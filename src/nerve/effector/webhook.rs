//! An outbound HTTP call

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tracing::debug;

use super::{Act, ActError, Action};
use crate::nerve::BuildError;
use crate::nerve::signal::{Command, Payload, Signal};
use crate::nerve::wire;

struct WebhookShared {
    url: wire::Url,
    client: wire::Client,
    headers: Vec<(String, String)>,
}

/// An outbound HTTP call: `POST url` with the command's `args` as JSON, or the signal's
/// payload as it is (JSON, text or bytes) for a plain signal. 5xx, 429 and connection
/// errors are retried; other 4xx are final.
///
/// Plain HTTP/1.1 only: no HTTPS (put a local proxy in front for that) and no redirects, so a
/// 3xx is a final failure, and enough of them in a row open the breaker.
pub struct Webhook {
    name: String,
    shared: Arc<WebhookShared>,
}

impl Webhook {
    /// `url` is `http://host[:port]/path`; each call has 5 seconds, the response included. An
    /// error for any other URL, an `https://` one among them.
    pub fn new(name: &str, url: &str) -> Result<Self, BuildError> {
        Self::with_timeout(name, url, Duration::from_secs(5))
    }

    /// [`Webhook::new`] with another bound on each call, the response included
    pub fn with_timeout(name: &str, url: &str, timeout: Duration) -> Result<Self, BuildError> {
        let url = wire::Url::parse(url).map_err(|e| BuildError::Url {
            name: format!("webhook '{name}'"),
            reason: e.to_string(),
        })?;
        Ok(Self {
            name: name.to_string(),
            shared: Arc::new(WebhookShared {
                url,
                client: wire::Client::new(timeout),
                headers: Vec::new(),
            }),
        })
    }

    /// Send this header with every call (an authorization token, say)
    ///
    /// # Panics
    ///
    /// Once the webhook is shared: configure it before handing it to an [`Effector`](super::Effector).
    pub fn header(mut self, name: &str, value: &str) -> Self {
        Arc::get_mut(&mut self.shared)
            .expect("configured before use")
            .headers
            .push((name.to_string(), value.to_string()));
        self
    }
}

impl Act for Webhook {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let (content_type, body) = if command.args.is_null() {
            match &signal.payload {
                Payload::Json(_) | Payload::Command(_) => {
                    ("application/json", signal.payload.to_bytes())
                }
                Payload::Text(_) => ("text/plain; charset=utf-8", signal.payload.to_bytes()),
                other => ("application/octet-stream", other.to_bytes()),
            }
        } else {
            ("application/json", Bytes::from(command.args.to_string()))
        };
        let shared = self.shared.clone();
        Action::pending(async move {
            let mut headers: Vec<(&str, &str)> = vec![("content-type", content_type)];
            headers.extend(shared.headers.iter().map(|(n, v)| (n.as_str(), v.as_str())));
            match shared
                .client
                .request("POST", &shared.url, &headers, &body)
                .await
            {
                // A JSON answer is what the act returns, to the motor program that sent it
                Ok(response) if (200..300).contains(&response.status) => {
                    Ok(serde_json::from_slice(&response.body).unwrap_or_default())
                }
                Ok(response) => Err(ActError::Status(response.status)),
                Err(wire::Error::Timeout(what)) => {
                    Err(ActError::Timeout(format!("{what} to {}", shared.url.host)))
                }
                Err(e @ wire::Error::Io(_)) => Err(ActError::Unreachable(e.to_string())),
                // An answer the client cannot read: the target may answer better next time
                Err(e) => Err(ActError::retry(e.to_string())),
            }
        })
    }

    fn prepare(&self, runtime: &tokio::runtime::Handle) {
        let shared = self.shared.clone();
        runtime.spawn(async move {
            if let Err(e) = shared.client.warm(&shared.url).await {
                debug!("nerve: webhook warm-up to {}: {e}", shared.url.host);
            }
        });
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "webhook": format!("http://{}:{}{}", self.shared.url.host, self.shared.url.port, self.shared.url.path),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Effector, Retry};
    use super::*;
    use crate::nerve::testing::{command, fabric_with, wait_until};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn webhook_posts_json_and_retries_5xx() {
        use axum::Router;
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::routing::post;
        type Hook = (Arc<Mutex<Vec<String>>>, Arc<AtomicUsize>);
        let hits = Arc::new(Mutex::new(Vec::<String>::new()));
        let fails = Arc::new(AtomicUsize::new(1));
        let app = Router::new()
            .route(
                "/hook",
                post(
                    |State((hits, fails)): State<Hook>, body: String| async move {
                        if fails
                            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                            .is_ok()
                        {
                            return StatusCode::SERVICE_UNAVAILABLE;
                        }
                        hits.lock().unwrap().push(body);
                        StatusCode::NO_CONTENT
                    },
                ),
            )
            .with_state((hits.clone(), fails));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let effector = Effector::new(
            Webhook::new("hook", &format!("http://127.0.0.1:{port}/hook"))
                .unwrap()
                .header("x-token", "t"),
        )
        .retry(Retry {
            attempts: 3,
            backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(5),
        });
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("hook", serde_json::json!({"level": 3})));
        tokio::task::spawn_blocking({
            let effector = effector.clone();
            move || wait_until(|| effector.metrics().acted == 1)
        })
        .await
        .unwrap();
        assert_eq!(effector.metrics().retries, 1);
        assert_eq!(hits.lock().unwrap().as_slice(), [r#"{"level":3}"#]);
        drop(fabric);
    }

    #[test]
    fn a_url_the_client_cannot_call_is_an_error() {
        for url in ["https://example.com/hook", "example.com", "http://:80/"] {
            assert!(
                matches!(Webhook::new("hook", url), Err(BuildError::Url { .. })),
                "{url}"
            );
        }
    }
}
