//! An outbound HTTP call

#[cfg(feature = "tls")]
use std::path::Path;
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
/// HTTP/1.1, over TLS for an `https://` URL with the `tls` feature: TLS 1.3, through the same
/// pure-Rust stack as the server's TLS, to a server whose certificate chain is signed with ECDSA
/// or Ed25519 (RSA is refused), from one of Mozilla's certificate authorities or one given with
/// [`trust`](Self::trust). A certificate that does not pass is a final failure, not retried.
/// No redirects: a 3xx is a final failure, and enough of them in a row open the breaker.
pub struct Webhook {
    name: String,
    shared: Arc<WebhookShared>,
}

impl Webhook {
    /// `url` is `http://host[:port]/path`, or `https://` with the `tls` feature; each call has
    /// 5 seconds, the TLS handshake and the response included. An error for any other URL.
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

    /// Trust the certificate authorities in the PEM file `ca_file` too, for an `https://` URL:
    /// a private CA's certificate, say. Mozilla's authorities stay trusted.
    ///
    /// # Panics
    ///
    /// Once the webhook is shared: configure it before handing it to an [`Effector`](super::Effector).
    #[cfg(feature = "tls")]
    pub fn trust(mut self, ca_file: impl AsRef<Path>) -> Result<Self, BuildError> {
        let refused = |reason: String| BuildError::Certificate {
            name: format!("webhook '{}'", self.name),
            reason,
        };
        let authorities =
            crate::tls::read_certs(ca_file.as_ref()).map_err(|e| refused(e.to_string()))?;
        Arc::get_mut(&mut self.shared)
            .expect("configured before use")
            .client
            .trust(authorities)
            .map_err(|e| refused(format!("{}: {e}", ca_file.as_ref().display())))?;
        Ok(self)
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
                // A certificate that did not pass will not pass on the next try
                Err(e @ wire::Error::Tls(_)) => Err(ActError::Fatal(e.to_string())),
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
        serde_json::json!({ "webhook": self.shared.url.to_string() })
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
        for url in ["ftp://example.com/hook", "example.com", "http://:80/"] {
            assert!(
                matches!(Webhook::new("hook", url), Err(BuildError::Url { .. })),
                "{url}"
            );
        }
        let https = Webhook::new("hook", "https://example.com/hook");
        if cfg!(feature = "tls") {
            assert!(https.is_ok());
        } else {
            assert!(matches!(https, Err(BuildError::Url { .. })));
        }
    }

    /// A test certificate authority's file: `ca` signed the test servers' certificates
    #[cfg(feature = "tls")]
    fn fixture(name: &str) -> String {
        format!("{}/tests/fixtures/tls/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    /// An HTTPS server (the server's own TLS, the test CA's certificate for localhost) whose
    /// `/hook` records each body; the bodies, and its URL
    #[cfg(feature = "tls")]
    async fn https_hook() -> (Arc<Mutex<Vec<String>>>, String) {
        use crate::tls::{TlsConfigs, TlsListener, tests::settings};
        use axum::Router;
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::routing::post;
        let hits = Arc::new(Mutex::new(Vec::<String>::new()));
        let app = Router::new()
            .route(
                "/hook",
                post(
                    |State(hits): State<Arc<Mutex<Vec<String>>>>, body: String| async move {
                        hits.lock().unwrap().push(body);
                        StatusCode::NO_CONTENT
                    },
                ),
            )
            .with_state(hits.clone());
        let configs = TlsConfigs::load(settings("server-p256", false)).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let listener = TlsListener::new(listener, configs).unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (hits, format!("https://localhost:{port}/hook"))
    }

    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn https_webhook_posts_to_a_server_of_a_trusted_authority() {
        let (hits, url) = https_hook().await;
        let effector = Effector::new(
            Webhook::new("hook", &url)
                .unwrap()
                .trust(fixture("ca.crt"))
                .unwrap(),
        );
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("hook", serde_json::json!({"level": 3})));
        tokio::task::spawn_blocking({
            let effector = effector.clone();
            move || wait_until(|| effector.metrics().acted == 1)
        })
        .await
        .unwrap();
        assert_eq!(hits.lock().unwrap().as_slice(), [r#"{"level":3}"#]);
        drop(fabric);
    }

    /// A certificate from an authority the webhook does not trust fails the command at once:
    /// no retry, nothing reaches the server
    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn an_untrusted_certificate_is_a_final_failure() {
        let (hits, url) = https_hook().await;
        let effector = Effector::new(Webhook::new("hook", &url).unwrap()).retry(Retry {
            attempts: 3,
            backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(5),
        });
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("hook", serde_json::json!({"level": 3})));
        tokio::task::spawn_blocking({
            let effector = effector.clone();
            move || wait_until(|| effector.metrics().failed == 1)
        })
        .await
        .unwrap();
        assert_eq!(effector.metrics().retries, 0);
        assert!(hits.lock().unwrap().is_empty());
        drop(fabric);
    }

    #[cfg(feature = "tls")]
    #[test]
    fn trust_needs_a_file_of_certificates() {
        let url = "https://localhost/hook";
        for file in [fixture("missing.crt"), fixture("client.key")] {
            assert!(
                matches!(
                    Webhook::new("hook", url).unwrap().trust(&file),
                    Err(BuildError::Certificate { .. })
                ),
                "{file}"
            );
        }
    }
}
