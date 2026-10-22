//! An upstream Server-Sent-Events feed

use std::time::Duration;

use tracing::{debug, warn};

use super::{Afferent, Backoff, RateMeter, Receptor, Source, UrgencyRule, kinds};
use crate::nerve::neuron::{Cx, Firing, Neuron, NeuronKind};
use crate::nerve::signal::{Feature, Payload, Signal};
use crate::nerve::wire;

/// An upstream event feed: `GET url` as `text/event-stream`, one signal per event (the
/// `data` as text, the feature a sketch of it), reconnecting with backoff and
/// `Last-Event-ID`. Runs on the fabric's afferent runtime.
pub struct SseIn {
    source: Source,
    url: String,
    event: Option<String>,
}

impl SseIn {
    /// Follow `url`, a plain `http://host[:port]/path` (no HTTPS); a URL of another form is
    /// logged when the receptor starts, and nothing is read
    pub fn new(name: &str, url: &str) -> Self {
        Self {
            source: Source::new(name),
            url: url.to_string(),
            event: None,
        }
    }

    /// How each event's urgency is stamped
    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Only events of this `event:` type
    pub fn event(mut self, event: &str) -> Self {
        self.event = Some(event.to_string());
        self
    }
}

impl Neuron for SseIn {
    fn name(&self) -> &str {
        &self.source.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Receptor
    }

    fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
        Firing::one(signal)
    }

    fn describe(&self) -> serde_json::Value {
        self.source.describe(serde_json::json!({
            "source": format!("sse {}", self.url),
            "event": self.event,
        }))
    }
}

/// One parsed SSE event
#[derive(Debug, Default, PartialEq, Eq)]
struct SseEvent {
    event: Option<String>,
    data: String,
    id: Option<String>,
}

/// Incremental parser of the `text/event-stream` format
#[derive(Default)]
struct SseParser {
    buffer: String,
    current: SseEvent,
    has_data: bool,
}

impl SseParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut events = Vec::new();
        while let Some(i) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=i).collect();
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if self.has_data {
                    let mut event = std::mem::take(&mut self.current);
                    if event.data.ends_with('\n') {
                        event.data.pop();
                    }
                    events.push(event);
                }
                self.has_data = false;
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "data" => {
                    self.current.data.push_str(value);
                    self.current.data.push('\n');
                    self.has_data = true;
                }
                "event" => self.current.event = Some(value.to_string()),
                "id" => self.current.id = Some(value.to_string()),
                _ => {}
            }
        }
        events
    }
}

impl Receptor for SseIn {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent.clone());
        let Some(runtime) = afferent.runtime() else {
            return;
        };
        let url = match wire::Url::parse(&self.url) {
            Ok(url) => url,
            Err(e) => {
                warn!("nerve: sse receptor: {e}");
                return;
            }
        };
        let (rule, wanted) = (self.source.rule, self.event.clone());
        runtime.spawn(async move {
            let client = wire::Client::new(Duration::from_secs(3600));
            let meter = RateMeter::new();
            let mut last_id: Option<String> = None;
            let mut backoff = Backoff::new(Duration::from_millis(200));
            loop {
                let mut headers = vec![("accept", "text/event-stream")];
                let id = last_id.clone().unwrap_or_default();
                if last_id.is_some() {
                    headers.push(("last-event-id", &id));
                }
                match client.send("GET", &url, &headers, b"").await {
                    Ok(mut streaming) if streaming.status == 200 => {
                        backoff.reset();
                        let mut parser = SseParser::default();
                        loop {
                            match streaming.next().await {
                                Ok(Some(piece)) => {
                                    for event in parser.push(&piece) {
                                        if let Some(id) = &event.id {
                                            last_id = Some(id.clone());
                                        }
                                        if wanted.as_ref().is_some_and(|w| {
                                            event.event.as_deref().unwrap_or("message") != w
                                        }) {
                                            continue;
                                        }
                                        let signal = rule.signal(
                                            &meter,
                                            Feature::sketch_as(kinds::SSE, event.data.as_bytes()),
                                            Payload::text(&event.data),
                                        );
                                        if afferent.emit(signal).is_none() {
                                            return;
                                        }
                                    }
                                }
                                Ok(None) => break,
                                Err(e) => {
                                    debug!("nerve: sse receptor: {e}");
                                    break;
                                }
                            }
                        }
                    }
                    Ok(streaming) => {
                        debug!(
                            "nerve: sse receptor: upstream answered {}",
                            streaming.status
                        );
                    }
                    Err(e) => debug!("nerve: sse receptor: {e}"),
                }
                if afferent.fabric().is_none() {
                    return;
                }
                tokio::time::sleep(backoff.next()).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::testing::{fabric_with_receptor, wait_until};
    use bytes::Bytes;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    #[test]
    fn sse_parser_handles_multi_line_data_ids_and_comments() {
        let mut parser = SseParser::default();
        let events = parser
            .push(b": keep-alive\nid: 7\nevent: temp\ndata: 21.5\ndata: more\n\ndata: plain\n\n");
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("temp".into()),
                    data: "21.5\nmore".into(),
                    id: Some("7".into())
                },
                SseEvent {
                    event: None,
                    data: "plain".into(),
                    id: None
                }
            ]
        );
        // A partial line waits for the rest
        assert!(parser.push(b"data: par").is_empty());
        assert_eq!(parser.push(b"tial\n\n")[0].data, "partial");
    }

    #[tokio::test]
    async fn sse_receptor_follows_an_upstream_feed() {
        use axum::Router;
        use axum::body::Body;
        use axum::routing::get;
        let served = Arc::new(AtomicUsize::new(0));
        let app = Router::new().route(
            "/feed",
            get({
                let served = served.clone();
                move || {
                    served.fetch_add(1, Ordering::SeqCst);
                    async {
                        let events = (0..3).map(|i| {
                            Ok::<_, std::convert::Infallible>(Bytes::from(format!(
                                "id: {i}\ndata: event {i}\n\n"
                            )))
                        });
                        axum::response::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from_stream(tokio_stream::iter(events)))
                            .unwrap()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (fabric, _r, seen) =
            fabric_with_receptor(SseIn::new("feed", &format!("http://127.0.0.1:{port}/feed")));
        tokio::task::spawn_blocking({
            let seen = seen.clone();
            move || wait_until(|| seen.lock().unwrap().len() >= 3)
        })
        .await
        .unwrap();
        {
            let seen = seen.lock().unwrap();
            assert_eq!(seen[2].payload.as_text(), Some("event 2"));
            assert_eq!(seen[2].feature.kind, kinds::SSE);
        }
        // The stream ended: the receptor reconnects (and would resume from the last id)
        let deadline = Instant::now() + Duration::from_secs(5);
        while served.load(Ordering::SeqCst) < 2 {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(fabric);
    }
}
