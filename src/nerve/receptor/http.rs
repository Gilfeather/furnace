//! An HTTP route as a receptor

use std::sync::Arc;
use std::time::Duration;

use super::{Afferent, HttpMethod, HttpMount, Ingest, Receptor, Source, UrgencyRule};
use crate::nerve::neuron::{Cx, Firing, Neuron, NeuronKind};
use crate::nerve::signal::{Feature, Payload, Signal};

/// The header a client sets to say how soon it needs an action, in milliseconds
pub const DEADLINE_HEADER: &str = "x-deadline-ms";

type FeatureFn = dyn Fn(&Ingest<'_>) -> Feature + Send + Sync;

/// An HTTP route: each request is one signal. The feature is a 64-bit sketch of the body
/// unless the receptor is given its own; the urgency comes from the `x-deadline-ms` header,
/// else the route's default.
///
/// The response is whatever a [`Respond`](crate::nerve::effector::Respond) effector on the chain
/// answers; a chain that ends without one answers `202 Accepted` (queued to the brain) or
/// `204 No Content` (a reflex acted, or the signal died at the gate), with the outcome in the
/// `x-furnace-nerve` header.
pub struct Http {
    source: Source,
    mount: HttpMount,
    feature: Option<Arc<FeatureFn>>,
    text: bool,
    respond_within: Duration,
}

impl Http {
    /// `POST path`
    pub fn new(name: &str, path: &str) -> Self {
        Self {
            source: Source::new(name),
            mount: HttpMount {
                path: path.to_string(),
                method: HttpMethod::Post,
            },
            feature: None,
            text: false,
            respond_within: Duration::ZERO,
        }
    }

    /// Wait up to this long for a response when the signal went to a deliberative pathway
    /// (the brain answering through a `Respond` effector); by default such requests get
    /// `202 Accepted` at once
    pub fn respond_within(mut self, wait: Duration) -> Self {
        self.respond_within = wait;
        self
    }

    /// `GET path` instead of `POST`
    pub fn get(mut self) -> Self {
        self.mount.method = HttpMethod::Get;
        self
    }

    /// The urgency of requests without a deadline header
    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Extract the feature yourself (an embedding sketch, a field of the body)
    pub fn feature(mut self, f: impl Fn(&Ingest<'_>) -> Feature + Send + Sync + 'static) -> Self {
        self.feature = Some(Arc::new(f));
        self
    }

    /// Only accept UTF-8 bodies (`Payload::as_text` then always succeeds); others are ignored
    pub fn text(mut self) -> Self {
        self.text = true;
        self
    }
}

impl Neuron for Http {
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
            "route": format!("{:?} {}", self.mount.method, self.mount.path).to_uppercase(),
        }))
    }
}

impl Receptor for Http {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent);
    }

    fn mount(&self) -> Option<HttpMount> {
        Some(self.mount.clone())
    }

    fn respond_within(&self) -> Duration {
        self.respond_within
    }

    fn ingest(&self, request: Ingest<'_>) -> Option<Signal> {
        let feature = match &self.feature {
            Some(f) => f(&request),
            None => Feature::sketch(&request.body),
        };
        // Text stays in the request's own buffer: `Payload::as_text` reads bytes as UTF-8
        if self.text && std::str::from_utf8(&request.body).is_err() {
            return None;
        }
        let payload = Payload::Bytes(request.body);
        let mut signal = self.source.signal(feature, payload);
        if let Some(ms) = request
            .headers
            .get(DEADLINE_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|ms| ms.is_finite() && *ms >= 0.0)
        {
            signal.urgency.deadline = Some(signal.at + Duration::from_secs_f64(ms / 1e3));
        }
        Some(signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::time::Instant;

    #[test]
    fn http_receptor_sketches_the_body_and_reads_the_deadline_header() {
        let http = Http::new("in", "/events").urgency(UrgencyRule::within(Duration::from_secs(1)));
        let mut headers = axum::http::HeaderMap::new();
        let signal = http
            .ingest(Ingest {
                body: Bytes::from_static(b"hello"),
                content_type: None,
                headers: &headers,
            })
            .unwrap();
        assert_eq!(signal.feature, Feature::sketch(b"hello"));
        assert_eq!(signal.payload.as_text(), Some("hello"));
        let remaining = signal.urgency.remaining(Instant::now()).unwrap();
        assert!(remaining > Duration::from_millis(900), "{remaining:?}");
        headers.insert(DEADLINE_HEADER, "5".parse().unwrap());
        let urgent = http
            .ingest(Ingest {
                body: Bytes::from_static(b"hello"),
                content_type: None,
                headers: &headers,
            })
            .unwrap();
        assert!(urgent.urgency.remaining(Instant::now()).unwrap() <= Duration::from_millis(5));
        assert_eq!(http.mount().unwrap().path, "/events");
        let own = Http::new("own", "/x").feature(|_| Feature::new(9, 42));
        let signal = own
            .ingest(Ingest {
                body: Bytes::new(),
                content_type: None,
                headers: &headers,
            })
            .unwrap();
        assert_eq!(signal.feature, Feature::new(9, 42));
    }
}
