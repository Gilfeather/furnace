//! The nervous system (feature `nerve`): the layer between sensors, small fast models, the
//! large slow model (the brain, an LLM) and actuators.
//!
//! Not a server that loads models and answers requests, but what moves signals between
//! them: overwhelmingly light, so that the models are the only cost. Most signals must
//! never reach the brain. The mechanisms are a neuron's:
//!
//! - **Receptors** ([`receptor`]) extract a small [`Feature`] from the world and stamp an
//!   [`Urgency`]; the raw input stays a lazily shared [`Payload`].
//! - **Habituation**, **thresholds** and **refractory periods** ([`Mechanisms`]) live in
//!   every neuron's gate: a repeated feature attenuates before it propagates and recovers
//!   with time; weak signals cost nothing; a neuron that just fired ignores the next burst.
//! - **Reflex pathways** are direct calls on the firing thread, with no queue and no
//!   scheduler: microseconds from stimulus to effector.
//! - **Deliberative pathways** queue to a worker that sums a window of signals into one
//!   firing of a brain ([`NeuronKind::Brain`]) (temporal summation).
//! - **Timing selects the pathway**: a pathway whose measured p99 does not fit in the
//!   signal's remaining deadline is not eligible; among the eligible, the heaviest weight
//!   wins. No classifier decides "is this urgent".
//! - **Effectors** ([`effector`]) act on the world (a webhook, a command, a file, a
//!   response) with their own local loops: retry, circuit breaker, rate limit, refractory
//!   period. The brain is never in a retry loop.
//!
//! ```no_run
//! use std::time::Duration;
//! use furnace::nerve::{
//!     effector, receptor, Fabric, Firing, FnNeuron, Habituation, Mechanisms, NeuronKind,
//!     Summation,
//! };
//!
//! let mut fabric = Fabric::builder();
//! // POST /events: a 64-bit sketch of the body is the feature; repeats habituate here
//! let events = fabric.receptor(
//!     receptor::Http::new("events", "/events"),
//!     Mechanisms::NONE
//!         .threshold(0.3)
//!         .habituation(Habituation::new(Duration::from_secs(30))),
//! );
//! // A reflex: in process, no blocking; urgent lines become commands at once
//! let classify = fabric.neuron(
//!     FnNeuron::reflex("classify", |signal, _cx| {
//!         if signal.payload.as_text().is_some_and(|t| t.contains("ALERT")) {
//!             Firing::one(signal.command("alert", serde_json::json!({"text": "alert"})))
//!         } else {
//!             Firing::one(signal)
//!         }
//!     }),
//!     Mechanisms::NONE,
//! );
//! let alert = fabric.neuron(
//!     effector::Effector::new(effector::Webhook::new("alert", "http://127.0.0.1:9000/hook")),
//!     Mechanisms::NONE.refractory(Duration::from_millis(100)),
//! );
//! // The brain: slow, so it sits behind a queue that sums a window of signals into one
//! // firing (a closure here; `brain::Llm` or `brain::ModelNeuron` in practice)
//! let brain = fabric.neuron(
//!     FnNeuron::reflex("brain", |signal, _cx| {
//!         let text = signal.payload.as_text().unwrap_or_default().to_string();
//!         Firing::one(signal.command("alert", serde_json::json!({"text": text})))
//!     })
//!     .with_kind(NeuronKind::Brain),
//!     Mechanisms::NONE,
//! );
//! fabric.reflex(events, classify, 1.0);
//! fabric.reflex(classify, alert, 1.0);
//! // A command takes the pathway to the effector it names; any other signal the heaviest
//! // pathway fast enough for its deadline: the brain's, as long as it keeps up
//! fabric.deliberative(classify, brain, 2.0, Summation::default());
//! let fabric = fabric.build();
//! # let _ = fabric;
//! ```
//!
//! See [docs/NERVE.md](https://github.com/Gilfeather/furnace/blob/main/docs/NERVE.md) for
//! the concepts, the demo and the measurements.

#![warn(missing_docs)]

pub mod brain;
pub mod consolidation;
pub mod effector;
pub mod expect;
mod fabric;
pub(crate) mod http;
mod latency;
mod metrics;
mod neuron;
pub mod receptor;
pub(crate) mod serve;
mod signal;
#[cfg(test)]
mod testing;
pub mod wire;

pub use fabric::{
    Announced, Fabric, FabricBuilder, MAX_HOPS, Outcome, Pathway, PathwayKind, Summation,
};
pub use http::{OUTCOME_HEADER, SIGNAL_HEADER};
pub use latency::{LatencyEstimate, LatencySnapshot};
pub use metrics::{
    ArcMetrics, Description, NerveMetrics, NeuronDescription, NeuronMetrics, PathwayMetrics,
};
pub use neuron::{Cx, Firing, FnNeuron, Habituation, Mechanisms, Neuron, NeuronKind, Quiet};
pub use receptor::{Afferent, Receptor};
pub use signal::{
    Answer, Command, Feature, NeuronId, PathwayId, Payload, Reply, Signal, SignalId, TRACE_HOPS,
    Trace, Urgency, sketch64,
};

/// Nanoseconds from `epoch` to `now`: 0 if `now` is earlier, `u64::MAX` past 584 years
pub(crate) fn nanos_since(epoch: std::time::Instant, now: std::time::Instant) -> u64 {
    u64::try_from(now.saturating_duration_since(epoch).as_nanos()).unwrap_or(u64::MAX)
}
