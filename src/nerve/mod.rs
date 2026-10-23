//! The nervous system: the layer between sensors, small fast models, the
//! large slow model (the brain, an LLM) and actuators.
//!
//! ![Receptors feed junctions of any depth that branch and merge; at every junction a signal
//! acts now or passes on; an optional brain is one possible end. Below, the brain talks to
//! the spine in actions, and a nerve per body part carries them to its motor junctions and
//! effectors](https://raw.githubusercontent.com/Gilfeather/furnace/main/docs/images/nervous-system-light.svg)
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
//! - **Motor neurons** ([`motor`]) stand between the brain and the effectors: the brain names
//!   an action ("get dressed"), and a motor neuron's program breaks it down into commands for
//!   the motor neurons and effectors below it, waits for what a receptor senses, asks a small
//!   question, retries locally and reports back up. The brain is told the actions declared
//!   with [`FabricBuilder::action`], not the joints, and reaches nothing else.
//! - **The skeleton**: [`FabricBuilder::spine`] is the brain's only way to the body (senses
//!   up, actions down, outcomes and questions up), and [`FabricBuilder::nerve`] the trunk to
//!   one part, which carries its own load, the urgent first, and closes the part's own loops.
//!
//! ```no_run
//! use std::time::Duration;
//! use furnace::nerve::prelude::*;
//! use serde_json::json;
//!
//! let mut b = Fabric::builder();
//! // POST /events: a 64-bit sketch of the body is the feature; repeats habituate here,
//! // attenuated below the threshold
//! let events = b.receptor(Http::new("events", "/events").text());
//! let habituation = Habituation::new(Duration::from_secs(30));
//! b.gate(events, Mechanisms::NONE.threshold(0.5).habituation(habituation));
//! // A reflex: in process, no blocking. Alerts become a command to `alert` at once; the rest
//! // go on to the brain
//! let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
//!     match signal.payload.as_text() {
//!         Some(t) if t.contains("ALERT") => signal.command("alert", json!({"text": t})),
//!         _ => signal,
//!     }
//! }));
//! // The brain: slow, so it sits behind a deliberative pathway, a queue whose worker sums a
//! // window of signals into one firing (a closure here; `Llm` or `ModelNeuron` in practice)
//! let brain = b.neuron(
//!     FnNeuron::reflex("brain", |signal, _| signal.command("alert", json!({"text": "look"})))
//!         .with_kind(NeuronKind::Brain),
//! );
//! b.effector(Webhook::new("alert", "http://127.0.0.1:9000/hook")?);
//! b.reflex(events, triage);
//! b.deliberative(triage, brain);
//! let fabric = b.build()?;
//! # let _ = fabric;
//! # Ok::<(), furnace::nerve::BuildError>(())
//! ```
//!
//! See [docs/NERVE.md](https://github.com/Gilfeather/furnace/blob/main/docs/NERVE.md) for
//! the concepts, the demo and the measurements.

#![warn(missing_docs)]

pub mod brain;
pub mod effector;
mod fabric;
pub(crate) mod http;
mod latency;
mod metrics;
pub mod motor;
mod neuron;
pub mod receptor;
pub(crate) mod serve;
mod signal;
mod sync;
#[cfg(test)]
mod testing;
pub(crate) mod wire;

/// The nervous system's own HTTP/1.1 client, for the nerve bench's raw baseline. Not part of
/// the API: it may change in any release.
#[doc(hidden)]
pub mod bench_wire {
    pub use super::wire::{Client, Error, Response, Url};
}

/// What building a fabric usually takes, in one import: `use furnace::nerve::prelude::*;`
pub mod prelude {
    #[cfg(feature = "llm")]
    pub use super::brain::Llm;
    pub use super::brain::ModelNeuron;
    pub use super::effector::{
        ChannelOut, Effector, FileAppend, Process, Respond, SseOut, Webhook,
    };
    pub use super::motor::{Motion, MotionError};
    pub use super::receptor::{ChannelIn, Http, Lines, SseIn, Timer};
    pub use super::{
        BuildError, Command, Fabric, Firing, FnNeuron, Habituation, Mechanisms, Nerve, NeuronKind,
        Payload, Signal, Summation, Urgency,
    };
}

pub(crate) use fabric::Announced;
pub use fabric::{
    BuildError, Fabric, FabricBuilder, Link, Nerve, NerveLaneMetrics, Outcome, PathwayKind, Spine,
    Summation,
};
pub use http::{OUTCOME_HEADER, SIGNAL_HEADER};
pub use latency::LatencySnapshot;
pub use metrics::{ArcMetrics, NerveMetrics, NeuronMetrics, PathwayMetrics};
pub use neuron::{Cx, Firing, FnNeuron, Habituation, Mechanisms, Neuron, NeuronKind, Quiet};
pub use receptor::{Afferent, Receptor};
pub use signal::{
    Answer, Command, Feature, NeuronId, Payload, ReplyHandle, Signal, SignalId, Trace, Urgency,
};

/// Nanoseconds from `epoch` to `now`: 0 if `now` is earlier, `u64::MAX` past 584 years
pub(crate) fn nanos_since(epoch: std::time::Instant, now: std::time::Instant) -> u64 {
    u64::try_from(now.saturating_duration_since(epoch).as_nanos()).unwrap_or(u64::MAX)
}
