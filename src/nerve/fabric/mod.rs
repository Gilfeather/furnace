//! The fabric: neurons, the pathways between them, and how a signal moves.
//!
//! A reflex pathway is a direct call: the next neuron's `stimulate` runs on the thread that
//! fired, with no channel in between. A deliberative pathway is a queue to a worker thread
//! that sums the signals of a short window into one firing of its neuron (temporal
//! summation). Which pathway a signal takes is decided by timing: a pathway whose measured
//! p99 does not fit in the signal's remaining deadline is not eligible; among the eligible,
//! the heaviest weight wins.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::latency::Latency;
use super::motor::{ActionSpec, Taps};
use super::neuron::{Gate, Neuron, NeuronKind, Quiet};
use super::receptor::{Afferent, Receptor, Stop};
use super::signal::{NeuronId, PathwayId};

/// Hops a signal may take before the fabric assumes a loop and drops it
pub(crate) const MAX_HOPS: usize = 32;

mod builder;
mod pathway;
mod route;
mod summing;
#[cfg(test)]
mod tests;

pub use builder::{BuildError, FabricBuilder, Link};
use pathway::Pathway;
pub use pathway::{PathwayKind, Summation};
pub(crate) use route::Announced;

/// Per-neuron counters
#[derive(Default)]
pub(crate) struct Counters {
    pub received: AtomicU64,
    pub fired: AtomicU64,
    pub computed: AtomicU64,
    pub quiet_threshold: AtomicU64,
    pub quiet_habituation: AtomicU64,
    pub quiet_refractory: AtomicU64,
    pub quiet_not_interested: AtomicU64,
}

impl Counters {
    fn quiet(&self, reason: Quiet) {
        let cell = match reason {
            Quiet::Threshold => &self.quiet_threshold,
            Quiet::Habituation => &self.quiet_habituation,
            Quiet::Refractory => &self.quiet_refractory,
            Quiet::NotInterested => &self.quiet_not_interested,
        };
        cell.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) struct Cell {
    pub neuron: Arc<dyn Neuron>,
    pub kind: NeuronKind,
    pub gate: Gate,
    pub counters: Counters,
    /// Pathways leaving this neuron
    pub out: Vec<PathwayId>,
    /// Nanoseconds since the epoch of the last `prepare`
    last_prepared: AtomicU64,
    /// For effectors: stimulus-to-action latency per origin (reflex arcs only), by the
    /// origin's index
    pub arcs: Vec<OnceLock<Latency>>,
    /// The motor programs sensing this neuron
    pub taps: Taps,
}

pub(crate) struct Inner {
    pub epoch: Instant,
    pub cells: Vec<Cell>,
    pub pathways: Vec<Pathway>,
    pub by_name: HashMap<String, NeuronId>,
    receptors: Vec<(NeuronId, Arc<dyn Receptor>)>,
    /// What a brain may call, in the order declared
    actions: Vec<ActionSpec>,
    started: AtomicBool,
    /// Signals that entered through a receptor
    pub received: AtomicU64,
    /// Signals a brain neuron saw
    pub reached_brain: AtomicU64,
    /// Commands addressed to a name no effector or motor neuron has
    pub unrouted: AtomicU64,
    /// Signals with no eligible pathway (all too slow for the deadline)
    pub no_eligible: AtomicU64,
    /// Chains cut at [`MAX_HOPS`]
    pub loops: AtomicU64,
    /// Commands delivered to their effector without a declared pathway
    pub commands_direct: AtomicU64,
    /// Raised on drop, for the receptors started with [`Fabric::start`]
    stop: Arc<Stop>,
    efferent: Mutex<Option<tokio::runtime::Runtime>>,
    efferent_handle: tokio::runtime::Handle,
    afferent: Mutex<Option<tokio::runtime::Runtime>>,
    afferent_handle: tokio::runtime::Handle,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Receptors' threads and tasks end on this
        self.stop.raise();
        let take = |runtime: &mut Mutex<Option<tokio::runtime::Runtime>>| {
            runtime.get_mut().ok().and_then(Option::take)
        };
        let runtimes = (take(&mut self.efferent), take(&mut self.afferent));
        // A runtime cannot be dropped from inside another runtime's thread
        if tokio::runtime::Handle::try_current().is_ok() {
            std::thread::spawn(move || drop(runtimes));
        } else {
            drop(runtimes);
        }
    }
}

/// What happened to a signal handed to the fabric: how far it got synchronously, and what
/// is still pending on deliberative pathways
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The first neuron that was quiet, and why
    pub quiet: Option<(NeuronId, Quiet)>,
    /// The longest synchronous chain, in neurons stimulated
    pub hops: usize,
    /// Signals now waiting on a deliberative pathway
    pub queued: usize,
    /// Effectors and motor neurons stimulated synchronously
    pub acted: usize,
    /// Commands with no effector or motor neuron of that name
    pub unrouted: usize,
    /// Signals with no pathway fast enough for their deadline
    pub no_pathway: usize,
    /// Signals dropped because a deliberative queue was full
    pub dropped: usize,
}

impl Outcome {
    /// Whether the receptor's own gate dropped the signal
    pub fn is_quiet(&self) -> bool {
        self.hops == 0
    }

    fn merge(&mut self, other: Outcome) {
        if self.quiet.is_none() {
            self.quiet = other.quiet;
        }
        self.hops = self.hops.max(other.hops);
        self.queued += other.queued;
        self.acted += other.acted;
        self.unrouted += other.unrouted;
        self.no_pathway += other.no_pathway;
        self.dropped += other.dropped;
    }
}

/// Neurons and pathways, built once and then shared; [`Fabric::stimulate`] moves a signal
/// through them. Cloning shares the same fabric.
#[derive(Clone)]
pub struct Fabric(pub(crate) Arc<Inner>);

impl Fabric {
    /// An empty [`FabricBuilder`]
    pub fn builder() -> FabricBuilder {
        FabricBuilder::new()
    }

    /// Start every receptor's sampling (idempotent). [`Server::nerve`](crate::Server::nerve)
    /// does this when the router is built.
    pub fn start(&self) {
        if self.0.started.swap(true, Ordering::AcqRel) {
            return;
        }
        for (id, receptor) in &self.0.receptors {
            receptor.start(Afferent::new(
                Arc::downgrade(&self.0),
                *id,
                self.0.stop.clone(),
            ));
        }
    }

    /// The receptors, for mounting their routes
    pub(crate) fn receptors(&self) -> &[(NeuronId, Arc<dyn Receptor>)] {
        &self.0.receptors
    }

    /// A neuron by name
    pub fn neuron(&self, name: &str) -> Option<NeuronId> {
        self.0.by_name.get(name).copied()
    }

    /// A neuron's name
    pub(crate) fn name(&self, id: NeuronId) -> &str {
        self.0.cells[id.index()].neuron.name()
    }

    /// The actions a brain may call: the motor neurons and effectors declared with
    /// [`FabricBuilder::action`], in that order
    pub fn actions(&self) -> &[ActionSpec] {
        &self.0.actions
    }

    /// Stop the program running at the motor neuron named `name`, and everything it
    /// commanded; whether one was running
    pub fn stop(&self, name: &str) -> bool {
        self.neuron(name)
            .is_some_and(|id| self.0.cells[id.index()].neuron.stop())
    }

    /// Subscribe to a neuron that publishes events (an SSE effector)
    pub(crate) fn events(
        &self,
        id: NeuronId,
    ) -> Option<tokio::sync::broadcast::Receiver<bytes::Bytes>> {
        self.0.cells.get(id.index())?.neuron.events()
    }

    /// The motor side's runtime
    pub(crate) fn efferent(&self) -> &tokio::runtime::Handle {
        &self.0.efferent_handle
    }

    /// The receptive side's runtime
    pub(crate) fn afferent(&self) -> &tokio::runtime::Handle {
        &self.0.afferent_handle
    }

    /// Nanoseconds since the fabric was built
    pub(crate) fn nanos(&self, now: Instant) -> u64 {
        super::nanos_since(self.0.epoch, now)
    }

    /// Signals that entered through receptors, and those that reached a brain
    pub fn brain_fraction(&self) -> f64 {
        let received = self.0.received.load(Ordering::Relaxed);
        if received == 0 {
            0.0
        } else {
            self.0.reached_brain.load(Ordering::Relaxed) as f64 / received as f64
        }
    }

    /// Signals waiting in deliberative queues
    pub fn waiting(&self) -> u64 {
        self.0
            .pathways
            .iter()
            .filter_map(|p| p.queue.as_ref())
            .map(|q| q.waiting() as u64)
            .sum()
    }
}
