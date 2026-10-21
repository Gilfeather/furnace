//! Pathway kinds, temporal summation, and the pathway itself

use std::sync::atomic::AtomicU64;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::batching::engine::Engine;
use crate::nerve::latency::LatencyEstimate;
use crate::nerve::signal::{NeuronId, PathwayId, Signal};

/// How signals move between two neurons
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathwayKind {
    /// A direct call on the firing thread: no queue, no scheduler
    Reflex,
    /// A queue to a worker that sums a window of signals into one firing
    Deliberative,
}

impl PathwayKind {
    /// The name in JSON and in the Prometheus `kind` label
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Reflex => "reflex",
            Self::Deliberative => "deliberative",
        }
    }
}

/// Temporal summation on a deliberative pathway
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summation {
    /// Most signals summed into one firing
    pub max_batch: usize,
    /// How long after the first signal the worker waits for more; `None` learns it from the
    /// pathway's latency (an eighth of the p50, within the bounds below)
    pub window: Option<Duration>,
    /// Lower bound of a learned window
    pub min_window: Duration,
    /// Upper bound of a learned window
    pub max_window: Duration,
    /// Signals waiting at once; beyond it, new ones are dropped (and counted) rather than
    /// queued without bound
    pub capacity: usize,
    /// Firings of the neuron that may run at the same time. While one of these workers is
    /// idle, a signal fires on its own at once; signals are summed only while all are busy.
    pub workers: usize,
    /// Wait out the window only for signals announced as on their way (the server's model
    /// routes announce theirs): a lone signal then never waits. Off, the worker always waits
    /// out the window for more.
    pub(crate) announced_only: bool,
}

impl Default for Summation {
    fn default() -> Self {
        Self {
            max_batch: 8,
            window: None,
            min_window: Duration::from_micros(200),
            max_window: Duration::from_millis(20),
            capacity: 1024,
            workers: 1,
            announced_only: false,
        }
    }
}

impl Summation {
    /// No summation: every signal fires the neuron on its own, as soon as the worker is free
    pub fn none() -> Self {
        Self {
            max_batch: 1,
            window: Some(Duration::ZERO),
            ..Self::default()
        }
    }

    /// Sum at most `n` signals into one firing; at least 1
    pub fn max_batch(mut self, n: usize) -> Self {
        self.max_batch = n.max(1);
        self
    }

    /// A fixed summation window instead of a learned one
    pub fn window(mut self, window: Duration) -> Self {
        self.window = Some(window);
        self
    }

    /// Signals waiting at once before new ones are dropped; at least 1
    pub fn capacity(mut self, n: usize) -> Self {
        self.capacity = n.max(1);
        self
    }

    /// Firings that may run at the same time; at least 1
    pub fn workers(mut self, n: usize) -> Self {
        self.workers = n.max(1);
        self
    }

    /// See [`Summation::announced_only`]
    pub(crate) fn announced_only(mut self) -> Self {
        self.announced_only = true;
        self
    }
}

/// A directed connection from one neuron to the next, with a measured latency
pub(crate) struct Pathway {
    /// Its index in the fabric
    pub id: PathwayId,
    /// The neuron that fires into it
    pub from: NeuronId,
    /// The neuron it stimulates
    pub to: NeuronId,
    /// Direct call or queue
    pub kind: PathwayKind,
    /// Priority among eligible pathways from the same neuron
    pub weight: f32,
    /// Signal-in to signal-out on this hop: for a reflex pathway the next neuron's whole
    /// synchronous chain, for a deliberative one the wait in the queue plus the firing
    pub(crate) latency: LatencyEstimate,
    pub(crate) signals: AtomicU64,
    /// Signals that could not take this pathway: too slow for their deadline
    pub(crate) ineligible: AtomicU64,
    /// Dropped because the queue was full
    pub(crate) overflow: AtomicU64,
    pub(crate) summation: Option<Summation>,
    /// A deliberative pathway's queue and its summation worker
    pub(super) queue: Option<Engine<Signal>>,
}

impl Pathway {
    /// The window the worker currently waits for more signals
    pub(crate) fn window(&self) -> Option<Duration> {
        self.summation.map(|s| {
            s.window.unwrap_or_else(|| {
                self.latency.p50().map_or(s.min_window, |p50| {
                    (p50 / 8).clamp(s.min_window, s.max_window)
                })
            })
        })
    }

    pub(super) fn fits(&self, remaining: Option<Duration>) -> bool {
        match (remaining, self.latency.p99()) {
            (None, _) | (_, None) => true,
            (Some(remaining), Some(p99)) => p99 <= remaining,
        }
    }
}
