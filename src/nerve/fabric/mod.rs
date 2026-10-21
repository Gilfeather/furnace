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
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use super::effector::Effector;
use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Gate, Mechanisms, Neuron, NeuronKind, Quiet};
use super::receptor::{Afferent, Receptor, Stop};
use super::signal::{NeuronId, PathwayId, Signal};
use crate::batching::engine::{self, Batching, Engine, Queued};

/// Hops a signal may take before the fabric assumes a loop and drops it
pub(crate) const MAX_HOPS: usize = 32;

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
    queue: Option<Engine<Signal>>,
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

    fn fits(&self, remaining: Option<Duration>) -> bool {
        match (remaining, self.latency.p99()) {
            (None, _) | (_, None) => true,
            (Some(remaining), Some(p99)) => p99 <= remaining,
        }
    }
}

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
    pub arcs: Vec<OnceLock<LatencyEstimate>>,
}

pub(crate) struct Inner {
    pub epoch: Instant,
    pub cells: Vec<Cell>,
    pub pathways: Vec<Pathway>,
    pub by_name: HashMap<String, NeuronId>,
    receptors: Vec<(NeuronId, Arc<dyn Receptor>)>,
    started: AtomicBool,
    /// Signals that entered through a receptor
    pub received: AtomicU64,
    /// Signals a brain neuron saw
    pub reached_brain: AtomicU64,
    /// Commands addressed to a name no effector has
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
    /// Effectors stimulated synchronously
    pub acted: usize,
    /// Commands with no effector of that name
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

/// Why a part of the nervous system cannot be built
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BuildError {
    /// Two neurons have the same name; effectors are addressed by name
    #[error("two neurons are named '{0}'")]
    DuplicateName(String),
    /// A pathway names a neuron that was not added to this builder
    #[error("pathway {from} -> {to} names a neuron that was not added")]
    UnknownNeuron {
        /// The pathway's source
        from: NeuronId,
        /// The pathway's target
        to: NeuronId,
    },
    /// A URL the nervous system's HTTP client cannot call: not `http://host[:port]/path`
    /// (HTTPS among them)
    #[error("{name}: {reason}")]
    Url {
        /// The effector or receptor
        name: String,
        /// What is wrong with the URL
        reason: String,
    },
    /// A runtime of the fabric could not be started
    #[error("the {0} runtime: {1}")]
    Runtime(&'static str, #[source] std::io::Error),
}

/// Builds a [`Fabric`]: add neurons, connect them, build
pub struct FabricBuilder {
    neurons: Vec<(Arc<dyn Neuron>, Mechanisms)>,
    receptors: Vec<(NeuronId, Arc<dyn Receptor>)>,
    pathways: Vec<(NeuronId, NeuronId, PathwayKind, f32, Option<Summation>)>,
    efferent_threads: usize,
    efferent: Option<tokio::runtime::Handle>,
}

impl Default for FabricBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FabricBuilder {
    /// An empty builder
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            receptors: Vec::new(),
            pathways: Vec::new(),
            efferent_threads: 2,
            efferent: None,
        }
    }

    /// Threads of the motor side's runtime (2 by default)
    pub fn efferent_threads(mut self, threads: usize) -> Self {
        self.efferent_threads = threads.max(1);
        self
    }

    /// Run the motor side (effectors' calls, retries, backoff) on a runtime of your own, the
    /// server's say, instead of one the fabric starts. The runtime must outlive the fabric;
    /// [`efferent_threads`](Self::efferent_threads) no longer applies.
    pub fn efferent(mut self, runtime: tokio::runtime::Handle) -> Self {
        self.efferent = Some(runtime);
        self
    }

    /// Add a neuron, with no mechanisms in front of it ([`gate`](Self::gate) adds them).
    /// Names must be unique, since effectors are addressed by name in commands:
    /// [`build`](Self::build) refuses two of the same name.
    pub fn neuron(&mut self, neuron: impl Neuron) -> NeuronId {
        self.add(Arc::new(neuron))
    }

    /// [`FabricBuilder::neuron`] for a neuron already shared
    pub(crate) fn add(&mut self, neuron: Arc<dyn Neuron>) -> NeuronId {
        let id = NeuronId(u32::try_from(self.neurons.len()).expect("fewer than 4G neurons"));
        self.neurons.push((neuron, Mechanisms::NONE));
        id
    }

    /// Add a receptor: a neuron that is also started with the fabric and may be mounted as a
    /// route. Habituation belongs here ([`gate`](Self::gate)).
    pub fn receptor<R: Receptor>(&mut self, receptor: R) -> NeuronId {
        let receptor: Arc<R> = Arc::new(receptor);
        let neuron: Arc<dyn Neuron> = receptor.clone();
        let id = self.add(neuron);
        self.receptors.push((id, receptor));
        id
    }

    /// Add an effector: an [`Act`](super::effector::Act) with the default local loop
    /// (`b.effector(Webhook::new(..)?)`), or an [`Effector`] configured first. Commands
    /// reach it by its name from any neuron, with or without a pathway to it.
    pub fn effector(&mut self, effector: impl Into<Effector>) -> NeuronId {
        self.neuron(effector.into())
    }

    /// Put `mechanisms` (threshold, habituation, refractory period) in front of neuron `id`
    ///
    /// # Panics
    ///
    /// For an id this builder did not hand out.
    pub fn gate(&mut self, id: NeuronId, mechanisms: Mechanisms) -> &mut Self {
        self.neurons[id.index()].1 = mechanisms;
        self
    }

    /// The id the next pathway added will have
    pub(crate) fn next_pathway(&self) -> PathwayId {
        PathwayId(u32::try_from(self.pathways.len()).expect("fewer than 4G pathways"))
    }

    /// A reflex pathway: `to` is called directly when `from` fires. Weight 1 unless
    /// [`Link::weight`] says otherwise.
    pub fn reflex(&mut self, from: NeuronId, to: NeuronId) -> Link<'_> {
        self.link(from, to, PathwayKind::Reflex, REFLEX_WEIGHT, None)
    }

    /// A deliberative pathway: signals from `from` queue to a worker that sums them into
    /// firings of `to`. Weight 2, so it wins over a reflex pathway whenever the signal's
    /// deadline allows it, and the default [`Summation`]; [`Link`] changes either.
    pub fn deliberative(&mut self, from: NeuronId, to: NeuronId) -> Link<'_> {
        self.link(
            from,
            to,
            PathwayKind::Deliberative,
            DELIBERATIVE_WEIGHT,
            Some(Summation::default()),
        )
    }

    fn link(
        &mut self,
        from: NeuronId,
        to: NeuronId,
        kind: PathwayKind,
        weight: f32,
        summation: Option<Summation>,
    ) -> Link<'_> {
        self.pathways.push((from, to, kind, weight, summation));
        Link {
            pathway: self.pathways.last_mut().expect("just pushed"),
        }
    }

    /// Wire the fabric and start its runtimes and its deliberative workers. An error for two
    /// neurons of the same name, a pathway to a neuron of another builder, or a runtime that
    /// cannot be started.
    pub fn build(self) -> Result<Fabric, BuildError> {
        let count = self.neurons.len();
        let mut names = std::collections::HashSet::new();
        for (neuron, _) in &self.neurons {
            if !names.insert(neuron.name()) {
                return Err(BuildError::DuplicateName(neuron.name().to_string()));
            }
        }
        drop(names);
        for &(from, to, ..) in &self.pathways {
            if from.index() >= count || to.index() >= count {
                return Err(BuildError::UnknownNeuron { from, to });
            }
        }
        let (efferent, efferent_handle) = match self.efferent {
            Some(handle) => (None, handle),
            None => {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(self.efferent_threads)
                    .thread_name("furnace-efferent")
                    .enable_all()
                    .build()
                    .map_err(|e| BuildError::Runtime("efferent", e))?;
                let handle = runtime.handle().clone();
                (Some(runtime), handle)
            }
        };
        let afferent = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("furnace-afferent")
            .enable_all()
            .build()
            .map_err(|e| BuildError::Runtime("afferent", e))?;
        let mut by_name = HashMap::new();
        let mut cells: Vec<Cell> = self
            .neurons
            .into_iter()
            .enumerate()
            .map(|(i, (neuron, mechanisms))| {
                by_name.insert(neuron.name().to_string(), NeuronId(i as u32));
                let kind = neuron.kind();
                Cell {
                    arcs: if kind == NeuronKind::Effector {
                        (0..count).map(|_| OnceLock::new()).collect()
                    } else {
                        Vec::new()
                    },
                    neuron,
                    kind,
                    gate: Gate::new(mechanisms),
                    counters: Counters::default(),
                    out: Vec::new(),
                    last_prepared: AtomicU64::new(0),
                }
            })
            .collect();
        let inner = Arc::new_cyclic(|weak: &Weak<Inner>| {
            let pathways: Vec<Pathway> = self
                .pathways
                .into_iter()
                .enumerate()
                .map(|(i, (from, to, kind, weight, summation))| {
                    let id = PathwayId(i as u32);
                    cells[from.index()].out.push(id);
                    let queue = summation.map(|s| {
                        Engine::start(
                            Summing {
                                fabric: weak.clone(),
                                pathway: id,
                            },
                            engine::Config {
                                max_batch: s.max_batch,
                                workers: s.workers,
                                capacity: Some(s.capacity),
                                waits_unannounced: !s.announced_only,
                            },
                            move |_| format!("furnace-nerve-{i}"),
                        )
                    });
                    Pathway {
                        id,
                        from,
                        to,
                        kind,
                        weight,
                        latency: LatencyEstimate::new(),
                        signals: AtomicU64::new(0),
                        ineligible: AtomicU64::new(0),
                        overflow: AtomicU64::new(0),
                        summation,
                        queue,
                    }
                })
                .collect();
            Inner {
                epoch: Instant::now(),
                cells,
                pathways,
                by_name,
                receptors: self.receptors,
                started: AtomicBool::new(false),
                received: AtomicU64::new(0),
                reached_brain: AtomicU64::new(0),
                unrouted: AtomicU64::new(0),
                no_eligible: AtomicU64::new(0),
                loops: AtomicU64::new(0),
                commands_direct: AtomicU64::new(0),
                stop: Arc::new(Stop::default()),
                efferent_handle,
                efferent: Mutex::new(efferent),
                afferent_handle: afferent.handle().clone(),
                afferent: Mutex::new(Some(afferent)),
            }
        });
        Ok(Fabric(inner))
    }
}

/// The weight of a reflex pathway unless [`Link::weight`] changes it
const REFLEX_WEIGHT: f32 = 1.0;
/// The weight of a deliberative pathway unless [`Link::weight`] changes it: above a
/// reflex's, so the brain is chosen whenever there is time for it
const DELIBERATIVE_WEIGHT: f32 = 2.0;

/// A pathway just added, to change its weight or summation
pub struct Link<'a> {
    pathway: &'a mut (NeuronId, NeuronId, PathwayKind, f32, Option<Summation>),
}

impl Link<'_> {
    /// Priority among the eligible pathways from the same neuron: the heaviest wins
    pub fn weight(self, weight: f32) -> Self {
        self.pathway.3 = weight;
        self
    }

    /// How a deliberative pathway sums its queue into firings. On a reflex pathway it makes
    /// the pathway deliberative.
    pub fn summation(self, summation: Summation) -> Self {
        self.pathway.2 = PathwayKind::Deliberative;
        self.pathway.4 = Some(summation);
        self
    }
}

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

    /// The names effectors were registered under
    pub(crate) fn effector_names(&self) -> Vec<String> {
        self.0
            .cells
            .iter()
            .filter(|c| c.kind == NeuronKind::Effector)
            .map(|c| c.neuron.name().to_string())
            .collect()
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

    /// Hand a signal to a receptor: the whole synchronous chain runs before this returns.
    /// Receptors call this through their [`Afferent`] handle; tests and in-process sources
    /// call it directly.
    pub fn stimulate(&self, receptor: NeuronId, signal: Signal) -> Outcome {
        self.stimulate_announced(receptor, signal, None)
    }

    /// Announce a signal that will take `pathway` once it is ready (its request's body is
    /// still being received or decoded): a worker collecting a batch there waits for it.
    /// Hold the guard while preparing the signal, then hand both to
    /// [`Fabric::stimulate_announced`]. `None` for a reflex pathway, which has no queue.
    pub(crate) fn announce(&self, pathway: PathwayId) -> Option<Announced<'_>> {
        let queue = self.0.pathways.get(pathway.index())?.queue.as_ref()?;
        Some(Announced {
            pathway,
            arrival: queue.arrival(),
        })
    }

    /// [`Fabric::stimulate`] for a signal announced with [`Fabric::announce`]: if it takes
    /// the announced pathway, it is queued as the arrival a worker may be waiting for
    pub(crate) fn stimulate_announced(
        &self,
        receptor: NeuronId,
        mut signal: Signal,
        announced: Option<Announced<'_>>,
    ) -> Outcome {
        signal.origin = receptor;
        self.0.received.fetch_add(1, Ordering::Relaxed);
        let mut announced = announced;
        self.drive(receptor, signal, 0, &mut announced, Instant::now())
    }

    /// [`Fabric::stimulate`] by the receptor's name
    pub fn stimulate_named(&self, receptor: &str, signal: Signal) -> Option<Outcome> {
        Some(self.stimulate(self.neuron(receptor)?, signal))
    }

    /// Run `id`'s gate and `stimulate`, then propagate what it fires; `now` is when the
    /// signal arrived, which the caller has read already
    fn drive(
        &self,
        id: NeuronId,
        mut signal: Signal,
        depth: usize,
        announced: &mut Option<Announced<'_>>,
        now: Instant,
    ) -> Outcome {
        let cell = &self.0.cells[id.index()];
        let nanos = self.nanos(now);
        cell.counters.received.fetch_add(1, Ordering::Relaxed);
        if let Err(reason) = cell.gate.admit(&mut signal, nanos) {
            cell.counters.quiet(reason);
            return Outcome {
                quiet: Some((id, reason)),
                hops: depth,
                ..Outcome::default()
            };
        }
        signal.trace.push(id);
        let mut acted = 0;
        match cell.kind {
            NeuronKind::Brain => {
                signal.trace.mark_brain();
                self.0.reached_brain.fetch_add(1, Ordering::Relaxed);
            }
            NeuronKind::Effector => {
                acted = 1;
                if !signal.trace.was_queued()
                    && let Some(arc) = cell.arcs.get(signal.origin.index())
                {
                    arc.get_or_init(LatencyEstimate::new)
                        .observe(now.saturating_duration_since(signal.at));
                }
            }
            NeuronKind::Receptor | NeuronKind::Reflex => {}
        }
        let cx = Cx {
            now,
            id,
            fabric: self,
        };
        let firing = cell.neuron.stimulate(signal, &cx);
        if matches!(firing, Firing::Quiet) {
            cell.gate.stayed_quiet(nanos);
        }
        let mut outcome = self.after_firing(id, firing, depth, announced);
        outcome.acted += acted;
        outcome
    }

    /// Propagate what `id` fired
    fn after_firing(
        &self,
        id: NeuronId,
        firing: Firing,
        depth: usize,
        announced: &mut Option<Announced<'_>>,
    ) -> Outcome {
        let cell = &self.0.cells[id.index()];
        let emitted = match firing {
            Firing::Quiet => {
                cell.counters.quiet(Quiet::NotInterested);
                return Outcome {
                    quiet: Some((id, Quiet::NotInterested)),
                    hops: depth + 1,
                    ..Outcome::default()
                };
            }
            Firing::Fire(signals) => signals,
            Firing::Compute(work) => {
                cell.counters.computed.fetch_add(1, Ordering::Relaxed);
                work()
            }
        };
        cell.counters.fired.fetch_add(1, Ordering::Relaxed);
        let mut outcome = Outcome {
            hops: depth + 1,
            ..Outcome::default()
        };
        for signal in emitted {
            outcome.merge(self.route(id, signal, depth + 1, announced));
        }
        outcome
    }

    /// Send a signal `from` fired along the pathway its timing selects
    fn route(
        &self,
        from: NeuronId,
        mut signal: Signal,
        depth: usize,
        announced: &mut Option<Announced<'_>>,
    ) -> Outcome {
        // The depth restarts after a deliberative queue; the trace keeps the whole count, so a
        // loop through a queue is cut too
        if depth >= MAX_HOPS || signal.trace.len() >= MAX_HOPS {
            self.0.loops.fetch_add(1, Ordering::Relaxed);
            warn!(
                "nerve: signal {} dropped after {MAX_HOPS} hops (a loop?)",
                signal.id.0
            );
            return Outcome::default();
        }
        let now = Instant::now();
        let cell = &self.0.cells[from.index()];
        let selected = match signal.payload.as_command() {
            // A command names its effector; the pathway to it is the one (declared or not)
            Some(command) => match self.0.by_name.get(&command.effector) {
                Some(&target) if self.0.cells[target.index()].kind == NeuronKind::Effector => {
                    match cell
                        .out
                        .iter()
                        .map(|&p| &self.0.pathways[p.index()])
                        .find(|p| p.to == target)
                    {
                        Some(pathway) => Some(pathway),
                        None => {
                            self.0.commands_direct.fetch_add(1, Ordering::Relaxed);
                            return self.drive(target, signal, depth, announced, now);
                        }
                    }
                }
                _ => {
                    self.0.unrouted.fetch_add(1, Ordering::Relaxed);
                    debug!(
                        "nerve: no effector named '{}' for a command from {}",
                        command.effector,
                        cell.neuron.name()
                    );
                    return Outcome {
                        unrouted: 1,
                        ..Outcome::default()
                    };
                }
            },
            None => self.select(cell, &signal, now),
        };
        let Some(pathway) = selected else {
            if cell.out.is_empty() {
                // A terminal neuron: nothing to do with what it fired
                return Outcome::default();
            }
            self.0.no_eligible.fetch_add(1, Ordering::Relaxed);
            return Outcome {
                no_pathway: 1,
                ..Outcome::default()
            };
        };
        pathway.signals.fetch_add(1, Ordering::Relaxed);
        match &pathway.queue {
            None => {
                let outcome = self.drive(pathway.to, signal, depth, announced, now);
                pathway.latency.observe(now.elapsed());
                outcome
            }
            Some(queue) => {
                self.prepare_for(pathway, &signal, now);
                signal.trace.mark_queued();
                // The arrival a worker may be waiting for, if this is the announced pathway
                let arrival = match announced.take() {
                    Some(a) if a.pathway == pathway.id => Some(a.arrival),
                    other => {
                        *announced = other;
                        None
                    }
                };
                match queue.submit(arrival, signal) {
                    Ok(()) => Outcome {
                        queued: 1,
                        ..Outcome::default()
                    },
                    // Full: dropped rather than queued without bound
                    Err(_) => {
                        pathway.overflow.fetch_add(1, Ordering::Relaxed);
                        Outcome {
                            dropped: 1,
                            ..Outcome::default()
                        }
                    }
                }
            }
        }
    }

    /// The short-mode / long-mode rule: among the pathways leaving `cell` whose p99 fits in
    /// the signal's remaining deadline, the heaviest weight
    fn select(&self, cell: &Cell, signal: &Signal, now: Instant) -> Option<&Pathway> {
        let remaining = signal.urgency.remaining(now);
        let mut best: Option<&Pathway> = None;
        for &id in &cell.out {
            let pathway = &self.0.pathways[id.index()];
            if !pathway.fits(remaining) {
                pathway.ineligible.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if best.is_none_or(|b| pathway.weight > b.weight) {
                best = Some(pathway);
            }
        }
        best
    }

    /// Motor preparation: a signal with time to spare that takes a deliberative pathway
    /// warms the neurons it will probably reach, at most once a second each
    fn prepare_for(&self, pathway: &Pathway, signal: &Signal, now: Instant) {
        let far = match (signal.urgency.remaining(now), pathway.latency.p99()) {
            (None, _) | (_, None) => true,
            (Some(remaining), Some(p99)) => remaining > 2 * p99,
        };
        if !far {
            return;
        }
        let nanos = self.nanos(now);
        let to = &self.0.cells[pathway.to.index()];
        let downstream = to
            .out
            .iter()
            .map(|&p| self.0.pathways[p.index()].to)
            .chain(std::iter::once(pathway.to));
        for id in downstream {
            let cell = &self.0.cells[id.index()];
            let last = cell.last_prepared.load(Ordering::Relaxed);
            if nanos.saturating_sub(last) < 1_000_000_000 && last != 0 {
                continue;
            }
            cell.last_prepared.store(nanos.max(1), Ordering::Relaxed);
            let cx = Cx {
                now,
                id,
                fabric: self,
            };
            cell.neuron.prepare(&cx);
        }
    }

    /// Fire `to` with a batch a worker summed, and propagate the results
    fn fire_summed(&self, pathway: PathwayId, batch: Vec<Queued<Signal>>) {
        let pathway = &self.0.pathways[pathway.index()];
        let cell = &self.0.cells[pathway.to.index()];
        let now = Instant::now();
        let nanos = self.nanos(now);
        let mut admitted = Vec::with_capacity(batch.len());
        let mut queued_at = Vec::with_capacity(batch.len());
        for Queued {
            item: mut signal,
            queued_at: at,
        } in batch
        {
            cell.counters.received.fetch_add(1, Ordering::Relaxed);
            match cell.gate.admit(&mut signal, nanos) {
                Ok(()) => {
                    signal.trace.push(pathway.to);
                    if cell.kind == NeuronKind::Brain {
                        signal.trace.mark_brain();
                        self.0.reached_brain.fetch_add(1, Ordering::Relaxed);
                    }
                    admitted.push(signal);
                    queued_at.push(at);
                }
                Err(reason) => {
                    cell.counters.quiet(reason);
                    pathway.latency.observe(now.saturating_duration_since(at));
                }
            }
        }
        if admitted.is_empty() {
            return;
        }
        let cx = Cx {
            now,
            id: pathway.to,
            fabric: self,
        };
        let firings = cell.neuron.stimulate_many(admitted, &cx);
        if firings.iter().all(|f| matches!(f, Firing::Quiet)) {
            cell.gate.stayed_quiet(nanos);
        }
        for (firing, at) in firings.into_iter().zip(queued_at) {
            self.after_firing(pathway.to, firing, 1, &mut None);
            pathway.latency.observe(at.elapsed());
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

    #[cfg(test)]
    pub(crate) fn seed_latency(&self, pathway: usize, latency: Duration) {
        for _ in 0..4 {
            self.0.pathways[pathway].latency.observe(latency);
        }
    }
}

/// A signal announced on a deliberative pathway, see [`Fabric::announce`]
pub(crate) struct Announced<'a> {
    pathway: PathwayId,
    arrival: engine::Arrival<'a, Signal>,
}

/// A deliberative pathway's summation: the batching engine over the pathway's queue, with
/// the window learned from the pathway's latency, firing the pathway's neuron with each
/// batch and propagating what it fires
struct Summing {
    fabric: Weak<Inner>,
    pathway: PathwayId,
}

impl Batching for Summing {
    type Item = Signal;

    /// A signal of tensors counts its items (the batch dimension); any other counts one
    fn weight(&self, signal: &Signal) -> usize {
        signal
            .payload
            .as_tensors()
            .and_then(<[_]>::first)
            .map_or(1, |t| t.batch().max(1))
    }

    /// A signal that exists to answer a request nobody waits for any more is skipped
    fn wanted(&self, signal: &Signal) -> bool {
        signal
            .reply
            .as_ref()
            .is_none_or(super::signal::ReplyHandle::wanted)
    }

    fn window(&self) -> Duration {
        self.fabric
            .upgrade()
            .and_then(|inner| inner.pathways[self.pathway.index()].window())
            .unwrap_or(Duration::ZERO)
    }

    fn run(&self, batch: Vec<Queued<Signal>>) {
        if let Some(inner) = self.fabric.upgrade() {
            Fabric(inner).fire_summed(self.pathway, batch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::neuron::{FnNeuron, Habituation};
    use crate::nerve::signal::{Feature, Payload};
    use crate::nerve::testing::{pass, wait_until};
    use std::sync::atomic::AtomicUsize;
    use std::thread::ThreadId;

    /// Counts what reaches it and remembers the thread it ran on
    struct Sink {
        name: &'static str,
        kind: NeuronKind,
        hits: AtomicUsize,
        thread: Mutex<Option<ThreadId>>,
        queued: AtomicUsize,
        seen: Mutex<Vec<Signal>>,
    }

    impl Sink {
        fn new(name: &'static str, kind: NeuronKind) -> Arc<Self> {
            Arc::new(Self {
                name,
                kind,
                hits: AtomicUsize::new(0),
                thread: Mutex::new(None),
                queued: AtomicUsize::new(0),
                seen: Mutex::new(Vec::new()),
            })
        }

        fn hits(&self) -> usize {
            self.hits.load(Ordering::SeqCst)
        }

        fn wait_for(&self, hits: usize) {
            wait_until(|| self.hits() >= hits);
        }
    }

    impl Neuron for Sink {
        fn name(&self) -> &str {
            self.name
        }
        fn kind(&self) -> NeuronKind {
            self.kind
        }
        fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
            self.hits.fetch_add(1, Ordering::SeqCst);
            *self.thread.lock().unwrap() = Some(std::thread::current().id());
            if signal.trace.was_queued() {
                self.queued.fetch_add(1, Ordering::SeqCst);
            }
            self.seen.lock().unwrap().push(signal);
            Firing::Quiet
        }
    }

    fn receptor(name: &'static str) -> Arc<Sink> {
        Sink::new(name, NeuronKind::Receptor)
    }

    fn signal(bytes: &[u8]) -> Signal {
        Signal::new(Feature::sketch(bytes), Payload::bytes(bytes.to_vec()))
    }

    #[test]
    fn a_reflex_arc_is_a_direct_call_on_the_callers_thread() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let reflex = b.neuron(FnNeuron::reflex("classify", |s, _| {
            Firing::one(s.command("act", 1.into()))
        }));
        let effector = Sink::new("act", NeuronKind::Effector);
        let e = b.add(effector.clone());
        b.reflex(r, reflex);
        b.reflex(reflex, e);
        let fabric = b.build().unwrap();

        let outcome = fabric.stimulate(r, signal(b"stimulus"));
        // The effector ran before stimulate returned, on this thread, never queued
        assert_eq!(effector.hits(), 1);
        assert_eq!(
            *effector.thread.lock().unwrap(),
            Some(std::thread::current().id())
        );
        assert_eq!(effector.queued.load(Ordering::SeqCst), 0);
        assert_eq!(outcome.hops, 3);
        assert_eq!(outcome.acted, 1);
        assert_eq!(outcome.queued, 0);
        let seen = effector.seen.lock().unwrap();
        assert_eq!(seen[0].trace.hops(), &[r, reflex, e]);
        assert!(!seen[0].trace.reached_brain());
        // The arc's stimulus-to-action latency was measured
        let arc = fabric.0.cells[e.index()].arcs[r.index()].get().unwrap();
        assert_eq!(arc.count(), 1);
        assert_eq!(fabric.0.pathways[1].latency.count(), 1);
        assert_eq!(fabric.brain_fraction(), 0.0);
    }

    #[test]
    fn habituation_at_the_receptor_keeps_repeats_from_the_brain() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        b.gate(
            r,
            Mechanisms::NONE
                .threshold(0.6)
                .habituation(Habituation::new(Duration::from_secs(60))),
        );
        let brain = Sink::new("brain", NeuronKind::Brain);
        let bid = b.add(brain.clone());
        b.deliberative(r, bid)
            .summation(Summation::default().window(Duration::ZERO));
        let fabric = b.build().unwrap();

        let mut queued = 0;
        for _ in 0..10 {
            queued += fabric.stimulate(r, signal(b"same")).queued;
        }
        queued += fabric.stimulate(r, signal(b"new")).queued;
        // Only two were queued, so no more than two can reach the brain
        assert_eq!(queued, 2);
        brain.wait_for(2);
        assert_eq!(brain.hits(), 2);
        assert_eq!(brain.queued.load(Ordering::SeqCst), 2);
        let c = &fabric.0.cells[r.index()].counters;
        assert_eq!(c.quiet_habituation.load(Ordering::SeqCst), 9);
        assert_eq!(c.fired.load(Ordering::SeqCst), 2);
        assert!((fabric.brain_fraction() - 2.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn selection_picks_the_heaviest_pathway_that_fits_the_deadline() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let fast = Sink::new("fast", NeuronKind::Effector);
        let slow = Sink::new("slow", NeuronKind::Effector);
        let f = b.add(fast.clone());
        let s = b.add(slow.clone());
        // The slow path is preferred (heavier) when there is time
        b.reflex(r, f).weight(0.5);
        b.reflex(r, s);
        let fabric = b.build().unwrap();
        fabric.seed_latency(0, Duration::from_micros(10));
        fabric.seed_latency(1, Duration::from_millis(50));

        fabric.stimulate(r, signal(b"a"));
        assert_eq!((fast.hits(), slow.hits()), (0, 1));
        // A deadline the slow path cannot meet selects the fast one
        fabric.stimulate(
            r,
            signal(b"b").with_urgency(super::super::signal::Urgency::within(
                Duration::from_millis(5),
            )),
        );
        assert_eq!((fast.hits(), slow.hits()), (1, 1));
        assert_eq!(fabric.0.pathways[1].ineligible.load(Ordering::SeqCst), 1);
        // A deadline already past goes nowhere
        let outcome = fabric.stimulate(
            r,
            signal(b"c").with_urgency(super::super::signal::Urgency::at(Instant::now())),
        );
        assert_eq!(outcome.no_pathway, 1);
        assert_eq!((fast.hits(), slow.hits()), (1, 1));
    }

    #[test]
    fn a_pathway_fits_a_deadline_its_p99_equals() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let e = b.add(Sink::new("act", NeuronKind::Effector));
        b.reflex(r, e);
        let fabric = b.build().unwrap();
        let pathway = &fabric.0.pathways[0];
        // Unmeasured, it fits anything
        assert!(pathway.fits(Some(Duration::ZERO)));
        fabric.seed_latency(0, Duration::from_millis(50));
        assert_eq!(pathway.latency.p99(), Some(Duration::from_millis(50)));
        assert!(pathway.fits(None));
        assert!(pathway.fits(Some(Duration::from_millis(50))));
        assert!(!pathway.fits(Some(Duration::from_millis(50) - Duration::from_nanos(1))));
    }

    #[test]
    fn compute_runs_inline_on_a_reflex_arc_and_its_signals_propagate() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let ran_on = Arc::new(Mutex::new(None));
        let model = b.neuron(FnNeuron::reflex("model", {
            let ran_on = ran_on.clone();
            move |s, _| {
                let ran_on = ran_on.clone();
                Firing::compute(move || {
                    *ran_on.lock().unwrap() = Some(std::thread::current().id());
                    vec![s.command("act", 1.into())]
                })
            }
        }));
        let effector = Sink::new("act", NeuronKind::Effector);
        let e = b.add(effector.clone());
        b.reflex(r, model);
        b.reflex(model, e);
        let fabric = b.build().unwrap();

        let outcome = fabric.stimulate(r, signal(b"x"));
        assert_eq!((outcome.acted, effector.hits()), (1, 1));
        assert_eq!(*ran_on.lock().unwrap(), Some(std::thread::current().id()));
        let c = &fabric.0.cells[model.index()].counters;
        assert_eq!(c.computed.load(Ordering::SeqCst), 1);
        assert_eq!(c.fired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn commands_reach_the_named_effector() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let reflex = b.neuron(FnNeuron::reflex("decide", |s, _| {
            Firing::Fire(vec![
                s.command("webhook", serde_json::json!({"n": 1})),
                s.command("journal", serde_json::json!({"n": 2})),
                s.command("nobody", serde_json::Value::Null),
            ])
        }));
        let webhook = Sink::new("webhook", NeuronKind::Effector);
        let journal = Sink::new("journal", NeuronKind::Effector);
        let w = b.add(webhook.clone());
        b.add(journal.clone());
        b.reflex(r, reflex);
        b.reflex(reflex, w);
        let fabric = b.build().unwrap();

        let outcome = fabric.stimulate(r, signal(b"x"));
        assert_eq!((webhook.hits(), journal.hits()), (1, 1));
        assert_eq!(outcome.unrouted, 1);
        assert_eq!(outcome.acted, 2);
        // The journal had no declared pathway: delivered directly
        assert_eq!(fabric.0.commands_direct.load(Ordering::SeqCst), 1);
        assert_eq!(fabric.0.pathways[1].signals.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn refractory_period_drops_a_burst_after_the_first() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        b.gate(r, Mechanisms::NONE.refractory(Duration::from_secs(5)));
        let e = Sink::new("act", NeuronKind::Effector);
        let eid = b.add(e.clone());
        b.reflex(r, eid);
        let fabric = b.build().unwrap();
        for i in 0..5u8 {
            fabric.stimulate(r, signal(&[i]));
        }
        assert_eq!(e.hits(), 1);
        let c = &fabric.0.cells[r.index()].counters;
        assert_eq!(c.quiet_refractory.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn deliberative_pathway_sums_a_window_into_one_firing() {
        struct Batchy {
            sizes: Mutex<Vec<usize>>,
        }
        impl Neuron for Batchy {
            fn name(&self) -> &str {
                "brain"
            }
            fn kind(&self) -> NeuronKind {
                NeuronKind::Brain
            }
            fn stimulate(&self, _: Signal, _: &Cx<'_>) -> Firing {
                unreachable!("stimulate_many is used")
            }
            fn stimulate_many(&self, signals: Vec<Signal>, _: &Cx<'_>) -> Vec<Firing> {
                self.sizes.lock().unwrap().push(signals.len());
                signals
                    .into_iter()
                    .map(|_| Firing::Fire(Vec::new()))
                    .collect()
            }
        }
        let batchy = Arc::new(Batchy {
            sizes: Mutex::new(Vec::new()),
        });
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let brain = b.add(batchy.clone());
        b.deliberative(r, brain).summation(
            Summation::default()
                .max_batch(4)
                .window(Duration::from_millis(50)),
        );
        let fabric = b.build().unwrap();
        for i in 0..6u8 {
            assert_eq!(fabric.stimulate(r, signal(&[i])).queued, 1);
        }
        // The pathway's latency is measured last, after the batch has fired
        wait_until(|| fabric.0.pathways[0].latency.count() == 6);
        assert_eq!(*batchy.sizes.lock().unwrap(), vec![4, 2]);
        assert_eq!(fabric.waiting(), 0);
    }

    #[test]
    fn a_loop_is_cut() {
        let mut b = Fabric::builder();
        let a = b.neuron(pass("a"));
        let c = b.neuron(FnNeuron::reflex("b", |s, _| Firing::one(s)));
        b.reflex(a, c);
        b.reflex(c, a);
        let fabric = b.build().unwrap();
        let outcome = fabric.stimulate(a, signal(b"round"));
        assert_eq!(outcome.hops, MAX_HOPS);
        assert_eq!(fabric.0.loops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_loop_through_a_queue_is_cut() {
        let mut b = Fabric::builder();
        let reflex = b.neuron(pass("reflex"));
        let brain =
            b.neuron(FnNeuron::reflex("brain", |s, _| Firing::one(s)).with_kind(NeuronKind::Brain));
        b.deliberative(reflex, brain).summation(Summation::none());
        b.reflex(brain, reflex);
        let fabric = b.build().unwrap();
        fabric.stimulate(reflex, signal(b"round"));
        // Each pass through the queue restarts the depth; the trace still counts every hop
        wait_until(|| fabric.0.loops.load(Ordering::SeqCst) > 0);
        assert_eq!(fabric.0.loops.load(Ordering::SeqCst), 1);
        assert!(fabric.0.reached_brain.load(Ordering::SeqCst) < MAX_HOPS as u64);
    }

    #[test]
    fn prepare_runs_once_for_far_signals_on_deliberative_pathways() {
        struct Prep(AtomicUsize);
        impl Neuron for Prep {
            fn name(&self) -> &str {
                "brain"
            }
            fn kind(&self) -> NeuronKind {
                NeuronKind::Brain
            }
            fn stimulate(&self, _: Signal, _: &Cx<'_>) -> Firing {
                Firing::Quiet
            }
            fn prepare(&self, _: &Cx<'_>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let prep = Arc::new(Prep(AtomicUsize::new(0)));
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        let brain = b.add(prep.clone());
        b.deliberative(r, brain).summation(Summation::none());
        let fabric = b.build().unwrap();
        for i in 0..3u8 {
            fabric.stimulate(r, signal(&[i]));
        }
        // Prepared before the first signal was queued; the next two were within the second
        assert_eq!(prep.0.load(Ordering::SeqCst), 1);
        let _ = receptor("unused");
    }

    #[test]
    fn a_builder_with_two_neurons_of_one_name_or_a_foreign_pathway_is_an_error() {
        let mut b = Fabric::builder();
        b.neuron(pass("in"));
        b.neuron(pass("in"));
        assert!(matches!(b.build(), Err(BuildError::DuplicateName(name)) if name == "in"));

        let mut other = Fabric::builder();
        let a = other.neuron(pass("a"));
        let b2 = other.neuron(pass("b"));
        let mut b = Fabric::builder();
        let only = b.neuron(pass("only"));
        b.reflex(only, b2);
        assert!(matches!(
            b.build(),
            Err(BuildError::UnknownNeuron { from, to }) if from == only && to == b2
        ));
        let _ = a;
    }

    #[test]
    fn the_builder_defaults_wire_a_reflex_arc_and_prefer_the_brain() {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"));
        // Option<Signal>: a command for alerts, quiet for the rest
        let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
            let text = signal.payload.as_text()?;
            text.contains("ALERT")
                .then(|| signal.command("alarm", serde_json::json!({"text": text})))
        }));
        let (channel, _deliveries) = crate::nerve::effector::ChannelOut::new("alarm", 4);
        // No pathway to the effector: the command finds it by name
        b.effector(channel);
        let brain = b.add(Sink::new("brain", NeuronKind::Brain));
        b.reflex(r, triage);
        b.deliberative(r, brain);
        b.reflex(triage, brain).weight(0.5);
        let fabric = b.build().unwrap();

        let weights: Vec<(PathwayKind, f32)> = fabric
            .0
            .pathways
            .iter()
            .map(|p| (p.kind, p.weight))
            .collect();
        assert_eq!(
            weights,
            [
                (PathwayKind::Reflex, 1.0),
                (PathwayKind::Deliberative, 2.0),
                (PathwayKind::Reflex, 0.5)
            ]
        );
        let text = |t: &str| Signal::new(Feature::sketch(t.as_bytes()), Payload::text(t));
        let acted = fabric.stimulate(triage, text("ALERT: smoke"));
        assert_eq!((acted.acted, acted.quiet), (1, None));
        let quiet = fabric.stimulate(triage, text("all is well"));
        assert_eq!(
            (quiet.acted, quiet.quiet),
            (0, Some((triage, Quiet::NotInterested)))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_motor_side_can_run_on_a_runtime_of_ones_own() {
        let mut b = Fabric::builder().efferent(tokio::runtime::Handle::current());
        b.neuron(pass("in"));
        let fabric = b.build().unwrap();
        assert_eq!(
            fabric.efferent().id(),
            tokio::runtime::Handle::current().id()
        );
        // Dropping it on this runtime leaves the runtime alone
        drop(fabric);
        tokio::task::yield_now().await;
    }
}
