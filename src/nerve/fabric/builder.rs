//! Building a fabric: neurons, receptors, effectors, and the pathways between them

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

use super::pathway::{Pathway, PathwayKind, Summation};
use super::summing::Summing;
use super::{Cell, Counters, Fabric, Inner};
use crate::nerve::effector::Effector;
use crate::nerve::latency::LatencyEstimate;
use crate::nerve::neuron::{Gate, Mechanisms, Neuron, NeuronKind};
use crate::nerve::receptor::{Receptor, Stop};
use crate::nerve::signal::{NeuronId, PathwayId};
use furnace_batch::Engine;

/// Why a part of the nervous system cannot be built
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// Two neurons have the same name; effectors are addressed by name
    DuplicateName(String),
    /// A pathway names a neuron that was not added to this builder
    UnknownNeuron {
        /// The pathway's source
        from: NeuronId,
        /// The pathway's target
        to: NeuronId,
    },
    /// A URL the nervous system's HTTP client cannot call: not `http://host[:port]/path`
    /// (HTTPS among them)
    Url {
        /// The effector or receptor
        name: String,
        /// What is wrong with the URL
        reason: String,
    },
    /// A runtime of the fabric could not be started
    Runtime(&'static str, std::io::Error),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "two neurons are named '{name}'"),
            Self::UnknownNeuron { from, to } => {
                write!(
                    f,
                    "pathway {from} -> {to} names a neuron that was not added"
                )
            }
            Self::Url { name, reason } => write!(f, "{name}: {reason}"),
            Self::Runtime(which, e) => write!(f, "the {which} runtime: {e}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(_, e) => Some(e),
            _ => None,
        }
    }
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

    /// Add an effector: an [`Act`](crate::nerve::effector::Act) with the default local loop
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
                            furnace_batch::Config {
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
