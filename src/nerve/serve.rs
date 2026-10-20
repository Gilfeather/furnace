//! The server's own pathways: every model the server serves is a neuron of its nervous system.
//!
//! For each model the server wires an entrance (the neuron the protocols stimulate: `/predict`,
//! the Open Inference Protocol over HTTP and gRPC), a pathway from it to the model's neuron, and
//! a reflex pathway from there to the `respond` effector, which answers the request with the
//! model's outputs. A failed inference answers with its own error.
//!
//! With batching on (`--max-batch-size` above 1) and a model that can join requests along the
//! batch dimension, the pathway to the model is deliberative: concurrent requests sum into one
//! forward pass, a request whose body is still arriving is announced so a batch about to start
//! waits for it, and a lone request never waits. Otherwise the pathway is a reflex: the request
//! runs its forward pass on a blocking thread of its own, as a hand-written server would.

use std::collections::HashMap;
use std::sync::Arc;

use tracing::error;

use super::effector::{Effector, Respond};
use super::fabric::{Announced, Fabric, FabricBuilder, Summation};
use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Mechanisms, Neuron, NeuronKind};
use super::signal::{Answer, Feature, NeuronId, PathwayId, Payload, Reply, Signal};
use crate::batching::BatchConfig;
use crate::error::Result;
use crate::model::Model;
use crate::tensor::HostTensor;

/// Where a served neuron gets its model: the one currently loaded, so a reload swaps it under
/// the pathway without rebuilding the fabric
pub(crate) type ModelSource = Arc<dyn Fn() -> Arc<Model> + Send + Sync>;

/// The name of the effector that answers the server's requests
pub const RESPOND: &str = "respond";

/// The neuron a protocol stimulates for `model`
fn entrance_name(model: &str) -> String {
    format!("in:{model}")
}

/// The neuron that runs `model`
fn model_name(model: &str) -> String {
    format!("model:{model}")
}

/// A protocol's way in: passes the request's signal on
struct Entrance {
    name: String,
}

impl Neuron for Entrance {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Receptor
    }

    fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
        Firing::one(signal)
    }
}

/// A served model as a neuron: tensors in, tensors out
struct Served {
    name: String,
    source: ModelSource,
    latency: LatencyEstimate,
}

impl Served {
    /// The request's tensors, taken out of the signal without a copy when it holds the only
    /// handle to them
    fn take_inputs(signal: &mut Signal) -> Option<Vec<HostTensor>> {
        match std::mem::take(&mut signal.payload) {
            Payload::Tensors(tensors) => {
                Some(Arc::try_unwrap(tensors).unwrap_or_else(|shared| (*shared).clone()))
            }
            other => {
                signal.payload = other;
                None
            }
        }
    }

    /// The outputs go on to the `respond` effector; a failure answers the request itself
    fn fire(signal: Signal, result: Result<Vec<HostTensor>>) -> Firing {
        match result {
            Ok(outputs) => Firing::one(signal.derive(Payload::tensors(outputs))),
            Err(e) => {
                signal.answer(Answer::failed(e));
                Firing::Fire(Vec::new())
            }
        }
    }
}

impl Neuron for Served {
    fn name(&self) -> &str {
        &self.name
    }

    fn stimulate(&self, mut signal: Signal, _cx: &Cx<'_>) -> Firing {
        let Some(inputs) = Self::take_inputs(&mut signal) else {
            return Firing::Quiet;
        };
        let started = std::time::Instant::now();
        let result = (self.source)().infer(inputs);
        self.latency.observe(started.elapsed());
        Self::fire(signal, result)
    }

    /// A summed window: one forward pass for every request in it, each checked on its own
    fn stimulate_many(&self, signals: Vec<Signal>, _cx: &Cx<'_>) -> Vec<Firing> {
        let mut firings: Vec<Option<Firing>> = (0..signals.len()).map(|_| None).collect();
        let mut kept = Vec::with_capacity(signals.len());
        let mut requests = Vec::with_capacity(signals.len());
        for (i, mut signal) in signals.into_iter().enumerate() {
            match Self::take_inputs(&mut signal) {
                Some(inputs) => {
                    requests.push(inputs);
                    kept.push((i, signal));
                }
                None => firings[i] = Some(Firing::Quiet),
            }
        }
        if !requests.is_empty() {
            let started = std::time::Instant::now();
            let results = (self.source)().infer_many(requests);
            self.latency.observe(started.elapsed());
            for ((i, signal), result) in kept.into_iter().zip(results) {
                firings[i] = Some(Self::fire(signal, result));
            }
        }
        firings
            .into_iter()
            .map(|f| f.unwrap_or(Firing::Quiet))
            .collect()
    }

    fn describe(&self) -> serde_json::Value {
        let model = (self.source)();
        let info = model.get_info();
        serde_json::json!({
            "model": info.name,
            "backend": info.backend,
            "inputs": model.inputs().iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            "outputs": model.outputs().iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            "forward": self.latency.snapshot(),
        })
    }
}

/// A model to wire into the fabric
pub(crate) struct ServedModel {
    pub name: String,
    pub source: ModelSource,
    /// Whether concurrent requests can be joined into one forward pass
    pub batches: bool,
}

/// How a model's requests enter the fabric
#[derive(Clone, Copy, Debug)]
struct Route {
    entrance: NeuronId,
    /// From the entrance to the model
    pathway: PathwayId,
    deliberative: bool,
}

/// Why a request got no answer of the model's
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The model's queue was full
    Busy,
    /// The forward pass panicked, or the chain ended without answering
    Lost,
}

/// The server's models wired into its fabric
pub(crate) struct Serving {
    fabric: Fabric,
    routes: HashMap<String, Route>,
}

impl Serving {
    /// Add an entrance and a neuron for each model, and the `respond` effector, to `builder`;
    /// then build the fabric. `capacity` bounds each deliberative queue (the server's
    /// concurrency limit already bounds what can wait).
    pub(crate) fn build(
        mut builder: FabricBuilder,
        models: Vec<ServedModel>,
        batching: BatchConfig,
        capacity: usize,
    ) -> Self {
        let respond = builder.neuron(Effector::new(Respond::new(RESPOND)), Mechanisms::NONE);
        let mut wired = Vec::with_capacity(models.len());
        for model in models {
            let entrance = builder.neuron(
                Entrance {
                    name: entrance_name(&model.name),
                },
                Mechanisms::NONE,
            );
            let served = builder.neuron(
                Served {
                    name: model_name(&model.name),
                    source: model.source,
                    latency: LatencyEstimate::new(),
                },
                Mechanisms::NONE,
            );
            let deliberative = batching.enabled() && model.batches;
            let pathway = builder.next_pathway();
            if deliberative {
                builder.deliberative(
                    entrance,
                    served,
                    1.0,
                    Summation::default()
                        .max_batch(batching.max_batch_size)
                        .window(batching.batch_timeout)
                        .workers(batching.workers)
                        .capacity(capacity.max(1))
                        .announced_only(),
                );
            } else {
                builder.reflex(entrance, served, 1.0);
            }
            builder.reflex(served, respond, 1.0);
            wired.push((
                model.name,
                Route {
                    entrance,
                    pathway,
                    deliberative,
                },
            ));
        }
        Self {
            fabric: builder.build(),
            routes: wired.into_iter().collect(),
        }
    }

    pub(crate) fn fabric(&self) -> &Fabric {
        &self.fabric
    }

    /// Announce a request for `model` whose body is still being received or decoded, so a
    /// batch about to start waits for it; `None` when the model's requests are not batched
    pub(crate) fn announce(&self, model: &str) -> Option<Announced<'_>> {
        let route = self.routes.get(model)?;
        route
            .deliberative
            .then(|| self.fabric.announce(route.pathway))
            .flatten()
    }

    /// Run one request of `model`: its tensors through the model's pathway, and its outputs
    /// or its own error back
    pub(crate) async fn infer(
        &self,
        model: &str,
        inputs: Vec<HostTensor>,
        announced: Option<Announced<'_>>,
    ) -> std::result::Result<Result<Vec<HostTensor>>, Refused> {
        let Some(&route) = self.routes.get(model) else {
            error!("nerve: no pathway for model '{model}'");
            return Err(Refused::Lost);
        };
        let (reply, receiver) = Reply::required();
        let signal = Signal::new(Feature::NONE, Payload::tensors(inputs)).with_reply(reply);
        let outcome = if route.deliberative {
            self.fabric
                .stimulate_announced(route.entrance, signal, announced)
        } else {
            // The forward pass runs on the arc, so off the async runtime's threads
            drop(announced);
            let fabric = self.fabric.clone();
            match tokio::task::spawn_blocking(move || fabric.stimulate(route.entrance, signal))
                .await
            {
                Ok(outcome) => outcome,
                Err(e) => {
                    error!("Inference task failed: {e}");
                    return Err(Refused::Lost);
                }
            }
        };
        match receiver.await {
            Ok(Answer::Tensors(outputs)) => Ok(Ok(outputs)),
            Ok(Answer::Failed(e)) => Ok(Err(*e)),
            Ok(Answer::Http { .. }) => {
                error!("nerve: model '{model}' was answered with an HTTP response");
                Err(Refused::Lost)
            }
            Err(_) if outcome.dropped > 0 => Err(Refused::Busy),
            Err(_) => Err(Refused::Lost),
        }
    }
}
