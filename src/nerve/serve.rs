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
//!
//! An LLM (feature `llm`) is the brain: the OpenAI routes stimulate its entrance `in:{llm}`,
//! and its neuron `llm:{llm}` asks the engine to generate. The tokens do not hop from neuron to
//! neuron: the engine's stream goes straight back to the route, which writes it to the client.

use std::collections::HashMap;
use std::sync::Arc;

use tracing::error;

use super::effector::{Effector, Respond};
use super::fabric::{Announced, Fabric, FabricBuilder, Outcome, Summation};
use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Mechanisms, Neuron, NeuronKind};
use super::signal::{Answer, Feature, NeuronId, PathwayId, Payload, ReplyHandle, Signal};
use crate::batching::BatchConfig;
use crate::error::Result;
use crate::model::Model;
use crate::tensor::HostTensor;

/// Where a served neuron gets its model: the one currently loaded, so a reload swaps it under
/// the pathway without rebuilding the fabric
pub(crate) type ModelSource = Arc<dyn Fn() -> Arc<Model> + Send + Sync>;

/// The name of the effector that answers the server's requests; prefixed, so that it never
/// clashes with a `respond` effector of your own
pub const RESPOND: &str = "serve:respond";

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
            "forward_latency_ms": self.latency.snapshot(),
        })
    }
}

/// An LLM as the brain: a generation request in, the engine's stream of events back
#[cfg(feature = "llm")]
struct Speaking {
    name: String,
    engine: Arc<dyn crate::llm::LlmEngine>,
}

#[cfg(feature = "llm")]
impl Neuron for Speaking {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Brain
    }

    /// Asks the engine, which must not block (it is the route's thread), and answers with the
    /// stream itself
    fn stimulate(&self, mut signal: Signal, _cx: &Cx<'_>) -> Firing {
        let request = match std::mem::take(&mut signal.payload) {
            Payload::Prompt(request) => {
                Arc::try_unwrap(request).unwrap_or_else(|shared| (*shared).clone())
            }
            other => {
                signal.payload = other;
                return Firing::Quiet;
            }
        };
        signal.answer(Answer::Generation(self.engine.generate(request)));
        Firing::Fire(Vec::new())
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "engine": self.engine.name(),
            "max_context": self.engine.max_context(),
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

/// A request's chain: handed to the fabric already, or running on a blocking thread
enum Chain {
    Ran(Outcome),
    Running(tokio::task::JoinHandle<Outcome>),
}

/// The server's models wired into its fabric
pub(crate) struct Serving {
    fabric: Fabric,
    routes: HashMap<String, Route>,
    /// Each LLM's entrance
    #[cfg(feature = "llm")]
    llms: HashMap<String, NeuronId>,
}

impl Serving {
    /// Add an entrance and a neuron for each model, and the `respond` effector, to `builder`;
    /// then build the fabric. `capacity` bounds each deliberative queue (the server's
    /// concurrency limit already bounds what can wait).
    pub(crate) fn build(
        mut builder: FabricBuilder,
        models: Vec<ServedModel>,
        #[cfg(feature = "llm")] llms: &[Arc<dyn crate::llm::LlmEngine>],
        batching: BatchConfig,
        capacity: usize,
    ) -> std::result::Result<Self, crate::nerve::BuildError> {
        // Only a server with models answers requests through the fabric
        let respond = (!models.is_empty())
            .then(|| builder.neuron(Effector::new(Respond::new(RESPOND)), Mechanisms::NONE));
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
            if let Some(respond) = respond {
                builder.reflex(served, respond, 1.0);
            }
            wired.push((
                model.name,
                Route {
                    entrance,
                    pathway,
                    deliberative,
                },
            ));
        }
        #[cfg(feature = "llm")]
        let llms = llms
            .iter()
            .map(|engine| {
                let name = engine.name().to_string();
                let entrance = builder.neuron(
                    Entrance {
                        name: entrance_name(&name),
                    },
                    Mechanisms::NONE,
                );
                let brain = builder.neuron(
                    Speaking {
                        name: format!("llm:{name}"),
                        engine: engine.clone(),
                    },
                    Mechanisms::NONE,
                );
                builder.reflex(entrance, brain, 1.0);
                (name, entrance)
            })
            .collect();
        Ok(Self {
            fabric: builder.build()?,
            routes: wired.into_iter().collect(),
            #[cfg(feature = "llm")]
            llms,
        })
    }

    /// [`generate`](Self::generate) on a blocking thread: an engine renders the chat template
    /// and tokenizes the prompt before it returns, milliseconds for a long prompt during which
    /// a runtime worker would deliver no other stream's tokens
    #[cfg(feature = "llm")]
    pub(crate) async fn generate_off_runtime(
        self: Arc<Self>,
        name: String,
        request: crate::llm::GenerateRequest,
    ) -> std::result::Result<tokio::sync::mpsc::Receiver<crate::llm::Event>, crate::llm::EngineError>
    {
        tokio::task::spawn_blocking(move || self.generate(&name, request))
            .await
            .unwrap_or_else(|_| Err(crate::llm::EngineError::failed("the engine panicked")))
    }

    /// Ask the LLM `name` to generate, through its pathway: the engine's stream of events, or
    /// why it refused
    #[cfg(feature = "llm")]
    pub(crate) fn generate(
        &self,
        name: &str,
        request: crate::llm::GenerateRequest,
    ) -> std::result::Result<tokio::sync::mpsc::Receiver<crate::llm::Event>, crate::llm::EngineError>
    {
        let lost = || {
            crate::llm::EngineError::failed(format!("the generation for '{name}' got no answer"))
        };
        let Some(&entrance) = self.llms.get(name) else {
            return Err(lost());
        };
        let (reply, mut receiver) = ReplyHandle::required();
        let signal =
            Signal::new(Feature::NONE, Payload::Prompt(Arc::new(request))).with_reply(reply);
        // The chain is synchronous: the brain has answered when this returns
        self.fabric.stimulate(entrance, signal);
        match receiver.try_recv() {
            Ok(Answer::Generation(result)) => result,
            _ => Err(lost()),
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
        let (reply, receiver) = ReplyHandle::required();
        let signal = Signal::new(Feature::NONE, Payload::tensors(inputs)).with_reply(reply);
        let chain = if route.deliberative {
            Chain::Ran(
                self.fabric
                    .stimulate_announced(route.entrance, signal, announced),
            )
        } else {
            // The forward pass runs on the arc, so off the async runtime's threads
            drop(announced);
            let fabric = self.fabric.clone();
            Chain::Running(tokio::task::spawn_blocking(move || {
                fabric.stimulate(route.entrance, signal)
            }))
        };
        // The answer, as soon as the effector sends it: the chain's bookkeeping after that
        // (counters, latencies) is not waited for. Waking on the task's end instead costs a
        // few microseconds a request, more than the whole chain on the fabric.
        match receiver.await {
            Ok(Answer::Tensors(outputs)) => Ok(Ok(outputs)),
            Ok(Answer::Failed(e)) => Ok(Err(*e)),
            #[cfg(feature = "llm")]
            Ok(Answer::Generation(_)) => {
                error!("nerve: model '{model}' was answered with a generation");
                Err(Refused::Lost)
            }
            Ok(Answer::Http { .. }) => {
                error!("nerve: model '{model}' was answered with an HTTP response");
                Err(Refused::Lost)
            }
            // No answer: the chain ended, so its outcome says why
            Err(_) => {
                let outcome = match chain {
                    Chain::Ran(outcome) => outcome,
                    Chain::Running(task) => match task.await {
                        Ok(outcome) => outcome,
                        Err(e) => {
                            error!("Inference task failed: {e}");
                            return Err(Refused::Lost);
                        }
                    },
                };
                Err(if outcome.dropped > 0 {
                    Refused::Busy
                } else {
                    Refused::Lost
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use crate::model::BurnModel;
    use crate::tensor::{IoSpec, Values};
    use std::sync::Condvar;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    const WIDTH: usize = 4;

    /// Doubles its input and records every batch size it sees, and the most passes that ran
    /// at once. It can be slowed down or held, and it panics on an input of -1, so tests can
    /// check what was summed with what.
    #[derive(Debug)]
    struct Doubler {
        batches: Arc<Mutex<Vec<usize>>>,
        delay: Duration,
        running: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        holds: Arc<Holds>,
    }

    /// Input values whose passes wait until released, so a test can keep workers busy for
    /// exactly as long as it needs without depending on timing
    #[derive(Debug, Default)]
    struct Holds {
        held: Mutex<Vec<f32>>,
        released: Condvar,
    }

    impl Holds {
        fn hold(&self, value: f32) {
            self.held.lock().unwrap().push(value);
        }

        fn release(&self, value: f32) {
            self.held.lock().unwrap().retain(|&v| v != value);
            self.released.notify_all();
        }

        /// Block while any of the values is held
        fn wait(&self, values: &[f32]) {
            let mut held = self.held.lock().unwrap();
            while values.iter().any(|v| held.contains(v)) {
                held = self.released.wait(held).unwrap();
            }
        }
    }

    impl BurnModel for Doubler {
        fn name(&self) -> &str {
            "doubler"
        }
        fn backend(&self) -> BackendKind {
            BackendKind::Cpu
        }
        fn inputs(&self) -> Vec<IoSpec> {
            vec![IoSpec::fp32("input", &[WIDTH])]
        }
        fn outputs(&self) -> Vec<IoSpec> {
            vec![IoSpec::fp32("output", &[WIDTH])]
        }
        fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
            let input = inputs.into_iter().next().unwrap();
            self.batches.lock().unwrap().push(input.batch());
            let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            let Values::F32(values) = input.values else {
                panic!("f32 input");
            };
            self.holds.wait(&values);
            std::thread::sleep(self.delay);
            self.running.fetch_sub(1, Ordering::SeqCst);
            assert!(!values.contains(&-1.0), "poisoned input");
            Ok(vec![HostTensor {
                shape: input.shape,
                values: Values::F32(values.iter().map(|v| v * 2.0).collect()),
            }])
        }
    }

    struct Setup {
        serving: Arc<Serving>,
        model: Arc<Model>,
        batches: Arc<Mutex<Vec<usize>>>,
        running: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        holds: Arc<Holds>,
    }

    fn serving_of(model: Arc<Model>, batching: BatchConfig) -> Arc<Serving> {
        let name = model.get_info().name.clone();
        let source = model.clone();
        Arc::new(
            Serving::build(
                FabricBuilder::new(),
                vec![ServedModel {
                    name,
                    batches: model.is_simple() || model.batches_named(),
                    source: Arc::new(move || source.clone()),
                }],
                #[cfg(feature = "llm")]
                &[],
                batching,
                1000,
            )
            .unwrap(),
        )
    }

    fn setup(max_batch_size: usize, batch_timeout: Duration, delay: Duration) -> Setup {
        setup_with_workers(1, max_batch_size, batch_timeout, delay)
    }

    fn setup_with_workers(
        workers: usize,
        max_batch_size: usize,
        batch_timeout: Duration,
        delay: Duration,
    ) -> Setup {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let holds = Arc::new(Holds::default());
        let model = Arc::new(Model::new(
            Box::new(Doubler {
                batches: batches.clone(),
                delay,
                running: running.clone(),
                peak: peak.clone(),
                holds: holds.clone(),
            }),
            "test",
            0,
        ));
        let serving = serving_of(
            model.clone(),
            BatchConfig {
                max_batch_size,
                batch_timeout,
                workers,
            },
        );
        Setup {
            serving,
            model,
            batches,
            running,
            peak,
            holds,
        }
    }

    /// One request whose items are filled with the given values
    fn request(items: &[f32]) -> Vec<HostTensor> {
        vec![HostTensor {
            shape: vec![items.len(), WIDTH],
            values: Values::F32(items.iter().flat_map(|&v| [v; WIDTH]).collect()),
        }]
    }

    fn expected(items: &[f32]) -> Vec<f32> {
        items.iter().flat_map(|&v| [2.0 * v; WIDTH]).collect()
    }

    fn values(result: std::result::Result<Result<Vec<HostTensor>>, Refused>) -> Vec<f32> {
        let outputs = result.expect("answered").expect("inferred");
        match outputs.into_iter().next().unwrap().values {
            Values::F32(v) => v,
            Values::I64(_) => panic!("f32 output"),
        }
    }

    type Answered = tokio::task::JoinHandle<std::result::Result<Result<Vec<HostTensor>>, Refused>>;

    impl Setup {
        /// Run a request on its own task, announced first as the server does
        fn spawn(&self, items: &[f32]) -> Answered {
            let serving = self.serving.clone();
            let inputs = request(items);
            tokio::spawn(async move {
                let announced = serving.announce("doubler");
                serving.infer("doubler", inputs, announced).await
            })
        }

        fn batches(&self) -> Vec<usize> {
            self.batches.lock().unwrap().clone()
        }
    }

    /// Let the worker pick up what was submitted so far
    async fn settle() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    #[tokio::test]
    async fn concurrent_requests_get_their_own_outputs_in_order() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        // The first request occupies the worker while the rest queue up behind it
        let first = s.spawn(&[100.0]);
        settle().await;
        let requests: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let i = i as f32;
                if i == 2.0 {
                    vec![i, i + 0.5, i + 0.25] // a client-side batch of 3
                } else {
                    vec![i]
                }
            })
            .collect();
        let answers: Vec<_> = requests.iter().map(|r| s.spawn(r)).collect();
        assert_eq!(values(first.await.unwrap()), expected(&[100.0]));
        for (request, answer) in requests.iter().zip(answers) {
            assert_eq!(values(answer.await.unwrap()), expected(request));
        }
        // 1 alone, then 8 items fill the next batch
        assert_eq!(s.batches(), vec![1, 8]);
        let stats = s.model.get_stats();
        assert_eq!(stats.success_count, 7);
        assert_eq!(stats.batch_count, 2);
        assert_eq!(stats.batched_items, 9);
    }

    #[tokio::test]
    async fn invalid_request_fails_alone() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        let first = s.spawn(&[1.0]);
        settle().await;
        let good = s.spawn(&[2.0]);
        let not_finite = s.spawn(&[f32::NAN]);
        let serving = s.serving.clone();
        let wrong_length = tokio::spawn(async move {
            let inputs = vec![HostTensor {
                shape: vec![1, WIDTH],
                values: Values::F32(vec![1.0; WIDTH + 1]),
            }];
            serving.infer("doubler", inputs, None).await
        });
        let also_good = s.spawn(&[3.0]);

        assert_eq!(values(first.await.unwrap()), expected(&[1.0]));
        assert_eq!(values(good.await.unwrap()), expected(&[2.0]));
        let err = not_finite.await.unwrap().unwrap().unwrap_err();
        assert!(err.is_client_error(), "{err}");
        assert!(err.to_string().contains("non-finite"), "{err}");
        assert!(wrong_length.await.unwrap().unwrap().is_err());
        assert_eq!(values(also_good.await.unwrap()), expected(&[3.0]));
        assert_eq!(s.batches(), vec![1, 2]);
        assert_eq!(s.model.get_stats().error_count, 2);
    }

    #[tokio::test]
    async fn batches_respect_max_size_and_keep_order() {
        let s = setup(4, Duration::ZERO, Duration::from_millis(30));
        let first = s.spawn(&[0.0]);
        settle().await;
        let sizes = [3, 2, 6, 1, 1];
        // Polled in order, so they are queued in order
        let answers = futures_util::future::join_all(sizes.iter().enumerate().map(|(i, &n)| {
            s.serving
                .infer("doubler", request(&vec![i as f32; n]), None)
        }))
        .await;
        values(first.await.unwrap());
        for (i, (&n, answer)) in sizes.iter().zip(answers).enumerate() {
            assert_eq!(values(answer), expected(&vec![i as f32; n]));
        }
        // 3 (2 does not fit), 2, then 6 alone (above the maximum), then 1 + 1
        assert_eq!(s.batches(), vec![1, 3, 2, 6, 2]);
    }

    #[tokio::test]
    async fn lone_request_does_not_wait_for_the_window() {
        let s = setup(8, Duration::from_secs(5), Duration::ZERO);
        let start = Instant::now();
        assert_eq!(values(s.spawn(&[1.0]).await.unwrap()), expected(&[1.0]));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn waits_for_a_request_on_its_way_in() {
        let s = setup(8, Duration::from_secs(5), Duration::ZERO);
        // A second request is still being received while the first is queued
        let arriving = s.serving.announce("doubler");
        let first = s.spawn(&[1.0]);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let second = s.serving.infer("doubler", request(&[2.0]), arriving).await;
        assert_eq!(values(first.await.unwrap()), expected(&[1.0]));
        assert_eq!(values(second), expected(&[2.0]));
        assert_eq!(s.batches(), vec![2]);
    }

    #[tokio::test]
    async fn abandoned_arrival_ends_the_wait() {
        let s = setup(8, Duration::from_secs(5), Duration::ZERO);
        let arriving = s.serving.announce("doubler");
        let start = Instant::now();
        let first = s.spawn(&[1.0]);
        settle().await;
        // E.g. its body failed to decode
        drop(arriving);
        assert_eq!(values(first.await.unwrap()), expected(&[1.0]));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn wait_is_bounded_by_the_timeout() {
        let s = setup(8, Duration::from_millis(20), Duration::ZERO);
        let _never_submitted = s.serving.announce("doubler");
        let start = Instant::now();
        assert_eq!(values(s.spawn(&[1.0]).await.unwrap()), expected(&[1.0]));
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(20), "{elapsed:?}");
        assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
    }

    #[tokio::test]
    async fn abandoned_request_is_skipped() {
        let s = setup(8, Duration::ZERO, Duration::ZERO);
        // The first pass is held, so the next two queue behind it
        s.holds.hold(1.0);
        let first = s.spawn(&[1.0]);
        let deadline = Instant::now() + Duration::from_secs(10);
        while s.running.load(Ordering::SeqCst) < 1 {
            assert!(Instant::now() < deadline, "the first never ran");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        // The client times out while queued: its future is dropped with its reply
        let gave_up = tokio::time::timeout(
            Duration::from_millis(1),
            s.serving.infer("doubler", request(&[2.0]), None),
        )
        .await;
        assert!(gave_up.is_err());
        let kept = s.spawn(&[3.0]);
        settle().await;
        s.holds.release(1.0);
        values(first.await.unwrap());
        assert_eq!(values(kept.await.unwrap()), expected(&[3.0]));
        assert_eq!(s.batches(), vec![1, 1]);
    }

    #[tokio::test]
    async fn idle_workers_run_requests_side_by_side() {
        let s = setup_with_workers(2, 8, Duration::from_secs(5), Duration::ZERO);
        // The first two passes are held, so both run until released
        s.holds.hold(1.0);
        s.holds.hold(2.0);
        let first = s.spawn(&[1.0]);
        let second = s.spawn(&[2.0]);
        // Each went to an idle worker: both passes run side by side
        let deadline = Instant::now() + Duration::from_secs(10);
        while s.running.load(Ordering::SeqCst) < 2 {
            assert!(
                Instant::now() < deadline,
                "the first two never ran together"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        // Both workers are busy, so these two wait in the queue
        let third = s.spawn(&[3.0]);
        let fourth = s.spawn(&[4.0]);
        settle().await;
        // The worker that frees up takes both as one batch while the other is still busy.
        // (Freeing both at once could leave two idle workers, which rightly run them apart.)
        s.holds.release(1.0);
        assert_eq!(values(first.await.unwrap()), expected(&[1.0]));
        assert_eq!(values(third.await.unwrap()), expected(&[3.0]));
        assert_eq!(values(fourth.await.unwrap()), expected(&[4.0]));
        s.holds.release(2.0);
        assert_eq!(values(second.await.unwrap()), expected(&[2.0]));
        assert_eq!(s.peak.load(Ordering::SeqCst), 2);
        assert_eq!(s.batches(), vec![1, 1, 2]);
    }

    #[tokio::test]
    async fn panicking_batch_fails_only_its_requests() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        let first = s.spawn(&[1.0]);
        settle().await;
        let poisoned = s.spawn(&[-1.0]);
        let neighbour = s.spawn(&[2.0]);
        assert!(first.await.unwrap().is_ok());
        // Both were in the batch that panicked, so neither was answered
        assert_eq!(poisoned.await.unwrap().unwrap_err(), Refused::Lost);
        assert_eq!(neighbour.await.unwrap().unwrap_err(), Refused::Lost);
        // The worker survives
        assert_eq!(values(s.spawn(&[3.0]).await.unwrap()), expected(&[3.0]));
    }

    #[tokio::test]
    async fn named_tensor_requests_arriving_together_share_a_batch() {
        let model = Arc::new(Model::new(
            Box::new(crate::model::DummyTensorsModel::new(None)),
            "dummy",
            0,
        ));
        let name = model.get_info().name.clone();
        let serving = serving_of(
            model.clone(),
            BatchConfig {
                max_batch_size: 8,
                batch_timeout: Duration::from_secs(5),
                workers: 1,
            },
        );
        let request = |id: i64| {
            vec![
                HostTensor {
                    shape: vec![1, 4],
                    values: Values::I64(vec![id; 4]),
                },
                HostTensor {
                    shape: vec![1],
                    values: Values::F32(vec![1.0]),
                },
            ]
        };
        // All three are announced before any is queued, so the worker waits for them all
        let arrivals: Vec<_> = (0..3).map(|_| serving.announce(&name)).collect();
        let answers = futures_util::future::join_all(
            arrivals
                .into_iter()
                .zip(1..)
                .map(|(announced, id)| serving.infer(&name, request(id), announced)),
        )
        .await;
        for (answer, id) in answers.into_iter().zip(1i64..) {
            let outputs = answer.unwrap().unwrap();
            assert_eq!(outputs[1].values, Values::I64(vec![4 * id]));
        }
        let stats = model.get_stats();
        assert_eq!((stats.batch_count, stats.batched_items), (1, 3));
    }

    #[tokio::test]
    async fn without_batching_every_request_runs_alone_at_once() {
        let s = setup_with_workers(1, 1, Duration::from_secs(5), Duration::from_millis(30));
        assert!(s.serving.announce("doubler").is_none());
        let answers: Vec<_> = (0..4).map(|i| s.spawn(&[i as f32])).collect();
        for (i, answer) in answers.into_iter().enumerate() {
            assert_eq!(values(answer.await.unwrap()), expected(&[i as f32]));
        }
        assert_eq!(s.batches(), vec![1, 1, 1, 1]);
        // Each on a blocking thread of its own, side by side
        assert!(s.peak.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn the_server_s_neurons_are_named_apart_from_yours() {
        let model = Arc::new(Model::new(
            Box::new(crate::model::DummyModel::new("m".into(), vec![2], vec![2])),
            "dummy",
            0,
        ));
        let mut yours = FabricBuilder::new();
        yours.neuron(Effector::new(Respond::new("respond")), Mechanisms::NONE);
        let source = model.clone();
        let serving = Serving::build(
            yours,
            vec![ServedModel {
                name: "m".into(),
                batches: true,
                source: Arc::new(move || source.clone()),
            }],
            #[cfg(feature = "llm")]
            &[],
            BatchConfig::default(),
            10,
        )
        .unwrap();
        let names: Vec<String> = serving
            .fabric()
            .neurons()
            .map(|(_, n)| n.name().to_string())
            .collect();
        for name in ["respond", RESPOND, "in:m", "model:m"] {
            assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
        }
    }
}
