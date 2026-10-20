//! Brains and model neurons: the slow, large model whose output is commands, and any
//! [`Model`] as a reflex or a brain.
//!
//! - `Llm` (feature `llm`) wraps an `LlmEngine`: the mock engines of
//!   `furnace::llm::mock`, a `Scheduled` step model, the candle adapter. The signal's payload
//!   becomes the user message, the effectors' names go into the system prompt, and the
//!   model's answer is parsed as commands: a JSON command object
//!   `{"effector": "...", "args": {...}}`, an array of them, or OpenAI-style tool calls.
//! - [`ModelNeuron`] wraps a [`Model`]: a user function makes the input tensors from the
//!   signal and another turns the outputs into commands (or any signals). As a reflex it runs
//!   inline on the arc; as a brain, on a deliberative pathway, it batches the signals of a
//!   window into one forward pass.
//!
//! A brain's work is blocking and runs where it is stimulated: on the summation worker of a
//! deliberative pathway, which is where brains belong. The signals of one summation window
//! are generated at once (an engine with a scheduler runs them as one batch) or run as one
//! forward pass. A brain on a reflex pathway would block the firing thread for the whole
//! generation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::debug;

use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Neuron, NeuronKind};
use super::signal::{Command, Payload, Signal};
use crate::model::Model;
use crate::tensor::HostTensor;

/// Commands in a model's output text: the whole text as JSON, or every JSON object or array
/// found in it. Accepted shapes, in any of those:
///
/// - `{"effector": "name", "args": ...}`
/// - `{"name": "name", "arguments": ...}` and `{"function": {"name": ..., "arguments": ...}}`
///   (OpenAI tool calls; string `arguments` are parsed as JSON)
/// - `{"tool_calls": [...]}` and `{"commands": [...]}`
/// - an array of any of the above
pub fn parse_commands(text: &str) -> Vec<Command> {
    let mut commands = Vec::new();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim()) {
        collect(&value, &mut commands);
        return commands;
    }
    for candidate in json_candidates(text) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(candidate) {
            collect(&value, &mut commands);
        }
    }
    commands
}

/// Balanced `{...}` and `[...]` spans of `text`, outermost only, strings respected
fn json_candidates(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            match b {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => in_string = false,
                _ => escaped = false,
            }
            continue;
        }
        match b {
            b'"' if depth > 0 => in_string = true,
            b'{' | b'[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' | b']' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    spans.push(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    spans
}

fn collect(value: &serde_json::Value, out: &mut Vec<Command>) {
    match value {
        serde_json::Value::Array(items) => items.iter().for_each(|item| collect(item, out)),
        serde_json::Value::Object(object) => {
            if let Some(serde_json::Value::String(effector)) = object.get("effector") {
                out.push(Command::new(
                    effector,
                    object
                        .get("args")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                ));
                return;
            }
            if let Some(function) = object.get("function").and_then(|f| f.as_object())
                && let Some(serde_json::Value::String(name)) = function.get("name")
            {
                out.push(Command::new(name, arguments(function)));
                return;
            }
            if let Some(serde_json::Value::String(name)) = object.get("name")
                && (object.contains_key("arguments") || object.contains_key("args"))
            {
                out.push(Command::new(name, arguments(object)));
                return;
            }
            for key in ["tool_calls", "commands"] {
                if let Some(items) = object.get(key) {
                    collect(items, out);
                }
            }
        }
        _ => {}
    }
}

/// `arguments` or `args`, a JSON string parsed if it is one
fn arguments(object: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    match object.get("arguments").or_else(|| object.get("args")) {
        Some(serde_json::Value::String(text)) => {
            serde_json::from_str(text).unwrap_or(serde_json::Value::String(text.clone()))
        }
        Some(value) => value.clone(),
        None => serde_json::Value::Null,
    }
}

/// The text a brain shows its model for a signal
pub fn signal_text(signal: &Signal) -> String {
    match &signal.payload {
        Payload::Text(text) => text.to_string(),
        Payload::Json(value) => value.to_string(),
        Payload::Command(command) => serde_json::to_string(&**command).unwrap_or_default(),
        Payload::Bytes(bytes) => match std::str::from_utf8(bytes) {
            Ok(text) => text.to_string(),
            Err(_) => format!("[binary payload of {} bytes]", bytes.len()),
        },
        Payload::Tensors(tensors) => format!("[{} tensor(s)]", tensors.len()),
        Payload::None => String::new(),
        #[cfg(feature = "llm")]
        Payload::Prompt(_) => "[a generation request]".to_string(),
    }
}

#[derive(Default)]
struct BrainCounters {
    generations: AtomicU64,
    failures: AtomicU64,
    without_commands: AtomicU64,
    commands: AtomicU64,
    completion_tokens: AtomicU64,
}

impl BrainCounters {
    fn json(&self, latency: &LatencyEstimate) -> serde_json::Value {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "generations": load(&self.generations),
            "failures": load(&self.failures),
            "without_commands": load(&self.without_commands),
            "commands": load(&self.commands),
            "completion_tokens": load(&self.completion_tokens),
            "generation": latency.snapshot(),
        })
    }
}

#[cfg(feature = "llm")]
pub use llm::Llm;

#[cfg(feature = "llm")]
mod llm {
    use super::*;
    use crate::llm::{Event, GenerateRequest, Input, LlmEngine, Message, SamplingParams};

    /// The default system prompt; `{effectors}` is replaced by the effectors' names
    pub const SYSTEM_PROMPT: &str = "You are the brain of a nervous system. You see one signal \
        at a time and may act on the world only through these effectors: {effectors}. Answer \
        with JSON only: one command object {\"effector\": \"<name>\", \"args\": {...}}, or a \
        JSON array of command objects, or [] when no action is needed.";

    type UserFn = dyn Fn(&Signal) -> String + Send + Sync;

    /// A brain over an [`LlmEngine`]: prompt, generate, parse commands
    pub struct Llm {
        name: String,
        engine: Arc<dyn LlmEngine>,
        system: String,
        user: Arc<UserFn>,
        params: SamplingParams,
        /// Use `/v1/completions`-style raw text (system prompt, a blank line, the signal)
        /// instead of a chat; for engines without a chat template
        completion: bool,
        shared: Arc<Shared>,
    }

    #[derive(Default)]
    struct Shared {
        counters: BrainCounters,
        latency: LatencyEstimate,
    }

    impl Llm {
        /// A brain named `name` over `engine`, greedy, at most 256 tokens per answer
        pub fn new(name: &str, engine: Arc<dyn LlmEngine>) -> Self {
            Self {
                name: name.to_string(),
                engine,
                system: SYSTEM_PROMPT.to_string(),
                user: Arc::new(signal_text),
                params: SamplingParams::greedy().with_max_tokens(256),
                completion: false,
                shared: Arc::default(),
            }
        }

        /// Your own system prompt; `{effectors}` is replaced by the effectors' names
        pub fn system(mut self, prompt: &str) -> Self {
            self.system = prompt.to_string();
            self
        }

        /// How a signal becomes the user message (by default its payload as text)
        pub fn user(mut self, f: impl Fn(&Signal) -> String + Send + Sync + 'static) -> Self {
            self.user = Arc::new(f);
            self
        }

        /// How to sample the reply (by default greedy, at most 256 tokens)
        pub fn params(mut self, params: SamplingParams) -> Self {
            self.params = params;
            self
        }

        /// Raw-text prompts instead of a chat
        pub fn completion(mut self) -> Self {
            self.completion = true;
            self
        }

        fn request(&self, signal: &Signal, effectors: &[String]) -> GenerateRequest {
            let system = self.system.replace("{effectors}", &effectors.join(", "));
            let user = (self.user)(signal);
            let input = if self.completion {
                Input::Text(format!("{system}\n\n{user}\n"))
            } else {
                Input::Chat(vec![
                    Message::new("system", &system),
                    Message::new("user", &user),
                ])
            };
            GenerateRequest {
                input,
                params: self.params.clone(),
                stop: Vec::new(),
            }
        }
    }

    /// Collect one generation's events
    async fn collect(
        mut events: tokio::sync::mpsc::Receiver<Event>,
    ) -> Result<(String, usize), String> {
        let mut text = String::new();
        let mut tokens = 0;
        while let Some(event) = events.recv().await {
            match event {
                Event::Started { .. } => {}
                Event::Text(piece) => text.push_str(&piece),
                Event::Finished {
                    completion_tokens, ..
                } => tokens = completion_tokens,
                Event::Error(e) => return Err(e.to_string()),
            }
        }
        Ok((text, tokens))
    }

    /// Run every request's generation at once on `runtime` (an engine with a scheduler
    /// batches them) and wait for them all, from any thread
    fn generate_all(
        engine: &dyn LlmEngine,
        runtime: &tokio::runtime::Handle,
        requests: Vec<GenerateRequest>,
    ) -> Vec<Result<(String, usize), String>> {
        let run = || {
            let _guard = runtime.enter();
            let started: Vec<Result<tokio::sync::mpsc::Receiver<Event>, String>> = requests
                .into_iter()
                .map(|request| engine.generate(request).map_err(|e| e.to_string()))
                .collect();
            runtime.block_on(async move {
                let mut results = Vec::with_capacity(started.len());
                let mut tasks = Vec::new();
                for (i, events) in started.into_iter().enumerate() {
                    match events {
                        Ok(events) => tasks.push((i, tokio::spawn(collect(events)))),
                        Err(e) => results.push((i, Err(e))),
                    }
                }
                for (i, task) in tasks {
                    results.push((
                        i,
                        task.await
                            .unwrap_or_else(|_| Err("generation panicked".to_string())),
                    ));
                }
                results.sort_by_key(|(i, _)| *i);
                results.into_iter().map(|(_, r)| r).collect()
            })
        };
        // Blocking on a runtime's own worker thread is not allowed (a brain on a reflex
        // pathway stimulated from a request handler): do it on a thread of our own
        if tokio::runtime::Handle::try_current().is_ok() {
            std::thread::scope(|scope| scope.spawn(run).join().unwrap_or_else(|_| Vec::new()))
        } else {
            run()
        }
    }

    impl Llm {
        /// Turn one generation's result into the signals to fire
        fn fired(
            &self,
            signal: &Signal,
            result: Result<(String, usize), String>,
            started: std::time::Instant,
        ) -> Vec<Signal> {
            let counters = &self.shared.counters;
            match result {
                Ok((text, tokens)) => {
                    counters.generations.fetch_add(1, Ordering::Relaxed);
                    counters
                        .completion_tokens
                        .fetch_add(tokens as u64, Ordering::Relaxed);
                    self.shared.latency.observe(started.elapsed());
                    let commands = parse_commands(&text);
                    if commands.is_empty() {
                        counters.without_commands.fetch_add(1, Ordering::Relaxed);
                        debug!(
                            "nerve: brain '{}' answered without a command: {text:?}",
                            self.name
                        );
                    }
                    counters
                        .commands
                        .fetch_add(commands.len() as u64, Ordering::Relaxed);
                    commands
                        .into_iter()
                        .map(|c| signal.derive(Payload::command(c)))
                        .collect()
                }
                Err(e) => {
                    counters.failures.fetch_add(1, Ordering::Relaxed);
                    debug!("nerve: brain '{}' failed: {e}", self.name);
                    Vec::new()
                }
            }
        }
    }

    impl Neuron for Llm {
        fn name(&self) -> &str {
            &self.name
        }

        fn kind(&self) -> NeuronKind {
            NeuronKind::Brain
        }

        /// One signal: the generation runs here, on the pathway's worker (or inline on a
        /// reflex pathway)
        fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
            let started = std::time::Instant::now();
            let request = self.request(&signal, &cx.effectors());
            let result = generate_all(&*self.engine, cx.efferent(), vec![request])
                .pop()
                .unwrap_or_else(|| Err("no result".to_string()));
            Firing::Fire(self.fired(&signal, result, started))
        }

        /// The signals of a summation window: all their generations at once. An engine with
        /// a scheduler runs them as one batch; the window's signals wait for the slowest.
        fn stimulate_many(&self, signals: Vec<Signal>, cx: &Cx<'_>) -> Vec<Firing> {
            let started = std::time::Instant::now();
            let effectors = cx.effectors();
            let requests = signals
                .iter()
                .map(|s| self.request(s, &effectors))
                .collect();
            let results = generate_all(&*self.engine, cx.efferent(), requests);
            signals
                .iter()
                .zip(results)
                .map(|(signal, result)| Firing::Fire(self.fired(signal, result, started)))
                .collect()
        }

        fn describe(&self) -> serde_json::Value {
            serde_json::json!({
                "engine": self.engine.name(),
                "max_context": self.engine.max_context(),
                "max_tokens": self.params.max_tokens,
                "completion": self.completion,
            })
        }

        fn metrics(&self) -> serde_json::Value {
            self.shared.counters.json(&self.shared.latency)
        }
    }
}

type InputFn = dyn Fn(&Signal) -> Option<Vec<HostTensor>> + Send + Sync;
type OutputFn = dyn Fn(&Signal, Vec<HostTensor>) -> Vec<Signal> + Send + Sync;

/// A [`Model`] as a neuron: `input` makes the model's input tensors from a signal (`None`
/// to stay quiet), `output` turns the outputs into the signals to fire, usually commands
/// made with [`Signal::command`]. A reflex runs inline; a brain batches on a deliberative
/// pathway.
pub struct ModelNeuron {
    name: String,
    kind: NeuronKind,
    model: Arc<Model>,
    input: Arc<InputFn>,
    output: Arc<OutputFn>,
    counters: BrainCounters,
    latency: LatencyEstimate,
}

impl ModelNeuron {
    /// A reflex: small and fast, run inline on the arc
    pub fn reflex(
        name: &str,
        model: Arc<Model>,
        input: impl Fn(&Signal) -> Option<Vec<HostTensor>> + Send + Sync + 'static,
        output: impl Fn(&Signal, Vec<HostTensor>) -> Vec<Signal> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.to_string(),
            kind: NeuronKind::Reflex,
            model,
            input: Arc::new(input),
            output: Arc::new(output),
            counters: BrainCounters::default(),
            latency: LatencyEstimate::new(),
        }
    }

    /// A brain: large and slow, put behind a deliberative pathway, where the signals of a
    /// window share one forward pass
    pub fn brain(
        name: &str,
        model: Arc<Model>,
        input: impl Fn(&Signal) -> Option<Vec<HostTensor>> + Send + Sync + 'static,
        output: impl Fn(&Signal, Vec<HostTensor>) -> Vec<Signal> + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind: NeuronKind::Brain,
            ..Self::reflex(name, model, input, output)
        }
    }

    fn fired(&self, signal: &Signal, result: crate::error::Result<Vec<HostTensor>>) -> Vec<Signal> {
        match result {
            Ok(outputs) => {
                self.counters.generations.fetch_add(1, Ordering::Relaxed);
                let signals = (self.output)(signal, outputs);
                let commands = signals
                    .iter()
                    .filter(|s| s.payload.as_command().is_some())
                    .count();
                if signals.is_empty() {
                    self.counters
                        .without_commands
                        .fetch_add(1, Ordering::Relaxed);
                }
                self.counters
                    .commands
                    .fetch_add(commands as u64, Ordering::Relaxed);
                signals
            }
            Err(e) => {
                self.counters.failures.fetch_add(1, Ordering::Relaxed);
                debug!("nerve: model neuron '{}' failed: {e}", self.name);
                Vec::new()
            }
        }
    }
}

impl Neuron for ModelNeuron {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NeuronKind {
        self.kind
    }

    fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
        let Some(inputs) = (self.input)(&signal) else {
            return Firing::Quiet;
        };
        // On a reflex pathway the forward pass runs inline, as the arc's work; on a
        // deliberative one it runs on the pathway's worker thread
        let started = std::time::Instant::now();
        let result = self.model.infer(inputs);
        self.latency.observe(started.elapsed());
        Firing::Fire(self.fired(&signal, result))
    }

    fn stimulate_many(&self, signals: Vec<Signal>, _cx: &Cx<'_>) -> Vec<Firing> {
        let mut firings: Vec<Option<Firing>> = (0..signals.len()).map(|_| None).collect();
        let mut requests = Vec::new();
        let mut indexes = Vec::new();
        for (i, signal) in signals.iter().enumerate() {
            match (self.input)(signal) {
                Some(inputs) => {
                    requests.push(inputs);
                    indexes.push(i);
                }
                None => firings[i] = Some(Firing::Quiet),
            }
        }
        if !requests.is_empty() {
            let started = std::time::Instant::now();
            let results = self.model.infer_many(requests);
            self.latency.observe(started.elapsed());
            for (i, result) in indexes.into_iter().zip(results) {
                firings[i] = Some(Firing::Fire(self.fired(&signals[i], result)));
            }
        }
        firings
            .into_iter()
            .map(|f| f.unwrap_or(Firing::Quiet))
            .collect()
    }

    fn describe(&self) -> serde_json::Value {
        let info = self.model.get_info();
        serde_json::json!({
            "model": info.name,
            "backend": info.backend,
            "inputs": self.model.inputs().iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            "outputs": self.model.outputs().iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
        })
    }

    fn metrics(&self) -> serde_json::Value {
        self.counters.json(&self.latency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_command_objects_arrays_and_tool_calls() {
        let one = parse_commands(r#"{"effector": "alarm", "args": {"level": 2}}"#);
        assert_eq!(
            one,
            vec![Command::new("alarm", serde_json::json!({"level": 2}))]
        );
        let many = parse_commands(
            r#"Sure. [{"effector": "log", "args": "a"}, {"name": "webhook", "arguments": "{\"x\": 1}"}] done"#,
        );
        assert_eq!(many.len(), 2);
        assert_eq!(many[1].effector, "webhook");
        assert_eq!(many[1].args, serde_json::json!({"x": 1}));
        let tool_calls = parse_commands(
            r#"{"tool_calls": [{"id": "1", "type": "function", "function": {"name": "cmd", "arguments": "{\"args\": [\"on\"]}"}}]}"#,
        );
        assert_eq!(
            tool_calls,
            vec![Command::new("cmd", serde_json::json!({"args": ["on"]}))]
        );
        assert!(parse_commands("nothing to do").is_empty());
        assert!(parse_commands("[]").is_empty());
        // Braces inside strings do not confuse the scan
        let tricky = parse_commands(r#"text {"effector": "log", "args": "a } b"} tail"#);
        assert_eq!(tricky[0].args, "a } b");
    }

    use crate::nerve::effector::{Channel, Effector};
    use crate::nerve::fabric::{Fabric, Summation};
    use crate::nerve::neuron::Mechanisms;
    use crate::nerve::signal::{Feature, NeuronId};
    use crate::nerve::testing::pass;
    use std::time::{Duration, Instant};

    /// A receptor, a deliberative pathway to `brain`, and a channel effector named `out`
    fn fabric_with_brain(
        brain: impl Neuron,
    ) -> (
        Fabric,
        NeuronId,
        std::sync::mpsc::Receiver<crate::nerve::effector::Delivery>,
    ) {
        let mut b = Fabric::builder();
        let r = b.neuron(pass("in"), Mechanisms::NONE);
        let brain = b.neuron(brain, Mechanisms::NONE);
        let (channel, receiver) = Channel::new("out", 64);
        b.neuron(Effector::new(channel), Mechanisms::NONE);
        b.deliberative(
            r,
            brain,
            1.0,
            Summation::default().window(Duration::from_millis(5)),
        );
        (b.build(), r, receiver)
    }

    fn signal(text: &str) -> Signal {
        Signal::new(Feature::sketch(text.as_bytes()), Payload::text(text))
    }

    #[cfg(feature = "llm")]
    #[test]
    fn llm_brain_turns_the_answer_into_commands() {
        use crate::llm::mock::MockEngine;
        let engine = MockEngine::new(
            "mock",
            &[
                "I will act. ",
                r#"{"effector": "out", "#,
                r#""args": {"level": 2}}"#,
            ],
        );
        let brain = Llm::new("brain", Arc::new(engine)).system("Effectors: {effectors}.");
        let (fabric, r, receiver) = fabric_with_brain(brain);
        let outcome = fabric.stimulate(r, signal("something new"));
        assert_eq!(outcome.queued, 1);
        let delivery = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(
            delivery.command,
            Command::new("out", serde_json::json!({"level": 2}))
        );
        assert!(delivery.signal.trace.reached_brain());
        assert!(delivery.signal.trace.was_queued());
        assert_eq!(delivery.signal.payload.as_text(), None);
        let metrics = fabric.metrics();
        let brain = metrics.neurons.iter().find(|n| n.name == "brain").unwrap();
        assert_eq!(brain.extra["generations"], 1);
        assert_eq!(brain.extra["commands"], 1);
        assert_eq!(metrics.brain_fraction, 1.0);
    }

    #[test]
    fn model_brain_batches_a_window_into_one_forward_pass() {
        let model = Arc::new(Model::new(
            Box::new(crate::model::DummyModel::new(
                "dummy".into(),
                vec![4],
                vec![1],
            )),
            "test",
            0,
        ));
        let brain = ModelNeuron::brain(
            "brain",
            model.clone(),
            |signal| {
                let n = signal.payload.as_text()?.len() as f32;
                Some(vec![HostTensor {
                    shape: vec![1, 4],
                    values: crate::tensor::Values::F32(vec![n; 4]),
                }])
            },
            |signal, outputs| {
                let crate::tensor::Values::F32(values) = &outputs[0].values else {
                    return Vec::new();
                };
                vec![signal.command("out", serde_json::json!({"score": values[0]}))]
            },
        );
        let (fabric, r, receiver) = fabric_with_brain(brain);
        for text in ["a", "bb", "ccc"] {
            fabric.stimulate(r, signal(text));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut delivered = Vec::new();
        while delivered.len() < 3 {
            assert!(Instant::now() < deadline);
            if let Ok(d) = receiver.recv_timeout(Duration::from_millis(50)) {
                delivered.push(d);
            }
        }
        assert!(delivered.iter().all(|d| d.command.args["score"] == 0.5));
        // One forward pass for the three (the dummy's batch statistics)
        let stats = model.get_stats();
        assert_eq!(stats.batch_count, 1);
        assert_eq!(stats.batched_items, 3);
    }
}
