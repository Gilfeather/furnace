//! The nervous system, end to end: a stream of events (text lines and image frames) goes
//! through a habituating receptor and a small Burn reflex classifier; urgent events take the
//! reflex arc straight to an effector, novel ones go to the brain (a mock LLM), which emits
//! commands that effectors carry out (a webhook, a local command, a journal file); repeated
//! events die of habituation at the receptor and never reach anything.
//!
//! ```bash
//! cargo run --release --features llm --example nerve
//! # Keep serving after the demo stream, to poke at it with curl:
//! cargo run --release --features llm --example nerve -- --serve
//! curl -X POST localhost:3000/events -d 'ALERT: smoke in rack 4'
//! curl -X POST localhost:3000/events -d 'door 7 opened'
//! curl localhost:3000/nerve | jq .
//! ```
//!
//! The example runs three things in one process:
//!
//! 1. A tiny webhook receiver on port 3001 (`POST /hook`) that prints every command that
//!    arrives, standing in for the service or device the nervous system commands.
//! 2. The furnace server on port 3000 with the fabric below.
//! 3. A demo stream: 200 text events of which half are repeats, posted to `/events`, and 200
//!    synthetic 28x28 frames fed through an in-process channel; then a summary from
//!    `/metrics`.
//!
//! The fabric:
//!
//! ```text
//! POST /events ──▶ events (habituation) ──▶ triage (reflex) ──┬─▶ alarm (webhook)     urgent
//!                                                             └─▶ brain (mock LLM) ──▶ webhook, command, log
//! frames (channel, habituation) ──▶ frame-classifier (Burn reflex) ──┬─▶ alarm (webhook)  bright frames
//!                                                                    └─▶ brain
//! ```
//!
//! Urgent signals carry a 2 ms deadline: the brain's pathway, measured in hundreds of
//! milliseconds (100 ms of mock thinking plus the queue), is not eligible for them, so the
//! reflex arc wins. Novel signals carry no
//! deadline, so the heavier (brain) pathway wins. No code decides "is this urgent": the
//! signal's timing does.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::routing::post;
use burn::tensor::{Tensor, TensorData};
use furnace::llm::{
    EngineError, Event, FinishReason, GenerateRequest, LlmEngine, Prompt, SamplingParams,
};
use furnace::nerve::brain::{Llm, ModelNeuron};
use furnace::nerve::effector::{Effector, FileAppend, Process, Respond, Webhook};
use furnace::nerve::receptor::{ChannelIn, Http, kinds};
use furnace::nerve::{
    Fabric, FabricBuilder, Feature, Firing, FnNeuron, Habituation, Mechanisms, Payload, Signal,
    Summation, Urgency,
};
use furnace::tensor::{HostTensor, Values};
use furnace::{BackendKind, Model, Server};

const PORT: u16 = 3000;
const HOOK_PORT: u16 = 3001;
/// An urgent signal wants its action within this: far below what the brain takes
const URGENT: Duration = Duration::from_millis(2);

// ---------------------------------------------------------------------------------------------
// The brain: a mock LLM that "thinks" for a while and answers with commands

/// Stands in for a real LLM (`furnace::llm::Scheduled` over your Burn model, or the candle
/// adapter's GGUF engine; both implement `LlmEngine`). It takes `delay` to answer, and
/// always decides to investigate through the webhook and to note it in the journal.
#[derive(Debug)]
struct MockBrain {
    delay: Duration,
}

impl LlmEngine for MockBrain {
    fn name(&self) -> &str {
        "mock-brain"
    }

    fn max_context(&self) -> usize {
        4096
    }

    fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Event>, EngineError> {
        let text = match &request.input {
            Prompt::Chat(messages) => messages.last().map(|m| m.content.clone()),
            Prompt::Text(text) => Some(text.clone()),
        }
        .unwrap_or_default();
        let answer = serde_json::json!([
            {"effector": "webhook", "args": {"action": "investigate", "about": text}},
            {"effector": "command", "args": {"args": [format!("brain: looked at '{text}'")]}},
            {"effector": "log", "args": {"line": format!("brain: {text}")}},
        ])
        .to_string();
        let delay = self.delay;
        let (events, receiver) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            let _ = events.send(Event::Started { prompt_tokens: 32 }).await;
            tokio::time::sleep(delay).await;
            let _ = events.send(Event::Text(answer)).await;
            let _ = events
                .send(Event::Finished {
                    reason: FinishReason::Stop,
                    completion_tokens: 48,
                })
                .await;
        });
        Ok(receiver)
    }
}

// ---------------------------------------------------------------------------------------------
// The reflex classifier: a small Burn model over 28x28 frames

/// `[batch, 784] -> [batch, 3]` logits for known / novel / urgent, from the frame's mean
/// brightness: dark frames are known, mid ones novel, bright ones urgent. A linear layer
/// with weights set by hand, so the demo is deterministic; your trained model goes here.
struct FrameNet {
    weight: Tensor<2>,
    bias: Tensor<2>,
}

impl FrameNet {
    fn new(device: &burn::tensor::Device) -> Self {
        let s = 20.0f32;
        let mut weight = Vec::with_capacity(784 * 3);
        for _ in 0..784 {
            weight.extend([-s / 784.0, 0.0, s / 784.0]);
        }
        Self {
            weight: Tensor::from_data(TensorData::new(weight, [784, 3]), device),
            bias: Tensor::from_data(
                TensorData::new(vec![0.5 * s, 0.2 * s, -0.5 * s], [1, 3]),
                device,
            ),
        }
    }

    fn forward(&self, frames: Tensor<2>) -> Tensor<2> {
        frames.matmul(self.weight.clone()) + self.bias.clone()
    }
}

/// What the classifier decided
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Known,
    Novel,
    Urgent,
}

fn verdict(logits: &[f32]) -> Verdict {
    let (best, _) =
        logits.iter().enumerate().fold(
            (0, f32::MIN),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        );
    match best {
        0 => Verdict::Known,
        1 => Verdict::Novel,
        _ => Verdict::Urgent,
    }
}

/// What a classifier fires for its verdict: nothing for known, the signal on (to the brain,
/// by weight) for novel, and an urgent command straight to the alarm
fn act_on(verdict: Verdict, signal: Signal, what: &str) -> Firing {
    match verdict {
        Verdict::Known => Firing::Quiet,
        Verdict::Novel => Firing::one(signal),
        Verdict::Urgent => Firing::one(
            signal
                .command("alarm", serde_json::json!({"urgent": what}))
                .with_urgency(Urgency::within(URGENT)),
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// The fabric

fn build_fabric(
    journal: &std::path::Path,
) -> (FabricBuilder, furnace::nerve::receptor::ChannelSender) {
    let mut b = Fabric::builder();
    let habituation = Mechanisms::NONE
        .threshold(0.6)
        .habituation(Habituation::new(Duration::from_secs(30)).gain(0.5));

    // Receptors
    let events = b.receptor(Http::new("events", "/events").text(), habituation);
    // Frames carry no deadline of their own: a novel frame may take the brain's time. A
    // sensor loop that must act before its next sample would add
    // `ChannelIn::with_urgency(.., UrgencyRule::before_next(1.0))`, and the brain's pathway
    // would then never be eligible for it.
    let (frames, frame_sender) = ChannelIn::new("frames", 1024);
    let frames = b.receptor(frames, habituation);

    // Reflexes
    let triage = b.neuron(
        FnNeuron::reflex("triage", |signal, _| {
            let text = signal.payload.as_text().unwrap_or_default();
            let verdict = if text.contains("ALERT") || text.contains("URGENT") {
                Verdict::Urgent
            } else {
                Verdict::Novel
            };
            act_on(verdict, signal, "text")
        }),
        Mechanisms::NONE,
    );
    let device = BackendKind::Cpu.device();
    let net = FrameNet::new(&device);
    let model = Model::from_fn(
        "frame-classifier",
        BackendKind::Cpu,
        &[784],
        &[3],
        move |x: Tensor<2>| net.forward(x),
    )
    .expect("a valid model");
    let frame_classifier = b.neuron(
        ModelNeuron::reflex(
            "frame-classifier",
            Arc::new(model),
            |signal| {
                let frame = signal.payload.as_tensors()?.first()?.clone();
                Some(vec![frame])
            },
            |signal, outputs| {
                let Values::F32(logits) = &outputs[0].values else {
                    return Vec::new();
                };
                match act_on(verdict(logits), signal.clone(), "frame") {
                    Firing::Fire(signals) => signals,
                    _ => Vec::new(),
                }
            },
        ),
        Mechanisms::NONE,
    );

    // The brain
    let brain = b.neuron(
        Llm::new(
            "brain",
            Arc::new(MockBrain {
                delay: Duration::from_millis(100),
            }),
        )
        .params(SamplingParams::greedy().with_max_tokens(128)),
        Mechanisms::NONE,
    );

    // Effectors
    let hook = format!("http://127.0.0.1:{HOOK_PORT}/hook");
    let alarm = b.neuron(
        Effector::new(Webhook::new("alarm", &hook).expect("an http:// URL"))
            .refractory(Duration::from_millis(50)),
        Mechanisms::NONE,
    );
    b.neuron(
        Effector::new(Webhook::new("webhook", &hook).expect("an http:// URL")),
        Mechanisms::NONE,
    );
    let (program, args): (&str, &[&str]) = if cfg!(windows) {
        ("cmd", &["/C", "echo"])
    } else {
        ("echo", &[])
    };
    b.neuron(
        Effector::new(Process::new("command", program).args(args)).no_retry(),
        Mechanisms::NONE,
    );
    b.neuron(
        Effector::new(FileAppend::new("log", journal)),
        Mechanisms::NONE,
    );
    b.neuron(Effector::new(Respond::new("respond")), Mechanisms::NONE);

    // Pathways: the brain's weighs more, so it wins whenever the deadline allows it
    b.reflex(events, triage, 1.0);
    b.reflex(frames, frame_classifier, 1.0);
    for classifier in [triage, frame_classifier] {
        b.reflex(classifier, alarm, 1.0);
        b.deliberative(classifier, brain, 2.0, Summation::default().max_batch(16));
    }
    (b, frame_sender)
}

// ---------------------------------------------------------------------------------------------
// The tiny webhook receiver

async fn hook_receiver(received: Arc<AtomicUsize>) {
    let app = Router::new().route(
        "/hook",
        post(move |body: String| {
            let received = received.clone();
            async move {
                let n = received.fetch_add(1, Ordering::Relaxed) + 1;
                let shown = if body.len() > 100 {
                    format!("{}…", &body[..100])
                } else {
                    body
                };
                println!("  [webhook #{n}] {shown}");
                axum::http::StatusCode::NO_CONTENT
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", HOOK_PORT))
        .await
        .expect("bind the webhook receiver");
    axum::serve(listener, app)
        .await
        .expect("serve the webhook receiver");
}

// ---------------------------------------------------------------------------------------------
// The demo stream

/// A synthetic 28x28 frame of the given mean brightness, with a little texture
fn frame(mean: f32, seed: u32) -> HostTensor {
    let values: Vec<f32> = (0..784u32)
        .map(|i| {
            let noise = ((i.wrapping_mul(2654435761) ^ seed) % 97) as f32 / 97.0 - 0.5;
            (mean + 0.05 * noise).clamp(0.0, 1.0)
        })
        .collect();
    HostTensor {
        shape: vec![1, 784],
        values: Values::F32(values),
    }
}

async fn demo_stream(frames: furnace::nerve::receptor::ChannelSender) {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{PORT}/events");
    println!("\nText events: 200, half of them repeats of 10 lines, 5% urgent");
    let mut statuses = [0usize; 3];
    for i in 0..200 {
        let text = if i % 2 == 0 {
            format!("sensor {} reports nominal", i % 10)
        } else if i % 20 == 1 {
            format!("ALERT: pressure spike at valve {i}")
        } else {
            format!("event {i}: door opened")
        };
        let response = client
            .post(&url)
            .body(text)
            .send()
            .await
            .expect("the server answers");
        statuses[match response.status().as_u16() {
            204 => 0,
            202 => 1,
            _ => 2,
        }] += 1;
    }
    println!(
        "  204 (quiet or reflex): {}, 202 (to the brain): {}, other: {}",
        statuses[0], statuses[1], statuses[2]
    );

    println!("\nFrames: 200 through the in-process channel, 50% repeats of 5 known frames");
    let start = Instant::now();
    let mut outcomes = [0usize; 3];
    for i in 0..200u32 {
        let (mean, seed) = if i % 2 == 0 {
            (0.1, i % 5) // a known dark frame, one of five
        } else if i % 40 == 1 {
            (0.9, i) // a bright, urgent frame
        } else {
            (0.5, i) // a novel frame
        };
        let tensor = frame(mean, seed);
        // The feature: a sketch of the quantized frame, so equal frames habituate
        let bytes: Vec<u8> = match &tensor.values {
            Values::F32(v) => v.iter().map(|x| (x * 255.0) as u8).collect(),
            Values::I64(_) => Vec::new(),
        };
        let signal = frames.signal(
            Feature::sketch_as(kinds::BYTES + 10, &bytes),
            Payload::tensors(vec![tensor]),
        );
        if let Some(outcome) = frames.emit(signal) {
            outcomes[if outcome.is_quiet() {
                0
            } else if outcome.queued > 0 {
                1
            } else {
                2
            }] += 1;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    println!(
        "  quiet at the receptor: {}, to the brain: {}, reflex: {} ({:.1} ms for all)",
        outcomes[0],
        outcomes[1],
        outcomes[2],
        start.elapsed().as_secs_f64() * 1e3
    );
}

async fn summary() {
    // Let the brain finish what is queued
    let deadline = Instant::now() + Duration::from_secs(30);
    let metrics: serde_json::Value = loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let metrics: serde_json::Value = reqwest::get(format!("http://127.0.0.1:{PORT}/metrics"))
            .await
            .expect("metrics")
            .json()
            .await
            .expect("json");
        if metrics["nerve"]["waiting"] == 0 || Instant::now() > deadline {
            break metrics;
        }
    };
    let nerve = &metrics["nerve"];
    println!("\nFrom /metrics:");
    println!(
        "  signals received {}, reached the brain {}: brain fraction {:.2}",
        nerve["received"], nerve["reached_brain"], nerve["brain_fraction"]
    );
    for neuron in nerve["neurons"].as_array().into_iter().flatten() {
        println!(
            "  {:<17} {:<9} fired {:>4}  quiet: habituation {:>3}, threshold {:>3}, refractory {:>3}",
            neuron["name"].as_str().unwrap_or(""),
            neuron["kind"].as_str().unwrap_or(""),
            neuron["fired"],
            neuron["quiet_by_habituation"],
            neuron["quiet_by_threshold"],
            neuron["quiet_by_refractory"]
        );
    }
    for pathway in nerve["pathways"].as_array().into_iter().flatten() {
        println!(
            "  pathway {} -> {} ({}): {} signals, p50 {:.1} µs, p99 {:.1} µs",
            pathway["from"].as_str().unwrap_or(""),
            pathway["to"].as_str().unwrap_or(""),
            pathway["kind"].as_str().unwrap_or(""),
            pathway["signals"],
            pathway["latency_ms"]["p50"].as_f64().unwrap_or(0.0) * 1e3,
            pathway["latency_ms"]["p99"].as_f64().unwrap_or(0.0) * 1e3
        );
    }
    for arc in nerve["arcs"].as_array().into_iter().flatten() {
        println!(
            "  reflex arc {} -> {}: stimulus to effector call p50 {:.1} µs, p99 {:.1} µs ({} times)",
            arc["from"].as_str().unwrap_or(""),
            arc["to"].as_str().unwrap_or(""),
            arc["latency_ms"]["p50"].as_f64().unwrap_or(0.0) * 1e3,
            arc["latency_ms"]["p99"].as_f64().unwrap_or(0.0) * 1e3,
            arc["latency_ms"]["count"]
        );
    }
}

#[tokio::main]
async fn main() {
    let serve_only = std::env::args().any(|a| a == "--serve");
    let journal = std::env::temp_dir().join("furnace-nerve-journal.log");
    let _ = std::fs::remove_file(&journal);
    let received = Arc::new(AtomicUsize::new(0));
    tokio::spawn(hook_receiver(received.clone()));

    let (fabric, frames) = build_fabric(&journal);
    let server = Server::new().host("127.0.0.1").port(PORT).nerve(fabric);
    tokio::spawn(async move {
        if let Err(e) = server.serve().await {
            eprintln!("server: {e}");
            std::process::exit(1);
        }
    });
    // Wait for the server
    let client = reqwest::Client::new();
    while client
        .get(format!("http://127.0.0.1:{PORT}/healthz"))
        .send()
        .await
        .is_err()
    {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    println!("furnace nerve demo: server on :{PORT}, webhook receiver on :{HOOK_PORT}");
    println!("journal: {}", journal.display());

    if serve_only {
        println!("Serving; POST text to /events, GET /nerve. Ctrl-C to stop.");
        tokio::signal::ctrl_c().await.ok();
        return;
    }
    demo_stream(frames).await;
    summary().await;
    println!(
        "\nThe webhook receiver got {} commands; the journal has {} lines.",
        received.load(Ordering::Relaxed),
        std::fs::read_to_string(&journal)
            .map(|s| s.lines().count())
            .unwrap_or(0)
    );
}
