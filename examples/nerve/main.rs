//! The nervous system, end to end: text events and image frames go through habituating
//! receptors and two reflexes (a closure and a small Burn classifier); urgent ones act at once
//! through a webhook, novel ones go to the brain (a mock LLM), whose commands effectors carry
//! out (a webhook, a local command, a journal file). Repeats die of habituation at the
//! receptor and never reach anything.
//!
//! ```bash
//! cargo run --release --features llm --example nerve
//! # Keep serving after the demo stream, to poke at it with curl:
//! cargo run --release --features llm --example nerve -- --serve
//! curl -X POST localhost:3000/events -d 'ALERT: smoke in rack 4'
//! curl localhost:3000/nerve | jq .
//! ```
//!
//! ```text
//! POST /events ─▶ events (habituation) ─▶ triage (reflex) ───────────┬─▶ alarm (webhook)  urgent
//! frames (channel, habituation) ─▶ frame-classifier (Burn reflex) ──┴─▶ brain (mock LLM) ─▶ webhook, command, log
//! ```
//!
//! Urgent signals are commands to `alarm` with a 2 ms deadline; novel ones carry none, so the
//! heavier (brain) pathway takes them. A webhook receiver on port 3001 prints what arrives.

use std::time::{Duration, Instant};

use axum::routing::post;
use burn::tensor::{Tensor, TensorData};
use furnace::llm::mock::MockEngine;
use furnace::nerve::prelude::*;
use furnace::nerve::receptor::{ChannelSender, kinds};
use furnace::nerve::{FabricBuilder, Feature};
use furnace::tensor::{HostTensor, Values};
use furnace::{BackendKind, Model, Server};
use serde_json::json;

const PORT: u16 = 3000;
const HOOK_PORT: u16 = 3001;

/// What a classifier fires for its class: nothing for known (0), the signal on to the brain
/// for novel (1), a command to the alarm with a deadline the brain cannot meet for urgent (2)
fn act_on(class: usize, signal: Signal, what: &str) -> Option<Signal> {
    match class {
        0 => None,
        1 => Some(signal),
        _ => Some(
            signal
                .command("alarm", json!({"urgent": what}))
                .with_urgency(Urgency::within(Duration::from_millis(2))),
        ),
    }
}

/// `[batch, 784] -> [batch, 3]` logits for known / novel / urgent from a frame's mean
/// brightness: a linear layer with weights set by hand, so the demo is deterministic. Your
/// trained model goes here.
fn frame_classifier() -> furnace::Result<Model> {
    let device = BackendKind::Cpu.device();
    let s = 20.0f32;
    let weight: Vec<f32> = (0..784)
        .flat_map(|_| [-s / 784.0, 0.0, s / 784.0])
        .collect();
    let weight = Tensor::<2>::from_data(TensorData::new(weight, [784, 3]), &device);
    let bias = Tensor::<2>::from_data(
        TensorData::new(vec![0.5 * s, 0.2 * s, -0.5 * s], [1, 3]),
        &device,
    );
    Model::from_fn(
        "frame-classifier",
        BackendKind::Cpu,
        &[784],
        &[3],
        move |x: Tensor<2>| x.matmul(weight.clone()) + bias.clone(),
    )
}

fn build_fabric(journal: &std::path::Path) -> furnace::Result<(FabricBuilder, ChannelSender)> {
    let mut b = Fabric::builder();
    let habituation = Mechanisms::NONE
        .threshold(0.6)
        .habituation(Habituation::new(Duration::from_secs(30)).gain(0.5));
    let events = b.receptor(Http::new("events", "/events").text());
    let (frames, frame_sender) = ChannelIn::new("frames", 1024);
    let frames = b.receptor(frames);
    b.gate(events, habituation).gate(frames, habituation);

    let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
        let urgent = signal
            .payload
            .as_text()
            .is_some_and(|t| t.contains("ALERT") || t.contains("URGENT"));
        act_on(if urgent { 2 } else { 1 }, signal, "text")
    }));
    let classifier = b.neuron(ModelNeuron::reflex(
        "frame-classifier",
        frame_classifier()?,
        |signal| Some(vec![signal.payload.as_tensors()?.first()?.clone()]),
        |signal, outputs| {
            let Values::F32(logits) = &outputs[0].values else {
                return Vec::new();
            };
            let class = (0..3)
                .max_by(|&i, &j| logits[i].total_cmp(&logits[j]))
                .unwrap_or(0);
            act_on(class, signal.clone(), "frame").into_iter().collect()
        },
    ));
    // Stands in for a real LLM (`furnace::llm::Scheduled` over your Burn model, or the candle
    // adapter): 100 ms of thinking, then the same three commands
    let answer = json!([
        {"effector": "webhook", "args": {"action": "investigate"}},
        {"effector": "command", "args": {"args": ["brain: looked"]}},
        {"effector": "log", "args": {"line": "brain: looked"}},
    ])
    .to_string();
    let engine = MockEngine::new("mock-brain", &[&answer]).first_delay(Duration::from_millis(100));
    let brain = b.neuron(Llm::new("brain", engine));

    let hook = format!("http://127.0.0.1:{HOOK_PORT}/hook");
    let alarm = b.effector(
        Effector::new(Webhook::new("alarm", &hook)?).refractory(Duration::from_millis(50)),
    );
    b.effector(Webhook::new("webhook", &hook)?);
    let echo = if cfg!(windows) {
        Process::new("command", "cmd").args(&["/C", "echo"])
    } else {
        Process::new("command", "echo")
    };
    b.effector(Effector::new(echo).no_retry());
    b.effector(FileAppend::new("log", journal));
    b.effector(Respond::new("respond"));

    b.reflex(events, triage);
    b.reflex(frames, classifier);
    for from in [triage, classifier] {
        b.reflex(from, alarm);
        b.deliberative(from, brain)
            .summation(Summation::default().max_batch(16));
    }
    Ok((b, frame_sender))
}

/// A synthetic 28x28 frame of the given mean brightness, with a little texture
fn frame(mean: f32, seed: u32) -> HostTensor {
    let values = (0..784u32)
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

/// 200 text events (half repeats of 10 lines, 5% urgent) over HTTP, then 200 frames (half
/// repeats of 5 known dark frames, a few bright urgent ones) through the channel
async fn demo_stream(frames: ChannelSender) {
    let client = reqwest::Client::new();
    let mut statuses = std::collections::BTreeMap::<u16, usize>::new();
    for i in 0..200 {
        let text = match i {
            _ if i % 2 == 0 => format!("sensor {} reports nominal", i % 10),
            _ if i % 20 == 1 => format!("ALERT: pressure spike at valve {i}"),
            _ => format!("event {i}: door opened"),
        };
        let url = format!("http://127.0.0.1:{PORT}/events");
        let response = client
            .post(url)
            .body(text)
            .send()
            .await
            .expect("the server answers");
        *statuses.entry(response.status().as_u16()).or_default() += 1;
    }
    println!("\nText events by status (204 quiet or reflex, 202 to the brain): {statuses:?}");

    let (mut quiet, mut queued, start) = (0, 0, Instant::now());
    for i in 0..200u32 {
        let (mean, seed) = match i {
            _ if i % 2 == 0 => (0.1, i % 5),
            _ if i % 40 == 1 => (0.9, i),
            _ => (0.5, i),
        };
        let tensor = frame(mean, seed);
        // The feature: a sketch of the quantized frame, so equal frames habituate
        let Values::F32(v) = &tensor.values else {
            continue;
        };
        let bytes: Vec<u8> = v.iter().map(|x| (x * 255.0) as u8).collect();
        let feature = Feature::sketch_as(kinds::BYTES + 10, &bytes);
        if let Some(outcome) = frames.emit(frames.signal(feature, Payload::tensors(vec![tensor]))) {
            quiet += usize::from(outcome.is_quiet());
            queued += usize::from(outcome.queued > 0);
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!("Frames: {quiet} quiet at the receptor, {queued} to the brain ({ms:.1} ms for all)");
}

/// Wait for the brain's queue to drain, then print what `/metrics` counted
async fn summary() {
    let deadline = Instant::now() + Duration::from_secs(30);
    let nerve = loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let url = format!("http://127.0.0.1:{PORT}/metrics");
        let metrics: serde_json::Value = reqwest::get(url)
            .await
            .expect("metrics")
            .json()
            .await
            .expect("json");
        if metrics["nerve"]["waiting"] == 0 || Instant::now() > deadline {
            break metrics["nerve"].clone();
        }
    };
    println!(
        "\nSignals received {}, reached the brain {}: brain fraction {:.2}",
        nerve["received"], nerve["reached_brain"], nerve["brain_fraction"]
    );
    for n in nerve["neurons"].as_array().into_iter().flatten() {
        println!(
            "  {:<17} {:<9} fired {:>4}, quiet by habituation {:>3}",
            n["name"].as_str().unwrap_or(""),
            n["kind"].as_str().unwrap_or(""),
            n["fired"],
            n["quiet_by_habituation"]
        );
    }
    println!("Pathway and reflex-arc latencies: GET /nerve");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let journal = std::env::temp_dir().join("furnace-nerve-journal.log");
    let _ = std::fs::remove_file(&journal);
    // The webhook receiver standing in for the service or device the nervous system commands
    let hooks = axum::Router::new().route(
        "/hook",
        post(|body: String| async move {
            println!("  [webhook] {}", body.chars().take(100).collect::<String>());
            axum::http::StatusCode::NO_CONTENT
        }),
    );
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", HOOK_PORT)).await?;
    tokio::spawn(async move { axum::serve(listener, hooks).await });

    let (fabric, frames) = build_fabric(&journal)?;
    let server = Server::new().host("127.0.0.1").port(PORT).nerve(fabric);
    tokio::spawn(async move {
        if let Err(e) = server.serve().await {
            eprintln!("server: {e}");
            std::process::exit(1);
        }
    });
    let health = format!("http://127.0.0.1:{PORT}/healthz");
    while reqwest::get(&health).await.is_err() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    println!("furnace nerve demo: server on :{PORT}, webhook receiver on :{HOOK_PORT}");

    if std::env::args().any(|a| a == "--serve") {
        println!("Serving; POST text to /events, GET /nerve. Ctrl-C to stop.");
        tokio::signal::ctrl_c().await?;
        return Ok(());
    }
    demo_stream(frames).await;
    summary().await;
    let lines = std::fs::read_to_string(&journal).map_or(0, |s| s.lines().count());
    println!("The journal ({}) has {lines} lines.", journal.display());
    Ok(())
}
