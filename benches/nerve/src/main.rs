//! The nervous system against a hand-written server doing the same work in a handler, and
//! FastAPI where Python can do it. The work is the demo's triage: a 64-bit sketch of the
//! input as its feature, habituation (a repeated feature attenuates and dies below the
//! threshold), then `ALERT` lines act at once and everything else goes to the brain.
//!
//! ```bash
//! cargo build --release
//! target/release/furnace-nerve-bench [hop|arc|brain|memory|webhook|serve-bench|all]
//!     [--runs 3] [--clients 1,4,16] [--servers furnace,hand-written,fastapi] [--json PATH]
//! ```
//!
//! Every number is the median of `--runs` runs. The measurements:
//!
//! - *hop*: propagation cost per hop with no work: a chain of 8 pass-through reflex neurons,
//!   against calling 8 boxed closures in a loop.
//! - *arc*: a reflex arc, stimulus to effector call, in process: receptor → triage → channel
//!   effector, against a function doing the same sketch, table and send.
//! - *brain*: the brain fraction on a stream of 10,000 signals of which half are repeats, as
//!   it falls with habituation; a plain request/response server's is 1.0 by construction.
//! - *memory*: private memory this process gains while a fabric carries 1,000 signals/s for
//!   10 s, habituation table and deliberative queue included.
//! - *webhook*: stimulus to outbound command on a reflex arc: the time from the stimulus to
//!   the webhook's arrival at a local receiver, against the raw HTTP call with the same client.
//! - *serve-bench*: the demo served over HTTP by furnace (`Server::nerve`), by a hand-written
//!   axum server and by FastAPI, loaded by N clients on keep-alive connections.
//!
//! `serve KIND PORT` runs one server on its own.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use furnace::Server;
use furnace::nerve::effector::{Channel as ChannelEffector, Effector, Respond, Webhook};
use furnace::nerve::receptor::{Channel, Http};
use furnace::nerve::wire;
use furnace::nerve::{
    Fabric, Feature, Firing, FnNeuron, Habituation, Mechanisms, NeuronKind, Payload, Signal,
    Summation, Urgency, sketch64,
};
use furnace_bench_common::{
    fail, free_port, furnace_version, memory, percentile, python, stop, wait_until_healthy,
    write_json,
};
use serde::Serialize;

const THRESHOLD: f32 = 0.6;
const TAU: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const WARMUP: usize = 5;

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

// ---------------------------------------------------------------------------------------------
// The work

/// The demo's events: half repeats from a pool of 50, 5% alerts, the rest novel
fn event(i: usize) -> String {
    if i.is_multiple_of(2) {
        format!("sensor {} reports nominal", i % 100 / 2)
    } else if i % 20 == 1 {
        format!("ALERT: pressure spike at valve {i}")
    } else {
        format!("event {i}: door opened")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Quiet,
    Urgent,
    Novel,
}

/// The hand-written side's habituation: the same table as furnace's gate (set-associative,
/// 4 ways, exponential decay), written out in a handler
struct Table {
    slots: Vec<(u64, f32, Instant)>,
    sets: usize,
}

impl Table {
    fn new(capacity: usize) -> Self {
        let sets = capacity / 4;
        Self {
            slots: vec![(0, 0.0, Instant::now()); sets * 4],
            sets,
        }
    }

    /// Attenuation for `sketch` seen now
    fn attenuation(&mut self, sketch: u64, now: Instant) -> f32 {
        let set = sketch as usize % self.sets;
        let slots = &mut self.slots[set * 4..set * 4 + 4];
        let mut victim = 0;
        let mut victim_seen = None;
        for (i, slot) in slots.iter_mut().enumerate() {
            if slot.1 > 0.0 && slot.0 == sketch {
                let dt = now.saturating_duration_since(slot.2).as_secs_f32();
                let decayed = slot.1 * (-dt / TAU.as_secs_f32()).exp();
                slot.1 = decayed + (1.0 - decayed) * 0.5;
                slot.2 = now;
                return decayed;
            }
            // An empty slot first, else the least recently seen
            let seen = (slot.1 > 0.0).then_some(slot.2);
            if victim_seen.is_none_or(|v: Option<Instant>| seen < v) {
                victim = i;
                victim_seen = Some(seen);
            }
        }
        slots[victim] = (sketch, 0.5, now);
        0.0
    }
}

/// The hand-written triage: sketch, habituate, classify
fn triage(table: &Mutex<Table>, body: &[u8]) -> Verdict {
    let sketch = sketch64(body);
    let attenuation = table.lock().unwrap().attenuation(sketch, Instant::now());
    if 1.0 - attenuation < THRESHOLD {
        return Verdict::Quiet;
    }
    if body.windows(5).any(|w| w == b"ALERT") {
        Verdict::Urgent
    } else {
        Verdict::Novel
    }
}

fn is_alert(text: &str) -> bool {
    text.contains("ALERT")
}

/// The demo's fabric: receptor (habituation) → triage reflex → `alarm` effector for urgent
/// signals, deliberative pathway to a counting brain for the rest. `alarm` is whatever the
/// measurement needs; the receptor is an HTTP route or an in-process channel.
struct Demo {
    builder: furnace::nerve::FabricBuilder,
    brain_hits: Arc<AtomicU64>,
}

/// A built demo, for driving it in this process
struct Built {
    fabric: Fabric,
    brain_hits: Arc<AtomicU64>,
}

impl Demo {
    fn build(self) -> Built {
        Built {
            fabric: self.builder.build(),
            brain_hits: self.brain_hits,
        }
    }
}

fn demo_fabric(
    receptor: impl FnOnce(&mut furnace::nerve::FabricBuilder, Mechanisms) -> furnace::nerve::NeuronId,
    alarm: Effector,
    respond: bool,
) -> Demo {
    let mut b = Fabric::builder();
    let r = receptor(
        &mut b,
        Mechanisms::NONE
            .threshold(THRESHOLD)
            .habituation(Habituation::new(TAU).capacity(1024)),
    );
    let triage = b.neuron(
        FnNeuron::reflex("triage", move |signal, _| {
            if signal.payload.as_text().is_some_and(is_alert) {
                let mut fired = vec![
                    signal
                        .command("alarm", serde_json::json!({"alarm": true}))
                        .with_urgency(Urgency::within(Duration::from_millis(2))),
                ];
                if respond {
                    fired.push(signal.command(
                        "respond",
                        serde_json::json!({"status": 200, "body": "alarm"}),
                    ));
                }
                Firing::Fire(fired)
            } else {
                Firing::one(signal)
            }
        }),
        Mechanisms::NONE,
    );
    let brain_hits = Arc::new(AtomicU64::new(0));
    let hits = brain_hits.clone();
    let brain = b.neuron(
        FnNeuron::reflex("brain", move |_, _| {
            hits.fetch_add(1, Ordering::Relaxed);
            Firing::Fire(Vec::new())
        })
        .with_kind(NeuronKind::Brain),
        Mechanisms::NONE,
    );
    let alarm = b.neuron(alarm, Mechanisms::NONE);
    if respond {
        b.neuron(Effector::new(Respond::new("respond")), Mechanisms::NONE);
    }
    b.reflex(r, triage, 1.0);
    b.reflex(triage, alarm, 1.0);
    b.deliberative(triage, brain, 2.0, Summation::default());
    Demo {
        builder: b,
        brain_hits,
    }
}

// ---------------------------------------------------------------------------------------------
// hop: propagation cost per hop, no work

const HOPS: usize = 8;
const HOP_SIGNALS: usize = 200_000;

#[derive(Debug, Serialize, Clone)]
struct Hop {
    /// Fabric: nanoseconds per hop, the receptor included
    fabric_ns_per_hop: f64,
    /// A chain of boxed closures: nanoseconds per call
    closures_ns_per_hop: f64,
}

fn bench_hop() -> Hop {
    let mut b = Fabric::builder();
    let pass = |name: String| FnNeuron::reflex(&name, |s, _| Firing::one(s));
    let first = b.neuron(
        FnNeuron::reflex("in", |s, _| Firing::one(s)).with_kind(NeuronKind::Receptor),
        Mechanisms::NONE,
    );
    let mut previous = first;
    for i in 1..HOPS {
        let next = b.neuron(pass(format!("hop{i}")), Mechanisms::NONE);
        b.reflex(previous, next, 1.0);
        previous = next;
    }
    let fabric = b.build();
    let signal = || Signal::new(Feature::NONE, Payload::None);
    for _ in 0..1000 {
        fabric.stimulate(first, signal());
    }
    let start = Instant::now();
    for _ in 0..HOP_SIGNALS {
        std::hint::black_box(fabric.stimulate(first, signal()));
    }
    let fabric_ns = start.elapsed().as_nanos() as f64 / (HOP_SIGNALS * HOPS) as f64;

    type Hopper = Box<dyn Fn(Signal) -> Signal + Send + Sync>;
    let chain: Vec<Hopper> = (0..HOPS)
        .map(|_| Box::new(|s: Signal| s) as Hopper)
        .collect();
    let start = Instant::now();
    for _ in 0..HOP_SIGNALS {
        let mut s = signal();
        for hop in &chain {
            s = hop(s);
        }
        std::hint::black_box(s);
    }
    let closures_ns = start.elapsed().as_nanos() as f64 / (HOP_SIGNALS * HOPS) as f64;
    Hop {
        fabric_ns_per_hop: fabric_ns,
        closures_ns_per_hop: closures_ns,
    }
}

// ---------------------------------------------------------------------------------------------
// arc: a reflex arc, stimulus to effector call, in process

const ARC_SIGNALS: usize = 100_000;

#[derive(Debug, Serialize, Clone)]
struct Arc_ {
    /// Per stimulus, from the caller's clock: the whole synchronous chain
    fabric_p50_us: f64,
    fabric_p99_us: f64,
    /// The fabric's own stimulus-to-effector-call measurement of the arc
    arc_p50_us: f64,
    arc_p99_us: f64,
    /// The hand-written function: sketch, table, classify, channel send
    hand_written_p50_us: f64,
    hand_written_p99_us: f64,
    /// Of the stimuli, how many reached the effector (alerts that were not habituated)
    acted: u64,
}

fn bench_arc() -> Arc_ {
    let (channel, receiver) = ChannelEffector::new("alarm", 4096);
    let (receptor, sender) = Channel::new("in", 16);
    let demo = demo_fabric(
        |b, mechanisms| b.receptor(receptor, mechanisms),
        Effector::new(channel),
        false,
    )
    .build();
    demo.fabric.start();
    // Alerts only: every stimulus is meant to reach the effector; repeats die at the gate
    let text = |i: usize| format!("ALERT {}", i % 50_000);
    let mut times = Vec::with_capacity(ARC_SIGNALS);
    for i in 0..ARC_SIGNALS {
        let t = text(i);
        let signal = sender.signal(Feature::sketch(t.as_bytes()), Payload::text(&t));
        let start = Instant::now();
        std::hint::black_box(sender.emit(signal));
        times.push(start.elapsed());
        if i % 1000 == 999 {
            receiver.try_iter().count();
        }
    }
    let acted = receiver.try_iter().count() as u64
        + demo
            .fabric
            .metrics()
            .neurons
            .iter()
            .find(|n| n.name == "alarm")
            .map_or(0, |n| n.fired)
        - receiver.try_iter().count() as u64;
    let acted = acted.min(ARC_SIGNALS as u64);
    times.sort();
    let arc = demo
        .fabric
        .metrics()
        .arcs
        .into_iter()
        .find(|a| a.to == "alarm")
        .map(|a| a.latency)
        .unwrap_or_default();

    // Hand-written: the same work as a function
    let table = Mutex::new(Table::new(1024));
    let (tx, rx) = std::sync::mpsc::sync_channel::<(String, Instant)>(1 << 16);
    let mut hand = Vec::with_capacity(ARC_SIGNALS);
    for i in 0..ARC_SIGNALS {
        let t = text(i);
        let start = Instant::now();
        let verdict = triage(&table, t.as_bytes());
        if verdict == Verdict::Urgent {
            let _ = tx.try_send((t, start));
        }
        std::hint::black_box(verdict);
        hand.push(start.elapsed());
        if i % 1000 == 999 {
            rx.try_iter().count();
        }
    }
    hand.sort();
    let _ = demo.brain_hits;
    Arc_ {
        fabric_p50_us: us(percentile(&times, 0.5)),
        fabric_p99_us: us(percentile(&times, 0.99)),
        arc_p50_us: arc.p50_us.unwrap_or(0.0),
        arc_p99_us: arc.p99_us.unwrap_or(0.0),
        hand_written_p50_us: us(percentile(&hand, 0.5)),
        hand_written_p99_us: us(percentile(&hand, 0.99)),
        acted,
    }
}

// ---------------------------------------------------------------------------------------------
// brain: the brain fraction on a stream with 50% repeats

const BRAIN_SIGNALS: usize = 10_000;

#[derive(Debug, Serialize, Clone)]
struct BrainFraction {
    /// After 1k, 2k, 5k and 10k signals
    at: Vec<(usize, f64)>,
    /// Of the whole stream
    overall: f64,
    /// Repeats in the stream: what a plain server would still send to its model
    repeat_fraction: f64,
}

fn bench_brain() -> BrainFraction {
    let (channel, _receiver) = ChannelEffector::new("alarm", 4096);
    let (receptor, sender) = Channel::new("in", 16);
    let demo = demo_fabric(
        |b, mechanisms| b.receptor(receptor, mechanisms),
        Effector::new(channel),
        false,
    )
    .build();
    demo.fabric.start();
    let mut at = Vec::new();
    let mut repeats = 0;
    for i in 0..BRAIN_SIGNALS {
        let t = event(i);
        if i.is_multiple_of(2) && i >= 100 {
            repeats += 1;
        }
        sender.emit(sender.signal(Feature::sketch(t.as_bytes()), Payload::text(&t)));
        if [1000, 2000, 5000, 10_000].contains(&(i + 1)) {
            at.push((i + 1, demo.fabric.brain_fraction()));
        }
    }
    // The deliberative queue drains; the fraction counts what reached the brain
    let deadline = Instant::now() + Duration::from_secs(10);
    while demo.fabric.waiting() > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    BrainFraction {
        overall: demo.fabric.brain_fraction(),
        at,
        repeat_fraction: repeats as f64 / BRAIN_SIGNALS as f64,
    }
}

// ---------------------------------------------------------------------------------------------
// memory: the fabric at 1k signals/s

#[derive(Debug, Serialize, Clone)]
struct Memory {
    /// Private bytes of this process before the fabric was built
    base_mb: f64,
    /// With the fabric built and started, idle: its runtimes and threads
    built_mb: f64,
    /// After 10 s at 1,000 signals/s
    after_mb: f64,
    added_mb: f64,
    signals: u64,
}

fn bench_memory() -> Memory {
    let base = memory::of_process(std::process::id()).private_mb;
    let (channel, receiver) = ChannelEffector::new("alarm", 4096);
    let (receptor, sender) = Channel::new("in", 16);
    let demo = demo_fabric(
        |b, mechanisms| b.receptor(receptor, mechanisms),
        Effector::new(channel),
        false,
    )
    .build();
    demo.fabric.start();
    std::thread::sleep(Duration::from_millis(100));
    let built = memory::of_process(std::process::id()).private_mb;
    let start = Instant::now();
    let mut sent = 0u64;
    while start.elapsed() < Duration::from_secs(10) {
        let t = event(sent as usize);
        sender.emit(sender.signal(Feature::sketch(t.as_bytes()), Payload::text(&t)));
        sent += 1;
        receiver.try_iter().count();
        let next = start + Duration::from_micros(1000 * sent);
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        }
    }
    let after = memory::of_process(std::process::id()).private_mb;
    Memory {
        base_mb: base,
        built_mb: built,
        after_mb: after,
        added_mb: after - base,
        signals: sent,
    }
}

// ---------------------------------------------------------------------------------------------
// webhook: stimulus to outbound command, reflex arc, localhost

const WEBHOOK_CALLS: usize = 2000;
/// Between webhook calls, so that the runtimes park between them as they would for sporadic
/// signals, in every variant alike
const PACE: Duration = Duration::from_millis(1);

#[derive(Debug, Serialize, Clone)]
struct WebhookLatency {
    /// The raw HTTP call with the same client, on the calling runtime: send to arrival at
    /// the receiver
    raw_p50_us: f64,
    raw_p99_us: f64,
    /// The raw call handed to another runtime's thread (what the motor side does): the
    /// thread hop alone
    handed_p50_us: f64,
    handed_p99_us: f64,
    /// Through the fabric: stimulus to arrival at the receiver
    fabric_p50_us: f64,
    fabric_p99_us: f64,
    added_p50_us: f64,
}

/// A local receiver that records when each numbered command arrived
async fn webhook_receiver() -> (u16, Arc<Mutex<Vec<Option<Instant>>>>) {
    let arrivals = Arc::new(Mutex::new(vec![None; WEBHOOK_CALLS + WARMUP]));
    let app = Router::new()
        .route(
            "/hook",
            post(
                |State(arrivals): State<Arc<Mutex<Vec<Option<Instant>>>>>, body: String| async move {
                    let now = Instant::now();
                    let n: usize = serde_json::from_str::<serde_json::Value>(&body)
                        .ok()
                        .and_then(|v| v["n"].as_u64())
                        .map_or(usize::MAX, |n| n as usize);
                    if let Some(slot) = arrivals.lock().unwrap().get_mut(n) {
                        *slot = Some(now);
                    }
                    StatusCode::NO_CONTENT
                },
            ),
        )
        .with_state(arrivals.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (port, arrivals)
}

async fn bench_webhook() -> WebhookLatency {
    let (port, arrivals) = webhook_receiver().await;
    let url = wire::Url::parse(&format!("http://127.0.0.1:{port}/hook")).unwrap();

    // Raw: one call at a time, the same client the effector uses
    let client = wire::Client::default();
    let mut raw = Vec::with_capacity(WEBHOOK_CALLS);
    for n in 0..WEBHOOK_CALLS + WARMUP {
        // One call per millisecond: every path pays the same idle-runtime wake-ups
        tokio::time::sleep(PACE).await;
        let body = format!("{{\"n\":{n}}}");
        let start = Instant::now();
        client
            .request(
                "POST",
                &url,
                &[("content-type", "application/json")],
                body.as_bytes(),
            )
            .await
            .expect("the receiver answers");
        let arrived = arrivals.lock().unwrap()[n].take().expect("recorded");
        if n >= WARMUP {
            raw.push(arrived.saturating_duration_since(start));
        }
    }
    raw.sort();

    // The same call handed to another runtime, as the efferent side does
    let other = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let client = Arc::new(wire::Client::default());
    let mut handed = Vec::with_capacity(WEBHOOK_CALLS);
    for n in 0..WEBHOOK_CALLS + WARMUP {
        // One call per millisecond: every path pays the same idle-runtime wake-ups
        tokio::time::sleep(PACE).await;
        let body = format!("{{\"n\":{n}}}");
        let (client, url) = (client.clone(), url.clone());
        let start = Instant::now();
        other
            .spawn(async move {
                client
                    .request(
                        "POST",
                        &url,
                        &[("content-type", "application/json")],
                        body.as_bytes(),
                    )
                    .await
                    .expect("the receiver answers");
            })
            .await
            .unwrap();
        let arrived = arrivals.lock().unwrap()[n].take().expect("recorded");
        if n >= WARMUP {
            handed.push(arrived.saturating_duration_since(start));
        }
    }
    handed.sort();
    std::thread::spawn(move || drop(other));

    // Through the fabric: channel receptor → triage → webhook effector, one at a time
    let (receptor, sender) = Channel::new("in", 16);
    let demo = demo_fabric(
        |b, mechanisms| b.receptor(receptor, mechanisms),
        Effector::new(Webhook::new(
            "alarm",
            &format!("http://127.0.0.1:{port}/hook"),
        ))
        .no_retry(),
        false,
    )
    .build();
    demo.fabric.start();
    let mut through = Vec::with_capacity(WEBHOOK_CALLS);
    for n in 0..WEBHOOK_CALLS + WARMUP {
        // One call per millisecond: every path pays the same idle-runtime wake-ups
        tokio::time::sleep(PACE).await;
        // Every stimulus is a distinct alert, so none habituates; the command carries `n`
        let text = format!("ALERT {n}");
        let signal = sender.signal(Feature::sketch(text.as_bytes()), Payload::text(&text));
        let start = Instant::now();
        // The triage reflex emits a fixed command; the webhook needs the number, so the
        // receptor's signal is stimulated through a reflex of this measurement's own
        let signal = signal.command("alarm", serde_json::json!({"n": n}));
        // Synchronous and short: the reflex chain and the hand-off to the motor side
        sender.emit(signal);
        let arrived = loop {
            if let Some(at) = arrivals.lock().unwrap()[n] {
                break at;
            }
            tokio::time::sleep(Duration::from_micros(20)).await;
            if start.elapsed() > Duration::from_secs(5) {
                fail("the webhook never arrived");
            }
        };
        if n >= WARMUP {
            through.push(arrived.saturating_duration_since(start));
        }
    }
    through.sort();
    WebhookLatency {
        raw_p50_us: us(percentile(&raw, 0.5)),
        raw_p99_us: us(percentile(&raw, 0.99)),
        handed_p50_us: us(percentile(&handed, 0.5)),
        handed_p99_us: us(percentile(&handed, 0.99)),
        fabric_p50_us: us(percentile(&through, 0.5)),
        fabric_p99_us: us(percentile(&through, 0.99)),
        added_p50_us: us(percentile(&through, 0.5)) - us(percentile(&raw, 0.5)),
    }
}

// ---------------------------------------------------------------------------------------------
// Servers

/// The hand-written server: the triage in a handler
struct HandWritten {
    table: Mutex<Table>,
    brain: AtomicU64,
}

async fn hand_written_events(State(server): State<Arc<HandWritten>>, body: Bytes) -> StatusCode {
    match triage(&server.table, &body) {
        Verdict::Quiet => StatusCode::NO_CONTENT,
        Verdict::Urgent => StatusCode::OK,
        Verdict::Novel => {
            server.brain.fetch_add(1, Ordering::Relaxed);
            StatusCode::ACCEPTED
        }
    }
}

async fn serve(kind: &str, port: u16) {
    match kind {
        "furnace" => {
            let (channel, _receiver) = ChannelEffector::new("alarm", 4096);
            let demo = demo_fabric(
                |b, mechanisms| b.receptor(Http::new("events", "/events").text(), mechanisms),
                Effector::new(channel),
                true,
            );
            let result = Server::new()
                .host("127.0.0.1")
                .port(port)
                .nerve(demo.builder)
                .serve()
                .await;
            if let Err(e) = result {
                fail(&format!("furnace: {e}"));
            }
        }
        "hand-written" => {
            let state = Arc::new(HandWritten {
                table: Mutex::new(Table::new(1024)),
                brain: AtomicU64::new(0),
            });
            let app = Router::new()
                .route("/events", post(hand_written_events))
                .route("/healthz", get(|| async { "ok" }))
                .with_state(state);
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
                .await
                .unwrap_or_else(|e| fail(&format!("bind {port}: {e}")));
            axum::serve(listener, app).await.expect("serve");
        }
        _ => fail(&format!("unknown server {kind}: furnace or hand-written")),
    }
}

// ---------------------------------------------------------------------------------------------
// Load

#[derive(Debug, Clone, Serialize)]
struct Load {
    clients: usize,
    requests_per_second: f64,
    p50_ms: f64,
    p95_ms: f64,
    /// Responses by status: 204 (quiet), 200 (alarm), 202 (to the brain)
    quiet: usize,
    alarm: usize,
    brain: usize,
}

async fn load(url: Arc<String>, clients: usize, per_client: usize) -> Load {
    let connect = || {
        reqwest::Client::builder()
            .pool_max_idle_per_host(1)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("client")
    };
    let call = move |client: reqwest::Client, c: usize, i: usize| {
        let url = url.clone();
        async move {
            let start = Instant::now();
            let response = client
                .post(url.as_str())
                .body(event(c * 100_000 + i))
                .send()
                .await
                .expect("request failed");
            let status = response.status().as_u16();
            let _ = response.bytes().await;
            (start.elapsed(), status)
        }
    };
    let run = furnace_bench_common::load(clients, WARMUP, per_client, connect, call, || ()).await;
    let (times, wall) = (run.results, run.wall);
    let count = |status: u16| times.iter().filter(|(_, s)| *s == status).count();
    let (quiet, alarm, brain) = (count(204), count(200), count(202));
    let mut durations: Vec<Duration> = times.into_iter().map(|(d, _)| d).collect();
    durations.sort();
    Load {
        clients,
        requests_per_second: durations.len() as f64 / wall.as_secs_f64(),
        p50_ms: us(percentile(&durations, 0.5)) / 1e3,
        p95_ms: us(percentile(&durations, 0.95)) / 1e3,
        quiet,
        alarm,
        brain,
    }
}

#[derive(Debug, Serialize, Clone)]
struct ServerResult {
    server: String,
    ready_ms: f64,
    peak_private_mb: f64,
    load: Vec<Load>,
}

async fn bench_server(kind: &str, clients: &[usize], per_client: usize) -> ServerResult {
    let port = free_port();
    let exe = std::env::current_exe().expect("own path");
    let start = Instant::now();
    let mut command = if kind == "fastapi" {
        let mut command = std::process::Command::new(python());
        command
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/fastapi/triage.py"))
            .args(["--port", &port.to_string()]);
        command
    } else {
        let mut command = std::process::Command::new(exe);
        command.args(["serve", kind, &port.to_string()]);
        command
    };
    let mut child = command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("start server");
    let pid = wait_until_healthy(kind, &mut child, port, Duration::from_secs(120));
    let ready_ms = start.elapsed().as_secs_f64() * 1e3;
    let url = Arc::new(format!("http://127.0.0.1:{port}/events"));
    let mut results = Vec::new();
    for &n in clients {
        results.push(load(url.clone(), n, per_client).await);
    }
    let peak_private_mb = memory::of_process(pid).peak_private_mb;
    stop(&mut child);
    ServerResult {
        server: kind.to_string(),
        ready_ms,
        peak_private_mb,
        load: results,
    }
}

// ---------------------------------------------------------------------------------------------
// The report

#[derive(Debug, Serialize)]
struct Report {
    furnace_version: &'static str,
    runs: usize,
    hop: Option<Hop>,
    arc: Option<Arc_>,
    brain: Option<BrainFraction>,
    memory: Option<Memory>,
    webhook: Option<WebhookLatency>,
    servers: Vec<ServerResult>,
}

/// Run `f` `runs` times and keep the run whose `key` is the median
fn median_run<T: Clone>(runs: usize, mut f: impl FnMut() -> T, key: impl Fn(&T) -> f64) -> T {
    let mut results: Vec<T> = (0..runs).map(|_| f()).collect();
    results.sort_by(|a, b| key(a).total_cmp(&key(b)));
    results[results.len() / 2].clone()
}

fn print_report(report: &Report) {
    println!(
        "furnace {} (nerve), medians of {} runs",
        report.furnace_version, report.runs
    );
    if let Some(hop) = &report.hop {
        println!(
            "\nPropagation per hop, no work: fabric {:.0} ns, a chain of boxed closures {:.0} ns \
             (target <= 1000 ns)",
            hop.fabric_ns_per_hop, hop.closures_ns_per_hop
        );
    }
    if let Some(arc) = &report.arc {
        println!(
            "\nReflex arc, stimulus to effector call, in process (target <= 10 us):\n  \
             fabric, caller's clock: p50 {:.2} us, p99 {:.2} us\n  \
             fabric, its own arc measurement: p50 {:.2} us, p99 {:.2} us\n  \
             hand-written function: p50 {:.2} us, p99 {:.2} us\n  ({} of {} stimuli reached the effector)",
            arc.fabric_p50_us,
            arc.fabric_p99_us,
            arc.arc_p50_us,
            arc.arc_p99_us,
            arc.hand_written_p50_us,
            arc.hand_written_p99_us,
            arc.acted,
            ARC_SIGNALS
        );
    }
    if let Some(brain) = &report.brain {
        let at: Vec<String> = brain
            .at
            .iter()
            .map(|(n, f)| format!("{n}: {f:.3}"))
            .collect();
        println!(
            "\nBrain fraction on {} signals with {:.0}% repeats (target <= 0.5, falling): {} \
             (a plain server: 1.000)",
            BRAIN_SIGNALS,
            brain.repeat_fraction * 100.0,
            at.join(", ")
        );
    }
    if let Some(memory) = &report.memory {
        println!(
            "\nMemory at 1k signals/s for 10 s ({} signals): {:.1} MB before, {:.1} MB with the \
             fabric built and idle, {:.1} MB after: {:.1} MB added (target <= 10 MB)",
            memory.signals, memory.base_mb, memory.built_mb, memory.after_mb, memory.added_mb
        );
    }
    if let Some(webhook) = &report.webhook {
        println!(
            "\nStimulus to webhook arrival, localhost (target <= 100 us added):\n  \
             raw call on the calling runtime: p50 {:.0} us (p99 {:.0})\n  \
             raw call handed to another runtime's thread: p50 {:.0} us (p99 {:.0})\n  \
             through the fabric: p50 {:.0} us (p99 {:.0}): {:.0} us added over the raw call",
            webhook.raw_p50_us,
            webhook.raw_p99_us,
            webhook.handed_p50_us,
            webhook.handed_p99_us,
            webhook.fabric_p50_us,
            webhook.fabric_p99_us,
            webhook.added_p50_us
        );
    }
    if !report.servers.is_empty() {
        println!("\nThe demo over HTTP (POST /events; 204 quiet, 200 alarm, 202 to the brain):");
        println!(
            "{:<14} {:>8} {:>8} {:>10} {:>9} {:>9}   responses",
            "server", "clients", "ready ms", "req/s", "p50 ms", "p95 ms"
        );
        for server in &report.servers {
            for load in &server.load {
                println!(
                    "{:<14} {:>8} {:>8.0} {:>10.0} {:>9.3} {:>9.3}   {} / {} / {}",
                    server.server,
                    load.clients,
                    server.ready_ms,
                    load.requests_per_second,
                    load.p50_ms,
                    load.p95_ms,
                    load.quiet,
                    load.alarm,
                    load.brain
                );
            }
            println!(
                "{:<14} peak private memory {:.1} MB",
                "", server.peak_private_mb
            );
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    if args.first().map(String::as_str) == Some("serve") {
        let (Some(kind), Some(port)) = (args.get(1), args.get(2).and_then(|p| p.parse().ok()))
        else {
            fail("usage: serve KIND PORT");
        };
        serve(kind, port).await;
        return;
    }
    let what = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "all".to_string());
    let runs: usize = flag("--runs").map_or(3, |r| r.parse().unwrap_or_else(|_| fail("--runs")));
    let clients: Vec<usize> = flag("--clients")
        .unwrap_or_else(|| "1,4,16".to_string())
        .split(',')
        .map(|c| {
            c.parse()
                .unwrap_or_else(|_| fail("--clients takes numbers"))
        })
        .collect();
    let servers = flag("--servers").unwrap_or_else(|| "furnace,hand-written,fastapi".to_string());
    let per_client: usize =
        flag("--requests").map_or(2000, |r| r.parse().unwrap_or_else(|_| fail("--requests")));
    let wants = |name: &str| what == "all" || what == name;

    let mut report = Report {
        furnace_version: furnace_version(),
        runs,
        hop: None,
        arc: None,
        brain: None,
        memory: None,
        webhook: None,
        servers: Vec::new(),
    };
    if wants("hop") {
        eprintln!("hop");
        report.hop = Some(tokio::task::block_in_place(|| {
            median_run(runs, bench_hop, |h| h.fabric_ns_per_hop)
        }));
    }
    if wants("arc") {
        eprintln!("arc");
        report.arc = Some(tokio::task::block_in_place(|| {
            median_run(runs, bench_arc, |a| a.fabric_p50_us)
        }));
    }
    if wants("brain") {
        eprintln!("brain");
        report.brain = Some(tokio::task::block_in_place(|| {
            median_run(runs, bench_brain, |b| b.overall)
        }));
    }
    if wants("memory") {
        eprintln!("memory");
        report.memory = Some(tokio::task::block_in_place(|| {
            median_run(runs, bench_memory, |m| m.added_mb)
        }));
    }
    if wants("webhook") {
        eprintln!("webhook");
        let mut results = Vec::new();
        for _ in 0..runs {
            results.push(bench_webhook().await);
        }
        results.sort_by(|a, b| a.added_p50_us.total_cmp(&b.added_p50_us));
        report.webhook = Some(results[results.len() / 2].clone());
    }
    if wants("serve-bench") {
        for kind in servers.split(',') {
            eprintln!("{kind}");
            let mut results = Vec::new();
            for _ in 0..runs {
                results.push(bench_server(kind, &clients, per_client).await);
            }
            // The run whose throughput at the most clients is the median
            results.sort_by(|a, b| {
                let last = |r: &ServerResult| r.load.last().map_or(0.0, |l| l.requests_per_second);
                last(a).total_cmp(&last(b))
            });
            report.servers.push(results[results.len() / 2].clone());
        }
    }
    print_report(&report);
    if let Some(path) = flag("--json") {
        write_json(&path, &report);
    }
}
