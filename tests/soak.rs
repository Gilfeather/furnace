//! A soak run: the server and the nervous system under a steady load for as long as asked,
//! checking that they keep answering, answer without errors, and that the process's resident
//! memory levels off instead of growing for as long as the run lasts.
//!
//! Ignored by default; `.github/workflows/soak.yml` runs it by hand and weekly:
//!
//! ```sh
//! FURNACE_SOAK_SECS=3600 cargo test --profile ci --test soak -- --ignored --nocapture
//! ```
//!
//! - `FURNACE_SOAK_SECS`: how long to run (default 20)
//! - `FURNACE_SOAK_INTERVALS`: how many intervals to cut the run into (default 10)
//! - `FURNACE_SOAK_RATE`: requests a second on each route, and signals a second into the
//!   in-process receptor (default 200)
//! - `FURNACE_SOAK_OUT`: a file to write one JSON line per interval to
//!
//! The pass mark is about shape, not speed: no request fails, every interval completes
//! requests, and the largest resident size of the last quarter of the run is within 10% (or
//! 16 MiB) of the largest of the first quarter after the warm-up interval. Resident memory is
//! read from `/proc/self/status` on Linux and `ps` elsewhere; where neither works (Windows),
//! it is not judged. The numbers are written out for the record only.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use furnace::batching::BatchConfig;
use furnace::nerve::effector::{ChannelOut, Respond};
use furnace::nerve::receptor::{ChannelIn, Http};
use furnace::nerve::{
    Fabric, Feature, Firing, FnNeuron, Habituation, Mechanisms, NeuronKind, Payload, Summation,
};
use furnace::{HostTensor, IoSpec, Model, Server, ServerConfig, TensorModel};

const WIDTH: usize = 16;

/// Answers with its input
#[derive(Debug)]
struct Echo;

impl TensorModel for Echo {
    fn name(&self) -> &str {
        "echo"
    }
    fn backend(&self) -> &str {
        "cpu"
    }
    fn inputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("input", &[WIDTH])]
    }
    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("output", &[WIDTH])]
    }
    fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
        Ok(inputs)
    }
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// This process's resident memory in KiB, where it can be read
fn resident_kib() -> Option<u64> {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
        return line.split_whitespace().nth(1)?.parse().ok();
    }
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

#[derive(Default)]
struct Counts {
    sent: AtomicU64,
    ok: AtomicU64,
    errors: AtomicU64,
}

impl Counts {
    fn record(&self, ok: bool, what: impl FnOnce() -> String) {
        if ok {
            self.ok.fetch_add(1, Ordering::Relaxed);
        } else if self.errors.fetch_add(1, Ordering::Relaxed) < 10 {
            eprintln!("soak: {}", what());
        }
    }
}

/// The nervous system under load: POST /events (habituating, a reflex for alerts, a brain
/// behind a deliberative pathway for the rest) and an in-process receptor with a reflex to a
/// device channel
fn fabric(
    device: ChannelIn,
) -> (
    furnace::nerve::FabricBuilder,
    Vec<std::sync::mpsc::Receiver<furnace::nerve::effector::Delivery>>,
) {
    let mut b = Fabric::builder();
    let events = b.receptor(Http::new("events", "/events").text());
    b.gate(
        events,
        Mechanisms::NONE.habituation(Habituation::new(Duration::from_secs(1))),
    );
    let classify = b.neuron(FnNeuron::reflex("classify", |signal, _| {
        let text = signal.payload.as_text().unwrap_or_default();
        if text.contains("ALERT") {
            signal.command(
                "respond",
                serde_json::json!({ "status": 200, "body": "alarm raised" }),
            )
        } else {
            signal
        }
    }));
    let brain = b.neuron(
        FnNeuron::reflex("brain", |signal, _| {
            let text = signal.payload.as_text().unwrap_or_default().to_string();
            signal.command("journal", serde_json::json!({ "line": text }))
        })
        .with_kind(NeuronKind::Brain),
    );
    b.effector(Respond::new("respond"));
    let (journal, journal_rx) = ChannelOut::new("journal", 1024);
    b.effector(journal);
    b.reflex(events, classify);
    b.deliberative(classify, brain)
        .summation(Summation::default().window(Duration::from_millis(1)));

    let device = b.receptor(device);
    let (motor, motor_rx) = ChannelOut::new("motor", 1024);
    let motor = b.effector(motor);
    let reflex = b.neuron(FnNeuron::reflex("steer", |signal, _| Firing::one(signal)));
    b.reflex(device, reflex);
    b.reflex(reflex, motor);
    (b, vec![journal_rx, motor_rx])
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a soak run, minutes to hours: see .github/workflows/soak.yml"]
async fn the_server_and_the_nerve_hold_steady() {
    let secs = env("FURNACE_SOAK_SECS", 20).max(1);
    let intervals = env("FURNACE_SOAK_INTERVALS", 10).max(1);
    let rate = env("FURNACE_SOAK_RATE", 200).max(1);
    let out = std::env::var("FURNACE_SOAK_OUT").ok();
    let interval = Duration::from_secs(secs) / intervals as u32;

    let (device, device_tx) = ChannelIn::new("device", 1024);
    let (nerve, outlets) = fabric(device);
    // The effectors' receivers are drained as a device would
    for outlet in outlets {
        std::thread::spawn(move || for _ in outlet {});
    }
    let config = ServerConfig {
        batching: BatchConfig {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(1),
            workers: 2,
        },
        ..ServerConfig::default()
    };
    let model = Model::new(Box::new(Echo), "test", 0);
    let app = Server::with_config(config)
        .model(model)
        .nerve(nerve)
        .router()
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let counts = Arc::new(Counts::default());
    let predict = serde_json::json!({ "input": vec![0.5; WIDTH] }).to_string();

    let started = Instant::now();
    let mut ticks = tokio::time::interval(Duration::from_millis(10));
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut records = Vec::new();
    let mut n = 0u64;
    let mut done_before = 0;
    for index in 0..intervals {
        let end = started + interval * (index as u32 + 1);
        let mut max_kib = resident_kib();
        let mut next_sample = Instant::now();
        while Instant::now() < end {
            ticks.tick().await;
            if Instant::now() >= next_sample {
                max_kib = max_kib.max(resident_kib());
                next_sample += Duration::from_millis(250);
            }
            // As many as the rate owes by now, so the timer's resolution does not change it
            let due = (started.elapsed().as_secs_f64() * rate as f64) as u64;
            while n < due {
                n += 1;
                // One request on each route and one signal into the fabric
                counts.sent.fetch_add(2, Ordering::Relaxed);
                let (c, k) = (client.clone(), counts.clone());
                let body = predict.clone();
                let url = format!("{base}/predict");
                tokio::spawn(async move {
                    let result = c
                        .post(url)
                        .header("content-type", "application/json")
                        .body(body)
                        .send()
                        .await;
                    match result {
                        Ok(r) => {
                            let status = r.status();
                            k.record(status == 200, || format!("/predict: {status}"));
                        }
                        Err(e) => k.record(false, || format!("/predict: {e}")),
                    }
                });
                // Lines repeat (and habituate) in cycles; one in ten is an alert
                let line = if n.is_multiple_of(10) {
                    format!("ALERT {}", n % 50)
                } else {
                    format!("reading {}", n % 500)
                };
                let (c, k) = (client.clone(), counts.clone());
                let url = format!("{base}/events");
                tokio::spawn(async move {
                    match c
                        .post(url)
                        .header("content-type", "text/plain")
                        .body(line)
                        .send()
                        .await
                    {
                        Ok(r) => {
                            let status = r.status();
                            k.record(status.is_success(), || format!("/events: {status}"));
                        }
                        Err(e) => k.record(false, || format!("/events: {e}")),
                    }
                });
                let signal =
                    device_tx.signal(Feature::sketch(&n.to_le_bytes()), Payload::text("turn"));
                if device_tx.send(signal).is_err() {
                    counts.record(false, || "device: the receptor's queue is full".to_string());
                }
            }
        }
        let max_kib = max_kib.max(resident_kib());
        let ok = counts.ok.load(Ordering::Relaxed);
        let errors = counts.errors.load(Ordering::Relaxed);
        let done = ok + errors;
        let record = serde_json::json!({
            "interval": index,
            "elapsed_s": started.elapsed().as_secs_f64(),
            "sent": counts.sent.load(Ordering::Relaxed),
            "ok": ok,
            "errors": errors,
            "completed_in_interval": done - done_before,
            "max_rss_kib": max_kib,
        });
        println!("soak: {record}");
        if let Some(path) = &out {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            writeln!(file, "{record}").unwrap();
        }
        assert!(
            done > done_before,
            "no request completed in interval {index}: a stall"
        );
        done_before = done;
        records.push(max_kib);
    }

    let errors = counts.errors.load(Ordering::Relaxed);
    assert_eq!(
        errors, 0,
        "{errors} request(s) failed; the first are printed above"
    );

    let rss: Vec<u64> = records.iter().flatten().copied().collect();
    if rss.len() < 4 || rss.len() != records.len() {
        println!(
            "soak: too few memory samples to judge ({} of {})",
            rss.len(),
            records.len()
        );
        return;
    }
    let quarter = rss.len().div_ceil(4);
    let early = *rss[1..=quarter.min(rss.len() - 1)].iter().max().unwrap();
    let late = *rss[rss.len() - quarter..].iter().max().unwrap();
    let allowed = early + (early / 10).max(16 * 1024);
    println!(
        "soak: largest resident size {early} KiB early, {late} KiB late, {allowed} KiB allowed"
    );
    assert!(
        late <= allowed,
        "resident memory kept growing: {early} KiB early in the run, {late} KiB at its end"
    );
}
