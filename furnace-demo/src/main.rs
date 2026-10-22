//! furnace's nervous system in one process, in a few seconds of terminal output.
//!
//! ```text
//! logs  (no deadline) ─┐                    ┌─ reflex ──────────── (µs) ─┐
//!                      ├─▶ triage junction ─┤                            ├─▶ actuator
//! smoke (5 ms)        ─┘                    └─ brain (mock LLM) ─ (ms) ──┘
//! ```
//!
//! Signals come in through two receptors. At the `triage` junction each one takes the
//! heaviest pathway whose measured p99 fits in its remaining deadline: with no deadline that
//! is the brain; with 5 ms, once the brain has been measured, only the reflex fits. Repeats
//! habituate at the receptor and go no further. The brain is the LLM mock engine, so there is
//! no model to download; put a real `LlmEngine` (or any neuron) in its place.
//!
//! ```bash
//! cargo run --release -p furnace-demo
//! ```

use std::io::IsTerminal;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use furnace::llm::mock::MockEngine;
use furnace::nerve::effector::Delivery;
use furnace::nerve::prelude::*;
use furnace::nerve::receptor::{ChannelSender, UrgencyRule};
use furnace::nerve::{Feature, Outcome, Quiet};
use serde_json::json;

/// What the `smoke` receptor stamps on every signal
const DEADLINE: Duration = Duration::from_millis(5);

/// The stimuli, in order: which receptor, and what it saw
const SCRIPT: &[(&str, &str)] = &[
    ("logs", "disk 71% full on node-3"),
    ("logs", "disk 71% full on node-3"),
    ("logs", "login failed for admin from 10.0.0.7"),
    ("smoke", "smoke in rack 4"),
    ("smoke", "smoke in rack 4"),
    ("smoke", "smoke in rack 4"),
    ("logs", "deploy finished: api v2.3"),
    ("smoke", "81 °C in rack 2"),
    ("logs", "disk 71% full on node-3"),
];

/// Pause between stimuli, so the output can be read as it comes
const PACE: Duration = Duration::from_millis(400);

struct Demo {
    fabric: Fabric,
    logs: ChannelSender,
    smoke: ChannelSender,
    actuator: Receiver<Delivery>,
}

fn build() -> Result<Demo, BuildError> {
    let mut b = Fabric::builder();
    let (logs_in, logs) = ChannelIn::new("logs", 64);
    let (smoke_in, smoke) = ChannelIn::with_urgency("smoke", 64, UrgencyRule::within(DEADLINE));
    let logs_id = b.receptor(logs_in);
    let smoke_id = b.receptor(smoke_in);
    // A repeat is attenuated to half its intensity, below the threshold: it stops here
    let habituation = Mechanisms::NONE
        .threshold(0.6)
        .habituation(Habituation::new(Duration::from_secs(30)).gain(0.5));
    b.gate(logs_id, habituation).gate(smoke_id, habituation);

    // The junction passes every signal on; its pathways decide where
    let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| signal));
    let reflex = b.neuron(FnNeuron::reflex("reflex", |signal, _| {
        signal.command("actuator", json!({"do": "safe stop"}))
    }));
    // Stands in for an LLM: about 150 ms of thinking, then a command as JSON
    let engine = MockEngine::new(
        "mock-llm",
        &[
            r#"{"effector": "actuator", "#,
            r#""args": {"do": "open a ticket"}}"#,
        ],
    )
    .first_delay(Duration::from_millis(120))
    .piece_delay(Duration::from_millis(30));
    let brain = b.neuron(Llm::new("brain", engine));
    let (actuator, deliveries) = ChannelOut::new("actuator", 64);
    b.effector(actuator);

    b.reflex(logs_id, triage);
    b.reflex(smoke_id, triage);
    b.reflex(triage, reflex).weight(1.0);
    b.deliberative(triage, brain).weight(2.0);

    let fabric = b.build()?;
    fabric.start();
    Ok(Demo {
        fabric,
        logs,
        smoke,
        actuator: deliveries,
    })
}

/// ANSI colors when writing to a terminal that shows them, and `NO_COLOR` is not set
#[derive(Clone, Copy)]
struct Paint(bool);

impl Paint {
    fn detect() -> Self {
        let terminal = std::io::stdout().is_terminal()
            && std::env::var_os("NO_COLOR").is_none()
            // The old Windows console prints the escapes; Windows Terminal sets WT_SESSION
            && (!cfg!(windows) || std::env::var_os("WT_SESSION").is_some());
        Self(terminal)
    }

    fn paint(self, code: &str, text: &str) -> String {
        if self.0 {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    fn dim(self, text: &str) -> String {
        self.paint("2", text)
    }

    fn reflex(self, text: &str) -> String {
        self.paint("1;33", text)
    }

    fn brain(self, text: &str) -> String {
        self.paint("1;36", text)
    }
}

fn duration(d: Duration) -> String {
    if d < Duration::from_millis(1) {
        format!("{:.0} µs", d.as_secs_f64() * 1e6)
    } else if d.subsec_nanos().is_multiple_of(1_000_000) {
        format!("{} ms", d.as_millis())
    } else {
        format!("{:.1} ms", d.as_secs_f64() * 1e3)
    }
}

/// Milliseconds as the metrics report them, in the unit that reads best
fn ms(value: Option<f64>) -> String {
    value.map_or_else(
        || "-".to_string(),
        |ms| duration(Duration::from_secs_f64(ms / 1e3)),
    )
}

/// The brain pathway's measured p99, as the junction sees it when it chooses
fn brain_p99(fabric: &Fabric) -> Option<f64> {
    fabric
        .metrics()
        .pathways
        .into_iter()
        .find(|p| p.to == "brain")
        .and_then(|p| p.latency_ms.percentiles.p99)
}

/// Where the signal went, and why
fn route(fabric: &Fabric, outcome: &Outcome, deadline: bool, paint: Paint) -> String {
    match outcome.quiet {
        Some((_, Quiet::Habituation)) if outcome.is_quiet() => {
            paint.dim("quiet: a repeat, habituated at the receptor")
        }
        Some((_, reason)) => paint.dim(&format!("quiet: {reason:?}")),
        None if outcome.queued > 0 => format!(
            "{} {}",
            paint.brain("→ brain"),
            paint.dim("no deadline: the heaviest pathway")
        ),
        None if deadline => format!(
            "{} {}",
            paint.reflex("→ reflex"),
            paint.dim(&format!(
                "the brain's p99 ({}) does not fit in {}",
                ms(brain_p99(fabric)),
                duration(DEADLINE)
            ))
        ),
        None => paint.reflex("→ reflex"),
    }
}

fn summary(fabric: &Fabric, paint: Paint) {
    let metrics = fabric.metrics();
    println!(
        "\n{}",
        paint.dim("What the fabric measured (a server shows the same at GET /nerve):")
    );
    for p in &metrics.pathways {
        println!(
            "  {:<17} {:<13} {:>2} signals  p50 {:>9}  p99 {:>9}{}",
            format!("{} → {}", p.from, p.to),
            format!("{:?}", p.kind).to_lowercase(),
            p.signals,
            ms(p.latency_ms.percentiles.p50),
            ms(p.latency_ms.percentiles.p99),
            match p.ineligible {
                0 => String::new(),
                n => format!("  {n} too slow for a deadline"),
            },
        );
    }
    for arc in &metrics.arcs {
        println!(
            "  {:<31} p50 {:>9}  (stimulus to effector, inside the fabric)",
            format!("arc {} → {}", arc.from, arc.to),
            ms(arc.latency_ms.percentiles.p50),
        );
    }
    let habituated: u64 = metrics.neurons.iter().map(|n| n.quiet_by_habituation).sum();
    println!(
        "  {} signals in, {} quiet by habituation, {} reached the brain (brain fraction {:.2})",
        metrics.received, habituated, metrics.reached_brain, metrics.brain_fraction
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paint = Paint::detect();
    let demo = build()?;
    println!("furnace demo: a nervous system in one process");
    println!(
        "{}",
        paint.dim(&format!(
            "  receptors logs (no deadline) and smoke ({} deadline) → triage junction → \
             reflex (in place) or brain (mock LLM) → actuator\n  \
             the junction takes the heaviest pathway whose measured p99 fits the deadline\n",
            duration(DEADLINE)
        ))
    );

    for &(receptor, text) in SCRIPT {
        let sender = if receptor == "smoke" {
            &demo.smoke
        } else {
            &demo.logs
        };
        let signal = sender.signal(Feature::sketch(text.as_bytes()), Payload::text(text));
        let deadline = signal.urgency.deadline.is_some();
        let outcome = sender.emit(signal).ok_or("the fabric is not running")?;
        println!(
            "{:<6} {:<40} {}",
            receptor,
            format!("\"{text}\""),
            route(&demo.fabric, &outcome, deadline, paint)
        );
        if !outcome.is_quiet() {
            let delivery = demo.actuator.recv_timeout(Duration::from_secs(10))?;
            let by = if delivery.signal.trace.reached_brain() {
                paint.brain("by the brain,")
            } else {
                paint.reflex("by the reflex,")
            };
            println!(
                "       {:<40} {by} {} after the stimulus",
                format!(
                    "actuator: {}",
                    delivery.command.args["do"].as_str().unwrap_or("?")
                ),
                duration(delivery.signal.age()),
            );
        }
        std::thread::sleep(PACE);
    }
    summary(&demo.fabric, paint);
    println!(
        "\n{}",
        paint.dim(
            "Your own receptors (HTTP, SSE, lines, timers), models and effectors (webhooks, \
             commands, files): https://github.com/Gilfeather/furnace/blob/main/docs/NERVE.md"
        )
    );
    Ok(())
}
