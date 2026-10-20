//! Consolidation on real logs: Loghub's HDFS and BGL samples (2,000 lines each) replayed
//! through a `Lines` receptor at a fixed rate into a log-triage brain, once with
//! consolidation off and once on.
//!
//! - A reflex makes each line's feature its *template*: the line with every token that
//!   holds a digit masked (dates, times, block ids, addresses, sizes).
//! - The brain is a mock LLM with a realistic latency (a generation of a summed window
//!   takes `BRAIN_BASE` plus `BRAIN_PER_ITEM` per further line). It decides on the raw
//!   line: `page` for errors and failures, `ticket` for warnings and for HDFS blocks short of
//!   the 64 MB block size, `ignore` for the rest. It also writes a `reason` with the line's numbers in
//!   it, a volatile field. The block size is masked out of the template, so that template's
//!   engram meets disagreement: the case consolidation must not get wrong.
//! - Effectors: `page`, `ticket`, `ignore`, in-process channels.
//!
//! Reported per log and mode: brain calls in all and per 200 lines, line-to-action p50 and
//! p99, how many actions equal what the brain decides for that very line, and the shadow
//! checks' agreement.
//!
//! The logs are not in the repository. From <https://github.com/logpai/loghub>:
//! `HDFS/HDFS_2k.log` and `BGL/BGL_2k.log`, into one directory passed as `--logs DIR`.

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use furnace::nerve::consolidation::{Consolidation, ConsolidationMetrics, EngramStore};
use furnace::nerve::effector::{Channel as ChannelEffector, Delivery, Effector};
use furnace::nerve::receptor::Lines;
use furnace::nerve::{
    Cx, Fabric, Feature, Firing, FnNeuron, Mechanisms, Neuron, NeuronKind, Signal, Summation,
};
use furnace_bench_common::{fail, free_port, percentile};
use serde::Serialize;

/// One generation of a window of lines
const BRAIN_BASE: Duration = Duration::from_millis(400);
/// Each further line in the same window
const BRAIN_PER_ITEM: Duration = Duration::from_millis(20);
/// Generations at once
const BRAIN_WORKERS: usize = 4;
/// Lines summed into one generation at most
const BRAIN_BATCH: usize = 8;
/// Lines per second sent to the receptor
const RATE: f64 = 50.0;
/// Lines per bucket of the brain-calls timeline
const BUCKET: u64 = 200;
const TEMPLATE_KIND: u16 = 0x10;
const ACTIONS: [&str; 3] = ["page", "ticket", "ignore"];

/// What the brain decides for a raw line, and why
fn triage(line: &str) -> (&'static str, String) {
    let lower = line.to_ascii_lowercase();
    if ["fatal", "error", "exception", "failed", "failure", "severe"]
        .iter()
        .any(|w| lower.contains(w))
    {
        return ("page", format!("failure in: {line}"));
    }
    if lower.contains("warn") {
        return ("ticket", format!("warning in: {line}"));
    }
    // HDFS: a block stored short of the 64 MB block size (a partial block)
    if let Some(size) = line
        .split_whitespace()
        .skip_while(|t| *t != "size")
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        && size < 64 << 20
    {
        return ("ticket", format!("partial block ({size} bytes) in: {line}"));
    }
    ("ignore", format!("routine: {line}"))
}

/// The line with every token holding a digit masked
fn template(line: &str) -> String {
    line.split_whitespace()
        .map(|t| {
            if t.bytes().any(|b| b.is_ascii_digit()) {
                "<*>"
            } else {
                t
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The mock LLM: decides per line, sleeps for a generation per summed window, counts calls
struct MockLlm {
    calls: Arc<Mutex<Vec<u64>>>,
}

impl Neuron for MockLlm {
    fn name(&self) -> &str {
        "brain"
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Brain
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        self.stimulate_many(vec![signal], cx)
            .pop()
            .unwrap_or(Firing::Quiet)
    }

    fn stimulate_many(&self, signals: Vec<Signal>, _cx: &Cx<'_>) -> Vec<Firing> {
        std::thread::sleep(BRAIN_BASE + BRAIN_PER_ITEM * (signals.len() as u32 - 1));
        self.calls
            .lock()
            .unwrap()
            .extend(signals.iter().map(|s| s.id.0));
        signals
            .into_iter()
            .map(|s| {
                let line = s.payload.as_text().unwrap_or_default();
                let (action, reason) = triage(line);
                Firing::one(s.command(action, serde_json::json!({"reason": reason})))
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Run {
    pub consolidation: bool,
    pub lines: u64,
    pub actions: u64,
    /// Lines the brain was called for, shadow checks included
    pub brain_calls: u64,
    /// Brain calls per bucket of 200 lines, in order of arrival
    pub brain_calls_per_200: Vec<u64>,
    pub line_to_action_p50_ms: f64,
    pub line_to_action_p99_ms: f64,
    /// Of actions taken by the reflex (not the brain), p50 and p99
    pub reflex_p50_ms: Option<f64>,
    pub reflex_p99_ms: Option<f64>,
    /// Actions equal to the brain's decision for that very line
    pub same_as_brain: f64,
    pub engrams: Option<ConsolidationMetrics>,
    /// agreements / (agreements + disagreements)
    pub shadow_agreement: Option<f64>,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LogResult {
    pub log: String,
    pub lines: usize,
    pub templates: usize,
    pub off: Run,
    pub on: Run,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub rate_lines_per_s: f64,
    pub brain_base_ms: u64,
    pub brain_per_item_ms: u64,
    pub brain_workers: usize,
    pub brain_batch: usize,
    pub logs: Vec<LogResult>,
}

/// The log's lines as the brain should see them: BGL's first column is a label added by the
/// dataset's authors (the answer), not part of the log, and is dropped
fn read_log(path: &std::path::Path) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| fail(&format!("cannot read {}: {e}", path.display())));
    let labelled = path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with("BGL"));
    text.lines()
        .map(|l| {
            let l = l.trim_end();
            if labelled {
                l.split_once(' ').map_or(l, |(_, rest)| rest).to_string()
            } else {
                l.to_string()
            }
        })
        .filter(|l| !l.is_empty())
        .collect()
}

fn run(lines: &[String], consolidate: bool) -> Run {
    let engrams = consolidate.then(|| {
        EngramStore::new(
            Consolidation::new()
                .effectors(ACTIONS)
                .after(3)
                .shadow(0.05)
                .first_shadows(3)
                .volatile(["reason"]),
        )
    });
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut b = Fabric::builder();
    let logs = b.receptor(Lines::listen("logs", &addr), Mechanisms::NONE);
    // Which line each signal was, for comparing actions with the brain's decision
    let seen: Arc<Mutex<HashMap<u64, Arc<str>>>> = Arc::default();
    let shape = {
        let seen = seen.clone();
        FnNeuron::reflex("template", move |signal: Signal, _: &Cx<'_>| {
            let line = signal.payload.as_text().unwrap_or_default();
            let feature = Feature::sketch_as(TEMPLATE_KIND, template(line).as_bytes());
            seen.lock().unwrap().insert(signal.id.0, Arc::from(line));
            Firing::one(signal.with_feature(feature))
        })
    };
    let shape = b.neuron(shape, Mechanisms::NONE);
    b.reflex(logs, shape, 1.0);
    let (sender, deliveries) = mpsc::channel::<(Delivery, Instant)>();
    for action in ACTIONS {
        let (act, receiver) = ChannelEffector::new(action, 4096);
        let effector = match &engrams {
            Some(engrams) => Effector::new(engrams.watch(act)),
            None => Effector::new(act),
        };
        b.neuron(effector, Mechanisms::NONE);
        let sender = sender.clone();
        std::thread::spawn(move || {
            while let Ok(d) = receiver.recv() {
                if sender.send((d, Instant::now())).is_err() {
                    return;
                }
            }
        });
    }
    drop(sender);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let brain = MockLlm {
        calls: calls.clone(),
    };
    let summation = Summation::default()
        .max_batch(BRAIN_BATCH)
        .workers(BRAIN_WORKERS)
        .capacity(4096);
    match &engrams {
        Some(engrams) => {
            let (recall, _) = b.consolidate(brain, Mechanisms::NONE, summation, engrams);
            b.reflex(shape, recall, 1.0);
        }
        None => {
            let brain = b.neuron(brain, Mechanisms::NONE);
            b.deliberative(shape, brain, 1.0, summation);
        }
    }
    let fabric = b.build();
    fabric.start();

    let mut stream = (0..200)
        .find_map(|_| {
            std::net::TcpStream::connect(&addr).ok().or_else(|| {
                std::thread::sleep(Duration::from_millis(10));
                None
            })
        })
        .unwrap_or_else(|| fail("the Lines receptor does not listen"));
    let _ = stream.set_nodelay(true);
    let started = Instant::now();
    let period = Duration::from_secs_f64(1.0 / RATE);
    for (i, line) in lines.iter().enumerate() {
        let due = started + period * i as u32;
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
        writeln!(stream, "{line}").unwrap_or_else(|e| fail(&format!("send: {e}")));
    }
    drop(stream);

    let mut received = Vec::with_capacity(lines.len());
    let deadline = Instant::now() + Duration::from_secs(60);
    while received.len() < lines.len() {
        let left = deadline.saturating_duration_since(Instant::now());
        match deliveries.recv_timeout(left) {
            Ok(d) => received.push(d),
            Err(_) => break,
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    // Shadow checks still on their way back
    if let Some(engrams) = &engrams {
        let wait = Instant::now() + Duration::from_secs(10);
        while fabric.waiting() > 0 && Instant::now() < wait {
            std::thread::sleep(Duration::from_millis(5));
        }
        let m = engrams.snapshot();
        let checked = Instant::now() + Duration::from_secs(5);
        while engrams.snapshot().agreements + engrams.snapshot().disagreements < m.shadows
            && Instant::now() < checked
        {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    let seen = seen.lock().unwrap();
    let first = seen.keys().min().copied().unwrap_or(0);
    let bucket = |id: u64| ((id - first) / BUCKET) as usize;
    let buckets = lines.len().div_ceil(BUCKET as usize);
    let mut per_bucket = vec![0u64; buckets];
    let calls = calls.lock().unwrap();
    for &id in calls.iter() {
        if let Some(b) = per_bucket.get_mut(bucket(id)) {
            *b += 1;
        }
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut all: Vec<Duration> = Vec::new();
    let mut reflex: Vec<Duration> = Vec::new();
    let mut same = 0u64;
    for (d, at) in &received {
        let latency = at.saturating_duration_since(d.signal.at);
        all.push(latency);
        if !d.signal.trace.reached_brain() {
            reflex.push(latency);
        }
        if let Some(line) = seen.get(&d.signal.id.0)
            && triage(line).0 == d.command.effector
        {
            same += 1;
        }
    }
    all.sort();
    reflex.sort();
    let engram_metrics = engrams.as_ref().map(EngramStore::snapshot);
    Run {
        consolidation: consolidate,
        lines: lines.len() as u64,
        actions: received.len() as u64,
        brain_calls: calls.len() as u64,
        brain_calls_per_200: per_bucket,
        line_to_action_p50_ms: all.first().map_or(0.0, |_| ms(percentile(&all, 0.5))),
        line_to_action_p99_ms: all.first().map_or(0.0, |_| ms(percentile(&all, 0.99))),
        reflex_p50_ms: reflex.first().map(|_| ms(percentile(&reflex, 0.5))),
        reflex_p99_ms: reflex.first().map(|_| ms(percentile(&reflex, 0.99))),
        same_as_brain: same as f64 / received.len().max(1) as f64,
        shadow_agreement: engram_metrics
            .as_ref()
            .map(|m| m.agreements as f64 / (m.agreements + m.disagreements).max(1) as f64),
        engrams: engram_metrics,
        seconds,
    }
}

pub fn bench(dir: &str) -> Report {
    let mut logs = Vec::new();
    for name in ["HDFS_2k.log", "BGL_2k.log"] {
        let path = std::path::Path::new(dir).join(name);
        let lines = read_log(&path);
        let templates: BTreeSet<String> = lines.iter().map(|l| template(l)).collect();
        eprintln!(
            "{name}: {} lines, {} templates",
            lines.len(),
            templates.len()
        );
        let off = run(&lines, false);
        eprintln!("  off: {} brain calls", off.brain_calls);
        let on = run(&lines, true);
        eprintln!("  on:  {} brain calls", on.brain_calls);
        logs.push(LogResult {
            log: name.to_string(),
            lines: lines.len(),
            templates: templates.len(),
            off,
            on,
        });
    }
    Report {
        rate_lines_per_s: RATE,
        brain_base_ms: BRAIN_BASE.as_millis() as u64,
        brain_per_item_ms: BRAIN_PER_ITEM.as_millis() as u64,
        brain_workers: BRAIN_WORKERS,
        brain_batch: BRAIN_BATCH,
        logs,
    }
}

pub fn print(report: &Report) {
    println!(
        "consolidation: {} lines/s, brain {} ms + {} ms per further line, {} workers, batch {}",
        report.rate_lines_per_s,
        report.brain_base_ms,
        report.brain_per_item_ms,
        report.brain_workers,
        report.brain_batch
    );
    for log in &report.logs {
        println!(
            "\n{} ({} lines, {} templates)",
            log.log, log.lines, log.templates
        );
        println!(
            "  {:<4} {:>7} {:>11} {:>9} {:>9} {:>10} {:>9}",
            "", "actions", "brain calls", "p50 ms", "p99 ms", "same as br", "shadow ok"
        );
        for (mode, r) in [("off", &log.off), ("on", &log.on)] {
            println!(
                "  {:<4} {:>7} {:>11} {:>9.1} {:>9.1} {:>9.1}% {:>9}",
                mode,
                r.actions,
                r.brain_calls,
                r.line_to_action_p50_ms,
                r.line_to_action_p99_ms,
                r.same_as_brain * 100.0,
                r.shadow_agreement
                    .map_or("-".to_string(), |a| format!("{:.1}%", a * 100.0))
            );
            println!(
                "       brain calls per 200 lines: {:?}",
                r.brain_calls_per_200
            );
        }
        if let Some(m) = &log.on.engrams {
            println!(
                "  engrams: {} consolidated, {} dissolved, {} hits, {} shadows ({} agree, {} disagree), reflex p50 {:?} ms, brain time saved {:.0} s",
                m.consolidated,
                m.dissolved,
                m.hits,
                m.shadows,
                m.agreements,
                m.disagreements,
                log.on.reflex_p50_ms,
                m.brain_time_saved_ms / 1e3
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_mask_numbers_and_ids() {
        assert_eq!(
            template(
                "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-6952295868487656571 terminating"
            ),
            "<*> <*> <*> INFO dfs.DataNode$PacketResponder: PacketResponder <*> for block <*> terminating"
        );
        assert_eq!(triage("x WARN y").0, "ticket");
        assert_eq!(triage("a is added to blk_1 size 67108864").0, "ignore");
        assert_eq!(triage("a is added to blk_1 size 1024").0, "ticket");
        assert_eq!(triage("ciod: Error loading").0, "page");
    }
}
