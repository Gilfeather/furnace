//! Expectations on Loghub's BGL log (4,747,963 lines over 215 days, each labelled alert or
//! not), against a brain that sees every line and against habituation alone.
//!
//! The brain is a mock LLM that pages on a line whose severity is `FATAL` or `FAILURE`, the
//! only severities BGL's alert labels fall on, and otherwise does nothing. With expectations
//! it also writes rules: a `match` for each template it has judged (acting with `page` for an
//! alerting template, absorbing a quiet one) and an `every` for a heartbeat. A synthetic
//! heartbeat from one service, every minute of log time, drops out 20 times for 15 minutes.
//!
//! Two parts:
//!
//! - **Replay**: the whole log in log time, the three setups side by side, the brain answering
//!   at once (its latency is the second part's subject). Brain calls, line-level precision
//!   and recall against the labels, the same per alert episode (alert lines less than a minute
//!   apart), and the dropouts found.
//! - **Live**: 3,000 consecutive lines (20% to 60% alerts) through a `Lines` receptor at 50
//!   lines/s, the brain a
//!   mock with an LLM's latency, then (with `--gguf`) a real model through candle: time from
//!   line to page.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use furnace::nerve::effector::{ChannelOut, Delivery, Effector};
use furnace::nerve::expect::fields::log_line;
use furnace::nerve::expect::{ExpectConfig, ExpectMetrics, Expectations};
use furnace::nerve::receptor::Lines;
use furnace::nerve::{
    Command, Cx, Fabric, Feature, Firing, FnNeuron, Habituation, Mechanisms, Neuron, NeuronKind,
    Payload, Signal, Summation,
};
use furnace_bench_common::{fail, free_port, percentile};
use serde::Serialize;
use serde_json::Value;

/// The synthetic heartbeat: one a minute, 20 dropouts of 15 minutes
const BEAT: u64 = 60;
const DROPOUTS: u64 = 20;
const DROPOUT: u64 = 15 * 60;
/// Alert lines less than this apart are one episode, and so are pages
const EPISODE: u64 = 60;
/// Habituation as the other benches set it
const HABITUATION_TAU: Duration = Duration::from_secs(30);
const THRESHOLD: f32 = 0.6;
/// The live part
const LIVE_LINES: usize = 3000;
const LIVE_RATE: f64 = 50.0;
const BRAIN_BASE: Duration = Duration::from_millis(400);
const BRAIN_PER_ITEM: Duration = Duration::from_millis(20);
const BRAIN_WORKERS: usize = 4;
const BRAIN_BATCH: usize = 8;

/// What the mock brain decides for a line: page on what BGL's operators label (FATAL and
/// FAILURE are the only severities alerts carry)
fn pages(text: &str) -> bool {
    text.split_whitespace()
        .any(|t| t == "FATAL" || t == "FAILURE")
}

fn is_heartbeat(text: &str) -> bool {
    text.starts_with("heartbeat from")
}

/// The mock brain with expectations: what it commands for a surprise
fn decide(surprise: &Value) -> Vec<Command> {
    let observation = &surprise["observation"];
    let template = observation["template"].as_str().unwrap_or_default();
    let text = observation["text"].as_str().unwrap_or_default();
    let id = format!("t{:016x}", furnace::nerve::sketch64(template.as_bytes()));
    let page = || Command::new("page", serde_json::json!({}));
    let add = |e: Value| Command::new("expect", serde_json::json!({ "add": e }));
    match surprise["reason"].as_str().unwrap_or_default() {
        "unexpected" if is_heartbeat(text) => vec![add(serde_json::json!({
            "id": "heartbeat", "kind": "every", "period_ms": BEAT * 1000,
            "tolerance_ms": BEAT * 500, "selector": {"template": template},
            "provenance": "a service's heartbeat"
        }))],
        "unexpected" if pages(text) => vec![
            page(),
            add(serde_json::json!({
                "id": id, "kind": "match", "selector": {"template": template},
                "on_match": {"act": [{"effector": "page", "args": {}}]},
                "provenance": "an alerting template"
            })),
        ],
        "unexpected" => vec![add(serde_json::json!({
            "id": id, "kind": "match", "selector": {"template": template},
            "provenance": "a routine template"
        }))],
        "violation" => vec![page()],
        "absence" => vec![Command::new("page", serde_json::json!({"absence": true}))],
        "shadow" => {
            // Would it have paged where the rule only absorbed?
            let acts = surprise["expectations"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|e| e["on_match"].get("act").is_some());
            if pages(text) && !acts {
                let rule = surprise["absorbed_by"].clone();
                vec![Command::new(
                    "expect",
                    serde_json::json!({"verdict": {"id": rule, "correct": false}}),
                )]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// One line of the log, or a synthetic heartbeat
struct Line {
    /// Seconds since the epoch, from the log
    at: u64,
    /// The line without its label column
    text: String,
    alert: bool,
}

/// The log's lines with the synthetic heartbeats merged in by time
struct Stream<R> {
    lines: std::io::Lines<R>,
    next_beat: u64,
    end: u64,
    dropouts: Vec<(u64, u64)>,
    pending: Option<Line>,
    last: u64,
}

fn parse(line: &str) -> Option<Line> {
    let (label, text) = line.split_once(' ')?;
    let at = text.split(' ').next()?.parse().ok()?;
    Some(Line {
        at,
        text: text.trim_end().to_string(),
        alert: label != "-",
    })
}

/// The span of the log, to place the dropouts
fn span(path: &Path) -> (u64, u64, usize) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| fail(&format!("{e}")));
    let (mut first, mut last, mut n) = (u64::MAX, 0, 0);
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        if let Some(l) = parse(&line) {
            first = first.min(l.at);
            last = last.max(l.at);
            n += 1;
        }
    }
    (first, last, n)
}

fn dropouts(first: u64, last: u64) -> Vec<(u64, u64)> {
    let step = (last - first) / (DROPOUTS + 1);
    (1..=DROPOUTS)
        .map(|k| {
            let start = first + k * step;
            (start, start + DROPOUT)
        })
        .collect()
}

impl<R: BufRead> Iterator for Stream<R> {
    type Item = Line;

    fn next(&mut self) -> Option<Line> {
        if self.pending.is_none() {
            self.pending = self
                .lines
                .by_ref()
                .map_while(Result::ok)
                .find_map(|l| parse(&l));
        }
        // Log time never runs backwards here: a line a little out of order keeps the last time
        let log_at = self.pending.as_ref().map(|l| l.at.max(self.last));
        if self.next_beat <= self.end && log_at.is_none_or(|at| self.next_beat <= at) {
            let at = self.next_beat;
            self.next_beat += BEAT;
            let silent = self.dropouts.iter().any(|&(s, e)| at >= s && at < e);
            if !silent {
                self.last = self.last.max(at);
                return Some(Line {
                    at,
                    text: "heartbeat from svc-a".to_string(),
                    alert: false,
                });
            }
            return self.next();
        }
        let mut line = self.pending.take()?;
        line.at = line.at.max(self.last);
        self.last = line.at;
        Some(line)
    }
}

fn stream(path: &Path, first: u64, last: u64) -> Stream<std::io::BufReader<std::fs::File>> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| fail(&format!("{e}")));
    Stream {
        lines: std::io::BufReader::with_capacity(1 << 20, file).lines(),
        next_beat: first,
        end: last,
        dropouts: dropouts(first, last),
        pending: None,
        last: first,
    }
}

/// Furnace's habituation table (src/nerve/neuron.rs), here with log time: a bounded,
/// 4-way set-associative table of attenuations that decay as `exp(-dt / tau)`
struct Habituate {
    slots: Vec<(u64, f32, u64)>,
    sets: usize,
    tau: f32,
    gain: f32,
}

impl Habituate {
    fn new(config: Habituation) -> Self {
        let sets = config.capacity.div_ceil(4);
        Self {
            slots: vec![(0, 0.0, u64::MAX); sets * 4],
            sets,
            tau: config.tau.as_secs_f32(),
            gain: config.gain,
        }
    }

    /// Whether a signal of `key` at `at` (seconds) passes a gate of [`THRESHOLD`]
    fn admit(&mut self, key: u64, at: u64) -> bool {
        let set = (key as usize) % self.sets;
        let slots = &mut self.slots[set * 4..set * 4 + 4];
        let mut victim = 0;
        let mut oldest = u64::MAX;
        for (i, slot) in slots.iter_mut().enumerate() {
            if slot.2 != u64::MAX && slot.0 == key {
                let decayed = slot.1 * (-(at.saturating_sub(slot.2) as f32) / self.tau).exp();
                slot.1 = decayed + (1.0 - decayed) * self.gain;
                slot.2 = at;
                return 1.0 - decayed >= THRESHOLD;
            }
            let age = if slot.2 == u64::MAX { 0 } else { slot.2 + 1 };
            if age < oldest {
                oldest = age;
                victim = i;
            }
        }
        slots[victim] = (key, self.gain, at);
        true
    }
}

/// What a setup did, counted the same way for all three
#[derive(Clone, Debug, Default, Serialize)]
pub struct Tally {
    pub lines: u64,
    pub heartbeats: u64,
    pub brain_calls: u64,
    /// Brain calls per tenth of the stream, in order
    pub brain_calls_by_tenth: Vec<u64>,
    pub true_pages: u64,
    pub false_pages: u64,
    pub missed: u64,
    pub precision: f64,
    pub recall: f64,
    pub alert_episodes: u64,
    pub alert_episodes_paged: u64,
    pub page_episodes: u64,
    pub page_episodes_true: u64,
    pub episode_precision: f64,
    pub episode_recall: f64,
    pub dropouts: u64,
    pub dropouts_found: u64,
    pub false_absences: u64,
    /// Seconds from the first missing beat to the absence, over the dropouts found
    pub absence_delay_s: Option<f64>,
    pub seconds: f64,
    pub expectations: Option<ExpectMetrics>,
}

#[derive(Default)]
struct Counter {
    t: Tally,
    tenth: u64,
    last_alert: Option<u64>,
    alert_episode_paged: bool,
    last_page: Option<u64>,
    page_episode_true: bool,
    delays: Vec<f64>,
    found: Vec<bool>,
}

impl Counter {
    fn line(&mut self, line: &Line, paged: bool, brain_calls: u64) {
        let t = &mut self.t;
        if is_heartbeat(&line.text) {
            t.heartbeats += 1;
        } else {
            t.lines += 1;
        }
        let bucket = ((t.lines + t.heartbeats) / self.tenth.max(1)).min(9) as usize;
        if t.brain_calls_by_tenth.len() <= bucket {
            t.brain_calls_by_tenth.resize(bucket + 1, 0);
        }
        t.brain_calls_by_tenth[bucket] += brain_calls;
        t.brain_calls += brain_calls;
        match (line.alert, paged) {
            (true, true) => t.true_pages += 1,
            (false, true) => t.false_pages += 1,
            (true, false) => t.missed += 1,
            (false, false) => {}
        }
        if line.alert {
            if self.last_alert.is_none_or(|a| line.at > a + EPISODE) {
                if self.last_alert.is_some() {
                    t.alert_episodes_paged += u64::from(self.alert_episode_paged);
                }
                t.alert_episodes += 1;
                self.alert_episode_paged = false;
            }
            self.alert_episode_paged |= paged;
            self.last_alert = Some(line.at);
        }
        if paged {
            if self.last_page.is_none_or(|p| line.at > p + EPISODE) {
                if self.last_page.is_some() {
                    t.page_episodes_true += u64::from(self.page_episode_true);
                }
                t.page_episodes += 1;
                self.page_episode_true = false;
            }
            self.page_episode_true |= line.alert;
            self.last_page = Some(line.at);
        }
    }

    /// An absence at `at`, against the dropouts
    fn absence(&mut self, at: u64, dropouts: &[(u64, u64)]) {
        self.t.brain_calls += 1;
        match dropouts
            .iter()
            .position(|&(s, e)| at >= s && at <= e + 2 * BEAT)
        {
            Some(i) if !self.found[i] => {
                self.found[i] = true;
                self.delays.push(at.saturating_sub(dropouts[i].0) as f64);
            }
            Some(_) => {}
            None => self.t.false_absences += 1,
        }
    }

    fn finish(mut self, started: Instant) -> Tally {
        let t = &mut self.t;
        t.alert_episodes_paged += u64::from(self.alert_episode_paged && self.last_alert.is_some());
        t.page_episodes_true += u64::from(self.page_episode_true && self.last_page.is_some());
        let ratio = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        t.precision = ratio(t.true_pages, t.true_pages + t.false_pages);
        t.recall = ratio(t.true_pages, t.true_pages + t.missed);
        t.episode_precision = ratio(t.page_episodes_true, t.page_episodes);
        t.episode_recall = ratio(t.alert_episodes_paged, t.alert_episodes);
        t.dropouts = DROPOUTS;
        t.dropouts_found = self.found.iter().filter(|f| **f).count() as u64;
        t.absence_delay_s = (!self.delays.is_empty())
            .then(|| self.delays.iter().sum::<f64>() / self.delays.len() as f64);
        t.seconds = started.elapsed().as_secs_f64();
        self.t
    }
}

fn counter(total: usize) -> Counter {
    Counter {
        tenth: (total as u64).div_ceil(10),
        found: vec![false; DROPOUTS as usize],
        ..Counter::default()
    }
}

/// Every line to the brain
fn replay_brain(path: &Path, first: u64, last: u64, total: usize) -> Tally {
    let started = Instant::now();
    let mut c = counter(total);
    for line in stream(path, first, last) {
        let paged = pages(&line.text);
        c.line(&line, paged, 1);
    }
    c.finish(started)
}

/// Habituation in front of the brain: a repeated template attenuates and stops reaching it
fn replay_habituation(path: &Path, first: u64, last: u64, total: usize) -> Tally {
    let started = Instant::now();
    let mut c = counter(total);
    let mut gate = Habituate::new(Habituation::new(HABITUATION_TAU));
    for line in stream(path, first, last) {
        let key = log_line(&line.text).template.unwrap_or_default();
        let reaches = gate.admit(key, line.at);
        c.line(&line, reaches && pages(&line.text), u64::from(reaches));
    }
    c.finish(started)
}

fn expect_config() -> ExpectConfig {
    ExpectConfig::new()
        .effectors(["page"])
        .max(8192)
        .writes_per_minute(600)
        .ttl(Duration::from_secs(24 * 3600))
}

/// Expectations in front of the brain, in log time
fn replay_expectations(path: &Path, first: u64, last: u64, total: usize) -> Tally {
    let started = Instant::now();
    let mut c = counter(total);
    let expectations = Expectations::new("bgl-expect", expect_config());
    let expect = expectations.effector("expect");
    let epoch = expectations.epoch();
    let at = |s: u64| epoch + Duration::from_secs(s - first);
    let drops = dropouts(first, last);
    // The brain's answer to one surprise at `now`: true if it paged
    let answer = |surprise_signal: &Signal, now: Instant| -> bool {
        let surprise = surprise_signal
            .payload
            .to_json()
            .and_then(|v| v.get("surprise").cloned())
            .unwrap_or_default();
        let mut paged = false;
        for command in decide(&surprise) {
            if command.effector == "expect" {
                let _ = expect.act_at(&command, surprise_signal, now);
            } else {
                paged = true;
            }
        }
        paged
    };
    for line in stream(path, first, last) {
        let now = at(line.at);
        for signal in expectations.due(now) {
            if signal.payload.as_command().is_none() && answer(&signal, now) {
                c.absence(line.at, &drops);
            }
        }
        let signal = Signal::new(Feature::NONE, Payload::text(&line.text));
        let mut paged = false;
        let mut calls = 0;
        for fired in expectations.react(signal, now) {
            match fired.payload.as_command() {
                Some(command) => paged |= command.effector == "page",
                None => {
                    calls += 1;
                    paged |= answer(&fired, now);
                }
            }
        }
        c.line(&line, paged, calls);
    }
    let mut tally = c.finish(started);
    tally.expectations = Some(expectations.snapshot());
    tally
}

// ---------------------------------------------------------------------------------------------
// Live: time from line to page

/// The mock LLM: decides per line (or per surprise), a generation per summed window
struct MockLlm {
    expectations: bool,
    calls: Arc<Mutex<u64>>,
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
        *self.calls.lock().unwrap() += signals.len() as u64;
        signals
            .into_iter()
            .map(|s| {
                let commands = if self.expectations {
                    let surprise = s
                        .payload
                        .to_json()
                        .and_then(|v| v.get("surprise").cloned())
                        .unwrap_or_default();
                    decide(&surprise)
                } else if pages(s.payload.as_text().unwrap_or_default()) {
                    vec![Command::new("page", serde_json::json!({}))]
                } else {
                    Vec::new()
                };
                Firing::Fire(
                    commands
                        .into_iter()
                        .map(|c| s.derive(Payload::command(c)))
                        .collect(),
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Live {
    pub setup: String,
    pub brain: String,
    pub lines: u64,
    pub alerts: u64,
    pub brain_calls: u64,
    pub pages: u64,
    pub true_pages: u64,
    pub precision: f64,
    pub recall: f64,
    /// Line to page, over the pages
    pub page_p50_ms: Option<f64>,
    pub page_p99_ms: Option<f64>,
    /// Pages a rule fired without the brain, and their p50
    pub reflex_pages: u64,
    pub reflex_p50_ms: Option<f64>,
    pub dropped: u64,
    pub expectations: Option<ExpectMetrics>,
    /// The brain neuron's own counters (an LLM's generations and commands)
    pub brain_metrics: Value,
    pub seconds: f64,
}

/// The live slice: the first `LIVE_LINES` consecutive lines in the second half of the log
/// (from a multiple of 500) of which 20% to 60% are alerts, so that it has alerts and quiet
fn slice(path: &Path, total: usize) -> Vec<Line> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| fail(&format!("{e}")));
    let lines: Vec<Line> = std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .skip(total / 2)
        .filter_map(|l| parse(&l))
        .collect();
    let alerts: Vec<u32> = std::iter::once(0)
        .chain(lines.iter().scan(0u32, |n, l| {
            *n += u32::from(l.alert);
            Some(*n)
        }))
        .collect();
    let start = (0..lines.len().saturating_sub(LIVE_LINES))
        .step_by(500)
        .find(|&s| {
            let share = f64::from(alerts[s + LIVE_LINES] - alerts[s]) / LIVE_LINES as f64;
            (0.2..=0.6).contains(&share)
        })
        .unwrap_or(0);
    lines.into_iter().skip(start).take(LIVE_LINES).collect()
}

type BrainMaker = Box<dyn FnOnce(Arc<Mutex<u64>>) -> Arc<dyn Neuron>>;

fn live(lines: &[Line], setup: &str, brain_name: &str, make: BrainMaker) -> Live {
    let calls = Arc::new(Mutex::new(0u64));
    let mut b = Fabric::builder();
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let logs = b.receptor(Lines::listen("logs", &addr), Mechanisms::NONE);
    // Which line each signal is, for the labels
    let index: Arc<Mutex<HashMap<u64, usize>>> = Arc::default();
    let counter = Arc::new(Mutex::new(0usize));
    let tag = {
        let (index, counter) = (index.clone(), counter.clone());
        FnNeuron::reflex("tag", move |s: Signal, _: &Cx<'_>| {
            let mut n = counter.lock().unwrap();
            index.lock().unwrap().insert(s.id.0, *n);
            *n += 1;
            let key = log_line(s.payload.as_text().unwrap_or_default())
                .template
                .unwrap_or_default();
            Firing::one(s.with_feature(Feature::new(0x10, key)))
        })
    };
    let tag = b.neuron(tag, Mechanisms::NONE);
    b.reflex(logs, tag, 1.0);
    let (page_act, deliveries) = ChannelOut::new("page", 1 << 16);
    b.neuron(Effector::new(page_act), Mechanisms::NONE);
    let summation = Summation::default()
        .max_batch(BRAIN_BATCH)
        .workers(BRAIN_WORKERS)
        .capacity(1 << 14);
    let brain = b.add(make(calls.clone()), Mechanisms::NONE);
    let expectations = match setup {
        "brain" => {
            b.deliberative(tag, brain, 1.0, summation);
            None
        }
        "habituation" => {
            let gate = b.neuron(
                FnNeuron::reflex("habituate", |s, _| Firing::one(s)),
                Mechanisms::NONE
                    .threshold(THRESHOLD)
                    .habituation(Habituation::new(HABITUATION_TAU)),
            );
            b.reflex(tag, gate, 1.0);
            b.deliberative(gate, brain, 1.0, summation);
            None
        }
        _ => {
            let expectations = Expectations::new("bgl-expect", expect_config());
            let e = b.receptor(expectations.clone(), Mechanisms::NONE);
            b.reflex(tag, e, 1.0);
            b.neuron(
                Effector::new(expectations.effector("expect")),
                Mechanisms::NONE,
            );
            b.deliberative(e, brain, 1.0, summation);
            Some(expectations)
        }
    };
    let fabric = b.build().expect("the fabric");
    fabric.start();
    let (sender, received) = mpsc::channel::<(Delivery, Instant)>();
    std::thread::spawn(move || {
        while let Ok(d) = deliveries.recv() {
            if sender.send((d, Instant::now())).is_err() {
                return;
            }
        }
    });
    let mut stream = (0..200)
        .find_map(|_| {
            std::net::TcpStream::connect(&addr).ok().or_else(|| {
                std::thread::sleep(Duration::from_millis(10));
                None
            })
        })
        .unwrap_or_else(|| fail("the Lines receptor does not listen"));
    let started = Instant::now();
    let period = Duration::from_secs_f64(1.0 / LIVE_RATE);
    for (i, line) in lines.iter().enumerate() {
        let due = started + period * i as u32;
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
        writeln!(stream, "{}", line.text).unwrap_or_else(|e| fail(&format!("send: {e}")));
    }
    drop(stream);
    // Let the brain's queue drain
    let deadline = Instant::now() + Duration::from_secs(600);
    while (fabric.waiting() > 0 || *counter.lock().unwrap() < lines.len())
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(500));
    let seconds = started.elapsed().as_secs_f64();
    let index = index.lock().unwrap();
    let mut paged = vec![false; lines.len()];
    let (mut all, mut reflex) = (Vec::new(), Vec::new());
    while let Ok((d, at)) = received.try_recv() {
        let Some(&i) = index.get(&d.signal.id.0) else {
            continue;
        };
        if paged[i] {
            continue;
        }
        paged[i] = true;
        let latency = at.saturating_duration_since(d.signal.at).as_secs_f64() * 1e3;
        all.push(latency);
        if !d.signal.trace.reached_brain() {
            reflex.push(latency);
        }
    }
    all.sort_by(f64::total_cmp);
    reflex.sort_by(f64::total_cmp);
    let alerts = lines.iter().filter(|l| l.alert).count() as u64;
    let true_pages = lines
        .iter()
        .zip(&paged)
        .filter(|(l, p)| l.alert && **p)
        .count() as u64;
    let pages = paged.iter().filter(|p| **p).count() as u64;
    let brain_calls = *calls.lock().unwrap();
    Live {
        setup: setup.to_string(),
        brain: brain_name.to_string(),
        lines: lines.len() as u64,
        alerts,
        brain_calls,
        pages,
        true_pages,
        precision: if pages == 0 {
            0.0
        } else {
            true_pages as f64 / pages as f64
        },
        recall: if alerts == 0 {
            0.0
        } else {
            true_pages as f64 / alerts as f64
        },
        page_p50_ms: (!all.is_empty()).then(|| percentile(&all, 0.5)),
        page_p99_ms: (!all.is_empty()).then(|| percentile(&all, 0.99)),
        reflex_pages: reflex.len() as u64,
        reflex_p50_ms: (!reflex.is_empty()).then(|| percentile(&reflex, 0.5)),
        dropped: lines.len() as u64 - (*counter.lock().unwrap() as u64).min(lines.len() as u64),
        expectations: expectations.map(|e| e.snapshot()),
        brain_metrics: fabric
            .neuron("brain")
            .map(|id| fabric.describe().neurons[id.index()].counters.extra.clone())
            .unwrap_or_default(),
        seconds,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub lines: usize,
    pub first: u64,
    pub last: u64,
    pub replay: Vec<(String, Tally)>,
    pub live: Vec<Live>,
}

pub fn bench(dir: &str, gguf: Option<&str>) -> Report {
    let path = Path::new(dir).join("BGL.log");
    let (first, last, total) = span(&path);
    eprintln!("BGL: {total} lines, {} days", (last - first) / 86_400);
    let mut replay = Vec::new();
    for (name, run) in [
        ("brain", replay_brain as fn(&Path, u64, u64, usize) -> Tally),
        ("habituation", replay_habituation),
        ("expectations", replay_expectations),
    ] {
        eprintln!("replay: {name}");
        replay.push((name.to_string(), run(&path, first, last, total)));
    }
    let lines = slice(&path, total);
    eprintln!(
        "live: {} lines, {} alerts",
        lines.len(),
        lines.iter().filter(|l| l.alert).count()
    );
    let mut live_runs = Vec::new();
    for setup in ["brain", "habituation", "expectations"] {
        eprintln!("live: {setup}");
        let expectations = setup == "expectations";
        live_runs.push(live(
            &lines,
            setup,
            "mock LLM",
            Box::new(move |calls| {
                Arc::new(MockLlm {
                    expectations,
                    calls,
                })
            }),
        ));
    }
    if let Some(gguf) = gguf {
        eprintln!("live: expectations with {gguf}");
        live_runs.push(live(
            &lines,
            "expectations",
            &format!(
                "{} (candle)",
                Path::new(gguf)
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
            ),
            {
                let gguf = gguf.to_string();
                Box::new(move |calls| crate::expect_llm::brain(&gguf, calls))
            },
        ));
    }
    Report {
        lines: total,
        first,
        last,
        replay,
        live: live_runs,
    }
}

pub fn print(report: &Report) {
    println!(
        "BGL: {} lines over {} days, plus a heartbeat a minute with {DROPOUTS} dropouts of {} min",
        report.lines,
        (report.last - report.first) / 86_400,
        DROPOUT / 60
    );
    println!(
        "\nreplay      {:>10} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>9}",
        "brain", "precision", "recall", "ep. prec", "ep. rec", "dropouts", "seconds"
    );
    for (name, t) in &report.replay {
        println!(
            "{name:<11} {:>10} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>6}/{:<2}  {:>9.1}",
            t.brain_calls,
            t.precision,
            t.recall,
            t.episode_precision,
            t.episode_recall,
            t.dropouts_found,
            t.dropouts,
            t.seconds
        );
        println!(
            "            brain calls by tenth: {:?}",
            t.brain_calls_by_tenth
        );
    }
    println!(
        "\nlive ({} lines at {LIVE_RATE} lines/s)",
        report.live.first().map_or(0, |l| l.lines)
    );
    for l in &report.live {
        println!(
            "{:<13} {:<34} calls {:>5} pages {:>5} prec {:.3} rec {:.3} p50 {:?} p99 {:?} reflex {} at {:?}",
            l.setup,
            l.brain,
            l.brain_calls,
            l.pages,
            l.precision,
            l.recall,
            l.page_p50_ms,
            l.page_p99_ms,
            l.reflex_pages,
            l.reflex_p50_ms
        );
    }
}
