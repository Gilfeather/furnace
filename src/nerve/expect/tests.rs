//! Expectations with a scripted brain: no LLM, a function from surprise to commands

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use super::*;
use crate::nerve::effector::{self, Effector};
use crate::nerve::{Fabric, FnNeuron, Mechanisms, NeuronId, Summation};

const SECOND: Duration = Duration::from_secs(1);

type Script = Box<dyn Fn(&Value, &Signal) -> Vec<Signal> + Send + Sync>;

struct Rig {
    fabric: Fabric,
    input: NeuronId,
    expect: NeuronId,
    expectations: Expectations,
    /// The surprises the brain saw, in order
    seen: Arc<Mutex<Vec<Value>>>,
    calls: Arc<AtomicUsize>,
    page: mpsc::Receiver<effector::Delivery>,
    ticket: mpsc::Receiver<effector::Delivery>,
}

fn rig(config: ExpectConfig, script: Script) -> Rig {
    let expectations = Expectations::new("logs-expect", config);
    let mut b = Fabric::builder();
    let input = b.neuron(
        FnNeuron::reflex("logs", |s, _| Firing::one(s)),
        Mechanisms::NONE,
    );
    let expect = b.receptor(expectations.clone(), Mechanisms::NONE);
    b.reflex(input, expect, 1.0);
    b.neuron(
        Effector::new(expectations.effector("expect")).no_retry(),
        Mechanisms::NONE,
    );
    let (page_act, page) = effector::ChannelOut::new("page", 64);
    b.neuron(Effector::new(page_act), Mechanisms::NONE);
    let (ticket_act, ticket) = effector::ChannelOut::new("ticket", 64);
    b.neuron(Effector::new(ticket_act), Mechanisms::NONE);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let brain = {
        let (seen, calls) = (seen.clone(), calls.clone());
        FnNeuron::reflex("brain", move |s: Signal, _: &Cx<'_>| {
            let surprise = s
                .payload
                .to_json()
                .and_then(|v| v.get("surprise").cloned())
                .unwrap_or_default();
            let fired = script(&surprise, &s);
            seen.lock().unwrap().push(surprise);
            calls.fetch_add(1, Ordering::SeqCst);
            Firing::Fire(fired)
        })
        .with_kind(NeuronKind::Brain)
    };
    let brain = b.neuron(brain, Mechanisms::NONE);
    b.deliberative(expect, brain, 1.0, Summation::none());
    Rig {
        fabric: b.build().unwrap(),
        input,
        expect,
        expectations,
        seen,
        calls,
        page,
        ticket,
    }
}

impl Rig {
    fn line(&self, text: &str) {
        self.fabric
            .stimulate(self.input, Signal::new(Feature::NONE, Payload::text(text)));
    }

    fn json(&self, value: Value) {
        self.fabric
            .stimulate(self.input, Signal::new(Feature::NONE, Payload::json(value)));
    }

    /// Wait until the brain has been called `n` times in all and its queue is empty
    fn brain_calls(&self, n: usize) {
        let deadline = Instant::now() + 10 * SECOND;
        while (self.calls.load(Ordering::SeqCst) < n || self.fabric.waiting() > 0)
            && Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        assert_eq!(self.calls.load(Ordering::SeqCst), n, "brain calls");
    }

    /// Wait until what the brain fired has taken effect: `done` holds of the counters
    fn settle(&self, done: impl Fn(&ExpectMetrics) -> bool) {
        let deadline = Instant::now() + 10 * SECOND;
        while !done(&self.expectations.snapshot()) && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            done(&self.expectations.snapshot()),
            "{:?}",
            self.expectations.snapshot()
        );
    }

    fn reasons(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|s| s["reason"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    /// The absences due at `at` from now, into the fabric
    fn tick(&self, at: Duration) {
        for signal in self.expectations.due(Instant::now() + at) {
            self.fabric.stimulate(self.expect, signal);
        }
    }
}

/// The brain's answer: an `expect` command
fn expect(signal: &Signal, args: Value) -> Signal {
    signal.command("expect", args)
}

fn none() -> Script {
    Box::new(|_, _| Vec::new())
}

fn quiet() -> ExpectConfig {
    ExpectConfig::new()
        .effectors(["ticket", "page"])
        .shadow(0.0)
        .young(0)
}

fn parse(value: Value) -> Expectation {
    Expectation::parse(&value).unwrap()
}

#[test]
fn a_brain_installs_a_match_and_the_template_stops_reaching_it() {
    // On an unexpected line, the brain absorbs its template from now on
    let rig = rig(
        quiet(),
        Box::new(|surprise, s| {
            let template = surprise["observation"]["template"].clone();
            vec![expect(
                s,
                serde_json::json!({"add": {
                    "id": format!("routine-{}", template.as_str().unwrap_or_default().len()),
                    "kind": "match",
                    "selector": {"template": template},
                    "provenance": {"reason": surprise["reason"]},
                }}),
            )]
        }),
    );
    rig.line("081109 203615 148 INFO PacketResponder 1 for block blk_1 terminating");
    rig.brain_calls(1);
    rig.settle(|m| m.installed == 1);
    for i in 0..50 {
        rig.line(&format!(
            "081109 2036{i:02} 1{i} INFO PacketResponder {i} for block blk_{i}9 terminating"
        ));
    }
    rig.brain_calls(1);
    // Another template still surprises, with its context
    rig.line("081109 203700 99 WARN disk 7 nearly full");
    rig.brain_calls(2);
    rig.settle(|m| m.installed == 2);
    assert_eq!(rig.reasons(), ["unexpected", "unexpected"]);
    let seen = rig.seen.lock().unwrap();
    assert_eq!(
        seen[1]["observation"]["template"],
        "<*> <*> <*> WARN disk <*> nearly full"
    );
    drop(seen);
    let m = rig.expectations.snapshot();
    assert_eq!(
        (m.signals, m.absorbed, m.unexpected, m.installed),
        (52, 50, 2, 2)
    );
    assert!((m.surprise_rate - 2.0 / 52.0).abs() < 1e-9);
    // GET /nerve lists them with their counters and provenance
    let described = rig.fabric.describe();
    let shown = &described.neurons[rig.expect.index()].describe;
    let listed = shown["expectations"].as_array().unwrap();
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().any(|e| e["matches"] == 50
        && e["expectation"]["provenance"]["reason"] == "unexpected"));
}

#[test]
fn a_heartbeat_that_does_not_come_is_a_surprise_once() {
    let rig = rig(quiet(), none());
    rig.expectations
        .install(
            parse(serde_json::json!({
                "id": "hb-a", "kind": "every", "period_ms": 5000, "tolerance_ms": 1000,
                "selector": {"template": "heartbeat from svc-a"}
            })),
            Instant::now(),
        )
        .unwrap();
    rig.line("heartbeat from svc-a");
    rig.tick(4 * SECOND);
    rig.brain_calls(0);
    // Silent past period + tolerance: an absence reaches the brain, once
    rig.tick(7 * SECOND);
    rig.brain_calls(1);
    rig.tick(20 * SECOND);
    rig.brain_calls(1);
    assert_eq!(rig.reasons(), ["absence"]);
    let seen = rig.seen.lock().unwrap();
    assert_eq!(seen[0]["broken"][0]["expectation"], "hb-a");
    assert_eq!(seen[0]["broken"][0]["kind"], "every");
    drop(seen);
    // It beats again: absorbed, and armed again
    rig.line("heartbeat from svc-a");
    rig.brain_calls(1);
    assert_eq!(rig.expectations.snapshot().absences, 1);
}

#[test]
fn a_value_out_of_range_acts_and_escalates_and_a_false_alarm_teaches_the_bound() {
    let rig = rig(
        quiet(),
        Box::new(|surprise, s| {
            // The brain judges 29 degrees normal: a false alarm
            if surprise["observation"]["fields"]["t"] == 29 {
                vec![expect(
                    s,
                    serde_json::json!({"verdict": {"id": "temp", "normal": true}}),
                )]
            } else {
                Vec::new()
            }
        }),
    );
    rig.expectations
        .install(
            parse(serde_json::json!({
                "id": "temp", "kind": "range", "field": "t", "min": 18.0, "max": 27.0,
                "selector": {"fields": {"room": "lab"}},
                "on_violation": {"act": [{"effector": "page", "args": {"why": "temperature"}}]},
                "learn": {"max": [27.0, 30.0]}
            })),
            Instant::now(),
        )
        .unwrap();
    rig.json(serde_json::json!({"room": "lab", "t": 22}));
    rig.brain_calls(0);
    rig.json(serde_json::json!({"room": "lab", "t": 29}));
    // The page goes at once, the surprise to the brain
    assert_eq!(
        rig.page.recv_timeout(5 * SECOND).unwrap().command.args["why"],
        "temperature"
    );
    rig.brain_calls(1);
    rig.settle(|m| m.learned == 1);
    assert_eq!(
        rig.expectations.get("temp").unwrap().param("max"),
        Some(29.0)
    );
    // Now 29 is within range; 35 is not, and learning never goes past the brain's bound
    rig.json(serde_json::json!({"room": "lab", "t": 29}));
    rig.brain_calls(1);
    rig.json(serde_json::json!({"room": "lab", "t": 35}));
    rig.brain_calls(2);
    let m = rig.expectations.snapshot();
    assert_eq!(
        (m.false_alarms, m.learned, m.violations["range"]),
        (1, 1, 2)
    );
}

#[test]
fn rates_break_above_their_max_at_once_and_below_their_min_at_the_window_s_end() {
    let rig = rig(quiet(), none());
    for e in [
        serde_json::json!({"id": "warn-disk", "kind": "rate", "window_ms": 60000, "max": 3,
            "selector": {"template": "WARN disk <*>"}}),
        serde_json::json!({"id": "orders", "kind": "rate", "window_ms": 10000, "min": 2,
            "selector": {"template": "order <*> placed"}}),
    ] {
        rig.expectations.install(parse(e), Instant::now()).unwrap();
    }
    for i in 0..3 {
        rig.line(&format!("WARN disk {i}"));
    }
    rig.brain_calls(0);
    rig.line("WARN disk 9");
    rig.brain_calls(1);
    // One order in the window, two expected: at the window's end, a surprise
    rig.line("order 1 placed");
    rig.tick(11 * SECOND);
    rig.brain_calls(2);
    assert_eq!(rig.reasons(), ["violation", "absence"]);
    let seen = rig.seen.lock().unwrap();
    assert_eq!(seen[1]["broken"][0]["observed"], 1);
}

#[test]
fn what_should_follow_and_what_should_never_come() {
    let rig = rig(quiet(), none());
    for e in [
        serde_json::json!({"id": "job", "kind": "then", "within_ms": 600000,
            "selector": {"template": "job <*> started"},
            "then": {"template": "job <*> finished"}}),
        serde_json::json!({"id": "no-root", "kind": "never",
            "selector": {"template": "auth failure for root"}}),
    ] {
        rig.expectations.install(parse(e), Instant::now()).unwrap();
    }
    rig.line("job 1 started");
    rig.line("job 1 finished");
    rig.tick(11 * 60 * SECOND);
    rig.brain_calls(0);
    rig.line("job 2 started");
    rig.tick(11 * 60 * SECOND);
    rig.brain_calls(1);
    rig.line("auth failure for root");
    rig.brain_calls(2);
    assert_eq!(rig.reasons(), ["absence", "violation"]);
}

#[test]
fn the_scope_limits_hold_against_a_brain_that_writes_too_much() {
    let rig = rig(
        quiet().max(3),
        Box::new(|surprise, s| {
            let template = surprise["observation"]["template"].clone();
            vec![expect(
                s,
                serde_json::json!({"add": [
                    // absorb everything
                    {"id": "all", "kind": "match"},
                    // pay from a log line
                    {"id": "pay", "kind": "match", "selector": {"template": template},
                     "on_match": {"act": [{"effector": "pay"}]}},
                    // not what surprised: another template
                    {"id": "other", "kind": "match", "selector": {"template": "something else"}},
                    // fine
                    {"id": "ok", "kind": "match", "selector": {"template": template}},
                ], "approve": "all"}),
            )]
        }),
    );
    rig.line("login failed for alice");
    rig.brain_calls(1);
    rig.settle(|m| m.refused == 3 && m.installed == 1);
    assert_eq!(rig.expectations.ids(), ["ok"]);
}

#[test]
fn the_budgets_and_the_approval_mode() {
    let e = |id: &str| {
        parse(serde_json::json!({"id": id, "kind": "match", "selector": {"template": id}}))
    };
    let x = Expectations::new("x", ExpectConfig::new().max(2).writes_per_minute(3));
    let now = x.epoch();
    for id in ["a", "b"] {
        x.write(Write::Add(e(id)), None, true, now).unwrap();
    }
    assert!(x.write(Write::Add(e("c")), None, true, now).is_err(), "max");
    // A refused write does not count: replacing one is not a new one, and is the third
    assert!(x.write(Write::Add(e("a")), None, true, now).is_ok());
    assert!(
        x.write(Write::Add(e("b")), None, true, now).is_err(),
        "per minute"
    );
    // A minute later
    assert!(
        x.write(Write::Add(e("b")), None, true, now + 61 * SECOND)
            .is_ok()
    );

    let y = Expectations::new("y", ExpectConfig::new().approval(true));
    y.write(Write::Add(e("p")), None, true, now).unwrap();
    assert!(y.ids().is_empty());
    assert_eq!(y.pending(), ["p"]);
    y.approve("p", now).unwrap();
    assert_eq!(y.ids(), ["p"]);
    // The operator installs without waiting
    y.install(e("q"), now).unwrap();
    assert_eq!(y.ids(), ["p", "q"]);
}

#[test]
fn shadow_checks_retire_what_the_brain_says_was_wrongly_absorbed() {
    let rig = rig(
        quiet().young(2).retire_after(2),
        Box::new(|surprise, s| match surprise["reason"].as_str() {
            Some("shadow") => vec![expect(
                s,
                serde_json::json!({"verdict": {"id": surprise["absorbed_by"], "correct": false}}),
            )],
            _ => Vec::new(),
        }),
    );
    rig.expectations
        .install(
            parse(serde_json::json!({
                "id": "quiet-disk", "kind": "match", "selector": {"template": "disk <*> ok"}
            })),
            Instant::now(),
        )
        .unwrap();
    // Young: both matches are checked, the brain says no twice, the expectation retires
    rig.line("disk 1 ok");
    rig.brain_calls(1);
    rig.settle(|m| m.shadow_wrong == 1);
    rig.line("disk 2 ok");
    rig.brain_calls(2);
    rig.settle(|m| m.retired == 1);
    assert!(rig.expectations.ids().is_empty());
    rig.line("disk 3 ok");
    rig.brain_calls(3);
    assert_eq!(rig.reasons(), ["shadow", "shadow", "unexpected"]);
    assert!(rig.ticket.try_recv().is_err());
    let m = rig.expectations.snapshot();
    assert_eq!((m.shadows, m.shadow_wrong, m.retired), (2, 2, 1));
}

#[test]
fn rules_act_on_their_own_and_expire_and_retire_with_their_dependency() {
    let x = Expectations::new(
        "x",
        ExpectConfig::new()
            .effectors(["ticket"])
            .shadow(0.0)
            .young(0)
            .ttl(10 * SECOND),
    );
    let t0 = x.epoch();
    x.install(
        parse(serde_json::json!({
            "id": "vip", "kind": "match", "selector": {"fields": {"plan": "vip"}},
            "on_match": {"act": [{"effector": "ticket", "args": {"queue": "vip"}}]},
            "depends_on": ["customers"]
        })),
        t0,
    )
    .unwrap();
    let vip = Signal::new(
        Feature::NONE,
        Payload::json(serde_json::json!({"plan": "vip"})),
    );
    let (verdict, ..) = x.take(&vip, t0 + SECOND);
    assert!(matches!(verdict, Verdict::Absorbed(ref c, None) if c[0].args["queue"] == "vip"));
    // Unmatched for longer than the ttl: expired
    let (verdict, ..) = x.take(&vip, t0 + 12 * SECOND);
    assert!(matches!(verdict, Verdict::Surprise(Reason::Unexpected, ..)));
    assert_eq!(x.snapshot().expired, 1);
    // And a dependency's change retires what depends on it
    x.install(
        parse(serde_json::json!({
            "id": "vip", "kind": "match", "selector": {"fields": {"plan": "vip"}},
            "depends_on": ["customers"]
        })),
        t0 + 12 * SECOND,
    )
    .unwrap();
    assert_eq!(x.changed("prices"), 0);
    assert_eq!(x.changed("customers"), 1);
    assert!(x.ids().is_empty());
}

#[test]
fn the_period_follows_the_observed_interval_within_the_brain_s_bounds() {
    let x = Expectations::new("x", ExpectConfig::new().shadow(0.0).young(0));
    let t0 = x.epoch();
    x.install(
        parse(serde_json::json!({
            "id": "hb", "kind": "every", "period_ms": 5000,
            "selector": {"template": "beat"}, "learn": {"period_ms": [4000, 8000]}
        })),
        t0,
    )
    .unwrap();
    let beat = Signal::new(Feature::NONE, Payload::text("beat"));
    for i in 0..20 {
        x.take(&beat, t0 + Duration::from_millis(7000 * i));
    }
    let period = x.get("hb").unwrap().param("period_ms").unwrap();
    assert!((6500.0..=7000.0).contains(&period), "{period}");
    for i in 20..40 {
        x.take(
            &beat,
            t0 + Duration::from_millis(140_000 + 20_000 * (i - 20)),
        );
    }
    assert_eq!(x.get("hb").unwrap().param("period_ms"), Some(8000.0));
}

// ---------------------------------------------------------------------------------------------
// A synthetic stream with repeats, an absence and anomalies

/// The scripted brain of the synthetic stream, and what it did
struct Sim {
    expectations: Expectations,
    expect: Expect,
    brain_calls: u64,
    pages: u64,
}

impl Sim {
    /// The brain learns the shape of what is normal and pages on what is not
    fn brain(&mut self, reason: Reason, template: &str, signal: &Signal) {
        self.brain_calls += 1;
        let add = |e: Value| Command::new("expect", serde_json::json!({ "add": e }));
        let commands = match reason {
            Reason::Shadow | Reason::Absence => Vec::new(),
            Reason::Violation => vec![Command::new("page", Value::Null)],
            Reason::Unexpected if template.contains("panic") => {
                vec![Command::new("page", Value::Null)]
            }
            Reason::Unexpected if template.contains("root") => vec![
                Command::new("page", Value::Null),
                add(serde_json::json!({
                    "id": "no-root", "kind": "never", "selector": {"template": template}})),
            ],
            Reason::Unexpected if template.starts_with("heartbeat") => {
                vec![add(serde_json::json!({
                    "id": "hb", "kind": "every", "period_ms": 1000, "tolerance_ms": 500,
                    "selector": {"template": template}}))]
            }
            Reason::Unexpected if template.starts_with("temp") => {
                vec![add(serde_json::json!({
                    "id": "temp", "kind": "range", "field": "v1", "min": 15.0, "max": 30.0,
                    "selector": {"template": template}}))]
            }
            Reason::Unexpected => vec![add(serde_json::json!({
                "id": format!("routine-{:016x}", sketch64(template.as_bytes())),
                "kind": "match", "selector": {"template": template}}))],
        };
        for command in commands {
            if command.effector == "expect" {
                let _ = self.expect.act(&command, signal);
            } else {
                self.pages += 1;
            }
        }
    }

    /// One line at `at`; whether it was escalated (a shadow check is not)
    fn line(&mut self, text: &str, at: Instant) -> bool {
        let signal = Signal::new(Feature::NONE, Payload::text(text));
        let (verdict, observation, ..) = self.expectations.take(&signal, at);
        let template = observation.template_text.unwrap_or_default();
        match verdict {
            Verdict::Absorbed(_, None) => false,
            Verdict::Absorbed(_, Some(_)) => {
                self.brain(Reason::Shadow, &template, &signal);
                false
            }
            Verdict::Surprise(reason, ..) => {
                self.brain(reason, &template, &signal);
                true
            }
        }
    }
}

/// Ten minutes in steps of 100 ms: routine lines of five templates (0 to 2 per step), a
/// heartbeat every second that drops out for ten seconds at minute 5, a sensor reading every
/// 2 s that leaves its range three times, three kernel panics and two root logins.
/// Returns (signals, brain calls, pages, anomalies escalated, absences).
fn synthetic() -> (u64, u64, u64, u64, u64) {
    let expectations = Expectations::new(
        "synthetic",
        ExpectConfig::new()
            .effectors(["page"])
            .shadow(0.01)
            .young(3),
    );
    let mut sim = Sim {
        expect: expectations.effector("expect"),
        expectations,
        brain_calls: 0,
        pages: 0,
    };
    let t0 = sim.expectations.epoch();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let routine = [
        "INFO request <n> served in <n> ms",
        "INFO cache hit for key <n>",
        "INFO user <n> logged in",
        "DEBUG gc pause <n> ms",
        "INFO job <n> started",
    ];
    let (mut signals, mut anomalies, mut absences) = (0, 0, 0);
    for tick in 0..6000u64 {
        let at = t0 + Duration::from_millis(100 * tick);
        for absence in sim.expectations.due(at) {
            if absence.payload.as_command().is_none() {
                absences += 1;
                sim.brain(Reason::Absence, "", &absence);
            }
        }
        let mut lines: Vec<(String, bool)> = Vec::new();
        for _ in 0..(next() % 3) {
            let mut line = routine[(next() % 5) as usize].to_string();
            while line.contains("<n>") {
                line = line.replacen("<n>", &(next() % 10_000).to_string(), 1);
            }
            lines.push((line, false));
        }
        if tick % 10 == 0 && !(3000..3100).contains(&tick) {
            lines.push(("heartbeat from svc-a".into(), false));
        }
        if tick % 20 == 5 {
            let anomalous = [1205, 3305, 5105].contains(&tick);
            let reading = if anomalous {
                41.5
            } else {
                20.0 + (next() % 50) as f64 / 10.0
            };
            lines.push((format!("temp sensor-3 reading {reading}"), anomalous));
        }
        if [2000, 4000, 5800].contains(&tick) {
            lines.push((format!("kernel panic on node {}", tick % 7), true));
        }
        if [2500, 4500].contains(&tick) {
            lines.push(("auth failure for root".into(), true));
        }
        for (line, anomalous) in lines {
            signals += 1;
            if sim.line(&line, at) && anomalous {
                anomalies += 1;
            }
        }
    }
    (signals, sim.brain_calls, sim.pages, anomalies, absences)
}

#[test]
fn a_synthetic_stream_reaches_the_brain_with_its_surprises_only() {
    let (signals, brain_calls, pages, anomalies, absences) = synthetic();
    // Every anomaly reached the brain and was paged, the dropout was one absence, nothing
    // normal was paged
    assert_eq!((signals, pages, anomalies, absences), (6846, 8, 8, 1));
    // The brain saw 16 surprises (7 shapes to learn, 3 panics, a root login, 4 violations,
    // the absence), 21 checks of young expectations, and a 1% sample of the rest: under 2%
    // of the stream. The sample follows signal ids, which other tests share, so it varies.
    assert!((37..=signals / 50).contains(&brain_calls), "{brain_calls}");
}
