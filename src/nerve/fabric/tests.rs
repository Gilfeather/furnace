use super::*;
use crate::nerve::neuron::{Cx, Firing, FnNeuron, Habituation, Mechanisms};
use crate::nerve::signal::{Feature, Payload, Signal};
use crate::nerve::testing::{pass, wait_until};
use std::sync::atomic::AtomicUsize;
use std::thread::ThreadId;
use std::time::Duration;

impl Fabric {
    pub(crate) fn seed_latency(&self, pathway: usize, latency: Duration) {
        for _ in 0..4 {
            self.0.pathways[pathway].latency.observe(latency);
        }
    }
}

/// Counts what reaches it and remembers the thread it ran on
struct Sink {
    name: &'static str,
    kind: NeuronKind,
    hits: AtomicUsize,
    thread: Mutex<Option<ThreadId>>,
    queued: AtomicUsize,
    seen: Mutex<Vec<Signal>>,
}

impl Sink {
    fn new(name: &'static str, kind: NeuronKind) -> Arc<Self> {
        Arc::new(Self {
            name,
            kind,
            hits: AtomicUsize::new(0),
            thread: Mutex::new(None),
            queued: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    fn wait_for(&self, hits: usize) {
        wait_until(|| self.hits() >= hits);
    }
}

impl Neuron for Sink {
    fn name(&self) -> &str {
        self.name
    }
    fn kind(&self) -> NeuronKind {
        self.kind
    }
    fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
        self.hits.fetch_add(1, Ordering::SeqCst);
        *self.thread.lock().unwrap() = Some(std::thread::current().id());
        if signal.trace.was_queued() {
            self.queued.fetch_add(1, Ordering::SeqCst);
        }
        self.seen.lock().unwrap().push(signal);
        Firing::Quiet
    }
}

fn receptor(name: &'static str) -> Arc<Sink> {
    Sink::new(name, NeuronKind::Receptor)
}

fn signal(bytes: &[u8]) -> Signal {
    Signal::new(Feature::sketch(bytes), Payload::bytes(bytes.to_vec()))
}

#[test]
fn a_reflex_arc_is_a_direct_call_on_the_callers_thread() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let reflex = b.neuron(FnNeuron::reflex("classify", |s, _| {
        Firing::one(s.command("act", 1.into()))
    }));
    let effector = Sink::new("act", NeuronKind::Effector);
    let e = b.add(effector.clone());
    b.reflex(r, reflex);
    b.reflex(reflex, e);
    let fabric = b.build().unwrap();

    let outcome = fabric.stimulate(r, signal(b"stimulus"));
    // The effector ran before stimulate returned, on this thread, never queued
    assert_eq!(effector.hits(), 1);
    assert_eq!(
        *effector.thread.lock().unwrap(),
        Some(std::thread::current().id())
    );
    assert_eq!(effector.queued.load(Ordering::SeqCst), 0);
    assert_eq!(outcome.hops, 3);
    assert_eq!(outcome.acted, 1);
    assert_eq!(outcome.queued, 0);
    let seen = effector.seen.lock().unwrap();
    assert_eq!(seen[0].trace.hops(), &[r, reflex, e]);
    assert!(!seen[0].trace.reached_brain());
    // The arc's stimulus-to-action latency was measured
    let arc = fabric.0.cells[e.index()].arcs[r.index()].get().unwrap();
    assert_eq!(arc.count(), 1);
    assert_eq!(fabric.0.pathways[1].latency.count(), 1);
    assert_eq!(fabric.brain_fraction(), 0.0);
}

#[test]
fn habituation_at_the_receptor_keeps_repeats_from_the_brain() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    b.gate(
        r,
        Mechanisms::NONE
            .threshold(0.6)
            .habituation(Habituation::new(Duration::from_secs(60))),
    );
    let brain = Sink::new("brain", NeuronKind::Brain);
    let bid = b.add(brain.clone());
    b.deliberative(r, bid)
        .summation(Summation::default().window(Duration::ZERO));
    let fabric = b.build().unwrap();

    let mut queued = 0;
    for _ in 0..10 {
        queued += fabric.stimulate(r, signal(b"same")).queued;
    }
    queued += fabric.stimulate(r, signal(b"new")).queued;
    // Only two were queued, so no more than two can reach the brain
    assert_eq!(queued, 2);
    brain.wait_for(2);
    assert_eq!(brain.hits(), 2);
    assert_eq!(brain.queued.load(Ordering::SeqCst), 2);
    let c = &fabric.0.cells[r.index()].counters;
    assert_eq!(c.quiet_habituation.load(Ordering::SeqCst), 9);
    assert_eq!(c.fired.load(Ordering::SeqCst), 2);
    assert!((fabric.brain_fraction() - 2.0 / 11.0).abs() < 1e-9);
}

#[test]
fn selection_picks_the_heaviest_pathway_that_fits_the_deadline() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let fast = Sink::new("fast", NeuronKind::Effector);
    let slow = Sink::new("slow", NeuronKind::Effector);
    let f = b.add(fast.clone());
    let s = b.add(slow.clone());
    // The slow path is preferred (heavier) when there is time
    b.reflex(r, f).weight(0.5);
    b.reflex(r, s);
    let fabric = b.build().unwrap();
    fabric.seed_latency(0, Duration::from_micros(10));
    fabric.seed_latency(1, Duration::from_millis(50));

    fabric.stimulate(r, signal(b"a"));
    assert_eq!((fast.hits(), slow.hits()), (0, 1));
    // A deadline the slow path cannot meet selects the fast one
    fabric.stimulate(
        r,
        signal(b"b").with_urgency(super::super::signal::Urgency::within(
            Duration::from_millis(5),
        )),
    );
    assert_eq!((fast.hits(), slow.hits()), (1, 1));
    assert_eq!(fabric.0.pathways[1].ineligible.load(Ordering::SeqCst), 1);
    // A deadline already past goes nowhere
    let outcome = fabric.stimulate(
        r,
        signal(b"c").with_urgency(super::super::signal::Urgency::at(Instant::now())),
    );
    assert_eq!(outcome.no_pathway, 1);
    assert_eq!((fast.hits(), slow.hits()), (1, 1));
}

#[test]
fn a_pathway_fits_a_deadline_its_p99_equals() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let e = b.add(Sink::new("act", NeuronKind::Effector));
    b.reflex(r, e);
    let fabric = b.build().unwrap();
    let pathway = &fabric.0.pathways[0];
    // Unmeasured, it fits anything
    assert!(pathway.fits(Some(Duration::ZERO)));
    fabric.seed_latency(0, Duration::from_millis(50));
    assert_eq!(pathway.latency.p99(), Some(Duration::from_millis(50)));
    assert!(pathway.fits(None));
    assert!(pathway.fits(Some(Duration::from_millis(50))));
    assert!(!pathway.fits(Some(Duration::from_millis(50) - Duration::from_nanos(1))));
}

#[test]
fn compute_runs_inline_on_a_reflex_arc_and_its_signals_propagate() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let ran_on = Arc::new(Mutex::new(None));
    let model = b.neuron(FnNeuron::reflex("model", {
        let ran_on = ran_on.clone();
        move |s, _| {
            let ran_on = ran_on.clone();
            Firing::compute(move || {
                *ran_on.lock().unwrap() = Some(std::thread::current().id());
                vec![s.command("act", 1.into())]
            })
        }
    }));
    let effector = Sink::new("act", NeuronKind::Effector);
    let e = b.add(effector.clone());
    b.reflex(r, model);
    b.reflex(model, e);
    let fabric = b.build().unwrap();

    let outcome = fabric.stimulate(r, signal(b"x"));
    assert_eq!((outcome.acted, effector.hits()), (1, 1));
    assert_eq!(*ran_on.lock().unwrap(), Some(std::thread::current().id()));
    let c = &fabric.0.cells[model.index()].counters;
    assert_eq!(c.computed.load(Ordering::SeqCst), 1);
    assert_eq!(c.fired.load(Ordering::SeqCst), 1);
}

#[test]
fn commands_reach_the_named_effector() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let reflex = b.neuron(FnNeuron::reflex("decide", |s, _| {
        Firing::Fire(vec![
            s.command("webhook", serde_json::json!({"n": 1})),
            s.command("journal", serde_json::json!({"n": 2})),
            s.command("nobody", serde_json::Value::Null),
        ])
    }));
    let webhook = Sink::new("webhook", NeuronKind::Effector);
    let journal = Sink::new("journal", NeuronKind::Effector);
    let w = b.add(webhook.clone());
    b.add(journal.clone());
    b.reflex(r, reflex);
    b.reflex(reflex, w);
    let fabric = b.build().unwrap();

    let outcome = fabric.stimulate(r, signal(b"x"));
    assert_eq!((webhook.hits(), journal.hits()), (1, 1));
    assert_eq!(outcome.unrouted, 1);
    assert_eq!(outcome.acted, 2);
    // The journal had no declared pathway: delivered directly
    assert_eq!(fabric.0.commands_direct.load(Ordering::SeqCst), 1);
    assert_eq!(fabric.0.pathways[1].signals.load(Ordering::SeqCst), 1);
}

#[test]
fn refractory_period_drops_a_burst_after_the_first() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    b.gate(r, Mechanisms::NONE.refractory(Duration::from_secs(5)));
    let e = Sink::new("act", NeuronKind::Effector);
    let eid = b.add(e.clone());
    b.reflex(r, eid);
    let fabric = b.build().unwrap();
    for i in 0..5u8 {
        fabric.stimulate(r, signal(&[i]));
    }
    assert_eq!(e.hits(), 1);
    let c = &fabric.0.cells[r.index()].counters;
    assert_eq!(c.quiet_refractory.load(Ordering::SeqCst), 4);
}

#[test]
fn deliberative_pathway_sums_a_window_into_one_firing() {
    struct Batchy {
        sizes: Mutex<Vec<usize>>,
    }
    impl Neuron for Batchy {
        fn name(&self) -> &str {
            "brain"
        }
        fn kind(&self) -> NeuronKind {
            NeuronKind::Brain
        }
        fn stimulate(&self, _: Signal, _: &Cx<'_>) -> Firing {
            unreachable!("stimulate_many is used")
        }
        fn stimulate_many(&self, signals: Vec<Signal>, _: &Cx<'_>) -> Vec<Firing> {
            self.sizes.lock().unwrap().push(signals.len());
            signals
                .into_iter()
                .map(|_| Firing::Fire(Vec::new()))
                .collect()
        }
    }
    let batchy = Arc::new(Batchy {
        sizes: Mutex::new(Vec::new()),
    });
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let brain = b.add(batchy.clone());
    b.deliberative(r, brain).summation(
        Summation::default()
            .max_batch(4)
            .window(Duration::from_millis(50)),
    );
    let fabric = b.build().unwrap();
    for i in 0..6u8 {
        assert_eq!(fabric.stimulate(r, signal(&[i])).queued, 1);
    }
    // The pathway's latency is measured last, after the batch has fired
    wait_until(|| fabric.0.pathways[0].latency.count() == 6);
    assert_eq!(*batchy.sizes.lock().unwrap(), vec![4, 2]);
    assert_eq!(fabric.waiting(), 0);
}

#[test]
fn a_loop_is_cut() {
    let mut b = Fabric::builder();
    let a = b.neuron(pass("a"));
    let c = b.neuron(FnNeuron::reflex("b", |s, _| Firing::one(s)));
    b.reflex(a, c);
    b.reflex(c, a);
    let fabric = b.build().unwrap();
    let outcome = fabric.stimulate(a, signal(b"round"));
    assert_eq!(outcome.hops, MAX_HOPS);
    assert_eq!(fabric.0.loops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_loop_through_a_queue_is_cut() {
    let mut b = Fabric::builder();
    let reflex = b.neuron(pass("reflex"));
    let brain =
        b.neuron(FnNeuron::reflex("brain", |s, _| Firing::one(s)).with_kind(NeuronKind::Brain));
    b.deliberative(reflex, brain).summation(Summation::none());
    b.reflex(brain, reflex);
    let fabric = b.build().unwrap();
    fabric.stimulate(reflex, signal(b"round"));
    // Each pass through the queue restarts the depth; the trace still counts every hop
    wait_until(|| fabric.0.loops.load(Ordering::SeqCst) > 0);
    assert_eq!(fabric.0.loops.load(Ordering::SeqCst), 1);
    assert!(fabric.0.reached_brain.load(Ordering::SeqCst) < MAX_HOPS as u64);
}

#[test]
fn prepare_runs_once_for_far_signals_on_deliberative_pathways() {
    struct Prep(AtomicUsize);
    impl Neuron for Prep {
        fn name(&self) -> &str {
            "brain"
        }
        fn kind(&self) -> NeuronKind {
            NeuronKind::Brain
        }
        fn stimulate(&self, _: Signal, _: &Cx<'_>) -> Firing {
            Firing::Quiet
        }
        fn prepare(&self, _: &Cx<'_>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let prep = Arc::new(Prep(AtomicUsize::new(0)));
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let brain = b.add(prep.clone());
    b.deliberative(r, brain).summation(Summation::none());
    let fabric = b.build().unwrap();
    for i in 0..3u8 {
        fabric.stimulate(r, signal(&[i]));
    }
    // Prepared before the first signal was queued; the next two were within the second
    assert_eq!(prep.0.load(Ordering::SeqCst), 1);
    let _ = receptor("unused");
}

#[test]
fn a_builder_with_two_neurons_of_one_name_or_a_foreign_pathway_is_an_error() {
    let mut b = Fabric::builder();
    b.neuron(pass("in"));
    b.neuron(pass("in"));
    assert!(matches!(b.build(), Err(BuildError::DuplicateName(name)) if name == "in"));

    let mut other = Fabric::builder();
    let a = other.neuron(pass("a"));
    let b2 = other.neuron(pass("b"));
    let mut b = Fabric::builder();
    let only = b.neuron(pass("only"));
    b.reflex(only, b2);
    assert!(matches!(
        b.build(),
        Err(BuildError::UnknownNeuron { from, to }) if from == only && to == b2
    ));
    let _ = a;
}

#[test]
fn the_builder_defaults_wire_a_reflex_arc_and_prefer_the_brain() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    // Option<Signal>: a command for alerts, quiet for the rest
    let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
        let text = signal.payload.as_text()?;
        text.contains("ALERT")
            .then(|| signal.command("alarm", serde_json::json!({"text": text})))
    }));
    let (channel, _deliveries) = crate::nerve::effector::ChannelOut::new("alarm", 4);
    // No pathway to the effector: the command finds it by name
    b.effector(channel);
    let brain = b.add(Sink::new("brain", NeuronKind::Brain));
    b.reflex(r, triage);
    b.deliberative(r, brain);
    b.reflex(triage, brain).weight(0.5);
    let fabric = b.build().unwrap();

    let weights: Vec<(PathwayKind, f32)> = fabric
        .0
        .pathways
        .iter()
        .map(|p| (p.kind, p.weight))
        .collect();
    assert_eq!(
        weights,
        [
            (PathwayKind::Reflex, 1.0),
            (PathwayKind::Deliberative, 2.0),
            (PathwayKind::Reflex, 0.5)
        ]
    );
    let text = |t: &str| Signal::new(Feature::sketch(t.as_bytes()), Payload::text(t));
    let acted = fabric.stimulate(triage, text("ALERT: smoke"));
    assert_eq!((acted.acted, acted.quiet), (1, None));
    let quiet = fabric.stimulate(triage, text("all is well"));
    assert_eq!(
        (quiet.acted, quiet.quiet),
        (0, Some((triage, Quiet::NotInterested)))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_motor_side_can_run_on_a_runtime_of_ones_own() {
    let mut b = Fabric::builder().efferent(tokio::runtime::Handle::current());
    b.neuron(pass("in"));
    let fabric = b.build().unwrap();
    assert_eq!(
        fabric.efferent().id(),
        tokio::runtime::Handle::current().id()
    );
    // Dropping it on this runtime leaves the runtime alone
    drop(fabric);
    tokio::task::yield_now().await;
}

/// A brain that hangs: each firing tells the test it started, then blocks until the test opens
/// the gate
struct Stuck {
    entered: Mutex<std::sync::mpsc::Sender<()>>,
    open: Mutex<bool>,
    opened: std::sync::Condvar,
}

impl Stuck {
    fn new() -> (Arc<Self>, std::sync::mpsc::Receiver<()>) {
        let (entered, receiver) = std::sync::mpsc::channel();
        let stuck = Arc::new(Self {
            entered: Mutex::new(entered),
            open: Mutex::new(false),
            opened: std::sync::Condvar::new(),
        });
        (stuck, receiver)
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }
}

impl Neuron for Stuck {
    fn name(&self) -> &str {
        "brain"
    }
    fn kind(&self) -> NeuronKind {
        NeuronKind::Brain
    }
    fn stimulate(&self, _: Signal, _: &Cx<'_>) -> Firing {
        let _ = self.entered.lock().unwrap().send(());
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.opened.wait(open).unwrap();
        }
        Firing::Quiet
    }
}

/// Wait for the next firing of a [`Stuck`] brain; fails the test after 10 seconds
fn entered(firings: &std::sync::mpsc::Receiver<()>) {
    firings
        .recv_timeout(Duration::from_secs(10))
        .expect("the brain fired within 10 s");
}

#[test]
fn a_hung_brain_overflows_its_queue_while_deadlines_take_the_reflex() {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let (stuck, firings) = Stuck::new();
    let brain = b.add(stuck.clone());
    let reflex = Sink::new("reflex", NeuronKind::Effector);
    let fallback = b.add(reflex.clone());
    b.deliberative(r, brain)
        .weight(2.0)
        .summation(Summation::none().capacity(2));
    b.reflex(r, fallback).weight(1.0);
    let fabric = b.build().unwrap();

    // The brain takes the first signal and hangs on it
    assert_eq!(fabric.stimulate(r, signal(b"first")).queued, 1);
    entered(&firings);
    // It was measured slow before: ten seconds, far more than the deadlines below allow
    fabric.seed_latency(0, Duration::from_secs(10));

    // Signals with a deadline take the reflex at once, however long the brain hangs
    let urgent = || {
        signal(b"urgent").with_urgency(super::super::signal::Urgency::within(Duration::from_secs(
            1,
        )))
    };
    for _ in 0..10 {
        let outcome = fabric.stimulate(r, urgent());
        assert_eq!((outcome.queued, outcome.dropped), (0, 0));
    }
    assert_eq!(reflex.hits(), 10);

    // Signals without one queue for the brain up to the capacity; the rest are dropped and
    // counted rather than queued without bound
    let mut dropped = 0;
    for i in 0..50u8 {
        dropped += fabric.stimulate(r, signal(&[i])).dropped;
        assert!(fabric.waiting() <= 2, "{} waiting", fabric.waiting());
    }
    assert_eq!(dropped, 48);
    let pathway = &fabric.metrics().pathways[0];
    assert_eq!((pathway.overflow, pathway.ineligible), (48, 10));
    assert_eq!(fabric.waiting(), 2);
    // The reflex still answers while the queue is full
    fabric.stimulate(r, urgent());
    assert_eq!(reflex.hits(), 11);

    // Released, the brain works through what was kept, and nothing more
    stuck.open();
    entered(&firings);
    entered(&firings);
    drop(fabric);
    assert!(firings.try_recv().is_err());
}
