//! Scaffolding for the nerve's unit tests

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::fabric::Fabric;
use super::neuron::{Firing, FnNeuron, Mechanisms, Neuron, NeuronKind};
use super::receptor::Receptor;
use super::signal::{Feature, NeuronId, Payload, Signal};

/// Wait for `f` to hold, checking every millisecond; fails the test after 10 seconds
pub(crate) fn wait_until(f: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !f() {
        assert!(Instant::now() < deadline, "still not true after 10 s");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A receptor that passes everything on
pub(crate) fn pass(name: &str) -> impl Neuron {
    FnNeuron::reflex(name, |s, _| Firing::one(s)).with_kind(NeuronKind::Receptor)
}

/// A fabric where a receptor `in` reaches `neuron` through a reflex; stimulate the id returned
pub(crate) fn fabric_with(neuron: impl Neuron) -> (Fabric, NeuronId) {
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"), Mechanisms::NONE);
    let n = b.neuron(neuron, Mechanisms::NONE);
    b.reflex(r, n, 1.0);
    (b.build(), r)
}

/// A started fabric with `receptor` and a sink behind it that records what arrives
pub(crate) fn fabric_with_receptor<R: Receptor>(
    receptor: R,
) -> (Fabric, NeuronId, Arc<Mutex<Vec<Signal>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut b = Fabric::builder();
    let r = b.receptor(receptor, Mechanisms::NONE);
    let sink = {
        let seen = seen.clone();
        b.neuron(
            FnNeuron::reflex("sink", move |s, _| {
                seen.lock().unwrap().push(s);
                Firing::Quiet
            })
            .with_kind(NeuronKind::Effector),
            Mechanisms::NONE,
        )
    };
    b.reflex(r, sink, 1.0);
    let fabric = b.build();
    fabric.start();
    (fabric, r, seen)
}

/// A command for the effector named `effector`
pub(crate) fn command(effector: &str, args: serde_json::Value) -> Signal {
    Signal::new(Feature::NONE, Payload::None).command(effector, args)
}
