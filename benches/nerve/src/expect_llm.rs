//! A real model as the brain that writes expectations: a GGUF file through candle
//! (furnace-candle), behind furnace's expectation adapter (`furnace::nerve::expect::llm`)

use std::path::Path;
use std::sync::{Arc, Mutex};

use furnace::nerve::{Cx, Firing, Neuron, NeuronKind, Signal};
use furnace_bench_common::fail;

/// Counts the signals a brain is asked about
struct Counted<N> {
    brain: N,
    calls: Arc<Mutex<u64>>,
}

impl<N: Neuron> Neuron for Counted<N> {
    fn name(&self) -> &str {
        self.brain.name()
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Brain
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        *self.calls.lock().unwrap() += 1;
        self.brain.stimulate(signal, cx)
    }

    fn stimulate_many(&self, signals: Vec<Signal>, cx: &Cx<'_>) -> Vec<Firing> {
        *self.calls.lock().unwrap() += signals.len() as u64;
        self.brain.stimulate_many(signals, cx)
    }

    fn metrics(&self) -> serde_json::Value {
        self.brain.metrics()
    }
}

/// The brain: the GGUF file at `path`, four sequences at once, greedy
pub fn brain(path: &str, calls: Arc<Mutex<u64>>) -> Arc<dyn Neuron> {
    let engine = furnace_candle::Gguf::new(Path::new(path))
        .name("brain")
        .max_sequences(4)
        .max_context(4096)
        .load()
        .unwrap_or_else(|e| fail(&format!("loading {path}: {e}")));
    Arc::new(Counted {
        brain: furnace::nerve::expect::llm::brain("brain", Arc::new(engine)),
        calls,
    })
}
