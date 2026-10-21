//! A deliberative pathway's worker: summing a window of signals into one firing

use std::sync::Weak;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::{Fabric, Inner};
use crate::nerve::neuron::{Cx, Firing, NeuronKind};
use crate::nerve::signal::{PathwayId, ReplyHandle, Signal};
use furnace_batch::{Batching, Queued};

impl Fabric {
    /// Fire `to` with a batch a worker summed, and propagate the results
    fn fire_summed(&self, pathway: PathwayId, batch: Vec<Queued<Signal>>) {
        let pathway = &self.0.pathways[pathway.index()];
        let cell = &self.0.cells[pathway.to.index()];
        let now = Instant::now();
        let nanos = self.nanos(now);
        let mut admitted = Vec::with_capacity(batch.len());
        let mut queued_at = Vec::with_capacity(batch.len());
        for Queued {
            item: mut signal,
            queued_at: at,
        } in batch
        {
            cell.counters.received.fetch_add(1, Ordering::Relaxed);
            match cell.gate.admit(&mut signal, nanos) {
                Ok(()) => {
                    signal.trace.push(pathway.to);
                    if cell.kind == NeuronKind::Brain {
                        signal.trace.mark_brain();
                        self.0.reached_brain.fetch_add(1, Ordering::Relaxed);
                    }
                    admitted.push(signal);
                    queued_at.push(at);
                }
                Err(reason) => {
                    cell.counters.quiet(reason);
                    pathway.latency.observe(now.saturating_duration_since(at));
                }
            }
        }
        if admitted.is_empty() {
            return;
        }
        let cx = Cx {
            now,
            id: pathway.to,
            fabric: self,
        };
        let firings = cell.neuron.stimulate_many(admitted, &cx);
        if firings.iter().all(|f| matches!(f, Firing::Quiet)) {
            cell.gate.stayed_quiet(nanos);
        }
        for (firing, at) in firings.into_iter().zip(queued_at) {
            self.after_firing(pathway.to, firing, 1, &mut None);
            pathway.latency.observe(at.elapsed());
        }
    }
}

/// A deliberative pathway's summation: the batching engine over the pathway's queue, with
/// the window learned from the pathway's latency, firing the pathway's neuron with each
/// batch and propagating what it fires
pub(super) struct Summing {
    pub(super) fabric: Weak<Inner>,
    pub(super) pathway: PathwayId,
}

impl Batching for Summing {
    type Item = Signal;

    /// A signal of tensors counts its items (the batch dimension); any other counts one
    fn weight(&self, signal: &Signal) -> usize {
        signal
            .payload
            .as_tensors()
            .and_then(<[_]>::first)
            .map_or(1, |t| t.batch().max(1))
    }

    /// A signal that exists to answer a request nobody waits for any more is skipped
    fn wanted(&self, signal: &Signal) -> bool {
        signal.reply.as_ref().is_none_or(ReplyHandle::wanted)
    }

    fn window(&self) -> Duration {
        self.fabric
            .upgrade()
            .and_then(|inner| inner.pathways[self.pathway.index()].window())
            .unwrap_or(Duration::ZERO)
    }

    fn run(&self, batch: Vec<Queued<Signal>>) {
        if let Some(inner) = self.fabric.upgrade() {
            Fabric(inner).fire_summed(self.pathway, batch);
        }
    }

    fn panicked(&self, weight: usize) {
        tracing::error!("A batch of {weight} item(s) panicked; its results are lost");
    }
}
