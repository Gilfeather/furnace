//! A deliberative pathway's worker: summing a window of signals into one firing

use std::sync::Weak;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::{Fabric, Inner};
use crate::nerve::motor::MotionError;
use crate::nerve::neuron::{Cx, Firing, NeuronKind};
use crate::nerve::signal::{PathwayId, ReplyHandle, Signal};
use amate::{Batching, Queued};

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
            item: signal,
            queued_at: at,
        } in batch
        {
            // An act takes its nerve's lane, as on any other way in; one that waits there is
            // carried on from it
            let mut signal = match self.enter(pathway.to, signal) {
                Ok(signal) => signal,
                Err(_) => continue,
            };
            cell.counters.received.fetch_add(1, Ordering::Relaxed);
            match cell.gate.admit(&mut signal, nanos) {
                Ok(()) => {
                    signal.trace.push(pathway.to);
                    cell.taps.copy(&signal);
                    if cell.kind == NeuronKind::Brain {
                        signal.trace.mark_brain();
                        self.0.reached_brain.fetch_add(1, Ordering::Relaxed);
                    }
                    admitted.push(signal);
                    queued_at.push(at);
                }
                Err(reason) => {
                    cell.counters.quiet(reason);
                    signal.report(Err(MotionError::failed(format!(
                        "quiet at '{}': {}",
                        cell.neuron.name(),
                        reason.as_str()
                    ))));
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

    /// A command is skipped once whoever sent it stops waiting, and not before: it is an act
    /// to carry out, whether or not the request behind it is still waited for. Any other
    /// signal that exists to answer a request nobody waits for any more is skipped.
    fn wanted(&self, signal: &Signal) -> bool {
        if signal.payload.as_command().is_some() {
            return !signal.abandoned();
        }
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
