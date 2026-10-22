//! How a signal moves: the gate, the firing, and the pathway its timing selects

use std::sync::atomic::Ordering;
use std::time::Instant;

use tracing::{debug, warn};

use super::{Cell, Fabric, MAX_HOPS, Outcome, Pathway};
use crate::nerve::latency::Latency;
use crate::nerve::motor::MotionError;
use crate::nerve::neuron::{Cx, Firing, NeuronKind, Quiet};
use crate::nerve::signal::{NeuronId, PathwayId, Signal};

impl Fabric {
    /// Hand a signal to a receptor: the whole synchronous chain runs before this returns.
    /// Receptors call this through their [`Afferent`](crate::nerve::receptor::Afferent)
    /// handle; tests and in-process sources call it directly.
    pub fn stimulate(&self, receptor: NeuronId, signal: Signal) -> Outcome {
        self.stimulate_announced(receptor, signal, None)
    }

    /// Announce a signal that will take `pathway` once it is ready (its request's body is
    /// still being received or decoded): a worker collecting a batch there waits for it.
    /// Hold the guard while preparing the signal, then hand both to
    /// [`Fabric::stimulate_announced`]. `None` for a reflex pathway, which has no queue.
    pub(crate) fn announce(&self, pathway: PathwayId) -> Option<Announced<'_>> {
        let queue = self.0.pathways.get(pathway.index())?.queue.as_ref()?;
        Some(Announced {
            pathway,
            arrival: queue.arrival(),
        })
    }

    /// [`Fabric::stimulate`] for a signal announced with [`Fabric::announce`]: if it takes
    /// the announced pathway, it is queued as the arrival a worker may be waiting for
    pub(crate) fn stimulate_announced(
        &self,
        receptor: NeuronId,
        mut signal: Signal,
        announced: Option<Announced<'_>>,
    ) -> Outcome {
        signal.origin = receptor;
        self.0.received.fetch_add(1, Ordering::Relaxed);
        let mut announced = announced;
        self.drive(receptor, signal, 0, &mut announced, Instant::now())
    }

    /// [`Fabric::stimulate`] by the receptor's name
    pub fn stimulate_named(&self, receptor: &str, signal: Signal) -> Option<Outcome> {
        Some(self.stimulate(self.neuron(receptor)?, signal))
    }

    /// Run `id`'s gate and `stimulate`, then propagate what it fires; `now` is when the
    /// signal arrived, which the caller has read already
    fn drive(
        &self,
        id: NeuronId,
        mut signal: Signal,
        depth: usize,
        announced: &mut Option<Announced<'_>>,
        now: Instant,
    ) -> Outcome {
        let cell = &self.0.cells[id.index()];
        let nanos = self.nanos(now);
        cell.counters.received.fetch_add(1, Ordering::Relaxed);
        if let Err(reason) = cell.gate.admit(&mut signal, nanos) {
            cell.counters.quiet(reason);
            signal.report(Err(MotionError::failed(format!(
                "quiet at '{}': {}",
                cell.neuron.name(),
                reason.as_str()
            ))));
            return Outcome {
                quiet: Some((id, reason)),
                hops: depth,
                ..Outcome::default()
            };
        }
        signal.trace.push(id);
        cell.taps.copy(&signal);
        let mut acted = 0;
        match cell.kind {
            NeuronKind::Brain => {
                signal.trace.mark_brain();
                self.0.reached_brain.fetch_add(1, Ordering::Relaxed);
            }
            NeuronKind::Motor => acted = 1,
            NeuronKind::Effector => {
                acted = 1;
                if !signal.trace.was_queued()
                    && let Some(arc) = cell.arcs.get(signal.origin.index())
                {
                    arc.get_or_init(Latency::new)
                        .observe(now.saturating_duration_since(signal.at));
                }
            }
            NeuronKind::Receptor | NeuronKind::Reflex => {}
        }
        let cx = Cx {
            now,
            id,
            fabric: self,
        };
        let firing = cell.neuron.stimulate(signal, &cx);
        if matches!(firing, Firing::Quiet) {
            cell.gate.stayed_quiet(nanos);
        }
        let mut outcome = self.after_firing(id, firing, depth, announced);
        outcome.acted += acted;
        outcome
    }

    /// Propagate what `id` fired
    pub(super) fn after_firing(
        &self,
        id: NeuronId,
        firing: Firing,
        depth: usize,
        announced: &mut Option<Announced<'_>>,
    ) -> Outcome {
        let cell = &self.0.cells[id.index()];
        let emitted = match firing {
            Firing::Quiet => {
                cell.counters.quiet(Quiet::NotInterested);
                return Outcome {
                    quiet: Some((id, Quiet::NotInterested)),
                    hops: depth + 1,
                    ..Outcome::default()
                };
            }
            Firing::Fire(signals) => signals,
            Firing::Compute(work) => {
                cell.counters.computed.fetch_add(1, Ordering::Relaxed);
                work()
            }
        };
        cell.counters.fired.fetch_add(1, Ordering::Relaxed);
        let mut outcome = Outcome {
            hops: depth + 1,
            ..Outcome::default()
        };
        for signal in emitted {
            outcome.merge(self.route(id, signal, depth + 1, announced));
        }
        outcome
    }

    /// Fire `signal` from `from` as if `from` had fired it: a motor program's commands,
    /// questions and reports leave its motor neuron this way
    pub(crate) fn emit(&self, from: NeuronId, signal: Signal) -> Outcome {
        self.route(from, signal, 1, &mut None)
    }

    /// Send a signal `from` fired along the pathway its timing selects
    fn route(
        &self,
        from: NeuronId,
        mut signal: Signal,
        depth: usize,
        announced: &mut Option<Announced<'_>>,
    ) -> Outcome {
        // The depth restarts after a deliberative queue; the trace keeps the whole count, so a
        // loop through a queue is cut too
        if depth >= MAX_HOPS || signal.trace.len() >= MAX_HOPS {
            self.0.loops.fetch_add(1, Ordering::Relaxed);
            warn!(
                "nerve: signal {} dropped after {MAX_HOPS} hops (a loop?)",
                signal.id.0
            );
            signal.report(Err(MotionError::failed(format!(
                "dropped after {MAX_HOPS} hops (a loop?)"
            ))));
            return Outcome::default();
        }
        // What a neuron fires in answer to a motor neuron's question goes back to it
        if signal.answers_from(from) {
            return Outcome::default();
        }
        let now = Instant::now();
        let cell = &self.0.cells[from.index()];
        let selected = match signal.payload.as_command() {
            // A command names its effector or motor neuron; the pathway to it is the one
            // (declared or not)
            Some(command) => match self.0.by_name.get(&command.effector) {
                Some(&target)
                    if matches!(
                        self.0.cells[target.index()].kind,
                        NeuronKind::Effector | NeuronKind::Motor
                    ) =>
                {
                    match cell
                        .out
                        .iter()
                        .map(|&p| &self.0.pathways[p.index()])
                        .find(|p| p.to == target)
                    {
                        Some(pathway) => Some(pathway),
                        None => {
                            self.0.commands_direct.fetch_add(1, Ordering::Relaxed);
                            return self.drive(target, signal, depth, announced, now);
                        }
                    }
                }
                _ => {
                    self.0.unrouted.fetch_add(1, Ordering::Relaxed);
                    debug!(
                        "nerve: no effector or motor neuron named '{}' for a command from {}",
                        command.effector,
                        cell.neuron.name()
                    );
                    signal.report(Err(MotionError::Unrouted(command.effector.clone())));
                    return Outcome {
                        unrouted: 1,
                        ..Outcome::default()
                    };
                }
            },
            None => self.select(cell, &signal, now),
        };
        let Some(pathway) = selected else {
            if cell.out.is_empty() {
                // A terminal neuron: nothing to do with what it fired
                return Outcome::default();
            }
            self.0.no_eligible.fetch_add(1, Ordering::Relaxed);
            return Outcome {
                no_pathway: 1,
                ..Outcome::default()
            };
        };
        pathway.signals.fetch_add(1, Ordering::Relaxed);
        match &pathway.queue {
            None => {
                let outcome = self.drive(pathway.to, signal, depth, announced, now);
                pathway.latency.observe(now.elapsed());
                outcome
            }
            Some(queue) => {
                self.prepare_for(pathway, &signal, now);
                signal.trace.mark_queued();
                // The arrival a worker may be waiting for, if this is the announced pathway
                let arrival = match announced.take() {
                    Some(a) if a.pathway == pathway.id => Some(a.arrival),
                    other => {
                        *announced = other;
                        None
                    }
                };
                match queue.submit(arrival, signal) {
                    Ok(()) => Outcome {
                        queued: 1,
                        ..Outcome::default()
                    },
                    // Full: dropped rather than queued without bound
                    Err(signal) => {
                        pathway.overflow.fetch_add(1, Ordering::Relaxed);
                        signal.report(Err(MotionError::failed("the pathway's queue is full")));
                        Outcome {
                            dropped: 1,
                            ..Outcome::default()
                        }
                    }
                }
            }
        }
    }

    /// The short-mode / long-mode rule: among the pathways leaving `cell` whose p99 fits in
    /// the signal's remaining deadline, the heaviest weight
    fn select(&self, cell: &Cell, signal: &Signal, now: Instant) -> Option<&Pathway> {
        let remaining = signal.urgency.remaining(now);
        let mut best: Option<&Pathway> = None;
        for &id in &cell.out {
            let pathway = &self.0.pathways[id.index()];
            if !pathway.fits(remaining) {
                pathway.ineligible.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if best.is_none_or(|b| pathway.weight > b.weight) {
                best = Some(pathway);
            }
        }
        best
    }

    /// Motor preparation: a signal with time to spare that takes a deliberative pathway
    /// warms the neurons it will probably reach, at most once a second each
    fn prepare_for(&self, pathway: &Pathway, signal: &Signal, now: Instant) {
        let far = match (signal.urgency.remaining(now), pathway.latency.p99()) {
            (None, _) | (_, None) => true,
            (Some(remaining), Some(p99)) => remaining > 2 * p99,
        };
        if !far {
            return;
        }
        let nanos = self.nanos(now);
        let to = &self.0.cells[pathway.to.index()];
        let downstream = to
            .out
            .iter()
            .map(|&p| self.0.pathways[p.index()].to)
            .chain(std::iter::once(pathway.to));
        for id in downstream {
            let cell = &self.0.cells[id.index()];
            let last = cell.last_prepared.load(Ordering::Relaxed);
            if nanos.saturating_sub(last) < 1_000_000_000 && last != 0 {
                continue;
            }
            cell.last_prepared.store(nanos.max(1), Ordering::Relaxed);
            let cx = Cx {
                now,
                id,
                fabric: self,
            };
            cell.neuron.prepare(&cx);
        }
    }
}

/// A signal announced on a deliberative pathway, see [`Fabric::announce`]
pub(crate) struct Announced<'a> {
    pathway: PathwayId,
    arrival: amate::Arrival<'a, Signal>,
}
