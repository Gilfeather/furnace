//! Nerves: the trunks that carry signals into one part of the body (the right arm, the left
//! leg), and the spine between the brain and them.
//!
//! What enters a nerve from outside it (a command from the brain's side, from another part)
//! takes the nerve's lane: at most [`Nerve::in_flight`] of them are carried out at once, and
//! the rest wait, the most urgent (the nearest deadline) first. An idle lane costs nothing:
//! the signal goes straight on. A part that is flooded makes its own signals wait and no
//! other part's. What stays inside the nerve (the hand commanding its fingers, a fingertip's
//! pressure reaching the hand) never takes the lane: the part's own loops close there.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{Fabric, Inner, Outcome};
use crate::nerve::motor::MotionError;
use crate::nerve::signal::{NeuronId, Signal};

/// Signals that may wait in a nerve's lane; beyond it, the least urgent is dropped
const WAITING: usize = 1024;

/// A nerve: the trunk to one part of the body. Give it to
/// [`FabricBuilder::nerve`](super::FabricBuilder::nerve), or just its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nerve {
    pub(super) name: String,
    pub(super) in_flight: usize,
}

impl Nerve {
    /// A nerve named `name` that carries out 16 signals from outside at once
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            in_flight: 16,
        }
    }

    /// Signals from outside carried out at once (at least 1): a command taken counts until
    /// its act or program ends, everything it commands inside the nerve included
    pub fn in_flight(mut self, n: usize) -> Self {
        self.in_flight = n.max(1);
        self
    }
}

impl From<&str> for Nerve {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

/// Urgency order: the nearest deadline first, then no deadline, each in order of arrival
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    /// `false` with a deadline, so that those come first
    whenever: bool,
    deadline: Option<Instant>,
    sequence: u64,
}

struct LaneState {
    running: usize,
    waiting: BTreeMap<Key, (NeuronId, Signal)>,
    sequence: u64,
}

/// A nerve at run time: its lane and counters
pub(crate) struct Lane {
    pub(crate) name: String,
    limit: usize,
    state: Mutex<LaneState>,
    waited: AtomicU64,
    dropped: AtomicU64,
}

impl Lane {
    pub(super) fn new(nerve: Nerve) -> Self {
        Self {
            name: nerve.name,
            limit: nerve.in_flight,
            state: Mutex::new(LaneState {
                running: 0,
                waiting: BTreeMap::new(),
                sequence: 0,
            }),
            waited: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, LaneState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn metrics(&self) -> NerveLaneMetrics {
        let state = self.state();
        NerveLaneMetrics {
            name: self.name.clone(),
            in_flight: state.running as u64,
            limit: self.limit as u64,
            waiting: state.waiting.len() as u64,
            waited: self.waited.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

/// A nerve's lane in `GET /metrics` and `GET /nerve`
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NerveLaneMetrics {
    /// The nerve's name
    pub name: String,
    /// Signals from outside being carried out now
    pub in_flight: u64,
    /// How many may be at once
    pub limit: u64,
    /// Signals waiting for a place now
    pub waiting: u64,
    /// Signals that had to wait, in all
    pub waited: u64,
    /// Signals dropped: the lane was full, or their deadline passed while they waited
    pub dropped: u64,
}

/// A place in a nerve's lane, held by the signal that took it and every signal derived from
/// it: the place is free again once the last of them is gone
#[derive(Clone)]
pub(crate) struct Ticket(Arc<Held>);

struct Held {
    fabric: Weak<Inner>,
    nerve: usize,
}

impl Drop for Held {
    fn drop(&mut self) {
        if let Some(inner) = self.fabric.upgrade() {
            Fabric(inner).release(self.nerve);
        }
    }
}

impl std::fmt::Debug for Ticket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ticket({})", self.0.nerve)
    }
}

impl Fabric {
    /// Let `signal` from `from` into `to`'s nerve: `Ok` to carry it on now, or what became of
    /// it (waiting in the lane, or dropped). Signals within one nerve, and to neurons in no
    /// nerve, always go on.
    pub(super) fn enter(
        &self,
        from: NeuronId,
        to: NeuronId,
        mut signal: Signal,
    ) -> Result<Signal, Outcome> {
        let Some(nerve) = self.0.cells[to.index()].nerve else {
            return Ok(signal);
        };
        if self.0.cells[from.index()].nerve == Some(nerve) {
            return Ok(signal);
        }
        let lane = &self.0.nerves[nerve];
        let mut state = lane.state();
        if state.running < lane.limit && state.waiting.is_empty() {
            state.running += 1;
            drop(state);
            signal.ticket = Some(self.ticket(nerve));
            return Ok(signal);
        }
        state.sequence += 1;
        let key = Key {
            whenever: signal.urgency.deadline.is_none(),
            deadline: signal.urgency.deadline,
            sequence: state.sequence,
        };
        if state.waiting.len() >= WAITING {
            let least = state.waiting.last_key_value().map(|(k, _)| *k);
            match least {
                // The newcomer is more urgent than the least urgent waiting: that one goes
                Some(least) if key < least => {
                    if let Some((_, (_, dropped))) = state.waiting.pop_last() {
                        dropped.report(Err(full(&lane.name)));
                    }
                }
                _ => {
                    drop(state);
                    lane.dropped.fetch_add(1, Ordering::Relaxed);
                    signal.report(Err(full(&lane.name)));
                    return Err(Outcome {
                        dropped: 1,
                        ..Outcome::default()
                    });
                }
            }
            lane.dropped.fetch_add(1, Ordering::Relaxed);
        }
        state.waiting.insert(key, (to, signal));
        lane.waited.fetch_add(1, Ordering::Relaxed);
        Err(Outcome {
            queued: 1,
            ..Outcome::default()
        })
    }

    fn ticket(&self, nerve: usize) -> Ticket {
        Ticket(Arc::new(Held {
            fabric: Arc::downgrade(&self.0),
            nerve,
        }))
    }

    /// A place in `nerve`'s lane is free: the most urgent signal waiting takes it, on the
    /// motor side's runtime
    fn release(&self, nerve: usize) {
        let lane = &self.0.nerves[nerve];
        let mut state = lane.state();
        state.running = state.running.saturating_sub(1);
        let now = Instant::now();
        while state.running < lane.limit {
            let Some((_, (to, mut signal))) = state.waiting.pop_first() else {
                return;
            };
            if signal.urgency.deadline.is_some_and(|d| d <= now) {
                lane.dropped.fetch_add(1, Ordering::Relaxed);
                signal.report(Err(MotionError::TimedOut));
                continue;
            }
            if signal.abandoned() {
                continue;
            }
            state.running += 1;
            drop(state);
            signal.ticket = Some(self.ticket(nerve));
            let fabric = self.clone();
            self.efferent().spawn(async move {
                fabric.deliver(to, signal);
            });
            return;
        }
    }
}

fn full(nerve: &str) -> MotionError {
    MotionError::failed(format!("the nerve '{nerve}' is full"))
}
