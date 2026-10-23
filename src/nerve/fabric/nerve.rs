//! Nerves: the trunks that carry signals into one part of the body (the right arm, the left
//! leg), and the spine between the brain and them.
//!
//! A nerve's lane counts the acts of its effectors, nothing else: every act entering the
//! nerve takes a place in the lane, whoever commanded it, from inside the part or outside.
//! At most [`Nerve::in_flight`] acts are carried out at once, and the rest wait, the most
//! urgent (the nearest deadline) first. An idle lane costs nothing: the act goes straight
//! on. A part that is flooded makes its own acts wait and no other part's. An act waits for
//! no other command and always ends, so the places it holds always come back. A motor
//! neuron's program never takes a place and never waits in a lane: one runs per motor
//! neuron, and a newer command stops it at once.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{Fabric, Inner, Outcome};
use crate::nerve::motor::MotionError;
use crate::nerve::neuron::NeuronKind;
use crate::nerve::signal::{NeuronId, Signal};
use crate::nerve::sync::{Mutex, MutexGuard};

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
    /// A nerve named `name` that carries out 16 acts at once
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            in_flight: 16,
        }
    }

    /// Acts carried out at once (at least 1): an act counts until it ends, retries included
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

struct LaneState<T> {
    running: usize,
    waiting: BTreeMap<Key, T>,
    sequence: u64,
}

/// A nerve at run time: its lane and counters. What waits in it (a signal and the neuron it
/// goes to) is handed back out of the lane's lock whenever it leaves, so that whatever
/// dropping or answering it does never runs under the lock.
pub(crate) struct Lane<T = (NeuronId, Signal)> {
    pub(crate) name: String,
    limit: usize,
    /// Most waiting at once
    room: usize,
    state: Mutex<LaneState<T>>,
    waited: AtomicU64,
    dropped: AtomicU64,
}

/// What became of something entering a lane
pub(crate) enum Entered<T> {
    /// A place was free and it took it: carry it on
    Go(T),
    /// It waits for a place; the least urgent of those waiting was dropped for it, when the
    /// lane was full
    Waits { dropped: Option<T> },
    /// The lane is full of more urgent ones: it is dropped
    Full(T),
}

/// What freeing a place in a lane let go of
pub(crate) struct Released<T> {
    /// The most urgent waiting that is still due and wanted: it took the place
    pub(crate) next: Option<T>,
    /// Those whose deadline passed while they waited
    pub(crate) expired: Vec<T>,
    /// Those nobody wanted any more
    pub(crate) unwanted: Vec<T>,
}

impl<T> Lane<T> {
    pub(super) fn new(nerve: Nerve) -> Self {
        Self::with_room(nerve, WAITING)
    }

    fn with_room(nerve: Nerve, room: usize) -> Self {
        Self {
            name: nerve.name,
            limit: nerve.in_flight,
            room,
            state: Mutex::new(LaneState {
                running: 0,
                waiting: BTreeMap::new(),
                sequence: 0,
            }),
            waited: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    fn state(&self) -> MutexGuard<'_, LaneState<T>> {
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

    /// Take a place for `item`, due by `deadline`, or wait for one. An idle lane lets it
    /// straight on.
    fn enter(&self, deadline: Option<Instant>, item: T) -> Entered<T> {
        let mut state = self.state();
        if state.running < self.limit && state.waiting.is_empty() {
            state.running += 1;
            return Entered::Go(item);
        }
        state.sequence += 1;
        let key = Key {
            whenever: deadline.is_none(),
            deadline,
            sequence: state.sequence,
        };
        let mut dropped = None;
        if state.waiting.len() >= self.room {
            match state.waiting.last_key_value().map(|(k, _)| *k) {
                // The newcomer is more urgent than the least urgent waiting: that one goes
                Some(least) if key < least => {
                    dropped = state.waiting.pop_last().map(|(_, item)| item);
                }
                _ => {
                    drop(state);
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    return Entered::Full(item);
                }
            }
        }
        state.waiting.insert(key, item);
        drop(state);
        if dropped.is_some() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        self.waited.fetch_add(1, Ordering::Relaxed);
        Entered::Waits { dropped }
    }

    /// A place is free: the most urgent waiting that is still due at `now` and `wanted`
    /// takes it
    fn release(&self, now: Instant, wanted: impl Fn(&T) -> bool) -> Released<T> {
        let mut released = Released {
            next: None,
            expired: Vec::new(),
            unwanted: Vec::new(),
        };
        let mut state = self.state();
        state.running = state.running.saturating_sub(1);
        while state.running < self.limit {
            let Some((key, item)) = state.waiting.pop_first() else {
                break;
            };
            if key.deadline.is_some_and(|d| d <= now) {
                released.expired.push(item);
            } else if !wanted(&item) {
                released.unwanted.push(item);
            } else {
                state.running += 1;
                released.next = Some(item);
                break;
            }
        }
        drop(state);
        self.dropped
            .fetch_add(released.expired.len() as u64, Ordering::Relaxed);
        released
    }
}

/// A nerve's lane in `GET /metrics` and `GET /nerve`
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NerveLaneMetrics {
    /// The nerve's name
    pub name: String,
    /// Acts being carried out now
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

/// A place in a nerve's lane, held by the act that took it (every copy of its signal, its
/// retries' included): the place is free again once the act ends
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
    /// Let `signal` into `to`'s nerve: `Ok` to carry it on now, or what became of it
    /// (waiting in the lane, or dropped, which fails it). Only an effector's act takes the
    /// lane; signals to any other neuron, and to neurons in no nerve, always go on.
    pub(super) fn enter(&self, to: NeuronId, mut signal: Signal) -> Result<Signal, Outcome> {
        let cell = &self.0.cells[to.index()];
        let Some(nerve) = cell.nerve.filter(|_| cell.kind == NeuronKind::Effector) else {
            return Ok(signal);
        };
        // Its place is its own: one a signal it derives from held is not carried in
        signal.ticket = None;
        let lane = &self.0.nerves[nerve];
        match lane.enter(signal.urgency.deadline, (to, signal)) {
            Entered::Go((_, mut signal)) => {
                signal.ticket = Some(self.ticket(nerve));
                Ok(signal)
            }
            Entered::Waits { dropped } => {
                if let Some((_, dropped)) = dropped {
                    dropped.report(Err(full(&lane.name)));
                }
                Err(Outcome {
                    queued: 1,
                    ..Outcome::default()
                })
            }
            Entered::Full((_, signal)) => {
                signal.report(Err(full(&lane.name)));
                Err(Outcome {
                    dropped: 1,
                    ..Outcome::default()
                })
            }
        }
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
        let released = lane.release(Instant::now(), |(_, signal)| !signal.abandoned());
        for (_, signal) in released.expired {
            signal.report(Err(MotionError::TimedOut));
        }
        if let Some((to, mut signal)) = released.next {
            signal.ticket = Some(self.ticket(nerve));
            let fabric = self.clone();
            self.efferent().spawn(async move {
                fabric.deliver(to, signal);
            });
        }
    }
}

fn full(nerve: &str) -> MotionError {
    MotionError::failed(format!("the nerve '{nerve}' is full"))
}

#[cfg(all(test, furnace_loom))]
mod model_check;

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    /// Locks its lane when it is dropped, as a signal holding the last of another place in
    /// the lane does when it frees that place
    struct Probe(Arc<Lane<Probe>>);

    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.metrics();
        }
    }

    /// What leaves a lane leaves it out of the lane's lock: dropping it may lock the lane
    /// again (free a place in it) without deadlocking
    #[test]
    fn what_leaves_a_lane_is_dropped_out_of_its_lock() {
        let lane = Arc::new(Lane::with_room(Nerve::new("arm").in_flight(1), 1));
        let probe = || Probe(lane.clone());
        let soon = Some(Instant::now() + Duration::from_secs(3600));
        let later = Some(Instant::now() + Duration::from_secs(7200));
        let Entered::Go(running) = lane.enter(None, probe()) else {
            panic!("an idle lane lets it on");
        };
        assert!(matches!(
            lane.enter(later, probe()),
            Entered::Waits { dropped: None }
        ));
        // More urgent: the one waiting is dropped for it, and handed back
        let Entered::Waits { dropped } = lane.enter(soon, probe()) else {
            panic!("the more urgent one waits");
        };
        drop(dropped.expect("the least urgent is dropped"));
        // Less urgent than all waiting: handed back
        let Entered::Full(full) = lane.enter(later, probe()) else {
            panic!("the lane is full");
        };
        drop(full);
        drop(running);
        // Freed after its deadline, the one waiting expired and is handed back too
        let released = lane.release(Instant::now() + Duration::from_secs(4000), |_| true);
        assert!(released.next.is_none());
        assert_eq!(released.expired.len(), 1);
        drop(released);
        let metrics = lane.metrics();
        assert_eq!(
            (metrics.in_flight, metrics.waiting, metrics.dropped),
            (0, 0, 3)
        );
    }
}
