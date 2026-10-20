//! Consolidation, level 1: a verified cache of the brain's decisions.
//!
//! The fabric watches what the brain decides for each kind of signal (level 1: the signal's
//! [`Feature`], exactly, or a key of your own). Once the brain has decided the same thing for
//! the same key [`Consolidation::after`] times in a row, the decision is an *engram*, and the
//! next signal with that key is answered by [`Recall`] in front of the brain, in microseconds
//! and without it. A sample of those answers is still sent to the brain after the reflex
//! acted (a *shadow check*); the brain's answer is compared, never acted on, and an engram
//! the brain disagrees with is dissolved. So is one whose effector keeps failing.
//!
//! This is a cache with verification, and the foundation for what comes later (keys by
//! similarity, expectations the brain writes into the periphery), not more than that.
//!
//! ```text
//! receptor ──► Recall ──(consolidated engram)──────────────────────────► effectors
//!                │
//!                └──(miss, or shadow check)──► Record(brain) ──(commands)──► effectors
//!                     deliberative pathway          │
//!                                                   ▼
//!                                         Memory (EngramStore)
//! changes ──► Inhibit ──(tags, dependency versions)──┘
//! ```
//!
//! **What a decision may depend on.** A brain without [`Consolidation::depends_on`] is taken
//! to decide from its input alone: the same key, the same answer. A brain that consults
//! external state (a database table) declares it, and every change of that state must reach
//! the store, either as a tag (`customer:42`, [`Memory::inhibit`]) that dissolves the engrams
//! carrying it at once, or as a new version of the dependency ([`Memory::advance`]) that
//! invalidates every engram made under the old one. [`Inhibit`] does both from change
//! notifications arriving at a receptor. Forgetting ([`Consolidation::forget_after`]) and the
//! shadow checks are the last line against a dependency nobody declared, not the mechanism.
//!
//! [`Record`] wraps the brain rather than following it: a brain's commands are routed by
//! name straight to their effectors, so a neuron on a pathway behind the brain would never
//! see them. Wrapping it sees each firing next to the signal that caused it, which is the
//! decision.
//!
//! [`Recall`], [`Record`], [`Inhibit`] and [`Watched`] talk to a [`Memory`], a trait:
//! [`EngramStore`] is the one built in (an exact key and a list of commands), and a memory
//! with other keys or other contents can take its place.
//!
//! ```no_run
//! use std::time::Duration;
//! use furnace::nerve::consolidation::{Consolidation, EngramStore, Inhibit};
//! use furnace::nerve::{effector, Fabric, Firing, FnNeuron, Mechanisms, NeuronKind, Summation};
//!
//! let engrams = EngramStore::new(
//!     Consolidation::new()
//!         .effectors(["ticket", "ignore"])
//!         .after(3)
//!         .shadow(0.05)
//!         .depends_on(["customers"])
//!         .tags(|s| s.payload.as_text().map(|t| vec![format!("customer:{t}")]).unwrap_or_default())
//!         .forget_after(Duration::from_secs(3600)),
//! );
//! let mut b = Fabric::builder();
//! let (ticket, _deliveries) = effector::ChannelOut::new("ticket", 64);
//! // Watched: the effector's failures reach the engrams it acts for
//! b.neuron(effector::Effector::new(engrams.watch(ticket)), Mechanisms::NONE);
//! let brain = FnNeuron::reflex("brain", |s, _| Firing::one(s.command("ticket", serde_json::json!({}))))
//!     .with_kind(NeuronKind::Brain);
//! let (recall, _brain) = b.consolidate(brain, Mechanisms::NONE, Summation::default(), &engrams);
//! // Change notifications ({"inhibit": ["customer:42"]}, {"advance": ["customers"]}) from a
//! // receptor of their own
//! let changes = b.neuron(Inhibit::new("changes", &engrams), Mechanisms::NONE);
//! // ... b.reflex(receptor, recall, 1.0); b.reflex(change_receptor, changes, 1.0)
//! let _ = (recall, changes);
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tracing::warn;

use super::effector::{Act, Action};
use super::fabric::{FabricBuilder, Summation};
use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Mechanisms, Neuron, NeuronKind};
use super::signal::{Command, Feature, NeuronId, Payload, Signal, SignalId, sketch64};

/// Entries per set of the engram store, as in the habituation table
const WAYS: usize = 4;

/// Signals per window of the brain-fraction timeline
pub const WINDOW: u64 = 1000;

/// Windows of the timeline kept
const WINDOWS: usize = 16;

/// Engrams shown by `GET /nerve`
const TOP: usize = 8;

// ---------------------------------------------------------------------------------------------
// The memory: what Recall, Record, Inhibit and Watched need of a store

/// What a decision is filed under, worked out from the signal before the brain sees it
#[derive(Clone, Debug, PartialEq)]
pub struct Cause {
    /// The memory's key for the signal
    pub key: Feature,
    /// The signal's id
    pub id: SignalId,
    /// When the receptor saw the stimulus
    pub at: Instant,
    /// What the decision is about, hashed ([`Memory::inhibit`] by one of them)
    pub tags: Vec<u64>,
    /// The versions of the declared dependencies the decision is made under
    pub generation: u64,
}

/// What a [`Memory`] answers a signal with
#[derive(Debug, Clone)]
pub enum Recalled {
    /// Ask the brain
    Miss,
    /// Fire these; with `shadow`, also send the signal to the brain to check
    Hit {
        /// The remembered decision's signals
        fire: Vec<Signal>,
        /// Whether this answer is a shadow check
        shadow: bool,
    },
}

impl Recalled {
    /// Whether an engram answered
    pub fn is_hit(&self) -> bool {
        matches!(self, Self::Hit { .. })
    }
}

/// Where the brain's decisions are kept and looked up: the part of consolidation that
/// decides what a key is and what is stored. [`EngramStore`] is the one built in.
pub trait Memory: Send + Sync + 'static {
    /// The key, tags and dependency versions a signal's decision is filed under
    fn cause(&self, signal: &Signal) -> Cause;

    /// Look a signal up before the brain: fire a remembered decision, or ask the brain
    fn recall(&self, signal: &Signal, cause: &Cause, now: Instant) -> Recalled;

    /// The brain fired `fired` for the signal of `cause`. `true`: act on it; `false`: it was
    /// a shadow check, compared and not to be acted on.
    fn decided(&self, cause: &Cause, fired: &[Signal], now: Instant) -> bool;

    /// An effector acted for a signal whose feature is `key` (a fired decision carries its
    /// key as its feature), and succeeded or failed
    fn outcome(&self, _key: Feature, _ok: bool, _now: Instant) {}

    /// The state tagged `tag` changed: forget what was decided about it. Returns how many
    /// remembered decisions that dissolved.
    fn inhibit(&self, _tag: &str, _now: Instant) -> usize {
        0
    }

    /// The declared dependency `name` changed: every decision made under its old version is
    /// void. Returns its new version, `None` if it was not declared.
    fn advance(&self, _dependency: &str, _now: Instant) -> Option<u64> {
        None
    }

    /// For `GET /nerve`
    fn describe(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// For `/metrics`
    fn metrics(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
}

// ---------------------------------------------------------------------------------------------
// The policy

type KeyFn = Arc<dyn Fn(&Signal) -> Feature + Send + Sync>;
type TagsFn = Arc<dyn Fn(&Signal) -> Vec<String> + Send + Sync>;

/// When and what the brain's decisions may become reflexes: the policy of an [`EngramStore`]
#[derive(Clone)]
pub struct Consolidation {
    /// Effectors whose commands may be answered from an engram; nothing else ever is
    pub effectors: Vec<String>,
    /// Agreeing brain decisions, in a row, before a key's decision is consolidated. After
    /// each dissolution of the same key by the brain or by failures it takes this many more.
    pub after: u32,
    /// Fraction of reflex answers that are also sent to the brain to check, 0..=1
    pub shadow: f64,
    /// The first answers of every engram are always checked
    pub first_shadows: u32,
    /// An engram not seen for this long is forgotten
    pub forget_after: Duration,
    /// Failures in a row of an effector acting for an engram that dissolve it
    pub failures: u32,
    /// Argument fields left out when comparing two decisions: timestamps, ids, free text.
    /// Top-level keys of each command's `args`.
    pub volatile: Vec<String>,
    /// Engrams remembered at once, rounded up to a multiple of 4; the least recently seen in
    /// a key's set is replaced
    pub capacity: usize,
    /// External state the brain's decisions depend on, each with a version
    pub depends_on: Vec<String>,
    key: Option<KeyFn>,
    tags: Option<TagsFn>,
}

impl std::fmt::Debug for Consolidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Consolidation")
            .field("effectors", &self.effectors)
            .field("after", &self.after)
            .field("shadow", &self.shadow)
            .field("first_shadows", &self.first_shadows)
            .field("forget_after", &self.forget_after)
            .field("failures", &self.failures)
            .field("volatile", &self.volatile)
            .field("capacity", &self.capacity)
            .field("depends_on", &self.depends_on)
            .field("key", &self.key.as_ref().map(|_| "fn"))
            .field("tags", &self.tags.as_ref().map(|_| "fn"))
            .finish()
    }
}

impl Default for Consolidation {
    fn default() -> Self {
        Self {
            effectors: Vec::new(),
            after: 3,
            shadow: 0.05,
            first_shadows: 3,
            forget_after: Duration::from_secs(3600),
            failures: 3,
            volatile: Vec::new(),
            capacity: 4096,
            depends_on: Vec::new(),
            key: None,
            tags: None,
        }
    }
}

impl Consolidation {
    /// The defaults, with no effector allowed: nothing consolidates until some are
    pub fn new() -> Self {
        Self::default()
    }

    /// Effectors whose commands may be answered from an engram
    pub fn effectors<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.effectors = names.into_iter().map(Into::into).collect();
        self
    }

    /// Agreeing brain decisions in a row before consolidating, at least 1
    pub fn after(mut self, n: u32) -> Self {
        self.after = n.max(1);
        self
    }

    /// Fraction of reflex answers also checked by the brain, clamped to 0..=1
    pub fn shadow(mut self, fraction: f64) -> Self {
        self.shadow = fraction.clamp(0.0, 1.0);
        self
    }

    /// Answers of a new reflex that are always checked
    pub fn first_shadows(mut self, n: u32) -> Self {
        self.first_shadows = n;
        self
    }

    /// How long an unseen engram is kept
    pub fn forget_after(mut self, ttl: Duration) -> Self {
        self.forget_after = ttl;
        self
    }

    /// Effector failures in a row that dissolve an engram, at least 1
    pub fn failures(mut self, k: u32) -> Self {
        self.failures = k.max(1);
        self
    }

    /// Top-level argument fields ignored when comparing decisions
    pub fn volatile<S: Into<String>>(mut self, fields: impl IntoIterator<Item = S>) -> Self {
        self.volatile = fields.into_iter().map(Into::into).collect();
        self
    }

    /// Engrams remembered at once, at least 4, rounded up to a multiple of 4
    pub fn capacity(mut self, n: usize) -> Self {
        self.capacity = n.max(WAYS);
        self
    }

    /// Declare the external state the brain consults (`"customers"`): each gets a version,
    /// and [`Memory::advance`] voids every engram made under the old one. Without any, the
    /// brain is taken to decide from its input alone.
    pub fn depends_on<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.depends_on = names.into_iter().map(Into::into).collect();
        self
    }

    /// The key a signal's decision is filed under; the signal's feature by default
    pub fn key(mut self, key: impl Fn(&Signal) -> Feature + Send + Sync + 'static) -> Self {
        self.key = Some(Arc::new(key));
        self
    }

    /// What a signal's decision is about (`customer:42`): [`Memory::inhibit`] with one of
    /// these dissolves the engram at once
    pub fn tags(mut self, tags: impl Fn(&Signal) -> Vec<String> + Send + Sync + 'static) -> Self {
        self.tags = Some(Arc::new(tags));
        self
    }
}

// ---------------------------------------------------------------------------------------------
// The engram store

/// Where an engram is in its life
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EngramState {
    /// The brain has decided the same thing a few times, not yet enough
    Forming,
    /// Answered by the reflex
    Consolidated,
    /// The brain disagreed on a check, the effector kept failing, or what it was about
    /// changed: back to the brain
    Dissolved,
}

#[derive(Clone)]
struct Engram {
    key: Feature,
    state: EngramState,
    /// Normalized decision
    decision: u64,
    /// The last brain decision with that normalized form, as fired
    commands: Arc<[Arc<Command>]>,
    tags: Box<[u64]>,
    /// The dependency versions it was made under
    generation: u64,
    /// Agreeing brain decisions in a row
    agree: u32,
    dissolutions: u32,
    /// Reflex answers since it was consolidated
    hits: u64,
    /// Effector failures in a row
    failures: u32,
    /// Nanoseconds since the store's epoch; `u64::MAX` marks an empty slot
    seen: u64,
    formed: u64,
}

const EMPTY: u64 = u64::MAX;

impl Engram {
    fn empty() -> Self {
        Self {
            key: Feature::NONE,
            state: EngramState::Forming,
            decision: 0,
            commands: Arc::from(Vec::new()),
            tags: Box::default(),
            generation: 0,
            agree: 0,
            dissolutions: 0,
            hits: 0,
            failures: 0,
            seen: EMPTY,
            formed: 0,
        }
    }

    fn live(&self) -> bool {
        self.seen != EMPTY
    }

    /// Back to the brain because the world changed, not because the decision was wrong: no
    /// penalty for consolidating it again
    fn void(&mut self) {
        self.state = EngramState::Dissolved;
        self.agree = 0;
        self.failures = 0;
    }
}

/// Counters of an [`EngramStore`]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ConsolidationMetrics {
    /// Signals that reached [`Recall`]
    pub signals: u64,
    /// Of those, sent to the brain to decide (no consolidated engram)
    pub brain: u64,
    /// Answered by an engram
    pub hits: u64,
    /// Reflex answers also sent to the brain to check
    pub shadows: u64,
    /// Checks where the brain decided the same
    pub agreements: u64,
    /// Checks where it did not (each dissolves the engram)
    pub disagreements: u64,
    /// Engrams that became reflexes
    pub consolidated: u64,
    /// Engrams dissolved by disagreement or by effector failures
    pub dissolved: u64,
    /// Of those, by effector failures
    pub dissolved_by_failures: u64,
    /// Engrams dissolved by a change of what they were about ([`Memory::inhibit`])
    pub inhibited: u64,
    /// Engrams found made under an old dependency version, and brain decisions that came
    /// back after their dependency changed (not recorded)
    pub stale: u64,
    /// Engrams forgotten after `forget_after` without being seen
    pub forgotten: u64,
    /// Brain decisions that named an effector not allowed to consolidate
    pub forbidden: u64,
    /// Brain decisions recorded
    pub recorded: u64,
    /// Engrams in the store now, and of those consolidated
    pub engrams: u64,
    /// Of the engrams, those consolidated
    pub reflexes: u64,
    /// `brain / signals`
    pub brain_fraction: f64,
    /// `(brain + shadows) / signals`: the brain's whole load
    pub brain_load: f64,
    /// `brain / signals` per window of [`WINDOW`] signals, oldest first (the last few)
    pub timeline: Vec<f64>,
    /// Signal-to-decision time through the brain (p50), and what the hits saved at that
    pub brain_p50_ms: Option<f64>,
    /// Brain time the hits saved, at the p50: `hits * brain_p50_ms`
    pub brain_time_saved_ms: f64,
}

#[derive(Default)]
struct Counts {
    signals: u64,
    brain: u64,
    hits: u64,
    shadows: u64,
    agreements: u64,
    disagreements: u64,
    consolidated: u64,
    dissolved: u64,
    dissolved_by_failures: u64,
    inhibited: u64,
    stale: u64,
    forgotten: u64,
    forbidden: u64,
    recorded: u64,
    window_signals: u64,
    window_brain: u64,
    timeline: VecDeque<f64>,
}

struct Store {
    slots: Vec<Engram>,
    sets: usize,
    /// Shadow checks on their way to the brain: the decision the reflex fired
    pending: HashMap<SignalId, (Feature, u64)>,
    pending_order: VecDeque<SignalId>,
    /// When each tag was last inhibited (nanoseconds since the epoch): a decision about it
    /// for a stimulus seen before then comes back too late to be recorded
    inhibited: HashMap<u64, u64>,
    /// Every stimulus before this is treated as inhibited (the table above overflowed)
    inhibited_all: u64,
    counts: Counts,
}

impl Store {
    fn set(&self, key: Feature) -> std::ops::Range<usize> {
        let set = (key.sketch ^ (u64::from(key.kind) << 56)) as usize % self.sets;
        set * WAYS..(set + 1) * WAYS
    }

    /// The live engram for `key`, forgetting it first if it has not been seen for `ttl`
    fn find(&mut self, key: Feature, now: u64, ttl: u64) -> Option<usize> {
        let i = self
            .set(key)
            .find(|&i| self.slots[i].live() && self.slots[i].key == key)?;
        if now.saturating_sub(self.slots[i].seen) > ttl {
            self.slots[i] = Engram::empty();
            self.counts.forgotten += 1;
            return None;
        }
        Some(i)
    }

    /// A slot for a new engram: an empty one first, else the least recently seen (an
    /// expired one counts as forgotten)
    fn claim(&mut self, key: Feature, now: u64, ttl: u64) -> usize {
        let victim = self
            .set(key)
            .min_by_key(|&i| {
                let seen = self.slots[i].seen;
                if seen == EMPTY {
                    0
                } else {
                    seen.saturating_add(1)
                }
            })
            .expect("a set has ways");
        if self.slots[victim].live() && now.saturating_sub(self.slots[victim].seen) > ttl {
            self.counts.forgotten += 1;
        }
        victim
    }

    fn dissolve(&mut self, i: usize, by_failures: bool) {
        let engram = &mut self.slots[i];
        engram.void();
        engram.dissolutions = engram.dissolutions.saturating_add(1);
        self.counts.dissolved += 1;
        if by_failures {
            self.counts.dissolved_by_failures += 1;
        }
    }

    /// Whether a decision for a stimulus seen at `at` about `tags` was overtaken by a change
    fn overtaken(&self, tags: &[u64], at: u64) -> bool {
        at < self.inhibited_all
            || tags
                .iter()
                .any(|t| self.inhibited.get(t).is_some_and(|&when| at <= when))
    }
}

struct Shared {
    policy: Consolidation,
    epoch: Instant,
    ttl: u64,
    store: Mutex<Store>,
    /// The declared dependencies' versions, in `policy.depends_on`'s order
    versions: Vec<AtomicU64>,
    /// The sum of the versions: it changes whenever one does
    generation: AtomicU64,
    /// Signal-to-decision time of the brain's decisions
    latency: LatencyEstimate,
}

/// The built-in [`Memory`]: engrams keyed exactly, bounded and set-associative like a
/// habituation table, holding the commands the brain decided. Shared by a brain's
/// [`Recall`], [`Record`], [`Inhibit`] and the effectors it [`watch`](EngramStore::watch)es.
/// Cloning shares the same store.
#[derive(Clone)]
pub struct EngramStore(Arc<Shared>);

impl EngramStore {
    /// An empty store under `policy`
    pub fn new(policy: Consolidation) -> Self {
        let sets = policy.capacity.div_ceil(WAYS).max(1);
        Self(Arc::new(Shared {
            ttl: u64::try_from(policy.forget_after.as_nanos()).unwrap_or(u64::MAX),
            versions: policy
                .depends_on
                .iter()
                .map(|_| AtomicU64::new(0))
                .collect(),
            generation: AtomicU64::new(0),
            policy,
            epoch: Instant::now(),
            store: Mutex::new(Store {
                slots: vec![Engram::empty(); sets * WAYS],
                sets,
                pending: HashMap::new(),
                pending_order: VecDeque::new(),
                inhibited: HashMap::new(),
                inhibited_all: 0,
                counts: Counts::default(),
            }),
            latency: LatencyEstimate::new(),
        }))
    }

    /// The policy the store was made with
    pub fn policy(&self) -> &Consolidation {
        &self.0.policy
    }

    /// When the store was made: times passed to it count from here
    pub fn epoch(&self) -> Instant {
        self.0.epoch
    }

    fn nanos(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.0.epoch).as_nanos()).unwrap_or(u64::MAX)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Store> {
        self.0.store.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Acquire)
    }

    /// Wrap an effector's act so that its failures reach the engrams it acts for: `failures`
    /// in a row for one key dissolve that key's engram
    pub fn watch<A: Act>(&self, act: A) -> Watched<A> {
        Watched::new(act, self)
    }

    /// The version of a declared dependency
    pub fn version(&self, dependency: &str) -> Option<u64> {
        let i = self
            .0
            .policy
            .depends_on
            .iter()
            .position(|d| d == dependency)?;
        Some(self.0.versions[i].load(Ordering::Acquire))
    }

    /// The state of the engram for `key`, if there is one
    pub fn state(&self, key: Feature) -> Option<EngramState> {
        let generation = self.generation();
        let store = self.lock();
        store
            .slots
            .get(store.set(key))?
            .iter()
            .find(|e| e.live() && e.key == key)
            .map(|e| {
                if e.generation == generation {
                    e.state
                } else {
                    EngramState::Dissolved
                }
            })
    }

    /// The counters and rates now
    pub fn snapshot(&self) -> ConsolidationMetrics {
        let p50 = self.0.latency.p50();
        let store = self.lock();
        let c = &store.counts;
        let ratio = |n: u64| {
            if c.signals == 0 {
                0.0
            } else {
                n as f64 / c.signals as f64
            }
        };
        let generation = self.generation();
        let live = store.slots.iter().filter(|e| e.live());
        ConsolidationMetrics {
            signals: c.signals,
            brain: c.brain,
            hits: c.hits,
            shadows: c.shadows,
            agreements: c.agreements,
            disagreements: c.disagreements,
            consolidated: c.consolidated,
            dissolved: c.dissolved,
            dissolved_by_failures: c.dissolved_by_failures,
            inhibited: c.inhibited,
            stale: c.stale,
            forgotten: c.forgotten,
            forbidden: c.forbidden,
            recorded: c.recorded,
            engrams: live.clone().count() as u64,
            reflexes: live
                .filter(|e| e.state == EngramState::Consolidated && e.generation == generation)
                .count() as u64,
            brain_fraction: ratio(c.brain),
            brain_load: ratio(c.brain + c.shadows),
            timeline: c.timeline.iter().copied().collect(),
            brain_p50_ms: p50.map(|d| d.as_secs_f64() * 1e3),
            brain_time_saved_ms: p50.map_or(0.0, |d| d.as_secs_f64() * 1e3 * c.hits as f64),
        }
    }

    /// Record the brain's decision `decision` for `cause` (`None`: a firing that is not a
    /// decision); `true` to act on it, `false` for a shadow check's answer
    fn decide(&self, cause: &Cause, decision: Option<&[&Command]>, now: Instant) -> bool {
        let now = self.nanos(now);
        let policy = &self.0.policy;
        let normalized = decision.map(|d| normalize(d, &policy.volatile));
        let mut store = self.lock();
        if let Some((key, expected)) = store.pending.remove(&cause.id) {
            store.pending_order.retain(|&p| p != cause.id);
            if normalized == Some(expected) {
                store.counts.agreements += 1;
            } else {
                store.counts.disagreements += 1;
                if let Some(i) = store.find(key, now, self.0.ttl)
                    && store.slots[i].state == EngramState::Consolidated
                    && store.slots[i].decision == expected
                {
                    store.dissolve(i, false);
                }
            }
            return false;
        }
        let (Some(decision), Some(normalized)) = (decision, normalized) else {
            return true;
        };
        if cause.key.is_none() {
            return true;
        }
        if decision
            .iter()
            .any(|c| !policy.effectors.contains(&c.effector))
        {
            store.counts.forbidden += 1;
            return true;
        }
        // Made under a dependency version or about a state that has changed since
        if cause.generation != self.generation()
            || store.overtaken(&cause.tags, self.nanos(cause.at))
        {
            store.counts.stale += 1;
            return true;
        }
        store.counts.recorded += 1;
        let commands: Arc<[Arc<Command>]> = decision.iter().map(|&c| Arc::new(c.clone())).collect();
        let i = match store.find(cause.key, now, self.0.ttl) {
            Some(i) => i,
            None => {
                let i = store.claim(cause.key, now, self.0.ttl);
                store.slots[i] = Engram {
                    key: cause.key,
                    decision: normalized,
                    commands: commands.clone(),
                    generation: cause.generation,
                    ..Engram::empty()
                };
                i
            }
        };
        if store.slots[i].generation != cause.generation {
            // An engram of an older version: it starts again under this one
            store.slots[i].void();
            store.slots[i].generation = cause.generation;
        }
        let needed = policy
            .after
            .saturating_mul(store.slots[i].dissolutions.saturating_add(1));
        let engram = &mut store.slots[i];
        engram.seen = now;
        engram.tags = cause.tags.clone().into_boxed_slice();
        match engram.state {
            EngramState::Consolidated if engram.decision == normalized => {
                engram.commands = commands;
            }
            EngramState::Consolidated => {
                // The brain was asked while the reflex formed, and now says otherwise
                store.counts.disagreements += 1;
                store.dissolve(i, false);
                let engram = &mut store.slots[i];
                engram.decision = normalized;
                engram.commands = commands;
                engram.agree = 1;
            }
            EngramState::Forming | EngramState::Dissolved => {
                if engram.decision == normalized && engram.agree > 0 {
                    engram.agree += 1;
                } else {
                    engram.decision = normalized;
                    engram.agree = 1;
                }
                engram.commands = commands;
                engram.state = EngramState::Forming;
                if engram.agree >= needed {
                    engram.state = EngramState::Consolidated;
                    engram.hits = 0;
                    engram.formed = now;
                    store.counts.consolidated += 1;
                }
            }
        }
        true
    }
}

impl Memory for EngramStore {
    fn cause(&self, signal: &Signal) -> Cause {
        let policy = &self.0.policy;
        Cause {
            key: policy
                .key
                .as_ref()
                .map_or(signal.feature, |key| key(signal)),
            id: signal.id,
            at: signal.at,
            tags: policy
                .tags
                .as_ref()
                .map(|tags| {
                    tags(signal)
                        .iter()
                        .map(|t| sketch64(t.as_bytes()))
                        .collect()
                })
                .unwrap_or_default(),
            generation: self.generation(),
        }
    }

    fn recall(&self, signal: &Signal, cause: &Cause, now: Instant) -> Recalled {
        let now = self.nanos(now);
        let policy = &self.0.policy;
        let generation = self.generation();
        let mut store = self.lock();
        store.counts.signals += 1;
        store.counts.window_signals += 1;
        let found = if cause.key.is_none() {
            None
        } else {
            store.find(cause.key, now, self.0.ttl)
        };
        if let Some(i) = found
            && store.slots[i].state == EngramState::Consolidated
            && store.slots[i].generation != generation
        {
            store.slots[i].void();
            store.counts.stale += 1;
        }
        let recalled = match found {
            Some(i) if store.slots[i].state == EngramState::Consolidated => {
                let engram = &mut store.slots[i];
                engram.seen = now;
                engram.hits += 1;
                let shadow = engram.hits <= u64::from(policy.first_shadows)
                    || sample(cause.id.0, policy.shadow);
                let (commands, decision) = (engram.commands.clone(), engram.decision);
                store.counts.hits += 1;
                if shadow {
                    store.counts.shadows += 1;
                    store.pending.insert(cause.id, (cause.key, decision));
                    store.pending_order.push_back(cause.id);
                    // A check that never came back (its queue was full) is let go
                    while store.pending_order.len() > store.slots.len()
                        && let Some(old) = store.pending_order.pop_front()
                    {
                        store.pending.remove(&old);
                    }
                }
                // A fired decision carries its key as its feature, for `outcome`
                let fire = commands
                    .iter()
                    .map(|c| {
                        signal
                            .derive(Payload::Command(c.clone()))
                            .with_feature(cause.key)
                    })
                    .collect();
                Recalled::Hit { fire, shadow }
            }
            _ => {
                store.counts.brain += 1;
                store.counts.window_brain += 1;
                Recalled::Miss
            }
        };
        if store.counts.window_signals >= WINDOW {
            let counts = &mut store.counts;
            let fraction = counts.window_brain as f64 / counts.window_signals as f64;
            counts.timeline.push_back(fraction);
            if counts.timeline.len() > WINDOWS {
                counts.timeline.pop_front();
            }
            counts.window_signals = 0;
            counts.window_brain = 0;
        }
        recalled
    }

    fn decided(&self, cause: &Cause, fired: &[Signal], now: Instant) -> bool {
        let commands: Option<Vec<&Command>> =
            fired.iter().map(|s| s.payload.as_command()).collect();
        let decision = commands.as_deref().filter(|c| !c.is_empty());
        let act = self.decide(cause, decision, now);
        if act && decision.is_some() {
            self.0
                .latency
                .observe(now.saturating_duration_since(cause.at));
        }
        act
    }

    fn outcome(&self, key: Feature, ok: bool, now: Instant) {
        if key.is_none() {
            return;
        }
        let now = self.nanos(now);
        let limit = self.0.policy.failures;
        let mut store = self.lock();
        let Some(i) = store.find(key, now, self.0.ttl) else {
            return;
        };
        let engram = &mut store.slots[i];
        if ok {
            engram.failures = 0;
            return;
        }
        engram.failures += 1;
        if engram.failures >= limit && engram.state == EngramState::Consolidated {
            store.dissolve(i, true);
        }
    }

    fn inhibit(&self, tag: &str, now: Instant) -> usize {
        let now = self.nanos(now);
        let tag = sketch64(tag.as_bytes());
        let mut store = self.lock();
        let mut dissolved = 0;
        for engram in store.slots.iter_mut() {
            if engram.live() && engram.state != EngramState::Dissolved && engram.tags.contains(&tag)
            {
                engram.void();
                dissolved += 1;
            }
        }
        store.counts.inhibited += dissolved as u64;
        if store.inhibited.len() >= store.slots.len() {
            store.inhibited.clear();
            store.inhibited_all = now;
        }
        store.inhibited.insert(tag, now);
        dissolved
    }

    fn advance(&self, dependency: &str, _now: Instant) -> Option<u64> {
        let Some(i) = self
            .0
            .policy
            .depends_on
            .iter()
            .position(|d| d == dependency)
        else {
            warn!("nerve: consolidation has no dependency named '{dependency}'");
            return None;
        };
        // Under the store's lock, so no decision is recorded between the two increments
        let _store = self.lock();
        let version = self.0.versions[i].fetch_add(1, Ordering::AcqRel) + 1;
        self.0.generation.fetch_add(1, Ordering::AcqRel);
        Some(version)
    }

    /// The policy, the counters and the most used engrams, for `GET /nerve`
    fn describe(&self) -> serde_json::Value {
        let now = self.nanos(Instant::now());
        let metrics = self.snapshot();
        let generation = self.generation();
        let store = self.lock();
        let mut top: Vec<&Engram> = store.slots.iter().filter(|e| e.live()).collect();
        top.sort_by_key(|e| (std::cmp::Reverse(e.hits), std::cmp::Reverse(e.agree)));
        let top: Vec<serde_json::Value> = top
            .into_iter()
            .take(TOP)
            .map(|e| {
                let state = if e.generation == generation {
                    e.state
                } else {
                    EngramState::Dissolved
                };
                serde_json::json!({
                    "key": format!("{}:{:016x}", e.key.kind, e.key.sketch),
                    "state": state,
                    "hits": e.hits,
                    "agree": e.agree,
                    "dissolutions": e.dissolutions,
                    "effectors": e.commands.iter().map(|c| c.effector.as_str()).collect::<Vec<_>>(),
                    "tags": e.tags.len(),
                    "idle_s": now.saturating_sub(e.seen) as f64 / 1e9,
                    "age_s": (state == EngramState::Consolidated)
                        .then(|| now.saturating_sub(e.formed) as f64 / 1e9),
                })
            })
            .collect();
        let policy = &self.0.policy;
        let versions: serde_json::Map<String, serde_json::Value> = policy
            .depends_on
            .iter()
            .zip(&self.0.versions)
            .map(|(name, v)| (name.clone(), v.load(Ordering::Acquire).into()))
            .collect();
        serde_json::json!({
            "consolidation": {
                "effectors": policy.effectors,
                "after": policy.after,
                "shadow": policy.shadow,
                "first_shadows": policy.first_shadows,
                "forget_after_s": policy.forget_after.as_secs_f64(),
                "failures": policy.failures,
                "volatile": policy.volatile,
                "capacity": store.slots.len(),
                "depends_on": versions,
                "keyed_by": if policy.key.is_some() { "key fn" } else { "feature" },
                "tagged": policy.tags.is_some(),
            },
            "metrics": metrics,
            "top": top,
        })
    }

    fn metrics(&self) -> serde_json::Value {
        serde_json::to_value(self.snapshot()).unwrap_or_default()
    }
}

/// Whether the signal numbered `id` is in a sample of `fraction`: deterministic, so a run
/// repeats
fn sample(id: u64, fraction: f64) -> bool {
    if fraction >= 1.0 {
        return true;
    }
    let mut z = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 / (1u64 << 53) as f64) < fraction
}

/// A decision's normal form: its commands, in order, with the volatile argument fields left
/// out and object keys sorted, hashed
fn normalize(decision: &[&Command], volatile: &[String]) -> u64 {
    let mut bytes = Vec::with_capacity(64);
    for command in decision {
        bytes.extend_from_slice(command.effector.as_bytes());
        bytes.push(0);
        match &command.args {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> =
                    map.keys().filter(|k| !volatile.contains(*k)).collect();
                keys.sort();
                bytes.push(b'{');
                for key in keys {
                    bytes.extend_from_slice(key.as_bytes());
                    bytes.push(0);
                    canonical(&map[key], &mut bytes);
                }
                bytes.push(b'}');
            }
            other => canonical(other, &mut bytes),
        }
        bytes.push(1);
    }
    sketch64(&bytes)
}

/// JSON with object keys sorted, whatever map order serde_json was built with
fn canonical(value: &serde_json::Value, out: &mut Vec<u8>) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for key in keys {
                out.extend_from_slice(key.as_bytes());
                out.push(0);
                canonical(&map[key], out);
            }
            out.push(b'}');
        }
        serde_json::Value::Array(items) => {
            out.push(b'[');
            for item in items {
                canonical(item, out);
                out.push(b',');
            }
            out.push(b']');
        }
        scalar => out.extend_from_slice(scalar.to_string().as_bytes()),
    }
}

// ---------------------------------------------------------------------------------------------
// The neurons

/// The reflex in front of a brain: a signal its [`Memory`] remembers a decision for fires
/// that decision at once; any other goes on to the brain. A shadow check fires the decision
/// and also sends the signal on, without its reply (the reflex answered).
pub struct Recall {
    name: String,
    memory: Arc<dyn Memory>,
}

impl Recall {
    /// A recall neuron named `name`, over a clone of `memory`
    pub fn new(name: &str, memory: &(impl Memory + Clone)) -> Self {
        Self::shared(name, Arc::new(memory.clone()))
    }

    /// A recall neuron named `name`, over a memory already shared
    pub fn shared(name: &str, memory: Arc<dyn Memory>) -> Self {
        Self {
            name: name.to_string(),
            memory,
        }
    }
}

impl Neuron for Recall {
    fn name(&self) -> &str {
        &self.name
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        let cause = self.memory.cause(&signal);
        match self.memory.recall(&signal, &cause, cx.now) {
            Recalled::Miss => Firing::one(signal),
            Recalled::Hit { mut fire, shadow } => {
                if shadow {
                    let mut check = signal;
                    check.reply = None;
                    fire.push(check);
                }
                Firing::Fire(fire)
            }
        }
    }

    fn describe(&self) -> serde_json::Value {
        self.memory.describe()
    }

    fn metrics(&self) -> serde_json::Value {
        self.memory.metrics()
    }
}

/// A brain whose decisions are recorded: each firing, next to the signal that caused it, is
/// a decision for that signal's key. A shadow check's firing is compared and dropped.
pub struct Record<N> {
    brain: N,
    memory: Arc<dyn Memory>,
}

impl<N: Neuron> Record<N> {
    /// `brain`, its decisions recorded in a clone of `memory`
    pub fn new(brain: N, memory: &(impl Memory + Clone)) -> Self {
        Self::shared(brain, Arc::new(memory.clone()))
    }

    /// `brain`, its decisions recorded in a memory already shared
    pub fn shared(brain: N, memory: Arc<dyn Memory>) -> Self {
        Self { brain, memory }
    }
}

/// Record what the brain fired for `cause`; what is to be acted on, with the decision's key
/// as its feature
fn record(memory: &dyn Memory, cause: &Cause, mut fired: Vec<Signal>) -> Vec<Signal> {
    if !memory.decided(cause, &fired, Instant::now()) {
        return Vec::new();
    }
    for signal in &mut fired {
        if signal.payload.as_command().is_some() {
            signal.feature = cause.key;
        }
    }
    fired
}

impl<N> Record<N> {
    fn observe(&self, cause: Cause, firing: Firing) -> Firing {
        match firing {
            // A check the brain had nothing to say to disagrees; anything else stays quiet
            Firing::Quiet => {
                if self.memory.decided(&cause, &[], Instant::now()) {
                    Firing::Quiet
                } else {
                    Firing::Fire(Vec::new())
                }
            }
            Firing::Fire(fired) => Firing::Fire(record(&*self.memory, &cause, fired)),
            Firing::Compute(work) => {
                let memory = self.memory.clone();
                Firing::compute(move || record(&*memory, &cause, work()))
            }
        }
    }
}

impl<N: Neuron> Neuron for Record<N> {
    fn name(&self) -> &str {
        self.brain.name()
    }

    fn kind(&self) -> NeuronKind {
        self.brain.kind()
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        let cause = self.memory.cause(&signal);
        let firing = self.brain.stimulate(signal, cx);
        self.observe(cause, firing)
    }

    fn stimulate_many(&self, signals: Vec<Signal>, cx: &Cx<'_>) -> Vec<Firing> {
        let causes: Vec<Cause> = signals.iter().map(|s| self.memory.cause(s)).collect();
        self.brain
            .stimulate_many(signals, cx)
            .into_iter()
            .zip(causes)
            .map(|(firing, cause)| self.observe(cause, firing))
            .collect()
    }

    fn prepare(&self, cx: &Cx<'_>) {
        self.brain.prepare(cx);
    }

    fn describe(&self) -> serde_json::Value {
        self.brain.describe()
    }

    fn metrics(&self) -> serde_json::Value {
        self.brain.metrics()
    }

    fn render_prometheus(&self, out: &mut String, labels: &str) {
        self.brain.render_prometheus(out, labels);
    }

    fn events(&self) -> Option<tokio::sync::broadcast::Receiver<bytes::Bytes>> {
        self.brain.events()
    }
}

/// Inhibition: change notifications void what the brain decided about the state that
/// changed. A signal whose payload is JSON with `"inhibit": ["customer:42", ...]` dissolves
/// the engrams tagged so; `"advance": ["customers", ...]` raises those dependencies'
/// versions. A string stands for a list of one. Fires nothing.
pub struct Inhibit {
    name: String,
    memory: Arc<dyn Memory>,
    inhibited: AtomicU64,
    advanced: AtomicU64,
}

impl Inhibit {
    /// An inhibition neuron named `name`, over a clone of `memory`
    pub fn new(name: &str, memory: &(impl Memory + Clone)) -> Self {
        Self::shared(name, Arc::new(memory.clone()))
    }

    /// An inhibition neuron named `name`, over a memory already shared
    pub fn shared(name: &str, memory: Arc<dyn Memory>) -> Self {
        Self {
            name: name.to_string(),
            memory,
            inhibited: AtomicU64::new(0),
            advanced: AtomicU64::new(0),
        }
    }
}

/// A JSON string or array of strings
fn names(value: Option<&serde_json::Value>) -> Vec<&str> {
    match value {
        Some(serde_json::Value::String(s)) => vec![s.as_str()],
        Some(serde_json::Value::Array(items)) => {
            items.iter().filter_map(serde_json::Value::as_str).collect()
        }
        _ => Vec::new(),
    }
}

impl Neuron for Inhibit {
    fn name(&self) -> &str {
        &self.name
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        let Some(change) = signal.payload.to_json() else {
            return Firing::Quiet;
        };
        // A command addressed to this neuron carries the change in its arguments
        let change = match change.get("args") {
            Some(args) if change.get("effector").is_some() => args.clone(),
            _ => change,
        };
        let (tags, dependencies) = (names(change.get("inhibit")), names(change.get("advance")));
        if tags.is_empty() && dependencies.is_empty() {
            return Firing::Quiet;
        }
        for tag in tags {
            self.memory.inhibit(tag, cx.now);
            self.inhibited.fetch_add(1, Ordering::Relaxed);
        }
        for dependency in dependencies {
            if self.memory.advance(dependency, cx.now).is_some() {
                self.advanced.fetch_add(1, Ordering::Relaxed);
            }
        }
        Firing::Fire(Vec::new())
    }

    fn metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "inhibited_tags": self.inhibited.load(Ordering::Relaxed),
            "advanced": self.advanced.load(Ordering::Relaxed),
        })
    }
}

/// An effector's act whose outcomes reach a [`Memory`], see [`EngramStore::watch`]. Every
/// attempt counts, retries included.
pub struct Watched<A> {
    act: A,
    memory: Arc<dyn Memory>,
}

impl<A: Act> Watched<A> {
    /// `act`, its outcomes reported to a clone of `memory`
    pub fn new(act: A, memory: &(impl Memory + Clone)) -> Self {
        Self::shared(act, Arc::new(memory.clone()))
    }

    /// `act`, its outcomes reported to a memory already shared
    pub fn shared(act: A, memory: Arc<dyn Memory>) -> Self {
        Self { act, memory }
    }
}

impl<A: Act> Act for Watched<A> {
    fn name(&self) -> &str {
        self.act.name()
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let key = signal.feature;
        match self.act.act(command, signal) {
            Action::Done(result) => {
                self.memory.outcome(key, result.is_ok(), Instant::now());
                Action::Done(result)
            }
            Action::Pending(future) => {
                let memory = self.memory.clone();
                Action::pending(async move {
                    let result = future.await;
                    memory.outcome(key, result.is_ok(), Instant::now());
                    result
                })
            }
        }
    }

    fn prepare(&self, runtime: &tokio::runtime::Handle) {
        self.act.prepare(runtime);
    }

    fn describe(&self) -> serde_json::Value {
        self.act.describe()
    }

    fn events(&self) -> Option<tokio::sync::broadcast::Receiver<bytes::Bytes>> {
        self.act.events()
    }
}

impl FabricBuilder {
    /// Put a brain behind consolidation: add it, wrapped in [`Record`], and a [`Recall`]
    /// reflex in front of it (named `<brain>-recall`), connected by a deliberative pathway
    /// with `summation`. Returns `(recall, brain)`: connect what feeds the brain to `recall`.
    pub fn consolidate(
        &mut self,
        brain: impl Neuron,
        mechanisms: Mechanisms,
        summation: Summation,
        memory: &(impl Memory + Clone),
    ) -> (NeuronId, NeuronId) {
        let memory: Arc<dyn Memory> = Arc::new(memory.clone());
        let recall = Recall::shared(&format!("{}-recall", brain.name()), memory.clone());
        let recall = self.neuron(recall, Mechanisms::NONE);
        let brain = self.neuron(Record::shared(brain, memory), mechanisms);
        self.deliberative(recall, brain, 1.0, summation);
        (recall, brain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::effector::{self, ActError, Effector};
    use crate::nerve::{Fabric, FnNeuron};
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    const SECOND: Duration = Duration::from_secs(1);

    fn feature(n: u64) -> Feature {
        Feature::new(7, n)
    }

    fn command(effector: &str, args: serde_json::Value) -> Command {
        Command::new(effector, args)
    }

    /// A signal with feature `f`, seen at `at` after the store's epoch
    fn signal_at(store: &EngramStore, f: Feature, at: Duration) -> Signal {
        let mut s = Signal::new(f, Payload::text("line"));
        s.at = store.epoch() + at;
        s
    }

    /// Record `decision` for `f` as the brain's, at `at` after the epoch
    fn decide(store: &EngramStore, f: Feature, decision: &[&Command], at: Duration) -> bool {
        let signal = signal_at(store, f, at);
        let fired: Vec<Signal> = decision
            .iter()
            .map(|&c| signal.derive(Payload::command(c.clone())))
            .collect();
        store.decided(&store.cause(&signal), &fired, store.epoch() + at)
    }

    /// What the store answers a signal with `f` at `at`
    fn recall(store: &EngramStore, f: Feature, at: Duration) -> Recalled {
        let signal = signal_at(store, f, at);
        store.recall(&signal, &store.cause(&signal), store.epoch() + at)
    }

    fn fired(recalled: Recalled) -> Vec<Command> {
        let Recalled::Hit { fire, .. } = recalled else {
            panic!("a consolidated engram answers");
        };
        fire.iter()
            .map(|s| s.payload.as_command().unwrap().clone())
            .collect()
    }

    #[test]
    fn decisions_consolidate_after_n_agreeing_ones_and_disagreement_resets() {
        let store = EngramStore::new(Consolidation::new().effectors(["ticket", "page"]).after(3));
        let ticket = command("ticket", serde_json::json!({"queue": "ops"}));
        let page = command("page", serde_json::json!({}));
        let f = feature(1);
        decide(&store, f, &[&ticket], SECOND);
        decide(&store, f, &[&ticket], 2 * SECOND);
        // A different decision starts over
        decide(&store, f, &[&page], 3 * SECOND);
        assert_eq!(store.state(f), Some(EngramState::Forming));
        decide(&store, f, &[&page], 4 * SECOND);
        decide(&store, f, &[&page], 5 * SECOND);
        assert_eq!(store.state(f), Some(EngramState::Consolidated));
        let commands = fired(recall(&store, f, 6 * SECOND));
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].effector, "page");
        assert_eq!(store.snapshot().consolidated, 1);
    }

    #[test]
    fn volatile_fields_do_not_tell_decisions_apart() {
        let store = EngramStore::new(
            Consolidation::new()
                .effectors(["ticket"])
                .after(2)
                .volatile(["at"]),
        );
        let f = feature(2);
        let a = command(
            "ticket",
            serde_json::json!({"at": 1, "queue": {"x": 1, "y": 2}}),
        );
        let b = command(
            "ticket",
            serde_json::json!({"queue": {"y": 2, "x": 1}, "at": 2}),
        );
        decide(&store, f, &[&a], SECOND);
        decide(&store, f, &[&b], 2 * SECOND);
        assert_eq!(store.state(f), Some(EngramState::Consolidated));
        // The engram fires the latest decision as the brain made it
        assert_eq!(fired(recall(&store, f, 3 * SECOND))[0].args["at"], 2);
    }

    #[test]
    fn forbidden_effectors_never_consolidate() {
        let store = EngramStore::new(Consolidation::new().effectors(["ticket"]).after(1));
        let f = feature(3);
        let ticket = command("ticket", serde_json::json!({}));
        let delete = command("delete", serde_json::json!({}));
        for i in 1..=5 {
            // Mixed with an allowed one, still not
            assert!(decide(&store, f, &[&ticket, &delete], i * SECOND));
        }
        assert_eq!(store.state(f), None);
        assert!(!recall(&store, f, 6 * SECOND).is_hit());
        assert_eq!(store.snapshot().forbidden, 5);
    }

    #[test]
    fn engrams_are_forgotten_after_ttl_and_kept_alive_by_use() {
        let store = EngramStore::new(
            Consolidation::new()
                .effectors(["ticket"])
                .after(1)
                .forget_after(10 * SECOND),
        );
        let f = feature(4);
        let ticket = command("ticket", serde_json::json!({}));
        decide(&store, f, &[&ticket], SECOND);
        // Used within the ttl, again and again: alive
        for s in [9, 18, 27] {
            assert!(recall(&store, f, s * SECOND).is_hit());
        }
        // Then idle for longer than the ttl: forgotten, back to the brain
        assert!(!recall(&store, f, 38 * SECOND).is_hit());
        assert_eq!(store.state(f), None);
        assert_eq!(store.snapshot().forgotten, 1);
    }

    #[test]
    fn the_store_is_bounded() {
        let store = EngramStore::new(Consolidation::new().effectors(["t"]).after(1).capacity(8));
        let t = command("t", serde_json::json!({}));
        for i in 0..1000u64 {
            decide(
                &store,
                Feature::sketch(&i.to_le_bytes()),
                &[&t],
                Duration::from_millis(i + 1),
            );
        }
        let m = store.snapshot();
        assert_eq!(m.engrams, 8);
        assert_eq!(m.recorded, 1000);
    }

    #[test]
    fn shadow_checks_are_sampled_deterministically() {
        let hits = (0..100_000u64).filter(|&i| sample(i, 0.05)).count();
        assert!((4_500..5_500).contains(&hits), "{hits}");
        assert!(!(0..1000u64).any(|i| sample(i, 0.0)));
        assert!((0..1000u64).all(|i| sample(i, 1.0)));
        assert_eq!(sample(42, 0.3), sample(42, 0.3));
    }

    #[test]
    fn a_key_of_your_own_files_decisions() {
        // Keyed by the text's first word, whatever the feature
        let store = EngramStore::new(Consolidation::new().effectors(["t"]).after(2).key(|s| {
            let text = s.payload.as_text().unwrap_or_default();
            Feature::sketch_as(9, text.split(' ').next().unwrap_or("").as_bytes())
        }));
        let t = command("t", serde_json::json!({}));
        for (i, text) in ["disk full on a", "disk full on b"].iter().enumerate() {
            let signal = Signal::new(feature(i as u64), Payload::text(text));
            let fired = vec![signal.derive(Payload::command(t.clone()))];
            store.decided(&store.cause(&signal), &fired, Instant::now());
        }
        let other = Signal::new(feature(99), Payload::text("disk again"));
        let hit = store.recall(&other, &store.cause(&other), Instant::now());
        let Recalled::Hit { fire, .. } = hit else {
            panic!("the same key");
        };
        // A fired decision carries its key as its feature
        assert_eq!(fire[0].feature, Feature::sketch_as(9, b"disk"));
    }

    #[test]
    fn an_inhibited_tag_dissolves_its_engrams_at_once() {
        let store = EngramStore::new(
            Consolidation::new()
                .effectors(["t"])
                .after(1)
                .depends_on(["customers"])
                .tags(|s| vec![format!("customer:{}", s.feature.sketch)]),
        );
        let t = command("t", serde_json::json!({}));
        decide(&store, feature(42), &[&t], SECOND);
        decide(&store, feature(43), &[&t], SECOND);
        assert!(recall(&store, feature(42), 2 * SECOND).is_hit());
        assert_eq!(store.inhibit("customer:42", store.epoch() + 3 * SECOND), 1);
        assert_eq!(store.state(feature(42)), Some(EngramState::Dissolved));
        assert!(!recall(&store, feature(42), 4 * SECOND).is_hit());
        // Another customer's is untouched
        assert!(recall(&store, feature(43), 4 * SECOND).is_hit());
        // A decision for a stimulus seen before the change comes back too late: not recorded
        decide(&store, feature(42), &[&t], 2 * SECOND + SECOND / 2);
        assert!(!recall(&store, feature(42), 5 * SECOND).is_hit());
        // One for a stimulus after it is, with no penalty for the change
        decide(&store, feature(42), &[&t], 6 * SECOND);
        assert!(recall(&store, feature(42), 7 * SECOND).is_hit());
        let m = store.snapshot();
        assert_eq!((m.inhibited, m.stale, m.dissolved), (1, 1, 0));
    }

    #[test]
    fn a_new_dependency_version_voids_the_old_engrams() {
        let store = EngramStore::new(
            Consolidation::new()
                .effectors(["t"])
                .after(1)
                .depends_on(["customers", "prices"]),
        );
        let t = command("t", serde_json::json!({}));
        decide(&store, feature(1), &[&t], SECOND);
        // A decision made under version 0 whose answer is still on its way
        let late = signal_at(&store, feature(2), SECOND);
        let late_cause = store.cause(&late);
        assert!(recall(&store, feature(1), 2 * SECOND).is_hit());
        assert_eq!(store.advance("prices", store.epoch() + 3 * SECOND), Some(1));
        assert_eq!(store.version("prices"), Some(1));
        assert_eq!(store.version("customers"), Some(0));
        assert_eq!(store.advance("weather", Instant::now()), None);
        // The old engram is not used any more
        assert_eq!(store.state(feature(1)), Some(EngramState::Dissolved));
        assert!(!recall(&store, feature(1), 4 * SECOND).is_hit());
        // Nor recorded from a decision made under the old version
        let fired = vec![late.derive(Payload::command(t.clone()))];
        store.decided(&late_cause, &fired, store.epoch() + 4 * SECOND);
        assert!(!recall(&store, feature(2), 5 * SECOND).is_hit());
        // Under the new version it forms again
        decide(&store, feature(1), &[&t], 6 * SECOND);
        assert!(recall(&store, feature(1), 7 * SECOND).is_hit());
        assert_eq!(store.snapshot().stale, 2);
    }

    // ---------------------------------------------------------------------------------------
    // In a fabric, with a brain that counts its calls

    struct Rig {
        fabric: Fabric,
        input: NeuronId,
        changes: NeuronId,
        engrams: EngramStore,
        calls: Arc<AtomicUsize>,
        /// What the brain answers: the effector name
        answer: Arc<Mutex<&'static str>>,
        ticket: mpsc::Receiver<effector::Delivery>,
        page: mpsc::Receiver<effector::Delivery>,
    }

    /// Succeeds or fails as `results` say, in order, then succeeds
    struct Flaky {
        results: Mutex<VecDeque<bool>>,
        sender: mpsc::SyncSender<effector::Delivery>,
    }

    impl Act for Flaky {
        fn name(&self) -> &str {
            "ticket"
        }

        fn act(&self, command: &Command, signal: &Signal) -> Action {
            let _ = self.sender.try_send(effector::Delivery {
                command: command.clone(),
                signal: signal.clone(),
            });
            if self.results.lock().unwrap().pop_front().unwrap_or(true) {
                Action::ok()
            } else {
                Action::Done(Err(ActError::fatal("down")))
            }
        }
    }

    fn rig(policy: Consolidation, ticket_results: &[bool]) -> Rig {
        let engrams = EngramStore::new(policy);
        let mut b = Fabric::builder();
        let input = b.neuron(
            FnNeuron::reflex("input", |s, _| Firing::one(s)),
            Mechanisms::NONE,
        );
        let (ticket_tx, ticket) = mpsc::sync_channel(64);
        let flaky = Flaky {
            results: Mutex::new(ticket_results.iter().copied().collect()),
            sender: ticket_tx,
        };
        b.neuron(
            Effector::new(engrams.watch(flaky)).no_retry(),
            Mechanisms::NONE,
        );
        let (page_act, page) = effector::ChannelOut::new("page", 64);
        b.neuron(Effector::new(engrams.watch(page_act)), Mechanisms::NONE);
        let calls = Arc::new(AtomicUsize::new(0));
        let answer = Arc::new(Mutex::new("ticket"));
        let brain = {
            let (calls, answer) = (calls.clone(), answer.clone());
            FnNeuron::reflex("brain", move |s: Signal, _: &Cx<'_>| {
                calls.fetch_add(1, Ordering::SeqCst);
                let effector = *answer.lock().unwrap();
                Firing::one(s.command(effector, serde_json::json!({"line": "x"})))
            })
            .with_kind(NeuronKind::Brain)
        };
        let (recall, _) = b.consolidate(brain, Mechanisms::NONE, Summation::none(), &engrams);
        b.reflex(input, recall, 1.0);
        let changes = b.neuron(Inhibit::new("changes", &engrams), Mechanisms::NONE);
        Rig {
            fabric: b.build().unwrap(),
            input,
            changes,
            engrams,
            calls,
            answer,
            ticket,
            page,
        }
    }

    impl Rig {
        fn send(&self, key: u64) -> SignalId {
            let signal = Signal::new(feature(key), Payload::text("line"));
            let id = signal.id;
            self.fabric.stimulate(self.input, signal);
            id
        }

        /// Wait for the brain to have been called `n` times in all, and for its queue to
        /// drain, without sleeping a fixed time
        fn brain_calls(&self, n: usize) {
            let deadline = Instant::now() + 10 * SECOND;
            while (self.calls.load(Ordering::SeqCst) < n || self.fabric.waiting() > 0)
                && Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            assert_eq!(self.calls.load(Ordering::SeqCst), n);
        }

        /// The next delivery on `channel`, waiting at most a few seconds
        fn delivered(channel: &mpsc::Receiver<effector::Delivery>) -> effector::Delivery {
            channel.recv_timeout(5 * SECOND).expect("a delivery")
        }

        /// Wait until the shadow checks sent so far have come back
        fn checks_settled(&self) {
            let deadline = Instant::now() + 10 * SECOND;
            loop {
                let m = self.engrams.snapshot();
                if m.agreements + m.disagreements == m.shadows || Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
        }

        /// Consolidate `key` with `n` brain decisions, the brain having been called `before`
        /// times already
        fn consolidate(&self, key: u64, n: usize, before: usize) {
            for i in 1..=n {
                self.send(key);
                Self::delivered(&self.ticket);
                self.brain_calls(before + i);
            }
            assert_eq!(
                self.engrams.state(feature(key)),
                Some(EngramState::Consolidated)
            );
        }
    }

    fn quiet_policy() -> Consolidation {
        Consolidation::new()
            .effectors(["ticket", "page"])
            .shadow(0.0)
            .first_shadows(0)
    }

    #[test]
    fn a_consolidated_decision_answers_without_the_brain() {
        let rig = rig(quiet_policy().after(3), &[]);
        rig.consolidate(1, 3, 0);
        // From now on the reflex acts and the brain is not called
        for _ in 0..20 {
            let id = rig.send(1);
            let d = Rig::delivered(&rig.ticket);
            assert_eq!(d.signal.id, id);
            assert!(!d.signal.trace.reached_brain());
        }
        rig.brain_calls(3);
        // Another feature still goes to the brain
        rig.send(2);
        Rig::delivered(&rig.ticket);
        rig.brain_calls(4);
        let m = rig.engrams.snapshot();
        assert_eq!((m.signals, m.brain, m.hits, m.shadows), (24, 4, 20, 0));
        assert!(m.brain_p50_ms.is_some() && m.brain_time_saved_ms > 0.0);
        // GET /nerve shows the engrams on the recall neuron
        let recall = rig.fabric.neuron("brain-recall").unwrap();
        let described = rig.fabric.describe();
        let shown = &described.neurons[recall.index()].describe;
        assert_eq!(shown["metrics"]["hits"], 20);
        assert_eq!(shown["top"][0]["state"], "consolidated");
    }

    #[test]
    fn shadow_checks_ask_the_brain_without_acting_twice() {
        let rig = rig(quiet_policy().after(2).first_shadows(2), &[]);
        rig.consolidate(1, 2, 0);
        // The first two answers are checked: the brain is called, the effector acts once
        for i in 3..=4 {
            rig.send(1);
            Rig::delivered(&rig.ticket);
            rig.brain_calls(i);
        }
        rig.checks_settled();
        rig.send(1);
        Rig::delivered(&rig.ticket);
        rig.brain_calls(4);
        assert!(rig.ticket.try_recv().is_err(), "a check is never acted on");
        let m = rig.engrams.snapshot();
        assert_eq!(
            (m.hits, m.shadows, m.agreements, m.disagreements),
            (3, 2, 2, 0)
        );
        assert_eq!(m.brain_load, 4.0 / 5.0);
    }

    #[test]
    fn a_disagreeing_check_dissolves_the_engram() {
        let rig = rig(quiet_policy().after(2).shadow(1.0), &[]);
        rig.consolidate(1, 2, 0);
        // The world changed: the brain now pages for this feature
        *rig.answer.lock().unwrap() = "page";
        rig.send(1);
        // The reflex still files a ticket; the check disagrees and dissolves it
        Rig::delivered(&rig.ticket);
        rig.brain_calls(3);
        rig.checks_settled();
        assert!(
            rig.page.try_recv().is_err(),
            "a check's answer is not acted on"
        );
        assert_eq!(rig.engrams.state(feature(1)), Some(EngramState::Dissolved));
        // Back to the brain, which pages; after twice `after` agreeing decisions it is a
        // reflex again
        for i in 4..=7 {
            rig.send(1);
            let d = Rig::delivered(&rig.page);
            assert!(d.signal.trace.reached_brain());
            rig.brain_calls(i);
        }
        assert_eq!(
            rig.engrams.state(feature(1)),
            Some(EngramState::Consolidated)
        );
        let m = rig.engrams.snapshot();
        assert_eq!((m.disagreements, m.dissolved, m.consolidated), (1, 1, 2));
    }

    #[test]
    fn decisions_for_effectors_not_allowed_always_reach_the_brain() {
        let rig = rig(Consolidation::new().effectors(["ticket"]).after(1), &[]);
        *rig.answer.lock().unwrap() = "page";
        for i in 1..=5 {
            rig.send(1);
            Rig::delivered(&rig.page);
            rig.brain_calls(i);
        }
        let m = rig.engrams.snapshot();
        assert_eq!((m.hits, m.forbidden, m.consolidated), (0, 5, 0));
    }

    #[test]
    fn an_effector_that_keeps_failing_dissolves_its_engram() {
        // The brain's own decision acts and succeeds; the reflex's next three acts fail
        let rig = rig(
            quiet_policy().after(1).failures(3),
            &[true, false, false, false],
        );
        rig.consolidate(1, 1, 0);
        for _ in 0..3 {
            rig.send(1);
            Rig::delivered(&rig.ticket);
        }
        rig.brain_calls(1);
        assert_eq!(rig.engrams.state(feature(1)), Some(EngramState::Dissolved));
        assert_eq!(rig.engrams.snapshot().dissolved_by_failures, 1);
        // The next signal goes back to the brain
        rig.send(1);
        let d = Rig::delivered(&rig.ticket);
        assert!(d.signal.trace.reached_brain());
        rig.brain_calls(2);
    }

    #[test]
    fn a_change_notification_removes_the_reflex_and_the_next_signal_asks_the_brain() {
        let rig = rig(
            quiet_policy()
                .after(2)
                .depends_on(["customers"])
                .tags(|s| vec![format!("customer:{}", s.feature.sketch)]),
            &[],
        );
        rig.consolidate(42, 2, 0);
        rig.consolidate(43, 2, 2);
        rig.send(42);
        assert!(!Rig::delivered(&rig.ticket).signal.trace.reached_brain());
        // The database row of customer 42 changed: a notification arrives
        let change = Signal::new(
            Feature::NONE,
            Payload::json(serde_json::json!({"inhibit": "customer:42"})),
        );
        rig.fabric.stimulate(rig.changes, change);
        assert_eq!(rig.engrams.state(feature(42)), Some(EngramState::Dissolved));
        // The next signal for customer 42 goes to the brain; customer 43's stays a reflex
        rig.send(42);
        assert!(Rig::delivered(&rig.ticket).signal.trace.reached_brain());
        rig.brain_calls(5);
        rig.send(43);
        assert!(!Rig::delivered(&rig.ticket).signal.trace.reached_brain());
        rig.brain_calls(5);
        // The whole table changed (a migration): a new version, every engram void
        let change = Signal::new(
            Feature::NONE,
            Payload::json(serde_json::json!({"advance": ["customers"]})),
        );
        rig.fabric.stimulate(rig.changes, change);
        assert_eq!(rig.engrams.version("customers"), Some(1));
        rig.send(43);
        assert!(Rig::delivered(&rig.ticket).signal.trace.reached_brain());
        rig.brain_calls(6);
        let shown = rig.fabric.describe();
        let recall = rig.fabric.neuron("brain-recall").unwrap();
        assert_eq!(
            shown.neurons[recall.index()].describe["consolidation"]["depends_on"]["customers"],
            1
        );
    }
}
