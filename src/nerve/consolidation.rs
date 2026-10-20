//! Consolidation: the brain's repeated decisions become reflexes.
//!
//! The fabric watches what the brain decides for each kind of signal (level 1: the signal's
//! [`Feature`], exactly). Once the brain has decided the same thing for the same feature
//! [`Consolidation::after`] times in a row, the decision is an *engram*, and the next signal
//! with that feature is answered by [`Recall`] in front of the brain, in microseconds and
//! without it. A sample of those answers is still sent to the brain after the reflex acted
//! (a *shadow check*); the brain's answer is compared, never acted on, and an engram the
//! brain disagrees with is dissolved. So is one whose effector keeps failing, and one not
//! seen for [`Consolidation::forget_after`].
//!
//! ```text
//! receptor ──► Recall ──(consolidated engram)──────────────────────────► effectors
//!                │
//!                └──(miss, or shadow check)──► Record(brain) ──(commands)──► effectors
//!                     deliberative pathway          │
//!                                                   ▼
//!                                              EngramStore
//! ```
//!
//! [`Record`] wraps the brain rather than following it: a brain's commands are routed by
//! name straight to their effectors, so a neuron on a pathway behind the brain would never
//! see them. Wrapping it sees each firing next to the signal that caused it, which is the
//! decision.
//!
//! Consolidation is opt-in per effector ([`Consolidation::effectors`]): a decision with a
//! command for any other effector is never answered from an engram.
//!
//! ```no_run
//! use std::time::Duration;
//! use furnace::nerve::consolidation::{Consolidation, EngramStore};
//! use furnace::nerve::{effector, Fabric, Firing, FnNeuron, Mechanisms, NeuronKind, Summation};
//!
//! let engrams = EngramStore::new(
//!     Consolidation::new()
//!         .effectors(["ticket", "ignore"])
//!         .after(3)
//!         .shadow(0.05)
//!         .forget_after(Duration::from_secs(3600)),
//! );
//! let mut b = Fabric::builder();
//! let (ticket, _deliveries) = effector::Channel::new("ticket", 64);
//! // Watched: the effector's failures reach the engrams it acts for
//! b.neuron(effector::Effector::new(engrams.watch(ticket)), Mechanisms::NONE);
//! let brain = FnNeuron::reflex("brain", |s, _| Firing::one(s.command("ticket", serde_json::json!({}))))
//!     .with_kind(NeuronKind::Brain);
//! let (recall, _brain) = b.consolidate(brain, Mechanisms::NONE, Summation::default(), &engrams);
//! // ... b.reflex(receptor, recall, 1.0)
//! let _ = recall;
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

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

/// When and what the brain's decisions may become reflexes: the policy of an [`EngramStore`]
#[derive(Clone, Debug, PartialEq)]
pub struct Consolidation {
    /// Effectors whose commands may be answered from an engram; nothing else ever is
    pub effectors: Vec<String>,
    /// Agreeing brain decisions, in a row, before a feature's decision is consolidated.
    /// After each dissolution of the same feature it takes this many more.
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
    /// a feature's set is replaced
    pub capacity: usize,
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
        }
    }
}

impl Consolidation {
    /// The defaults, with no effector allowed: nothing consolidates until some are
    pub fn new() -> Self {
        Self::default()
    }

    pub fn effectors<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.effectors = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn after(mut self, n: u32) -> Self {
        self.after = n.max(1);
        self
    }

    pub fn shadow(mut self, fraction: f64) -> Self {
        self.shadow = fraction.clamp(0.0, 1.0);
        self
    }

    pub fn first_shadows(mut self, n: u32) -> Self {
        self.first_shadows = n;
        self
    }

    pub fn forget_after(mut self, ttl: Duration) -> Self {
        self.forget_after = ttl;
        self
    }

    pub fn failures(mut self, k: u32) -> Self {
        self.failures = k.max(1);
        self
    }

    pub fn volatile<S: Into<String>>(mut self, fields: impl IntoIterator<Item = S>) -> Self {
        self.volatile = fields.into_iter().map(Into::into).collect();
        self
    }

    pub fn capacity(mut self, n: usize) -> Self {
        self.capacity = n.max(WAYS);
        self
    }
}

/// Where an engram is in its life
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EngramState {
    /// The brain has decided the same thing a few times, not yet enough
    Forming,
    /// Answered by the reflex
    Consolidated,
    /// The brain disagreed on a check, or the effector kept failing: back to the brain
    Dissolved,
}

#[derive(Clone)]
struct Engram {
    feature: Feature,
    state: EngramState,
    /// Normalized decision
    decision: u64,
    /// The last brain decision with that normalized form, as fired
    commands: Arc<[Arc<Command>]>,
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
            feature: Feature::NONE,
            state: EngramState::Forming,
            decision: 0,
            commands: Arc::from(Vec::new()),
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
    /// Engrams dissolved, by disagreement or by effector failures
    pub dissolved: u64,
    /// Of those, by effector failures
    pub dissolved_by_failures: u64,
    /// Engrams forgotten after `forget_after` without being seen
    pub forgotten: u64,
    /// Brain decisions that named an effector not allowed to consolidate
    pub forbidden: u64,
    /// Brain decisions recorded
    pub recorded: u64,
    /// Engrams in the store now, and of those consolidated
    pub engrams: u64,
    pub reflexes: u64,
    /// `brain / signals`
    pub brain_fraction: f64,
    /// `(brain + shadows) / signals`: the brain's whole load
    pub brain_load: f64,
    /// `brain / signals` per window of [`WINDOW`] signals, oldest first (the last few)
    pub timeline: Vec<f64>,
    /// Signal-to-decision time through the brain (p50), and what the hits saved at that
    pub brain_p50_ms: Option<f64>,
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
    counts: Counts,
}

impl Store {
    fn set(&self, feature: Feature) -> std::ops::Range<usize> {
        let set = (feature.sketch ^ (u64::from(feature.kind) << 56)) as usize % self.sets;
        set * WAYS..(set + 1) * WAYS
    }

    /// The live engram for `feature`, forgetting it first if it has not been seen for `ttl`
    fn find(&mut self, feature: Feature, now: u64, ttl: u64) -> Option<usize> {
        let range = self.set(feature);
        let i = range
            .clone()
            .find(|&i| self.slots[i].live() && self.slots[i].feature == feature)?;
        if now.saturating_sub(self.slots[i].seen) > ttl {
            self.slots[i] = Engram::empty();
            self.counts.forgotten += 1;
            return None;
        }
        Some(i)
    }

    /// A slot for a new engram: an empty one first, else the least recently seen (an
    /// expired one counts as forgotten)
    fn claim(&mut self, feature: Feature, now: u64, ttl: u64) -> usize {
        let range = self.set(feature);
        let victim = range
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
        engram.state = EngramState::Dissolved;
        engram.agree = 0;
        engram.dissolutions = engram.dissolutions.saturating_add(1);
        engram.failures = 0;
        self.counts.dissolved += 1;
        if by_failures {
            self.counts.dissolved_by_failures += 1;
        }
    }
}

/// What [`EngramStore::recall`] found for a signal
#[derive(Debug, Clone, PartialEq)]
pub enum Recalled {
    /// Ask the brain
    Miss,
    /// Fire these commands; with `shadow`, also send the signal to the brain to check
    Hit {
        commands: Arc<[Arc<Command>]>,
        shadow: bool,
    },
}

struct Shared {
    policy: Consolidation,
    epoch: Instant,
    ttl: u64,
    store: Mutex<Store>,
    /// Signal-to-decision time of the brain's decisions
    latency: LatencyEstimate,
}

/// The engrams of one brain: bounded and set-associative like a habituation table, shared
/// by its [`Recall`], its [`Record`] and the effectors it [`watch`](EngramStore::watch)es.
/// Cloning shares the same store.
#[derive(Clone)]
pub struct EngramStore(Arc<Shared>);

impl EngramStore {
    pub fn new(policy: Consolidation) -> Self {
        let sets = policy.capacity.div_ceil(WAYS).max(1);
        Self(Arc::new(Shared {
            ttl: u64::try_from(policy.forget_after.as_nanos()).unwrap_or(u64::MAX),
            policy,
            epoch: Instant::now(),
            store: Mutex::new(Store {
                slots: vec![Engram::empty(); sets * WAYS],
                sets,
                pending: HashMap::new(),
                pending_order: VecDeque::new(),
                counts: Counts::default(),
            }),
            latency: LatencyEstimate::new(),
        }))
    }

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

    /// Wrap an effector's act so that its failures reach the engrams it acts for: `failures`
    /// in a row for one feature dissolve that feature's engram
    pub fn watch<A: Act>(&self, act: A) -> Watched<A> {
        Watched {
            act,
            engrams: self.clone(),
        }
    }

    /// Look a signal up: a consolidated engram for its feature answers it
    pub fn recall(&self, feature: Feature, id: SignalId, now: Instant) -> Recalled {
        let now = self.nanos(now);
        let policy = &self.0.policy;
        let mut store = self.lock();
        store.counts.signals += 1;
        store.counts.window_signals += 1;
        let found = if feature.is_none() {
            None
        } else {
            store.find(feature, now, self.0.ttl)
        };
        let recalled = match found {
            Some(i) if store.slots[i].state == EngramState::Consolidated => {
                let engram = &mut store.slots[i];
                engram.seen = now;
                engram.hits += 1;
                let shadow =
                    engram.hits <= u64::from(policy.first_shadows) || sample(id.0, policy.shadow);
                let (commands, decision) = (engram.commands.clone(), engram.decision);
                store.counts.hits += 1;
                if shadow {
                    store.counts.shadows += 1;
                    store.pending.insert(id, (feature, decision));
                    store.pending_order.push_back(id);
                    // A check that never came back (its queue was full) is let go
                    while store.pending_order.len() > store.slots.len()
                        && let Some(old) = store.pending_order.pop_front()
                    {
                        store.pending.remove(&old);
                    }
                }
                Recalled::Hit { commands, shadow }
            }
            _ => {
                store.counts.brain += 1;
                store.counts.window_brain += 1;
                Recalled::Miss
            }
        };
        if store.counts.window_signals >= WINDOW {
            let fraction = store.counts.window_brain as f64 / store.counts.window_signals as f64;
            let counts = &mut store.counts;
            counts.timeline.push_back(fraction);
            if counts.timeline.len() > WINDOWS {
                counts.timeline.pop_front();
            }
            counts.window_signals = 0;
            counts.window_brain = 0;
        }
        recalled
    }

    /// The brain decided `decision` for the signal `id` with `feature`: either the answer to a
    /// shadow check (compared, and `false`: not to be acted on) or a decision to record
    /// (`true`: act on it). `None` is a firing that is not a decision (no commands, or a
    /// signal that is not a command).
    pub fn decided(
        &self,
        feature: Feature,
        id: SignalId,
        decision: Option<&[&Command]>,
        now: Instant,
    ) -> bool {
        let now = self.nanos(now);
        let policy = &self.0.policy;
        let normalized = decision.map(|d| normalize(d, &policy.volatile));
        let mut store = self.lock();
        if let Some((feature, expected)) = store.pending.remove(&id) {
            store.pending_order.retain(|&p| p != id);
            if normalized == Some(expected) {
                store.counts.agreements += 1;
            } else {
                store.counts.disagreements += 1;
                if let Some(i) = store.find(feature, now, self.0.ttl)
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
        if feature.is_none() {
            return true;
        }
        if decision
            .iter()
            .any(|c| !policy.effectors.iter().any(|e| *e == c.effector))
        {
            store.counts.forbidden += 1;
            return true;
        }
        store.counts.recorded += 1;
        let commands: Arc<[Arc<Command>]> = decision.iter().map(|&c| Arc::new(c.clone())).collect();
        let i = match store.find(feature, now, self.0.ttl) {
            Some(i) => i,
            None => {
                let i = store.claim(feature, now, self.0.ttl);
                store.slots[i] = Engram {
                    feature,
                    decision: normalized,
                    commands: commands.clone(),
                    ..Engram::empty()
                };
                i
            }
        };
        let needed = policy
            .after
            .saturating_mul(store.slots[i].dissolutions.saturating_add(1));
        let engram = &mut store.slots[i];
        engram.seen = now;
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

    /// An effector acted for a signal with `feature`, and succeeded or failed
    pub fn outcome(&self, feature: Feature, ok: bool, now: Instant) {
        if feature.is_none() {
            return;
        }
        let now = self.nanos(now);
        let limit = self.0.policy.failures;
        let mut store = self.lock();
        let Some(i) = store.find(feature, now, self.0.ttl) else {
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

    /// The state of the engram for `feature`, if there is one
    pub fn state(&self, feature: Feature) -> Option<EngramState> {
        let store = self.lock();
        store
            .slots
            .get(store.set(feature))?
            .iter()
            .find(|e| e.live() && e.feature == feature)
            .map(|e| e.state)
    }

    pub fn metrics(&self) -> ConsolidationMetrics {
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
            forgotten: c.forgotten,
            forbidden: c.forbidden,
            recorded: c.recorded,
            engrams: live.clone().count() as u64,
            reflexes: live
                .filter(|e| e.state == EngramState::Consolidated)
                .count() as u64,
            brain_fraction: ratio(c.brain),
            brain_load: ratio(c.brain + c.shadows),
            timeline: c.timeline.iter().copied().collect(),
            brain_p50_ms: p50.map(|d| d.as_secs_f64() * 1e3),
            brain_time_saved_ms: p50.map_or(0.0, |d| d.as_secs_f64() * 1e3 * c.hits as f64),
        }
    }

    /// The policy, the counters and the most used engrams, for `GET /nerve`
    pub fn describe(&self) -> serde_json::Value {
        let now = self.nanos(Instant::now());
        let metrics = self.metrics();
        let store = self.lock();
        let mut top: Vec<&Engram> = store.slots.iter().filter(|e| e.live()).collect();
        top.sort_by_key(|e| (std::cmp::Reverse(e.hits), std::cmp::Reverse(e.agree)));
        let top: Vec<serde_json::Value> = top
            .into_iter()
            .take(TOP)
            .map(|e| {
                serde_json::json!({
                    "feature": format!("{}:{:016x}", e.feature.kind, e.feature.sketch),
                    "state": e.state,
                    "hits": e.hits,
                    "agree": e.agree,
                    "dissolutions": e.dissolutions,
                    "effectors": e.commands.iter().map(|c| c.effector.as_str()).collect::<Vec<_>>(),
                    "idle_s": now.saturating_sub(e.seen) as f64 / 1e9,
                    "age_s": (e.state == EngramState::Consolidated)
                        .then(|| now.saturating_sub(e.formed) as f64 / 1e9),
                })
            })
            .collect();
        let policy = &self.0.policy;
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
            },
            "metrics": metrics,
            "top": top,
        })
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
                let mut keys: Vec<&String> = map
                    .keys()
                    .filter(|k| !volatile.iter().any(|v| v == *k))
                    .collect();
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

/// The reflex in front of a brain: a signal whose feature has a consolidated engram fires
/// the engram's commands at once; any other goes on to the brain. A shadow check fires the
/// commands and also sends the signal on, without its reply (the reflex answered).
pub struct Recall {
    name: String,
    engrams: EngramStore,
}

impl Recall {
    pub fn new(name: &str, engrams: &EngramStore) -> Self {
        Self {
            name: name.to_string(),
            engrams: engrams.clone(),
        }
    }
}

impl Neuron for Recall {
    fn name(&self) -> &str {
        &self.name
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        match self.engrams.recall(signal.feature, signal.id, cx.now) {
            Recalled::Miss => Firing::one(signal),
            Recalled::Hit { commands, shadow } => {
                let mut fired: Vec<Signal> = commands
                    .iter()
                    .map(|c| signal.derive(Payload::Command(c.clone())))
                    .collect();
                if shadow {
                    let mut check = signal;
                    check.reply = None;
                    fired.push(check);
                }
                Firing::Fire(fired)
            }
        }
    }

    fn describe(&self) -> serde_json::Value {
        self.engrams.describe()
    }

    fn metrics(&self) -> serde_json::Value {
        serde_json::to_value(self.engrams.metrics()).unwrap_or_default()
    }
}

/// A brain whose decisions are recorded: each firing, next to the signal that caused it, is
/// a decision for that signal's feature. A shadow check's firing is compared and dropped.
pub struct Record<N> {
    brain: N,
    engrams: EngramStore,
}

impl<N: Neuron> Record<N> {
    pub fn new(brain: N, engrams: &EngramStore) -> Self {
        Self {
            brain,
            engrams: engrams.clone(),
        }
    }
}

/// What a decision is recorded by: the signal's feature, id and time of stimulus
#[derive(Clone, Copy)]
struct Cause {
    feature: Feature,
    id: SignalId,
    at: Instant,
}

impl Cause {
    fn of(signal: &Signal) -> Self {
        Self {
            feature: signal.feature,
            id: signal.id,
            at: signal.at,
        }
    }
}

fn record(engrams: &EngramStore, cause: Cause, fired: Vec<Signal>) -> Vec<Signal> {
    let now = Instant::now();
    let commands: Option<Vec<&Command>> = fired.iter().map(|s| s.payload.as_command()).collect();
    let decision = commands.as_deref().filter(|c| !c.is_empty());
    let act = engrams.decided(cause.feature, cause.id, decision, now);
    if !act {
        return Vec::new();
    }
    if decision.is_some() {
        engrams
            .0
            .latency
            .observe(now.saturating_duration_since(cause.at));
    }
    fired
}

impl<N> Record<N> {
    fn observe(&self, cause: Cause, firing: Firing) -> Firing {
        match firing {
            Firing::Quiet => {
                // A check the brain had nothing to say to disagrees; anything else stays quiet
                if !self
                    .engrams
                    .decided(cause.feature, cause.id, None, Instant::now())
                {
                    return Firing::Fire(Vec::new());
                }
                Firing::Quiet
            }
            Firing::Fire(fired) => Firing::Fire(record(&self.engrams, cause, fired)),
            Firing::Compute(work) => {
                let engrams = self.engrams.clone();
                Firing::compute(move || record(&engrams, cause, work()))
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
        let cause = Cause::of(&signal);
        let firing = self.brain.stimulate(signal, cx);
        self.observe(cause, firing)
    }

    fn stimulate_many(&self, signals: Vec<Signal>, cx: &Cx<'_>) -> Vec<Firing> {
        let causes: Vec<Cause> = signals.iter().map(Cause::of).collect();
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

/// An effector's act whose outcomes reach an [`EngramStore`], see [`EngramStore::watch`].
/// Every attempt counts, retries included.
pub struct Watched<A> {
    act: A,
    engrams: EngramStore,
}

impl<A: Act> Act for Watched<A> {
    fn name(&self) -> &str {
        self.act.name()
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let feature = signal.feature;
        match self.act.act(command, signal) {
            Action::Done(result) => {
                self.engrams
                    .outcome(feature, result.is_ok(), Instant::now());
                Action::Done(result)
            }
            Action::Pending(future) => {
                let engrams = self.engrams.clone();
                Action::pending(async move {
                    let result = future.await;
                    engrams.outcome(feature, result.is_ok(), Instant::now());
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
        engrams: &EngramStore,
    ) -> (NeuronId, NeuronId) {
        let recall = Recall::new(&format!("{}-recall", brain.name()), engrams);
        let recall = self.neuron(recall, Mechanisms::NONE);
        let brain = self.neuron(Record::new(brain, engrams), mechanisms);
        self.deliberative(recall, brain, 1.0, summation);
        (recall, brain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::effector::{self, ActError, Effector};
    use crate::nerve::{Fabric, FnNeuron};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    const SECOND: Duration = Duration::from_secs(1);

    fn feature(n: u64) -> Feature {
        Feature::new(7, n)
    }

    fn command(effector: &str, args: serde_json::Value) -> Command {
        Command::new(effector, args)
    }

    /// Record `decision` for `feature` as the brain's, at `at` after the epoch
    fn decide(store: &EngramStore, f: Feature, decision: &[&Command], at: Duration) -> bool {
        let id = SignalId(u64::MAX - at.as_nanos() as u64);
        store.decided(f, id, Some(decision), store.epoch() + at)
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
        let Recalled::Hit { commands, .. } =
            store.recall(f, SignalId(1), store.epoch() + 6 * SECOND)
        else {
            panic!("a consolidated engram answers");
        };
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].effector, "page");
        assert_eq!(store.metrics().consolidated, 1);
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
        let Recalled::Hit { commands, .. } =
            store.recall(f, SignalId(1), store.epoch() + 3 * SECOND)
        else {
            panic!("consolidated");
        };
        assert_eq!(commands[0].args["at"], 2);
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
        assert_eq!(
            store.recall(f, SignalId(1), store.epoch() + 6 * SECOND),
            Recalled::Miss
        );
        assert_eq!(store.metrics().forbidden, 5);
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
        let at = |d: Duration| store.epoch() + d;
        decide(&store, f, &[&ticket], SECOND);
        // Used within the ttl, again and again: alive
        for s in [9, 18, 27] {
            assert!(matches!(
                store.recall(f, SignalId(s), at(Duration::from_secs(s))),
                Recalled::Hit { .. }
            ));
        }
        // Then idle for longer than the ttl: forgotten, back to the brain
        assert_eq!(
            store.recall(f, SignalId(99), at(38 * SECOND)),
            Recalled::Miss
        );
        assert_eq!(store.state(f), None);
        assert_eq!(store.metrics().forgotten, 1);
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
        let m = store.metrics();
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

    // ---------------------------------------------------------------------------------------
    // In a fabric, with a brain that counts its calls

    struct Rig {
        fabric: Fabric,
        input: NeuronId,
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
        let (page_act, page) = effector::Channel::new("page", 64);
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
        Rig {
            fabric: b.build(),
            input,
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
                let m = self.engrams.metrics();
                if m.agreements + m.disagreements == m.shadows || Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
        }
    }

    #[test]
    fn a_consolidated_decision_answers_without_the_brain() {
        let rig = rig(
            Consolidation::new()
                .effectors(["ticket", "page"])
                .after(3)
                .shadow(0.0)
                .first_shadows(0),
            &[],
        );
        for i in 1..=3 {
            rig.send(1);
            let d = Rig::delivered(&rig.ticket);
            assert!(d.signal.trace.reached_brain());
            rig.brain_calls(i);
        }
        assert_eq!(
            rig.engrams.state(feature(1)),
            Some(EngramState::Consolidated)
        );
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
        let m = rig.engrams.metrics();
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
        let rig = rig(
            Consolidation::new()
                .effectors(["ticket"])
                .after(2)
                .shadow(0.0)
                .first_shadows(2),
            &[],
        );
        for i in 1..=2 {
            rig.send(1);
            Rig::delivered(&rig.ticket);
            rig.brain_calls(i);
        }
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
        let m = rig.engrams.metrics();
        assert_eq!(
            (m.hits, m.shadows, m.agreements, m.disagreements),
            (3, 2, 2, 0)
        );
        assert_eq!(m.brain_load, 4.0 / 5.0);
    }

    #[test]
    fn a_disagreeing_check_dissolves_the_engram() {
        let rig = rig(
            Consolidation::new()
                .effectors(["ticket", "page"])
                .after(2)
                .shadow(1.0),
            &[],
        );
        for i in 1..=2 {
            rig.send(1);
            Rig::delivered(&rig.ticket);
            rig.brain_calls(i);
        }
        // The world changed: the brain now pages for this feature
        *rig.answer.lock().unwrap() = "page";
        rig.send(1);
        // The reflex still files a ticket; the check disagrees and dissolves it
        Rig::delivered(&rig.ticket);
        rig.brain_calls(3);
        rig.checks_settled();
        assert!(
            rig.page.try_recv().is_err(),
            "the check's answer is not acted on"
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
        let m = rig.engrams.metrics();
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
        let m = rig.engrams.metrics();
        assert_eq!((m.hits, m.forbidden, m.consolidated), (0, 5, 0));
    }

    #[test]
    fn an_effector_that_keeps_failing_dissolves_its_engram() {
        // The brain's own decision acts and succeeds; the reflex's next three acts fail
        let rig = rig(
            Consolidation::new()
                .effectors(["ticket"])
                .after(1)
                .shadow(0.0)
                .first_shadows(0)
                .failures(3),
            &[true, false, false, false],
        );
        rig.send(1);
        Rig::delivered(&rig.ticket);
        rig.brain_calls(1);
        assert_eq!(
            rig.engrams.state(feature(1)),
            Some(EngramState::Consolidated)
        );
        for _ in 0..3 {
            rig.send(1);
            Rig::delivered(&rig.ticket);
        }
        rig.brain_calls(1);
        assert_eq!(rig.engrams.state(feature(1)), Some(EngramState::Dissolved));
        assert_eq!(rig.engrams.metrics().dissolved_by_failures, 1);
        // The next signal goes back to the brain
        rig.send(1);
        let d = Rig::delivered(&rig.ticket);
        assert!(d.signal.trace.reached_brain());
        rig.brain_calls(2);
    }
}
