//! Expectations (v1): the brain programs the periphery.
//!
//! Consolidation remembers the brain's past answers ([`super::consolidation`]); here the brain
//! writes down how the stream should behave from now on, and only what breaks that reaches it.
//! When the brain looks at something new it commands, as before, and also installs
//! *expectations* through the `expect` effector: small declarative rules (the
//! [`language`]) that run next to the receptor in microseconds. A signal that matches is
//! absorbed (or acted on by the rule's own commands) and goes no further. A signal that
//! breaks one, a signal no expectation covers, and an expected signal that *did not come*
//! become a [`Surprise`]: a JSON payload with what broke, the observation and a window of
//! recent signals of the same template, sent on to the brain.
//!
//! ```text
//!              top-down: expectations, through the `expect` effector
//!        ┌──────────────────────────────────────────────────────────────┐
//!        ▼                                                              │
//! receptor ─► Expectations ──matched: absorb / the rule's commands──► effectors
//!                │   ▲ deadlines (absences)                             │
//!                └──surprise──► brain ──commands──► effectors           │
//!                                    └──expect commands────────────────┘
//! ```
//!
//! Kept honest:
//! - **Shadow checks**: a sample of absorbed signals (1%, and every match of a young
//!   expectation) also goes to the brain as a `shadow` surprise. The brain answers with
//!   `{"verdict": {"id": ..., "correct": false}}`; [`ExpectConfig::retire_after`] such answers
//!   retire the expectation.
//! - **False alarms**: `{"verdict": {"id": ..., "normal": true}}` answers a violation; a
//!   parameter the expectation lets the periphery `learn` moves to cover what was observed,
//!   within the brain's bounds.
//! - **Scope limits**: an absorbing expectation must name a template or fields, at least as
//!   specifically as the surprise it answers (no wildcard absorb); its commands only go to
//!   [`ExpectConfig::effectors`]; at most [`ExpectConfig::max`] expectations and
//!   [`ExpectConfig::writes_per_minute`] writes; with [`ExpectConfig::approval`], new
//!   absorbing expectations wait for [`Expectations::approve`] by the operator.
//! - **Forgetting**: unmatched expectations expire after their `ttl_s` (or the default), and
//!   those that `depends_on` a state retire when it changes ([`Expectations::changed`]).
//!
//! ```no_run
//! use furnace::nerve::effector::Effector;
//! use furnace::nerve::expect::{ExpectConfig, Expectations};
//! use furnace::nerve::{Fabric, Mechanisms, Summation};
//! # fn brain() -> furnace::nerve::FnNeuron<fn(furnace::nerve::Signal, &furnace::nerve::Cx<'_>) -> furnace::nerve::Firing> {
//! #     furnace::nerve::FnNeuron::reflex("brain", |_, _| furnace::nerve::Firing::Quiet)
//! # }
//!
//! let mut b = Fabric::builder();
//! let logs = b.receptor(furnace::nerve::receptor::Lines::stdin("logs"), Mechanisms::NONE);
//! let expectations = Expectations::new("logs-expect", ExpectConfig::new().effectors(["ticket"]));
//! let expect = b.receptor(expectations.clone(), Mechanisms::NONE);
//! b.reflex(logs, expect, 1.0);
//! // The top-down channel: the brain commands `expect`
//! b.neuron(Effector::new(expectations.effector("expect")), Mechanisms::NONE);
//! let brain = b.neuron(brain(), Mechanisms::NONE);
//! b.deliberative(expect, brain, 1.0, Summation::default());
//! ```

pub mod fields;
pub mod language;
#[cfg(feature = "llm")]
pub mod llm;

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;

use self::fields::{Extraction, Observation, observe};
use self::language::{Expectation, Kind, OnMatch, OnViolation, Refused, Selector};
use super::effector::{Act, ActError, Action};
use super::neuron::{Cx, Firing, Neuron, NeuronKind};
use super::receptor::{Afferent, Receptor};
use super::signal::{Command, Feature, Payload, Signal, SignalId, sketch64};

/// The feature kind of an absence surprise, which the timer makes: it passes through
pub const ABSENCE: u16 = 0x5E;

/// Signals per window of the surprise-rate timeline
pub const WINDOW: u64 = 1000;
const WINDOWS: usize = 16;

/// Surprises remembered for the specificity rule and for verdicts
const REMEMBERED: usize = 4096;

/// Retirements shown in `GET /nerve`
const RETIRED_SHOWN: usize = 16;

/// The policy of an [`Expectations`] neuron
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectConfig {
    /// Effectors an expectation's own commands may go to; nothing else
    pub effectors: Vec<String>,
    /// How signals are read into templates and fields
    pub extraction: Extraction,
    /// Fraction of absorbed signals also sent to the brain as a shadow check
    pub shadow: f64,
    /// An expectation's first matches are all checked
    pub young: u64,
    /// Wrong shadow verdicts that retire an expectation
    pub retire_after: u64,
    /// Expectations installed and waiting for approval, at most
    pub max: usize,
    /// Writes (add, replace, retire) through the `expect` effector per minute, at most
    pub writes_per_minute: usize,
    /// New absorbing expectations from the brain wait for [`Expectations::approve`]
    pub approval: bool,
    /// An expectation not matched for this long expires, unless it sets its own `ttl_s`
    pub ttl: Duration,
    /// Recent signals of the same template given to the brain with a surprise
    pub context: usize,
}

impl Default for ExpectConfig {
    fn default() -> Self {
        Self {
            effectors: Vec::new(),
            extraction: Extraction::Auto,
            shadow: 0.01,
            young: 5,
            retire_after: 2,
            max: 256,
            writes_per_minute: 60,
            approval: false,
            ttl: Duration::from_secs(3600),
            context: 8,
        }
    }
}

impl ExpectConfig {
    /// The defaults: no effector allowed for an expectation's own commands
    pub fn new() -> Self {
        Self::default()
    }

    /// Effectors an expectation's own commands may go to
    pub fn effectors<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.effectors = names.into_iter().map(Into::into).collect();
        self
    }

    /// How signals are read
    pub fn extraction(mut self, how: Extraction) -> Self {
        self.extraction = how;
        self
    }

    /// Fraction of absorbed signals checked by the brain, clamped to 0..=1
    pub fn shadow(mut self, fraction: f64) -> Self {
        self.shadow = fraction.clamp(0.0, 1.0);
        self
    }

    /// First matches of an expectation that are all checked
    pub fn young(mut self, matches: u64) -> Self {
        self.young = matches;
        self
    }

    /// Wrong shadow verdicts that retire an expectation, at least 1
    pub fn retire_after(mut self, wrong: u64) -> Self {
        self.retire_after = wrong.max(1);
        self
    }

    /// Expectations installed and waiting for approval, at most
    pub fn max(mut self, n: usize) -> Self {
        self.max = n;
        self
    }

    /// Writes through the `expect` effector per minute, at most
    pub fn writes_per_minute(mut self, n: usize) -> Self {
        self.writes_per_minute = n;
        self
    }

    /// Whether new absorbing expectations from the brain wait for the operator
    pub fn approval(mut self, on: bool) -> Self {
        self.approval = on;
        self
    }

    /// How long an unmatched expectation lives, unless it sets its own `ttl_s`
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Recent signals of the same template sent with a surprise
    pub fn context(mut self, n: usize) -> Self {
        self.context = n;
        self
    }
}

/// Why a signal goes to the brain
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Reason {
    /// No expectation covers it
    Unexpected,
    /// It broke one
    Violation,
    /// Something expected did not come
    Absence,
    /// It was absorbed; is that right?
    Shadow,
}

/// One broken expectation in a surprise
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Broken {
    /// The broken expectation's id
    pub expectation: String,
    /// Its kind, as [`Kind::name`](language::Kind::name) writes it
    pub kind: &'static str,
    /// What was seen: a value, a count, a gap in milliseconds
    pub observed: Value,
    /// What the expectation allows: a value, a bound, a period
    pub expected: Value,
}

/// What reaches the brain: the payload of a surprise signal, as JSON under `"surprise"`
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Surprise {
    /// Why the signal goes to the brain
    pub reason: Reason,
    /// The receptor's expectations neuron
    pub at: String,
    /// The expectations it broke (empty unless a violation or an absence)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub broken: Vec<Broken>,
    /// For a shadow check: the expectation that absorbed it
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absorbed_by: Option<String>,
    /// The signal (none for an absence)
    #[serde(skip_serializing_if = "Value::is_null")]
    pub observation: Value,
    /// Recent signals of the same template, oldest first
    pub context: Vec<String>,
    /// The expectations installed for this template now
    pub expectations: Vec<Value>,
}

/// Counters of an [`Expectations`] neuron
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ExpectMetrics {
    /// Signals seen
    pub signals: u64,
    /// Matched and absorbed
    pub absorbed: u64,
    /// Matched, and acted on by the rule's commands
    pub acted: u64,
    /// Signals no expectation covers
    pub unexpected: u64,
    /// Violations, by the broken expectation's kind
    pub violations: BTreeMap<&'static str, u64>,
    /// Expected signals that did not come
    pub absences: u64,
    /// Signals sent to the brain: surprises and shadow checks
    pub surprises: u64,
    /// Of the surprises, shadow checks
    pub shadows: u64,
    /// Shadow verdicts saying the absorption was wrong
    pub shadow_wrong: u64,
    /// Violations the brain answered as normal
    pub false_alarms: u64,
    /// Parameters moved by `learn`
    pub learned: u64,
    /// Expectations installed new (replacements not counted)
    pub installed: u64,
    /// Expectations replaced
    pub replaced: u64,
    /// Expectations retired: by the brain, by the operator, by wrong verdicts or by a changed dependency
    pub retired: u64,
    /// Expectations expired after their ttl without a match
    pub expired: u64,
    /// Writes refused
    pub refused: u64,
    /// Expectations installed now
    pub active: u64,
    /// Expectations waiting for approval now
    pub pending_approval: u64,
    /// `(unexpected + violations + absences) / signals`
    pub surprise_rate: f64,
    /// The same per window of [`WINDOW`] signals, oldest first (the last few)
    pub timeline: Vec<f64>,
}

#[derive(Clone)]
struct Installed {
    expectation: Expectation,
    /// Nanoseconds since the neuron's epoch
    since: u64,
    last_match: u64,
    matches: u64,
    violations: u64,
    absences: u64,
    false_alarms: u64,
    shadows: u64,
    wrong: u64,
    /// Rate: the current window
    window_start: u64,
    window_count: u64,
    previous_count: u64,
    /// Then: deadlines of the signals waiting for what follows
    armed: VecDeque<u64>,
    /// Every: smoothed interval between matches, in nanoseconds
    interval: Option<f64>,
    /// What the last violation saw, for learning on a false alarm: (parameter, value)
    last_violation: Option<(&'static str, f64)>,
    /// Deadlines of an older version of this expectation are ignored
    version: u64,
}

impl Installed {
    fn new(expectation: Expectation, now: u64, version: u64) -> Self {
        Self {
            expectation,
            since: now,
            last_match: now,
            matches: 0,
            violations: 0,
            absences: 0,
            false_alarms: 0,
            shadows: 0,
            wrong: 0,
            window_start: now,
            window_count: 0,
            previous_count: 0,
            armed: VecDeque::new(),
            interval: None,
            last_violation: None,
            version,
        }
    }

    /// Move the rate window on to the one holding `t`; windows are aligned to installation.
    /// The window just ended keeps its count in `previous_count`.
    fn roll(&mut self, t: u64, window: u64) {
        let window = window.max(1);
        if t < self.window_start + window {
            return;
        }
        let k = (t - self.window_start) / window;
        self.previous_count = if k == 1 { self.window_count } else { 0 };
        self.window_start += k * window;
        self.window_count = 0;
    }

    /// The deadline `deadline` of expectation `id` passed: what broke, if anything. Deadlines
    /// to arm next go to `later`.
    fn absent(
        &mut self,
        id: &str,
        deadline: u64,
        now: u64,
        later: &mut Vec<(u64, String, u64)>,
    ) -> Option<Broken> {
        let learnable = |p: &str| self.expectation.learn.contains_key(p);
        match self.expectation.kind.clone() {
            Kind::Every {
                period_ms,
                tolerance_ms,
            } => {
                // Only the deadline of the last match counts
                if self.last_match + ms(period_ms + tolerance_ms) > deadline {
                    return None;
                }
                let gap = now.saturating_sub(self.last_match);
                let gap_ms = gap as f64 / 1e6;
                self.last_violation = Some(if learnable("tolerance_ms") {
                    ("tolerance_ms", (gap_ms - period_ms as f64).max(0.0))
                } else {
                    ("period_ms", gap_ms)
                });
                Some(Broken {
                    expectation: id.to_string(),
                    kind: "every",
                    observed: serde_json::json!({"silent_ms": gap / 1_000_000}),
                    expected: serde_json::json!({"period_ms": period_ms, "tolerance_ms": tolerance_ms}),
                })
            }
            Kind::Then { within_ms, .. } => {
                if self.armed.front() != Some(&deadline) {
                    return None;
                }
                self.armed.pop_front();
                self.last_violation = Some(("within_ms", (within_ms * 2) as f64));
                Some(Broken {
                    expectation: id.to_string(),
                    kind: "then",
                    observed: serde_json::json!({"followed": false}),
                    expected: serde_json::json!({"within_ms": within_ms}),
                })
            }
            Kind::Rate { window_ms, min, .. } => {
                let window = ms(window_ms);
                // The window that ends at this deadline
                self.roll(deadline, window);
                let count = self.previous_count;
                later.push((deadline + window, id.to_string(), self.version));
                if count >= min {
                    return None;
                }
                self.last_violation = Some(("min", count as f64));
                Some(Broken {
                    expectation: id.to_string(),
                    kind: "rate",
                    observed: Value::from(count),
                    expected: serde_json::json!({"min": min, "window_ms": window_ms}),
                })
            }
            Kind::Match | Kind::Range { .. } | Kind::Never => None,
        }
    }

    fn describe(&self, now: u64) -> Value {
        let e = &self.expectation;
        serde_json::json!({
            "expectation": e,
            "matches": self.matches,
            "violations": self.violations,
            "absences": self.absences,
            "false_alarms": self.false_alarms,
            "shadows": self.shadows,
            "wrong": self.wrong,
            "age_s": now.saturating_sub(self.since) as f64 / 1e9,
            "idle_s": now.saturating_sub(self.last_match) as f64 / 1e9,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    /// The expectation's own selector
    Primary,
    /// A `then` expectation's follow-up selector
    Follow,
}

#[derive(Default)]
struct Counts {
    signals: u64,
    absorbed: u64,
    acted: u64,
    unexpected: u64,
    violations: BTreeMap<&'static str, u64>,
    absences: u64,
    surprises: u64,
    shadows: u64,
    shadow_wrong: u64,
    false_alarms: u64,
    learned: u64,
    installed: u64,
    replaced: u64,
    retired: u64,
    expired: u64,
    refused: u64,
    window_signals: u64,
    window_surprises: u64,
    timeline: VecDeque<f64>,
}

#[derive(Default)]
struct State {
    installed: BTreeMap<String, Installed>,
    /// Template key → expectations whose selector names it
    by_template: HashMap<u64, Vec<(String, Role)>>,
    /// Expectations whose selector names no template
    general: Vec<(String, Role)>,
    pending_approval: BTreeMap<String, Expectation>,
    /// (deadline, expectation, version)
    deadlines: BinaryHeap<Reverse<(u64, String, u64)>>,
    next_version: u64,
    /// Times of recent writes
    writes: VecDeque<u64>,
    /// Template key → recent texts
    context: HashMap<u64, VecDeque<Arc<str>>>,
    /// Surprises sent: the selector an answer to them must be at least as specific as
    surprised: HashMap<SignalId, Selector>,
    surprised_order: VecDeque<SignalId>,
    retired_log: VecDeque<Value>,
    counts: Counts,
}

impl State {
    fn reindex(&mut self) {
        self.by_template.clear();
        self.general.clear();
        for (id, installed) in &self.installed {
            let e = &installed.expectation;
            let mut put = |selector: &Selector, role| match selector.template_key() {
                Some(key) => self
                    .by_template
                    .entry(key)
                    .or_default()
                    .push((id.clone(), role)),
                None => self.general.push((id.clone(), role)),
            };
            put(&e.selector, Role::Primary);
            if let Kind::Then { then, .. } = &e.kind {
                put(then, Role::Follow);
            }
        }
    }

    fn schedule(&mut self, deadline: u64, id: &str, version: u64) {
        self.deadlines
            .push(Reverse((deadline, id.to_string(), version)));
    }

    fn retire(&mut self, id: &str, reason: &str) -> bool {
        if self.installed.remove(id).is_none() {
            return false;
        }
        self.counts.retired += 1;
        self.retired_log
            .push_back(serde_json::json!({"id": id, "reason": reason}));
        if self.retired_log.len() > RETIRED_SHOWN {
            self.retired_log.pop_front();
        }
        self.reindex();
        true
    }

    fn remember_surprise(&mut self, id: SignalId, selector: Selector) {
        self.surprised.insert(id, selector);
        self.surprised_order.push_back(id);
        while self.surprised_order.len() > REMEMBERED
            && let Some(old) = self.surprised_order.pop_front()
        {
            self.surprised.remove(&old);
        }
    }
}

struct Inner {
    name: String,
    config: ExpectConfig,
    epoch: Instant,
    state: Mutex<State>,
    /// Woken when a deadline earlier than the timer's next is scheduled
    wake: Condvar,
}

/// Expectations next to a receptor: matches every signal against what the brain installed,
/// absorbs or acts on what fits and sends the rest, and the absences, to the brain as
/// [`Surprise`]s. Add it with [`FabricBuilder::receptor`](super::FabricBuilder::receptor)
/// (it is a receptor too: its timer raises absences) behind the receptor it watches, with
/// a deliberative pathway to the brain, and an [`Expectations::effector`] for the brain to
/// command. Cloning shares the same expectations.
#[derive(Clone)]
pub struct Expectations(Arc<Inner>);

/// What a signal did at the expectations
enum Verdict {
    Absorbed(Vec<Command>, Option<String>),
    Surprise(Reason, Vec<Broken>, Vec<Command>),
}

impl Expectations {
    /// An expectations neuron named `name`, with nothing installed
    pub fn new(name: &str, config: ExpectConfig) -> Self {
        Self(Arc::new(Inner {
            name: name.to_string(),
            config,
            epoch: Instant::now(),
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        }))
    }

    /// The top-down channel: an effector named `name` that installs, replaces and retires
    /// expectations, and takes the brain's verdicts
    pub fn effector(&self, name: &str) -> Expect {
        Expect {
            name: name.to_string(),
            expectations: self.clone(),
        }
    }

    /// The policy it was made with
    pub fn config(&self) -> &ExpectConfig {
        &self.0.config
    }

    /// When the neuron was made: times passed to it count from here
    pub fn epoch(&self) -> Instant {
        self.0.epoch
    }

    fn nanos(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.0.epoch).as_nanos()).unwrap_or(u64::MAX)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Install an expectation as the operator: the scope rules apply, the budget and approval
    /// do not
    pub fn install(&self, expectation: Expectation, now: Instant) -> Result<(), Refused> {
        self.write(Write::Add(expectation), None, false, now)
    }

    /// Approve an expectation waiting for approval
    pub fn approve(&self, id: &str, now: Instant) -> Result<(), Refused> {
        let expectation = self
            .lock()
            .pending_approval
            .remove(id)
            .ok_or_else(|| Refused(format!("nothing named '{id}' waits for approval")))?;
        self.write(Write::Add(expectation), None, false, now)
    }

    /// Reject an expectation waiting for approval
    pub fn reject(&self, id: &str) -> bool {
        self.lock().pending_approval.remove(id).is_some()
    }

    /// Retire an expectation
    pub fn retire(&self, id: &str) -> bool {
        self.lock().retire(id, "retired")
    }

    /// External state changed: retire every expectation that depends on it. Returns how many.
    pub fn changed(&self, dependency: &str) -> usize {
        let mut guard = self.lock();
        let state = &mut *guard;
        let ids: Vec<String> = state
            .installed
            .iter()
            .filter(|(_, i)| i.expectation.depends_on.iter().any(|d| d == dependency))
            .map(|(id, _)| id.clone())
            .collect();
        for id in &ids {
            state.retire(id, &format!("{dependency} changed"));
        }
        ids.len()
    }

    /// The installed expectations' ids
    pub fn ids(&self) -> Vec<String> {
        self.lock().installed.keys().cloned().collect()
    }

    /// Ids waiting for approval
    pub fn pending(&self) -> Vec<String> {
        self.lock().pending_approval.keys().cloned().collect()
    }

    /// The installed expectation `id`
    pub fn get(&self, id: &str) -> Option<Expectation> {
        self.lock().installed.get(id).map(|i| i.expectation.clone())
    }

    fn write(
        &self,
        write: Write,
        answering: Option<SignalId>,
        from_brain: bool,
        now: Instant,
    ) -> Result<(), Refused> {
        let config = &self.0.config;
        let now = self.nanos(now);
        let mut guard = self.lock();
        let state = &mut *guard;
        let result = (|| {
            if from_brain {
                while state
                    .writes
                    .front()
                    .is_some_and(|&t| now.saturating_sub(t) >= 60_000_000_000)
                {
                    state.writes.pop_front();
                }
                if state.writes.len() >= config.writes_per_minute {
                    return Err(Refused("over the budget of writes per minute".into()));
                }
            }
            match write {
                Write::Retire(id) => {
                    if state.retire(&id, "retired by the brain") {
                        Ok(())
                    } else {
                        Err(Refused(format!("no expectation named '{id}'")))
                    }
                }
                Write::Add(expectation) | Write::Replace(expectation) => {
                    let replace = state.installed.contains_key(&expectation.id);
                    let surprise = answering.and_then(|id| state.surprised.get(&id).cloned());
                    expectation.check(&config.effectors, surprise.as_ref())?;
                    if !replace
                        && state.installed.len() + state.pending_approval.len() >= config.max
                    {
                        return Err(Refused(format!(
                            "over the budget of {} expectations",
                            config.max
                        )));
                    }
                    if from_brain && config.approval && expectation.absorbs() {
                        state
                            .pending_approval
                            .insert(expectation.id.clone(), expectation);
                        return Ok(());
                    }
                    self.put(state, expectation, now, replace);
                    Ok(())
                }
            }
        })();
        // Only accepted writes count against the budget: a brain retrying refused ones on
        // every surprise would otherwise keep itself locked out
        match &result {
            Ok(()) if from_brain => state.writes.push_back(now),
            Ok(()) => {}
            Err(_) => state.counts.refused += 1,
        }
        result
    }

    /// Install (or replace) `expectation` and arm its deadlines
    fn put(&self, state: &mut State, expectation: Expectation, now: u64, replace: bool) {
        state.next_version += 1;
        let version = state.next_version;
        let id = expectation.id.clone();
        let first = match &expectation.kind {
            Kind::Every {
                period_ms,
                tolerance_ms,
            } => Some(now + ms(*period_ms + *tolerance_ms)),
            Kind::Rate { window_ms, min, .. } if *min > 0 => Some(now + ms(*window_ms)),
            _ => None,
        };
        state
            .installed
            .insert(id.clone(), Installed::new(expectation, now, version));
        if replace {
            state.counts.replaced += 1;
        } else {
            state.counts.installed += 1;
        }
        state.reindex();
        if let Some(deadline) = first {
            state.schedule(deadline, &id, version);
            self.0.wake.notify_one();
        }
    }

    /// Match a signal at `now`
    fn take(
        &self,
        signal: &Signal,
        now: Instant,
    ) -> (Verdict, Observation, Vec<String>, Vec<Value>) {
        let config = &self.0.config;
        let observation = observe(signal, config.extraction);
        let now = self.nanos(now);
        let mut guard = self.lock();
        let state = &mut *guard;
        state.counts.signals += 1;
        state.counts.window_signals += 1;
        // Context: what came before, then this
        let key = observation.template.unwrap_or(signal.feature.sketch);
        let context: Vec<String> = state
            .context
            .get(&key)
            .map(|c| c.iter().map(|t| t.to_string()).collect())
            .unwrap_or_default();
        if config.context > 0 {
            if state.context.len() > REMEMBERED && !state.context.contains_key(&key) {
                state.context.clear();
            }
            let recent = state.context.entry(key).or_default();
            recent.push_back(Arc::from(observation.text.as_deref().unwrap_or_default()));
            if recent.len() > config.context {
                recent.pop_front();
            }
        }
        let candidates: Vec<(String, Role)> = observation
            .template
            .and_then(|t| state.by_template.get(&t))
            .into_iter()
            .flatten()
            .chain(state.general.iter())
            .cloned()
            .collect();
        let mut matched: Vec<String> = Vec::new();
        let mut broken: Vec<Broken> = Vec::new();
        let mut expired: Vec<String> = Vec::new();
        let mut schedule: Vec<(u64, String, u64)> = Vec::new();
        let mut commands: Vec<Command> = Vec::new();
        let mut on_violation: Vec<Command> = Vec::new();
        let mut learned = 0;
        for (id, role) in candidates {
            let Some(installed) = state.installed.get_mut(&id) else {
                continue;
            };
            let ttl = installed
                .expectation
                .ttl_s
                .map_or(config.ttl, Duration::from_secs);
            if now.saturating_sub(installed.last_match)
                > u64::try_from(ttl.as_nanos()).unwrap_or(u64::MAX)
            {
                expired.push(id);
                continue;
            }
            let selector = match (&installed.expectation.kind, role) {
                (Kind::Then { then, .. }, Role::Follow) => then,
                _ => &installed.expectation.selector,
            };
            if !selector.matches(&observation) {
                continue;
            }
            let previous = installed.last_match;
            installed.last_match = now;
            let version = installed.version;
            let mut violation: Option<Broken> = None;
            let e = &mut installed.expectation;
            match (&mut e.kind, role) {
                (Kind::Then { .. }, Role::Follow) => {
                    installed.armed.pop_front();
                }
                (Kind::Then { within_ms, .. }, Role::Primary) => {
                    let deadline = now + ms(*within_ms);
                    installed.armed.push_back(deadline);
                    schedule.push((deadline, id.clone(), version));
                }
                (Kind::Match, _) => {}
                (Kind::Never, _) => {
                    violation = Some(Broken {
                        expectation: id.clone(),
                        kind: "never",
                        observed: Value::from(1),
                        expected: Value::from(0),
                    });
                }
                (Kind::Range { field, min, max }, _) => {
                    let value = observation.number(field);
                    match value {
                        Some(v) if v >= *min && v <= *max => {}
                        _ => {
                            installed.last_violation =
                                value.map(|v| (if v > *max { "max" } else { "min" }, v));
                            violation = Some(Broken {
                                expectation: id.clone(),
                                kind: "range",
                                observed: value.map_or(Value::Null, Value::from),
                                expected: serde_json::json!([*min, *max]),
                            });
                        }
                    }
                }
                (Kind::Rate { window_ms, max, .. }, _) => {
                    let (window_ms, max) = (*window_ms, *max);
                    installed.roll(now, ms(window_ms));
                    installed.window_count += 1;
                    if installed.window_count > max {
                        installed.last_violation = Some(("max", installed.window_count as f64));
                        violation = Some(Broken {
                            expectation: id.clone(),
                            kind: "rate",
                            observed: Value::from(installed.window_count),
                            expected: serde_json::json!({"max": max, "window_ms": window_ms}),
                        });
                    }
                }
                (
                    Kind::Every {
                        period_ms,
                        tolerance_ms,
                    },
                    _,
                ) => {
                    // Statistics set the numbers: the period follows the observed interval
                    // where the brain let it
                    if installed.matches > 0 {
                        let gap = now.saturating_sub(previous) as f64;
                        let smoothed = installed.interval.map_or(gap, |i| 0.8 * i + 0.2 * gap);
                        installed.interval = Some(smoothed);
                    }
                    let deadline = now + ms(*period_ms + *tolerance_ms);
                    schedule.push((deadline, id.clone(), version));
                    if let Some(interval) = installed.interval
                        && e.learn.contains_key("period_ms")
                    {
                        let before = e.param("period_ms");
                        if e.tune("period_ms", interval / 1e6) && e.param("period_ms") != before {
                            learned += 1;
                        }
                    }
                }
            }
            match violation {
                Some(b) => {
                    installed.violations += 1;
                    if let OnViolation::Act(c) = &installed.expectation.on_violation {
                        on_violation.extend(c.iter().cloned());
                    }
                    broken.push(b);
                }
                None => {
                    installed.matches += 1;
                    if let OnMatch::Act(c) = &installed.expectation.on_match {
                        commands.extend(c.iter().cloned());
                    }
                    matched.push(id);
                }
            }
        }
        if !schedule.is_empty() {
            for (deadline, id, version) in schedule {
                state.schedule(deadline, &id, version);
            }
            // The timer may be asleep until a later deadline
            self.0.wake.notify_one();
        }
        if !expired.is_empty() {
            for id in &expired {
                if state.installed.remove(id).is_some() {
                    state.counts.expired += 1;
                }
            }
            state.reindex();
        }
        state.counts.learned += learned;
        let verdict = if !broken.is_empty() {
            for b in &broken {
                *state.counts.violations.entry(b.kind).or_default() += 1;
            }
            Verdict::Surprise(Reason::Violation, broken, on_violation)
        } else if matched.is_empty() {
            state.counts.unexpected += 1;
            Verdict::Surprise(Reason::Unexpected, Vec::new(), Vec::new())
        } else {
            if commands.is_empty() {
                state.counts.absorbed += 1;
            } else {
                state.counts.acted += 1;
            }
            // A shadow check for the youngest expectation that absorbed it, or a sample
            let check = matched
                .iter()
                .find(|id| {
                    state
                        .installed
                        .get(*id)
                        .is_some_and(|i| i.matches <= config.young)
                })
                .or_else(|| sample(signal.id.0, config.shadow).then(|| &matched[0]))
                .cloned();
            if let Some(id) = &check
                && let Some(i) = state.installed.get_mut(id)
            {
                i.shadows += 1;
                state.counts.shadows += 1;
            }
            Verdict::Absorbed(commands, check)
        };
        let surprising = matches!(verdict, Verdict::Surprise(..));
        if surprising {
            state.counts.window_surprises += 1;
        }
        if matches!(
            verdict,
            Verdict::Surprise(..) | Verdict::Absorbed(_, Some(_))
        ) {
            state.counts.surprises += 1;
        }
        let current: Vec<Value> = observation
            .template
            .and_then(|t| state.by_template.get(&t))
            .into_iter()
            .flatten()
            .filter_map(|(id, _)| state.installed.get(id))
            .map(|i| serde_json::to_value(&i.expectation).unwrap_or_default())
            .collect();
        if surprising {
            let selector = Selector {
                template: observation.template_text.clone(),
                fields: BTreeMap::new(),
            };
            state.remember_surprise(signal.id, selector);
        }
        self.roll_window(state);
        drop(guard);
        (verdict, observation, context, current)
    }

    fn roll_window(&self, state: &mut State) {
        let counts = &mut state.counts;
        if counts.window_signals >= WINDOW {
            counts
                .timeline
                .push_back(counts.window_surprises as f64 / counts.window_signals as f64);
            if counts.timeline.len() > WINDOWS {
                counts.timeline.pop_front();
            }
            counts.window_signals = 0;
            counts.window_surprises = 0;
        }
    }

    /// The absences due by `now`: surprise signals to emit into the fabric. The timer calls
    /// this; tests call it with times of their own.
    pub fn due(&self, now: Instant) -> Vec<Signal> {
        let now = self.nanos(now);
        let mut guard = self.lock();
        let state = &mut *guard;
        let mut due = Vec::new();
        let mut later = Vec::new();
        while let Some(Reverse((deadline, _, _))) = state.deadlines.peek()
            && *deadline <= now
        {
            let Some(Reverse((deadline, id, version))) = state.deadlines.pop() else {
                break;
            };
            let Some(installed) = state.installed.get_mut(&id) else {
                continue;
            };
            if installed.version != version {
                continue;
            }
            let Some(broken) = installed.absent(&id, deadline, now, &mut later) else {
                continue;
            };
            installed.absences += 1;
            let commands = match &installed.expectation.on_violation {
                OnViolation::Act(c) => c.clone(),
                OnViolation::Escalate => Vec::new(),
            };
            let current = vec![serde_json::to_value(&installed.expectation).unwrap_or_default()];
            let selector = installed.expectation.selector.clone();
            state.counts.absences += 1;
            state.counts.surprises += 1;
            let surprise = Surprise {
                reason: Reason::Absence,
                at: self.0.name.clone(),
                broken: vec![broken],
                absorbed_by: None,
                observation: Value::Null,
                context: Vec::new(),
                expectations: current,
            };
            let signal = Signal::new(
                Feature::new(ABSENCE, sketch64(id.as_bytes())),
                Payload::json(serde_json::json!({ "surprise": surprise })),
            );
            state.remember_surprise(signal.id, selector);
            for command in commands {
                due.push(signal.derive(Payload::command(command)));
            }
            due.push(signal);
        }
        for (deadline, id, version) in later {
            state.schedule(deadline, &id, version);
        }
        due
    }

    /// When the next deadline is
    fn next_deadline(&self) -> Option<Instant> {
        let state = self.lock();
        state
            .deadlines
            .peek()
            .map(|Reverse((d, _, _))| self.0.epoch + Duration::from_nanos(*d))
    }

    /// A verdict of the brain on a shadow check or a surprise
    fn verdict(&self, args: &Value) -> Result<(), Refused> {
        let id = args
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| Refused("a verdict names the expectation's id".into()))?;
        let mut guard = self.lock();
        let state = &mut *guard;
        let retire_after = self.0.config.retire_after;
        let installed = state
            .installed
            .get_mut(id)
            .ok_or_else(|| Refused(format!("no expectation named '{id}'")))?;
        if args.get("correct").and_then(Value::as_bool) == Some(false) {
            installed.wrong += 1;
            let retire = installed.wrong >= retire_after;
            state.counts.shadow_wrong += 1;
            if retire {
                state.retire(id, "the brain disagreed with what it absorbed");
            }
        } else if args.get("normal").and_then(Value::as_bool) == Some(true) {
            installed.false_alarms += 1;
            let tuned = match installed.last_violation.take() {
                Some((param, value)) => installed.expectation.tune(param, value),
                None => false,
            };
            state.counts.false_alarms += 1;
            if tuned {
                state.counts.learned += 1;
            }
        }
        Ok(())
    }

    /// What this neuron fires for `signal` seen at `now`: the rule's commands for a signal
    /// that fits (and a shadow check when one is due), or the surprise. [`Neuron::stimulate`]
    /// is this at the fabric's clock; a replay calls it with times of its own, as [`Self::due`].
    pub fn react(&self, signal: Signal, now: Instant) -> Vec<Signal> {
        // An absence the timer raised: on to the brain
        if signal.feature.kind == ABSENCE {
            return vec![signal];
        }
        let (verdict, observation, context, current) = self.take(&signal, now);
        let surprise = |reason, broken, absorbed_by| Surprise {
            reason,
            at: self.0.name.clone(),
            broken,
            absorbed_by,
            observation: observation.to_json(),
            context: context.clone(),
            expectations: current.clone(),
        };
        match verdict {
            Verdict::Absorbed(commands, check) => {
                let mut fired: Vec<Signal> = commands
                    .into_iter()
                    .map(|c| signal.derive(Payload::command(c)))
                    .collect();
                if let Some(id) = check {
                    let mut shadow = signal.derive(Payload::json(
                        serde_json::json!({"surprise": surprise(Reason::Shadow, Vec::new(), Some(id))}),
                    ));
                    shadow.reply = None;
                    fired.push(shadow);
                }
                fired
            }
            Verdict::Surprise(reason, broken, commands) => {
                let mut fired: Vec<Signal> = commands
                    .into_iter()
                    .map(|c| signal.derive(Payload::command(c)))
                    .collect();
                fired.push(signal.derive(Payload::json(
                    serde_json::json!({"surprise": surprise(reason, broken, None)}),
                )));
                fired
            }
        }
    }

    /// The counters now
    pub fn snapshot(&self) -> ExpectMetrics {
        let state = self.lock();
        let c = &state.counts;
        let surprising = c.unexpected + c.violations.values().sum::<u64>() + c.absences;
        ExpectMetrics {
            signals: c.signals,
            absorbed: c.absorbed,
            acted: c.acted,
            unexpected: c.unexpected,
            violations: c.violations.clone(),
            absences: c.absences,
            surprises: c.surprises,
            shadows: c.shadows,
            shadow_wrong: c.shadow_wrong,
            false_alarms: c.false_alarms,
            learned: c.learned,
            installed: c.installed,
            replaced: c.replaced,
            retired: c.retired,
            expired: c.expired,
            refused: c.refused,
            active: state.installed.len() as u64,
            pending_approval: state.pending_approval.len() as u64,
            surprise_rate: if c.signals == 0 {
                0.0
            } else {
                surprising as f64 / c.signals as f64
            },
            timeline: c.timeline.iter().copied().collect(),
        }
    }
}

enum Write {
    Add(Expectation),
    Replace(Expectation),
    Retire(String),
}

fn ms(ms: u64) -> u64 {
    ms.saturating_mul(1_000_000)
}

/// Whether the signal numbered `id` is in a sample of `fraction`, deterministically
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

impl Neuron for Expectations {
    fn name(&self) -> &str {
        &self.0.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Receptor
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        Firing::Fire(self.react(signal, cx.now))
    }

    fn describe(&self) -> Value {
        let now = self.nanos(Instant::now());
        let metrics = self.snapshot();
        let state = self.lock();
        let c = &self.0.config;
        serde_json::json!({
            "config": {
                "effectors": c.effectors,
                "shadow": c.shadow,
                "young": c.young,
                "retire_after": c.retire_after,
                "max": c.max,
                "writes_per_minute": c.writes_per_minute,
                "approval": c.approval,
                "ttl_s": c.ttl.as_secs_f64(),
                "context": c.context,
            },
            "metrics": metrics,
            "expectations": state.installed.values().map(|i| i.describe(now)).collect::<Vec<_>>(),
            "pending_approval": state.pending_approval.values().collect::<Vec<_>>(),
            "retired": state.retired_log,
        })
    }

    fn metrics(&self) -> Value {
        serde_json::to_value(self.snapshot()).unwrap_or_default()
    }
}

impl Receptor for Expectations {
    /// The timer: raises absences when their deadlines pass
    fn start(&self, afferent: Afferent) {
        let expectations = self.clone();
        std::thread::Builder::new()
            .name(format!("furnace-expect-{}", self.0.name))
            .spawn(move || {
                loop {
                    let now = Instant::now();
                    for signal in expectations.due(now) {
                        if afferent.emit(signal).is_none() {
                            return;
                        }
                    }
                    if afferent.fabric().is_none() {
                        return;
                    }
                    let wait = expectations
                        .next_deadline()
                        .map_or(Duration::from_secs(1), |d| {
                            d.saturating_duration_since(Instant::now())
                                .min(Duration::from_secs(1))
                        });
                    let state = expectations.lock();
                    let _ = expectations.0.wake.wait_timeout(state, wait);
                }
            })
            .expect("failed to spawn the expectations timer");
    }
}

/// The `expect` effector: the brain's top-down channel. Its command's `args` hold one or more
/// of:
///
/// - `"add"`: an expectation or a list of them (one with an installed id replaces it)
/// - `"replace"`: the same, for an id that must be installed
/// - `"retire"`: an id or a list of ids
/// - `"verdict"`: `{"id": ..., "correct": false}` on a shadow check, or
///   `{"id": ..., "normal": true}` on a surprise that was a false alarm
/// - `"changed"`: a dependency or a list of them that changed
///
/// A refused write fails the command (and is counted); approval is not a command.
pub struct Expect {
    name: String,
    expectations: Expectations,
}

fn list(value: Option<&Value>) -> Vec<&Value> {
    match value {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(one) => vec![one],
    }
}

impl Expect {
    /// [`Act::act`] at `now`: what a replay calls, with times of its own
    pub fn act_at(&self, command: &Command, signal: &Signal, now: Instant) -> Action {
        let args = &command.args;
        let e = &self.expectations;
        let mut errors: Vec<String> = Vec::new();
        if args.get("approve").is_some() {
            errors.push("approval is the operator's, not a command".into());
        }
        for value in list(args.get("add")) {
            let result = Expectation::parse(value)
                .and_then(|x| e.write(Write::Add(x), Some(signal.id), true, now));
            errors.extend(result.err().map(|r| r.0));
        }
        for value in list(args.get("replace")) {
            let result = Expectation::parse(value).and_then(|x| {
                if e.lock().installed.contains_key(&x.id) {
                    e.write(Write::Replace(x), Some(signal.id), true, now)
                } else {
                    Err(Refused(format!(
                        "no expectation named '{}' to replace",
                        x.id
                    )))
                }
            });
            errors.extend(result.err().map(|r| r.0));
        }
        for value in list(args.get("retire")) {
            let result = match value.as_str() {
                Some(id) => e.write(Write::Retire(id.to_string()), None, true, now),
                None => Err(Refused("retire takes ids".into())),
            };
            errors.extend(result.err().map(|r| r.0));
        }
        for value in list(args.get("verdict")) {
            errors.extend(e.verdict(value).err().map(|r| r.0));
        }
        for value in list(args.get("changed")) {
            if let Some(dependency) = value.as_str() {
                e.changed(dependency);
            }
        }
        if errors.is_empty() {
            Action::ok()
        } else {
            Action::Done(Err(ActError::fatal(errors.join("; "))))
        }
    }
}

impl Act for Expect {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        self.act_at(command, signal, Instant::now())
    }

    fn describe(&self) -> Value {
        serde_json::json!({"expectations": self.expectations.0.name})
    }
}

#[cfg(test)]
mod tests;
