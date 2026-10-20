//! Neurons: anything that takes a signal and may fire, and the three small state machines
//! every neuron owns (threshold, habituation, refractory period).

use std::marker::PhantomData;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::fabric::Fabric;
use super::signal::{Feature, NeuronId, Signal};

/// What kind of neuron, for metrics (`brain_fraction`, reflex arcs) and for `GET /nerve`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NeuronKind {
    /// World → signal
    Receptor,
    /// Feature → command, in process, bounded in microseconds to low milliseconds
    Reflex,
    /// The large slow model: its output is commands
    Brain,
    /// Command → world
    Effector,
}

impl NeuronKind {
    /// The name in JSON and in the Prometheus `kind` label
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Receptor => "receptor",
            Self::Reflex => "reflex",
            Self::Brain => "brain",
            Self::Effector => "effector",
        }
    }
}

/// What a neuron does with a signal
pub enum Firing {
    /// Not interested (the fabric's gate already handled threshold, habituation and refractory)
    Quiet,
    /// Emitted signals, each propagated along the neuron's pathways
    Fire(Vec<Signal>),
    /// Blocking work (a model's forward, a generation): the fabric runs it where the pathway
    /// says (inline on a reflex arc, on the worker of a deliberative pathway) and propagates
    /// the result
    Compute(Box<dyn FnOnce() -> Vec<Signal> + Send>),
}

impl Firing {
    /// Fire one signal
    pub fn one(signal: Signal) -> Self {
        Self::Fire(vec![signal])
    }

    /// Blocking work to run where the pathway says
    pub fn compute(work: impl FnOnce() -> Vec<Signal> + Send + 'static) -> Self {
        Self::Compute(Box::new(work))
    }

    /// Whether the neuron was not interested
    pub fn is_quiet(&self) -> bool {
        matches!(self, Self::Quiet)
    }
}

impl From<Signal> for Firing {
    fn from(signal: Signal) -> Self {
        Self::one(signal)
    }
}

impl From<Option<Signal>> for Firing {
    fn from(signal: Option<Signal>) -> Self {
        signal.map_or(Self::Quiet, Self::one)
    }
}

impl From<Vec<Signal>> for Firing {
    fn from(signals: Vec<Signal>) -> Self {
        Self::Fire(signals)
    }
}

impl std::fmt::Debug for Firing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quiet => f.write_str("Quiet"),
            Self::Fire(signals) => f.debug_tuple("Fire").field(&signals.len()).finish(),
            Self::Compute(_) => f.write_str("Compute"),
        }
    }
}

/// What a neuron sees of the fabric while it is stimulated
pub struct Cx<'a> {
    /// The fabric's clock reading when the signal reached this neuron
    pub now: Instant,
    /// This neuron
    pub id: NeuronId,
    pub(crate) fabric: &'a Fabric,
}

impl Cx<'_> {
    /// The fabric this neuron is part of
    pub fn fabric(&self) -> &Fabric {
        self.fabric
    }

    /// The motor side's runtime: where effectors run their I/O and brains their engines
    pub fn efferent(&self) -> &tokio::runtime::Handle {
        self.fabric.efferent()
    }

    /// The names effectors are registered under, for a brain's prompt
    pub fn effectors(&self) -> Vec<String> {
        self.fabric.effector_names()
    }
}

/// Anything that takes a signal and may fire. Must be cheap on the no-fire path: the gate
/// (threshold, habituation, refractory) runs before `stimulate`, so a neuron only sees the
/// signals that got through.
pub trait Neuron: Send + Sync + 'static {
    /// Unique in the fabric: effectors are addressed by it in commands
    fn name(&self) -> &str;

    /// What kind of neuron this is; a reflex unless overridden
    fn kind(&self) -> NeuronKind {
        NeuronKind::Reflex
    }

    /// Decide and act
    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing;

    /// Several signals that a deliberative pathway summed into one firing. The default
    /// stimulates them one by one; a model that batches overrides it.
    fn stimulate_many(&self, signals: Vec<Signal>, cx: &Cx<'_>) -> Vec<Firing> {
        signals.into_iter().map(|s| self.stimulate(s, cx)).collect()
    }

    /// Warm up for a signal that is probably coming: open the connection, touch the
    /// weights. Called on the thread that selected the pathway, so it must not block; start
    /// the work on [`Cx::efferent`] if it takes time.
    fn prepare(&self, _cx: &Cx<'_>) {}

    /// Anything else `GET /nerve` should show about this neuron
    fn describe(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// Counters of the neuron's own (an effector's local loop), for `GET /metrics`
    fn metrics(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// The same in the Prometheus text format; `labels` names this neuron
    fn render_prometheus(&self, _out: &mut String, _labels: &str) {}

    /// For neurons that publish a stream of events (the SSE effector): subscribe to it
    fn events(&self) -> Option<tokio::sync::broadcast::Receiver<bytes::Bytes>> {
        None
    }
}

/// A neuron made from a closure: `Fn(Signal, &Cx) -> R`, where `R` is a [`Firing`] or
/// anything that becomes one: a [`Signal`] fires it, an `Option<Signal>` fires it or stays
/// quiet, a `Vec<Signal>` fires them all
pub struct FnNeuron<F, R = Firing> {
    name: String,
    kind: NeuronKind,
    f: F,
    returns: PhantomData<fn() -> R>,
}

impl<F, R> FnNeuron<F, R>
where
    F: Fn(Signal, &Cx<'_>) -> R + Send + Sync + 'static,
    R: Into<Firing> + 'static,
{
    /// A reflex: feature in, signals (or commands) out, in process
    pub fn reflex(name: &str, f: F) -> Self {
        Self {
            name: name.to_string(),
            kind: NeuronKind::Reflex,
            f,
            returns: PhantomData,
        }
    }

    /// The same neuron as another kind (a brain, an effector)
    pub fn with_kind(mut self, kind: NeuronKind) -> Self {
        self.kind = kind;
        self
    }
}

impl<F, R> Neuron for FnNeuron<F, R>
where
    F: Fn(Signal, &Cx<'_>) -> R + Send + Sync + 'static,
    R: Into<Firing> + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NeuronKind {
        self.kind
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        (self.f)(signal, cx).into()
    }
}

/// Habituation: a repeated feature attenuates before it propagates, and the attenuation
/// decays with time
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Habituation {
    /// Recovery time constant: an entry's attenuation decays as `exp(-dt / tau)`
    pub tau: Duration,
    /// How much of the remaining way to full attenuation one repeat covers (0..1). With 0.5,
    /// the second occurrence of a feature arrives at half intensity, the third at a quarter.
    pub gain: f32,
    /// Features remembered at once, rounded up to a multiple of 4 (a set's ways); the least
    /// recently seen in a feature's set is replaced
    pub capacity: usize,
}

impl Default for Habituation {
    fn default() -> Self {
        Self {
            tau: Duration::from_secs(30),
            gain: 0.5,
            capacity: 1024,
        }
    }
}

impl Habituation {
    /// Habituation recovering with time constant `tau`, the default gain and capacity
    pub fn new(tau: Duration) -> Self {
        Self {
            tau,
            ..Self::default()
        }
    }

    /// Attenuation per repeat, clamped to 0..=1
    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain.clamp(0.0, 1.0);
        self
    }

    /// Remember this many features at once, rounded up to a multiple of 4; at least 4
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(WAYS);
        self
    }
}

/// The three mechanisms a neuron can have, each optional and each a few words of state
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mechanisms {
    /// Fires only if `intensity >= threshold`; 0 lets everything through
    pub threshold: f32,
    /// Attenuation of repeated features; `None` never habituates
    pub habituation: Option<Habituation>,
    /// After firing, ignore signals for this long
    pub refractory: Option<Duration>,
}

impl Default for Mechanisms {
    fn default() -> Self {
        Self::NONE
    }
}

impl Mechanisms {
    /// Everything passes: a plain neuron
    pub const NONE: Self = Self {
        threshold: 0.0,
        habituation: None,
        refractory: None,
    };

    /// Fire only at or above this intensity
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Attenuate repeated features
    pub fn habituation(mut self, habituation: Habituation) -> Self {
        self.habituation = Some(habituation);
        self
    }

    /// Ignore signals for `period` after firing
    pub fn refractory(mut self, period: Duration) -> Self {
        self.refractory = Some(period);
        self
    }
}

/// Why a signal died at a neuron's gate
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quiet {
    /// Intensity below the threshold as it arrived
    Threshold,
    /// Attenuated below the threshold by habituation
    Habituation,
    /// The neuron fired too recently
    Refractory,
    /// The neuron looked and was not interested
    NotInterested,
}

impl Quiet {
    /// The name in JSON and in the Prometheus `reason` label
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Threshold => "threshold",
            Self::Habituation => "habituation",
            Self::Refractory => "refractory",
            Self::NotInterested => "not_interested",
        }
    }
}

/// Entries per set of the habituation table: a feature is remembered in one of these slots
const WAYS: usize = 4;

#[derive(Clone, Copy)]
struct Slot {
    feature: Feature,
    /// Attenuation 0..1 as of `seen`
    attenuation: f32,
    /// Nanoseconds since the fabric's epoch; `u64::MAX` marks an empty slot
    seen: u64,
}

const EMPTY: Slot = Slot {
    feature: Feature::NONE,
    attenuation: 0.0,
    seen: u64::MAX,
};

/// A bounded, set-associative table of attenuations keyed by feature
struct Table {
    slots: Vec<Slot>,
    sets: usize,
    tau_nanos: f32,
    gain: f32,
}

impl Table {
    fn new(config: Habituation) -> Self {
        let sets = config.capacity.div_ceil(WAYS).max(1);
        Self {
            slots: vec![EMPTY; sets * WAYS],
            sets,
            tau_nanos: config.tau.as_nanos().max(1) as f32,
            gain: config.gain,
        }
    }

    /// The attenuation to apply to a signal with `feature` seen `now`, 0..1, and remember
    /// the repeat
    fn attenuation(&mut self, feature: Feature, now: u64) -> f32 {
        let set = (feature.sketch ^ (u64::from(feature.kind) << 56)) as usize % self.sets;
        let slots = &mut self.slots[set * WAYS..(set + 1) * WAYS];
        let mut victim = 0;
        let mut victim_key = u64::MAX;
        for (i, slot) in slots.iter_mut().enumerate() {
            if slot.seen != u64::MAX && slot.feature == feature {
                let dt = now.saturating_sub(slot.seen) as f32;
                let decayed = slot.attenuation * (-dt / self.tau_nanos).exp();
                slot.attenuation = decayed + (1.0 - decayed) * self.gain;
                slot.seen = now;
                return decayed;
            }
            // An empty slot first, else the least recently seen
            let key = if slot.seen == u64::MAX {
                0
            } else {
                slot.seen.saturating_add(1)
            };
            if key < victim_key {
                victim = i;
                victim_key = key;
            }
        }
        slots[victim] = Slot {
            feature,
            attenuation: self.gain,
            seen: now,
        };
        0.0
    }
}

/// The per-neuron state of [`Mechanisms`]: what the fabric runs before `stimulate`
pub(crate) struct Gate {
    threshold: f32,
    table: Option<Mutex<Table>>,
    refractory_nanos: u64,
    /// Nanoseconds since the epoch of the last firing, or of the signal now being admitted;
    /// 0 before any
    last_fired: AtomicU64,
}

impl Gate {
    pub(crate) fn new(mechanisms: Mechanisms) -> Self {
        Self {
            threshold: mechanisms.threshold,
            table: mechanisms.habituation.map(|h| Mutex::new(Table::new(h))),
            refractory_nanos: mechanisms
                .refractory
                .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
            last_fired: AtomicU64::new(0),
        }
    }

    /// Attenuate `signal`, and say why it dies here if it does. `now` is nanoseconds since
    /// the fabric's epoch.
    pub(crate) fn admit(&self, signal: &mut Signal, now: u64) -> Result<(), Quiet> {
        if signal.intensity < self.threshold {
            return Err(Quiet::Threshold);
        }
        if let Some(table) = &self.table
            && !signal.feature.is_none()
        {
            let attenuation = table
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .attenuation(signal.feature, now);
            if attenuation > 0.0 {
                signal.intensity *= 1.0 - attenuation;
                if signal.intensity < self.threshold || signal.intensity <= 0.0 {
                    return Err(Quiet::Habituation);
                }
            }
        }
        if self.refractory_nanos > 0 {
            // The admitted signal takes the period at once, so two signals arriving together
            // cannot both pass; signals of the same instant are one firing (a summed batch)
            let claim = now.max(1);
            let mut last = self.last_fired.load(Ordering::Acquire);
            while last != claim {
                if last != 0 && now.saturating_sub(last) < self.refractory_nanos {
                    return Err(Quiet::Refractory);
                }
                match self.last_fired.compare_exchange_weak(
                    last,
                    claim,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(current) => last = current,
                }
            }
        }
        Ok(())
    }

    /// The neuron admitted at `now` stayed quiet: it did not fire, so it is not refractory.
    /// Whatever fired before was a full period ago (or the signal would not have been
    /// admitted), so forgetting it changes nothing.
    pub(crate) fn stayed_quiet(&self, now: u64) {
        if self.refractory_nanos > 0 {
            let _ = self.last_fired.compare_exchange(
                now.max(1),
                0,
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
        }
    }

    pub(crate) fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "threshold": self.threshold,
            "habituation": self.table.as_ref().map(|t| {
                let t = t.lock().unwrap_or_else(|e| e.into_inner());
                serde_json::json!({
                    "capacity": t.slots.len(),
                    "tau_ms": t.tau_nanos / 1e6,
                    "gain": t.gain,
                    "remembered": t.slots.iter().filter(|s| s.seen != u64::MAX).count(),
                })
            }),
            "refractory_ms": (self.refractory_nanos > 0).then(|| self.refractory_nanos as f64 / 1e6),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::signal::Payload;

    const SECOND: u64 = 1_000_000_000;

    fn signal(bytes: &[u8]) -> Signal {
        Signal::new(Feature::sketch(bytes), Payload::None)
    }

    #[test]
    fn habituation_attenuates_repeats_and_recovers_with_time() {
        let gate = Gate::new(
            Mechanisms::NONE
                .threshold(0.3)
                .habituation(Habituation::new(Duration::from_secs(10)).gain(0.5)),
        );
        let mut first = signal(b"same");
        assert_eq!(gate.admit(&mut first, SECOND), Ok(()));
        assert_eq!(first.intensity, 1.0);
        // The second occurrence, right away, arrives at half intensity
        let mut second = signal(b"same");
        assert_eq!(gate.admit(&mut second, SECOND + 1), Ok(()));
        assert!(
            (second.intensity - 0.5).abs() < 1e-3,
            "{}",
            second.intensity
        );
        // The third at a quarter: below the threshold, it dies of habituation
        let mut third = signal(b"same");
        assert_eq!(gate.admit(&mut third, SECOND + 2), Err(Quiet::Habituation));
        // A different feature is untouched
        let mut other = signal(b"other");
        assert_eq!(gate.admit(&mut other, SECOND + 3), Ok(()));
        assert_eq!(other.intensity, 1.0);
        // After several time constants the attenuation has decayed away
        let mut later = signal(b"same");
        assert_eq!(gate.admit(&mut later, SECOND + 60 * SECOND), Ok(()));
        assert!(later.intensity > 0.99, "{}", later.intensity);
    }

    #[test]
    fn habituation_recovers_along_exp_minus_dt_over_tau() {
        let tau = 10 * SECOND;
        let gate = Gate::new(
            Mechanisms::NONE.habituation(Habituation::new(Duration::from_nanos(tau)).gain(0.5)),
        );
        // Each feature is seen once at 1 s, then again `dt` later: the second arrives at
        // 1 - 0.5 * exp(-dt / tau)
        for (i, dt) in [0, tau / 2, tau, 2 * tau, 3 * tau, 6 * tau]
            .into_iter()
            .enumerate()
        {
            let feature = [i as u8];
            let mut first = signal(&feature);
            assert_eq!(gate.admit(&mut first, SECOND), Ok(()));
            let mut again = signal(&feature);
            assert_eq!(gate.admit(&mut again, SECOND + dt), Ok(()));
            let expected = 1.0 - 0.5 * (-(dt as f32) / tau as f32).exp();
            assert!(
                (again.intensity - expected).abs() < 1e-4,
                "dt = {dt}: {} != {expected}",
                again.intensity
            );
        }
    }

    #[test]
    fn habituation_table_is_bounded() {
        let gate = Gate::new(
            Mechanisms::NONE
                .threshold(0.6)
                .habituation(Habituation::new(Duration::from_secs(10)).capacity(8)),
        );
        // Far more distinct features than the table holds
        for i in 0..10_000u32 {
            let mut s = signal(&i.to_le_bytes());
            let _ = gate.admit(&mut s, SECOND + u64::from(i));
        }
        let described = gate.describe();
        assert_eq!(described["habituation"]["capacity"], 8);
        assert_eq!(described["habituation"]["remembered"], 8);
        // A feature seen just now is still remembered; an evicted one is forgotten
        let mut recent = signal(&9_999u32.to_le_bytes());
        assert_eq!(gate.admit(&mut recent, 2 * SECOND), Err(Quiet::Habituation));
    }

    #[test]
    fn signals_without_a_feature_never_habituate() {
        let gate = Gate::new(
            Mechanisms::NONE
                .threshold(0.5)
                .habituation(Habituation::default()),
        );
        for i in 0..5 {
            let mut s = Signal::new(Feature::NONE, Payload::None);
            assert_eq!(gate.admit(&mut s, SECOND + i), Ok(()));
            assert_eq!(s.intensity, 1.0);
        }
    }

    #[test]
    fn threshold_drops_weak_signals() {
        let gate = Gate::new(Mechanisms::NONE.threshold(0.5));
        let mut weak = signal(b"a").with_intensity(0.4);
        assert_eq!(gate.admit(&mut weak, SECOND), Err(Quiet::Threshold));
        let mut strong = signal(b"a").with_intensity(0.5);
        assert_eq!(gate.admit(&mut strong, SECOND), Ok(()));
    }

    #[test]
    fn refractory_period_ignores_signals_after_a_firing() {
        let gate = Gate::new(Mechanisms::NONE.refractory(Duration::from_millis(10)));
        let mut s = signal(b"a");
        assert_eq!(gate.admit(&mut s, SECOND), Ok(()));
        // Admitting took the period: a signal right behind it is refused even before the
        // neuron has fired, so two signals racing in cannot both pass
        let mut during = signal(b"b");
        assert_eq!(
            gate.admit(&mut during, SECOND + 5_000_000),
            Err(Quiet::Refractory)
        );
        // Signals of the same instant are one firing, as a summed batch is
        let mut same = signal(b"s");
        assert_eq!(gate.admit(&mut same, SECOND), Ok(()));
        let mut after = signal(b"c");
        assert_eq!(gate.admit(&mut after, SECOND + 10_000_000), Ok(()));
        // A neuron that stayed quiet did not fire, and is not refractory
        gate.stayed_quiet(SECOND + 10_000_000);
        let mut next = signal(b"d");
        assert_eq!(gate.admit(&mut next, SECOND + 11_000_000), Ok(()));
    }

    #[test]
    fn racing_signals_pass_a_refractory_gate_once() {
        let gate = Gate::new(Mechanisms::NONE.refractory(Duration::from_secs(60)));
        let passed = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for i in 0..8u64 {
                let (gate, passed) = (&gate, &passed);
                scope.spawn(move || {
                    let mut s = signal(b"x");
                    if gate.admit(&mut s, SECOND + i).is_ok() {
                        passed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(passed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn no_mechanisms_means_no_cost_and_no_quiet() {
        let gate = Gate::new(Mechanisms::NONE);
        for _ in 0..3 {
            let mut s = signal(b"same").with_intensity(0.0);
            assert_eq!(gate.admit(&mut s, SECOND), Ok(()));
        }
    }
}
