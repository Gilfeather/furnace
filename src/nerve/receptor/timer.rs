//! Periodic signals

use std::time::{Duration, Instant};

use super::{Afferent, RateMeter, Receptor, Source, UrgencyRule, kinds};
use crate::nerve::neuron::{Cx, Firing, Neuron, NeuronKind};
use crate::nerve::signal::{Feature, Payload, Signal};

/// Periodic signals, one per `period`, from a thread of their own. The payload is
/// `{"tick": n}`; the feature is the tick number by default, so ticks never habituate, or
/// one fixed feature with [`Timer::same_feature`] so they do.
pub struct Timer {
    source: Source,
    period: Duration,
    same_feature: bool,
}

impl Timer {
    /// One tick every `period` (at least 10 µs), the first one `period` after the start
    pub fn new(name: &str, period: Duration) -> Self {
        Self {
            source: Source::new(name),
            period: period.max(Duration::from_micros(10)),
            same_feature: false,
        }
    }

    /// How each tick's urgency is stamped
    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Every tick has the same feature (and habituates, if the timer has habituation)
    pub fn same_feature(mut self) -> Self {
        self.same_feature = true;
        self
    }
}

impl Neuron for Timer {
    fn name(&self) -> &str {
        &self.source.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Receptor
    }

    fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
        Firing::one(signal)
    }

    fn describe(&self) -> serde_json::Value {
        self.source.describe(serde_json::json!({
            "source": "timer",
            "period_ms": self.period.as_secs_f64() * 1e3,
        }))
    }
}

impl Receptor for Timer {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent.clone());
        let (period, same, rule) = (self.period, self.same_feature, self.source.rule);
        let meter = RateMeter::new();
        let name = self.source.name.clone();
        std::thread::Builder::new()
            .name(format!("furnace-receptor-{name}"))
            .spawn(move || {
                let mut next = Instant::now() + period;
                for tick in 0u64.. {
                    let now = Instant::now();
                    if next > now && !afferent.sleep(next - now) {
                        return;
                    }
                    next += period;
                    let feature = Feature::new(kinds::TIMER, if same { 0 } else { tick });
                    let signal = rule.signal(
                        &meter,
                        feature,
                        Payload::json(serde_json::json!({"tick": tick})),
                    );
                    if afferent.emit(signal).is_none() {
                        return;
                    }
                }
            })
            .expect("failed to spawn a receptor thread");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::testing::{fabric_with_receptor, wait_until};

    #[test]
    fn timer_receptor_ticks_and_stamps_the_rate() {
        let timer =
            Timer::new("tick", Duration::from_millis(5)).urgency(UrgencyRule::before_next(1.0));
        let (_fabric, _r, seen) = fabric_with_receptor(timer);
        wait_until(|| seen.lock().unwrap().len() >= 5);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].payload.to_json().unwrap()["tick"], 0);
        assert_ne!(seen[0].feature, seen[1].feature);
        // From the second tick on, the rate is known and the deadline follows it
        assert!(seen[4].urgency.rate > 0.0, "{}", seen[4].urgency.rate);
        assert!(seen[4].urgency.deadline.is_some());
        assert!(seen[0].urgency.deadline.is_none());
    }

    #[test]
    fn ticks_of_the_same_feature_habituate() {
        let mut b = crate::nerve::Fabric::builder();
        let timer = b.receptor(Timer::new("tick", Duration::from_millis(1)).same_feature());
        b.gate(
            timer,
            crate::nerve::Mechanisms::NONE
                .threshold(0.6)
                .habituation(crate::nerve::Habituation::new(Duration::from_secs(3600))),
        );
        let (seen, ticks) = std::sync::mpsc::channel();
        let sink = b.neuron(
            crate::nerve::FnNeuron::reflex("sink", move |s, _| {
                let _ = seen.send(s);
                Firing::Quiet
            })
            .with_kind(NeuronKind::Effector),
        );
        b.reflex(timer, sink);
        let fabric = b.build().unwrap();
        fabric.start();
        // Only the first tick passes; the repeats die at the timer's gate
        let first = ticks.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(first.payload.to_json().unwrap()["tick"], 0);
        let counters = &fabric.0.cells[timer.index()].counters;
        wait_until(|| {
            counters
                .quiet_habituation
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 3
        });
        assert!(ticks.try_recv().is_err());
    }
}
