//! The effector's local loop: retry with backoff, the circuit breaker, the rate limit and the
//! refractory period, and the attempts that go on on the motor side

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use serde::Serialize;
use tracing::{debug, warn};

use super::{ActError, ActFuture, Action, EffectorInner};
use crate::nerve::motor::MotionError;
use crate::nerve::nanos_since;
use crate::nerve::signal::{Command, Signal};

/// Retry with exponential backoff
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retry {
    /// Tries in all, the first included; 1 means no retry
    pub attempts: u32,
    /// Wait before the second try; doubles each time
    pub backoff: Duration,
    /// Ceiling on the doubled wait
    pub max_backoff: Duration,
}

impl Default for Retry {
    fn default() -> Self {
        Self {
            attempts: 3,
            backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
        }
    }
}

/// Stop trying for a while after too many failures in a row
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Breaker {
    /// Consecutive final failures that open the circuit
    pub failures: u32,
    /// How long it stays open; then one command is let through as a trial
    pub cooldown: Duration,
}

impl Default for Breaker {
    fn default() -> Self {
        Self {
            failures: 5,
            cooldown: Duration::from_secs(10),
        }
    }
}

/// A token bucket: `per_second` commands sustained, `burst` at once; a command that would
/// have to wait longer than `max_wait` for a token is dropped
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActRateLimit {
    /// Sustained commands a second
    pub per_second: f64,
    /// Commands at once: the bucket's size
    pub burst: u32,
    /// Longest a command waits for a token before it is dropped
    pub max_wait: Duration,
}

impl ActRateLimit {
    /// `per_second` sustained, `burst` at once, waiting at most 1 s for a token
    pub fn new(per_second: f64, burst: u32) -> Self {
        Self {
            per_second: per_second.max(1e-6),
            burst: burst.max(1),
            max_wait: Duration::from_secs(1),
        }
    }

    /// Wait at most this long for a token, then drop the command
    pub fn max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }
}

/// Where a circuit breaker stands
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerState {
    /// Commands go through
    Closed,
    /// Commands are dropped until the cooldown ends
    Open,
    /// Cooldown over: one trial command decides whether it closes or opens again
    HalfOpen,
}

impl EffectorInner {
    /// Whether to act now, after how long, or not at all
    pub(super) fn admit(&self, now: Instant) -> Admission {
        if let Some(refractory) = self.refractory {
            let last = self.last_acted.load(Ordering::Relaxed);
            if last != 0
                && now.saturating_duration_since(self.epoch + Duration::from_nanos(last))
                    < refractory
            {
                self.counters
                    .quiet_refractory
                    .fetch_add(1, Ordering::Relaxed);
                return Admission::Refractory;
            }
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match state.open_until {
            Some(until) if now < until || state.trial => {
                self.counters.dropped_open.fetch_add(1, Ordering::Relaxed);
                return Admission::Open;
            }
            Some(_) => state.trial = true,
            None => {}
        }
        let mut delay = Duration::ZERO;
        if let Some(rate) = self.rate {
            let elapsed = now
                .saturating_duration_since(state.refilled_at)
                .as_secs_f64();
            state.tokens = (state.tokens + elapsed * rate.per_second).min(f64::from(rate.burst));
            state.refilled_at = now;
            if state.tokens < 1.0 {
                let wait = Duration::from_secs_f64((1.0 - state.tokens) / rate.per_second);
                if wait > rate.max_wait {
                    self.counters.dropped_rate.fetch_add(1, Ordering::Relaxed);
                    if state.trial {
                        state.trial = false;
                    }
                    return Admission::RateLimited;
                }
                delay = wait;
            }
            state.tokens -= 1.0;
        }
        self.last_acted
            .store(nanos_since(self.epoch, now).max(1), Ordering::Relaxed);
        self.counters.inflight.fetch_add(1, Ordering::Relaxed);
        Admission::After(delay)
    }

    pub(super) fn succeeded(&self, started: Instant) {
        self.counters.inflight.fetch_sub(1, Ordering::Relaxed);
        self.counters.acted.fetch_add(1, Ordering::Relaxed);
        self.latency.observe(started.elapsed());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.consecutive_failures = 0;
        state.open_until = None;
        state.trial = false;
    }

    pub(super) fn failed(&self, error: &ActError) {
        self.counters.inflight.fetch_sub(1, Ordering::Relaxed);
        self.counters.failed.fetch_add(1, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.consecutive_failures += 1;
        state.trial = false;
        if state.consecutive_failures >= self.breaker.failures {
            state.open_until = Some(Instant::now() + self.breaker.cooldown);
            warn!(
                "nerve: effector '{}' failed {} times in a row ({}); open for {:?}",
                self.act.name(),
                state.consecutive_failures,
                error,
                self.breaker.cooldown
            );
        } else {
            debug!("nerve: effector '{}' failed: {}", self.act.name(), error);
        }
    }

    /// The motor program that sent the command stopped waiting: no more tries
    fn abandoned(&self) {
        self.counters.inflight.fetch_sub(1, Ordering::Relaxed);
        debug!(
            "nerve: effector '{}': nobody waits for the command any more",
            self.act.name()
        );
    }

    pub(super) fn backoff(&self, attempt: u32) -> Duration {
        let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
        self.retry
            .backoff
            .saturating_mul(factor)
            .min(self.retry.max_backoff)
    }
}

pub(super) enum Admission {
    After(Duration),
    Refractory,
    Open,
    RateLimited,
}

/// The attempts after the first, or a delayed first one, on the motor side's runtime; the
/// outcome goes back to the motor program that sent the command, if one did
pub(super) async fn run(
    inner: Arc<EffectorInner>,
    command: Command,
    signal: Signal,
    mut pending: Option<ActFuture>,
    mut attempt: u32,
    started: Instant,
    delay: Duration,
) {
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
    loop {
        let result = match pending.take() {
            Some(future) => future.await,
            None if signal.abandoned() => return inner.abandoned(),
            None => match inner.act.act(&command, &signal) {
                Action::Done(result) => result,
                Action::Pending(future) => future.await,
            },
        };
        match result {
            Ok(value) => {
                inner.succeeded(started);
                return signal.report(Ok(value));
            }
            Err(error) => {
                attempt += 1;
                if !error.retryable() || attempt >= inner.retry.attempts {
                    inner.failed(&error);
                    return signal.report(Err(MotionError::failed(format!(
                        "{}: {error}",
                        inner.act.name()
                    ))));
                }
                inner.counters.retries.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(inner.backoff(attempt)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Act, ChannelOut, Effector};
    use super::*;
    use crate::nerve::fabric::Fabric;
    use crate::nerve::neuron::{Cx, Firing, Neuron};
    use crate::nerve::testing::{command, fabric_with, wait_until};
    use std::sync::atomic::AtomicUsize;

    const HOUR: Duration = Duration::from_secs(3600);

    /// An act that fails a configured number of times, then succeeds
    struct Flaky {
        fail: AtomicUsize,
        retryable: bool,
    }

    impl Flaky {
        fn new(fail: usize, retryable: bool) -> Self {
            Self {
                fail: AtomicUsize::new(fail),
                retryable,
            }
        }
    }

    impl Act for Flaky {
        fn name(&self) -> &str {
            "flaky"
        }
        fn act(&self, _: &Command, _: &Signal) -> Action {
            let left = self.fail.load(Ordering::SeqCst);
            if left > 0 {
                self.fail.store(left - 1, Ordering::SeqCst);
                Action::Done(Err(if self.retryable {
                    ActError::retry("not yet")
                } else {
                    ActError::fatal("never")
                }))
            } else {
                Action::ok()
            }
        }
    }

    /// An effector in a fabric, stimulated as the fabric would stimulate it at a given time,
    /// so the loop's clock is the test's
    struct Clocked {
        effector: Effector,
        fabric: Fabric,
    }

    impl Clocked {
        fn new(effector: Effector) -> Self {
            let (fabric, _) = fabric_with(effector.clone());
            Self { effector, fabric }
        }

        fn at(&self, now: Instant, args: serde_json::Value) -> Firing {
            let name = Neuron::name(&self.effector).to_string();
            let cx = Cx {
                now,
                id: self.fabric.neuron(&name).unwrap(),
                fabric: &self.fabric,
            };
            Neuron::stimulate(&self.effector, command(&name, args), &cx)
        }
    }

    fn admitted(admission: Admission) -> Option<Duration> {
        match admission {
            Admission::After(delay) => Some(delay),
            Admission::Refractory | Admission::Open | Admission::RateLimited => None,
        }
    }

    #[test]
    fn retries_with_backoff_then_succeeds() {
        let effector = Effector::new(Flaky::new(2, true)).retry(Retry {
            attempts: 3,
            backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(20),
        });
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("flaky", serde_json::Value::Null));
        wait_until(|| effector.metrics().acted == 1);
        let m = effector.metrics();
        assert_eq!((m.retries, m.failed, m.inflight), (2, 0, 0));
        assert_eq!(m.breaker, BreakerState::Closed);
    }

    #[test]
    fn fatal_errors_are_not_retried_and_open_the_circuit() {
        let effector = Effector::new(Flaky::new(100, false)).breaker(Breaker {
            failures: 2,
            cooldown: HOUR,
        });
        let clocked = Clocked::new(effector.clone());
        let t0 = Instant::now();
        for _ in 0..5 {
            clocked.at(t0, serde_json::Value::Null);
        }
        let m = effector.metrics();
        // Two failures open the circuit; the other three are dropped
        assert_eq!((m.failed, m.retries, m.dropped_open), (2, 0, 3));
        assert_eq!(m.breaker, BreakerState::Open);
        // Still within the cooldown: dropped
        assert!(
            clocked
                .at(t0 + HOUR / 2, serde_json::Value::Null)
                .is_quiet()
        );
        assert_eq!(effector.metrics().dropped_open, 4);
        // After the cooldown one trial goes through, fails again, and opens the circuit again
        assert!(
            !clocked
                .at(t0 + 2 * HOUR, serde_json::Value::Null)
                .is_quiet()
        );
        let m = effector.metrics();
        assert_eq!((m.failed, m.dropped_open), (3, 4));
        assert_eq!(m.breaker, BreakerState::Open);
    }

    #[test]
    fn a_successful_trial_closes_the_circuit() {
        // A zero cooldown: half open as soon as it opened
        let effector = Effector::new(Flaky::new(2, false)).breaker(Breaker {
            failures: 2,
            cooldown: Duration::ZERO,
        });
        let clocked = Clocked::new(effector.clone());
        let t0 = Instant::now();
        clocked.at(t0, serde_json::Value::Null);
        clocked.at(t0, serde_json::Value::Null);
        assert_eq!(effector.metrics().breaker, BreakerState::HalfOpen);
        // The trial succeeds: closed, and the count of failures in a row starts again
        clocked.at(Instant::now(), serde_json::Value::Null);
        let m = effector.metrics();
        assert_eq!((m.acted, m.failed, m.dropped_open), (1, 2, 0));
        assert_eq!(m.breaker, BreakerState::Closed);
        let state = effector.inner.state.lock().unwrap();
        assert_eq!((state.consecutive_failures, state.trial), (0, false));
    }

    #[test]
    fn half_open_lets_one_trial_through_at_a_time() {
        let effector = Effector::new(Flaky::new(0, false)).breaker(Breaker {
            failures: 1,
            cooldown: HOUR,
        });
        let inner = &effector.inner;
        let t0 = Instant::now();
        assert_eq!(admitted(inner.admit(t0)), Some(Duration::ZERO));
        inner.failed(&ActError::fatal("down"));
        assert!(matches!(inner.admit(t0), Admission::Open));
        // The cooldown is over: the first is the trial, the next is dropped while it is out
        let later = t0 + 2 * HOUR;
        assert_eq!(admitted(inner.admit(later)), Some(Duration::ZERO));
        assert!(matches!(inner.admit(later), Admission::Open));
        inner.succeeded(later);
        assert_eq!(admitted(inner.admit(later)), Some(Duration::ZERO));
        assert_eq!(effector.metrics().dropped_open, 2);
    }

    #[test]
    fn the_rate_limit_delays_within_max_wait_and_drops_beyond_it() {
        let effector = Effector::new(Flaky::new(0, false))
            .rate_limit(ActRateLimit::new(1000.0, 2).max_wait(Duration::from_micros(3500)));
        let inner = &effector.inner;
        let t0 = Instant::now();
        let ms = |n: f64| Duration::from_secs_f64(n / 1e3);
        // The burst at once, then one token a millisecond: 1, 2 and 3 ms of waiting, and a
        // fourth would wait longer than allowed
        let delays: Vec<Option<Duration>> = (0..6).map(|_| admitted(inner.admit(t0))).collect();
        assert_eq!(
            delays,
            [
                Some(ms(0.0)),
                Some(ms(0.0)),
                Some(ms(1.0)),
                Some(ms(2.0)),
                Some(ms(3.0)),
                None
            ]
        );
        assert_eq!(effector.metrics().dropped_rate, 1);
        // The bucket refills with time: 5 ms later, three tokens are owed and five came back
        assert_eq!(
            admitted(inner.admit(t0 + Duration::from_millis(5))),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn rate_limit_paces_and_refractory_ignores() {
        let (channel, receiver) = ChannelOut::new("out", 64);
        let effector = Effector::new(channel)
            .rate_limit(ActRateLimit::new(1000.0, 2).max_wait(Duration::from_micros(3500)));
        let clocked = Clocked::new(effector.clone());
        let t0 = Instant::now();
        for i in 0..10 {
            clocked.at(t0, serde_json::json!(i));
        }
        // 2 at once, 3 more after 1, 2 and 3 ms of waiting on the motor side, 5 dropped
        assert_eq!(effector.metrics().dropped_rate, 5);
        let mut delivered: Vec<i64> = (0..5)
            .map(|_| receiver.recv_timeout(Duration::from_secs(10)).unwrap())
            .map(|d| d.command.args.as_i64().unwrap())
            .collect();
        assert_eq!(delivered[..2], [0, 1]);
        delivered[2..].sort_unstable();
        assert_eq!(delivered[2..], [2, 3, 4]);
        wait_until(|| effector.metrics().inflight == 0);
        assert_eq!(effector.metrics().acted, 5);
        assert!(receiver.try_recv().is_err());

        let (channel, receiver) = ChannelOut::new("out", 64);
        let effector = Effector::new(channel).refractory(Duration::from_secs(5));
        let clocked = Clocked::new(effector.clone());
        let t0 = Instant::now();
        for i in 0..3 {
            clocked.at(t0 + Duration::from_secs(i), serde_json::json!(i));
        }
        assert_eq!(receiver.try_iter().count(), 1);
        assert_eq!(effector.metrics().quiet_refractory, 2);
        // The period over, the next one acts
        clocked.at(t0 + Duration::from_secs(5), serde_json::json!(5));
        assert_eq!(receiver.try_iter().count(), 1);
    }

    /// A target that is down: retries, then the breaker opens, then, once it is back, the rate
    /// limit paces what comes after the trial; the fabric in front keeps taking signals all
    /// along. The motor side runs on this test's runtime with its clock paused, so the backoffs
    /// pass at once and in order; the loop's own clock is the times the test hands it.
    #[tokio::test(start_paused = true)]
    async fn a_failing_target_is_retried_then_cut_off_then_paced() {
        /// Fails, retryably, its first `down` tries, and counts them all
        struct Down {
            down: usize,
            tries: Arc<AtomicUsize>,
        }
        impl Act for Down {
            fn name(&self) -> &str {
                "down"
            }
            fn act(&self, _: &Command, _: &Signal) -> Action {
                if self.tries.fetch_add(1, Ordering::SeqCst) < self.down {
                    Action::Done(Err(ActError::Unreachable("connection refused".into())))
                } else {
                    Action::ok()
                }
            }
        }
        let tries = Arc::new(AtomicUsize::new(0));
        let effector = Effector::new(Down {
            down: 6,
            tries: tries.clone(),
        })
        .retry(Retry {
            attempts: 3,
            backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        })
        .breaker(Breaker {
            failures: 2,
            cooldown: HOUR,
        })
        .rate_limit(ActRateLimit::new(1.0, 1).max_wait(Duration::ZERO));
        let mut b = Fabric::builder().efferent(tokio::runtime::Handle::current());
        let upstream = b.neuron(crate::nerve::testing::pass("in"));
        let target = b.neuron(effector.clone());
        let fabric = b.build().unwrap();
        let at = |now: Instant| {
            let cx = Cx {
                now,
                id: target,
                fabric: &fabric,
            };
            Neuron::stimulate(&effector, command("down", serde_json::Value::Null), &cx)
        };
        // Lets the retries on the motor side run to their end: an hour of paused time
        let settle = || tokio::time::sleep(HOUR);
        let attempts = || tries.load(Ordering::SeqCst);
        let t0 = Instant::now();

        // 1. Retried: three tries, then one failure in a row
        assert!(!at(t0).is_quiet());
        settle().await;
        let m = effector.metrics();
        assert_eq!((attempts(), m.retries, m.failed), (3, 2, 1));
        assert_eq!(m.breaker, BreakerState::Closed);

        // 2. The second command fails the same way, and the circuit opens
        assert!(!at(t0 + Duration::from_secs(1)).is_quiet());
        settle().await;
        let m = effector.metrics();
        assert_eq!((attempts(), m.retries, m.failed), (6, 4, 2));
        assert_eq!(m.breaker, BreakerState::Open);

        // 3. Open: commands are dropped without a try, and the fabric in front of the
        // effector keeps taking signals
        for i in 0..100 {
            assert!(at(t0 + Duration::from_secs(2) + Duration::from_millis(i)).is_quiet());
            fabric.stimulate(upstream, command("down", serde_json::Value::Null));
        }
        settle().await;
        let m = effector.metrics();
        // Half of them dropped here, half on their way through the fabric
        assert_eq!((attempts(), m.dropped_open), (6, 200));
        assert_eq!(fabric.metrics().received, 100);

        // 4. The cooldown over and the target back: the trial succeeds and closes the circuit,
        // and the rate limit drops the burst that follows it within the same second
        let back = Instant::now() + 2 * HOUR;
        assert!(!at(back).is_quiet());
        assert_eq!(effector.metrics().breaker, BreakerState::Closed);
        for _ in 0..3 {
            assert!(at(back).is_quiet());
        }
        assert!(!at(back + Duration::from_secs(1)).is_quiet());
        settle().await;
        let m = effector.metrics();
        assert_eq!(
            (m.acted, m.failed, m.retries, m.dropped_open, m.dropped_rate),
            (2, 2, 4, 200, 3)
        );
        assert_eq!(m.inflight, 0);
    }
}
