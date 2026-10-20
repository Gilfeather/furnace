//! The effector's local loop: retry with backoff, the circuit breaker, the rate limit and the
//! refractory period, and the attempts that go on on the motor side

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use serde::Serialize;
use tracing::{debug, warn};

use super::{ActError, ActFuture, Action, EffectorInner};
use crate::nerve::nanos_since;
use crate::nerve::signal::{Command, Signal};

/// Retry with exponential backoff
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retry {
    /// Tries in all, the first included; 1 means no retry
    pub attempts: u32,
    /// Wait before the second try; doubles each time
    pub backoff: Duration,
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
pub struct RateLimit {
    pub per_second: f64,
    pub burst: u32,
    pub max_wait: Duration,
}

impl RateLimit {
    pub fn new(per_second: f64, burst: u32) -> Self {
        Self {
            per_second: per_second.max(1e-6),
            burst: burst.max(1),
            max_wait: Duration::from_secs(1),
        }
    }

    pub fn max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerState {
    Closed,
    Open,
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
                error.message,
                self.breaker.cooldown
            );
        } else {
            debug!(
                "nerve: effector '{}' failed: {}",
                self.act.name(),
                error.message
            );
        }
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

/// The attempts after the first, or a delayed first one, on the motor side's runtime
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
            None => match inner.act.act(&command, &signal) {
                Action::Done(result) => result,
                Action::Pending(future) => future.await,
            },
        };
        match result {
            Ok(()) => return inner.succeeded(started),
            Err(error) => {
                attempt += 1;
                if !error.retry || attempt >= inner.retry.attempts {
                    return inner.failed(&error);
                }
                inner.counters.retries.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(inner.backoff(attempt)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Act, Channel, Effector};
    use super::*;
    use crate::nerve::testing::{command, fabric_with, wait_until};
    use std::sync::atomic::AtomicUsize;

    /// An act that fails a configured number of times, then succeeds
    struct Flaky {
        fail: AtomicUsize,
        calls: AtomicUsize,
        retryable: bool,
    }

    impl Act for Flaky {
        fn name(&self) -> &str {
            "flaky"
        }
        fn act(&self, _: &Command, _: &Signal) -> Action {
            self.calls.fetch_add(1, Ordering::SeqCst);
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

    #[test]
    fn retries_with_backoff_then_succeeds() {
        let flaky = Flaky {
            fail: AtomicUsize::new(2),
            calls: AtomicUsize::new(0),
            retryable: true,
        };
        let effector = Effector::new(flaky).retry(Retry {
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
        let flaky = Flaky {
            fail: AtomicUsize::new(100),
            calls: AtomicUsize::new(0),
            retryable: false,
        };
        let effector = Effector::new(flaky).breaker(Breaker {
            failures: 2,
            cooldown: Duration::from_millis(50),
        });
        let (fabric, r) = fabric_with(effector.clone());
        for _ in 0..5 {
            fabric.stimulate(r, command("flaky", serde_json::Value::Null));
        }
        let m = effector.metrics();
        // Two failures open the circuit; the other three are dropped
        assert_eq!((m.failed, m.retries, m.dropped_open), (2, 0, 3));
        assert_eq!(m.breaker, BreakerState::Open);
        // After the cooldown one trial goes through (and fails again)
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(effector.metrics().breaker, BreakerState::HalfOpen);
        fabric.stimulate(r, command("flaky", serde_json::Value::Null));
        assert_eq!(effector.metrics().failed, 3);
        assert_eq!(effector.metrics().breaker, BreakerState::Open);
    }

    #[test]
    fn rate_limit_paces_and_refractory_ignores() {
        let (channel, receiver) = Channel::new("out", 64);
        let effector = Effector::new(channel)
            .rate_limit(RateLimit::new(1000.0, 2).max_wait(Duration::from_millis(3)));
        let (fabric, r) = fabric_with(effector.clone());
        for i in 0..10 {
            fabric.stimulate(r, command("out", serde_json::json!(i)));
        }
        // 2 at once, a few more within 3 ms of waiting, the rest dropped
        let m = effector.metrics();
        assert!(m.dropped_rate >= 4, "{m:?}");
        wait_until(|| effector.metrics().inflight == 0);
        let delivered = receiver.try_iter().count();
        assert_eq!(delivered as u64, effector.metrics().acted);
        assert!((2..=6).contains(&delivered), "{delivered}");

        let (channel, receiver) = Channel::new("out", 64);
        let effector = Effector::new(channel).refractory(Duration::from_secs(5));
        let (fabric, r) = fabric_with(effector.clone());
        for i in 0..3 {
            fabric.stimulate(r, command("out", serde_json::json!(i)));
        }
        assert_eq!(receiver.try_iter().count(), 1);
        assert_eq!(effector.metrics().quiet_refractory, 2);
    }
}
