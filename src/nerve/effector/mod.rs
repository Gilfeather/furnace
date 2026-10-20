//! Effectors: the efferent side. A command comes in, something happens in the world, and the
//! effector's own local loop deals with failure and pacing: retry with backoff, a circuit
//! breaker, a rate limit and a refractory period. The brain is never asked whether to
//! proceed, and never sits in a retry loop.
//!
//! An [`Effector`] wraps an [`Act`], the thing that happens:
//!
//! - [`Respond`]: the HTTP response to the request that started the chain
//! - [`Webhook`]: an outbound HTTP call (a webhook, another service's API), JSON or bytes
//! - [`Process`]: run a local command with arguments, for devices and scripts
//! - [`FileAppend`]: append to a file (a log, a journal)
//! - [`Channel`]: an in-process channel, for your own code
//! - [`Sse`]: push commands to Server-Sent-Events subscribers
//!
//! Effectors are addressed by name: a command `{ "effector": "alarm", "args": {...} }`
//! reaches the effector registered as `alarm`. A plain (non-command) signal that reaches an
//! effector acts with the signal's payload as its arguments.

mod channel;
mod file_append;
mod local_loop;
mod process;
mod respond;
mod sse;
mod webhook;

use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde::Serialize;

use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Neuron, NeuronKind};
use super::signal::{Command, Signal};

pub use channel::{Channel, Delivery};
pub use file_append::FileAppend;
use local_loop::{Admission, run};
pub use local_loop::{Breaker, BreakerState, RateLimit, Retry};
pub use process::Process;
pub use respond::Respond;
pub use sse::Sse;
pub use webhook::Webhook;

/// Why an act failed, and whether trying again could help
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActError {
    pub message: String,
    pub retry: bool,
}

impl ActError {
    /// A failure that may pass: a timeout, a 503, a full queue
    pub fn retry(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retry: true,
        }
    }

    /// A failure that will not pass: a 400, a bad command
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retry: false,
        }
    }
}

pub type ActFuture = Pin<Box<dyn Future<Output = Result<(), ActError>> + Send + 'static>>;

/// What an act returns: done at once (an answer, a channel send, a file write) or still
/// going on the motor side's runtime (a socket, a process)
pub enum Action {
    Done(Result<(), ActError>),
    Pending(ActFuture),
}

impl Action {
    pub fn ok() -> Self {
        Self::Done(Ok(()))
    }

    pub fn pending(future: impl Future<Output = Result<(), ActError>> + Send + 'static) -> Self {
        Self::Pending(Box::pin(future))
    }
}

/// The thing an effector does. Implement this for your own device or service; the
/// [`Effector`] around it supplies the loop.
pub trait Act: Send + Sync + 'static {
    fn name(&self) -> &str;

    /// Do it. Called again, with the same command, on a retry.
    fn act(&self, command: &Command, signal: &Signal) -> Action;

    /// Warm up (open the connection) on `runtime`; must not block
    fn prepare(&self, _runtime: &tokio::runtime::Handle) {}

    fn describe(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// For effectors that publish a stream (the SSE effector): subscribe to it
    fn events(&self) -> Option<tokio::sync::broadcast::Receiver<Bytes>> {
        None
    }
}

struct LoopState {
    consecutive_failures: u32,
    open_until: Option<Instant>,
    /// A trial command is out while half open
    trial: bool,
    tokens: f64,
    refilled_at: Instant,
}

#[derive(Default)]
struct LoopCounters {
    acted: AtomicU64,
    failed: AtomicU64,
    retries: AtomicU64,
    dropped_open: AtomicU64,
    dropped_rate: AtomicU64,
    quiet_refractory: AtomicU64,
    inflight: AtomicU64,
}

/// The effector's own counters, as `GET /metrics` shows them
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct EffectorMetrics {
    /// Commands that succeeded
    pub acted: u64,
    /// Commands that failed for good (after their retries)
    pub failed: u64,
    pub retries: u64,
    /// Dropped while the circuit was open
    pub dropped_open: u64,
    /// Dropped by the rate limit
    pub dropped_rate: u64,
    pub quiet_refractory: u64,
    /// Commands in progress on the motor side
    pub inflight: u64,
    pub breaker: BreakerState,
    /// Of successful commands, from the effector's call to the world's acknowledgement
    pub latency: super::latency::LatencySnapshot,
}

struct EffectorInner {
    act: Box<dyn Act>,
    /// The command a plain signal with no JSON payload (tensors, nothing) acts as
    plain: Command,
    retry: Retry,
    breaker: Breaker,
    rate: Option<RateLimit>,
    refractory: Option<Duration>,
    epoch: Instant,
    /// Nanoseconds since `epoch` of the last command admitted; 0 before any
    last_acted: AtomicU64,
    state: Mutex<LoopState>,
    counters: LoopCounters,
    latency: LatencyEstimate,
}

/// An [`Act`] with its local loop: retry with backoff, circuit breaker, rate limit,
/// refractory period, and metrics. Add it to the fabric like any neuron; its
/// [`Mechanisms`](super::Mechanisms) (habituation, threshold) apply in front of it.
#[derive(Clone)]
pub struct Effector {
    inner: Arc<EffectorInner>,
}

impl Effector {
    /// With the defaults: 3 attempts from 100 ms backoff, the circuit opens after 5 failures
    /// in a row for 10 s, no rate limit, no refractory period
    pub fn new(act: impl Act) -> Self {
        let now = Instant::now();
        let plain = Command::new(act.name(), serde_json::Value::Null);
        Self {
            inner: Arc::new(EffectorInner {
                act: Box::new(act),
                plain,
                retry: Retry::default(),
                breaker: Breaker::default(),
                rate: None,
                refractory: None,
                epoch: now,
                last_acted: AtomicU64::new(0),
                state: Mutex::new(LoopState {
                    consecutive_failures: 0,
                    open_until: None,
                    trial: false,
                    tokens: 0.0,
                    refilled_at: now,
                }),
                counters: LoopCounters::default(),
                latency: LatencyEstimate::new(),
            }),
        }
    }

    fn configure(mut self, f: impl FnOnce(&mut EffectorInner)) -> Self {
        f(Arc::get_mut(&mut self.inner).expect("configured before use"));
        self
    }

    /// How often, and how far apart, to try a command that failed in a way worth retrying
    ///
    /// # Panics
    ///
    /// Once the effector is shared: a clone, or the fabric, holds it. Configure it first.
    pub fn retry(self, retry: Retry) -> Self {
        self.configure(|i| i.retry = retry)
    }

    /// One try, no retry
    ///
    /// # Panics
    ///
    /// Once the effector is shared: a clone, or the fabric, holds it. Configure it first.
    pub fn no_retry(self) -> Self {
        self.retry(Retry {
            attempts: 1,
            ..Retry::default()
        })
    }

    /// After this many failures in a row, refuse commands for a while
    ///
    /// # Panics
    ///
    /// Once the effector is shared: a clone, or the fabric, holds it. Configure it first.
    pub fn breaker(self, breaker: Breaker) -> Self {
        self.configure(|i| i.breaker = breaker)
    }

    /// At most this many commands a second, in bursts of at most `burst`
    ///
    /// # Panics
    ///
    /// Once the effector is shared: a clone, or the fabric, holds it. Configure it first.
    pub fn rate_limit(self, rate: RateLimit) -> Self {
        self.configure(|i| {
            i.state.get_mut().unwrap_or_else(|e| e.into_inner()).tokens = f64::from(rate.burst);
            i.rate = Some(rate);
        })
    }

    /// After a command, ignore the next ones for this long
    ///
    /// # Panics
    ///
    /// Once the effector is shared: a clone, or the fabric, holds it. Configure it first.
    pub fn refractory(self, period: Duration) -> Self {
        self.configure(|i| i.refractory = Some(period))
    }

    pub fn metrics(&self) -> EffectorMetrics {
        let c = &self.inner.counters;
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let breaker = match state.open_until {
            None => BreakerState::Closed,
            Some(until) if Instant::now() < until => BreakerState::Open,
            Some(_) => BreakerState::HalfOpen,
        };
        EffectorMetrics {
            acted: load(&c.acted),
            failed: load(&c.failed),
            retries: load(&c.retries),
            dropped_open: load(&c.dropped_open),
            dropped_rate: load(&c.dropped_rate),
            quiet_refractory: load(&c.quiet_refractory),
            inflight: load(&c.inflight),
            breaker,
            latency: self.inner.latency.snapshot(),
        }
    }

    /// The act, for its own state (a channel's receiver, an SSE effector's subscribers)
    pub fn act(&self) -> &dyn Act {
        &*self.inner.act
    }
}

impl Neuron for Effector {
    fn name(&self) -> &str {
        self.inner.act.name()
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Effector
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        let inner = &self.inner;
        let delay = match inner.admit(cx.now) {
            Admission::After(delay) => delay,
            Admission::Refractory | Admission::Open | Admission::RateLimited => {
                return Firing::Quiet;
            }
        };
        // Borrowed unless a plain signal's payload has to become JSON arguments; owned only
        // for the motor side's runtime
        let command = match signal.payload.as_command() {
            Some(command) => Cow::Borrowed(command),
            // A plain signal: the payload is the argument
            None => match signal.payload.to_json() {
                Some(args) => Cow::Owned(Command::new(self.name(), args)),
                None => Cow::Borrowed(&inner.plain),
            },
        };
        let started = cx.now;
        let spawn = |pending: Option<ActFuture>, attempt: u32, delay: Duration| {
            cx.efferent().spawn(run(
                self.inner.clone(),
                command.clone().into_owned(),
                signal.clone(),
                pending,
                attempt,
                started,
                delay,
            ));
        };
        if !delay.is_zero() {
            spawn(None, 0, delay);
            return Firing::Fire(Vec::new());
        }
        match inner.act.act(&command, &signal) {
            Action::Done(Ok(())) => inner.succeeded(started),
            Action::Done(Err(error)) => {
                if error.retry && inner.retry.attempts > 1 {
                    inner.counters.retries.fetch_add(1, Ordering::Relaxed);
                    spawn(None, 1, inner.backoff(1));
                } else {
                    inner.failed(&error);
                }
            }
            Action::Pending(future) => spawn(Some(future), 0, Duration::ZERO),
        }
        Firing::Fire(Vec::new())
    }

    fn prepare(&self, cx: &Cx<'_>) {
        self.inner.act.prepare(cx.efferent());
    }

    fn describe(&self) -> serde_json::Value {
        let inner = &self.inner;
        serde_json::json!({
            "retry": {
                "attempts": inner.retry.attempts,
                "backoff_ms": inner.retry.backoff.as_secs_f64() * 1e3,
            },
            "breaker": {
                "failures": inner.breaker.failures,
                "cooldown_ms": inner.breaker.cooldown.as_secs_f64() * 1e3,
            },
            "rate_limit": inner.rate.map(|r| serde_json::json!({
                "per_second": r.per_second, "burst": r.burst,
            })),
            "refractory_ms": inner.refractory.map(|d| d.as_secs_f64() * 1e3),
            "act": inner.act.describe(),
        })
    }

    fn metrics(&self) -> serde_json::Value {
        serde_json::to_value(Effector::metrics(self)).unwrap_or_default()
    }

    fn events(&self) -> Option<tokio::sync::broadcast::Receiver<Bytes>> {
        self.inner.act.events()
    }

    fn render_prometheus(&self, out: &mut String, labels: &str) {
        use std::fmt::Write as _;
        let m = Effector::metrics(self);
        for (name, value) in [
            ("acted", m.acted),
            ("failed", m.failed),
            ("retries", m.retries),
            ("dropped_open", m.dropped_open),
            ("dropped_rate", m.dropped_rate),
            ("quiet_refractory", m.quiet_refractory),
        ] {
            let _ = writeln!(
                out,
                "furnace_nerve_effector_{name}_total{{{labels}}} {value}"
            );
        }
        let _ = writeln!(
            out,
            "furnace_nerve_effector_inflight{{{labels}}} {}",
            m.inflight
        );
        let _ = writeln!(
            out,
            "furnace_nerve_effector_circuit_open{{{labels}}} {}",
            u8::from(m.breaker != BreakerState::Closed)
        );
    }
}

/// The names of the Prometheus families effectors write, for `/metrics`' HELP lines
pub(crate) fn render_prometheus_help(out: &mut String) {
    use std::fmt::Write as _;
    for (name, help) in [
        ("acted_total", "Commands the effector carried out"),
        ("failed_total", "Commands that failed after their retries"),
        ("retries_total", "Retries"),
        (
            "dropped_open_total",
            "Commands dropped while the circuit was open",
        ),
        ("dropped_rate_total", "Commands dropped by the rate limit"),
        (
            "quiet_refractory_total",
            "Commands ignored in the refractory period",
        ),
    ] {
        let _ = writeln!(
            out,
            "# HELP furnace_nerve_effector_{name} {help}\n# TYPE furnace_nerve_effector_{name} counter"
        );
    }
    let _ = writeln!(
        out,
        "# HELP furnace_nerve_effector_inflight Commands in progress\n# TYPE furnace_nerve_effector_inflight gauge\n\
         # HELP furnace_nerve_effector_circuit_open 1 while the effector's circuit breaker is open or half open\n# TYPE furnace_nerve_effector_circuit_open gauge"
    );
}
