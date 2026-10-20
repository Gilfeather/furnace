//! Receptors: the afferent side. Each samples the world on its own task or thread, extracts
//! a feature, stamps the urgency and hands the signal to the fabric; habituation lives in
//! its gate (the [`Mechanisms`](super::Mechanisms) it is added with).
//!
//! - [`Http`]: a route; the feature is a 64-bit sketch of the body, the urgency comes from a
//!   deadline header or the route's default
//! - [`ChannelIn`]: an in-process stream, fed from your own code
//! - [`Timer`]: periodic signals
//! - [`Lines`]: a line stream, from stdin, a TCP connection or a TCP port
//! - [`SseIn`]: an upstream Server-Sent-Events feed
//!
//! A receptor's own `stimulate` passes the signal on: the mechanisms in front of it are what
//! make it more than a source.

mod channel;
mod http;
mod lines;
mod sse;
mod timer;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use bytes::Bytes;

use super::fabric::{Fabric, Inner, Outcome};
use super::neuron::Neuron;
use super::signal::{Feature, NeuronId, Payload, Signal, Urgency};

pub use channel::{ChannelIn, ChannelSender};
pub use http::{DEADLINE_HEADER, Http};
pub use lines::Lines;
pub use sse::SseIn;
pub use timer::Timer;

/// Feature kinds the receptors use, so sketches of different origins never collide
pub mod kinds {
    /// [`Feature::sketch`](super::Feature::sketch): a hash of bytes
    pub const BYTES: u16 = 1;
    /// A sketch of a text line ([`Lines`](super::Lines))
    pub const TEXT: u16 = 2;
    /// A timer's tick, or its one fixed feature ([`Timer`](super::Timer))
    pub const TIMER: u16 = 3;
    /// A sketch of an upstream event's data ([`SseIn`](super::SseIn))
    pub const SSE: u16 = 4;
}

/// The fabric's stop signal, raised when the fabric is dropped
#[derive(Default)]
pub(crate) struct Stop {
    raised: AtomicBool,
    /// For threads sleeping in [`Afferent::sleep`]
    lock: Mutex<()>,
    wake: Condvar,
    /// For tasks awaiting [`Afferent::stopped`]
    notify: tokio::sync::Notify,
}

impl Stop {
    pub(crate) fn raise(&self) {
        self.raised.store(true, Ordering::Release);
        // Under the lock, so a thread between its check and its wait cannot miss it
        drop(self.lock.lock().unwrap_or_else(|e| e.into_inner()));
        self.wake.notify_all();
        self.notify.notify_waiters();
    }

    fn is_raised(&self) -> bool {
        self.raised.load(Ordering::Acquire)
    }
}

/// A receptor's way into the fabric, given to it by [`Fabric::start`]. It holds the fabric
/// weakly, and carries the fabric's stop signal: once the fabric is dropped,
/// [`is_stopped`](Self::is_stopped) is true, [`sleep`](Self::sleep) returns `false` at once,
/// [`stopped`](Self::stopped) resolves and [`emit`](Self::emit) returns `None`. A receptor's
/// thread or task ends then.
#[derive(Clone)]
pub struct Afferent {
    inner: Weak<Inner>,
    id: NeuronId,
    stop: Arc<Stop>,
}

impl Afferent {
    pub(crate) fn new(inner: Weak<Inner>, id: NeuronId, stop: Arc<Stop>) -> Self {
        Self { inner, id, stop }
    }

    /// Whether the fabric was dropped: the receptor should end what it started
    pub fn is_stopped(&self) -> bool {
        self.stop.is_raised()
    }

    /// Block this thread for `duration`, or less if the fabric stops meanwhile. `false` once
    /// it stopped: a receptor's thread sleeps with this between samples rather than with
    /// `std::thread::sleep`, so that it ends with the fabric.
    pub fn sleep(&self, duration: Duration) -> bool {
        let deadline = Instant::now() + duration;
        let mut guard = self.stop.lock.lock().unwrap_or_else(|e| e.into_inner());
        while !self.stop.is_raised() {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return true;
            };
            guard = self
                .stop
                .wake
                .wait_timeout(guard, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        false
    }

    /// Resolves once the fabric was dropped; a receptor's task selects on it
    pub async fn stopped(&self) {
        loop {
            let notified = self.stop.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.stop.is_raised() {
                return;
            }
            notified.await;
        }
    }

    /// This receptor's id
    pub fn id(&self) -> NeuronId {
        self.id
    }

    /// The fabric, while it lives
    pub fn fabric(&self) -> Option<Fabric> {
        self.inner.upgrade().map(Fabric)
    }

    /// Stimulate the fabric through this receptor; the whole synchronous chain runs before
    /// this returns. `None` once the fabric is gone.
    pub fn emit(&self, signal: Signal) -> Option<Outcome> {
        let fabric = self.fabric()?;
        Some(fabric.stimulate(self.id, signal))
    }

    /// The receptive side's runtime, for receptors that read the network
    pub fn runtime(&self) -> Option<tokio::runtime::Handle> {
        self.fabric().map(|f| f.afferent().clone())
    }
}

/// A neuron that is also a source: started with the fabric, and possibly mounted as a route
pub trait Receptor: Neuron {
    /// Start sampling (a thread, or a task on [`Afferent::runtime`]); return at once. What it
    /// starts ends when the fabric is dropped: wait with [`Afferent::sleep`] or select on
    /// [`Afferent::stopped`], and stop when [`Afferent::emit`] returns `None`. Tasks on the
    /// fabric's runtime are cancelled with it.
    fn start(&self, _afferent: Afferent) {}

    /// Mount this receptor as an HTTP route: the fabric's router
    /// ([`Server::nerve`](crate::Server::nerve)) calls [`Receptor::ingest`] with each request
    fn mount(&self) -> Option<HttpMount> {
        None
    }

    /// Make the signal for an HTTP request (see [`Receptor::mount`]); `None` ignores it
    fn ingest(&self, _request: Ingest<'_>) -> Option<Signal> {
        None
    }

    /// How long a request whose signal was queued to a deliberative pathway waits for a
    /// response from that side; zero (the default) answers `202 Accepted` at once
    fn respond_within(&self) -> Duration {
        Duration::ZERO
    }
}

/// Where a receptor is mounted
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpMount {
    /// An axum route path, e.g. `/events`
    pub path: String,
    /// The HTTP method the route answers
    pub method: HttpMethod,
}

/// The HTTP method a mounted receptor answers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    /// `GET`
    Get,
    /// `POST`
    Post,
}

/// What an HTTP receptor sees of a request
pub struct Ingest<'a> {
    /// The request body, whole
    pub body: Bytes,
    /// The `Content-Type` header, if any and valid text
    pub content_type: Option<&'a str>,
    /// All the request's headers
    pub headers: &'a axum::http::HeaderMap,
}

/// How a receptor stamps urgency on what it sees
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UrgencyRule {
    /// Every signal wants an action within this
    pub deadline: Option<Duration>,
    /// Or within this many inter-arrival intervals at the measured rate ("act before the next
    /// sample"): 1.0 is one interval. Used when the rate is known; `deadline` otherwise.
    pub intervals: Option<f32>,
}

impl UrgencyRule {
    /// No deadline: signals are not urgent
    pub const NONE: Self = Self {
        deadline: None,
        intervals: None,
    };

    /// Every signal wants an action within `deadline`
    pub fn within(deadline: Duration) -> Self {
        Self {
            deadline: Some(deadline),
            intervals: None,
        }
    }

    /// Deadline from the stream's rate: `intervals` inter-arrival times from now
    pub fn before_next(intervals: f32) -> Self {
        Self {
            deadline: None,
            intervals: Some(intervals),
        }
    }

    fn stamp(&self, rate: f32, now: Instant) -> Urgency {
        let from_rate = self
            .intervals
            .filter(|_| rate > 0.0)
            .map(|k| Duration::from_secs_f32((k / rate).clamp(1e-6, 3600.0)));
        Urgency {
            deadline: from_rate.or(self.deadline).map(|d| now + d),
            rate,
        }
    }
}

/// The rate a stream arrives at: an exponential moving average of inter-arrival times, in
/// signals per second, without a clock in the hot path beyond one `Instant`
pub(crate) struct RateMeter {
    /// Nanoseconds since `epoch` of the last arrival; 0 before any
    last: AtomicU64,
    /// EWMA of the interval, nanoseconds as bits of an f64
    interval: AtomicU64,
    epoch: Instant,
}

impl Default for RateMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateMeter {
    /// No arrivals yet: rate 0
    pub fn new() -> Self {
        Self {
            last: AtomicU64::new(0),
            interval: AtomicU64::new(0f64.to_bits()),
            epoch: Instant::now(),
        }
    }

    /// Record an arrival and return the rate so far, in per second (0 for the first)
    pub fn tick(&self, now: Instant) -> f32 {
        let nanos = super::nanos_since(self.epoch, now).max(1);
        let last = self.last.swap(nanos, Ordering::Relaxed);
        if last == 0 {
            return 0.0;
        }
        let dt = nanos.saturating_sub(last) as f64;
        let previous = f64::from_bits(self.interval.load(Ordering::Relaxed));
        let interval = if previous == 0.0 {
            dt
        } else {
            previous * 0.8 + dt * 0.2
        };
        self.interval.store(interval.to_bits(), Ordering::Relaxed);
        if interval > 0.0 {
            (1e9 / interval) as f32
        } else {
            0.0
        }
    }

    /// The rate so far, in per second, without recording an arrival (0 before two)
    pub fn rate(&self) -> f32 {
        let interval = f64::from_bits(self.interval.load(Ordering::Relaxed));
        if interval > 0.0 {
            (1e9 / interval) as f32
        } else {
            0.0
        }
    }
}

/// The wait before reconnecting to a source that dropped or refused: from `initial`, doubling
/// up to 10 s, and back to `initial` once connected
struct Backoff {
    initial: Duration,
    next: Duration,
}

impl Backoff {
    fn new(initial: Duration) -> Self {
        Self {
            initial,
            next: initial,
        }
    }

    /// Connected: the next wait is the first one again
    fn reset(&mut self) {
        self.next = self.initial;
    }

    /// The wait before this attempt; the one after it is twice as long
    fn next(&mut self) -> Duration {
        let wait = self.next;
        self.next = (wait * 2).min(Duration::from_secs(10));
        wait
    }
}

/// What every receptor here shares: a name, how it stamps urgency, and its rate
struct Source {
    name: String,
    rule: UrgencyRule,
    meter: RateMeter,
    afferent: OnceLock<Afferent>,
}

impl Source {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            rule: UrgencyRule::NONE,
            meter: RateMeter::new(),
            afferent: OnceLock::new(),
        }
    }

    /// A signal seen now, stamped with the rule
    fn signal(&self, feature: Feature, payload: Payload) -> Signal {
        let now = Instant::now();
        let rate = self.meter.tick(now);
        let mut signal = Signal::new(feature, payload);
        signal.at = now;
        signal.urgency = self.rule.stamp(rate, now);
        signal
    }

    fn emit(&self, signal: Signal) -> Option<Outcome> {
        self.afferent.get()?.emit(signal)
    }

    fn describe(&self, extra: serde_json::Value) -> serde_json::Value {
        let mut value = serde_json::json!({
            "rate_per_s": self.meter.rate(),
            "deadline_ms": self.rule.deadline.map(|d| d.as_secs_f64() * 1e3),
            "deadline_intervals": self.rule.intervals,
        });
        if let (Some(a), Some(b)) = (value.as_object_mut(), extra.as_object()) {
            a.extend(b.clone());
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stop_signal_wakes_a_sleeping_receptor() {
        let stop = Arc::new(Stop::default());
        let afferent = Afferent::new(Weak::new(), NeuronId(0), stop.clone());
        assert!(afferent.sleep(Duration::from_millis(1)));
        let sleeper = std::thread::spawn({
            let afferent = afferent.clone();
            move || afferent.sleep(Duration::from_secs(3600))
        });
        // Raised while the sleeper waits or before it starts: either way it returns false
        stop.raise();
        assert!(!sleeper.join().unwrap());
        assert!(afferent.is_stopped());
        assert!(!afferent.sleep(Duration::from_secs(3600)));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(afferent.stopped());
    }
}
