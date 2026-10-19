//! Receptors: the afferent side. Each samples the world on its own task or thread, extracts
//! a feature, stamps the urgency and hands the signal to the fabric; habituation lives in
//! its gate (the [`Mechanisms`](super::Mechanisms) it is added with).
//!
//! - [`Http`]: a route; the feature is a 64-bit sketch of the body, the urgency comes from a
//!   deadline header or the route's default
//! - [`Channel`]: an in-process stream, fed from your own code
//! - [`Timer`]: periodic signals
//! - [`Lines`]: a line stream, from stdin, a TCP connection or a TCP port
//! - [`Sse`]: an upstream Server-Sent-Events feed
//!
//! A receptor's own `stimulate` passes the signal on: the mechanisms in front of it are what
//! make it more than a source.

use std::io::BufRead;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak, mpsc};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tracing::{debug, warn};

use super::fabric::{Fabric, Inner, Outcome};
use super::neuron::{Cx, Firing, Neuron, NeuronKind};
use super::signal::{Feature, NeuronId, Payload, Signal, Urgency};
use super::wire;

/// Feature kinds the receptors use, so sketches of different origins never collide
pub mod kinds {
    /// [`Feature::sketch`](super::Feature::sketch): a hash of bytes
    pub const BYTES: u16 = 1;
    pub const TEXT: u16 = 2;
    pub const TIMER: u16 = 3;
    pub const SSE: u16 = 4;
}

/// A receptor's way into the fabric, given to it by [`Fabric::start`]. It holds the fabric
/// weakly: once the fabric is gone, [`Afferent::emit`] returns `None` and the receptor's
/// task should stop.
#[derive(Clone)]
pub struct Afferent {
    inner: Weak<Inner>,
    id: NeuronId,
}

impl Afferent {
    pub(crate) fn new(inner: Weak<Inner>, id: NeuronId) -> Self {
        Self { inner, id }
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
    /// Start sampling (a thread, or a task on [`Afferent::runtime`]); return at once
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
    pub method: HttpMethod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

/// What an HTTP receptor sees of a request
pub struct Ingest<'a> {
    pub body: Bytes,
    pub content_type: Option<&'a str>,
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
    pub const NONE: Self = Self {
        deadline: None,
        intervals: None,
    };

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
pub struct RateMeter {
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
    pub fn new() -> Self {
        Self {
            last: AtomicU64::new(0),
            interval: AtomicU64::new(0f64.to_bits()),
            epoch: Instant::now(),
        }
    }

    /// Record an arrival and return the rate so far, in per second (0 for the first)
    pub fn tick(&self, now: Instant) -> f32 {
        let nanos = u64::try_from(now.saturating_duration_since(self.epoch).as_nanos())
            .unwrap_or(u64::MAX)
            .max(1);
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

    pub fn rate(&self) -> f32 {
        let interval = f64::from_bits(self.interval.load(Ordering::Relaxed));
        if interval > 0.0 {
            (1e9 / interval) as f32
        } else {
            0.0
        }
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

// ---------------------------------------------------------------------------------------------
// HTTP

/// The header a client sets to say how soon it needs an action, in milliseconds
pub const DEADLINE_HEADER: &str = "x-deadline-ms";

type FeatureFn = dyn Fn(&Ingest<'_>) -> Feature + Send + Sync;

/// An HTTP route: each request is one signal. The feature is a 64-bit sketch of the body
/// unless the receptor is given its own; the urgency comes from the `x-deadline-ms` header,
/// else the route's default.
///
/// The response is whatever a [`Respond`](super::effector::Respond) effector on the chain
/// answers; a chain that ends without one answers `202 Accepted` (queued to the brain) or
/// `204 No Content` (a reflex acted, or the signal died at the gate), with the outcome in the
/// `x-furnace-nerve` header.
pub struct Http {
    source: Source,
    mount: HttpMount,
    feature: Option<Arc<FeatureFn>>,
    text: bool,
    respond_within: Duration,
}

impl Http {
    /// `POST path`
    pub fn new(name: &str, path: &str) -> Self {
        Self {
            source: Source::new(name),
            mount: HttpMount {
                path: path.to_string(),
                method: HttpMethod::Post,
            },
            feature: None,
            text: false,
            respond_within: Duration::ZERO,
        }
    }

    /// Wait up to this long for a response when the signal went to a deliberative pathway
    /// (the brain answering through a `Respond` effector); by default such requests get
    /// `202 Accepted` at once
    pub fn respond_within(mut self, wait: Duration) -> Self {
        self.respond_within = wait;
        self
    }

    pub fn get(mut self) -> Self {
        self.mount.method = HttpMethod::Get;
        self
    }

    /// The urgency of requests without a deadline header
    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Extract the feature yourself (an embedding sketch, a field of the body)
    pub fn feature(mut self, f: impl Fn(&Ingest<'_>) -> Feature + Send + Sync + 'static) -> Self {
        self.feature = Some(Arc::new(f));
        self
    }

    /// Carry the body as text (`Payload::Text`) rather than bytes; non-UTF-8 bodies are
    /// ignored
    pub fn text(mut self) -> Self {
        self.text = true;
        self
    }
}

impl Neuron for Http {
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
            "route": format!("{:?} {}", self.mount.method, self.mount.path).to_uppercase(),
        }))
    }
}

impl Receptor for Http {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent);
    }

    fn mount(&self) -> Option<HttpMount> {
        Some(self.mount.clone())
    }

    fn respond_within(&self) -> Duration {
        self.respond_within
    }

    fn ingest(&self, request: Ingest<'_>) -> Option<Signal> {
        let feature = match &self.feature {
            Some(f) => f(&request),
            None => Feature::sketch(&request.body),
        };
        let payload = if self.text {
            Payload::Text(Arc::from(std::str::from_utf8(&request.body).ok()?))
        } else {
            Payload::Bytes(request.body)
        };
        let mut signal = self.source.signal(feature, payload);
        if let Some(ms) = request
            .headers
            .get(DEADLINE_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|ms| ms.is_finite() && *ms >= 0.0)
        {
            signal.urgency.deadline = Some(signal.at + Duration::from_secs_f64(ms / 1e3));
        }
        Some(signal)
    }
}

// ---------------------------------------------------------------------------------------------
// In-process channel

/// An in-process stream: your code feeds it signals, from a sensor loop, another thread, a
/// test. [`ChannelSender::emit`] runs the chain inline on the caller's thread (no queue at
/// all); [`ChannelSender::send`] queues to the receptor's own thread, which samples the
/// queue and never blocks the caller.
pub struct Channel {
    source: Arc<Source>,
    receiver: Mutex<Option<mpsc::Receiver<Signal>>>,
}

/// The feeding end of a [`Channel`]
#[derive(Clone)]
pub struct ChannelSender {
    source: Arc<Source>,
    sender: mpsc::SyncSender<Signal>,
}

impl Channel {
    /// A channel receptor and its sender; `capacity` bounds what waits for the sampling
    /// thread, and `send` fails beyond it rather than blocking
    pub fn new(name: &str, capacity: usize) -> (Self, ChannelSender) {
        Self::with_urgency(name, capacity, UrgencyRule::NONE)
    }

    /// [`Channel::new`] with the rule its signals are stamped by
    pub fn with_urgency(name: &str, capacity: usize, rule: UrgencyRule) -> (Self, ChannelSender) {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        let mut source = Source::new(name);
        source.rule = rule;
        let source = Arc::new(source);
        (
            Self {
                source: source.clone(),
                receiver: Mutex::new(Some(receiver)),
            },
            ChannelSender { source, sender },
        )
    }
}

impl ChannelSender {
    /// A signal of `payload` with `feature`, stamped by the receptor's rule, seen now
    pub fn signal(&self, feature: Feature, payload: Payload) -> Signal {
        self.source.signal(feature, payload)
    }

    /// Run the whole synchronous chain on this thread, now; `None` before the fabric is
    /// started or after it is gone
    pub fn emit(&self, signal: Signal) -> Option<Outcome> {
        self.source.emit(signal)
    }

    /// Queue the signal for the receptor's thread; `Err` with the signal if the queue is full
    /// or the receptor is gone
    pub fn send(&self, signal: Signal) -> Result<(), Box<Signal>> {
        self.sender.try_send(signal).map_err(|e| match e {
            mpsc::TrySendError::Full(s) | mpsc::TrySendError::Disconnected(s) => Box::new(s),
        })
    }
}

impl Neuron for Channel {
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
        self.source
            .describe(serde_json::json!({"source": "channel"}))
    }
}

impl Receptor for Channel {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent.clone());
        let Some(receiver) = self
            .receiver
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        else {
            return;
        };
        let name = self.source.name.clone();
        std::thread::Builder::new()
            .name(format!("furnace-receptor-{name}"))
            .spawn(move || {
                while let Ok(signal) = receiver.recv() {
                    if afferent.emit(signal).is_none() {
                        return;
                    }
                }
            })
            .expect("failed to spawn a receptor thread");
    }
}

// ---------------------------------------------------------------------------------------------
// Timer

/// Periodic signals, one per `period`, from a thread of their own. The payload is
/// `{"tick": n}`; the feature is the tick number by default, so ticks never habituate, or
/// one fixed feature with [`Timer::same_feature`] so they do.
pub struct Timer {
    source: Source,
    period: Duration,
    same_feature: bool,
}

impl Timer {
    pub fn new(name: &str, period: Duration) -> Self {
        Self {
            source: Source::new(name),
            period: period.max(Duration::from_micros(10)),
            same_feature: false,
        }
    }

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
                    if next > now {
                        std::thread::sleep(next - now);
                    }
                    next += period;
                    let now = Instant::now();
                    let feature = Feature::new(kinds::TIMER, if same { 0 } else { tick });
                    let mut signal =
                        Signal::new(feature, Payload::json(serde_json::json!({"tick": tick})));
                    signal.at = now;
                    signal.urgency = rule.stamp(meter.tick(now), now);
                    if afferent.emit(signal).is_none() {
                        return;
                    }
                }
            })
            .expect("failed to spawn a receptor thread");
    }
}

// ---------------------------------------------------------------------------------------------
// Line streams

enum LineSource {
    Stdin,
    Connect(String),
    Listen(String),
}

/// A stream of text lines: one signal per line, the feature a sketch of the line, the
/// payload the line as text. From stdin, from a TCP connection to a host, or from every
/// connection to a TCP port this receptor listens on. Reading runs on threads of its own.
pub struct Lines {
    source: Source,
    from: LineSource,
}

impl Lines {
    pub fn stdin(name: &str) -> Self {
        Self {
            source: Source::new(name),
            from: LineSource::Stdin,
        }
    }

    /// Connect to `addr` (`host:port`) and read lines, reconnecting with backoff when it drops
    pub fn connect(name: &str, addr: &str) -> Self {
        Self {
            source: Source::new(name),
            from: LineSource::Connect(addr.to_string()),
        }
    }

    /// Listen on `addr` and read lines from every connection
    pub fn listen(name: &str, addr: &str) -> Self {
        Self {
            source: Source::new(name),
            from: LineSource::Listen(addr.to_string()),
        }
    }

    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Make the signal for one line
    pub fn line_signal(rule: UrgencyRule, meter: &RateMeter, line: &str) -> Signal {
        let now = Instant::now();
        let mut signal = Signal::new(
            Feature::sketch_as(kinds::TEXT, line.as_bytes()),
            Payload::text(line),
        );
        signal.at = now;
        signal.urgency = rule.stamp(meter.tick(now), now);
        signal
    }
}

/// Read lines from `reader` into the fabric until it ends or the fabric is gone; `true` if
/// the reader ended
fn pump_lines(
    reader: impl std::io::Read,
    afferent: &Afferent,
    rule: UrgencyRule,
    meter: &RateMeter,
) -> bool {
    let reader = std::io::BufReader::new(reader);
    for line in reader.lines() {
        let Ok(line) = line else {
            return true;
        };
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if afferent
            .emit(Lines::line_signal(rule, meter, line))
            .is_none()
        {
            return false;
        }
    }
    true
}

impl Neuron for Lines {
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
        let source = match &self.from {
            LineSource::Stdin => "stdin".to_string(),
            LineSource::Connect(addr) => format!("tcp connect {addr}"),
            LineSource::Listen(addr) => format!("tcp listen {addr}"),
        };
        self.source.describe(serde_json::json!({"source": source}))
    }
}

impl Receptor for Lines {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent.clone());
        let rule = self.source.rule;
        let name = self.source.name.clone();
        let spawn = |f: Box<dyn FnOnce() + Send>| {
            std::thread::Builder::new()
                .name(format!("furnace-receptor-{name}"))
                .spawn(f)
                .expect("failed to spawn a receptor thread");
        };
        match &self.from {
            LineSource::Stdin => spawn(Box::new(move || {
                pump_lines(std::io::stdin().lock(), &afferent, rule, &RateMeter::new());
            })),
            LineSource::Connect(addr) => {
                let addr = addr.clone();
                spawn(Box::new(move || {
                    let meter = RateMeter::new();
                    let mut backoff = Duration::from_millis(100);
                    loop {
                        match std::net::TcpStream::connect(&addr) {
                            Ok(stream) => {
                                backoff = Duration::from_millis(100);
                                if !pump_lines(stream, &afferent, rule, &meter) {
                                    return;
                                }
                            }
                            Err(e) => debug!("nerve: lines receptor: connect {addr}: {e}"),
                        }
                        if afferent.fabric().is_none() {
                            return;
                        }
                        std::thread::sleep(backoff);
                        backoff = (backoff * 2).min(Duration::from_secs(10));
                    }
                }));
            }
            LineSource::Listen(addr) => {
                let listener = match std::net::TcpListener::bind(addr) {
                    Ok(listener) => listener,
                    Err(e) => {
                        warn!("nerve: lines receptor cannot listen on {addr}: {e}");
                        return;
                    }
                };
                spawn(Box::new(move || {
                    let meter = Arc::new(RateMeter::new());
                    for stream in listener.incoming() {
                        let Ok(stream) = stream else { continue };
                        if afferent.fabric().is_none() {
                            return;
                        }
                        let (afferent, meter) = (afferent.clone(), meter.clone());
                        std::thread::spawn(move || {
                            pump_lines(stream, &afferent, rule, &meter);
                        });
                    }
                }));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Server-Sent Events

/// An upstream event feed: `GET url` as `text/event-stream`, one signal per event (the
/// `data` as text, the feature a sketch of it), reconnecting with backoff and
/// `Last-Event-ID`. Runs on the fabric's afferent runtime.
pub struct Sse {
    source: Source,
    url: String,
    event: Option<String>,
}

impl Sse {
    pub fn new(name: &str, url: &str) -> Self {
        Self {
            source: Source::new(name),
            url: url.to_string(),
            event: None,
        }
    }

    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Only events of this `event:` type
    pub fn event(mut self, event: &str) -> Self {
        self.event = Some(event.to_string());
        self
    }
}

impl Neuron for Sse {
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
            "source": format!("sse {}", self.url),
            "event": self.event,
        }))
    }
}

/// One parsed SSE event
#[derive(Debug, Default, PartialEq, Eq)]
struct SseEvent {
    event: Option<String>,
    data: String,
    id: Option<String>,
}

/// Incremental parser of the `text/event-stream` format
#[derive(Default)]
struct SseParser {
    buffer: String,
    current: SseEvent,
    has_data: bool,
}

impl SseParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut events = Vec::new();
        while let Some(i) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=i).collect();
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if self.has_data {
                    let mut event = std::mem::take(&mut self.current);
                    if event.data.ends_with('\n') {
                        event.data.pop();
                    }
                    events.push(event);
                }
                self.has_data = false;
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "data" => {
                    self.current.data.push_str(value);
                    self.current.data.push('\n');
                    self.has_data = true;
                }
                "event" => self.current.event = Some(value.to_string()),
                "id" => self.current.id = Some(value.to_string()),
                _ => {}
            }
        }
        events
    }
}

impl Receptor for Sse {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent.clone());
        let Some(runtime) = afferent.runtime() else {
            return;
        };
        let url = match wire::Url::parse(&self.url) {
            Ok(url) => url,
            Err(e) => {
                warn!("nerve: sse receptor: {e}");
                return;
            }
        };
        let (rule, wanted) = (self.source.rule, self.event.clone());
        runtime.spawn(async move {
            let client = wire::Client::new(Duration::from_secs(3600));
            let meter = RateMeter::new();
            let mut last_id: Option<String> = None;
            let mut backoff = Duration::from_millis(200);
            loop {
                let mut headers = vec![("accept", "text/event-stream")];
                let id = last_id.clone().unwrap_or_default();
                if last_id.is_some() {
                    headers.push(("last-event-id", &id));
                }
                match client.send("GET", &url, &headers, b"").await {
                    Ok(mut streaming) if streaming.status == 200 => {
                        backoff = Duration::from_millis(200);
                        let mut parser = SseParser::default();
                        loop {
                            match streaming.next().await {
                                Ok(Some(piece)) => {
                                    for event in parser.push(&piece) {
                                        if let Some(id) = &event.id {
                                            last_id = Some(id.clone());
                                        }
                                        if wanted.as_ref().is_some_and(|w| {
                                            event.event.as_deref().unwrap_or("message") != w
                                        }) {
                                            continue;
                                        }
                                        let now = Instant::now();
                                        let mut signal = Signal::new(
                                            Feature::sketch_as(kinds::SSE, event.data.as_bytes()),
                                            Payload::text(&event.data),
                                        );
                                        signal.at = now;
                                        signal.urgency = rule.stamp(meter.tick(now), now);
                                        if afferent.emit(signal).is_none() {
                                            return;
                                        }
                                    }
                                }
                                Ok(None) => break,
                                Err(e) => {
                                    debug!("nerve: sse receptor: {e}");
                                    break;
                                }
                            }
                        }
                    }
                    Ok(streaming) => {
                        debug!(
                            "nerve: sse receptor: upstream answered {}",
                            streaming.status
                        );
                    }
                    Err(e) => debug!("nerve: sse receptor: {e}"),
                }
                if afferent.fabric().is_none() {
                    return;
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(10));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::fabric::FnNeuron;
    use crate::nerve::neuron::Mechanisms;
    use std::sync::atomic::AtomicUsize;

    /// A fabric with one receptor and a sink that records what arrives
    fn fabric_with<R: Receptor>(receptor: R) -> (Fabric, NeuronId, Arc<Mutex<Vec<Signal>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut b = Fabric::builder();
        let r = b.receptor(receptor, Mechanisms::NONE);
        let sink = {
            let seen = seen.clone();
            b.neuron(
                FnNeuron::reflex("sink", move |s, _| {
                    seen.lock().unwrap().push(s);
                    Firing::Quiet
                })
                .with_kind(NeuronKind::Effector),
                Mechanisms::NONE,
            )
        };
        b.reflex(r, sink, 1.0);
        let fabric = b.build();
        fabric.start();
        (fabric, r, seen)
    }

    fn wait_for(seen: &Mutex<Vec<Signal>>, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while seen.lock().unwrap().len() < n {
            assert!(
                Instant::now() < deadline,
                "{} signals",
                seen.lock().unwrap().len()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn http_receptor_sketches_the_body_and_reads_the_deadline_header() {
        let http = Http::new("in", "/events").urgency(UrgencyRule::within(Duration::from_secs(1)));
        let mut headers = axum::http::HeaderMap::new();
        let signal = http
            .ingest(Ingest {
                body: Bytes::from_static(b"hello"),
                content_type: None,
                headers: &headers,
            })
            .unwrap();
        assert_eq!(signal.feature, Feature::sketch(b"hello"));
        assert_eq!(signal.payload.as_text(), Some("hello"));
        let remaining = signal.urgency.remaining(Instant::now()).unwrap();
        assert!(remaining > Duration::from_millis(900), "{remaining:?}");
        headers.insert(DEADLINE_HEADER, "5".parse().unwrap());
        let urgent = http
            .ingest(Ingest {
                body: Bytes::from_static(b"hello"),
                content_type: None,
                headers: &headers,
            })
            .unwrap();
        assert!(urgent.urgency.remaining(Instant::now()).unwrap() <= Duration::from_millis(5));
        assert_eq!(http.mount().unwrap().path, "/events");
        let own = Http::new("own", "/x").feature(|_| Feature::new(9, 42));
        let signal = own
            .ingest(Ingest {
                body: Bytes::new(),
                content_type: None,
                headers: &headers,
            })
            .unwrap();
        assert_eq!(signal.feature, Feature::new(9, 42));
    }

    #[test]
    fn channel_receptor_emits_inline_and_through_its_thread() {
        let (channel, sender) = Channel::new("stream", 16);
        let (_fabric, r, seen) = fabric_with(channel);
        let outcome = sender
            .emit(sender.signal(Feature::sketch(b"a"), Payload::text("a")))
            .unwrap();
        assert_eq!(outcome.acted, 1);
        assert_eq!(seen.lock().unwrap().len(), 1);
        sender
            .send(sender.signal(Feature::sketch(b"b"), Payload::text("b")))
            .unwrap();
        wait_for(&seen, 2);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[1].origin, r);
        assert_eq!(seen[1].payload.as_text(), Some("b"));
    }

    #[test]
    fn timer_receptor_ticks_and_stamps_the_rate() {
        let timer =
            Timer::new("tick", Duration::from_millis(5)).urgency(UrgencyRule::before_next(1.0));
        let (_fabric, _r, seen) = fabric_with(timer);
        wait_for(&seen, 5);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].payload.to_json().unwrap()["tick"], 0);
        assert_ne!(seen[0].feature, seen[1].feature);
        // From the second tick on, the rate is known and the deadline follows it
        assert!(seen[4].urgency.rate > 50.0, "{}", seen[4].urgency.rate);
        assert!(seen[4].urgency.deadline.is_some());
        assert!(seen[0].urgency.deadline.is_none());
    }

    #[test]
    fn lines_receptor_reads_a_tcp_stream() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            use std::io::Write;
            let (mut stream, _) = listener.accept().unwrap();
            stream.write_all(b"first line\r\n\nsecond line\n").unwrap();
        });
        let (_fabric, _r, seen) = fabric_with(Lines::connect("lines", &addr));
        wait_for(&seen, 2);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].payload.as_text(), Some("first line"));
        assert_eq!(seen[1].payload.as_text(), Some("second line"));
        assert_eq!(seen[0].feature.kind, kinds::TEXT);
    }

    #[test]
    fn lines_receptor_listens() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let addr = format!("127.0.0.1:{port}");
        let (_fabric, _r, seen) = fabric_with(Lines::listen("lines", &addr));
        use std::io::Write;
        let mut stream = std::net::TcpStream::connect(&addr).unwrap();
        stream.write_all(b"hello\n").unwrap();
        wait_for(&seen, 1);
        assert_eq!(seen.lock().unwrap()[0].payload.as_text(), Some("hello"));
    }

    #[test]
    fn sse_parser_handles_multi_line_data_ids_and_comments() {
        let mut parser = SseParser::default();
        let events = parser
            .push(b": keep-alive\nid: 7\nevent: temp\ndata: 21.5\ndata: more\n\ndata: plain\n\n");
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("temp".into()),
                    data: "21.5\nmore".into(),
                    id: Some("7".into())
                },
                SseEvent {
                    event: None,
                    data: "plain".into(),
                    id: None
                }
            ]
        );
        // A partial line waits for the rest
        assert!(parser.push(b"data: par").is_empty());
        assert_eq!(parser.push(b"tial\n\n")[0].data, "partial");
    }

    #[tokio::test]
    async fn sse_receptor_follows_an_upstream_feed() {
        use axum::Router;
        use axum::body::Body;
        use axum::routing::get;
        let served = Arc::new(AtomicUsize::new(0));
        let app = Router::new().route(
            "/feed",
            get({
                let served = served.clone();
                move || {
                    served.fetch_add(1, Ordering::SeqCst);
                    async {
                        let events = (0..3).map(|i| {
                            Ok::<_, std::convert::Infallible>(Bytes::from(format!(
                                "id: {i}\ndata: event {i}\n\n"
                            )))
                        });
                        axum::response::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from_stream(tokio_stream::iter(events)))
                            .unwrap()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (fabric, _r, seen) =
            fabric_with(Sse::new("feed", &format!("http://127.0.0.1:{port}/feed")));
        tokio::task::spawn_blocking({
            let seen = seen.clone();
            move || wait_for(&seen, 3)
        })
        .await
        .unwrap();
        {
            let seen = seen.lock().unwrap();
            assert_eq!(seen[2].payload.as_text(), Some("event 2"));
            assert_eq!(seen[2].feature.kind, kinds::SSE);
        }
        // The stream ended: the receptor reconnects (and would resume from the last id)
        let deadline = Instant::now() + Duration::from_secs(5);
        while served.load(Ordering::SeqCst) < 2 {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(fabric);
    }
}
