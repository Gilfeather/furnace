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

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde::Serialize;
use tracing::{debug, warn};

use super::latency::LatencyEstimate;
use super::neuron::{Cx, Firing, Neuron, NeuronKind};
use super::signal::{Answer, Command, Payload, Signal};
use super::wire;

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

struct LoopState {
    consecutive_failures: u32,
    open_until: Option<Instant>,
    /// A trial command is out while half open
    trial: bool,
    tokens: f64,
    refilled_at: Instant,
}

#[derive(Default)]
struct Counters {
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

struct Inner {
    act: Box<dyn Act>,
    retry: Retry,
    breaker: Breaker,
    rate: Option<RateLimit>,
    refractory: Option<Duration>,
    epoch: Instant,
    /// Nanoseconds since `epoch` of the last command admitted; 0 before any
    last_acted: AtomicU64,
    state: Mutex<LoopState>,
    counters: Counters,
    latency: LatencyEstimate,
    runtime: OnceLock<tokio::runtime::Handle>,
}

/// An [`Act`] with its local loop: retry with backoff, circuit breaker, rate limit,
/// refractory period, and metrics. Add it to the fabric like any neuron; its
/// [`Mechanisms`](super::Mechanisms) (habituation, threshold) apply in front of it.
#[derive(Clone)]
pub struct Effector {
    inner: Arc<Inner>,
}

impl Effector {
    /// With the defaults: 3 attempts from 100 ms backoff, the circuit opens after 5 failures
    /// in a row for 10 s, no rate limit, no refractory period
    pub fn new(act: impl Act) -> Self {
        let now = Instant::now();
        Self {
            inner: Arc::new(Inner {
                act: Box::new(act),
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
                counters: Counters::default(),
                latency: LatencyEstimate::new(),
                runtime: OnceLock::new(),
            }),
        }
    }

    fn configure(mut self, f: impl FnOnce(&mut Inner)) -> Self {
        f(Arc::get_mut(&mut self.inner).expect("configured before use"));
        self
    }

    pub fn retry(self, retry: Retry) -> Self {
        self.configure(|i| i.retry = retry)
    }

    /// One try, no retry
    pub fn no_retry(self) -> Self {
        self.retry(Retry {
            attempts: 1,
            ..Retry::default()
        })
    }

    pub fn breaker(self, breaker: Breaker) -> Self {
        self.configure(|i| i.breaker = breaker)
    }

    pub fn rate_limit(self, rate: RateLimit) -> Self {
        self.configure(|i| {
            i.state.get_mut().unwrap_or_else(|e| e.into_inner()).tokens = f64::from(rate.burst);
            i.rate = Some(rate);
        })
    }

    /// After a command, ignore the next ones for this long
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

impl Inner {
    fn nanos(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.epoch).as_nanos()).unwrap_or(u64::MAX)
    }

    /// Whether to act now, after how long, or not at all
    fn admit(&self, now: Instant) -> Admission {
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
            .store(self.nanos(now).max(1), Ordering::Relaxed);
        self.counters.inflight.fetch_add(1, Ordering::Relaxed);
        Admission::After(delay)
    }

    fn succeeded(&self, started: Instant) {
        self.counters.inflight.fetch_sub(1, Ordering::Relaxed);
        self.counters.acted.fetch_add(1, Ordering::Relaxed);
        self.latency.observe(started.elapsed());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.consecutive_failures = 0;
        state.open_until = None;
        state.trial = false;
    }

    fn failed(&self, error: &ActError) {
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

    fn backoff(&self, attempt: u32) -> Duration {
        let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
        self.retry
            .backoff
            .saturating_mul(factor)
            .min(self.retry.max_backoff)
    }
}

enum Admission {
    After(Duration),
    Refractory,
    Open,
    RateLimited,
}

/// The attempts after the first, or a delayed first one, on the motor side's runtime
async fn run(
    inner: Arc<Inner>,
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

impl Neuron for Effector {
    fn name(&self) -> &str {
        self.inner.act.name()
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Effector
    }

    fn stimulate(&self, signal: Signal, cx: &Cx<'_>) -> Firing {
        let inner = &self.inner;
        let _ = inner.runtime.set(cx.efferent().clone());
        let delay = match inner.admit(cx.now) {
            Admission::After(delay) => delay,
            Admission::Refractory | Admission::Open | Admission::RateLimited => {
                return Firing::Quiet;
            }
        };
        let command = match signal.payload.as_command() {
            Some(command) => command.clone(),
            // A plain signal: the payload is the argument
            None => Command::new(
                self.name(),
                signal.payload.to_json().unwrap_or(serde_json::Value::Null),
            ),
        };
        let started = cx.now;
        let spawn = |pending: Option<ActFuture>, attempt: u32, delay: Duration| {
            cx.efferent().spawn(run(
                self.inner.clone(),
                command.clone(),
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

// ---------------------------------------------------------------------------------------------
// Respond

/// The HTTP response to the request that started the chain. The command's `args` is
/// `{"status": 200, "content_type": "...", "body": ...}` (a string body as it is, anything
/// else as JSON); a plain signal is answered with its payload.
pub struct Respond {
    name: String,
}

impl Respond {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl Act for Respond {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let answer = match &command.args {
            serde_json::Value::Object(args) => {
                let status = args
                    .get("status")
                    .and_then(|s| s.as_u64())
                    .and_then(|s| u16::try_from(s).ok())
                    .unwrap_or(200);
                match args.get("body") {
                    Some(serde_json::Value::String(text)) => Answer {
                        status,
                        content_type: args
                            .get("content_type")
                            .and_then(|c| c.as_str())
                            .unwrap_or("text/plain; charset=utf-8")
                            .to_string(),
                        body: Bytes::from(text.clone()),
                    },
                    Some(body) => Answer::json(status, body),
                    None => Answer::json(status, &command.args),
                }
            }
            serde_json::Value::String(text) => Answer::text(200, text),
            serde_json::Value::Null => match &signal.payload {
                Payload::Text(text) => Answer::text(200, text),
                Payload::Bytes(bytes) => {
                    Answer::bytes(200, "application/octet-stream", bytes.clone())
                }
                Payload::Json(value) => Answer::json(200, value),
                other => Answer::bytes(200, "application/octet-stream", other.to_bytes()),
            },
            other => Answer::json(200, other),
        };
        if signal.answer(answer) {
            Action::ok()
        } else {
            Action::Done(Err(ActError::fatal("nobody is waiting for a response")))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Webhook

struct WebhookShared {
    url: wire::Url,
    client: wire::Client,
    headers: Vec<(String, String)>,
}

/// An outbound HTTP call: `POST url` with the command's `args` as JSON, or the signal's
/// payload as it is (JSON, text or bytes) for a plain signal. 5xx, 429 and connection
/// errors are retried; other 4xx are final.
pub struct Webhook {
    name: String,
    shared: Arc<WebhookShared>,
}

impl Webhook {
    /// `url` is `http://host[:port]/path`
    pub fn new(name: &str, url: &str) -> Self {
        Self::with_timeout(name, url, Duration::from_secs(5))
    }

    pub fn with_timeout(name: &str, url: &str, timeout: Duration) -> Self {
        let url = wire::Url::parse(url).unwrap_or_else(|e| panic!("webhook '{name}': {e}"));
        Self {
            name: name.to_string(),
            shared: Arc::new(WebhookShared {
                url,
                client: wire::Client::new(timeout),
                headers: Vec::new(),
            }),
        }
    }

    /// Send this header with every call (an authorization token, say)
    pub fn header(mut self, name: &str, value: &str) -> Self {
        Arc::get_mut(&mut self.shared)
            .expect("configured before use")
            .headers
            .push((name.to_string(), value.to_string()));
        self
    }
}

impl Act for Webhook {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let (content_type, body) = if command.args.is_null() {
            match &signal.payload {
                Payload::Json(_) | Payload::Command(_) => {
                    ("application/json", signal.payload.to_bytes())
                }
                Payload::Text(_) => ("text/plain; charset=utf-8", signal.payload.to_bytes()),
                other => ("application/octet-stream", other.to_bytes()),
            }
        } else {
            ("application/json", Bytes::from(command.args.to_string()))
        };
        let shared = self.shared.clone();
        Action::pending(async move {
            let mut headers: Vec<(&str, &str)> = vec![("content-type", content_type)];
            headers.extend(shared.headers.iter().map(|(n, v)| (n.as_str(), v.as_str())));
            match shared
                .client
                .request("POST", &shared.url, &headers, &body)
                .await
            {
                Ok(response) if (200..300).contains(&response.status) => Ok(()),
                Ok(response) if response.status == 429 || response.status >= 500 => {
                    Err(ActError::retry(format!("status {}", response.status)))
                }
                Ok(response) => Err(ActError::fatal(format!("status {}", response.status))),
                Err(e) => Err(ActError::retry(e.to_string())),
            }
        })
    }

    fn prepare(&self, runtime: &tokio::runtime::Handle) {
        let shared = self.shared.clone();
        runtime.spawn(async move {
            if let Err(e) = shared.client.warm(&shared.url).await {
                debug!("nerve: webhook warm-up to {}: {e}", shared.url.host);
            }
        });
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "webhook": format!("http://{}:{}{}", self.shared.url.host, self.shared.url.port, self.shared.url.path),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Process

/// Run a local command. The program and its fixed arguments are set here; the command's
/// `args` adds `{"args": [...], "stdin": "..."}` (or is itself an array of arguments). A
/// non-zero exit is a final failure; a program that cannot be started is retried.
pub struct Process {
    name: String,
    program: String,
    base_args: Vec<String>,
    timeout: Duration,
    /// Feed the signal's payload on stdin when the command gives none
    stdin_payload: bool,
}

impl Process {
    pub fn new(name: &str, program: &str) -> Self {
        Self {
            name: name.to_string(),
            program: program.to_string(),
            base_args: Vec::new(),
            timeout: Duration::from_secs(30),
            stdin_payload: false,
        }
    }

    pub fn args(mut self, args: &[&str]) -> Self {
        self.base_args = args.iter().map(|a| a.to_string()).collect();
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Give the process the signal's payload on stdin
    pub fn stdin_payload(mut self) -> Self {
        self.stdin_payload = true;
        self
    }
}

impl Act for Process {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let strings = |v: &serde_json::Value| -> Vec<String> {
            v.as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| match x {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut args = self.base_args.clone();
        let mut stdin: Option<Bytes> = None;
        match &command.args {
            serde_json::Value::Array(_) => args.extend(strings(&command.args)),
            serde_json::Value::Object(object) => {
                if let Some(extra) = object.get("args") {
                    args.extend(strings(extra));
                }
                if let Some(serde_json::Value::String(input)) = object.get("stdin") {
                    stdin = Some(Bytes::from(input.clone()));
                }
            }
            _ => {}
        }
        if stdin.is_none() && self.stdin_payload {
            stdin = Some(signal.payload.to_bytes());
        }
        let program = self.program.clone();
        let timeout = self.timeout;
        Action::pending(async move {
            use tokio::io::AsyncWriteExt;
            let mut child = tokio::process::Command::new(&program)
                .args(&args)
                .stdin(if stdin.is_some() {
                    std::process::Stdio::piped()
                } else {
                    std::process::Stdio::null()
                })
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .map_err(|e| ActError::retry(format!("{program}: {e}")))?;
            if let (Some(input), Some(mut pipe)) = (stdin, child.stdin.take()) {
                let _ = pipe.write_all(&input).await;
            }
            match tokio::time::timeout(timeout, child.wait_with_output()).await {
                Ok(Ok(output)) if output.status.success() => Ok(()),
                Ok(Ok(output)) => Err(ActError::fatal(format!(
                    "{program} exited with {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr).trim()
                ))),
                Ok(Err(e)) => Err(ActError::retry(format!("{program}: {e}"))),
                Err(_) => Err(ActError::retry(format!("{program}: timed out"))),
            }
        })
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"process": self.program, "args": self.base_args})
    }
}

// ---------------------------------------------------------------------------------------------
// File append

/// Append a line per command to a file: `args.line` as it is if it is a string, else one
/// JSON object `{"at": ..., "signal": ..., "args": ...}`. The file is opened on the first
/// command and reopened after an error.
pub struct FileAppend {
    name: String,
    path: std::path::PathBuf,
    file: Mutex<Option<std::fs::File>>,
}

impl FileAppend {
    pub fn new(name: &str, path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            name: name.to_string(),
            path: path.into(),
            file: Mutex::new(None),
        }
    }
}

impl Act for FileAppend {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        use std::io::Write as _;
        let mut line = match command.args.get("line").and_then(|l| l.as_str()) {
            Some(line) => line.to_string(),
            None => serde_json::json!({
                "at": chrono::Utc::now().to_rfc3339(),
                "signal": signal.id.0,
                "args": command.args,
            })
            .to_string(),
        };
        line.push('\n');
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let result = (|| {
            if file.is_none() {
                *file = Some(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&self.path)?,
                );
            }
            file.as_mut()
                .expect("opened just now")
                .write_all(line.as_bytes())
        })();
        Action::Done(result.map_err(|e| {
            *file = None;
            ActError::retry(format!("{}: {e}", self.path.display()))
        }))
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"file": self.path.display().to_string()})
    }
}

// ---------------------------------------------------------------------------------------------
// In-process channel

/// What a [`Channel`] effector hands to your code
#[derive(Debug, Clone)]
pub struct Delivery {
    pub command: Command,
    pub signal: Signal,
}

/// An in-process channel: commands go to whatever code holds the receiver (a device loop, a
/// test). A full channel is retried; a dropped receiver is final.
pub struct Channel {
    name: String,
    sender: mpsc::SyncSender<Delivery>,
}

impl Channel {
    pub fn new(name: &str, capacity: usize) -> (Self, mpsc::Receiver<Delivery>) {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        (
            Self {
                name: name.to_string(),
                sender,
            },
            receiver,
        )
    }
}

impl Act for Channel {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        Action::Done(
            self.sender
                .try_send(Delivery {
                    command: command.clone(),
                    signal: signal.clone(),
                })
                .map_err(|e| match e {
                    mpsc::TrySendError::Full(_) => ActError::retry("channel full"),
                    mpsc::TrySendError::Disconnected(_) => ActError::fatal("receiver gone"),
                }),
        )
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"channel": true})
    }
}

// ---------------------------------------------------------------------------------------------
// Server-Sent Events

/// Push commands to subscribers as Server-Sent Events: `GET /nerve/effectors/{name}/events`
/// on the fabric's router, one `data:` event per command, as
/// `{"signal": id, "effector": name, "args": ...}`. Without subscribers, commands are
/// dropped silently (and counted as acted).
pub struct Sse {
    name: String,
    sender: tokio::sync::broadcast::Sender<Bytes>,
}

impl Sse {
    /// `capacity` events are kept for slow subscribers before they lose some
    pub fn new(name: &str, capacity: usize) -> Self {
        let (sender, _) = tokio::sync::broadcast::channel(capacity.max(1));
        Self {
            name: name.to_string(),
            sender,
        }
    }

    /// Subscribe in process (the route does this for HTTP clients)
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Bytes> {
        self.sender.subscribe()
    }

    pub fn subscribers(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Act for Sse {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let event = serde_json::json!({
            "signal": signal.id.0,
            "effector": command.effector,
            "args": command.args,
        });
        // No receivers is not a failure: nobody is listening right now
        let _ = self.sender.send(Bytes::from(event.to_string()));
        Action::ok()
    }

    fn events(&self) -> Option<tokio::sync::broadcast::Receiver<Bytes>> {
        Some(self.subscribe())
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"sse": true, "subscribers": self.subscribers()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::fabric::{Fabric, FnNeuron};
    use crate::nerve::neuron::Mechanisms;
    use crate::nerve::signal::{Feature, NeuronId, Reply};
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

    fn fabric_with(effector: Effector) -> (Fabric, NeuronId) {
        let mut b = Fabric::builder();
        let r = b.neuron(
            FnNeuron::reflex("in", |s, _| Firing::one(s)).with_kind(NeuronKind::Receptor),
            Mechanisms::NONE,
        );
        let e = b.neuron(effector, Mechanisms::NONE);
        b.reflex(r, e, 1.0);
        (b.build(), r)
    }

    fn command(effector: &str, args: serde_json::Value) -> Signal {
        Signal::new(Feature::NONE, Payload::None).command(effector, args)
    }

    fn wait_until(f: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !f() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
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

    #[test]
    fn respond_answers_the_waiting_request() {
        let effector = Effector::new(Respond::new("respond"));
        let (fabric, r) = fabric_with(effector.clone());
        let (reply, mut receiver) = Reply::new();
        let signal = Signal::new(Feature::NONE, Payload::text("payload")).with_reply(reply);
        fabric.stimulate(
            r,
            signal.command(
                "respond",
                serde_json::json!({"status": 201, "body": {"ok": true}}),
            ),
        );
        let answer = receiver.try_recv().unwrap();
        assert_eq!(answer.status, 201);
        assert_eq!(answer.content_type, "application/json");
        assert_eq!(answer.body, Bytes::from(r#"{"ok":true}"#));
        // Nobody waiting: a final failure
        fabric.stimulate(r, command("respond", serde_json::Value::Null));
        assert_eq!(effector.metrics().failed, 1);
        // A plain signal answers with its payload
        let (reply, mut receiver) = Reply::new();
        fabric.stimulate(
            r,
            Signal::new(Feature::NONE, Payload::text("plain")).with_reply(reply),
        );
        assert_eq!(receiver.try_recv().unwrap().body, Bytes::from("plain"));
    }

    #[tokio::test]
    async fn webhook_posts_json_and_retries_5xx() {
        use axum::Router;
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::routing::post;
        type Hook = (Arc<Mutex<Vec<String>>>, Arc<AtomicUsize>);
        let hits = Arc::new(Mutex::new(Vec::<String>::new()));
        let fails = Arc::new(AtomicUsize::new(1));
        let app = Router::new()
            .route(
                "/hook",
                post(
                    |State((hits, fails)): State<Hook>, body: String| async move {
                        if fails
                            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                            .is_ok()
                        {
                            return StatusCode::SERVICE_UNAVAILABLE;
                        }
                        hits.lock().unwrap().push(body);
                        StatusCode::NO_CONTENT
                    },
                ),
            )
            .with_state((hits.clone(), fails));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let effector = Effector::new(
            Webhook::new("hook", &format!("http://127.0.0.1:{port}/hook")).header("x-token", "t"),
        )
        .retry(Retry {
            attempts: 3,
            backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(5),
        });
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("hook", serde_json::json!({"level": 3})));
        tokio::task::spawn_blocking({
            let effector = effector.clone();
            move || wait_until(|| effector.metrics().acted == 1)
        })
        .await
        .unwrap();
        assert_eq!(effector.metrics().retries, 1);
        assert_eq!(hits.lock().unwrap().as_slice(), [r#"{"level":3}"#]);
        drop(fabric);
    }

    #[tokio::test]
    async fn process_runs_a_command_and_reports_failures() {
        let (program, ok_args, bad_args): (&str, &[&str], &[&str]) = if cfg!(windows) {
            ("cmd", &["/C", "exit", "0"], &["/C", "exit", "3"])
        } else {
            ("sh", &["-c", "exit 0"], &["-c", "exit 3"])
        };
        let effector = Effector::new(Process::new("cmd", program)).no_retry();
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("cmd", serde_json::json!({"args": ok_args})));
        fabric.stimulate(r, command("cmd", serde_json::json!(bad_args)));
        tokio::task::spawn_blocking({
            let effector = effector.clone();
            move || {
                wait_until(|| {
                    effector.metrics().inflight == 0
                        && effector.metrics().acted + effector.metrics().failed == 2
                })
            }
        })
        .await
        .unwrap();
        let m = effector.metrics();
        assert_eq!((m.acted, m.failed), (1, 1));
        drop(fabric);
    }

    #[test]
    fn file_append_writes_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.log");
        let effector = Effector::new(FileAppend::new("journal", &path));
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(
            r,
            command("journal", serde_json::json!({"line": "raw line"})),
        );
        fabric.stimulate(r, command("journal", serde_json::json!({"k": 1})));
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "raw line");
        let json: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(json["args"]["k"], 1);
        assert_eq!(effector.metrics().acted, 2);
    }

    #[test]
    fn sse_pushes_to_subscribers() {
        let sse = Sse::new("push", 16);
        let mut subscriber = sse.subscribe();
        let effector = Effector::new(sse);
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("push", serde_json::json!({"go": true})));
        let event = subscriber.try_recv().unwrap();
        let json: serde_json::Value = serde_json::from_slice(&event).unwrap();
        assert_eq!(json["effector"], "push");
        assert_eq!(json["args"]["go"], true);
        assert!(effector.act().events().is_some());
    }
}
