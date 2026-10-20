//! What moves through the fabric: one [`Signal`] type for text, bytes, tensors and commands.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::tensor::HostTensor;

/// A neuron's index in its [`Fabric`](super::Fabric)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NeuronId(pub(crate) u32);

impl NeuronId {
    /// A signal that no neuron has stamped yet
    pub(crate) const NONE: Self = Self(u32::MAX);

    /// The index as a `usize`, for slices of neurons
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NeuronId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// A pathway's index in its fabric
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct PathwayId(pub(crate) u32);

impl PathwayId {
    /// The index as a `usize`, for slices of pathways
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Unique per process, in order of creation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SignalId(pub u64);

static NEXT_SIGNAL: AtomicU64 = AtomicU64::new(1);

impl SignalId {
    fn next() -> Self {
        Self(NEXT_SIGNAL.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a receptor extracted from its input: small, fixed-size, cheap to compare. Habituation
/// tables and thresholds look at this and never at the payload.
///
/// The `sketch` is a 64-bit summary (a hash of the bytes for exact repeats, an embedding
/// sketch for near repeats, a quantized rate for a stream); `kind` tells sketches of different
/// origins apart, so the same number from a text line and from an image frame are not one
/// feature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Feature {
    /// Tells sketches of different origins apart (1 for [`Feature::sketch`])
    pub kind: u16,
    /// The 64-bit summary itself
    pub sketch: u64,
}

impl Feature {
    /// No feature: never habituates, since every such signal looks the same and a table entry
    /// for it would attenuate everything
    pub const NONE: Self = Self { kind: 0, sketch: 0 };

    /// A feature from a kind and a sketch of your own
    pub const fn new(kind: u16, sketch: u64) -> Self {
        Self { kind, sketch }
    }

    /// A 64-bit sketch of `bytes` (kind 1): equal inputs give equal features. About 1 ns per
    /// 8 bytes.
    pub fn sketch(bytes: &[u8]) -> Self {
        Self::new(1, sketch64(bytes))
    }

    /// [`Feature::sketch`] with a `kind` of your own
    pub fn sketch_as(kind: u16, bytes: &[u8]) -> Self {
        Self::new(kind, sketch64(bytes))
    }

    /// The same sketch under another `kind`
    pub fn with_kind(self, kind: u16) -> Self {
        Self { kind, ..self }
    }

    /// Whether this is [`Feature::NONE`]
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

/// A fast, non-cryptographic 64-bit hash (multiply-xorshift over 8-byte words). The sketch
/// has to be cheap: it is computed once per input at the receptor and then compared in
/// nanoseconds by every habituation table on the way.
pub(crate) fn sketch64(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = (bytes.len() as u64).wrapping_mul(K) ^ 0x2545_F491_4F6C_DD1D;
    let (chunks, rest) = bytes.as_chunks::<8>();
    for chunk in chunks {
        h = (h ^ u64::from_le_bytes(*chunk)).wrapping_mul(K);
        h ^= h >> 29;
    }
    if !rest.is_empty() {
        let mut word = [0u8; 8];
        word[..rest.len()].copy_from_slice(rest);
        h = (h ^ u64::from_le_bytes(word)).wrapping_mul(K);
        h ^= h >> 29;
    }
    h = (h ^ (h >> 32)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^ (h >> 31)
}

/// How fast a signal is "approaching": the time by which an action is useless, and the rate
/// of change the receptor saw. Pathway selection reads this and nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Urgency {
    /// After this, no action is worth taking; `None` is "whenever"
    pub deadline: Option<Instant>,
    /// Signals per second at the receptor, or whatever rate of change it measures; 0 when
    /// unknown. Informational for pathways; receptors may derive the deadline from it.
    pub rate: f32,
}

impl Urgency {
    /// No deadline and no known rate
    pub const NONE: Self = Self {
        deadline: None,
        rate: 0.0,
    };

    /// An action is wanted within `d` from now
    pub fn within(d: Duration) -> Self {
        Self {
            deadline: Some(Instant::now() + d),
            rate: 0.0,
        }
    }

    /// An action is wanted by `deadline`
    pub fn at(deadline: Instant) -> Self {
        Self {
            deadline: Some(deadline),
            rate: 0.0,
        }
    }

    /// The same urgency with this rate
    pub fn with_rate(self, rate: f32) -> Self {
        Self { rate, ..self }
    }

    /// Time left before the deadline; `None` without one, zero once it has passed
    pub fn remaining(&self, now: Instant) -> Option<Duration> {
        self.deadline.map(|d| d.saturating_duration_since(now))
    }
}

/// An effector and its arguments: what a brain or a reflex tells the motor side to do
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Command {
    /// The name an effector was registered under
    pub effector: String,
    /// Whatever that effector takes: a JSON body for a webhook, `args` for a process, a line
    /// for a journal
    #[serde(default)]
    pub args: serde_json::Value,
}

impl Command {
    /// A command for the effector registered as `effector`
    pub fn new(effector: &str, args: serde_json::Value) -> Self {
        Self {
            effector: effector.to_string(),
            args,
        }
    }
}

/// The raw input, held by shared handle and never copied between neurons. A reflex pathway
/// usually never looks at it.
#[derive(Clone, Debug, Default)]
pub enum Payload {
    /// Nothing: the signal is its feature alone
    #[default]
    None,
    /// Raw bytes
    Bytes(Bytes),
    /// UTF-8 text
    Text(Arc<str>),
    /// A JSON value
    Json(Arc<serde_json::Value>),
    /// Tensors, for a model
    Tensors(Arc<Vec<HostTensor>>),
    /// A command for an effector
    Command(Arc<Command>),
    /// A request for an LLM to generate (feature `llm`): what the OpenAI routes send the
    /// brain
    #[cfg(feature = "llm")]
    Prompt(Arc<crate::llm::GenerateRequest>),
}

impl Payload {
    /// A text payload
    pub fn text(s: &str) -> Self {
        Self::Text(Arc::from(s))
    }

    /// A bytes payload
    pub fn bytes(b: impl Into<Bytes>) -> Self {
        Self::Bytes(b.into())
    }

    /// A JSON payload
    pub fn json(v: serde_json::Value) -> Self {
        Self::Json(Arc::new(v))
    }

    /// A command payload
    pub fn command(c: Command) -> Self {
        Self::Command(Arc::new(c))
    }

    /// A tensors payload
    pub fn tensors(t: Vec<HostTensor>) -> Self {
        Self::Tensors(Arc::new(t))
    }

    /// The text, for `Text` and for `Bytes` that are UTF-8
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Bytes(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// The bytes, for `Bytes` and for `Text` as UTF-8
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(b) => Some(b),
            Self::Text(s) => Some(s.as_bytes()),
            _ => None,
        }
    }

    /// The command, for `Command`
    pub fn as_command(&self) -> Option<&Command> {
        match self {
            Self::Command(c) => Some(c),
            _ => None,
        }
    }

    /// Whether this payload is a command for `effector`
    pub fn commands(&self, effector: &str) -> bool {
        self.as_command().is_some_and(|c| c.effector == effector)
    }

    /// The tensors, for `Tensors`
    pub fn as_tensors(&self) -> Option<&[HostTensor]> {
        match self {
            Self::Tensors(t) => Some(t),
            _ => None,
        }
    }

    /// The JSON value, parsing `Text` and `Bytes` on demand; a `Command` as its JSON object
    pub fn to_json(&self) -> Option<serde_json::Value> {
        match self {
            Self::Json(v) => Some((**v).clone()),
            Self::Text(_) | Self::Bytes(_) => serde_json::from_slice(self.as_bytes()?).ok(),
            Self::Command(c) => serde_json::to_value(&**c).ok(),
            Self::None | Self::Tensors(_) => None,
            #[cfg(feature = "llm")]
            Self::Prompt(_) => None,
        }
    }

    /// Bytes on the wire, for effectors that send the payload as it is: text and bytes as
    /// they are, JSON and commands serialized, tensors as little-endian values
    pub fn to_bytes(&self) -> Bytes {
        match self {
            Self::None => Bytes::new(),
            Self::Bytes(b) => b.clone(),
            Self::Text(s) => Bytes::copy_from_slice(s.as_bytes()),
            Self::Json(v) => Bytes::from(v.to_string()),
            Self::Command(c) => Bytes::from(serde_json::to_vec(&**c).unwrap_or_default()),
            Self::Tensors(t) => Bytes::from(
                t.iter()
                    .flat_map(|t| match &t.values {
                        crate::tensor::Values::F32(v) => {
                            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
                        }
                        crate::tensor::Values::I64(v) => {
                            v.iter().flat_map(|x| x.to_le_bytes()).collect()
                        }
                    })
                    .collect::<Vec<u8>>(),
            ),
            #[cfg(feature = "llm")]
            Self::Prompt(_) => Bytes::new(),
        }
    }

    /// The variant's name, lowercase (`"text"`, `"command"`, ...)
    pub fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bytes(_) => "bytes",
            Self::Text(_) => "text",
            Self::Json(_) => "json",
            Self::Tensors(_) => "tensors",
            Self::Command(_) => "command",
            #[cfg(feature = "llm")]
            Self::Prompt(_) => "prompt",
        }
    }
}

/// Most hops a trace records inline; longer chains count the rest
pub(crate) const TRACE_HOPS: usize = 8;

/// Where a signal has been: the neurons that fired it on, in order, without an allocation
#[derive(Clone, Copy, Debug)]
pub struct Trace {
    hops: [NeuronId; TRACE_HOPS],
    len: u8,
    /// Hops beyond the first [`TRACE_HOPS`]
    overflow: u16,
    brain: bool,
    /// Whether the signal went through a deliberative queue (then it is not a reflex arc)
    queued: bool,
}

impl Default for Trace {
    fn default() -> Self {
        Self {
            hops: [NeuronId::NONE; TRACE_HOPS],
            len: 0,
            overflow: 0,
            brain: false,
            queued: false,
        }
    }
}

impl Trace {
    pub(crate) fn push(&mut self, id: NeuronId) {
        if (self.len as usize) < TRACE_HOPS {
            self.hops[self.len as usize] = id;
            self.len += 1;
        } else {
            self.overflow = self.overflow.saturating_add(1);
        }
    }

    pub(crate) fn mark_brain(&mut self) {
        self.brain = true;
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued = true;
    }

    /// The neurons passed so far (the first 8)
    pub fn hops(&self) -> &[NeuronId] {
        &self.hops[..self.len as usize]
    }

    /// Neurons passed in all
    pub fn len(&self) -> usize {
        self.len as usize + self.overflow as usize
    }

    /// Whether no neuron has been passed yet
    pub fn is_empty(&self) -> bool {
        self.len == 0 && self.overflow == 0
    }

    /// Whether a brain neuron saw this signal (or the one it was derived from)
    pub fn reached_brain(&self) -> bool {
        self.brain
    }

    /// Whether the signal waited in a deliberative queue
    pub fn was_queued(&self) -> bool {
        self.queued
    }

    /// The last neuron recorded in [`Trace::hops`]
    pub fn last(&self) -> Option<NeuronId> {
        self.hops().last().copied()
    }
}

/// What a chain sends back to whoever started it, when that was a request
#[derive(Debug)]
pub enum Answer {
    /// An HTTP response, as the [`Respond`](super::effector::Respond) effector makes it for
    /// the HTTP receptor
    Http {
        /// HTTP status code
        status: u16,
        /// The `Content-Type` header
        content_type: String,
        /// The response body
        body: Bytes,
    },
    /// Tensors, for a protocol that encodes them itself (the server's `/predict`, the
    /// Open Inference Protocol over HTTP and gRPC)
    Tensors(Vec<HostTensor>),
    /// The chain failed and says why: the request's own error
    Failed(Box<crate::error::FurnaceError>),
    /// An LLM's generation (feature `llm`): its events, which go straight from the engine to
    /// whoever asked, or why the engine refused it
    #[cfg(feature = "llm")]
    Generation(
        std::result::Result<
            tokio::sync::mpsc::Receiver<crate::llm::Event>,
            crate::llm::EngineError,
        >,
    ),
}

impl Answer {
    /// A `text/plain` response
    pub fn text(status: u16, body: &str) -> Self {
        Self::Http {
            status,
            content_type: "text/plain; charset=utf-8".to_string(),
            body: Bytes::copy_from_slice(body.as_bytes()),
        }
    }

    /// An `application/json` response
    pub fn json(status: u16, value: &serde_json::Value) -> Self {
        Self::Http {
            status,
            content_type: "application/json".to_string(),
            body: Bytes::from(value.to_string()),
        }
    }

    /// A response with a content type of your own
    pub fn bytes(status: u16, content_type: &str, body: Bytes) -> Self {
        Self::Http {
            status,
            content_type: content_type.to_string(),
            body,
        }
    }

    /// The chain failed with `error`
    pub fn failed(error: crate::error::FurnaceError) -> Self {
        Self::Failed(Box::new(error))
    }
}

struct ReplyInner {
    sender: Mutex<Option<oneshot::Sender<Answer>>>,
    /// Whether the chain exists to answer: a signal whose requester stopped waiting is then
    /// worth nothing, and a deliberative pathway skips it
    required: bool,
}

/// The way back to the request that started a chain, carried by the signal and every signal
/// derived from it. The first effector to answer wins; a chain that ends without answering
/// drops the sender, and the receptor sees that.
#[derive(Clone)]
pub struct ReplyHandle(Arc<ReplyInner>);

impl fmt::Debug for ReplyHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplyHandle")
            .field("open", &self.is_open())
            .field("required", &self.0.required)
            .finish()
    }
}

impl ReplyHandle {
    /// A way back that the chain may use: the chain still runs if nobody waits any more (a
    /// request answered `202 Accepted` before its signal reached the brain)
    pub fn new() -> (Self, oneshot::Receiver<Answer>) {
        Self::make(false)
    }

    /// A way back the chain exists for: once nobody waits, the signal is dropped before it
    /// reaches a deliberative pathway's neuron
    pub fn required() -> (Self, oneshot::Receiver<Answer>) {
        Self::make(true)
    }

    fn make(required: bool) -> (Self, oneshot::Receiver<Answer>) {
        let (sender, receiver) = oneshot::channel();
        let inner = ReplyInner {
            sender: Mutex::new(Some(sender)),
            required,
        };
        (Self(Arc::new(inner)), receiver)
    }

    /// Send the answer; `false` if one was sent already or nobody is waiting
    pub fn answer(&self, answer: Answer) -> bool {
        let sender = self
            .0
            .sender
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        sender.is_some_and(|s| s.send(answer).is_ok())
    }

    /// Whether the request still waits and no answer was sent yet
    pub fn is_open(&self) -> bool {
        self.0
            .sender
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|s| !s.is_closed())
    }

    /// Whether the signal carrying this is still worth running: always, unless the chain
    /// exists to answer and nobody waits any more
    pub fn wanted(&self) -> bool {
        !self.0.required || self.is_open()
    }
}

/// One thing that moves: a feature, an urgency, an intensity and a lazily shared payload.
///
/// Receptors make signals ([`Signal::new`]); neurons that fire make derived ones
/// ([`Signal::derive`]) that keep the id, origin, timing and trace, so a command that leaves
/// an effector can be traced back to the stimulus that caused it.
#[derive(Clone, Debug)]
pub struct Signal {
    /// Shared by every signal derived from the same stimulus
    pub id: SignalId,
    /// The receptor that made it
    pub origin: NeuronId,
    /// When the receptor saw the stimulus: stimulus-to-action latencies count from here
    pub at: Instant,
    /// When an action is due; pathway selection reads it
    pub urgency: Urgency,
    /// What the receptor extracted; habituation and thresholds look at it
    pub feature: Feature,
    /// Strength after attenuation, 0..=1. Below a neuron's threshold, nothing fires.
    pub intensity: f32,
    /// The raw input, shared and never copied
    pub payload: Payload,
    /// The neurons it has passed through
    pub trace: Trace,
    /// The way back to the request that started the chain, if any
    pub reply: Option<ReplyHandle>,
}

impl Signal {
    /// A signal of full intensity and no urgency, seen now
    pub fn new(feature: Feature, payload: Payload) -> Self {
        Self {
            id: SignalId::next(),
            origin: NeuronId::NONE,
            at: Instant::now(),
            urgency: Urgency::NONE,
            feature,
            intensity: 1.0,
            payload,
            trace: Trace::default(),
            reply: None,
        }
    }

    /// The same signal with this urgency
    pub fn with_urgency(mut self, urgency: Urgency) -> Self {
        self.urgency = urgency;
        self
    }

    /// The same signal with this intensity, 0..=1
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// The same signal with this feature
    pub fn with_feature(mut self, feature: Feature) -> Self {
        self.feature = feature;
        self
    }

    /// The same signal carrying a way back to the request
    pub fn with_reply(mut self, reply: ReplyHandle) -> Self {
        self.reply = Some(reply);
        self
    }

    /// A signal this one causes: same id, origin, timing, urgency, trace and reply, with a
    /// new payload. The feature and intensity carry over; change them with the builders.
    pub fn derive(&self, payload: Payload) -> Self {
        Self {
            id: self.id,
            origin: self.origin,
            at: self.at,
            urgency: self.urgency,
            feature: self.feature,
            intensity: self.intensity,
            payload,
            trace: self.trace,
            reply: self.reply.clone(),
        }
    }

    /// [`Signal::derive`] with a [`Command`] payload: tell the effector named in it what to do
    pub fn command(&self, effector: &str, args: serde_json::Value) -> Self {
        self.derive(Payload::command(Command::new(effector, args)))
    }

    /// Time since the receptor saw the stimulus
    pub fn age(&self) -> Duration {
        self.at.elapsed()
    }

    /// Answer the request that started the chain, if there is one waiting
    pub fn answer(&self, answer: Answer) -> bool {
        self.reply.as_ref().is_some_and(|r| r.answer(answer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sketches_tell_inputs_apart_and_repeat_for_equal_ones() {
        assert_eq!(Feature::sketch(b"hello"), Feature::sketch(b"hello"));
        assert_ne!(Feature::sketch(b"hello"), Feature::sketch(b"hellp"));
        assert_ne!(Feature::sketch(b""), Feature::sketch(b"\0"));
        assert_ne!(Feature::sketch(b"12345678"), Feature::sketch(b"123456789"));
        assert_ne!(Feature::sketch(b"a"), Feature::sketch_as(2, b"a"));
        assert!(!Feature::sketch(b"").is_none());
    }

    #[test]
    fn trace_records_eight_hops_inline_and_counts_the_rest() {
        let mut trace = Trace::default();
        for i in 0..12 {
            trace.push(NeuronId(i));
        }
        assert_eq!(trace.hops().len(), TRACE_HOPS);
        assert_eq!(trace.len(), 12);
        assert_eq!(trace.last(), Some(NeuronId(7)));
        assert!(!trace.reached_brain());
    }

    #[test]
    fn derived_signals_share_identity_and_reply() {
        let (reply, mut receiver) = ReplyHandle::new();
        let signal = Signal::new(Feature::sketch(b"x"), Payload::text("x"))
            .with_reply(reply)
            .with_urgency(Urgency::within(Duration::from_secs(1)));
        let child = signal.command("log", serde_json::json!({"line": "x"}));
        assert_eq!(child.id, signal.id);
        assert_eq!(child.payload.as_command().unwrap().effector, "log");
        assert!(child.urgency.remaining(Instant::now()).is_some());
        assert!(child.answer(Answer::text(200, "ok")));
        // Only the first answer counts
        assert!(!signal.answer(Answer::text(200, "again")));
        let Answer::Http { body, .. } = receiver.try_recv().unwrap() else {
            panic!("an HTTP answer");
        };
        assert_eq!(body, Bytes::from_static(b"ok"));
    }

    #[test]
    fn only_a_required_reply_makes_its_signal_unwanted_once_abandoned() {
        let (optional, receiver) = ReplyHandle::new();
        drop(receiver);
        assert!(!optional.is_open());
        assert!(optional.wanted());
        let (required, receiver) = ReplyHandle::required();
        assert!(required.wanted());
        drop(receiver);
        assert!(!required.wanted());
    }

    #[test]
    fn payload_conversions() {
        let json = Payload::text(r#"{"a":1}"#).to_json().unwrap();
        assert_eq!(json["a"], 1);
        assert_eq!(
            Payload::json(json.clone()).to_bytes(),
            Bytes::from(r#"{"a":1}"#)
        );
        let command = Payload::command(Command::new("w", serde_json::json!({"k": "v"})));
        assert_eq!(command.to_json().unwrap()["effector"], "w");
        assert!(Payload::None.as_text().is_none());
        assert_eq!(Payload::bytes(&b"abc"[..]).as_text(), Some("abc"));
    }
}
