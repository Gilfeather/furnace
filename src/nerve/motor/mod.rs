//! The motor side's hierarchy: motor neurons, which take a command and carry it out by
//! commanding the neurons below them.
//!
//! A brain says "get dressed", not "bend the left elbow by 20 degrees". A [`Motor`] neuron
//! takes commands addressed to its name, as an effector does, and runs a program for each on
//! the motor side's runtime. The program gets a [`Motion`] and breaks the command down:
//!
//! - **Commands below**, to other motor neurons or to effectors, by name, each answering with
//!   its outcome: the effector's result (a value it measured, or why it failed after its own
//!   retries), or the program's below. One after the other is `.await` in turn;
//!   [`Motion::together`] sends several at once.
//! - **Senses**: [`Motion::sense`] receives every signal that passes a neuron (a receptor, a
//!   reflex) from then on, for waiting until the eye sees the coat or for adjusting a grip by
//!   the pressure a fingertip reports: a local loop that never goes through the brain.
//! - **Questions**: [`Motion::ask`] fires a small question along the motor neuron's own
//!   pathways, chosen by the question's deadline and their measured latency as any signal's,
//!   and waits for what the neuron there fires in answer.
//! - **Deadlines**: the command's deadline bounds the program and everything it commands;
//!   [`Motion::within`] bounds one step.
//!
//! What the program returns goes back up: to the program that commanded it, or, when a brain
//! or a reflex sent the command, as a signal fired along the motor neuron's pathways (wire one
//! back to the brain and it hears `{"action", "status", ...}`).
//!
//! One program runs per motor neuron: a new command preempts the one running, as a new
//! descending command overrides the last; [`Fabric::stop`] stops it. Stopping a program stops
//! everything it commanded and still waits for, all the way down: a motor neuron per body
//! part, not per command.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::Poll;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::debug;

use super::fabric::{Fabric, Inner};
use super::neuron::{Cx, Firing, Neuron, NeuronKind};
use super::signal::{Command, Feedback, NeuronId, Payload, Reported, Signal};

/// Why a command did not come to a good end
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MotionError {
    /// It was tried and did not work: an act that failed after its retries, a command quiet
    /// at a gate, a program that gave up
    Failed(String),
    /// No effector or motor neuron has this name
    Unrouted(String),
    /// The deadline passed first
    TimedOut,
    /// Stopped: by a newer command to the same motor neuron, by [`Fabric::stop`], or because
    /// whoever commanded it stopped waiting
    Cancelled,
}

impl MotionError {
    /// A failure with this reason ([`Failed`](Self::Failed))
    pub fn failed(reason: impl Into<String>) -> Self {
        Self::Failed(reason.into())
    }

    /// The status a report up to a brain carries: `failed`, `timed_out` or `cancelled`
    fn status(&self) -> &'static str {
        match self {
            Self::Failed(_) | Self::Unrouted(_) => "failed",
            Self::TimedOut => "timed_out",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for MotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(reason) => f.write_str(reason),
            Self::Unrouted(name) => write!(f, "no effector or motor neuron named '{name}'"),
            Self::TimedOut => f.write_str("the deadline passed"),
            Self::Cancelled => f.write_str("stopped"),
        }
    }
}

impl std::error::Error for MotionError {}

/// An action a brain may call: a motor neuron or an effector, what it does and the shape of
/// its arguments. [`FabricBuilder::action`](super::FabricBuilder::action) declares one; a
/// brain's prompt lists them ([`Cx::actions`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionSpec {
    /// The name commands address
    pub name: String,
    /// What it does, for the brain
    pub description: String,
    /// The shape of its arguments, for the brain: a JSON Schema, or an example
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub args: serde_json::Value,
}

type MotionFuture = Pin<Box<dyn Future<Output = Reported> + Send + 'static>>;
type Program = dyn Fn(Motion) -> MotionFuture + Send + Sync;

/// The program running at a motor neuron, and the way to stop it: dropping `stop` does
struct Running {
    generation: u64,
    _stop: oneshot::Sender<()>,
}

#[derive(Default)]
struct MotorCounters {
    started: AtomicU64,
    done: AtomicU64,
    failed: AtomicU64,
    timed_out: AtomicU64,
    cancelled: AtomicU64,
}

struct MotorInner {
    name: Arc<str>,
    program: Box<Program>,
    running: Mutex<Option<Running>>,
    generations: AtomicU64,
    counters: MotorCounters,
}

impl MotorInner {
    fn running(&self) -> std::sync::MutexGuard<'_, Option<Running>> {
        self.running.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A motor neuron: commands addressed to its name start its program, which carries them out
/// by commanding the motor neurons and effectors below it (see the [module](self)). Add one
/// with [`FabricBuilder::motor`](super::FabricBuilder::motor).
pub struct Motor {
    inner: Arc<MotorInner>,
}

impl Motor {
    /// A motor neuron named `name` whose program is `program`: an async closure that takes
    /// the [`Motion`] and returns the outcome, a JSON value (`Value::Null` for nothing to
    /// say) or why it failed
    pub fn new<F, Fut>(name: &str, program: F) -> Self
    where
        F: Fn(Motion) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, MotionError>> + Send + 'static,
    {
        Self {
            inner: Arc::new(MotorInner {
                name: Arc::from(name),
                program: Box::new(move |motion| Box::pin(program(motion))),
                running: Mutex::new(None),
                generations: AtomicU64::new(0),
                counters: MotorCounters::default(),
            }),
        }
    }
}

impl Neuron for Motor {
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Motor
    }

    fn stimulate(&self, mut signal: Signal, cx: &Cx<'_>) -> Firing {
        let inner = &self.inner;
        // The commander's way to hear back; without one, the outcome goes up the pathways
        let outcome = signal.feedback.take().and_then(|f| f.take_outcome());
        let args = match signal.payload.as_command() {
            Some(command) => command.args.clone(),
            None => signal.payload.to_json().unwrap_or_default(),
        };
        let (stop, stopped) = oneshot::channel();
        let generation = inner.generations.fetch_add(1, Ordering::Relaxed);
        // The program running is preempted: its stop is dropped here
        let preempted = inner.running().replace(Running {
            generation,
            _stop: stop,
        });
        drop(preempted);
        inner.counters.started.fetch_add(1, Ordering::Relaxed);
        let deadline = signal.urgency.deadline;
        let upward = outcome.is_none().then(|| signal.clone());
        let program = (inner.program)(Motion {
            fabric: Arc::downgrade(&cx.fabric().0),
            id: cx.id,
            name: inner.name.clone(),
            args,
            signal,
        });
        let started = Started {
            inner: inner.clone(),
            fabric: Arc::downgrade(&cx.fabric().0),
            id: cx.id,
            generation,
            stopped,
            outcome,
            deadline,
            upward,
        };
        cx.efferent().spawn(started.run(program));
        Firing::Fire(Vec::new())
    }

    fn stop(&self) -> bool {
        self.inner.running().take().is_some()
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"running": self.inner.running().is_some()})
    }

    fn metrics(&self) -> serde_json::Value {
        let c = &self.inner.counters;
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "started": load(&c.started),
            "done": load(&c.done),
            "failed": load(&c.failed),
            "timed_out": load(&c.timed_out),
            "cancelled": load(&c.cancelled),
            "running": u8::from(self.inner.running().is_some()),
        })
    }
}

/// A program just started at a motor neuron, and what decides how it ends
struct Started {
    inner: Arc<MotorInner>,
    fabric: Weak<Inner>,
    id: NeuronId,
    generation: u64,
    /// Resolves once the program is preempted or stopped
    stopped: oneshot::Receiver<()>,
    /// The commander's way to hear back, when a motor program sent the command
    outcome: Option<oneshot::Sender<Reported>>,
    deadline: Option<Instant>,
    /// The command, when nobody waits for the outcome: it goes up the pathways
    upward: Option<Signal>,
}

impl Started {
    /// Run the program on the motor side's runtime until it ends, it is stopped, its
    /// commander stops waiting, or the deadline passes; then send the outcome up
    async fn run(mut self, program: MotionFuture) {
        let result = {
            let outcome = &mut self.outcome;
            let abandoned = async {
                match outcome.as_mut() {
                    Some(sender) => sender.closed().await,
                    None => std::future::pending().await,
                }
            };
            let deadline = self.deadline;
            let expired = async {
                match deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                    None => std::future::pending().await,
                }
            };
            // A program stopped, abandoned or out of time is not polled again
            tokio::select! {
                biased;
                _ = &mut self.stopped => Err(MotionError::Cancelled),
                () = abandoned => Err(MotionError::Cancelled),
                () = expired => Err(MotionError::TimedOut),
                result = program => result,
            }
        };
        let inner = &self.inner;
        {
            let mut running = inner.running();
            if running
                .as_ref()
                .is_some_and(|r| r.generation == self.generation)
            {
                *running = None;
            }
        }
        let counters = &inner.counters;
        let counter = match &result {
            Ok(_) => &counters.done,
            Err(MotionError::TimedOut) => &counters.timed_out,
            Err(MotionError::Cancelled) => &counters.cancelled,
            Err(_) => &counters.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        match &result {
            Ok(_) => debug!("nerve: motor '{}' done", inner.name),
            Err(e) => debug!("nerve: motor '{}' {}: {e}", inner.name, e.status()),
        }
        if let Some(sender) = self.outcome {
            // The commander may have stopped waiting
            let _ = sender.send(result);
        } else if let (Some(signal), Some(fabric)) = (self.upward, self.fabric.upgrade()) {
            let mut report = serde_json::json!({"action": &*inner.name});
            match result {
                Ok(value) => {
                    report["status"] = "done".into();
                    report["result"] = value;
                }
                Err(e) => {
                    report["status"] = e.status().into();
                    report["reason"] = e.to_string().into();
                }
            }
            Fabric(fabric).emit(self.id, signal.derive(Payload::json(report)));
        }
    }
}

/// A motor neuron's program at work: the command it carries out, and the ways to command,
/// sense and ask. Every method that waits ends with [`MotionError::Cancelled`] once the
/// fabric is gone.
pub struct Motion {
    fabric: Weak<Inner>,
    id: NeuronId,
    name: Arc<str>,
    args: serde_json::Value,
    signal: Signal,
}

impl Motion {
    /// The command's arguments (a plain signal's payload as JSON)
    pub fn args(&self) -> &serde_json::Value {
        &self.args
    }

    /// The signal that carried the command: its id, urgency, trace and payload
    pub fn signal(&self) -> &Signal {
        &self.signal
    }

    /// This motor neuron's name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// When the command is due; past it, the program is stopped with
    /// [`MotionError::TimedOut`]
    pub fn deadline(&self) -> Option<Instant> {
        self.signal.urgency.deadline
    }

    fn fabric(&self) -> Result<Fabric, MotionError> {
        self.fabric
            .upgrade()
            .map(Fabric)
            .ok_or(MotionError::Cancelled)
    }

    /// Send `to` (a motor neuron or an effector) a command at once; the outcome arrives on the
    /// receiver
    fn send(&self, to: &str, args: serde_json::Value) -> oneshot::Receiver<Reported> {
        let (feedback, receiver) = Feedback::outcome();
        let mut signal = self.signal.command(to, args);
        signal.feedback = Some(feedback);
        if let Ok(fabric) = self.fabric() {
            fabric.emit(self.id, signal);
        }
        receiver
    }

    /// Command `to`, a motor neuron or an effector, and wait for the outcome: what the
    /// effector's act returned, or what the motor neuron's program did. The command keeps
    /// this one's deadline.
    pub async fn command(
        &self,
        to: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, MotionError> {
        outcome(self.send(to, args).await)
    }

    /// Send all `commands` at once and wait for every outcome, in the commands' order. The
    /// first failure stops the others and is the result.
    pub async fn together(
        &self,
        commands: impl IntoIterator<Item = Command>,
    ) -> Result<Vec<serde_json::Value>, MotionError> {
        let mut pending: Vec<Option<oneshot::Receiver<Reported>>> = commands
            .into_iter()
            .map(|c| Some(self.send(&c.effector, c.args)))
            .collect();
        let mut values = vec![serde_json::Value::Null; pending.len()];
        // Returning drops the receivers still pending, which stops what they wait for
        std::future::poll_fn(|cx| {
            for (slot, value) in pending.iter_mut().zip(values.iter_mut()) {
                let Some(receiver) = slot else {
                    continue;
                };
                if let Poll::Ready(received) = Pin::new(receiver).poll(cx) {
                    *slot = None;
                    match outcome(received) {
                        Ok(v) => *value = v,
                        Err(e) => return Poll::Ready(Err(e)),
                    }
                }
            }
            if pending.iter().all(Option::is_none) {
                Poll::Ready(Ok(std::mem::take(&mut values)))
            } else {
                Poll::Pending
            }
        })
        .await
    }

    /// From now on, receive a copy of every signal that passes `neuron`'s gate (a receptor,
    /// a reflex): call it before the command whose effect it should see. The newest 64 wait
    /// to be read; a sense that falls behind misses signals, as a sense does.
    pub fn sense(&self, neuron: &str) -> Result<Sense, MotionError> {
        let fabric = self.fabric()?;
        let id = fabric
            .neuron(neuron)
            .ok_or_else(|| MotionError::Unrouted(neuron.to_string()))?;
        let (sender, receiver) = mpsc::channel(SENSE_CAPACITY);
        fabric.0.cells[id.index()].taps.add(sender);
        Ok(Sense { receiver })
    }

    /// Ask a question along this motor neuron's pathways (to a brain, a heavier model): the
    /// pathway is chosen as any signal's, by the deadline and the measured latencies, and
    /// whatever the neuron there fires in answer comes back here. No answer is an empty
    /// list. An error when no pathway leaves this neuron, none is fast enough for the
    /// deadline, or the chosen one's queue is full.
    pub async fn ask(&self, question: Payload) -> Result<Vec<Signal>, MotionError> {
        let (feedback, mut answers) = Feedback::answers(self.id);
        let mut signal = self.signal.derive(question);
        signal.feedback = Some(feedback);
        let outcome = {
            let fabric = self.fabric()?;
            if fabric.0.cells[self.id.index()].out.is_empty() {
                return Err(MotionError::failed(format!(
                    "'{}' has no pathway to ask along",
                    self.name
                )));
            }
            fabric.emit(self.id, signal)
        };
        if outcome.no_pathway > 0 {
            return Err(MotionError::failed(
                "no pathway is fast enough for the deadline",
            ));
        }
        if outcome.dropped > 0 {
            return Err(MotionError::failed("the pathway's queue is full"));
        }
        let mut received = Vec::new();
        while let Some(answer) = answers.recv().await {
            received.push(answer);
        }
        Ok(received)
    }

    /// `step`, or [`MotionError::TimedOut`] if it takes longer than `limit`
    pub async fn within<T>(
        &self,
        limit: Duration,
        step: impl Future<Output = Result<T, MotionError>>,
    ) -> Result<T, MotionError> {
        tokio::time::timeout(limit, step)
            .await
            .unwrap_or(Err(MotionError::TimedOut))
    }

    /// Stop the program running at the motor neuron named `action`, and everything it
    /// commanded; whether one was running
    pub fn stop(&self, action: &str) -> bool {
        self.fabric().is_ok_and(|f| f.stop(action))
    }
}

/// A command's outcome as its sender sees it: no report is done, with nothing to say
fn outcome(received: Result<Reported, oneshot::error::RecvError>) -> Reported {
    received.unwrap_or(Ok(serde_json::Value::Null))
}

/// Signals a [`Sense`] holds before it misses new ones
const SENSE_CAPACITY: usize = 64;

/// Copies of the signals that pass a neuron, for a motor program: see [`Motion::sense`]
pub struct Sense {
    receiver: mpsc::Receiver<Signal>,
}

impl Sense {
    /// The next signal; `None` once the fabric is gone
    pub async fn next(&mut self) -> Option<Signal> {
        self.receiver.recv().await
    }

    /// The next signal for which `f` holds; `None` once the fabric is gone
    pub async fn until(&mut self, mut f: impl FnMut(&Signal) -> bool) -> Option<Signal> {
        while let Some(signal) = self.next().await {
            if f(&signal) {
                return Some(signal);
            }
        }
        None
    }
}

/// Who senses a neuron: copies of the signals it admits go to them
#[derive(Default)]
pub(crate) struct Taps {
    /// How many senders there are, so that a neuron nobody senses pays one load
    count: AtomicU64,
    senders: Mutex<Vec<mpsc::Sender<Signal>>>,
}

impl Taps {
    fn add(&self, sender: mpsc::Sender<Signal>) {
        let mut senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        senders.push(sender);
        self.count.store(senders.len() as u64, Ordering::Relaxed);
    }

    /// A copy of `signal`, without its ways back, to everyone sensing; the senses that are
    /// gone are forgotten
    pub(crate) fn copy(&self, signal: &Signal) {
        if self.count.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        senders.retain(|s| {
            !matches!(
                s.try_send(signal.observed()),
                Err(mpsc::error::TrySendError::Closed(_))
            )
        });
        self.count.store(senders.len() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;
