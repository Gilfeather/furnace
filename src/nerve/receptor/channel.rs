//! An in-process stream, fed from your own code

use std::sync::{Arc, Mutex, mpsc};

use super::{Afferent, Receptor, Source, UrgencyRule};
use crate::nerve::Outcome;
use crate::nerve::neuron::{Cx, Firing, Neuron, NeuronKind};
use crate::nerve::signal::{Feature, Payload, Signal};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::testing::{fabric_with_receptor, wait_until};

    #[test]
    fn channel_receptor_emits_inline_and_through_its_thread() {
        let (channel, sender) = Channel::new("stream", 16);
        let (_fabric, r, seen) = fabric_with_receptor(channel);
        let outcome = sender
            .emit(sender.signal(Feature::sketch(b"a"), Payload::text("a")))
            .unwrap();
        assert_eq!(outcome.acted, 1);
        assert_eq!(seen.lock().unwrap().len(), 1);
        sender
            .send(sender.signal(Feature::sketch(b"b"), Payload::text("b")))
            .unwrap();
        wait_until(|| seen.lock().unwrap().len() >= 2);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[1].origin, r);
        assert_eq!(seen[1].payload.as_text(), Some("b"));
    }
}
