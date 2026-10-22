//! An in-process channel, for your own code

use std::sync::mpsc;

use super::{Act, ActError, Action};
use crate::nerve::signal::{Command, Signal};

/// What a [`ChannelOut`] effector hands to your code
#[derive(Debug, Clone)]
pub struct Delivery {
    /// What to do
    pub command: Command,
    /// The signal that carried it
    pub signal: Signal,
}

/// An in-process channel: commands go to whatever code holds the receiver (a device loop, a
/// test). A full channel is retried; a dropped receiver is final.
pub struct ChannelOut {
    name: String,
    sender: mpsc::SyncSender<Delivery>,
}

impl ChannelOut {
    /// Named `name`, holding up to `capacity` deliveries (at least 1); your code keeps the
    /// receiver
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

impl Act for ChannelOut {
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
                .map(|()| serde_json::Value::Null)
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
