//! Push commands to Server-Sent-Events subscribers

use bytes::Bytes;

use super::{Act, Action};
use crate::nerve::signal::{Command, Signal};

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
    use super::super::Effector;
    use super::*;
    use crate::nerve::testing::{command, fabric_with};

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
