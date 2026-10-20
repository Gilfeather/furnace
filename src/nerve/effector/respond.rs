//! The answer to the request that started the chain

use bytes::Bytes;
use tracing::debug;

use super::{Act, Action};
use crate::nerve::signal::{Answer, Command, Payload, Signal};

/// The answer to the request that started the chain. The command's `args` is
/// `{"status": 200, "content_type": "...", "body": ...}` (a string body as it is, anything
/// else as JSON); a plain signal is answered with its payload, tensors as tensors for the
/// protocol that asked.
///
/// A requester that stopped waiting (it timed out, or disconnected) is not a failure of the
/// effector: the answer is dropped, and the circuit breaker never counts it.
pub struct Respond {
    name: String,
}

impl Respond {
    /// Registered as `name`, the effector commands address
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
                    Some(serde_json::Value::String(text)) => Answer::Http {
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
                Payload::Tensors(tensors) => Answer::Tensors(tensors.to_vec()),
                Payload::Text(text) => Answer::text(200, text),
                Payload::Bytes(bytes) => {
                    Answer::bytes(200, "application/octet-stream", bytes.clone())
                }
                Payload::Json(value) => Answer::json(200, value),
                other => Answer::bytes(200, "application/octet-stream", other.to_bytes()),
            },
            other => Answer::json(200, other),
        };
        if !signal.answer(answer) {
            debug!("nerve: '{}': nobody is waiting for the answer", self.name);
        }
        Action::ok()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Effector;
    use super::*;
    use crate::nerve::signal::{Feature, ReplyHandle};
    use crate::nerve::testing::{command, fabric_with};

    #[test]
    fn respond_answers_the_waiting_request() {
        let effector = Effector::new(Respond::new("respond"));
        let (fabric, r) = fabric_with(effector.clone());
        let (reply, mut receiver) = ReplyHandle::new();
        let signal = Signal::new(Feature::NONE, Payload::text("payload")).with_reply(reply);
        fabric.stimulate(
            r,
            signal.command(
                "respond",
                serde_json::json!({"status": 201, "body": {"ok": true}}),
            ),
        );
        let Answer::Http {
            status,
            content_type,
            body,
        } = receiver.try_recv().unwrap()
        else {
            panic!("an HTTP answer");
        };
        assert_eq!(status, 201);
        assert_eq!(content_type, "application/json");
        assert_eq!(body, Bytes::from(r#"{"ok":true}"#));
        // Nobody waiting: not a failure of the effector, so the breaker never counts it
        fabric.stimulate(r, command("respond", serde_json::Value::Null));
        assert_eq!(effector.metrics().failed, 0);
        // A plain signal answers with its payload
        let (reply, mut receiver) = ReplyHandle::new();
        fabric.stimulate(
            r,
            Signal::new(Feature::NONE, Payload::text("plain")).with_reply(reply),
        );
        let Answer::Http { body, .. } = receiver.try_recv().unwrap() else {
            panic!("an HTTP answer");
        };
        assert_eq!(body, Bytes::from("plain"));
        // Tensors go back as tensors, for the protocol that asked
        let (reply, mut receiver) = ReplyHandle::new();
        let tensor = crate::tensor::HostTensor {
            shape: vec![1, 2],
            values: crate::tensor::Values::F32(vec![1.0, 2.0]),
        };
        fabric.stimulate(
            r,
            Signal::new(Feature::NONE, Payload::tensors(vec![tensor.clone()])).with_reply(reply),
        );
        let Answer::Tensors(tensors) = receiver.try_recv().unwrap() else {
            panic!("tensors");
        };
        assert_eq!(tensors, vec![tensor]);
    }
}
