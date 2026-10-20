//! Deterministic engines for tests and for measuring the server's own overhead: they stream a
//! fixed reply with configurable delays, and count the requests they hold.
//!
//! - [`MockStepModel`] is a [`StepModel`]: serve it through [`Scheduled`](super::Scheduled)
//!   with [`byte_pipeline`], and the whole furnace path runs (scheduler, sampling,
//!   detokenization, stop strings).
//! - [`MockEngine`] is an [`LlmEngine`] with its own loop, like an engine that schedules itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;

use super::{
    BoxError, EngineError, Event, FinishReason, GenerateRequest, LlmEngine, Logits, Prompt,
    StepModel, TextPipeline,
};
use crate::llm::pipeline::Tokenizer;

/// The end-of-sequence token of [`byte_tokenizer`]
pub const EOS_TOKEN: &str = "<|eos|>";
/// Its id: right after the 256 bytes
pub const EOS_ID: u32 = 256;

/// A byte-level tokenizer with one token per byte and [`EOS_TOKEN`]: every character of more
/// than one byte spans several tokens, as in real byte-level BPE vocabularies
pub fn byte_tokenizer() -> Tokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = (0u8..=255)
        .map(|b| (byte_char(b).to_string(), serde_json::Value::from(b)))
        .collect();
    let json = serde_json::json!({
        "version": "1.0",
        "added_tokens": [{
            "id": EOS_ID, "content": EOS_TOKEN, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        }],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
            "use_regex": false
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
            "use_regex": false
        },
        "model": {"type": "BPE", "vocab": vocab, "merges": []}
    });
    json.to_string()
        .parse()
        .expect("the byte-level test tokenizer is valid")
}

/// GPT-2's map from bytes to printable characters, which byte-level vocabularies are written in
fn byte_char(b: u8) -> char {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    if printable(b) {
        char::from(b)
    } else {
        let shift = (0..b).filter(|&x| !printable(x)).count() as u32;
        char::from_u32(256 + shift).expect("a valid character")
    }
}

/// [`byte_tokenizer`] with [`EOS_TOKEN`] and a simple chat template:
/// `<|role|>content\n` per message, then `<|assistant|>`
pub fn byte_pipeline() -> TextPipeline {
    TextPipeline::new(byte_tokenizer())
        .with_eos_token(EOS_TOKEN)
        .expect("the EOS token is in the vocabulary")
        .with_chat_template(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}\
             {% if add_generation_prompt %}<|assistant|>{% endif %}",
        )
        .expect("the chat template is valid")
}

/// A count shared with a mock model, readable after the model moved into an engine: what is
/// alive (requests, sequences) or, for [`MockStepModel::decodes`], decode calls so far
#[derive(Debug, Clone, Default)]
pub struct SharedCount(Arc<AtomicUsize>);

/// The old name of [`SharedCount`]
#[deprecated(since = "0.11.0", note = "renamed to `SharedCount`")]
pub type Live = SharedCount;

impl SharedCount {
    /// The count now
    pub fn get(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }

    fn guard(&self) -> CountGuard {
        self.0.fetch_add(1, Ordering::AcqRel);
        CountGuard(self.0.clone())
    }
}

#[derive(Debug)]
struct CountGuard(Arc<AtomicUsize>);

impl Drop for CountGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A [`StepModel`] that answers every prompt with the same tokens, then EOS
#[derive(Debug)]
pub struct MockStepModel {
    name: String,
    reply: Vec<u32>,
    vocab: usize,
    eos: u32,
    max_context: usize,
    max_sequences: usize,
    prefill_delay: Duration,
    token_delay: Duration,
    live: SharedCount,
    decodes: SharedCount,
}

/// A [`MockStepModel`] sequence: how far into the reply it is
#[derive(Debug)]
pub struct MockSequence {
    position: usize,
    _live: CountGuard,
}

impl MockStepModel {
    /// Answers `reply`, as tokenized by [`byte_tokenizer`], with no delays
    pub fn new(name: &str, reply: &str) -> Self {
        Self {
            name: name.to_string(),
            reply: reply.bytes().map(u32::from).collect(),
            vocab: EOS_ID as usize + 1,
            eos: EOS_ID,
            max_context: 4096,
            max_sequences: 16,
            prefill_delay: Duration::ZERO,
            token_delay: Duration::ZERO,
            live: SharedCount::default(),
            decodes: SharedCount::default(),
        }
    }

    /// Each prefill takes this long
    pub fn prefill_delay(mut self, delay: Duration) -> Self {
        self.prefill_delay = delay;
        self
    }

    /// Each decode step takes this long
    pub fn token_delay(mut self, delay: Duration) -> Self {
        self.token_delay = delay;
        self
    }

    /// The context it reports, prompt and output together (4096 by default)
    pub fn max_context(mut self, tokens: usize) -> Self {
        self.max_context = tokens;
        self
    }

    /// The sequences it holds at once (16 by default)
    pub fn max_sequences(mut self, sequences: usize) -> Self {
        self.max_sequences = sequences;
        self
    }

    /// The sequences alive, as a handle that outlives moving the model into an engine
    pub fn live(&self) -> SharedCount {
        self.live.clone()
    }

    /// Decode calls finished so far, as a handle like [`live`](Self::live): steps, counted
    /// without a clock
    pub fn decodes(&self) -> SharedCount {
        self.decodes.clone()
    }

    /// Logits that make `token` the clear choice
    fn logits_for(&self, position: usize) -> Logits {
        let token = self.reply.get(position).copied().unwrap_or(self.eos);
        let mut logits = vec![0.0; self.vocab];
        logits[token as usize] = 10.0;
        Logits(logits)
    }
}

impl StepModel for MockStepModel {
    type Sequence = MockSequence;

    fn name(&self) -> &str {
        &self.name
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn max_sequences(&self) -> usize {
        self.max_sequences
    }

    fn prefill(&self, _prompt: &[u32]) -> Result<(MockSequence, Logits), BoxError> {
        std::thread::sleep(self.prefill_delay);
        let sequence = MockSequence {
            position: 0,
            _live: self.live.guard(),
        };
        Ok((sequence, self.logits_for(0)))
    }

    fn decode(&self, sequence: &mut MockSequence, _token: u32) -> Result<Logits, BoxError> {
        std::thread::sleep(self.token_delay);
        sequence.position += 1;
        self.decodes.0.fetch_add(1, Ordering::AcqRel);
        Ok(self.logits_for(sequence.position))
    }
}

/// An [`LlmEngine`] with its own loop: `Started`, then the reply in pieces, each after a delay
#[derive(Debug)]
pub struct MockEngine {
    name: String,
    pieces: Vec<String>,
    first_delay: Duration,
    piece_delay: Duration,
    max_context: usize,
    live: SharedCount,
}

impl MockEngine {
    /// Streams `pieces`, one `Text` each, with no delays
    pub fn new(name: &str, pieces: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            pieces: pieces.iter().map(|p| p.to_string()).collect(),
            first_delay: Duration::ZERO,
            piece_delay: Duration::ZERO,
            max_context: 4096,
            live: SharedCount::default(),
        }
    }

    /// Before the first piece (the prefill)
    pub fn first_delay(mut self, delay: Duration) -> Self {
        self.first_delay = delay;
        self
    }

    /// Before every other piece
    pub fn piece_delay(mut self, delay: Duration) -> Self {
        self.piece_delay = delay;
        self
    }

    /// The generations running, as a handle that outlives moving the engine into a server
    pub fn live(&self) -> SharedCount {
        self.live.clone()
    }
}

impl LlmEngine for MockEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn generate(&self, request: GenerateRequest) -> Result<mpsc::Receiver<Event>, EngineError> {
        let prompt_tokens = match &request.input {
            Prompt::Chat(messages) => messages.iter().map(|m| m.content.len()).sum(),
            Prompt::Text(text) => text.len(),
        };
        if prompt_tokens >= self.max_context {
            return Err(EngineError::TooLong {
                prompt_tokens,
                max_context: self.max_context,
            });
        }
        let max_tokens = request.params.max_tokens.unwrap_or(usize::MAX);
        let pieces = self.pieces.clone();
        let (first_delay, piece_delay) = (self.first_delay, self.piece_delay);
        let live = self.live.guard();
        let (events, receiver) = mpsc::channel(16);
        tokio::spawn(async move {
            let _live = live;
            let send = |event| {
                let events = events.clone();
                async move {
                    tokio::select! {
                        sent = events.send(event) => sent.is_ok(),
                        () = events.closed() => false,
                    }
                }
            };
            if !send(Event::Started { prompt_tokens }).await {
                return;
            }
            for (i, piece) in pieces.iter().enumerate() {
                if i == max_tokens {
                    send(Event::Finished {
                        reason: FinishReason::Length,
                        completion_tokens: i,
                    })
                    .await;
                    return;
                }
                let delay = if i == 0 { first_delay } else { piece_delay };
                tokio::select! {
                    () = tokio::time::sleep(delay) => {}
                    () = events.closed() => return,
                }
                if !send(Event::Text(piece.clone())).await {
                    return;
                }
            }
            send(Event::Finished {
                reason: FinishReason::Stop,
                completion_tokens: pieces.len(),
            })
            .await;
        });
        Ok(receiver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_tokenizer_has_one_token_per_byte() {
        let tokenizer = byte_tokenizer();
        let text = "Hi é ✓";
        let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(ids, text.bytes().map(u32::from).collect::<Vec<_>>());
        assert_eq!(tokenizer.decode(&ids, true).unwrap(), text);
        assert_eq!(tokenizer.token_to_id(EOS_TOKEN), Some(EOS_ID));
    }

    #[test]
    fn incremental_detokenization_waits_for_whole_characters() {
        let tokenizer = byte_tokenizer();
        let mut stream = tokenizer.decode_stream(true);
        let mut out = Vec::new();
        for byte in "a✓".bytes() {
            out.push(stream.step(u32::from(byte)).unwrap());
        }
        assert_eq!(
            out,
            [Some("a".to_string()), None, None, Some("✓".to_string())]
        );
    }
}
