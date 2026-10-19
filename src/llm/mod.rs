//! Serving LLMs (feature `llm`): the OpenAI chat and completions API, with streaming, next to
//! the other models of the same server.
//!
//! Furnace owns the server side of an LLM: the OpenAI-compatible routes and their errors, SSE
//! streaming, cancellation when the client goes away, admission and backpressure, and the
//! metrics (time to first token, inter-token latency, tokens per second). The engine that runs
//! the model sits behind one of two traits:
//!
//! - [`LlmEngine`]: a request goes in, a stream of [`Event`]s comes out, and dropping the stream
//!   cancels the request. Engines with their own scheduler implement it directly.
//! - [`StepModel`]: a plain model that can prefill a prompt and decode one token at a time over
//!   an opaque per-sequence state. [`Scheduled`] turns one into an [`LlmEngine`], with a
//!   [`TextPipeline`] for the chat template, the tokenizer, sampling, incremental detokenization
//!   and stop strings. This is the way in for your own Burn LLM, and for the candle adapter
//!   (the `furnace-candle` crate).
//!
//! [`Server::llm`](crate::Server::llm) serves an engine on `POST /v1/chat/completions`,
//! `POST /v1/completions` and `GET /v1/models`, behind the server's API keys, rate limit,
//! concurrency limit and TLS.
//!
//! # Your own Burn LLM
//!
//! A toy: a Burn model that maps tokens to next-token logits, served with step-wise decoding.
//! It keeps no KV cache, so each step runs the whole sequence again; a real model keeps its
//! cache in the [`StepModel::Sequence`], and furnace drops it when the request ends or the
//! client disconnects.
//!
//! ```no_run
//! use burn::module::Module;
//! use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
//! use burn::tensor::{Device, Int, Tensor, TensorData};
//! use furnace::llm::{BoxError, Logits, Scheduled, StepModel, TextPipeline};
//! use furnace::{BackendKind, Server};
//!
//! #[derive(Module, Debug)]
//! struct TinyLm {
//!     embedding: Embedding,
//!     head: Linear,
//! }
//!
//! impl TinyLm {
//!     fn new(vocab: usize, device: &Device) -> Self {
//!         Self {
//!             embedding: EmbeddingConfig::new(vocab, 64).init(device),
//!             head: LinearConfig::new(64, vocab).init(device),
//!         }
//!     }
//!
//!     /// `[1, seq]` token ids -> `[1, seq, vocab]` logits
//!     fn forward(&self, ids: Tensor<2, Int>) -> Tensor<3> {
//!         self.head.forward(self.embedding.forward(ids))
//!     }
//! }
//!
//! #[derive(Debug)]
//! struct BurnLm {
//!     net: TinyLm,
//!     device: Device,
//! }
//!
//! impl BurnLm {
//!     /// The logits after the last of `tokens`
//!     fn next_logits(&self, tokens: &[u32]) -> Result<Logits, BoxError> {
//!         let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
//!         let ids = Tensor::from_data(TensorData::new(ids, [1, tokens.len()]), &self.device);
//!         let logits = self.net.forward(ids); // [1, seq, vocab]
//!         let [_, seq, vocab] = logits.dims();
//!         let last = logits.slice([0..1, seq - 1..seq, 0..vocab]).flatten::<1>(0, 2);
//!         Ok(Logits(last.into_data().try_into_vec::<f32>()?))
//!     }
//! }
//!
//! impl StepModel for BurnLm {
//!     /// The tokens so far; a real model keeps its KV cache and position here
//!     type Sequence = Vec<u32>;
//!
//!     fn name(&self) -> &str {
//!         "tiny-lm"
//!     }
//!
//!     fn max_context(&self) -> usize {
//!         512
//!     }
//!
//!     fn prefill(&self, prompt: &[u32]) -> Result<(Vec<u32>, Logits), BoxError> {
//!         Ok((prompt.to_vec(), self.next_logits(prompt)?))
//!     }
//!
//!     fn decode(&self, tokens: &mut Vec<u32>, token: u32) -> Result<Logits, BoxError> {
//!         tokens.push(token);
//!         self.next_logits(tokens)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), BoxError> {
//!     // The tokenizer and chat template your model was trained with
//!     let pipeline = TextPipeline::from_files("tokenizer.json", Some("tokenizer_config.json".as_ref()))?;
//!     let device = BackendKind::Cpu.device();
//!     let mut net = TinyLm::new(pipeline.tokenizer().get_vocab_size(true), &device);
//!     furnace::weights::load_burnpack(&mut net, "tiny-lm.bpk".as_ref())?;
//!
//!     let engine = Scheduled::new(BurnLm { net, device }, pipeline);
//!     Server::new().port(8080).llm(engine).serve().await?;
//!     Ok(())
//! }
//! ```

mod http;
mod metrics;
pub mod mock;
mod pipeline;
mod sampling;
mod scheduler;
mod stop;

use std::fmt;

pub(crate) use http::{LlmState, routes};
pub use metrics::{FinishedCounts, LlmMetrics};
pub use pipeline::{TextPipeline, Tokenizer};
pub use sampling::Sampler;
pub use scheduler::{Logits, Scheduled, SchedulerConfig, SchedulerStats, StepModel};
pub use stop::StopStrings;

/// Any error a model returns; furnace reports its message
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// One message of a chat
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Message {
    /// `system`, `user`, `assistant` or `tool`
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// What to generate from
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    /// A chat (`/v1/chat/completions`), to go through the model's chat template
    Chat(Vec<Message>),
    /// Raw text (`/v1/completions`)
    Text(String),
}

/// How to pick tokens; `None` means the engine's default
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct SamplingParams {
    /// Most tokens to generate; by default, up to the end of the context
    pub max_tokens: Option<usize>,
    /// 0 picks the most likely token every time; OpenAI's default is 1
    pub temperature: Option<f32>,
    /// Nucleus sampling: the smallest set of tokens whose probabilities add up to this
    pub top_p: Option<f32>,
    /// Only the `top_k` most likely tokens
    pub top_k: Option<usize>,
    /// Makes sampling repeatable
    pub seed: Option<u64>,
    /// Subtracted from the logit of every token generated already
    pub presence_penalty: f32,
    /// Subtracted from the logit of a token once per time it was generated
    pub frequency_penalty: f32,
}

/// A request for an engine, parsed and checked by the HTTP layer
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GenerateRequest {
    pub input: Input,
    pub params: SamplingParams,
    /// Generation stops before any of these; they are not part of the output
    pub stop: Vec<String>,
}

impl GenerateRequest {
    /// A chat request with the engine's default sampling
    pub fn chat(messages: Vec<Message>) -> Self {
        Self {
            input: Input::Chat(messages),
            params: SamplingParams::default(),
            stop: Vec::new(),
        }
    }

    /// A completion of raw text with the engine's default sampling
    pub fn text(prompt: &str) -> Self {
        Self {
            input: Input::Text(prompt.to_string()),
            params: SamplingParams::default(),
            stop: Vec::new(),
        }
    }

    pub fn with_params(mut self, params: SamplingParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }
}

impl SamplingParams {
    /// Always the most likely token
    pub fn greedy() -> Self {
        Self {
            temperature: Some(0.0),
            ..Self::default()
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// Why generation ended
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An end-of-sequence token or a stop string
    Stop,
    /// `max_tokens`, or the end of the context
    Length,
}

impl FinishReason {
    /// The OpenAI `finish_reason`
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// What an engine sends back, in order: `Started` once, `Text` any number of times, then
/// `Finished` or `Error`
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The request left the queue and the engine started on it
    Started {
        prompt_tokens: usize,
    },
    /// The next piece of the output (never empty)
    Text(String),
    Finished {
        reason: FinishReason,
        completion_tokens: usize,
    },
    Error(EngineError),
}

/// Why an engine refused or failed a request
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The prompt does not fit the context (400)
    #[error("the prompt is {prompt_tokens} tokens, the model's context is {max_context}")]
    TooLong {
        prompt_tokens: usize,
        max_context: usize,
    },
    /// The engine is full (503)
    #[error("{0}")]
    Busy(String),
    /// The request cannot be served as asked (400)
    #[error("{0}")]
    Invalid(String),
    /// The model failed (500)
    #[error("{0}")]
    Failed(String),
}

/// An LLM furnace can serve.
///
/// [`generate`](Self::generate) checks and queues a request and returns its events. Dropping the
/// receiver cancels the request: the engine must stop generating for it and free its memory.
/// Engines with their own scheduler implement this; plain models implement [`StepModel`] and are
/// wrapped in [`Scheduled`].
pub trait LlmEngine: Send + Sync + fmt::Debug + 'static {
    /// The name clients ask for in `model`
    fn name(&self) -> &str;

    /// Tokens of prompt and output together
    fn max_context(&self) -> usize;

    /// Reject before any stream exists: the prompt is too long, or the engine is full. Called
    /// on a Tokio runtime, and must not block it.
    fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<Event>, EngineError>;

    /// Model time per step, for engines furnace schedules ([`Scheduled`])
    fn scheduler_stats(&self) -> Option<&SchedulerStats> {
        None
    }
}
