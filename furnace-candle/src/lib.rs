//! GGUF LLMs on the CPU through [candle](https://github.com/huggingface/candle), served by
//! [furnace](https://github.com/Gilfeather/furnace): quantized models of the llama family
//! (Llama, SmolLM, Mistral and others with the `llama` architecture) and Qwen2 / Qwen2.5,
//! from local files.
//!
//! ```no_run
//! use furnace::Server;
//! use furnace_candle::Gguf;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), furnace::llm::BoxError> {
//!     // The tokenizer and the chat template come from the GGUF file too
//!     let engine = Gguf::new("qwen2.5-0.5b-instruct-q4_k_m.gguf").load()?;
//!     Server::new().port(8080).llm(engine).serve().await?;
//!     Ok(())
//! }
//! ```
//!
//! Each request is a sequence of its own: the weights are shared, and every sequence has its
//! own KV cache, which furnace drops when the request ends or the client disconnects. A
//! forward pass runs one sequence (batch 1); furnace's scheduler interleaves the running
//! sequences token by token.
//!
//! Everything is pure Rust: candle's CPU backend, without its CUDA, MKL, Accelerate or Metal
//! features, and tokenizers with fancy-regex instead of Oniguruma. Nothing is downloaded; get
//! the files yourself.

mod model;

use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use furnace::llm::{
    BoxError, Logits, Scheduled, SchedulerConfig, StepModel, TextPipeline, Tokenizer,
};

pub use model::{Architecture, Cache, Weights};

/// Default limit on a sequence's tokens, since the KV cache grows with it; the GGUF file's
/// trained context can be much longer (32k for Qwen2.5)
pub const DEFAULT_MAX_CONTEXT: usize = 4096;
/// Default limit on sequences at once
pub const DEFAULT_MAX_SEQUENCES: usize = 4;

/// How to load a GGUF file
#[derive(Debug, Clone)]
pub struct Gguf {
    path: PathBuf,
    name: Option<String>,
    tokenizer: Option<PathBuf>,
    max_context: usize,
    max_sequences: usize,
    scheduler: SchedulerConfig,
}

impl Gguf {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            name: None,
            tokenizer: None,
            max_context: DEFAULT_MAX_CONTEXT,
            max_sequences: DEFAULT_MAX_SEQUENCES,
            scheduler: SchedulerConfig::default(),
        }
    }

    /// The name clients ask for in `model`; by default the file name without `.gguf`
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Use this `tokenizer.json` instead of the tokenizer stored in the GGUF file
    pub fn tokenizer(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer = Some(path.as_ref().to_path_buf());
        self
    }

    /// Tokens per sequence, prompt and output together; at most the model's trained context
    pub fn max_context(mut self, tokens: usize) -> Self {
        self.max_context = tokens;
        self
    }

    /// Sequences at once; more requests wait for one to end
    pub fn max_sequences(mut self, sequences: usize) -> Self {
        self.max_sequences = sequences.max(1);
        self
    }

    /// Worker threads and queue length of furnace's scheduler
    pub fn scheduler(mut self, config: SchedulerConfig) -> Self {
        self.scheduler = config;
        self
    }

    /// The model, ready to serve: give it to [`furnace::Server::llm`]
    pub fn load(self) -> Result<Scheduled<GgufModel>, BoxError> {
        let scheduler = self.scheduler;
        let (model, pipeline) = self.load_parts()?;
        Ok(Scheduled::with_config(model, pipeline, scheduler))
    }

    /// The model and its text pipeline, to adjust before serving them
    pub fn load_parts(self) -> Result<(GgufModel, TextPipeline), BoxError> {
        let mut file = std::fs::File::open(&self.path)
            .map_err(|e| format!("cannot open {}: {e}", self.path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| format!("{} is not a GGUF file: {e}", self.path.display()))?;
        let pipeline = pipeline(&content, self.tokenizer.as_deref())?;
        let weights =
            Weights::from_gguf(&content, &mut file, Some(self.max_context), &Device::Cpu)?;
        let name = self.name.unwrap_or_else(|| {
            self.path
                .file_stem()
                .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned())
        });
        tracing::info!(
            "Loaded {name}: {:?}, {} tokens of context (trained for {}), up to {} sequences",
            weights.architecture,
            weights.max_context(),
            weights.context_length,
            self.max_sequences
        );
        let model = GgufModel {
            name,
            weights,
            max_sequences: self.max_sequences,
        };
        Ok((model, pipeline))
    }
}

/// The tokenizer, chat template and end-of-sequence tokens stored in a GGUF file
fn pipeline(
    content: &gguf_file::Content,
    tokenizer: Option<&Path>,
) -> Result<TextPipeline, BoxError> {
    let tokenizer = match tokenizer {
        Some(path) => Tokenizer::from_file(path)?,
        None => Tokenizer::from_gguf(content)
            .map_err(|e| format!("reading the tokenizer from the GGUF file: {e}"))?,
    };
    let metadata = |key: &str| content.metadata.get(key);
    let token_id = |key: &str| metadata(key).and_then(|v| v.to_u32().ok());
    let tokens = metadata("tokenizer.ggml.tokens").and_then(|v| v.to_vec().ok());
    let token_text = |id: u32| {
        tokens
            .and_then(|tokens| tokens.get(id as usize))
            .and_then(|t| t.to_string().ok())
            .cloned()
    };
    let mut pipeline = TextPipeline::new(tokenizer);
    if let Some(bos) = token_id("tokenizer.ggml.bos_token_id").and_then(token_text) {
        pipeline = pipeline.with_bos_token(&bos);
    }
    // The end of a turn, then the end of the text
    for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eos_token_id"] {
        if let Some(id) = token_id(key) {
            pipeline = match token_text(id) {
                Some(text) => pipeline.with_eos_token(&text)?,
                None => pipeline.with_eos_id(id),
            };
        }
    }
    if let Some(template) = metadata("tokenizer.chat_template").and_then(|v| v.to_string().ok()) {
        pipeline = pipeline.with_chat_template(template)?;
    }
    Ok(pipeline)
}

/// A GGUF model as a furnace [`StepModel`]
#[derive(Debug)]
pub struct GgufModel {
    name: String,
    weights: Weights,
    max_sequences: usize,
}

impl GgufModel {
    pub fn weights(&self) -> &Weights {
        &self.weights
    }
}

impl StepModel for GgufModel {
    type Sequence = Cache;

    fn name(&self) -> &str {
        &self.name
    }

    fn max_context(&self) -> usize {
        self.weights.max_context()
    }

    fn max_sequences(&self) -> usize {
        self.max_sequences
    }

    fn prefill(&self, prompt: &[u32]) -> Result<(Cache, Logits), BoxError> {
        let mut cache = self.weights.new_cache();
        let logits = self.weights.forward(&mut cache, prompt)?;
        Ok((cache, Logits(logits)))
    }

    fn decode(&self, cache: &mut Cache, token: u32) -> Result<Logits, BoxError> {
        Ok(Logits(self.weights.forward(cache, &[token])?))
    }
}
