//! Serve [Burn](https://burn.dev) models with [furnace](https://docs.rs/furnace-server).
//!
//! furnace-server is the server: the KServe / Triton Open Inference Protocol over HTTP and gRPC,
//! dynamic batching, reloads, metrics, API keys, TLS, the LLM routes and the nervous system.
//! This crate runs Burn models in it: a `forward` function ([`from_fn`]), a model of named
//! tensors ([`BurnModel`], served with [`custom`]), weights from Burnpack files ([`weights`]),
//! the file-based MLP ([`load_model`]) and, with the `onnx` feature, ONNX models compiled to
//! Burn at build time ([`load_built_in_model`]). Add both crates:
//!
//! ```toml
//! [dependencies]
//! furnace = { package = "furnace-server", version = "0.15" }
//! furnace-burn = "0.15"
//! ```
//!
//! # Serving a Burn model
//!
//! A model that takes one float tensor and returns one is served by its `forward`, through
//! [`from_fn`]. It answers `POST /predict` and `POST /v2/models/{name}/infer`, and concurrent
//! requests are batched together.
//!
//! ```no_run
//! use burn::module::Module;
//! use burn::nn::{Linear, LinearConfig};
//! use burn::tensor::{Device, Tensor};
//! use furnace::Server;
//! use furnace_burn::BackendKind;
//!
//! #[derive(Module, Debug)]
//! struct Net {
//!     linear: Linear,
//! }
//!
//! impl Net {
//!     fn new(device: &Device) -> Self {
//!         Self { linear: LinearConfig::new(784, 10).init(device) }
//!     }
//!
//!     fn forward(&self, images: Tensor<2>) -> Tensor<2> {
//!         self.linear.forward(images)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> furnace::Result<()> {
//!     // Logs to stderr: where the server listens, what it serves, why it stopped
//!     tracing_subscriber::fmt::init();
//!     let device = BackendKind::Cpu.device();
//!     let mut net = Net::new(&device);
//!     // The weights saved after training, as a Burnpack file (burn-store)
//!     furnace_burn::weights::load_burnpack(&mut net, "net.bpk")?;
//!
//!     // Per-item shapes, without the batch: [batch, 784] in, [batch, 10] out
//!     let model = furnace_burn::from_fn("digits", BackendKind::Cpu, &[784], &[10],
//!         move |images: Tensor<2>| net.forward(images))?;
//!     Server::new().port(3000).model(model).serve().await
//! }
//! ```
//!
//! # Several inputs or outputs, integer tensors
//!
//! Implement [`BurnModel`] to name a model's tensors: a text classifier taking token ids and a
//! mask, a detector returning boxes and scores. The server checks every request against
//! [`BurnModel::inputs`] before [`BurnModel::run`] sees it, and serves the model on
//! `/v2/models/{name}/infer`, batching requests when every input has a fixed shape.
//! [`HostTensorExt`] turns the server's tensors into Burn's and back.
//!
//! ```no_run
//! use burn::tensor::{Device, Int, Tensor};
//! use furnace::{DataType, HostTensor, IoSpec, Server};
//! use furnace_burn::{BackendKind, BurnModel, HostTensorExt};
//!
//! # #[derive(Debug)] struct TextNet;
//! # impl TextNet {
//! #     fn forward(&self, ids: Tensor<2, Int>, mask: Tensor<2, Int>) -> Tensor<2> { mask.float() }
//! # }
//! #[derive(Debug)]
//! struct Classifier {
//!     net: TextNet, // your Burn Module
//!     device: Device,
//! }
//!
//! impl BurnModel for Classifier {
//!     fn name(&self) -> &str {
//!         "classifier"
//!     }
//!
//!     fn backend(&self) -> BackendKind {
//!         BackendKind::Cpu
//!     }
//!
//!     fn inputs(&self) -> Vec<IoSpec> {
//!         vec![
//!             IoSpec::new("input_ids", DataType::Int64, &[Some(128)]),
//!             IoSpec::new("attention_mask", DataType::Int64, &[Some(128)]),
//!         ]
//!     }
//!
//!     fn outputs(&self) -> Vec<IoSpec> {
//!         vec![IoSpec::fp32("logits", &[2])]
//!     }
//!
//!     fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
//!         // Checked already: two tensors, in the order of inputs(), [batch, 128] each
//!         let mut inputs = inputs.into_iter();
//!         let ids = inputs.next().unwrap().into_int::<2>(&self.device)?;
//!         let mask = inputs.next().unwrap().into_int::<2>(&self.device)?;
//!         Ok(vec![HostTensor::from_float(self.net.forward(ids, mask))?])
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> furnace::Result<()> {
//!     tracing_subscriber::fmt::init();
//!     let device = BackendKind::Cpu.device();
//!     let model = furnace_burn::custom(Classifier { net: TextNet, device });
//!     Server::new().model(model).serve().await
//! }
//! ```
//!
//! # Your own Burn LLM
//!
//! furnace's `llm` module (feature `llm` of furnace-server) serves a `StepModel`. A toy: a Burn
//! model that maps tokens to next-token logits, served with step-wise decoding.
//! It keeps no KV cache, so each step runs the whole sequence again; a real model keeps its
//! cache in the `StepModel::Sequence`, and furnace drops it when the request ends or the
//! client disconnects.
//!
//! ```no_run
//! use burn::module::Module;
//! use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
//! use burn::tensor::{Device, Int, Tensor, TensorData};
//! use furnace::llm::{BoxError, Logits, Scheduled, StepModel, TextPipeline};
//! use furnace::Server;
//! use furnace_burn::BackendKind;
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
//!     furnace_burn::weights::load_burnpack(&mut net, "tiny-lm.bpk")?;
//!
//!     let engine = Scheduled::new(BurnLm { net, device }, pipeline);
//!     Server::new().port(8080).llm(engine).serve().await?;
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - `onnx`: compile the ONNX files in `models/` next to this crate's manifest (or
//!   `$FURNACE_MODELS_DIR`) into built-in models. Not needed to serve your own Burn models.
//! - `embed-weights`: compile built-in models' weights into the binary.
//! - `wgpu`, `metal`, `cuda`: Burn's GPU backends, chosen with [`BackendKind`].

pub mod backend;
pub mod mlp;
pub mod model;
// The built-in ONNX models' registry, for the `furnace` command; not part of the API
#[doc(hidden)]
pub mod models;
pub mod tensor;
pub mod weights;

pub use backend::BackendKind;
pub use model::{
    BurnModel, MlpModel, custom, from_fn, load_built_in_model, load_built_in_model_with_backend,
    load_model, load_model_with_backend,
};
pub use tensor::HostTensorExt;

// The Rust examples in the repository's README and examples/README.md use Burn, so they compile
// as doc-tests here rather than in furnace-server
#[cfg(doctest)]
#[doc = include_str!("../../README.md")]
mod readme {}
#[cfg(doctest)]
#[doc = include_str!("../../examples/README.md")]
mod examples_readme {}
