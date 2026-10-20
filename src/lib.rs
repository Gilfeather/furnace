//! An inference server for [Burn](https://burn.dev) models.
//!
//! Furnace serves the models you write in Burn, with what a production inference server needs:
//! the KServe / Triton Open Inference Protocol over HTTP and gRPC (so Triton and KServe clients
//! work unchanged), dynamic batching, several models per process with reloads that don't drop
//! a request, Prometheus metrics, readiness after a warm-up, API keys, rate limits and TLS, all
//! in pure Rust. It also ships as the `furnace` command, which serves ONNX models compiled to
//! Burn at build time; see the [README](https://github.com/Gilfeather/furnace#readme).
//!
//! # Serving a Burn model
//!
//! A model that takes one float tensor and returns one is served by its `forward`, through
//! [`Model::from_fn`]. It answers `POST /predict` and `POST /v2/models/{name}/infer`, and
//! concurrent requests are batched together.
//!
//! ```no_run
//! use burn::module::Module;
//! use burn::nn::{Linear, LinearConfig};
//! use burn::tensor::{Device, Tensor};
//! use furnace::{BackendKind, Model, Server};
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
//!     let device = BackendKind::Cpu.device();
//!     let mut net = Net::new(&device);
//!     // The weights saved after training, as a Burnpack file (burn-store)
//!     furnace::weights::load_burnpack(&mut net, "net.bpk".as_ref())?;
//!
//!     // Per-item shapes, without the batch: [batch, 784] in, [batch, 10] out
//!     let model = Model::from_fn("digits", BackendKind::Cpu, &[784], &[10],
//!         move |images: Tensor<2>| net.forward(images))?;
//!     Server::new().port(3000).model(model).serve().await
//! }
//! ```
//!
//! # Mounting it in your own axum app
//!
//! [`Server::router`] returns the inference API as an axum `Router`. Merge it with your own
//! routes and serve the result as usual: furnace answers `/predict`, `/v2/...`, `/healthz`,
//! `/readyz` and `/metrics`, with batching and backpressure, and your routes do the rest.
//!
//! ```no_run
//! use axum::{Json, Router, routing::get};
//! use burn::tensor::Tensor;
//! use furnace::{BackendKind, Model, Server};
//!
//! # use burn::module::Module;
//! # use burn::nn::{Linear, LinearConfig};
//! # use burn::tensor::Device;
//! # #[derive(Module, Debug)]
//! # struct Net { linear: Linear }
//! # impl Net {
//! #     fn new(device: &Device) -> Self { Self { linear: LinearConfig::new(784, 10).init(device) } }
//! #     fn forward(&self, images: Tensor<2>) -> Tensor<2> { self.linear.forward(images) }
//! # }
//! // One of your own routes: the class names for the 10 logits /predict returns
//! async fn labels() -> Json<Vec<String>> {
//!     Json((0..10).map(|digit| digit.to_string()).collect())
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let device = BackendKind::Cpu.device();
//!     let mut net = Net::new(&device);
//!     furnace::weights::load_burnpack(&mut net, "net.bpk".as_ref())?;
//!     let model = Model::from_fn("digits", BackendKind::Cpu, &[784], &[10],
//!         move |images: Tensor<2>| net.forward(images))?;
//!
//!     let app = Router::new()
//!         .route("/labels", get(labels))
//!         .merge(Server::new().model(model).router()?);
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
//!     axum::serve(listener, app).await?;
//!     Ok(())
//! }
//! ```
//!
//! # Several inputs or outputs, integer tensors
//!
//! Implement [`BurnModel`] to name a model's tensors: a text classifier taking token ids and a
//! mask, a detector returning boxes and scores. The server checks every request against
//! [`BurnModel::inputs`] before [`BurnModel::run`] sees it, and serves the model on
//! `/v2/models/{name}/infer`, batching requests when every input has a fixed shape.
//!
//! ```no_run
//! use burn::tensor::{Device, Int, Tensor};
//! use furnace::{BackendKind, BurnModel, DataType, HostTensor, IoSpec, Model, Server};
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
//!     let device = BackendKind::Cpu.device();
//!     let model = Model::custom(Classifier { net: TextNet, device });
//!     Server::new().model(model).serve().await
//! }
//! ```
//!
//! # Configuration
//!
//! [`Server::configure`] reaches every [`ServerConfig`] setting: limits, timeouts, batching,
//! gRPC (feature `grpc`), TLS (feature `tls`), API keys, rate limits, the admin token for
//! [`Server::reloadable_model`]. [`Server::router`] returns the HTTP API as an axum `Router`
//! to mount in your own application. The HTTP API is described in
//! [docs/API.md](https://github.com/Gilfeather/furnace/blob/main/docs/API.md).
//!
//! # Features
//!
//! - `onnx` (default): compile the ONNX files in `models/` (or `$FURNACE_MODELS_DIR`) into
//!   built-in models. Not needed to serve your own Burn models.
//! - `grpc`: the Open Inference Protocol over gRPC as well.
//! - `tls`: HTTPS and mutual TLS, with a pure-Rust crypto provider.
//! - `image`: JPEG / PNG input on `/predict`, with ImageNet preprocessing.
//! - `llm`: serve LLMs with the OpenAI chat and completions API, streaming included, next to
//!   the other models; see the `llm` module.
//! - `nerve`: kept for compatibility. The nervous system (the `nerve` module) is always
//!   built, since the server's own routes run on it.
//! - `wgpu`, `metal`, `cuda`: Burn's GPU backends, chosen with [`BackendKind`].
//! - `embed-weights`: compile built-in models' weights into the binary.
//! - `test-models`: `model::DummyModel` and `model::DummyTensorsModel`, constant models for
//!   testing and benchmarking the server itself.

pub mod access;
pub mod allocator;
pub mod api;
pub mod backend;
pub mod batching;
pub mod burn_model;
pub mod config;
pub mod error;
mod json_fast;
#[cfg(feature = "llm")]
pub mod llm;
pub mod model;
pub mod models;
pub mod nerve;
#[cfg(feature = "image")]
pub mod preprocess;
pub mod telemetry;
pub mod tensor;
pub mod threads;
#[cfg(feature = "tls")]
pub mod tls;
pub mod types;
pub mod weights;

pub use api::{Server, ServerConfig, start_server, start_server_with_config};
pub use backend::BackendKind;
pub use error::{ApiError, CliError, FurnaceError, ModelError, Result};
pub use model::{BurnModel, Model, ModelInfo, load_built_in_model, load_model};
pub use tensor::{DataType, HostTensor, IoSpec, TensorSpec, Values};
