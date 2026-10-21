//! An inference server and nervous system in pure Rust, on axum and tokio.
//!
//! Furnace serves models with what a production inference server needs: the KServe / Triton
//! Open Inference Protocol over HTTP and gRPC (so Triton and KServe clients work unchanged),
//! dynamic batching, several models per process with reloads that don't drop a request,
//! Prometheus metrics, readiness after a warm-up, API keys, rate limits and TLS. It serves LLMs
//! with the OpenAI API (the `llm` module), and connects sensors, models, the LLM and actuators
//! through its [`nerve`] module. See the [README](https://github.com/Gilfeather/furnace#readme).
//!
//! This crate runs no model itself: an engine implements [`TensorModel`]. Burn is the engine
//! furnace is built around: the `furnace-burn` crate serves a Burn model's `forward`, a Burn
//! model of named tensors, weights from Burnpack files and ONNX models compiled to Burn, and
//! the `furnace` command (crate `furnace-cli`) serves ONNX models from the command line.
//!
//! # Serving a model
//!
//! Implement [`TensorModel`] for a model and serve it with [`Server`]. It answers
//! `POST /predict` and `POST /v2/models/{name}/infer`, and concurrent requests are batched
//! together. With Burn, `furnace_burn::from_fn` does this for a `forward` function.
//!
//! ```no_run
//! use furnace::{HostTensor, IoSpec, Model, Server, TensorModel, Values};
//!
//! /// Sums each row of four values
//! #[derive(Debug)]
//! struct RowSum;
//!
//! impl TensorModel for RowSum {
//!     fn name(&self) -> &str { "rowsum" }
//!     fn backend(&self) -> &str { "cpu" }
//!     fn inputs(&self) -> Vec<IoSpec> { vec![IoSpec::fp32("input", &[4])] }
//!     fn outputs(&self) -> Vec<IoSpec> { vec![IoSpec::fp32("output", &[1])] }
//!     fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
//!         // Checked already: one FP32 tensor of [batch, 4]
//!         let Values::F32(values) = &inputs[0].values else { unreachable!() };
//!         let sums: Vec<f32> = values.chunks(4).map(|row| row.iter().sum()).collect();
//!         Ok(vec![HostTensor { shape: vec![sums.len(), 1], values: Values::F32(sums) }])
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> furnace::Result<()> {
//!     // Logs to stderr: where the server listens, what it serves, why it stopped
//!     tracing_subscriber::fmt::init();
//!     Server::new().port(3000).model(Model::custom(RowSum)).serve().await
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
//! use furnace::{Model, Server};
//!
//! # use furnace::{HostTensor, IoSpec, TensorModel, Values};
//! # #[derive(Debug)]
//! # struct Digits;
//! # impl TensorModel for Digits {
//! #     fn name(&self) -> &str { "digits" }
//! #     fn backend(&self) -> &str { "cpu" }
//! #     fn inputs(&self) -> Vec<IoSpec> { vec![IoSpec::fp32("input", &[784])] }
//! #     fn outputs(&self) -> Vec<IoSpec> { vec![IoSpec::fp32("output", &[10])] }
//! #     fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
//! #         let batch = inputs[0].batch();
//! #         Ok(vec![HostTensor { shape: vec![batch, 10], values: Values::F32(vec![0.0; batch * 10]) }])
//! #     }
//! # }
//! // One of your own routes: the class names for the 10 logits /predict returns
//! async fn labels() -> Json<Vec<String>> {
//!     Json((0..10).map(|digit| digit.to_string()).collect())
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     tracing_subscriber::fmt::init();
//!     let app = Router::new()
//!         .route("/labels", get(labels))
//!         .merge(Server::new().model(Model::custom(Digits)).router()?);
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
//!     axum::serve(listener, app).await?;
//!     Ok(())
//! }
//! ```
//!
//! # Several inputs or outputs, integer tensors
//!
//! A model names its tensors in [`TensorModel::inputs`] and [`TensorModel::outputs`]: a text
//! classifier taking token ids and a mask, a detector returning boxes and scores. The server
//! checks every request against them before [`TensorModel::run`] sees it, and serves the model
//! on `/v2/models/{name}/infer`, batching requests when every input has a fixed shape.
//!
//! ```
//! use furnace::{DataType, IoSpec};
//!
//! let inputs = vec![
//!     IoSpec::new("input_ids", DataType::Int64, &[Some(128)]),
//!     IoSpec::new("attention_mask", DataType::Int64, &[Some(128)]),
//! ];
//! let outputs = vec![IoSpec::fp32("logits", &[2])];
//! # assert_eq!(inputs[0].v2_shape(), [-1, 128]);
//! # assert_eq!(outputs[0].v2_shape(), [-1, 2]);
//! ```
//!
//! # The nervous system
//!
//! The [`nerve`] module connects receptors (an HTTP route, a channel, a timer, a stream) to
//! neurons (a closure, a Burn model, an LLM) and effectors (a webhook, a command, a file).
//! Most signals stop at a fast reflex; only what needs it reaches the slow brain.
//! [`Server::nerve`] serves the neurons next to the models, on the same port.
//!
//! ```no_run
//! use furnace::Server;
//! use furnace::nerve::prelude::*;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> furnace::Result<()> {
//!     tracing_subscriber::fmt::init();
//!     let mut b = Fabric::builder();
//!     // POST /events: each body is a signal
//!     let events = b.receptor(Http::new("events", "/events").text());
//!     // A reflex: alerts become a command to the `alert` effector, the rest stop here
//!     let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
//!         let text = signal.payload.as_text()?.to_string();
//!         text.contains("ALERT")
//!             .then(|| signal.command("alert", json!({ "text": text })))
//!     }));
//!     b.effector(Webhook::new("alert", "http://127.0.0.1:9000/hook")?);
//!     b.reflex(events, triage);
//!     Server::new().port(3000).nerve(b).serve().await
//! }
//! ```
//!
//! See [docs/NERVE.md](https://github.com/Gilfeather/furnace/blob/main/docs/NERVE.md) for
//! habituation, thresholds, deliberative pathways and the brain.
//!
//! # Configuration
//!
//! The common settings have their own methods: [`Server::host`], [`Server::port`],
//! [`Server::max_batch_size`], [`Server::request_timeout`]. [`Server::configure`] reaches
//! every [`ServerConfig`] setting: limits, timeouts, batching, gRPC (feature `grpc`), TLS
//! (feature `tls`), API keys, rate limits, the admin token for [`Server::reloadable_model`].
//! [`Server::router`] returns the HTTP API as an axum `Router` to mount in your own application. The HTTP API is described in
//! [docs/API.md](https://github.com/Gilfeather/furnace/blob/main/docs/API.md).
//!
//! # Features
//!
//! None are on by default.
//!
//! - `grpc`: the Open Inference Protocol over gRPC as well.
//! - `tls`: HTTPS and mutual TLS, with a pure-Rust crypto provider.
//! - `image`: JPEG / PNG input on `/predict`, with ImageNet preprocessing.
//! - `llm`: serve LLMs with the OpenAI chat and completions API, streaming included, next to
//!   the other models; see the `llm` module.
//! - `test-models`: `model::DummyModel` and `model::DummyTensorsModel`, constant models for
//!   testing and benchmarking the server itself.

pub mod access;
// The `furnace` command's own parts: its allocator, its TOML file and its thread pool. Public
// for the binary (crate furnace-cli) only; not part of the library's API.
#[doc(hidden)]
pub mod allocator;
pub mod api;
pub mod batching;
#[doc(hidden)]
pub mod config;
pub mod error;
mod json_fast;
#[cfg(feature = "llm")]
pub mod llm;
pub mod model;
pub mod nerve;
#[cfg(feature = "image")]
pub mod preprocess;
pub mod telemetry;
pub mod tensor;
#[doc(hidden)]
pub mod threads;
#[cfg(feature = "tls")]
pub mod tls;

pub use api::{Server, ServerConfig};
pub use error::{ApiError, FurnaceError, ModelError, Result};
pub use model::{Model, ModelInfo, TensorModel};
pub use tensor::{DataType, HostTensor, IoSpec, TensorSpec, Values};

// The Rust examples in the Markdown documents compile as doc-tests, so an API change that breaks
// one fails CI. docs/NERVE.md uses the `llm` mock engine; README.md and examples/README.md use
// Burn and compile in furnace-burn, docs/LLM.md uses furnace-candle and compiles there.
#[cfg(all(doctest, feature = "llm"))]
#[doc = include_str!("../docs/NERVE.md")]
mod nerve_md {}
#[cfg(doctest)]
#[doc = include_str!("../docs/API.md")]
mod api_md {}
#[cfg(doctest)]
#[doc = include_str!("../docs/MODELS.md")]
mod models_md {}
#[cfg(doctest)]
#[doc = include_str!("../docs/PERFORMANCE.md")]
mod performance_md {}
#[cfg(doctest)]
#[doc = include_str!("../docs/SERVERLESS.md")]
mod serverless_md {}
#[cfg(doctest)]
#[doc = include_str!("../CONTRIBUTING.md")]
mod contributing {}
