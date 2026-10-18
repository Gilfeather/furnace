//! Serve models written in Burn, from your own Rust code.
//!
//! ```bash
//! cargo run --release --example serve_burn_model
//! # or load trained weights saved with burn-store:
//! cargo run --release --example serve_burn_model -- --digits-weights digits.bpk
//! ```
//!
//! Two models, both with randomly initialized weights unless you pass a `.bpk` file:
//!
//! - `digits`, a small CNN for 28x28 grayscale images: one float tensor in, one out, so
//!   `Model::from_fn` serves its `forward` as it is, with `/predict` and dynamic batching.
//! - `sentiment`, a text classifier taking token ids and an attention mask (`INT64`): it
//!   implements `BurnModel` to name its tensors, and is served on `/v2/models/sentiment/infer`.
//!
//! ```bash
//! curl localhost:3000/v2/models/sentiment
//! curl -X POST localhost:3000/v2/models/sentiment/infer -H 'Content-Type: application/json' -d '{
//!   "inputs": [
//!     {"name": "input_ids", "datatype": "INT64", "shape": [1, 8], "data": [101, 7, 42, 9, 102, 0, 0, 0]},
//!     {"name": "attention_mask", "datatype": "INT64", "shape": [1, 8], "data": [1, 1, 1, 1, 1, 0, 0, 0]}
//!   ]}'
//! ```

mod models;

use std::path::PathBuf;

use burn::tensor::Tensor;
use furnace::{BackendKind, Model, Server};
use models::{DigitNet, Sentiment, SentimentNet};

fn weights_arg(flag: &str) -> Option<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}

#[tokio::main]
async fn main() -> furnace::Result<()> {
    tracing_subscriber::fmt::init();
    let backend = BackendKind::Cpu;
    let device = backend.device();

    let mut digits = DigitNet::new(&device);
    if let Some(path) = weights_arg("--digits-weights") {
        furnace::weights::load_burnpack(&mut digits, &path)?;
    }
    let mut sentiment = SentimentNet::new(&device);
    if let Some(path) = weights_arg("--sentiment-weights") {
        furnace::weights::load_burnpack(&mut sentiment, &path)?;
    }

    Server::new()
        .port(3000)
        .model(Model::from_fn(
            "digits",
            backend,
            &[1, 28, 28],
            &[10],
            move |images: Tensor<4>| digits.forward(images),
        )?)
        .model(Model::custom(Sentiment {
            net: sentiment,
            backend,
            device,
            seq_len: 8,
        }))
        .serve()
        .await
}
