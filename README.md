# Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/furnace-server)](https://crates.io/crates/furnace-server)
[![Burn](https://img.shields.io/badge/burn-0.22-orange)](https://github.com/tracel-ai/burn)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Furnace serves [Burn](https://github.com/tracel-ai/burn) models over HTTP and gRPC, as an axum
router you mount in your own Rust service or as a standalone server.

## Embed it in your Axum app

Your app keeps its own routes for pre- and post-processing, the database and the rest of your
API. Furnace adds the inference endpoints, with backpressure, metrics, optional dynamic
batching and the KServe / Triton protocol:

```rust
use axum::{Json, Router, routing::get};
use burn::tensor::Tensor;
use furnace::{BackendKind, Model, Server};

// One of your own routes: the class names for the 10 logits /predict returns
async fn labels() -> Json<Vec<String>> {
    Json((0..10).map(|digit| digit.to_string()).collect())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init(); // Furnace logs through `tracing`; without this it prints nothing
    let device = BackendKind::Cpu.device();
    let mut net = Net::new(&device); // your Burn Module
    furnace::weights::load_burnpack(&mut net, "net.bpk".as_ref())?;
    let model = Model::from_fn("digits", BackendKind::Cpu, &[784], &[10],
        move |images: Tensor<2>| net.forward(images))?;

    let app = Router::new()
        .route("/labels", get(labels))
        .merge(Server::new().model(model).router()?);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

```toml
[dependencies]
furnace-server = { version = "0.11", default-features = false }
burn = "0.22"
axum = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread", "net"] }
tracing-subscriber = "0.3"
```

The snippets in this README follow `main`, which is released as 0.11. Until 0.11 is on
crates.io, depend on the repository instead:
`furnace-server = { git = "https://github.com/Gilfeather/furnace", default-features = false }`.

The app now answers `GET /labels` itself and, through Furnace, `POST /predict`,
`POST /v2/models/digits/infer`, `GET /healthz`, `GET /readyz` and `GET /metrics`. Furnace's API
keys and rate limits guard only its own routes (for per-IP rate limits, serve
`app.into_make_service_with_connect_info::<furnace::api::ClientAddr>()`). Models with several
or integer inputs, batching, and every setting are covered [below](#your-own-burn-models).

## Or run it standalone

The `furnace` command serves the same API on its own, from a release binary,
`brew install gilfeather/tap/furnace`, `cargo binstall furnace-server` or Docker
([Install](#install)). No Python, no C++ runtime: it starts in milliseconds and uses a fraction
of the memory of a Python stack.

**Try it in 30 seconds** (ResNet-18 built in):

```bash
docker run --rm -p 3000:3000 ghcr.io/gilfeather/furnace
curl -s localhost:3000/v2/models/resnet18     # inputs, outputs, shapes
```

Then send it tensors with any Triton or KServe client, or JSON to `POST /predict`
([Quick start](#quick-start) below).

Measured as a server, with 16 concurrent clients on a 16-thread laptop CPU (AMD Ryzen 9 7940HS):

| | **Furnace** | Hand-written axum | FastAPI (uvicorn, 4 workers) |
|---|---|---|---|
| Model that returns its input: requests/s, median latency | **85,700, 0.17 ms** | 88,200, 0.17 ms | 24,900, 0.55 ms |
| Transformer text classifier in Burn: requests/s, p95 | **418, 46 ms** | 412, 48 ms | |
| ResNet-18 in Burn: requests/s, p95 | **61, 366 ms** | 61, 366 ms | |
| Ready after start | **30 ms** | 32 ms | 3.1 s (1.6 s with 1 worker) |
| Memory (peak private), returning its input | **6 MB** | 6 MB | 565 MB with 1 worker |

The hand-written server is the minimum a Burn user would write: one axum route calling `forward`
on a blocking thread. Furnace keeps up with it while adding request IDs, metrics, input checks,
backpressure and the Triton protocol, about 7 µs per request. For ONNX models, compiled to Burn,
ONNX Runtime's kernels are faster: ResNet-18 takes 9.8 ms per image over HTTP with Furnace and
8.6 ms with FastAPI + ONNX Runtime, which needs 1.8 s to start and 693 MB. Details and how to
reproduce: [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Starting in milliseconds makes scaling to
zero practical: [docs/SERVERLESS.md](docs/SERVERLESS.md).

## Features

- **Mount it in your own axum app** with `Server::router()`, or serve on its own with
  `Server::new().model(Model::from_fn(...)).serve()`.
- **Serve the Burn model you wrote**: one float tensor in and out through its `forward`, or
  named FP32 / INT64 / INT32 tensors through the `BurnModel` trait.
- **Also: compile an ONNX file in at build time**: put `foo.onnx` in `models/`, rebuild, run
  `furnace --model-name foo` (with [burn-onnx](https://github.com/tracel-ai/burn-onnx)).
- **KServe / Triton Open Inference Protocol** over HTTP and gRPC: Triton and KServe clients work
  unchanged. Tensors are FP32, INT64 or INT32, several in and out, in JSON or binary.
- **LLMs too**, from the library (feature `llm`; the `furnace` command does not serve them):
  the OpenAI chat and completions API with streaming, next to your other models in the same
  server. GGUF files run on the CPU through candle
  ([`furnace-candle`](furnace-candle)), and your own Burn LLM plugs in with prefill and
  decode. See [docs/LLM.md](docs/LLM.md).
- **A nervous system**: receptors, reflex arcs, a brain and effectors, with habituation,
  thresholds, refractory periods and pathway selection by timing, so that most signals never
  reach the LLM and the ones that must act do so in microseconds. The server's own routes run
  on it: every model is a neuron. See [docs/NERVE.md](docs/NERVE.md).
- **Several models per process**, each swappable for new weights without downtime.
- **Dynamic batching** of concurrent requests, for models where it pays (off by default).
- **Pure Rust**, CPU or GPU: `--backend cpu|wgpu|metal|cuda`.
- **Weights stay outside the binary** in a memory-mapped `.bpk` file (or embed them with
  `--features embed-weights`).
- **Plain `/predict`** with JSON or raw `float32`, and optionally JPEG/PNG with ImageNet
  preprocessing (`--features image`).
- **Production basics**: request size limits, input validation, backpressure (`503`),
  timeouts (`504`), Prometheus metrics, readiness after a warm-up, graceful shutdown.
- **Security**: HTTPS and mutual TLS with certificate reload, in pure Rust (`--features tls`);
  API keys; per-client rate limits.

## Your own Burn models

Add Furnace next to Burn; the ONNX import is not needed for this (until 0.11 is released, use
the `git` line [above](#embed-it-in-your-axum-app)):

```toml
[dependencies]
furnace-server = { version = "0.11", default-features = false }
burn = "0.22"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tracing-subscriber = "0.3"
```

A model that takes one float tensor and returns one is served by its `forward` as it is:

```rust
use burn::tensor::Tensor;
use furnace::{BackendKind, Model, Server};

#[tokio::main]
async fn main() -> furnace::Result<()> {
    tracing_subscriber::fmt::init(); // Furnace's startup and request logs
    let device = BackendKind::Cpu.device();
    let mut net = DigitNet::new(&device); // your Burn Module
    furnace::weights::load_burnpack(&mut net, "digits.bpk".as_ref())?;

    let model = Model::from_fn("digits", BackendKind::Cpu, &[1, 28, 28], &[10],
        move |images: Tensor<4>| net.forward(images))?;
    Server::new().port(3000).model(model).serve().await
}
```

It then answers `POST /predict` and `POST /v2/models/digits/infer`, one forward pass per
request. A model with several inputs or outputs, or integer ones (token ids, masks), implements
`furnace::BurnModel` to name its tensors and is served on `/v2/models/{name}/infer`.
[`examples/serve_burn_model`](examples/serve_burn_model/main.rs) runs both kinds:
`cargo run --release --example serve_burn_model`. `Server::router()` gives the same API as an
axum router to mount in your own app ([above](#embed-it-in-your-axum-app)).

**Batching is off by default.** `Server::new().max_batch_size(8)` merges concurrent requests
into shared forward passes once the server is busy (models whose inputs have fixed shapes).
It pays for convolutional networks and on GPUs; for a transformer on a CPU it makes requests
slower, so leave it off there. [docs/API.md](docs/API.md#dynamic-batching) has the details.
`Server::request_timeout(..)` sets the `504` limit, and `Server::configure` reaches every other
`ServerConfig` setting: limits, gRPC, TLS, API keys and the rest of the options below.

## The nervous system

Inference getting faster is worth nothing if the layer that moves signals to and from the
models is the bottleneck. Furnace is that layer: the nervous system
between sensors, small fast models, the large slow model (an LLM, the brain) and actuators.
Receptors extract a small feature from each input and habituate to repeats, so a repeated
stimulus dies before it reaches anything; reflex pathways are direct calls, microseconds from
stimulus to effector; a signal's deadline selects its pathway, so the brain only sees what is
new and not urgent; the brain's output is commands, carried out by effectors (webhooks,
local commands, files, the response) with their own retry, circuit-breaker and rate-limit
loops. Measured against a hand-written server doing the same work: 146 ns per hop, 0.8 µs from stimulus to effector call on a reflex arc, a brain fraction of 0.45 on a stream with half repeats, 47 µs added to a webhook; at 16 clients the fabric server serves 11% fewer requests per second than the bare handler (the server layer around it, not the fabric).

```rust
use std::time::Duration;

use furnace::Server;
use furnace::llm::mock::MockEngine;
use furnace::nerve::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> furnace::Result<()> {
    tracing_subscriber::fmt::init();
    // Stands in for a real LLM (a GGUF file through furnace-candle, or your Burn model);
    // the brain's answer is commands for the effectors
    let engine = MockEngine::new("mock", &[r#"{"effector": "log", "args": {"line": "seen"}}"#]);

    let mut b = Fabric::builder();
    let events = b.receptor(Http::new("events", "/events").text()); // POST /events
    let repeats = Habituation::new(Duration::from_secs(30));
    b.gate(events, Mechanisms::NONE.threshold(0.6).habituation(repeats)); // repeats die here
    let triage = b.neuron(FnNeuron::reflex("triage", |signal, _| {
        if signal.payload.as_text().is_some_and(|t| t.contains("ALERT")) {
            signal.command("alarm", json!({"text": signal.payload.as_text()})) // acts at once
        } else {
            signal // goes on to the brain
        }
    }));
    let brain = b.neuron(Llm::new("brain", engine));
    b.effector(Webhook::new("alarm", "http://127.0.0.1:9000/alarm")?);
    b.effector(FileAppend::new("log", "journal.log"));
    b.reflex(events, triage);
    b.deliberative(triage, brain);
    Server::new().port(3000).nerve(b).serve().await
}
```

```toml
[dependencies]
furnace-server = { version = "0.11", default-features = false, features = ["llm"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde_json = "1"
tracing-subscriber = "0.3"
```

[docs/NERVE.md](docs/NERVE.md) has the concepts, the API, the demo
(`cargo run --release --features llm --example nerve`), how the server's own routes run on
the fabric, and the measurements.

## Install

- **As a library**, to serve your own Burn models: see [above](#embed-it-in-your-axum-app) and
  [below](#your-own-burn-models).
- **Release binaries** for Linux (x86_64, musl, aarch64), macOS (Intel, Apple Silicon) and
  Windows come with `resnet18.bpk` on the [releases page](https://github.com/Gilfeather/furnace/releases).
  Put both in one directory and run `./furnace --model-name resnet18`. With
  [cargo-binstall](https://github.com/cargo-bins/cargo-binstall), `cargo binstall furnace-server`
  fetches the binary for your platform, and on macOS or Linux
  `brew install gilfeather/tap/furnace` installs it with the ResNet-18 weights. Each release
  carries checksums, an SBOM and build provenance you can verify ([SECURITY.md](SECURITY.md)).
- **Docker**: `ghcr.io/gilfeather/furnace`, for x86_64 and ARM64 ([below](#docker)).
- **Kubernetes**: a Helm chart and a KServe serving runtime in [`deploy/`](deploy/README.md).
- **crates.io** (package `furnace-server`, command `furnace`). ONNX models are compiled in, so
  point the build at your `.onnx` files:

  ```bash
  FURNACE_MODELS_DIR=/path/to/models cargo install furnace-server
  ```

  Builds from crates.io use Burn's stock CPU backend, which is several times slower than the
  release binaries until Furnace's CPU kernels ship in a Burn release.
- **From source**: below.

## Quick start

The standalone `furnace` command, built from source with an ONNX model compiled in:

Requires Rust 1.95 or later (on Windows, the Visual Studio C++ build tools).

```bash
git clone https://github.com/Gilfeather/furnace.git
cd furnace
mkdir -p models
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o models/resnet18.onnx
cargo build --release
./target/release/furnace --model-name resnet18
```

Send it an image as JSON floats (`/predict` takes the 150,528 values of one preprocessed image):

```bash
cargo run --example resnet18_sample_data      # writes resnet18_single_sample.json
curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" \
  --data-binary @resnet18_single_sample.json
# "output" holds 1000 logits
```

To send a JPEG or PNG as it is, with the ImageNet preprocessing done by the server, build with
the `image` feature (the release binaries and the Docker image above do not have it):

```bash
cargo build --release --features image
./target/release/furnace --model-name resnet18
curl -O https://github.com/pytorch/hub/raw/master/images/dog.jpg
curl -X POST http://localhost:3000/predict -H "Content-Type: image/jpeg" --data-binary @dog.jpg
# the largest of the 1000 logits is class 258 (Samoyed)
```

Or serve two models written in Burn, with no ONNX file at all:
`cargo run --release --example serve_burn_model` (see [below](#your-own-burn-models)).

## ONNX models

The `furnace` command can also serve ONNX files without any Rust: every `models/<name>.onnx`
becomes a built-in model at build time:

```bash
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx" -o models/mnist.onnx
cargo build --release
./target/release/furnace --model-name mnist
```

Inputs and outputs are `float32`, `int64` or `int32` tensors with the batch as their first
dimension, as many as the model has: an image classifier's one tensor in and out, a text model's token
ids and masks, a detector's several outputs. A model with one `float32` input and one output of
fixed shapes is served on `/predict` too, with dynamic batching; any other takes its named
tensors on [`/v2/models/{model}/infer`](docs/API.md#models-with-several-inputs-or-outputs).
Models that burn-onnx can't convert (it supports opsets 1–24) are skipped with a build warning
saying why. [docs/MODELS.md](docs/MODELS.md) lists the models known to work, and the ones that
don't yet. If conversion fails, [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
often helps. `furnace --help` lists the models in a build.

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Inference, for models with one float tensor in and out |
| `GET /healthz` / `GET /readyz` | Liveness / readiness (after the warm-up inference) |
| `GET /model/info` | Input/output shapes and statistics |
| `GET /models`, `POST /models/{model}/predict` | Several models side by side ([details](docs/API.md#several-models)) |
| `POST /admin/models/{model}/reload` | Swap in a model, e.g. new weights, without downtime ([details](docs/API.md#reloading-a-model)) |
| `GET /metrics` | Counters and latency percentiles as JSON, or the Prometheus format for a scraper |
| `/v2/...` | KServe / Triton Open Inference Protocol: any model, by named tensors; Triton and KServe clients work unchanged |

`/predict` takes each input flattened (ResNet-18: `3 × 224 × 224 = 150,528` floats, CHW,
ImageNet-normalized) and returns raw logits:

```text
{ "input": [0.12, -0.53, ...] }    →  { "output": [-1.59, -0.17, ...], "inference_time_ms": 8.1, ... }
{ "inputs": [[...], [...]] }       →  { "outputs": [[...], [...]], ... }
```

For speed, send `Content-Type: application/octet-stream` with little-endian `float32` values
(and `X-Batch-Size: N` for a batch); add `Accept: application/octet-stream` to get raw floats
back. Every response carries an `X-Request-Id` (yours, or a generated one) that also tags the
server's log lines for that request. Full reference, including errors:
[docs/API.md](docs/API.md); OpenAPI spec: [docs/openapi.json](docs/openapi.json).

## Options

| Option | Default | |
|---|---|---|
| `-m, --model-name <NAME>` | the only built-in model | Model to serve; `a,b` serves several, the first by `/predict` |
| `-w, --weights <PATH>` | `<name>.bpk` next to the binary, then in `models/` | Weight file (one model) |
| `-b, --backend <BACKEND>` | `cpu` | `cpu`, or `wgpu` / `metal` / `cuda` if built with that feature |
| `--host <HOST>` / `-p, --port <PORT>` | `127.0.0.1` / `3000` | Listen address |
| `--max-concurrent-requests <N>` | `100` | Beyond this, requests get `503` |
| `--max-body-mb <MB>` | `100` | Request size limit |
| `--request-timeout-ms <MS>` | none | Beyond this, requests get `504` |
| `--batch-workers <N>` | `4` | With batching on: forward passes running at once; beyond this, concurrent requests are batched ([details](docs/API.md#dynamic-batching)) |
| `--max-batch-size <N>` | `1` | Most items per batched pass; the default leaves batching off. Try `8` for convolutional models and on GPUs |
| `--batch-timeout-us <US>` | `2000` | Longest a batch waits for requests still arriving; a lone request never waits |
| `--threads <N>` | one per physical core | Inference threads (also `RAYON_NUM_THREADS`) |
| `--grpc-port <PORT>` | none | Also serve the KServe / Triton protocol over gRPC (builds with `--features grpc`; release binaries and the Docker image have it) |
| `--admin-token <TOKEN>` | none | Enable the reload API for this bearer token (also `FURNACE_ADMIN_TOKEN`) |
| `--api-key <KEY>` | none | Require a key on everything but health probes; repeatable (also `FURNACE_API_KEYS`) ([details](docs/API.md#api-keys-and-rate-limiting)) |
| `--rate-limit <PER_SECOND>` / `--rate-limit-burst <N>` | none | Per-client request rate (by API key, else IP address); beyond it, `429` |
| `--tls-cert <PEM>` / `--tls-key <PEM>` | none | Serve HTTPS and gRPC over TLS; reloaded when the files change ([details](docs/API.md#tls)) |
| `--tls-client-ca <PEM>` | none | Mutual TLS: only clients with a certificate from this CA |
| `-l, --log-level <LEVEL>` | `info` | `RUST_LOG` sets per-module levels; `FURNACE_ENV=production` switches to JSON logs |
| `--config <PATH>` | | Read options from a TOML file (also `FURNACE_CONFIG`); the command line overrides it. See [`furnace.example.toml`](furnace.example.toml) |

`--model-path <PATH>` serves the sample MLP format (`<PATH>.bpk` + `<PATH>.json`, see
`examples/basic_mnist_create.rs`). For your own Burn architectures, use the
[library](#your-own-burn-models).

## Deploying

A deployment is two files, the binary and the weights:

```bash
cargo build --release
mkdir -p dist
cp target/release/furnace target/release/build/furnace-server-*/out/models/resnet18.bpk dist/
```

The weights are memory-mapped, so replace the file by renaming a new one over it rather than
editing it in place (on Windows, stop the server first). For a single file, build with
`--features embed-weights`.

GPU backends are Cargo features: `--features wgpu` (Vulkan, DX12, Metal), `metal` or `cuda`.

### Docker

Each release publishes an image with ResNet-18 and MNIST built in, for x86_64 and ARM64:

```bash
docker run --rm -p 3000:3000 -p 8001:8001 ghcr.io/gilfeather/furnace:latest   # HTTP 3000, gRPC 8001
docker run --rm -p 3000:3000 ghcr.io/gilfeather/furnace:latest --model-name mnist
```

To compile in your own models, build from the [`Dockerfile`](Dockerfile) (distroless, non-root,
~80 MB with ResNet-18's weights). It compiles every `models/*.onnx` in, and a checkout has none;
these two are what the release image carries:

```bash
mkdir -p models
curl -fsSL https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx -o models/resnet18.onnx
curl -fsSL https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx -o models/mnist.onnx
docker build -t furnace .
docker run --rm -p 3000:3000 furnace                         # serves resnet18
docker run --rm -p 3000:3000 furnace --model-name mnist --threads 4
```

`--build-arg FEATURES="grpc tls image"` adds JPEG/PNG input. The server stops cleanly on `SIGTERM`.

### Kubernetes

```bash
helm install furnace ./deploy/helm/furnace
```

The chart wires the probes to `/healthz` and `/readyz`, runs the container non-root and
read-only, and turns API keys, rate limits, TLS from a cert-manager Secret, Prometheus scraping
and autoscaling on from values. For KServe, [`deploy/kserve`](deploy/kserve) registers Furnace
as a serving runtime for the Open Inference Protocol. Details: [deploy/README.md](deploy/README.md).

## Development

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test             # needs models/resnet18.onnx and models/mnist.onnx
```

Debug builds optimize the dependencies, so `cargo test` needs no `--release` and rebuilds in
seconds after a change. Generated model code is in `target/*/build/furnace-server-*/out/models/`. Furnace ships a
patched copy of Burn's CPU backend in `vendor/burn-flex` with faster convolutions; see
[vendor/README.md](vendor/README.md). Contributions are welcome, see
[CONTRIBUTING.md](CONTRIBUTING.md), and benchmark results from other CPUs especially
([#52](https://github.com/Gilfeather/furnace/issues/52)).

## License

MIT. Built on [Burn](https://github.com/tracel-ai/burn) and
[burn-onnx](https://github.com/tracel-ai/burn-onnx); models from the
[ONNX Model Zoo](https://github.com/onnx/models).
