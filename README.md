# Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/furnace-server)](https://crates.io/crates/furnace-server)
[![Burn](https://img.shields.io/badge/burn-0.22-orange)](https://github.com/tracel-ai/burn)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, fast inference server for [Burn](https://github.com/tracel-ai/burn) models, written
entirely in Rust.

Serve the Burn model you wrote with a few lines of Rust, or drop an ONNX file in `models/` and
Furnace compiles it to Burn at build time (with [burn-onnx](https://github.com/tracel-ai/burn-onnx)).
Either way you get the KServe / Triton protocol over HTTP and gRPC, dynamic batching, metrics,
TLS and more, from a single small binary. No Python, no C++ runtime: the server starts in
milliseconds and uses a fraction of the memory of a Python stack.

| ResNet-18 over HTTP, CPU | Batch 1 | Batch 4 | Ready after | Memory (peak private) | Ships as |
|---|---|---|---|---|---|
| **Furnace** | 8.8 ms | **27.0 ms** | **39 ms** | **78 MB** | **5.6 MB binary** + weights |
| FastAPI + ONNX Runtime | **8.6 ms** | 28.9 ms | 1,863 ms | 694 MB | Python + 102 MB of packages + model |

Median latency on a 16-thread laptop CPU (AMD Ryzen 9 7940HS). In process, Furnace takes
8.0 ms per image against ONNX Runtime's 7.1 ms, and the same 25 ms for four. Details, more
engines and how to reproduce: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Features

- **Serve your own Burn model** from Rust: `Server::new().model(Model::from_fn(...)).serve()`.
- **Or bring an ONNX model**: put `foo.onnx` in `models/`, rebuild, run
  `furnace --model-name foo`. No hand-written Rust.
- **KServe / Triton Open Inference Protocol** over HTTP and gRPC: Triton and KServe clients work
  unchanged. Tensors are FP32, INT64 or INT32, several in and out, in JSON or binary.
- **Several models per process**, each swappable for new weights without downtime.
- **Dynamic batching** of concurrent requests.
- **Pure Rust**, CPU or GPU: `--backend cpu|wgpu|metal|cuda`.
- **Weights stay outside the binary** in a memory-mapped `.bpk` file (or embed them with
  `--features embed-weights`).
- **Plain `/predict`** with JSON or raw `float32`, and optionally JPEG/PNG with ImageNet
  preprocessing (`--features image`).
- **Production basics**: request size limits, input validation, backpressure (`503`),
  timeouts (`504`), Prometheus metrics, readiness after a warm-up, graceful shutdown.
- **Security**: HTTPS and mutual TLS with certificate reload, in pure Rust (`--features tls`);
  API keys; per-client rate limits.

## Install

- **Release binaries** for Linux (x86_64, musl, aarch64), macOS (Intel, Apple Silicon) and
  Windows come with `resnet18.bpk` on the [releases page](https://github.com/Gilfeather/furnace/releases).
  Put both in one directory and run `./furnace --model-name resnet18`.
- **As a library**, to serve your own Burn models: see [below](#your-own-burn-models).
- **crates.io** (package `furnace-server`, command `furnace`). ONNX models are compiled in, so
  point the build at your `.onnx` files:

  ```bash
  FURNACE_MODELS_DIR=/path/to/models cargo install furnace-server
  ```

  Builds from crates.io use Burn's stock CPU backend, which is several times slower than the
  release binaries until Furnace's CPU kernels ship in a Burn release.
- **From source**: below.

## Quick start

Requires Rust 1.95 or later (on Windows, the Visual Studio C++ build tools).

```bash
git clone https://github.com/Gilfeather/furnace.git
cd furnace
mkdir -p models
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o models/resnet18.onnx
cargo build --release
./target/release/furnace --model-name resnet18
```

Send it an image (with a server built using `--features image`), or raw floats:

```bash
curl -O https://github.com/pytorch/hub/raw/master/images/dog.jpg
curl -X POST http://localhost:3000/predict -H "Content-Type: image/jpeg" --data-binary @dog.jpg
# "output" holds 1000 logits; the largest is class 258 (Samoyed)

cargo run --example resnet18_sample_data      # writes resnet18_single_sample.json
curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" \
  --data-binary @resnet18_single_sample.json
```

Or serve two models written in Burn, with no ONNX file at all:
`cargo run --release --example serve_burn_model` (see [below](#your-own-burn-models)).

## Your own Burn models

Furnace is also a library (from version 0.9). Add it next to Burn; the ONNX import is not needed
for this:

```toml
[dependencies]
furnace-server = { version = "0.9", default-features = false }
burn = "0.22"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

A model that takes one float tensor and returns one is served by its `forward` as it is:

```rust
use burn::tensor::Tensor;
use furnace::{BackendKind, Model, Server};

#[tokio::main]
async fn main() -> furnace::Result<()> {
    let device = BackendKind::Cpu.device();
    let mut net = DigitNet::new(&device); // your Burn Module
    furnace::weights::load_burnpack(&mut net, "digits.bpk".as_ref())?;

    let model = Model::from_fn("digits", BackendKind::Cpu, &[1, 28, 28], &[10],
        move |images: Tensor<4>| net.forward(images))?;
    Server::new().port(3000).model(model).serve().await
}
```

It then answers `POST /predict` and `POST /v2/models/digits/infer`, batching concurrent requests
together. A model with several inputs or outputs, or integer ones (token ids, masks), implements
`furnace::BurnModel` to name its tensors and is served on `/v2/models/{name}/infer`, batched too
when its inputs have fixed shapes.
[`examples/serve_burn_model`](examples/serve_burn_model/main.rs) runs both kinds:
`cargo run --release --example serve_burn_model`. `Server::router()` gives an axum router to
mount in your own app, and `Server::configure` reaches every `ServerConfig` setting: batching,
limits, gRPC, TLS, API keys and the rest of the options below.

## ONNX models

Every `models/<name>.onnx` becomes a built-in model at build time:

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
saying why. If conversion fails, [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
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
| `--batch-workers <N>` | `4` | Forward passes running at once; beyond this, concurrent requests are batched ([details](docs/API.md#dynamic-batching)) |
| `--max-batch-size <N>` | `8` | Most items per batched pass; `1` turns batching off |
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
~80 MB with ResNet-18's weights):

```bash
docker build -t furnace .
docker run --rm -p 3000:3000 furnace                         # serves resnet18
docker run --rm -p 3000:3000 furnace --model-name mnist --threads 4
```

`--build-arg FEATURES=image` adds JPEG/PNG input. The server stops cleanly on `SIGTERM`.

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
