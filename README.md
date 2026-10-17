# Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/furnace-server)](https://crates.io/crates/furnace-server)
[![Burn](https://img.shields.io/badge/burn-0.22-orange)](https://github.com/tracel-ai/burn)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, fast HTTP inference server for ONNX models, written entirely in Rust.

Furnace compiles your ONNX model into Rust at build time (with
[burn-onnx](https://github.com/tracel-ai/burn-onnx)) and serves it from a single ~5 MB binary.
No Python, no C++ runtime: the server starts in milliseconds and uses a fraction of the memory
of a Python stack.

| ResNet-18 over HTTP, CPU | Batch 1 | Batch 4 | Ready after | Memory (peak private) | Ships as |
|---|---|---|---|---|---|
| **Furnace** | 8.8 ms | **27.0 ms** | **39 ms** | **78 MB** | **5.6 MB binary** + weights |
| FastAPI + ONNX Runtime | **8.6 ms** | 28.9 ms | 1,863 ms | 694 MB | Python + 102 MB of packages + model |

Median latency on a 16-thread laptop CPU (AMD Ryzen 9 7940HS). In process, Furnace takes
8.0 ms per image against ONNX Runtime's 7.1 ms, and the same 25 ms for four. Details, more
engines and how to reproduce: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Features

- **Bring your own ONNX model**: put `foo.onnx` in `models/`, rebuild, run
  `furnace --model-name foo`. No hand-written Rust.
- **Pure Rust**, CPU or GPU: `--backend cpu|wgpu|metal|cuda`.
- **Weights stay outside the binary** in a memory-mapped `.bpk` file (or embed them with
  `--features embed-weights`).
- **JSON or raw `float32`** requests, and optionally JPEG/PNG with ImageNet preprocessing
  (`--features image`).
- **Production basics**: request size limits, input validation, backpressure (`503`),
  timeouts (`504`), `/metrics`, graceful shutdown.

## Install

- **Release binaries** for Linux (x86_64, musl, aarch64), macOS (Intel, Apple Silicon) and
  Windows come with `resnet18.bpk` on the [releases page](https://github.com/Gilfeather/furnace/releases).
  Put both in one directory and run `./furnace --model-name resnet18`.
- **crates.io** (package `furnace-server`, command `furnace`). Models are compiled in, so point
  the build at your `.onnx` files:

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

## Your own models

Every `models/<name>.onnx` becomes a built-in model at build time:

```bash
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx" -o models/mnist.onnx
cargo build --release
./target/release/furnace --model-name mnist
```

A model needs one `float32` input and one `float32` output, each with the batch as the first
dimension and every other dimension fixed (e.g. `[N, 3, 224, 224]`). Models that don't fit,
or that burn-onnx can't convert (it supports opsets 1–24), are skipped with a build warning
saying why. If conversion fails, [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
often helps. `furnace --help` lists the models in a build.

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Inference |
| `GET /healthz` | Health check |
| `GET /model/info` | Input/output shapes and statistics |
| `GET /metrics` | Request, error and latency counters |

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
| `-m, --model-name <NAME>` | the only built-in model | Model to serve |
| `-w, --weights <PATH>` | `<name>.bpk` next to the binary, then in `models/` | Weight file |
| `-b, --backend <BACKEND>` | `cpu` | `cpu`, or `wgpu` / `metal` / `cuda` if built with that feature |
| `--host <HOST>` / `-p, --port <PORT>` | `127.0.0.1` / `3000` | Listen address |
| `--max-concurrent-requests <N>` | `100` | Beyond this, requests get `503` |
| `--max-body-mb <MB>` | `100` | Request size limit |
| `--request-timeout-ms <MS>` | none | Beyond this, requests get `504` |
| `--batch-workers <N>` | `4` | Forward passes running at once; beyond this, concurrent requests are batched ([details](docs/API.md#dynamic-batching)) |
| `--max-batch-size <N>` | `8` | Most items per batched pass; `1` turns batching off |
| `--batch-timeout-us <US>` | `2000` | Longest a batch waits for requests still arriving; a lone request never waits |
| `--threads <N>` | one per physical core | Inference threads (also `RAYON_NUM_THREADS`) |
| `-l, --log-level <LEVEL>` | `info` | `RUST_LOG` sets per-module levels; `FURNACE_ENV=production` switches to JSON logs |
| `--config <PATH>` | | Read options from a TOML file (also `FURNACE_CONFIG`); the command line overrides it. See [`furnace.example.toml`](furnace.example.toml) |

`--model-path <PATH>` serves a Burn MLP saved as `<PATH>.bpk` + `<PATH>.json` instead (see
`examples/basic_mnist_create.rs`).

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

The [`Dockerfile`](Dockerfile) compiles every model in `models/` into a small image (distroless,
non-root, ~80 MB with ResNet-18's weights):

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
