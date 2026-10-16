# 🔥 Furnace

[![Build Status](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml/badge.svg)](https://github.com/Gilfeather/furnace/actions/workflows/ci.yml)
[![Burn](https://img.shields.io/badge/burn-0.22-orange)](https://github.com/tracel-ai/burn)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/Gilfeather/furnace/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Gilfeather/furnace?style=social)](https://github.com/Gilfeather/furnace/stargazers)

**A Rust HTTP inference server for ONNX and Burn models, with zero Python dependencies.**

Furnace compiles ONNX models into native Rust code at build time with
[`burn-onnx`](https://github.com/tracel-ai/burn-onnx) and serves them over a small JSON API.
The model is yours to choose; the server stays a ~5 MB binary with a runtime-selectable
CPU/GPU backend.

## ✨ Features

- 🦀 **Pure Rust**: ONNX graphs become Rust code at build time; no ONNX runtime, no Python
- 🪶 **Lightweight**: ~5 MB server binary; model weights live in a separate `.bpk` file
  (swap in fine-tuned weights without rebuilding, or embed them with `--features embed-weights`)
- 🔀 **Runtime backend selection**: `--backend cpu|wgpu|metal|cuda` (Burn 0.22 `Device` API)
- 📦 **Bring your own ONNX model**: drop `foo.onnx` in `models/`, rebuild, run
  `furnace --model-name foo`. No hand-written Rust
- 🖼️ **ResNet-18 out of the box**: ImageNet classifier, ~14 ms per image on a laptop CPU
- ⚡ **Binary I/O**: send raw `float32` (`application/octet-stream`) instead of JSON; ~1 ms of
  request overhead per ResNet-18 image
- 📷 **Image input** (opt-in `--features image`): POST a JPEG or PNG and the server does the
  ImageNet resize, crop and normalization, so clients need no Python or NumPy
- 🚦 **Backpressure**: bounded concurrency with immediate `503` instead of unbounded queues
- 🛡️ **Hardened API**: body size limit, input validation, timeouts, security headers,
  graceful shutdown

## 📥 Install

- **Prebuilt binaries** for Linux (x86_64, musl, aarch64), macOS (x86_64, Apple Silicon) and
  Windows, plus `resnet18.bpk`, are on the [releases page](https://github.com/Gilfeather/furnace/releases).
  Put the binary and the `.bpk` side by side and run `./furnace --model-name resnet18`.
- **From crates.io** (the package is `furnace-server`; the command is `furnace`). Models are
  compiled in at build time, so point the build at a directory of `.onnx` files:

  ```bash
  FURNACE_MODELS_DIR=/path/to/models cargo install furnace-server
  ```

  crates.io builds use upstream `burn-flex`, without the convolution patch in `vendor/`, so CPU
  inference is slower than the release binaries until that change ships in a Burn release.
- **From source**: see Quick Start below.

## 🚀 Quick Start

### Prerequisites

- **Rust 1.95+** (`rustup update stable`)
- On Windows: Visual Studio C++ Build Tools (the MSVC linker)
- ~50 MB disk space for the ResNet-18 ONNX model

### 1. Clone and download the model

```bash
git clone https://github.com/Gilfeather/furnace.git
cd furnace
mkdir -p models
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o models/resnet18.onnx
```

### 2. Build

```bash
cargo build --release
```

`build.rs` converts every `models/*.onnx` into Rust code plus a Burnpack weight file and
registers it as a built-in model (see [Built-in models](#built-in-models-from-onnx)). The
server binary is `target/release/furnace` (~5 MB). When run from the build machine it finds
the generated `resnet18.bpk` automatically.

### 3. Run

```bash
./target/release/furnace --model-name resnet18 --port 3000
```

```
INFO 🔥 Starting furnace inference server version="0.6.0"
INFO 📦 Loading model model_name=Some("resnet18") backend=cpu
INFO ✅ Model loaded successfully model_name=resnet18 input_shape=[3, 224, 224] output_shape=[1000] backend=cpu model_type=onnx
INFO 🚦 Concurrency limit set to 100 inferences
INFO ✅ Server running on http://127.0.0.1:3000
```

### 4. Call the API

```bash
# Generate sample request bodies (resnet18_single_sample.json, resnet18_batch_sample.json)
cargo run --example resnet18_sample_data

curl http://localhost:3000/healthz
curl http://localhost:3000/model/info

curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_single_sample.json

curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  --data-binary @resnet18_batch_sample.json
```

## ⚙️ CLI Options

| Option | Default | Description |
|---|---|---|
| `-m, --model-name <NAME>` | the only built-in model | Built-in model compiled from `models/<NAME>.onnx`; `--help` lists them. Required when the build has more than one |
| `-w, --weights <PATH>` | see below | Weight file (`.bpk`) for the built-in model |
| `--model-path <PATH>` | | File-based Burn model: `<PATH>.bpk` + `<PATH>.json` (conflicts with `--model-name`) |
| `-b, --backend <BACKEND>` | `cpu` | Compute backend; only backends compiled into the binary are accepted |
| `--host <HOST>` | `127.0.0.1` | Bind address (`0.0.0.0` for all interfaces) |
| `-p, --port <PORT>` | `3000` | Port (1024–65535) |
| `--max-concurrent-requests <N>` | `100` | Concurrent inferences; extra requests get `503 SERVER_BUSY` |
| `--max-body-mb <MB>` | `100` | Maximum request body size |
| `--request-timeout-ms <MS>` | none | Inference timeout; exceeded requests get `504 INFERENCE_TIMEOUT` |
| `-l, --log-level <LEVEL>` | `info` | `error`, `warn`, `info`, `debug`, `trace` |

Without `--weights`, Furnace looks for `<name>.bpk` next to the executable, then in
`models/`, then in the build output directory. A release install is therefore just two
files side by side: `furnace` and `resnet18.bpk`.

Set `FURNACE_ENV=production` for JSON-structured logs. `RUST_LOG` overrides the log filter.

## 📦 Deploying

```bash
cargo build --release
mkdir -p dist
cp target/release/furnace dist/
cp target/release/build/furnace-server-*/out/models/resnet18.bpk dist/
./dist/furnace --model-name resnet18
```

If you'd rather ship a single file, build with `--features embed-weights`. The weights are
then compiled into the binary, so it grows by the size of the model.

The weight file is memory-mapped, not copied into the heap, so don't edit it in place
while the server runs. On Linux and macOS, write the new file elsewhere and rename it over
the old one; the running process keeps the old weights until it restarts. Windows locks the
file while it is mapped, so stop the server first or point `--weights` at a new file.

## 🔀 Backends

The CPU backend ([burn-flex](https://github.com/tracel-ai/burn), pure Rust) is always
available. GPU backends are opt-in Cargo features:

```bash
cargo build --release --features wgpu    # Vulkan / DX12 / Metal via WebGPU
cargo build --release --features metal   # Apple Metal (native MSL)
cargo build --release --features cuda    # NVIDIA CUDA

./target/release/furnace --model-name resnet18 --backend wgpu
```

## 🌐 API

### `GET /healthz` (alias `GET /health`)

```json
{ "status": "healthy", "model_loaded": true, "backend": "cpu", "uptime_seconds": 42, "timestamp": "2026-10-14T03:34:47Z" }
```

### `GET /metrics`

Server counters for monitoring:

```json
{
  "uptime_seconds": 3600,
  "total_requests": 1520, "total_errors": 12, "rejected_busy": 9,
  "in_flight_inferences": 2, "max_concurrent_inferences": 100,
  "total_inferences": 1508, "average_inference_time_ms": 84.2,
  "last_inference_at": "2026-10-14T05:01:12Z",
  "timestamp": "2026-10-14T05:01:13Z"
}
```

`total_requests` and `total_errors` count every `/predict` request, including those rejected
before inference (400, 413, 503). `rejected_busy` is the 503 subset.

### `GET /model/info`

```json
{
  "model_info": {
    "name": "resnet18",
    "version": "1.0.0",
    "input_spec":  { "shape": [3, 224, 224], "dtype": "float32", "min_value": null, "max_value": null },
    "output_spec": { "shape": [1000], "dtype": "float32", "min_value": null, "max_value": null },
    "model_type": "onnx",
    "backend": "cpu",
    "created_at": "2026-10-14T03:34:41Z",
    "model_size_bytes": 0
  },
  "stats": {
    "inference_count": 7, "success_count": 7, "error_count": 0,
    "total_inference_time_ms": 1130.9, "average_inference_time_ms": 161.6,
    "min_inference_time_ms": 73.3, "max_inference_time_ms": 386.9,
    "last_inference_time": "2026-10-14T03:34:47Z"
  },
  "timestamp": "2026-10-14T03:34:48Z"
}
```

### `POST /predict`

Inputs are flattened per item. For ResNet-18 that is `3 × 224 × 224 = 150,528` floats in CHW
order, normalized with the usual ImageNet mean/std.

Single item:

```json
// request
{ "input": [0.12, -0.53, ...] }
// response
{ "output": [-1.59, -0.17, ...], "status": "success", "inference_time_ms": 74.8, "timestamp": "...", "batch_size": 1 }
```

Batch (up to 1000 items):

```json
// request
{ "inputs": [[...], [...], [...]] }
// response
{ "outputs": [[...], [...], [...]], "status": "success", "inference_time_ms": 260.3, "timestamp": "...", "batch_size": 3 }
```

Outputs are raw logits; apply softmax client-side for probabilities.

#### Binary input and output

For large inputs, skip JSON and send raw little-endian `float32` values with
`Content-Type: application/octet-stream`. The body is the flattened items back to back
(`batch × 150,528 × 4` bytes for ResNet-18). This avoids formatting and parsing ~10 bytes
of text per value, and the body is ~2.5× smaller.

| Request header | Meaning |
|---|---|
| `Content-Type: application/octet-stream` | Body is raw `float32` (little-endian) |
| `X-Batch-Size: N` | Optional. Body holds `N` items and the JSON response uses `outputs`. Without it the body is one item and the response uses `output`, as with `input` |
| `Accept: application/octet-stream` | Optional, works with JSON requests too. Respond with raw little-endian `float32` outputs (`batch × output_size`) instead of JSON |

A binary response carries its metadata in headers: `X-Batch-Size`, `X-Output-Shape` (per item,
comma separated, e.g. `1000`) and `X-Inference-Time-Ms`. Errors are always JSON.

```python
import numpy as np, requests

batch = np.random.randn(4, 3, 224, 224).astype("<f4")   # little-endian float32
r = requests.post("http://localhost:3000/predict", data=batch.tobytes(), headers={
    "Content-Type": "application/octet-stream",
    "X-Batch-Size": "4",
    "Accept": "application/octet-stream",
})
logits = np.frombuffer(r.content, dtype="<f4").reshape(4, 1000)
```

```bash
# Raw float32 file with one 3x224x224 image
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.f32
```

#### Image input

Build with `--features image` and POST a JPEG or PNG directly, one image per request:

```bash
cargo build --release --features image

curl -O https://github.com/pytorch/hub/raw/master/images/dog.jpg
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: image/jpeg" \
  --data-binary @dog.jpg
# argmax of "output" is 258 (Samoyed)
```

| Request header | Meaning |
|---|---|
| `Content-Type: image/jpeg` or `image/png` | Body is one encoded image. The response uses `output`, as with `input`; `Accept: application/octet-stream` works too |

The server applies standard ImageNet (torchvision) preprocessing: convert to RGB, resize the
shorter side to 256 with a bilinear filter (longer side truncated), crop the centre 224 × 224,
scale to `[0, 1]`, normalize with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`,
and lay out as CHW `float32`. For a model with input shape `[3, H, W]` the crop is `H × W`
and the shorter side becomes `min(H, W) × 256 / 224`. Models with any other input shape
reject images with 415.

Images count against `--max-body-mb` like any other body. Decoding is limited to 16,384
pixels per side and 128 MB of decoded pixels, and it runs off the async runtime, sharing the
`--max-concurrent-requests` slots with inference. `inference_time_ms` excludes decoding.

Without the feature, `image/*` requests get 415. The feature adds the
[`image`](https://crates.io/crates/image) crate with only its JPEG and PNG decoders,
about +0.4 MB to the release binary (5.5 MB to 5.9 MB on Windows x86-64).

### Errors

Errors share one shape:

```json
{ "error": "Input validation failed: expected shape [3, 224, 224], got [100]", "code": "INPUT_VALIDATION_FAILED", "details": { "expected_shape": [3, 224, 224], "received_size": 100, "batch_item": 0 }, "timestamp": "..." }
```

| Status | `code` | Cause |
|---|---|---|
| 400 | `INVALID_REQUEST` | Malformed JSON, or `X-Batch-Size` not an integer |
| 422 | `INVALID_REQUEST` | Valid JSON without exactly one of `input`/`inputs`, or non-numeric values |
| 415 | `INVALID_REQUEST` | `Content-Type` is not `application/json`, `application/octet-stream` or (with `--features image`) `image/jpeg` / `image/png`; or an image sent to a model whose input isn't `[3, H, W]` |
| 400 | `INVALID_IMAGE` | Image body that fails to decode, or exceeds the decoding limits |
| 400 | `INVALID_INPUT` | Empty input |
| 400 | `INPUT_VALIDATION_FAILED` | Wrong number of values (or bytes, for binary input) for the model |
| 400 | `INVALID_INPUT_VALUE` | NaN or infinite value |
| 400 | `BATCH_TOO_LARGE` | More than 1000 items |
| 413 | `INVALID_REQUEST` | Body larger than `--max-body-mb` |
| 503 | `SERVER_BUSY` | All `--max-concurrent-requests` slots in use; retry later |
| 504 | `INFERENCE_TIMEOUT` | Exceeded `--request-timeout-ms` |
| 500 | `INFERENCE_FAILED` | Model error |

## 🖼️ Models

### Built-in models (from ONNX)

Drop the `.onnx` file in `models/` and rebuild:

```bash
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx" -o models/mnist.onnx
cargo build --release
./target/release/furnace --model-name mnist
```

At build time every `models/<name>.onnx` is converted to Rust code plus `<name>.bpk` weights,
and its input and output shapes are read from the graph. `--model-name` is the file stem
(matched ignoring case, `-`, `_` and `.`), and `furnace --help` lists the models in the build.
Weights are found the same way as ResNet-18's (`--weights`, or `<name>.bpk` next to the
binary or in `models/`).

A model is supported when it has:

- exactly one input and one output,
- both `float32` tensors,
- a batch dimension first, with every other dimension static (e.g. `[N, 3, 224, 224]`).

Requests send each item flattened, like ResNet-18 (`1 × 28 × 28 = 784` floats for MNIST);
`GET /model/info` reports the shapes. If the graph was exported with a fixed batch size
(MNIST's input is `[1, 1, 28, 28]`), batches still work: Furnace runs them through the model
in chunks of that size. A graph that declares a dynamic batch but hard-codes it internally
(e.g. a constant `Reshape` to `[1, …]`) fails on batches larger than 1; re-export it with a
fixed batch dimension. Models that don't fit, or that `burn-onnx` can't
convert, are skipped with a `cargo:warning` that says why, and the rest of the build carries
on. CI builds and tests ResNet-18 and
[MNIST](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
from the ONNX Model Zoo.

Tips:

- `burn-onnx` supports ONNX opsets 1–24. If conversion fails, try
  [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) first.
- Inspect generated code in `target/release/build/furnace-server-*/out/models/`.

### File-based Burn models

`--model-path` loads an MLP stored as Burnpack weights plus JSON metadata:

```bash
cargo run --example basic_mnist_create      # writes examples/basic_mnist/model.{bpk,json}
./target/release/furnace --model-path examples/basic_mnist/model
cargo run --example basic_mnist_test
```

Models saved with Burn ≤0.21 recorders (`.mpk`) can't be read by Burn 0.22. Load them with
the older Burn version and export with `burn-store`'s `BurnpackStore`.

## 📊 Performance

### Compared with other engines (ResNet-18)

The same `models/resnet18.onnx` (resnet18-v1-7 from the ONNX Model Zoo) on furnace,
[tract](https://github.com/sonos/tract) and [ONNX Runtime](https://onnxruntime.ai) (through the
[`ort`](https://crates.io/crates/ort) crate). The input is PyTorch Hub's
[`dog.jpg`](https://github.com/pytorch/hub/raw/master/images/dog.jpg), decoded and preprocessed in
Rust (resize to 256, center crop 224, ImageNet normalization). Every engine returns top-1 class
258 (Samoyed) with the same logit, 12.42.

The wgpu row was measured with the 0.4.0 release.

Setup: AMD Ryzen 9 7940HS (8 cores / 16 threads, Radeon 780M iGPU), 30 GB RAM, Windows 11 Pro
(build 26200), Rust 1.99.0, release builds with fat LTO. Versions: furnace (unreleased, after
0.5.0) on Burn 0.22 (burn-flex CPU backend), tract-onnx 0.23.8, `ort` 2.0.0-rc.14 with ONNX
Runtime 1.31 (the prebuilt binaries `ort` downloads). Each engine runs in its own process: 10
warm-up runs, then 50 timed runs per batch size, one request at a time. Batch 4 is the same image
four times. The table is the median of three such runs: single runs on this laptop vary by about
±10%, and ONNX Runtime's batch 1 median ranged from 4.9 to 6.6 ms.

| Engine | Threads | Batch 1 median / p95 | Batch 4 median / p95 | img/s (batch 1 / 4) |
|---|---|---|---|---|
| **furnace**, CPU | all (rayon) | 13.9 / 16.1 ms | 36.6 / 43.0 ms | 72.0 / 109.3 |
| furnace 0.5.0 release, CPU | all (rayon) | 26.3 / 30.6 ms | 87.1 / 101.6 ms | 38.0 / 45.9 |
| furnace 0.4.0 release, CPU | all (rayon) | 49.9 / 56.4 ms | 93.5 / 97.2 ms | 20.0 / 42.8 |
| **furnace** server, HTTP binary I/O | all (rayon) | 13.7 / 17.5 ms | 38.8 / 42.6 ms | 73.1 / 103.0 |
| **furnace**, wgpu (Radeon 780M iGPU) | GPU | 41.4 / 46.5 ms | 137.9 / 149.2 ms | 24.2 / 29.0 |
| tract | 1 | 45.5 / 49.1 ms | 326.0 / 353.3 ms | 22.0 / 12.3 |
| tract | 16 | 18.2 / 20.4 ms | 204.5 / 213.2 ms | 54.8 / 19.6 |
| ONNX Runtime | default (per physical core) | **6.2 / 7.5 ms** | **20.4 / 25.7 ms** | **160.1 / 196.5** |
| ONNX Runtime | 1 | 26.6 / 28.2 ms | 101.3 / 105.1 ms | 37.5 / 39.5 |

| Engine | Load | First inference | Peak working set | Peak private | Binary | Pure Rust |
|---|---|---|---|---|---|---|
| **furnace**, CPU | **1 ms** | 28 ms | **80 MB** | **46 MB** | **3.7 MB** + 46 MB `.bpk` | yes |
| **furnace** server | 26 ms to healthy | 28 ms | 81 MB | 52 MB | 5.5 MB + 46 MB `.bpk` | yes |
| **furnace**, wgpu | 505 ms | 234 ms | 384 MB | 373 MB | 41.1 MB + 46 MB `.bpk` | yes |
| tract (16 threads) | 175 ms | 21 ms | 166 MB | 194 MB | 18.7 MB + 47 MB `.onnx` | yes |
| ONNX Runtime | 122 ms | 8 ms | 151 MB | 137 MB | 41.7 MB + 47 MB `.onnx` | no (C++) |

*Load* is reading the model until it is ready to run; for the server it is the time from
spawning the process to the first successful `/healthz`. *Peak private* is the peak commit
charge, i.e. memory that belongs to this process alone. furnace memory-maps its weights, so the
46 MB of weights count towards the working set but not towards private memory. *Binary* is a
minimal benchmark executable that links only that engine, built with the same profile. ONNX
Runtime is linked statically, but on Windows `ort` ships the DirectML build, which loads an
18.5 MB `DirectML.dll` even on the CPU path. That DLL is included in the 41.7 MB.

**ONNX Runtime is 2.2× faster than furnace at batch 1 and 1.8× faster at batch 4** (3.5× and
3.0× in 0.5.0). furnace beats tract with 16 threads at batch 1 and is 5.6× faster at batch 4.
furnace loads in about a millisecond, has the smallest binary, and uses the least memory. Over
HTTP with binary bodies it adds 1 to 3 ms to in-process inference.

What changed since 0.5.0 (the API is the same; both changes are in the vendored burn-flex):

- **Convolutions split into cache-sized tasks.** The patched burn-flex used to build one im2col
  buffer per convolution (up to 16 MiB) and hand it to one multithreaded GEMM. Allocating,
  zeroing and freeing that buffer was expensive on Windows, whose heap returns large blocks to
  the OS and page-faults them back in, and the GEMM scaled poorly on the 14×14 and 7×7 layers.
  Each convolution is now split into tasks of output positions × output channels; every task
  fills a 128 KiB im2col tile in a per-thread scratch buffer that is reused across calls, and
  runs a single-threaded GEMM on it. Batch 1 went from 27.4 ms to about 14 ms.
- **Max pooling without indices.** burn-flex computed (and threw away) the argmax index tensor
  on every max pool. Skipping it makes ResNet-18's max pool about 40% faster (0.8 to 0.5 ms).

Both burn-flex changes are proposed upstream and live in `vendor/burn-flex` until a Burn release
includes them.

Where the time goes now (batch 1, profiled build): convolutions about 11 ms (im2col and GEMM
together), max pool 0.5 ms, the separate bias pass 0.4 ms, ReLU and the residual Add 0.45 ms
combined, everything else under 0.3 ms. So operator fusion would save little; the convolutions
are what separates furnace from ONNX Runtime:

- **GEMM throughput.** The [`gemm`](https://crates.io/crates/gemm) crate runs ResNet-18's
  convolution shapes at about 100 to 120 GFLOP/s per core, and furnace reaches about 380 GFLOP/s
  on all cores. ONNX Runtime's MLAS uses hand-written convolution kernels on a blocked NCHWc
  layout and never materializes im2col.
- **Arithmetic.** 3×3 stride-1 convolutions are 3.0 of ResNet-18's 3.6 GFLOP. A Winograd
  transform would cut their multiplies by 2.25× to 4×, but it needs the transformed weights cached
  between calls, which burn-flex has no place for yet.

Ideas for closing the gap, roughly in order of payoff per effort:

1. A Winograd path for 3×3 stride-1 convolutions, with the transformed weights cached per weight
   tensor (memory grows by up to 4× for the layers it covers).
2. Batch concurrent requests in the server, since batch 4 costs only about three times as much
   as batch 1.
3. A direct convolution kernel on a blocked layout, as MLAS does.
4. For GPU hosts, the wgpu backend; a discrete GPU or the CUDA backend should do much better
   than the integrated GPU measured here.
5. An optional ONNX Runtime backend (behind a Cargo feature) would give MLAS speed for users
   who accept a C++ dependency, while the default build stays pure Rust.

Reproduce (the harness is its own Cargo workspace, so `ort` and `tract` never become furnace
dependencies):

```bash
curl -L https://github.com/pytorch/hub/raw/master/images/dog.jpg -o benches/compare/dog.jpg
cargo build --release            # the server binary for the HTTP row
cd benches/compare
cargo run --release              # all engines; add --features wgpu for the GPU row
```

`cargo run --release -- --only ort,tract --iters 100 --json results.json` selects engines and
writes the raw reports. Each engine is also its own binary, for example
`target/release/cmp-ort --threads 1`. See [`benches/compare/src/main.rs`](benches/compare/src/main.rs).

### Furnace micro-benchmarks

Measured on the same machine, `--release`, CPU backend.

Criterion, in-process `Model::predict_batch` (median):

| Benchmark | Batch 1 | Batch 4 / 32 | Throughput |
|---|---|---|---|
| ResNet-18 (`resnet18/*`) | 44 ms | 88 ms (×4) | ~23 img/s single, ~46 img/s batched |
| Server overhead only (`overhead/*`, ResNet-sized I/O) | 0.31 ms | 1.18 ms (×4) | |
| MLP 784→128→10 (`mlp/*`) | 9.5 µs | 80 µs (×32) | ~400k items/s batched |

End to end over HTTP on localhost, ResNet-18, release binary, keep-alive client
(`examples/http_bench.rs`, median of 30 requests). *I/O overhead* is end-to-end time minus
the server's `inference_time_ms`: sending, receiving and decoding the request and response.
Inference alone varies by ±10 ms between runs, so compare the overhead column. These rows were
measured before the CPU backend became multithreaded, so their end-to-end times are about
40 ms (batch 1) to 200 ms (batch 4) too high today; the overhead column still applies.
They also come from an earlier Python version of this benchmark, whose JSON has a space
after every comma; the Rust one sends 1.5 / 6.1 MB JSON bodies with the same overheads.
Sending `dog.jpg` (1546 × 1213 JPEG, 0.66 MB) as `image/jpeg` adds ~30 ms (28–40 ms
measured) of decoding and resizing on top of inference.

| Request | Body | End to end | I/O overhead |
|---|---|---|---|
| JSON, 1 image (two-pass parser, pre-release) | 1.7 MB | 102 ms | 14 ms |
| JSON, 1 image | 1.7 MB | 91–96 ms | 6 ms |
| Binary (`application/octet-stream`), 1 image | 0.6 MB | 79–80 ms | 1 ms |
| JSON, batch 4 (two-pass parser, pre-release) | 6.7 MB | 334 ms | 44 ms |
| JSON, batch 4 | 6.7 MB | 299–323 ms | 21 ms |
| Binary, batch 4 | 2.4 MB | 285–297 ms | 3 ms |

In process (`request_decode/*`), turning one ResNet-18 body into floats takes 4.0 ms for
JSON (about half the time of the previous derived deserializer) and 27 µs for binary. For latency-sensitive clients,
send binary.

Other numbers:

- **Startup**: ~40 ms from launch to a healthy `/healthz`, including mapping the 46 MB weight
  file. On Windows, a client that polls the port before the server listens can see ~0.5 s
  instead: a connect to a closed port retries for about that long before it fails.
- **Binary size**: ~5.5 MB (+ 46 MB `resnet18.bpk`)
- **Memory**: ~60 MB working set and ~25 MB private memory after 3 batch-1 `/predict` calls; the
  peak during the batch-4 benchmark above is 81 MB and 52 MB. (The single-threaded build used
  ~52 MB and ~4 MB; the difference is per-thread buffers.) The
  weight file is memory-mapped and tensors use its pages directly, so the 46 MB of ResNet-18
  weights are shared, file-backed pages that the OS can drop and reload under pressure and
  that multiple server processes share. In the single-threaded build, reading the weights into
  the heap instead used ~49 MB private (measured with `Process.PrivateMemorySize64`).

Reproduce:

```bash
cargo bench --bench inference_benchmark
```

```bash
# With a release server running on port 3000
cargo run --release --example http_bench -- 3000 json1 json4 bin1 bin4
# Image input (server built with --features image)
cargo run --release --example http_bench -- 3000 image=dog.jpg
```

The `overhead/*` group runs a constant-output model with ResNet-sized I/O, so it measures
validation and tensor conversion alone.

## 🛠️ Development

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --release            # unit + end-to-end tests (needs resnet18.onnx and mnist.onnx)
cargo test --release --features image   # also image input tests
```

Tests run in `--release` because ResNet-18 inference in a debug build is slow. The
end-to-end tests also serve MNIST as a second built-in model:

```bash
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx" -o models/mnist.onnx
```

### Layout

```
src/
├── main.rs          # CLI and startup
├── api.rs           # HTTP routes, limits, backpressure
├── preprocess.rs    # JPEG/PNG -> ImageNet-normalized CHW floats (`image` feature)
├── model.rs         # Model wrapper: validation, batching, stats
├── backend.rs       # Runtime backend -> Burn Device
├── burn_model.rs    # File-based MLP (Burnpack)
└── models/mod.rs    # Built-in model registry and BurnModel wrapper
build.rs             # models/*.onnx -> Rust code + .bpk weights + registry
build/registry.rs    # Which graphs are servable; registry source generation
```

## 🔧 Troubleshooting

**`unknown built-in model` / model missing**: `models/<name>.onnx` wasn't present at
build time, or it was skipped: look for `skipping model '<name>'` warnings in the build
output. Download it (Quick Start step 1), then `cargo build --release`.
`touch build.rs` forces regeneration.

**`no model selected`**: the build has several built-in models; pick one with `--model-name`.

**`length limit exceeded` / 413**: raise `--max-body-mb`. A ResNet-18 image is 1.5–3 MB of JSON.

**Lots of `503 SERVER_BUSY`**: increase `--max-concurrent-requests`, or batch on the client.

**`backend not available in this build`**: rebuild with the matching feature,
e.g. `--features wgpu`.

**Windows: `link.exe` not found**: install Visual Studio Build Tools with the
"Desktop development with C++" workload.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

- [Burn](https://github.com/tracel-ai/burn) and [burn-onnx](https://github.com/tracel-ai/burn-onnx)
- [ONNX Model Zoo](https://github.com/onnx/models) for ResNet-18 and MNIST
