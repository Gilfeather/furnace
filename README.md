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
- 🖼️ **Built-in ResNet-18**: ImageNet classifier, ~75–90 ms per image on a laptop CPU
- ⚡ **Binary I/O**: send raw `float32` (`application/octet-stream`) instead of JSON; ~1 ms of
  request overhead per ResNet-18 image
- 🚦 **Backpressure**: bounded concurrency with immediate `503` instead of unbounded queues
- 🛡️ **Hardened API**: body size limit, input validation, timeouts, security headers,
  graceful shutdown

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

`build.rs` converts every `models/*.onnx` into Rust code plus a Burnpack weight file. The
server binary is `target/release/furnace` (~5 MB). When run from the build machine it finds
the generated `resnet18.bpk` automatically.

### 3. Run

```bash
./target/release/furnace --model-name resnet18 --port 3000
```

```
INFO 🔥 Starting furnace inference server version="0.4.0"
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
| `-m, --model-name <NAME>` | `resnet18` | Built-in model compiled into the binary |
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
cp target/release/build/furnace-*/out/models/resnet18.bpk dist/
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

### Errors

Errors share one shape:

```json
{ "error": "Input validation failed: expected shape [3, 224, 224], got [100]", "code": "INPUT_VALIDATION_FAILED", "details": { "expected_shape": [3, 224, 224], "received_size": 100, "batch_item": 0 }, "timestamp": "..." }
```

| Status | `code` | Cause |
|---|---|---|
| 400 | `INVALID_REQUEST` | Malformed JSON, or `X-Batch-Size` not an integer |
| 422 | `INVALID_REQUEST` | Valid JSON without exactly one of `input`/`inputs`, or non-numeric values |
| 415 | `INVALID_REQUEST` | `Content-Type` is neither `application/json` nor `application/octet-stream` |
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

Any `models/<name>.onnx` is converted at build time; the conversion is skipped with a
build warning if `burn-onnx` doesn't support the graph. ResNet-18 is wired up out of the box.

To serve another ONNX model:

1. Put it in `models/`, e.g. `models/mobilenetv2.onnx`, and run `cargo build --release`.
   Look for `cargo:warning` lines if generation fails.
2. In `build.rs`, declare its cfg: `println!("cargo:rustc-check-cfg=cfg(model_mobilenetv2)");`
3. In `src/models/mod.rs`, include the generated module and add a `BurnModel` wrapper,
   following the `ResNet18` implementation:

   ```rust
   #[cfg(model_mobilenetv2)]
   pub mod mobilenetv2 {
       include!(concat!(env!("OUT_DIR"), "/models/mobilenetv2.rs"));
   }
   ```

   The generated `Model` exposes `new(&Device)` (load weights with `weights::load_burnpack`) and
   `forward(...)`.
4. Add a variant to `BuiltInModel` and handle it in `from_name`, `available_models` and
   `create_model`.

Tips:

- `burn-onnx` supports ONNX opsets 1–24. If conversion fails, try
  [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) first.
- Inspect generated code in `target/release/build/furnace-*/out/models/`.

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

Measured on an AMD Ryzen 9 7940HS (8C/16T), Windows 11, Rust 1.99, `--release`, CPU backend.

Criterion, in-process `Model::predict_batch` (median):

| Benchmark | Batch 1 | Batch 4 / 32 | Throughput |
|---|---|---|---|
| ResNet-18 (`resnet18/*`) | 91 ms | 294 ms (×4) | ~11 img/s single, ~14 img/s batched |
| Server overhead only (`overhead/*`, ResNet-sized I/O) | 0.31 ms | 1.18 ms (×4) | |
| MLP 784→128→10 (`mlp/*`) | 14 µs | 109 µs (×32) | ~290k items/s batched |

End to end over HTTP on localhost, ResNet-18, release binary, keep-alive client
(`benches/http_bench.py`, median of 30 requests). *I/O overhead* is end-to-end time minus
the server's `inference_time_ms`: sending, receiving and decoding the request and response.
Inference alone varies by ±10 ms between runs, so compare the overhead column.

| Request | Body | End to end | I/O overhead |
|---|---|---|---|
| JSON, 1 image (0.4.0 before this change) | 1.7 MB | 102 ms | 14 ms |
| JSON, 1 image | 1.7 MB | 91–96 ms | 6 ms |
| Binary (`application/octet-stream`), 1 image | 0.6 MB | 79–80 ms | 1 ms |
| JSON, batch 4 (0.4.0 before this change) | 6.7 MB | 334 ms | 44 ms |
| JSON, batch 4 | 6.7 MB | 299–323 ms | 21 ms |
| Binary, batch 4 | 2.4 MB | 285–297 ms | 3 ms |

In process (`request_decode/*`), turning one ResNet-18 body into floats takes 4.0 ms for
JSON (about half the time of the previous derived deserializer) and 27 µs for binary. For latency-sensitive clients,
send binary.

Other numbers:

- **Startup**: ~0.45 s to healthy, including loading the 46 MB weight file
- **Binary size**: ~5 MB (+ 46 MB `resnet18.bpk`)
- **Memory**: ~52 MB working set but only ~4 MB private memory after inference. The weight
  file is memory-mapped and tensors use its pages directly, so the 46 MB of ResNet-18 weights
  are shared, file-backed pages that the OS can drop and reload under pressure and that
  multiple server processes share. Reading the weights into the heap instead used ~49 MB
  private (measured with `Process.PrivateMemorySize64` after 3 `/predict` calls).

Reproduce:

```bash
cargo bench --bench inference_benchmark
```

```bash
# With a release server running on port 3000 (stdlib-only Python)
python benches/http_bench.py 3000 json1 json4 bin1 bin4
```

The `overhead/*` group runs a constant-output model with ResNet-sized I/O, so it measures
validation and tensor conversion alone.

## 🛠️ Development

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --release            # unit + end-to-end tests (needs models/resnet18.onnx)
```

Tests run in `--release` because ResNet-18 inference in a debug build is slow.

### Layout

```
src/
├── main.rs          # CLI and startup
├── api.rs           # HTTP routes, limits, backpressure
├── model.rs         # Model wrapper: validation, batching, stats
├── backend.rs       # Runtime backend -> Burn Device
├── burn_model.rs    # File-based MLP (Burnpack)
└── models/mod.rs    # Built-in models generated from ONNX
build.rs             # models/*.onnx -> Rust code + .bpk weights
```

## 🔧 Troubleshooting

**`unknown built-in model` / resnet18 missing**: `models/resnet18.onnx` wasn't present at
build time. Download it (Quick Start step 1), then `cargo build --release`.
`touch build.rs` forces regeneration.

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
- [ONNX Model Zoo](https://github.com/onnx/models) for ResNet-18
