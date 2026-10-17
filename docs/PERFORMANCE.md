# Performance

## ResNet-18 compared with other engines

The same `models/resnet18.onnx` (resnet18-v1-7 from the ONNX Model Zoo) on Furnace,
[tract](https://github.com/sonos/tract), [ONNX Runtime](https://onnxruntime.ai) (through the
[`ort`](https://crates.io/crates/ort) crate) and FastAPI + ONNX Runtime. The input is PyTorch
Hub's [`dog.jpg`](https://github.com/pytorch/hub/raw/master/images/dog.jpg), and every engine
returns class 258 (Samoyed) with the same logit, 12.42.

| Engine | Threads | Batch 1 median / p95 | Batch 4 median / p95 | Images/s (batch 1 / 4) |
|---|---|---|---|---|
| **Furnace** | all | 8.0 / 9.7 ms | 25.1 / 30.0 ms | 125 / 160 |
| **Furnace server**, HTTP | all | 8.8 / 10.2 ms | 27.0 / 29.6 ms | 114 / 148 |
| ONNX Runtime | default (physical cores) | 7.1 / 7.9 ms | 25.3 / 26.3 ms | 142 / 158 |
| ONNX Runtime | 1 | 27.3 / 28.8 ms | 107.0 / 114.7 ms | 37 / 37 |
| FastAPI + ONNX Runtime, HTTP | default | 8.6 / 9.4 ms | 28.9 / 31.1 ms | 116 / 138 |
| tract | 16 | 20.8 / 22.5 ms | 218.6 / 234.7 ms | 48 / 18 |
| tract | 1 | 46.0 / 53.0 ms | 339.3 / 392.3 ms | 22 / 12 |

| Engine | Load | First inference | Peak working set | Peak private | Size | Pure Rust |
|---|---|---|---|---|---|---|
| **Furnace** | **1 ms** | 18 ms | **130 MB** | **81 MB** | **3.8 MB** + 46 MB `.bpk` | yes |
| **Furnace server** | **39 ms** | 18 ms | **124 MB** | **78 MB** | **5.6 MB** + 46 MB `.bpk` | yes |
| ONNX Runtime | 148 ms | 7 ms | 153 MB | 139 MB | 41.7 MB + 47 MB `.onnx` | no (C++) |
| FastAPI + ONNX Runtime | 1,863 ms | 10 ms | 214 MB | 694 MB | 102 MB of packages + Python + 47 MB `.onnx` | no |
| tract (16 threads) | 197 ms | 21 ms | 166 MB | 194 MB | 18.7 MB + 47 MB `.onnx` | yes |

- *Load* is reading the model until it is ready to run; for the servers it is the time from
  starting the process to the first successful `/healthz`.
- *Peak private* is memory that belongs to the process alone. Furnace memory-maps its weights,
  so the 46 MB of weights count towards the working set but not towards private memory, and
  several server processes share them.
- *Size* is a minimal executable that links only that engine, built with the same profile. On
  Windows, `ort` ships the DirectML build of ONNX Runtime, whose 18.5 MB `DirectML.dll` loads
  even on the CPU path and is included.
- The HTTP rows use the same Rust client (one keep-alive connection, one request at a time,
  binary `float32` bodies). The FastAPI server is
  [`benches/compare/fastapi-ort/server.py`](../benches/compare/fastapi-ort/server.py).

Setup: AMD Ryzen 9 7940HS (8 cores, 16 threads), 30 GB RAM, Windows 11, Rust 1.99, release
builds with fat LTO. Versions: Furnace 0.7.0 on Burn 0.22, tract-onnx 0.23.8, `ort` 2.0.0-rc.14 with ONNX Runtime 1.31, and for the FastAPI row Python 3.14.3, FastAPI 0.143.3, uvicorn 0.54.0 and ONNX Runtime 1.31.0. Each engine runs in its own process: 10 warm-up runs, then 50
timed runs per batch size. Batch 4 is the same image four times. Tables show the median of
three such runs. Runs on this laptop vary by about ±10%, so compare engines within a table
rather than across releases.

### Reproduce

The harness is its own Cargo workspace, so `ort` and `tract` never become Furnace dependencies.

```bash
mkdir -p models
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o models/resnet18.onnx
curl -L https://github.com/pytorch/hub/raw/master/images/dog.jpg -o benches/compare/dog.jpg
cargo build --release            # the server, for the HTTP row
cd benches/compare
# Optional, for the FastAPI + ONNX Runtime row (skipped without it)
uv venv fastapi-ort/.venv
uv pip install -p fastapi-ort/.venv -r fastapi-ort/requirements.txt
cargo build --release            # every engine binary; add --features wgpu for the GPU row
target/release/compare
```

Build before running: `cargo run` would rebuild only `compare`, not the engine binaries it
starts. `target/release/compare --only furnace,ort --json results.json` selects engines and
writes the raw reports; each engine is also its own binary, e.g. `target/release/cmp-ort
--threads 1`. Please share results from other CPUs in
[#52](https://github.com/Gilfeather/furnace/issues/52).

## Where Furnace's speed comes from

Furnace's CPU backend is [burn-flex](https://github.com/tracel-ai/burn) with faster
convolutions and pooling, vendored in [`vendor/burn-flex`](../vendor/README.md):

- **Direct convolution, no im2col.** The input is copied once into a zero-padded buffer (split
  into phase planes for strides above 1), so every kernel tap is a plain vector load. AVX2 and
  AVX-512 microkernels written with `std::arch` broadcast the weights from their original
  layout. The block shape is picked per layer to waste the fewest vector lanes.
- **Winograd F(2x2, 3x3)** for the 3x3 layers of up to 128 channels, which needs 2.25× fewer
  multiplications. The transformed filters are computed once and cached. Wider layers stay
  direct: their transformed filters no longer fit in cache.
- **Max pooling** in two separable passes, without computing argmax indices it would discard.
- **Reused allocations.** Each operator allocates its output, and system allocators return
  blocks that large to the OS, so the next pass page-faults them back in. Furnace's global
  allocator keeps freed blocks of 256 KiB and up (at most 128 MiB) for reuse.
- **BatchNorm folded into convolutions** at build time (`FURNACE_NO_GRAPH_OPT=1` turns this off).

History on this benchmark at batch 1 (16 threads): 0.4.0 50 ms, 0.5.0 26 ms, 0.6.0 15 ms,
0.7.0 8.0 ms.

### Remaining gap to ONNX Runtime

ONNX Runtime's kernels (MLAS) run ResNet-18 single-threaded at about 135 GFLOP/s, close to
this CPU's peak, so the remaining gap is per-core work. Next steps, roughly by payoff:

1. Winograd F(4x4, 3x3) for the narrow layers (fewer multiplications again).
2. A faster first 7x7 convolution, which has only 3 input channels.
3. Faster max pooling.
4. In the server, batching concurrent requests: batch 4 costs well under 4× batch 1.

GPU backends (`--features wgpu|metal|cuda`) work but are not tuned; on the integrated Radeon
780M, wgpu took 41 ms per image (0.4.0).

## Request overhead

Over HTTP on localhost, binary bodies (`application/octet-stream`) add about 1 ms per ResNet-18
image on top of inference, and JSON about 6 ms (a 1.7 MB body of text). Decoding one request
body in process takes 27 µs as binary and 4 ms as JSON. Sending a JPEG
(`--features image`, 1546 × 1213) adds about 30 ms of decoding and resizing.

```bash
cargo bench --bench inference_benchmark                              # in-process criterion benches
cargo run --release --example http_bench -- 3000 json1 json4 bin1 bin4   # against a running server
```
