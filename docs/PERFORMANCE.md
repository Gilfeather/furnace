# Performance

## Measured on GitHub's runners

The numbers below come from one laptop. The same comparisons also run on GitHub's own hosted
runners, x86_64 (`ubuntu-latest`) and Arm (`ubuntu-24.04-arm`), every week, on every release
and on demand: the [Benchmarks workflow](https://github.com/Gilfeather/furnace/actions/workflows/bench.yml).
Open its latest run: the *Ratios, all runners* job summary has, for each runner:

- requests/s of Furnace over a hand-written axum server for every Burn model below, and over
  FastAPI (1 and 4 uvicorn workers) for the echo model;
- FastAPI's time to first token over Furnace's, streaming LLM tokens ([`benches/llm`](../benches/llm));
- start-up (ready after start, and the release binary's cold start with ResNet-18) and memory.

Each runner's own job summary has the CPU model and core count (`lscpu`) and the full tables;
the JSON reports are the run's artifacts. Every figure is the median of three runs. Shared
runners are noisy and their CPUs vary from run to run, so read the ratios between servers
measured on the same runner, not the absolute numbers. To run it on a fork:
`gh workflow run bench.yml`.

README.md's numbers (the server table, the sentences under it and the nervous system's
figures) are not typed by hand: they sit between `<!-- bench:NAME:start -->` and
`<!-- bench:NAME:end -->` markers and come from this workflow's x86_64 runner. Run it by hand
with *update_readme* ticked (Actions → Benchmarks → Run workflow, or
`gh workflow run bench.yml -f update_readme=true`): its last job rewrites them with
[`benches/readme-numbers.sh`](../benches/readme-numbers.sh) and opens a pull request from a
`bench/readme-<date>` branch. The weekly and release runs leave README.md alone. The repository
has to allow it once, in Settings → Actions → General → *Allow GitHub Actions to create and
approve pull requests*. A pull request opened by the workflow starts no CI: close and reopen it
to run the checks. From a run's `bench-x86_64` artifact, `benches/readme-numbers.sh DIR` does
the same locally (it needs jq).

## Serving a Burn model

Furnace is a server, so the question is what it adds around a model, and how it compares with
the other ways to put one behind HTTP. [`benches/burn`](../benches/burn) serves the same models
three ways, each in its own process:

- **Furnace**, through furnace-burn (`from_fn`, or a `BurnModel` for the two-input
  text model), with default settings. Requests use the KServe / Triton v2 protocol with binary
  tensors.
- **Hand-written axum**: one route taking raw tensor bytes and calling `forward` on
  `spawn_blocking`. About the least a server can do, and what a Burn user writes without
  Furnace.
- **FastAPI** under uvicorn, with 1 and 4 workers, for the model that returns its input (Python
  cannot run the Burn models).

The models are plain Burn `Module`s with random weights:

- *echo*, which returns its input, so only the server is measured;
- *digits*, a small CNN (`[1, 28, 28]` -> 10, 0.05 ms per image);
- *ResNet-18*, written in Burn (47 ms per image);
- *encoder*, a 4-layer transformer text classifier taking two INT64 inputs of 128 tokens
  (10 ms).

N clients, each on its own keep-alive connection, send batch-1 requests back to back. Medians of
three runs on an AMD Ryzen 9 7940HS (16 threads), Windows 11, Furnace with stock burn-flex 0.22,
as a crate depending on `furnace-server` gets it.

**16 clients: requests/s, latency p50 / p95**

| Model | Furnace | Hand-written axum | FastAPI, 4 workers | FastAPI, 1 worker |
|---|---|---|---|---|
| echo | **85,670**, 0.17 / 0.27 ms | 88,203, 0.17 / 0.27 ms | 24,931, 0.55 / 0.90 ms | 7,238, 2.10 / 2.89 ms |
| digits | 51,967, 0.27 / 0.52 ms | 54,682, 0.26 / 0.49 ms | | |
| ResNet-18 | 61, 254 / 366 ms | 61, 253 / 366 ms | | |
| encoder | **418**, 38 / 46 ms | 412, 38 / 48 ms | | |

| Echo model | Furnace | Hand-written axum | FastAPI, 4 workers | FastAPI, 1 worker |
|---|---|---|---|---|
| Ready after start | **30 ms** | 32 ms | 3,063 ms | 1,597 ms |
| Peak private memory | **6 MB** | 6 MB | | 565 MB |
| 1 client: requests/s, p50 | **12,203**, 0.07 ms | 9,868, 0.10 ms | 4,587, 0.21 ms | 3,817, 0.25 ms |

What this shows:

- **On real models Furnace runs within a few percent of the hand-written server**, while adding
  request IDs, metrics, input checks, backpressure and the Triton protocol. Its own work costs
  about 7 µs more per request: called in process one request at a time, the echo model takes
  17 µs through Furnace's router and 10 µs through the hand-written one. That is 0.07% of the
  encoder's time per request, and shows only on the smallest models (digits, 5% fewer requests).
- **Against FastAPI, as servers**: 3.4 times the requests of 4 uvicorn workers at a third of the
  latency, ready in 30 ms instead of seconds, in 6 MB instead of hundreds.

**Batching.** With `--max-batch-size 8` (two runs; the default is off):

| 16 clients | Batching off (default) | Batching on |
|---|---|---|
| ResNet-18 | 61 req/s, p95 366 ms | **66 req/s, p95 316 ms** |
| encoder | **418 req/s, p95 46 ms** | 314 req/s, p95 88 ms |

ResNet-18 runs a batch of 4 in about twice the time of one image, so merging requests serves
more; the encoder takes nearly 4 times as long for 4, so merging only makes requests wait. Hence
the default: off, matching a hand-written server, with batching one flag away for the models
it helps.

**The engine.** These numbers use stock burn-flex. This repository's binaries and Docker image
use a patched copy ([`furnace-burn/vendor/burn-flex`](../furnace-burn/vendor/README.md)) with
faster convolutions: the same Burn ResNet-18 takes 12 ms per image instead of 47 ms. A crate
depending on `furnace-burn` gets stock burn-flex until those changes reach a Burn release.

```bash
cd benches/burn
cargo build --release
target/release/furnace-burn-bench                          # all models, 1, 4 and 16 clients
target/release/furnace-burn-bench --max-batch 8            # with batching on
target/release/furnace-burn-bench overhead echo            # the server's own work, in process
# the patched engine:
cargo build --release --config 'patch.crates-io.burn-flex.path="../../furnace-burn/vendor/burn-flex"'
```

The FastAPI rows need Python with FastAPI, uvicorn and numpy: the virtual environment of
`benches/compare/fastapi-ort` (below) or `FURNACE_BENCH_PYTHON`.

## ResNet-18 compared with other engines

The same `resnet18.onnx` (resnet18-v1-7 from the ONNX Model Zoo) on Furnace,
[tract](https://github.com/sonos/tract), [ONNX Runtime](https://onnxruntime.ai) (through the
[`ort`](https://crates.io/crates/ort) crate) and FastAPI + ONNX Runtime. The input is PyTorch
Hub's [`dog.jpg`](https://github.com/pytorch/hub/raw/master/images/dog.jpg), and every engine
returns class 258 (Samoyed) with the same logit, 12.42.

| Engine | Threads | Batch 1 median / p95 | Batch 4 median / p95 | Images/s (batch 1 / 4) |
|---|---|---|---|---|
| **Furnace** | all | 8.6 / 10.7 ms | 26.3 / 30.0 ms | 116 / 152 |
| **Furnace server**, HTTP | all | 9.8 / 11.5 ms | 31.9 / 34.7 ms | 102 / 125 |
| ONNX Runtime | default (physical cores) | 7.4 / 7.9 ms | 25.6 / 26.5 ms | 135 / 156 |
| ONNX Runtime | 1 | 26.5 / 28.2 ms | 107.1 / 112.1 ms | 38 / 37 |
| FastAPI + ONNX Runtime, HTTP | default | 8.6 / 9.6 ms | 28.8 / 38.4 ms | 116 / 139 |
| tract | 16 | 19.2 / 20.4 ms | 232.2 / 248.2 ms | 52 / 17 |
| tract | 1 | 47.3 / 53.6 ms | 356.8 / 370.3 ms | 21 / 11 |

| Engine | Load | First inference | Peak working set | Peak private | Size | Pure Rust |
|---|---|---|---|---|---|---|
| **Furnace** | **1 ms** | 20 ms | **130 MB** | **81 MB** | **3.8 MB** + 46 MB `.bpk` | yes |
| **Furnace server** | **29 ms** | 14 ms | **121 MB** | **79 MB** | **8.7 MB** + 46 MB `.bpk` | yes |
| ONNX Runtime | 135 ms | 7 ms | 153 MB | 140 MB | 41.7 MB + 47 MB `.onnx` | no (C++) |
| FastAPI + ONNX Runtime | 1,761 ms | 10 ms | 213 MB | 693 MB | 102 MB of packages + Python + 47 MB `.onnx` | no |
| tract (16 threads) | 180 ms | 21 ms | 166 MB | 194 MB | 18.7 MB + 47 MB `.onnx` | yes |

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
builds with fat LTO. Measured 2026-10-19. Versions: Furnace 0.9.0 on Burn 0.22, tract-onnx 0.23.8, `ort` 2.0.0-rc.14 with ONNX Runtime 1.31, and for the FastAPI row Python 3.14.3, FastAPI 0.143.3, uvicorn 0.54.0 and ONNX Runtime 1.31.0. Each engine runs in its own process: 10 warm-up runs, then 50
timed runs per batch size. Batch 4 is the same image four times. Tables show the median of
three such runs. Runs on this laptop vary by about ±10%, so compare engines within a table
rather than across releases.

### Reproduce

The harness is its own Cargo workspace, so `ort` and `tract` never become Furnace dependencies.

```bash
mkdir -p furnace-burn/models
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o furnace-burn/models/resnet18.onnx
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

## Cold start

From launching the server to its answers, ResNet-18, median of 10 runs:

| | Ready (`/readyz` 200) | First inference answered |
|---|---|---|
| Furnace binary | 29 ms | 40 ms |
| Furnace in Docker | 427 ms | 458 ms |
| FastAPI + ONNX Runtime | 1,761 ms | |

Docker's own start dominates its row (a container that only prints the version takes 722 ms
here). `furnace-burn-bench cold-start` reproduces both Furnace rows (see
[SERVERLESS.md](SERVERLESS.md), which covers scaling to zero).

## Where Furnace's speed comes from

Furnace's CPU backend is [burn-flex](https://github.com/tracel-ai/burn) with faster
convolutions and pooling, vendored in
[`furnace-burn/vendor/burn-flex`](../furnace-burn/vendor/README.md):

- **Direct convolution, no im2col.** The input is copied once into a zero-padded buffer (split
  into phase planes for strides above 1), so every kernel tap is a plain vector load. AVX2 and
  AVX-512 microkernels written with `std::arch` broadcast the weights from their original
  layout. The block shape is picked per layer to waste the fewest vector lanes.
- **Winograd F(2x2, 3x3)** for the 3x3 layers of up to 128 channels, which needs 2.25× fewer
  multiplications. On ARM it runs on NEON microkernels; the direct convolution stays x86-only,
  because on ARM the `gemm` crate's im2col path is faster. The transformed filters are computed once and cached. Wider layers stay
  direct: their transformed filters no longer fit in cache.
- **Max pooling** in two separable passes, without computing argmax indices it would discard.
- **Reused allocations.** Each operator allocates its output, and system allocators return
  blocks that large to the OS, so the next pass page-faults them back in. Furnace's global
  allocator keeps freed blocks of 256 KiB and up (at most 128 MiB) for reuse.
- **BatchNorm folded into convolutions** at build time (`FURNACE_NO_GRAPH_OPT=1` turns this off).

History on this benchmark at batch 1 (16 threads): 0.4.0 50 ms, 0.5.0 26 ms, 0.6.0 15 ms,
0.7.0 8.0 ms, 0.9.0 8.6 ms (the same compute path: runs vary by about ±10%).

### Remaining gap to ONNX Runtime

ONNX Runtime's kernels (MLAS) run ResNet-18 single-threaded at about 135 GFLOP/s, close to
this CPU's peak, so the remaining gap is per-core work. Next steps, roughly by payoff:

1. Winograd F(4x4, 3x3) for the narrow layers (fewer multiplications again).
2. A faster first 7x7 convolution, which has only 3 input channels.
3. Faster max pooling.
4. Cheaper large batches. Batch 4 takes 26 ms against 8.6 ms for batch 1, a gain of only 1.3×,
   so the server's dynamic batching adds no throughput over running passes side by side
   (see below).

GPU backends (`--features wgpu|metal|cuda`) work but are not tuned; on the integrated Radeon
780M, wgpu took 41 ms per image (0.4.0).

## Concurrent load

ResNet-18 from ONNX on the patched CPU backend, measured with Furnace 0.8, when batching was on
by default. N clients, each on its own keep-alive connection, send ResNet-18 requests (one
image, binary body and response) back to back. Throughput is over 40 timed requests per client after a
warm-up; latency percentiles are over all of them. Medians of six runs; the servers ran in a
different order in each run, with a 10 s pause between them, because a laptop that has just
run a heavy load is slower for a while.

| Clients | Furnace, batching on | Furnace, batching off (now the default) | Furnace, 1 batch at a time | FastAPI + ONNX Runtime |
|---|---|---|---|---|
| 1 | 113 req/s, 8.9 / 9.9 ms | 117 req/s, 8.6 / 10.3 ms | 112 req/s, 8.9 / 10.5 ms | 125 req/s, 8.0 / 9.2 ms |
| 2 | 150 req/s, 12.6 / 19.1 ms | 149 req/s, 12.2 / 23.3 ms | 114 req/s, 16.9 / 24.4 ms | 129 req/s, 15.7 / 17.8 ms |
| 4 | 170 req/s, 23 / 28 ms | 177 req/s, 22 / 27 ms | 134 req/s, 30 / 35 ms | 126 req/s, 32 / 35 ms |
| 8 | 174 req/s, 44 / 63 ms | 179 req/s, 44 / 59 ms | 138 req/s, 57 / 62 ms | 128 req/s, 63 / 69 ms |
| 16 | 174 req/s, 91 / **116** ms | 173 req/s, 69 / 207 ms | 155 req/s, 103 / 111 ms | 129 req/s, 126 / 137 ms |
| 32 | 176 req/s, 176 / **236** ms | 172 req/s, 71 / 435 ms | 153 req/s, 210 / 219 ms | 124 req/s, 262 / 281 ms |

Latencies are p50 / p95 (p90 for one client). The configurations:

- *Batching on*: `--batch-workers 4 --max-batch-size 8`. Up to four passes run side by side;
  beyond that, queued requests are batched (see [API.md](API.md#dynamic-batching)).
- *Batching off*: `--max-batch-size 1`, every request runs its own pass at once. Now the
  default, because for some models batching costs throughput
  ([above](#serving-a-burn-model)).
- *1 batch at a time*: `--batch-workers 1`, pure dynamic batching with a single model worker.
- *FastAPI + ONNX Runtime*: [`server.py`](../benches/compare/fastapi-ort/server.py), which
  runs one inference at a time.

What this shows:

- **Under load Furnace serves 35-40% more requests than FastAPI + ONNX Runtime** (about 175
  against 127 per second from 4 clients up), with lower p95 latency at every client count
  from 4 up.
- On this CPU, throughput tops out at about 175 images/s however requests are grouped. Two
  batch-1 passes side by side (about 13 ms for two images over HTTP) beat one batch-2 pass
  (about 17 ms), so merging requests too early lowers throughput, as the *1 batch at a time*
  column shows. With batching on, Furnace therefore merges only once four passes are running.
- What batching buys is predictable latency. Without a limit every request starts at once and
  they all compete for the cores: with 16 or 32 clients, half the requests finish quickly
  while others wait several times longer (p95 207 and 435 ms). With batching on, p95 stays
  within 1.3-1.4x of p50 (116 and 236 ms).
- A lone request is not delayed: one client sees the same latency in every Furnace
  configuration, within run-to-run noise.

```bash
cargo run --release --example http_bench -- 3000 bin1o bin1o@4 bin1o@16
```

`MODE@N` runs N concurrent clients. On a GPU, where one large batch is much cheaper than
several small passes, try `--batch-workers 1` with a larger `--max-batch-size`.

## Request overhead

Over HTTP on localhost, binary bodies (`application/octet-stream`) add about 1 ms per ResNet-18
image on top of inference, and JSON about 6 ms (a 1.7 MB body of text). Decoding one request
body in process takes 27 µs as binary and 4 ms as JSON. Sending a JPEG
(`--features image`, 1546 × 1213) adds about 30 ms of decoding and resizing.

```bash
cargo bench -p furnace-burn --bench inference_benchmark --features onnx   # in-process criterion benches
cargo run --release --example http_bench -- 3000 json1 json4 bin1 bin4   # against a running server
```

## How the harnesses count

The four harnesses (`benches/burn`, `benches/llm`, `benches/nerve` and `benches/compare`) take
their measuring from one crate, [`benches/common`](../benches/common/src/lib.rs), so a figure
means the same thing in every table:

- **Percentiles** (p50, p95, p99) are nearest-rank: the smallest sample with at least that
  fraction of all samples at or below it, `sorted[⌈p·n⌉ − 1]`. Never interpolated, so always a
  measured value. `benches/compare` already counted this way; `benches/burn`, `llm` and
  `nerve` took `sorted[round((n − 1)·p)]` up to 0.10.0. Over the thousands of requests of a
  load run the two land on neighbouring samples; with few samples the new p50 is the lower of
  the two middle ones (one client streaming 4 times in `benches/llm`: the 2nd of 4, was the
  3rd). `benches/compare`'s server-reported time per request is now that same p50, so it can
  be subtracted from the median column (it was the upper middle sample).
- **Medians** of repeated runs (`--runs N`) and of `benches/burn`'s in-process timings are the
  middle value, or the mean of the two middle values for an even count. The in-process timings
  were the upper middle sample up to 0.10.0.
- **Peak private memory** is Windows' peak commit charge. Linux has no peak private counter, so
  there it is the anonymous resident memory (`RssAnon`) read after the load; `benches/nerve`'s
  servers reported their peak resident memory (`VmHWM`, shared pages included) instead before
  0.10.x.
- **Ready** is the time from launching a server to its first 2xx `/healthz`, polled every 5 ms
  with a 5 ms connect first (on Windows a refused connection takes about 0.5 s to report).
- A server is **stopped** with its whole process tree on Windows (`taskkill /T /F`), and with
  `SIGTERM` first elsewhere, on which uvicorn stops its workers. `benches/llm` and
  `benches/compare` killed it without `SIGTERM`, and `benches/compare` left FastAPI's
  interpreter running on Windows.

The tables above were measured with the earlier definitions.
