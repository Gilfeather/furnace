# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Kubernetes: a Helm chart (`deploy/helm/furnace`: probes, non-root read-only container, API
  keys, rate limits, TLS from a Secret, ServiceMonitor, autoscaling) and a KServe
  `ClusterServingRuntime` (`deploy/kserve`), both validated in CI
- `cargo binstall furnace-server` installs the release binary, and
  `brew install gilfeather/tap/furnace` installs it on macOS and Linux with the ResNet-18 weights;
  each release updates the tap's formula
- Releases carry build provenance attestations for every binary and image, SHA-256 checksums
  and an SPDX SBOM; `SECURITY.md` says how to report a vulnerability and verify a download
- `docs/MODELS.md`: models known to work, and those that don't yet; `ADOPTERS.md`
- `docs/SERVERLESS.md` and `benches/cold_start.py`: cold start measured (ready in 42 ms, first
  inference in 67 ms) and how to scale to zero on Knative / KServe, Cloud Run and Fly.io

### Fixed
- The release's Docker image was built without TLS; it now has the same features as the
  release binaries (`grpc tls`)

## [0.9.0] - 2026-10-18

Furnace is now a library for serving your own Burn models too, not only ONNX files: a
`forward` function, or any model of named tensors (several inputs and outputs, `int64` token
ids and masks), with everything the server does, batching included.

### Added
- Serve models written in Burn from your own Rust code: `furnace::Server` (builder, or
  `router()` to mount the HTTP API in your app), `Model::from_fn` for a `forward` from one float
  tensor to one (with `/predict` and dynamic batching), and `Model::custom` for any
  `furnace::BurnModel`. `examples/serve_burn_model` serves a CNN and a text classifier with
  integer inputs
- Models with several inputs or outputs, `int64` / `int32` inputs, and dimensions fixed per
  request. They take named tensors on `/v2/models/{model}/infer` (HTTP and gRPC), in
  JSON or binary, return every output or those requested, and describe their tensors in the
  model metadata and `/model/info` (`inputs`, `outputs`). When every input has a fixed shape per
  item, concurrent requests are batched together; an invalid one (wrong shape, an integer the
  backend cannot hold) is refused on its own without failing the rest. Models with one
  `float32` input and output work as before, `/predict` and dynamic batching included

### Changed
- `furnace::BurnModel`, the trait a served model implements, is now `name`, `backend`,
  `inputs`, `outputs` and `run` (plus an optional `fixed_batch`): named tensors in host memory
  instead of a flattened `Tensor<2>`. `predict`, `device`, `get_input_shape`,
  `get_output_shape` and `get_name` are gone

### Fixed
- A request still running on a model while it was reloaded was missing from the model's
  statistics: the old and the new model now share them

## [0.8.0] - 2026-10-18

Furnace grows from a fast model server into a production one. KServe and Triton clients work
against it unchanged, over HTTP and gRPC. One process serves several models and swaps them
without downtime. It can require API keys, rate-limit clients and speak TLS, mutual TLS
included, still in pure Rust. Existing command lines and requests work as before.

### Security
- TLS, in pure Rust (feature `tls`, on in the release binaries and the Docker image):
  `--tls-cert` and `--tls-key` serve HTTPS, and gRPC over TLS; `--tls-client-ca` requires
  client certificates (mutual TLS). Certificates are reloaded without a restart when the files
  change, and a renewal that fails to load keeps the previous one. rustls runs the protocol with
  a small crypto provider over RustCrypto primitives: TLS 1.3, AES-GCM and ChaCha20-Poly1305,
  X25519 / P-256 / P-384, ECDSA and Ed25519 keys. RSA keys are refused (RUSTSEC-2023-0071)
- API keys: with `--api-key` (repeatable, or `FURNACE_API_KEYS`), every route except the health
  probes needs `Authorization: Bearer <key>` or `X-API-Key: <key>`, over HTTP and gRPC alike;
  otherwise `401` / `UNAUTHENTICATED`
- Per-client rate limiting: `--rate-limit <PER_SECOND>` and `--rate-limit-burst <N>`, by API
  key or else by IP address; over the limit, `429` with `Retry-After` (gRPC
  `RESOURCE_EXHAUSTED`). `/metrics` counts the rejections

### Changed
- JSON bodies of the usual shape (`{"input": [...]}`, `{"inputs": [[...], ...]}`) are parsed
  by a dedicated scanner that converts numbers straight to f32 with `fast-float2` and splits
  large arrays across threads; other bodies still go through serde_json. The HTTP overhead of
  a ResNet-18 JSON request drops from about 4.7 ms to about 2.2 ms

### Added
- Several models from one process: `--model-name resnet18,mnist` (or repeated, or a list in the
  config file; `--model-path` can repeat too). `GET /models` lists them,
  `POST /models/{model}/predict` and `GET /models/{model}` address one, and the v2 protocol
  serves each by name. `/predict` and `/model/info` use the first. Metrics are per model
  (`model` label in Prometheus, a `models` list in the JSON)
- Reloading a model without downtime: `POST /admin/models/{model}/reload`, optionally with new
  `weights`, enabled by `--admin-token` (or `FURNACE_ADMIN_TOKEN`). The new model is loaded and
  warmed up beside the serving one and swapped in; requests in flight finish on the old one,
  and a failed reload leaves the old one serving
- `GET /metrics` serves the Prometheus text format to scrapers (requests that accept `text/plain`
  or `application/openmetrics-text`), with request and inference latency histograms; the JSON
  gains estimated p50/p95/p99 latencies and `ready`
- The KServe / Triton Open Inference Protocol over HTTP (`/v2/...`): server and model metadata,
  health, and inference with FP32 tensors, as JSON (flat or nested) or with Triton's binary
  tensor data extension. NVIDIA's `tritonclient.http` works against Furnace unchanged
- The same protocol over gRPC with `--grpc-port` (feature `grpc`, on in the release binaries
  and the Docker image): `inference.GRPCInferenceService` from KServe's proto, compiled with
  the pure-Rust `protox`, so building needs no `protoc`. `tritonclient.grpc` works unchanged
- `GET /readyz` for readiness probes: 503 until a warm-up inference, run in the background at
  startup, has initialized the model; `/healthz` stays the liveness probe

## [0.7.0] - 2026-10-18

CPU inference is about 2× faster again, and the server batches concurrent requests. Over HTTP,
Furnace now matches FastAPI + ONNX Runtime on ResNet-18 one request at a time (8.8 vs 8.6 ms
per image), serves 35-40% more requests per second under concurrent load, and starts in 39 ms
instead of 1.9 s. Existing API calls and CLI options work as before.

### Changed
- ResNet-18 goes from about 15 ms to 8.0 ms per image at batch 1, and from 40.6 ms to 25.1 ms
  at batch 4 (ONNX Runtime: 7.1 and 25.3 ms) on the benchmark laptop. Single-threaded, it goes
  from 47 ms to about 32 ms. Changes in the vendored burn-flex, for `f32` convolutions without
  groups or dilation on x86_64 with AVX2 or AVX-512:
  - Direct convolution without im2col: the input is copied once into a zero-padded buffer (split
    into phase planes for strides above 1), and `std::arch` microkernels read every kernel tap
    as a plain vector load
  - Winograd F(2x2, 3x3) for 3x3 stride-1 layers of up to 128 x 128 channels. The transformed
    filters are cached per weight tensor, holding only a weak reference, so changed weights are
    never served stale
  - Work buffers are reused from a small pool
  - 2D max pooling runs as two separable passes
- Inference uses one thread per physical core by default, like ONNX Runtime, instead of one
  per logical CPU. That costs nothing on bare metal and avoids stragglers on VMs: in a Linux VM
  on the benchmark laptop, ResNet-18 took 24 ms per image with 8 threads and 50 ms with 16
- On ARM (Apple Silicon, AWS Graviton, Ampere), the Winograd convolutions run on NEON
  microkernels, 1.15-1.5x faster than the im2col path on a Neoverse CI runner. The direct
  convolution stays x86-only: on ARM, burn-flex's im2col path is faster than it
- CI runs the vendored burn-flex tests and clippy on x86_64 and aarch64 runners
- The server uses a new global allocator, `furnace::allocator::BlockCache`, that keeps freed
  blocks of 256 KiB and up (at most 128 MiB) for reuse, instead of returning them to the OS and
  page-faulting them back in on every pass. It takes about 15% off batch-1 latency
- Peak private memory rises from about 50 to 78 MB for the server (the reused blocks and the
  Winograd filters); ONNX Runtime takes 139 MB and FastAPI + ONNX Runtime 694 MB
- The README is much shorter; the API reference moved to `docs/API.md` and benchmark details to
  `docs/PERFORMANCE.md`
- Development: `cargo test` no longer needs `--release` (debug builds optimize the dependencies
  and furnace lightly), and rebuilds in about 20 seconds after a change. CI builds its tests
  without fat LTO, checks only that beta and the MSRV compile, and runs coverage on main only
- `total_inferences` in `GET /metrics` is documented as requests that reached the model (it
  always counted requests); `in_flight_inferences` includes requests waiting to be batched,
  and `inference_time_ms` includes that wait

### Added
- `--config <PATH>` (or `FURNACE_CONFIG`) reads options from a TOML file; options on the command
  line override it, and unknown keys are an error. See `furnace.example.toml`
- `--threads <N>` sets the inference threads (also `RAYON_NUM_THREADS`, or `[server] threads`)
- Every response has an `X-Request-Id`: the client's (1-128 visible ASCII characters) or a
  generated UUID. The server's log lines for the request carry it in a `request` span
- A `Dockerfile`: every model in `models/` compiled in, on a distroless non-root image. Each
  release publishes `ghcr.io/gilfeather/furnace:<version>` and `:latest` (linux/amd64 and
  linux/arm64, with ResNet-18 and MNIST); the image there had not been updated since 2025-07
- `docs/openapi.json`, an OpenAPI 3 spec of the HTTP API, checked against the server's actual
  responses by a test
- Dynamic batching in the server. Up to `--batch-workers` forward passes (default 4) run at
  once; when all are busy, queued `/predict` requests are merged into batches of up to
  `--max-batch-size` items (default 8). A batch waits at most `--batch-timeout-us` (default
  2000) for requests still being received, and a lone request never waits.
  `--max-batch-size 1` restores the old behaviour. With 16 concurrent ResNet-18 clients, p95
  latency drops from 207 to 116 ms at the same throughput, about 175 images/s against 129 for
  FastAPI + ONNX Runtime
- `GET /metrics` reports `total_batches`, `average_batch_size` and `max_batch_size`;
  `GET /model/info` statistics gain `batch_count` and `batched_items`
- `Model::predict_many` runs several independent requests as one forward pass
- `examples/http_bench.rs` takes `MODE@N` for N concurrent clients and reports throughput and
  p50/p95 latency
- `benches/compare` also benchmarks FastAPI + ONNX Runtime over HTTP, with the same client as
  the Furnace server

### Fixed
- `furnace --help` showed a run of spaces in the `--model-name` description
- `benches/compare`: the documented steps ran stale engine binaries (`cargo run` rebuilds only
  `compare`), reports said furnace 0.4.0, and the benchmark-report issue template missed the
  input image

## [0.6.1] - 2026-10-17

No changes to furnace itself; release pipeline only.

### Changed
- crates.io releases are published through Trusted Publishing (OIDC) instead of a stored token
- GitHub Actions moved to their Node 24 runtimes (checkout v5, cache v5, action-gh-release v3,
  codecov-action v5, dependency-review-action v5, codeql-action v4)

## [0.6.0] - 2026-10-17

About 2× faster CPU inference again, and the first release on crates.io (as `furnace-server`).
The API and CLI are unchanged from 0.5.0.

### Added
- Security Audit checks RustSec advisories with `cargo deny check advisories` over the resolved
  dependency graph (all features and targets), with no ignored advisories
- Published on crates.io as **`furnace-server`** (the `furnace` crate name belongs to another
  project). The library and the binary are still called `furnace`
- `FURNACE_MODELS_DIR` points the build at the directory of `.onnx` models to compile in
  (default `models/`), so `FURNACE_MODELS_DIR=... cargo install furnace-server` works
- The release workflow publishes to crates.io when the `CARGO_REGISTRY_TOKEN` secret is set
- Issue template for benchmark reports (`benches/compare` results on your hardware)

### Changed
- CPU inference is about 2× faster again: ResNet-18 goes from 27.4 ms to about 14-15 ms per
  image at batch 1 and from 82.2 ms to about 40 ms at batch 4 on the benchmark laptop. Peak
  private memory also drops, from 49 to 46-48 MB in process and from ~72 to ~50 MB for the
  server. Three changes in the vendored burn-flex:
  - Each 2D convolution is split into tasks of output positions × output channels, each filling
    a 128 KiB im2col tile in a reused per-thread buffer and running a single-threaded GEMM,
    instead of one large im2col buffer per call and one multithreaded GEMM
  - Convolutions with many input channels build and multiply their im2col tiles one slice of
    input channels at a time, so every GEMM gets at least 16 output positions. ResNet-18's
    512-channel layers run 36% faster single-threaded (13.9 to 8.9 ms) and ~16% faster on
    16 threads
  - Max pooling no longer computes the argmax index tensor it then discarded

### Removed
- `burn-store`'s PyTorch `.pt` reader (and its `zip` dependency), which furnace never used; it
  only reads Burnpack weights

## [0.5.0] - 2026-10-16

About 2× faster CPU inference. The API and CLI are unchanged from 0.4.0.

### Changed
- CPU inference is about 2× faster: ResNet-18 goes from ~50 ms to ~27 ms per image (batch 4:
  ~94 ms to ~87 ms), and peak private memory drops from ~68 MB to ~49 MB. furnace now uses a
  patched burn-flex (`vendor/burn-flex`, burn-flex 0.22.0 plus one change to
  `src/ops/conv.rs`) whose 2D convolution builds im2col directly from NCHW input instead of
  re-laying out the weight and input on every call. The patch is proposed upstream; the
  vendored copy goes away once a Burn release includes it
- Inference-mode `Conv -> BatchNormalization` pairs are folded into a single Conv at build
  time (`build/optimize.rs`). Set `FURNACE_NO_GRAPH_OPT=1` to build from the original graph

### Fixed
- The new 2D convolution path in the patched burn-flex also caught 3D convolutions with depth
  padding whose single output slice lies entirely in the padding, and returned wrong values for
  them. It now only handles true 2D convolutions (found in upstream review)

## [0.4.0] - 2026-10-15

Migration to Burn 0.22 and a correctness pass over the server. Performance figures in
earlier releases were measured against a dummy fallback model and should be disregarded.

Highlights:
- Any ONNX model in `models/` is servable with no hand-written Rust
- ~5 MB server binary; weights live in a separate, memory-mapped `.bpk` file
- ResNet-18 at ~50 ms per image on a laptop CPU (multithreaded + SIMD), compared against
  tract and ONNX Runtime in `benches/compare`
- Binary request I/O (~1 ms overhead per image) and optional JPEG/PNG input
- No Python anywhere in the repository

### Changed
- **BREAKING**: Upgraded to Burn 0.22. ONNX import moved from `burn-import` to `burn-onnx`;
  the CPU backend is now `burn-flex` (replaces `burn-ndarray`)
- **BREAKING**: Cargo feature `burn-import` renamed to `onnx` (enabled by default)
- **BREAKING**: File-based models now use Burnpack (`<name>.bpk` + `<name>.json`) instead of
  MessagePack `.mpk`. Re-export old models with Burn ≤0.21 + `burn-store`, or regenerate them
- **BREAKING**: Rust edition 2024, MSRV raised to 1.95
- `--backend` now selects the device at runtime (`cpu`, plus `wgpu`/`metal`/`cuda` when built
  with the matching feature); only backends compiled into the binary are accepted
- Built-in model weights are loaded at startup from a separate `<name>.bpk` file
  (`--weights`, or found next to the executable / in `models/`). The server binary stays
  ~5 MB. Previously the binary read weights from an absolute path in the build directory, so
  it only worked on the machine that built it
- ResNet-18 `input_spec.shape` is now `[3, 224, 224]` (per item, without the batch dimension)
- `POST /predict` with `inputs` always returns `outputs` (even for a batch of one)
- Inference runs on a blocking thread pool instead of async runtime workers
- Models are shared across requests without a global mutex
- Stats (`/model/info`) are now updated accurately in release builds
- JSON request bodies are parsed in one pass. The derived `flatten` + `untagged` deserializer
  buffered the whole document into an intermediate tree first, roughly doubling parse time
- Requests whose `Content-Type` is neither JSON nor `application/octet-stream` get
  `415 INVALID_REQUEST`
- A JSON body with both `input` and `inputs` is rejected with 422
- Weight files are memory-mapped and used zero-copy by the CPU backend: private memory
  with ResNet-18 loaded drops from ~49 MB to ~4 MB
- `BuiltInModel` is a handle into the generated registry instead of an enum
  (`from_name`, `name`, `available_models` and `create_model` keep their signatures).
  The hand-written `models::ResNet18` wrapper is replaced by the generated
  `models::resnet18::Served`
- Without `--model-name` or `--model-path`, Furnace serves the built-in model if the build
  has exactly one (release builds: ResNet-18, as before) and otherwise asks for `--model-name`
- The CPU backend is multithreaded and uses SIMD: `burn` is now built with its `rayon` and
  `simd` features, which `default-features = false` had turned off. ResNet-18 goes from
  ~79 ms to ~50 ms per image (batch 4: ~267 ms to ~94 ms). Peak private memory rises from
  ~33 MB to ~68 MB (per-thread buffers), and the binary grows by ~0.7 MB

### Added
- `--weights <PATH>` to load built-in model weights from any `.bpk` file (e.g. fine-tuned)
- `embed-weights` Cargo feature for single-file binaries with compiled-in weights
- Release assets include `resnet18.bpk`
- `--model-path` for file-based Burnpack models
- `--max-body-mb` (default 100) and `--request-timeout-ms` (returns 504 on timeout)
- Backpressure: requests beyond `--max-concurrent-requests` get `503 SERVER_BUSY` immediately
- `GET /metrics` with request, error, busy-rejection and in-flight counters. Based on #32
  by @tamish560, adapted to the new API layer
- `GET /health` alias for `/healthz`; health response includes `backend`
- Structured `INVALID_REQUEST` errors for malformed JSON
- End-to-end tests with real ResNet-18 inference, batch consistency and backpressure checks
- CI: MSRV job, wgpu feature check, cargo-deny
- Binary input: `Content-Type: application/octet-stream` with raw little-endian `float32`
  values, optional `X-Batch-Size: N` header for batches
- Binary output: `Accept: application/octet-stream` returns raw little-endian `float32`
  outputs with `X-Batch-Size`, `X-Output-Shape` and `X-Inference-Time-Ms` headers (also
  exposed to browsers through CORS)
- `Model::predict_flat` runs a batch stored in one contiguous buffer, without per-item copies
- `request_decode/*` Criterion benchmarks and `benches/http_bench.py` for HTTP latency
- Any ONNX model in `models/` is servable with no hand-written Rust: `models/foo.onnx` +
  `cargo build --release`, then `furnace --model-name foo`. `build.rs` reads each graph's
  input and output shapes (via `onnx-ir`) and generates a model registry. Supported: one
  `float32` input and one `float32` output, batch dimension first, other dimensions static;
  other graphs are skipped with a `cargo:warning` that says why. Graphs exported with a
  static batch size (e.g. 1) still accept batches, run in chunks of that size
- `furnace --help` and unknown `--model-name` errors list the models in the build;
  names match ignoring case, `-`, `_` and `.`
- MNIST (ONNX Model Zoo `mnist-8`) as a second built-in model in CI and the end-to-end tests
- `benches/compare`: ResNet-18 comparison against tract and ONNX Runtime (latency, memory,
  startup, binary size, top-1 check on a real image). Its own Cargo workspace, so `ort` and
  `tract` never become furnace dependencies
- Image input behind the opt-in `image` Cargo feature: `Content-Type: image/jpeg` or
  `image/png` with one image per request. The server applies ImageNet preprocessing (shorter
  side to 256, bilinear; centre crop 224; ImageNet mean/std; CHW `float32`), so clients don't
  need Python/PIL. Only models with `[3, H, W]` input accept images (others return 415);
  undecodable images return `400 INVALID_IMAGE`
- `examples/http_bench.rs` replaces `benches/http_bench.py` for HTTP latency
  (`cargo run --release --example http_bench`), so the repository has no Python left

### Fixed
- Real ResNet-18 requests (~1.5–3 MB JSON) were rejected with `length limit exceeded` by
  axum's 2 MB default body limit
- Model load failures silently fell back to a dummy model returning constant `0.5` outputs
- Benchmarks measured the dummy fallback model instead of ResNet-18
- Release workflow built binaries without the ResNet-18 model and referenced a removed Dockerfile
- README referenced a `/health` endpoint that did not exist

### Removed
- `--enable-kernel-fusion` / `--enable-autotuning` are now hidden no-ops (they never had an
  effect); Burn configures fusion and autotuning through Cargo features
- Unused runtime dependencies: bollard, plotters, hdrhistogram, statistical, toml, futures
- Obsolete examples: `onnx_direct_inference`, `create_small_resnet_sample`,
  `create_resnet_test_sample`

## [0.3.5] - 2025-07-24

### Fixed
- Fixed ONNX model generation using proper OUT_DIR environment variable
- Resolved Docker build issues by removing Docker support
- Updated CLI examples in README to use --model-name instead of --model-path
- Corrected API response examples to match actual ResNet18 output format
- Fixed performance metrics to reflect actual ResNet18 inference times (~25s)

### Changed
- Simplified CI workflow by removing Docker build job
- Updated README.md with accurate CLI usage and API documentation
- Improved build process reliability for ONNX code generation

### Removed
- Docker support and Dockerfile (focusing on native binary distribution)
- create_sample_model dependency from benchmarks

## [0.3.0] - 2025-01-21

### Added
- ResNet-18 ONNX model support with automatic shape detection
- Comprehensive benchmark suite with Criterion
- Server overhead analysis to separate ONNX inference from processing time
- Secure performance optimizations with SIMD-optimized validation
- Detailed performance breakdown in documentation
- Troubleshooting section in README.md
- ResNet-18 sample data generation examples

### Changed
- **BREAKING**: Replaced MNIST focus with ResNet-18 specialization
- Updated inference time from ~4ms to ~0.17ms (171µs) with optimizations
- Binary size updated to 4.5MB (from 2.3MB) due to enhanced ONNX support
- Complete README.md overhaul with accurate performance metrics
- Improved API documentation with ResNet-18 specific examples

### Performance Improvements
- 50% server processing optimization while maintaining full security
- ONNX inference: ~14µs (pure model execution)
- Server overhead: ~157µs (optimized data processing, validation, tensor conversion)
- Total end-to-end time: ~171µs
- Maximum throughput: 14,300 req/s (8 threads on Intel MacBook Pro 2020)
- Latency percentiles: P50: 149µs, P95: 255µs, P99: 340µs

### Security
- SIMD-optimized input validation for NaN/infinity detection
- Complete input validation maintained while achieving performance gains
- Non-blocking statistics updates to prevent deadlocks
- Memory-efficient tensor operations

### Removed
- Custom benchmark system in favor of Criterion
- MNIST model references and examples
- Outdated performance claims and documentation

### Technical Details
- Test Environment: Intel MacBook Pro 2020
- Compiler: Rust 1.75+ (release mode with full optimizations)
- Model: ResNet-18 ONNX (45MB, 150,528 input values → 1,000 output classes)

## [0.2.0] - Previous Release
- Initial implementation with basic ML inference capabilities
- MNIST model support
- Basic HTTP API endpoints

## [0.1.0] - Initial Release
- Basic project structure
- Core inference server functionality