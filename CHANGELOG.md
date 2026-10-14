# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-14

Migration to Burn 0.22 and a correctness pass over the server. Performance figures in
earlier releases were measured against a dummy fallback model and should be disregarded.

Highlights: ~5 MB server binary with weights in a separate memory-mapped `.bpk` file
(~4 MB private memory with ResNet-18 loaded), and binary request I/O (~1 ms overhead per
ResNet-18 image vs ~6 ms for JSON).

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