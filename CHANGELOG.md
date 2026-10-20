# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The server runs on the nervous system: every model is a neuron, every route a receptor.

### Migrating from 0.10
The public API changes below break code written for 0.10; furnace keeps no compatibility
aliases before 1.0. Renames are listed as old name -> new name.

- `Server::router()`, `api::router()` and `api::router_with_models()` return
  `Result<Router>`: an error for a server with nothing to serve, as `serve()` gives (0.10
  panicked). Add `?` or `.expect(..)`:
  `Router::new().merge(Server::new().model(model).router()?)`.
- `Model::warm_up()` returns `furnace::Result<Option<Duration>>` (was a `String` error): the
  model's own error, as `infer` gives.
- `model::DummyModel` and `model::DummyTensorsModel` need the new `test-models` feature. They
  answer constants and are for testing the server; serve a model of your own with
  `Model::from_fn` or `Model::custom`. In tests,
  `furnace-server = { version = "0.11", features = ["test-models"] }` under
  `[dev-dependencies]`.
- `load_model` replaces a missing model with a sample MLP of random weights only for a path
  named exactly `sample_model`, with a warning (0.10: any name containing `sample`, logged at
  `info`). Any other missing path is `ModelError::FileNotFound`; write real files with
  `cargo run --bin create_sample_model`.
- `GET /metrics` JSON: `max_concurrent_inferences` -> `max_concurrent_requests`, the name the
  flag, the configuration and Prometheus use.
- `access::RateLimit` -> `access::ClientRateLimit` (`ServerConfig::rate_limit`), so it is not
  confused with the nervous system's `effector::RateLimit`.
- LLM layer (feature `llm`):
  - `llm::Input` -> `llm::Prompt`: `Prompt::Chat`, `Prompt::Text`. `GenerateRequest::input`
    keeps its name.
  - `llm::SchedulerStats` -> `llm::SchedulerCounters` (live values;
    `LlmEngine::scheduler_stats` keeps its name).
  - `llm::mock::Live` -> `llm::mock::SharedCount`: it counts decode calls too, which are not
    alive.
  - `Event` is `#[non_exhaustive]`: a `match` on it needs a `_` arm. It is no longer `Clone`,
    since an `Error` holds the model's error.
  - `SchedulerConfig` is `#[non_exhaustive]`: build it from `SchedulerConfig::default()` with
    the new `with_workers`, `with_max_waiting` and `with_max_prefill_tokens`, e.g.
    `SchedulerConfig::default().with_max_waiting(64)` for
    `SchedulerConfig { max_waiting: 64, ..Default::default() }`.
  - `EngineError::Failed` holds a `BoxError` (was `String`): the error a `StepModel` returned
    reaches the caller unchanged and can be downcast. Construct it with
    `EngineError::failed("message")` or `EngineError::Failed(error.into())`; match it with
    `EngineError::Failed(e)` and `e.to_string()` for the message. `EngineError` is no longer
    `Clone`; it is `PartialEq` as before (`Failed` compares messages).
- TLS (feature `tls`): `TlsConfigs::load` and `reload_if_changed` return `tls::Error` (was
  `String`), an enum that says which file is wrong and how (`Certificate`, `NoCertificate`,
  `PrivateKey`, `ClientCa`, `CertificateAndKey`, `Setup`); its message is the one 0.10 gave.
- Nervous system (`furnace::nerve`):
  - `FabricBuilder::build()` returns `Result<Fabric, BuildError>`: two neurons of one name, a
    pathway to a neuron of another builder, or a runtime that cannot start is an error (0.10
    panicked; `add` and `neuron` no longer check names themselves). Add `?` or `.expect(..)`.
  - `effector::Webhook::new` and `with_timeout` return `Result<Webhook, BuildError>`
    (`BuildError::Url` for an `https://` or malformed URL; 0.10 panicked):
    `Effector::new(Webhook::new("alarm", url)?)`.
  - `furnace::nerve::wire` is no longer public; it was the outbound client's internals.
  - `effector::ActError` is an enum: `Status(u16)`, `Timeout(String)`, `Unreachable(String)`,
    `Transient(String)`, `Fatal(String)`. `ActError::retry(msg)` and `ActError::fatal(msg)`
    still build the last two; read `error.retryable()` for `error.retry` and `error.to_string()`
    for `error.message`. `Webhook` reports the status, a timeout or an unreachable target, so
    a 429 and a refused connection can be told apart.
  - `Receptor::start` gets a stop signal: the `Afferent` it receives has `is_stopped()`,
    `sleep(duration) -> bool` and `stopped().await`, raised when the fabric is dropped. A
    receptor of your own should sleep with `afferent.sleep(..)` (it returns `false` once
    stopped) or select on `afferent.stopped()`. The built-in `Timer`, `ChannelIn`, `SseIn` and
    `Lines` (its listener closes its port, reconnecting stops) end with the fabric.
  - Renames, old -> new: `receptor::Channel` -> `receptor::ChannelIn`,
    `effector::Channel` -> `effector::ChannelOut`, `receptor::Sse` -> `receptor::SseIn`,
    `effector::Sse` -> `effector::SseOut`, `nerve::Reply` -> `nerve::ReplyHandle`,
    `effector::RateLimit` -> `effector::ActRateLimit`.
  - Metrics JSON (`/metrics` `nerve`, `GET /nerve`): a neuron's `quiet` -> `quiet_not_interested`
    (the not-interested reason, next to `quiet_by_threshold` and the others, not a total);
    pathways', arcs' and effectors' `latency` -> `latency_ms`, an LLM brain's `generation` ->
    `generation_latency_ms`, a model neuron's `forward` -> `forward_latency_ms`. A latency is
    `{"count", "p50", "p95", "p99", "mean", "max"}` in milliseconds (was `p50_us`, `p99_us`,
    `mean_us`, `max_us` in microseconds), the percentiles estimated from a histogram of every
    measurement (`telemetry::Histogram` with the new `FINE_LATENCY_BUCKETS`, 1 µs to 10 s)
    instead of the last 128. `LatencySnapshot` has `count`, `percentiles`, `mean`, `max`.
  - Prometheus: `furnace_nerve_pathway_latency_seconds` and `furnace_nerve_arc_latency_seconds`
    are histograms (`_bucket`, `_sum`, `_count`) instead of `quantile` gauges; query
    `histogram_quantile(0.99, rate(furnace_nerve_pathway_latency_seconds_bucket[5m]))`.
  - `NeuronMetrics::computed` counts a brain's firings too (`brain::Llm`,
    `brain::ModelNeuron`): they fire through `Firing::Compute`. It stayed at 0 for them.

### Added
- `FabricBuilder::efferent(Handle)`: run the motor side on a Tokio runtime of your own
  instead of one the fabric starts.
- `telemetry::Histogram::with_buckets`, `Histogram::count` and `FINE_LATENCY_BUCKETS`.
- `tls::MAX_HANDSHAKES` (1024): TLS handshakes in flight per listener. Past it the listener
  stops accepting until one ends, so clients that connect and never send a ClientHello cannot
  hold an unbounded number of tasks and sockets.

### Changed
- `/predict`, v2 over HTTP and v2 over gRPC share one inference path: the concurrency slot,
  the timeout, the latency of a success and the request counters are taken in one place, and
  one mapping turns an error into an HTTP status and `code`. What changes on the wire:
  - v2 error responses carry the `code` `/predict` gives for the same error
    (`INPUT_VALIDATION_FAILED`, `INVALID_INPUT_VALUE`, `INFERENCE_FAILED`,
    `INFERENCE_TIMEOUT`, `SERVER_BUSY`; `MODEL_NOT_FOUND` for an unknown model or version)
    instead of one picked from the status (`INVALID_REQUEST` for a 500, `NOT_FOUND`).
  - A model error that blames the request (`ModelError::InvalidArgument` too) is a 400
    `INVALID_INPUT_VALUE` on every route, as `FurnaceError::is_client_error` says.
  - gRPC: a model whose pathway is full is `RESOURCE_EXHAUSTED` (was `INTERNAL`), and the
    timeout and panic messages are the HTTP ones.
  - Named tensors are decoded by one function for v2 JSON and gRPC; a raw tensor of the wrong
    size reads "Input 'x' has N bytes of raw data; shape [..] of FP32 needs M" on both (v2 said
    "binary_data_size is .. bytes", gRPC "raw_input_contents has .. bytes"), and v2's
    "binary_data_size is N bytes but M are left in the body" when the body is short.
- A streaming OpenAI request answers its status after the engine's first event, so an engine
  that refuses only once it has looked at the request (`Event::Error` in place of `Started`,
  e.g. a prompt too long for the context) gets its 400 instead of an error payload after a
  200; a stream whose first event does not come within `--request-timeout-ms` is a `504`.
- `start_server_with_models` with no model returns an error instead of panicking.
- The fast JSON parser of `/predict` leaves numbers past `f32`'s range to serde_json, so it
  never accepts what the fallback would answer differently; both end as a 400.
- A Ctrl-C or SIGTERM handler that cannot be installed is logged, and the server keeps
  serving, instead of a panic in the signal task.
- Per request: `/predict`'s binary decoder reads the one header it needs instead of copying
  them all, a JSON batch response is written from the output buffer without a `Vec` per item,
  binary response headers are built without intermediate strings, and the `request` span
  borrows its fields instead of formatting them.
- `/predict`, `/models/{model}/predict`, v2 over HTTP and gRPC, and the OpenAI routes run on
  the fabric: each model is `in:{model}` → `model:{model}` → `serve:respond`, each LLM a brain
  `llm:{llm}` behind `in:{llm}`, all listed in `GET /nerve` with measured latencies. With
  batching off a model's pathway is a reflex (the forward pass on a blocking thread, as
  before); with `--max-batch-size` above 1 it is a deliberative pathway. The server's own work
  per request grew by about 4 µs on a laptop; over HTTP the difference is within noise.
- One batching engine (`src/batching/engine.rs`) behind the server's dynamic batching and
  every deliberative pathway. A brain that panics loses that window's signals instead of its
  pathway's worker. The internal `Batcher` is gone.
- The nervous system is always built; the `nerve` feature is kept and changes nothing.
  `GET /nerve` and the fabric's metrics are always served.
- `Server::nerve` takes a `FabricBuilder` (was a built `Fabric`), so the server can add its
  models and LLMs to it.
- `nerve::Answer` is an enum: an HTTP answer, tensors, a failure, or (feature `llm`) a
  generation's events. `Payload::Prompt` carries a generation request (feature `llm`).
- `HostTensor::batch()` is 0 for a tensor without dimensions (was 1).
- `--request-timeout-ms` covers the OpenAI routes: `504` for a whole answer, an error payload
  and `[DONE]` for a stream whose model goes quiet for that long.
- A refractory neuron admits one of several signals arriving together (compare-and-swap).
- `wire::Client::request` (and so `Webhook`) bounds the whole call, response body included;
  `Process` writes stdin within its timeout; `FileAppend` writes on a thread of its own.
- `Respond` with nobody waiting any more is not a failure, so late answers no longer open its
  breaker.
- The bench harnesses share `benches/common`: one percentile definition (nearest rank, which
  `benches/burn`, `llm` and `nerve` did not use), one median, one memory probe, and one way to
  start and stop a server ([PERFORMANCE.md](docs/PERFORMANCE.md#how-the-harnesses-count)).
  The crates.io package no longer carries `benches/burn`, `llm` and `nerve`.
- TLS: a PKCS#8 private key the server does not take is named by its algorithm, read from the
  key ("RSA keys are not supported", "unsupported EC key curve 1.3.132.0.35"), instead of
  "unsupported PKCS#8 key (RSA?)".
- Build output: no `onnx feature not enabled` warning for `default-features = false` builds,
  and no warning for a checkout without `models/` (only for a `FURNACE_MODELS_DIR` that does
  not exist). The build script no longer reruns on every build when `models/` is missing; a
  `models/` created later is picked up after `touch build.rs`.
- `tokio-stream` is part of the `grpc` feature (was `tls`); `tower` is no longer a
  dependency of the library.
- Prometheus: the `kind` label of `furnace_nerve_pathway_signals_total` is lowercase
  (`reflex`, `deliberative`), as in the JSON metrics and as the neurons' `kind` label already
  was; it was `Reflex` and `Deliberative`. Queries that match the old values need updating.
  The rest of the `furnace_nerve_*` text is unchanged byte for byte.

### Removed
- `ApiError::{InvalidRequest, JsonParsing, ModelNotLoaded}` and
  `CliError::{MissingArgument, InvalidModelPath}`: nothing constructed them.
- `FurnaceError` no longer implements axum's `IntoResponse`; no handler returned it, and its
  codes disagreed with the ones the routes send.

### Fixed
- An ONNX file named after a reserved Rust keyword (`final.onnx`, `override.onnx`, ...)
  generates `model_final` instead of a module that does not compile.
- A batch size a fixed-batch graph cannot run is a `400` on `/predict` too (was `500` with a
  doubled message).
- Warming up a simple model with a fixed batch runs that batch (was always 1).
- An input tensor without dimensions is refused with a clear message.
- A loop through a deliberative pathway is cut after `MAX_HOPS` like any other.
- Chat template rendering and tokenization no longer run on the async runtime.
- A rate limit without API keys logs a warning when the router may be served without
  `ClientAddr`, in which case every client shares one budget.
- docs/openapi.json lists the OpenAI routes, `GET /nerve/effectors/{name}/events`, the
  `llm` and `nerve` fields of `/metrics`, and every error `code`; a test derives the expected
  routes from the router's own list and the codes from the server's. docs/API.md documents
  `furnace_build_info{version}` and `furnace_model_info{model, backend}` (it said
  `furnace_build_info{version, model, backend}`), the missing error codes, and that 401 and 429
  are counted only in `rejected_unauthorized` and `rejected_rate_limited`.

## [0.10.0] - 2026-10-20

Furnace becomes the layer between sensors, models and actuators, not only a server that
answers requests: a nervous system (`nerve`) built from a neuron's mechanisms, an LLM layer
with the OpenAI API, and its own case measured as a server against a hand-written axum
server and FastAPI, on this laptop and on GitHub's runners.

### Added
- The nervous system (feature `nerve`): `furnace::nerve`, the layer between sensors, small
  fast models, an LLM (the brain) and actuators. One `Signal` type (a 64-bit feature, an
  urgency, an intensity, a lazily shared payload, a trace); neurons with per-neuron
  mechanisms (threshold, habituation as a bounded table with exponential decay, refractory
  period); reflex pathways as direct calls on the firing thread and deliberative pathways
  over a summation worker, selected by the signal's remaining deadline against each
  pathway's measured p99, then by weight; `prepare()` for signals with time to spare.
  Receptors (an HTTP route, an in-process channel, a timer, line streams, an upstream SSE
  feed), effectors with their own retry / circuit-breaker / rate-limit / refractory loops
  (the HTTP response, webhooks, local processes, file append, an in-process channel, SSE
  push), brains over any `LlmEngine` or `Model` whose output is commands, `Server::nerve`,
  `GET /nerve`, and metrics (per neuron fired and quiet by reason, per pathway latency,
  `brain_fraction`, stimulus-to-action per reflex arc) in JSON and Prometheus. The demo in
  `examples/nerve`; `benches/nerve` measures it against a hand-written server and FastAPI.
  See docs/NERVE.md
- Serve LLMs (feature `llm`): `POST /v1/chat/completions`, `POST /v1/completions` and
  `GET /v1/models` in the OpenAI format, with SSE streaming and usage, OpenAI-shaped errors,
  cancellation when the client disconnects, 503s at the concurrency limit or a full queue, and
  metrics (time to first token, inter-token latency, queue time, tokens) in Prometheus and in
  `/metrics`. `Server::llm` serves an engine next to the other models. Engines implement
  `furnace::llm::LlmEngine`, or `StepModel` (prefill and decode) to run through furnace's
  scheduler with a `TextPipeline`: chat templates (minijinja), a pure-Rust tokenizer, sampling,
  incremental detokenization and stop strings. Mock engines for tests and benchmarks are in
  `furnace::llm::mock`. See docs/LLM.md
- `furnace-candle`, a workspace crate (not published yet): quantized GGUF models of the llama
  and qwen2 families on the CPU through candle, from local files, with
  `examples/serve_gguf.rs`
- CI fails if a C or C++ crate (cc, ring, aws-lc, Oniguruma, OpenSSL, ...) enters the dependency
  tree of `furnace` with `llm` or of `furnace-candle`
- `benches/llm`: furnace's LLM server layer against a hand-written axum server and FastAPI,
  streaming the same paced mock tokens; furnace adds 0.3 µs per streamed token over the
  hand-written server, and matches its time to first token up to 256 streams (docs/LLM.md)
- `furnace-llm-bench real`: one GGUF file served by furnace + candle, llama.cpp and mistral.rs
  under the same streaming requests. On Qwen2.5-0.5B-Instruct Q4_K_M, furnace + candle runs at
  candle's own speed and matches mistral.rs on one stream; llama.cpp is 4.5 times faster per
  token; mistral.rs 0.9.5 fails most concurrent streams on the CPU (docs/LLM.md)
- `SchedulerConfig::max_prefill_tokens` (512): the scheduler prefills waiting prompts up to this
  many tokens between two decode steps, instead of one prompt per step, and new sequences join
  the decode step right after their prefill. With a batched step model at 256 streams, tokens/s
  go from 2,957 to 11,783 and the time to first token from 3.9 s to 22 ms; the inter-token p99
  from 40.7 ms (two steps) to 22.5 ms
- Kubernetes: a Helm chart (`deploy/helm/furnace`: probes, non-root read-only container, API
  keys, rate limits, TLS from a Secret, ServiceMonitor, autoscaling) and a KServe
  `ClusterServingRuntime` (`deploy/kserve`), both validated in CI
- `cargo binstall furnace-server` installs the release binary, and
  `brew install gilfeather/tap/furnace` installs it on macOS and Linux with the ResNet-18 weights;
  each release updates the tap's formula
- Releases carry build provenance attestations for every binary and image, SHA-256 checksums
  and an SPDX SBOM; `SECURITY.md` says how to report a vulnerability and verify a download
- `docs/MODELS.md`: models known to work, and those that don't yet; `ADOPTERS.md`
- `docs/SERVERLESS.md`: cold start measured (ready in 29 ms, first inference in 40 ms) and how
  to scale to zero on Knative / KServe, Cloud Run and Fly.io. `furnace-burn-bench cold-start`
  measures it
- `benches/burn`: Burn models served through the library API, measured against calling
  `forward` directly and against a minimal hand-written axum server

### Changed
- `GET /models` reports an empty `default_model` on a server that serves only LLMs; `/predict`
  and `/model/info` answer 404 there
- Dynamic batching is off by default (`--max-batch-size 1`, `BatchConfig::default()`). It
  pays only for some models: ResNet-18 on a CPU serves 10-20% more requests with it at 16
  clients, a transformer text encoder about 25% fewer. Pass `--max-batch-size 8` for
  convolutional models and on GPUs
- Less work per request: request ID, CORS and security headers in one layer, no access check
  without API keys or a rate limit, and the per-request log line at `debug` instead of `info`
  (at the command line's default level it cost about 9 µs per request). `tower-http` is no
  longer a dependency
- No Python outside the FastAPI servers the benchmarks compare against: the Homebrew formula is
  rendered by `packaging/homebrew/render.sh`, and cold start is measured by
  `furnace-burn-bench cold-start` instead of `benches/cold_start.py`

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