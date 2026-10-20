# Maintainability audit (v0.10.0)

A read-only pass over the whole repository at v0.10.0 (commit 749dbc3): `src/`, `build.rs` and
`build/`, `furnace-candle/`, `examples/`, `tests/`, the four benchmark workspaces, `deploy/`,
the CI workflows and `docs/`. About 35,000 lines of Rust were read; every line number below was
checked against the tree with `grep -n` or `sed -n`. Nothing was built or run, so claims about
runtime behaviour come from reading the code, not from observing it; the few places where a run
would settle a question say so.

The second half of this document is a plan: pull-request-sized waves, in order, that do not
touch each other's files and stay out of the batching unification that is in progress in
`src/batching.rs` and the summation worker of `src/nerve/fabric.rs`.

## Scope: what survives and what is replaced

The hand-written serving routes are technical debt and will be rebuilt on the nerve fabric as
receptor → pathway → neuron → effector, after which the handlers are retired. That work runs on
its own branch. The plan in section 12 therefore has **no wave that splits, renames or polishes
these handlers**:

- `src/api.rs`: `/predict` and its decoders, the middleware, admin reload, the metrics routes;
- `src/api/v2.rs` and `src/api/grpc.rs`: the Open Inference Protocol over HTTP and gRPC;
- `src/llm/http.rs` and the HTTP-side tracking in `src/llm/metrics.rs`: the OpenAI routes;
- the glue that serves them: `src/main.rs`'s config plumbing, `src/config.rs`, `docs/API.md`,
  `docs/openapi.json`.

Findings in that code stay in the document, marked **(replaced by the fabric)**: the structural
ones describe what the rebuild should not reproduce, and the correctness ones (wrong status
codes, missing timeouts, the rate limit that needs `ConnectInfo`, untested paths) are
requirements for the receptors and effectors that take over. Section 12.1 collects them as a
checklist for the rebuild.

What survives, and what the waves cover: `src/nerve/*`, `src/batching/*` (after the
unification), `src/model.rs`, `src/tensor.rs`, `src/weights.rs`, `src/allocator.rs`,
`src/error.rs`, the LLM engine side (`src/llm/{mod,scheduler,pipeline,sampling,stop,mock}.rs`,
which the brain neuron keeps using), `src/access.rs`, `src/telemetry.rs`, `src/tls`, `build.rs`
and `build/`, `furnace-candle`, the tests of all of these, CI, the bench harnesses and the docs
that describe surviving code.

Severity: **high** can bite a user or blocks maintenance; **medium** costs time or hides bugs;
**low** is cosmetic. Effort: **S** under an hour, **M** a few hours, **L** a day or more.
Proposals that would change behaviour or the public API say so explicitly; everything else is
a pure refactor.

## Summary

| Area | Lines | Main debt |
|---|---|---|
| Server (`src/api.rs`, `api/v2.rs`, `api/grpc.rs`, `access`, `telemetry`, `config`, `main`), the handlers to be replaced by the fabric | 7,000 | one 3,609-line file; the error-to-status mapping written four times; permit, timing and stats repeated per transport; a rate limit that silently degrades without `ConnectInfo`; no timeout on the OpenAI routes |
| Model core (`model.rs`, `tensor.rs`, `burn_model.rs`, `models/`, `build/`) | 3,500 | two parallel request pipelines inside `Model`; nine `#[allow]`s; five never-constructed error variants; a `/predict` validation error that comes back as a 500 |
| LLM (`src/llm`, `furnace-candle`) | 3,800 | sound layering; template rendering and tokenization on the Tokio runtime; `catch_unwind` that cannot run under `panic = "abort"`; four real-model tests that never run |
| Nerve (`src/nerve`) | 6,800 | two 1,200-line files bundling five or six independent kinds; the hand-written HTTP client is public API; a `Respond` failure mode that opens the breaker for live requests; body reads without a timeout |
| TLS, batching, CI, benches, deploy | 8,000 | feature sets never compiled; doc-tests never run; three Prometheus renderers; twelve helpers copied across four bench harnesses; stale `0.9` in the README |

Counts over non-test library code (`src/`, `build/`, `furnace-candle/src`):

| `unwrap()` | `expect(` | `unsafe` | `#[allow` | `#[ignore]` | `sleep` in tests | `todo!`/`dbg!` |
|---|---|---|---|---|---|---|
| 13 (9 in `main.rs` on clap defaults, 4 in `build/` on `write!` to a `String`) | 38 | 2 sites (`weights.rs`, `allocator.rs`), both with correct SAFETY comments | 9 | 4 (`furnace-candle/tests/gguf.rs`, need local GGUF files) | 19 | 0 |

No `unwrap()` or `expect()` touches request data anywhere in the server, LLM or nerve code; all
request checks are explicit. The two `unsafe` sites are necessary and documented. The house-rule
violations are the nine `#[allow]`s and the build-time `cargo:warning` for a documented
configuration.

---

## 1. Structure

### 1.1 `src/api.rs` is 3,609 lines and holds eight unrelated clusters (replaced by the fabric)
- Where: `src/api.rs:1-3609`
- Severity: high. Effort: L.
- Finding: one file carries the public builder and config, the request/response DTOs, all
  middleware, the `/predict` decode-and-run path, the admin reload, the metrics renderers, the
  LLM and nerve glue and 1,300 lines of tests (`#[cfg(test)]` at line 2308). Everything public
  is reachable as `furnace::api::*` through `pub mod api` (`src/lib.rs:177`), so sub-modules can
  `pub use` their items back into `api` with no public API change; `pub use
  crate::batching::BatchConfig` at `src/api.rs:35` already does this.
- Proposal: none for this code: it is replaced by the fabric, not split. The clusters below
  (line ranges in today's file) are kept as a map for the rebuild, which needs a home for each
  concern (configuration, served state, routing, middleware, decode, admin, metrics, errors):
  1. `config.rs`: `ServerConfig` 276-300, `TlsSettings` 302-311, `ClientAddr` and its
     `Connected` impls 313-334, `Default for ServerConfig` 336-353, `Reloader`/`ServedModel`
     357-374, `Server` and `Contents` 376-519.
  2. `state.rs`: `Loaded` 521-572, `ModelSlot` 574-597, `ServerState` 599-655, `server_state()`
     667-720, `in_flight()` 1991-1995.
  3. `routes.rs`: `router`/`router_with_models` 657-665, `router_with_state` 722-767,
     `PUBLIC_PATHS` 769-776, `start_server*` 906-925, `serve()` 927-1074, `shutdown_signal`
     1076-1098.
  4. `middleware.rs`: `guard` 778-797, `API_KEY_HEADER` 799-800, `rejection_response` 802-836,
     `EXPOSED_HEADERS` 838-839, `edge` 841-891, `new_request_id` 893-904, `count_requests`
     1764-1778, `REQUEST_ID_HEADER` 53-55.
  5. `predict.rs`: DTOs 57-151, header consts 44-52, handlers 1132-1323, `run_inference`,
     `run_general`, `InferenceTask` 1325-1441, the three canned responses 1401-1450,
     `RequestFormat` and content-type helpers 1452-1501, decoders 1503-1684,
     `binary_output_response` 1686-1713, `empty_input_response`/`check_batch_len`/
     `validate_inputs` 1715-1762. `run_inference`, `run_general`, `busy_response` and
     `check_batch_len` are used by v2.rs (569-590, 736-757, 825) and grpc.rs (276-284,
     413-421), so they stay `pub(super)`.
  6. `admin.rs`: `ReloadRequest` 2137-2141, `reload_model` 2143-2278, `authorized` 2280-2290,
     `ReloadResponse` 232-242.
  7. `metrics.rs`: `MetricsResponse`/`ModelMetrics` 153-211, `metrics`, `wants_prometheus`,
     `prometheus_metrics`, `json_metrics`, `average_batch_size` 1780-2083, health and readiness
     244-259 and 1100-1130, model listing 213-230, 261-266, 2085-2135.
  8. `errors.rs`: `ErrorResponse` 268-274, `create_error_response` 2292-2306,
     `no_tensor_model_response` 1139-1147, `unknown_model_response` 1161-1171, and the one
     error-to-status mapping proposed in 2.1.
  9. The LLM and nerve glue (lines 189-196, 397-400, 405-412, 421-424, 470-488, 619-624,
     672-684, 715-718, 747-756, 929-942, 1897-1904, 2065-2072) is thin and stays under
     `#[cfg(feature)]` in config, state, routes and metrics; no separate module.
  10. Tests 2308-3609 become `api/tests.rs`; `pub(crate) mod tests` is already shared with
      v2.rs:882 and grpc.rs:449.

### 1.2 Two parallel request pipelines inside `Model`
- Where: `src/model.rs:581-653` (`infer_many`) vs `844-921` (`predict_many`); `662-741`
  (`check_inputs`) vs `941-976` (`flatten_batch`, `validate_flat`); `655-659` (`run_checked`)
  vs `1021-1043` (`forward`); `990-1018` (`warm_up`, two branches).
- Severity: high. Effort: L.
- Finding: `BurnModel` (`src/model.rs:128-152`) is already the one core trait, and every way of
  defining a model (`Model::from_fn` through `FnModel`, `Model::custom`, the generated `Served`
  from `built_in_model!`, `MlpModel`/`DummyModel` through `single_f32_run`, `DummyTensorsModel`)
  ends in a `Box<dyn BurnModel>`. `ModelNeuron` in `src/nerve/brain.rs:427-558` uses only the
  named door (`infer`, `infer_many`, `get_info`, `inputs`, `outputs`). The duplication is inside
  `Model`: the flat door (`predict*`, `validate_flat`, `forward`) and the named door
  (`infer*`, `check_inputs`, `run_checked`) are the same 75-line algorithm written twice, with
  validation (non-finite scan 683-687 vs 966-973, length check 719-726 vs 957-965), output
  checks (`check_outputs` vs 1032-1042) and warm-up each duplicated. The flat path also erases
  error variants (see 5.3).
- Proposal: keep every public method; make the flat ones adapters over the named path
  (`predict_many` wraps each `FlatInput` as a `HostTensor { shape: [batch] ++ input_spec.shape,
  values: F32(data) }` without copying, calls `infer_many`, unwraps the single `F32` output);
  delete `validate_flat`, `forward` and the `simple` branch of `warm_up`. To keep the wire
  format, `check_inputs` must emit the legacy variants for simple models: today the flat path
  returns `InputValidation {..}` (`INPUT_VALIDATION_FAILED`) and `InvalidDataType("Invalid value
  at batch[i][j] ...")`, the named path returns `InvalidInput(String)`. With that mapping there
  is no public API or wire change.

### 1.3 Four ways to turn a `HostTensor` into a Burn tensor
- Where: `src/model.rs:193-200` (`FnModel::run`), `204-229` (`single_f32_run`),
  `build/registry.rs:262-273` and `286-292` (rendered `run_generated`), `src/tensor.rs:193-216`.
- Severity: medium. Effort: M.
- Finding: `single_f32_run` rebuilds `TensorData` by hand and reshapes to `[batch, per_item]`
  instead of calling `into_float`; `FnModel::run` and `single_f32_run` both re-implement
  "expected 1 input" (194-197 vs 210-213); the generated code renders its own `missing_input`
  closure (`src/models/mod.rs:54-56`).
- Proposal: `HostTensor::into_float_flat(device) -> Result<Tensor<2>>` and a shared
  `fn take_inputs<const N: usize>(Vec<HostTensor>) -> Result<[HostTensor; N]>` in
  `src/tensor.rs`; the two doc examples that destructure with `unwrap()` (`src/lib.rs:138-139`,
  `src/model.rs:121`) become `let [ids, mask] = take_inputs(inputs)?;`. No public API change.

### 1.4 `effector.rs` and `receptor.rs` each bundle five or six independent kinds
- Where: `src/nerve/effector.rs:606-1056` (banners at 606, 670, 767, 883, 947, 1001;
  `#[cfg(test)]` at 1058); `src/nerve/receptor.rs:265-943` (banners at 265, 393, 500, 585, 758;
  `#[cfg(test)]` at 945).
- Severity: medium. Effort: M.
- Finding: the loop (`effector.rs:34-604`) and the shared receptor plumbing (`receptor.rs:37-263`)
  are the only parts the others depend on; each Act and each receptor has its own test block.
  The SSE parser (`receptor.rs:812-864`) is a self-contained state machine.
- Proposal: `src/nerve/effector/mod.rs` (loop, `Act`, `Action`, `ActError`, `EffectorMetrics`)
  plus one file per Act; `src/nerve/receptor/mod.rs` plus one file per receptor with
  `SseParser` in `sse.rs`; re-exports keep every path (`furnace::nerve::effector::Webhook`).
  No public API change.

### 1.5 Smaller structure items

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/nerve/fabric.rs:852-900` | low | S | `FnNeuron` (852-893) belongs in `neuron.rs`, `impl Payload` (895-900) in `signal.rs` next to the other `Payload` methods (281-377). **After the batching unification merges**, since it edits this file. |
| `src/types.rs:1-10`, `src/lib.rs:197,205` | low | S | A 10-line module for one struct; move `TensorSpec` to `src/tensor.rs` and keep `pub mod types { pub use crate::tensor::TensorSpec; }` plus the root re-export. No public API change. |
| `src/api.rs:504-507, 510-516, 669` | medium | S | **(Replaced by the fabric.)** `Server::serve` returns `ApiError::ServerStartup("no model to serve")` for an empty server, but `Server::router` reaches `assert!(!contents.is_empty())` in `server_state` and panics in library code. Either `router() -> Result<Router>` (public API change) or a `# Panics` section (docs only). |
| `src/api.rs:657-665, 906-925`, `src/lib.rs:200` | low | S | **(Replaced by the fabric.)** `router`, `router_with_models`, `start_server`, `start_server_with_config` are one-line wrappers over the builder; only `start_server_with_models` has a caller (`src/main.rs:713`). Implement them through `Server::with_config(..).model(..)` so there is one construction path. |
| `src/main.rs:18-41, 682-698` | low | S | **(Replaced by the fabric.)** `CliArgs` mirrors `ServerConfig` field by field. Let `parse_args` return a `ServerConfig` directly. |
| `src/llm/scheduler.rs:149-150, 170-171, 222-228` | low | S | `Scheduled` copies `name` and `max_context` from the model although `self.shared.model` is at hand. Delegate. |
| `src/llm/scheduler.rs:383-391, 408-416` | low | S | `Slot { running: Arc<dyn Fn()> }` wraps a `Weak<Shared<M>>` only because `Running<S>` is generic over the sequence; make `Running` generic over `M` and hold the `Weak` directly. Private types. |
| `src/nerve/http.rs:35-36`, `src/api.rs:752-756` | low | S | `routes(fabric)` calls `fabric.start()` as a side effect of building a `Router`. Document on `Server::nerve`, or move `start()` to `serve()` (behaviour change for `router()` callers). |
| `furnace-candle/src/model.rs:95-302` | medium | L | A maintained copy of candle-transformers' llama/qwen2 forward pass, guarded only by four `#[ignore]`d tests (`furnace-candle/tests/gguf.rs:121,127,164,170`). See 8.6. |
| `src/tls/provider.rs:40, 173-208`, `src/tls/mod.rs:12` | low | S | `pub fn provider()` and three `pub static TLS13_*` live in a private module; make them `pub(super)`. |

---

## 2. Duplicated or near-duplicated code

### 2.1 The `FurnaceError` to status/code mapping is written four times, plus a dead fifth (replaced by the fabric)
- Where: `src/api.rs:1308-1319`, `src/api/v2.rs:579-587`, `src/api/v2.rs:746-754`,
  `src/api/grpc.rs:312-321`; `src/error.rs:125-247` (unused `impl IntoResponse for FurnaceError`).
- Severity: medium. Effort: S.
- Finding: each copy matches `ModelError::{InputValidation, InvalidDataType, InvalidInput,
  InferenceFailed}` to a status. The dead impl disagrees with the live mapping: `InvalidInput`
  is `INVALID_INPUT` there (`error.rs:168-170`) and `INVALID_INPUT_VALUE` in `api.rs:1314`;
  `InvalidDataType` is `INVALID_DATA_TYPE` vs `INVALID_INPUT_VALUE`. No handler returns
  `Result<_, FurnaceError>`, so the impl is unreachable, and it cannot be flagged by
  `dead_code` because it is a trait impl.
- Proposal: one `fn inference_error(&FurnaceError) -> (StatusCode, &'static str)` in
  `api/errors.rs`, used by predict, v2 and grpc (`Status` built from the `StatusCode`); delete
  the impl (public API change: `FurnaceError` stops implementing `IntoResponse`; no in-tree
  user) and the `axum` import in `src/error.rs:1-5`. Keeping the api.rs mapping means no wire
  change.

### 2.2 Semaphore acquisition, timing and `observe_inference` repeated in every transport (replaced by the fabric)
- Where: `src/api.rs:1248-1257`, `src/api/v2.rs:568-575`, `src/api/v2.rs:735-742`,
  `src/api/grpc.rs:275-280`, `src/api/grpc.rs:412-417`; the matching `observe_inference` at
  `api.rs:1266-1267`, `v2.rs:591`, `v2.rs:758`, `grpc.rs:285`, `grpc.rs:422`.
- Severity: medium. Effort: M.
- Finding: five copies of `Instant::now(); inference_slots.clone().try_acquire_owned() else
  busy` and five of `observe_inference(slot, start.elapsed())`. `run_inference` and
  `run_general` already take the permit.
- Proposal: move acquisition, timing and observation into `run_inference`/`run_general` (or a
  `ServerState::run(slot, loaded, job)`), returning a typed outcome `Busy | Timeout | Panicked |
  Done(Result<..>)` that each transport maps to its wire format. Only the `Arrival` plumbing
  touches the batcher's API; the batcher itself is not changed.

### 2.3 Named-tensor request decoding is implemented twice (replaced by the fabric)
- Where: `src/api/v2.rs:473-541` (`decode_named_inputs`), `src/api/grpc.rs:342-404`.
- Severity: medium. Effort: M.
- Finding: same algorithm (`vec![None; inputs.len()]`, `match_input`, "given twice",
  `request_shape`, size check, `decode_raw` or typed values, "Missing input"), differing only
  in where the bytes come from. The two error messages differ ("binary_data_size is .. bytes"
  vs "raw_input_contents has .. bytes").
- Proposal: a shared `assemble_inputs(model, tensors)` in v2.rs with `enum Source { Raw(&[u8]),
  Values(Values) }`, called by both. Harmonising the two messages is a behaviour change to say
  in the changelog.

### 2.4 Three hand-written Prometheus renderers
- Where: `src/telemetry.rs:107-146` (`render_value`, `Histogram::render`),
  `src/llm/metrics.rs:144-146, 156-158, 225-226`, `src/nerve/metrics.rs:241-331` (six `# HELP`/
  `# TYPE` pairs by hand), `src/nerve/effector.rs:548-604`, `src/nerve/metrics.rs:342-361`
  (`render_quantiles`).
- Severity: medium. Effort: M.
- Finding: `llm/metrics.rs` reuses `telemetry::Histogram`, `Percentiles`, `label` and
  `render_series` (good) but writes its own family headers because telemetry only renders
  unlabelled values and histograms; nerve writes them a third way and renders latency with its
  own `LatencySnapshot` (microseconds, p50/p99) and quantile gauges, so `/metrics` carries two
  percentile shapes.
- Proposal: `telemetry::render_family(out, name, kind, help)` and `render_labelled(out, name,
  labels, value)`, used by all three; byte-identical output is the acceptance test. Converging
  nerve latency on `Histogram` is a later behaviour change.

### 2.5 Twelve helpers copied across the four bench harnesses
- Where: memory probe (Win32 `unsafe` plus `/proc` parser): `benches/burn/src/main.rs:1031-1082`,
  `benches/llm/src/main.rs:930-981`, `benches/nerve/src/main.rs:1100-1165`,
  `benches/compare/src/lib.rs:445-516`; `load()`: `burn:487-525`, `llm:476-553`,
  `nerve:700-760`; start-up and health polling: `burn:598-668`, `nerve:808-860`,
  `compare/lib.rs:274-324`, `burn/cold_start.rs:92-103`; percentile: `burn:518`, `llm:471-474`,
  `nerve:63-66`, `compare/lib.rs:155-158`, `examples/http_bench.rs:129-136`; `fail`, `free_port`,
  `python`, `stop`, `median`, `median_server`, JSON output (anchors in the per-area notes).
- Severity: medium. Effort: M.
- Finding: five percentile implementations use three definitions (round-to-index, nearest-rank
  with ceil, Python's exclusive interpolation), so a "p95" in one table is not the "p95" of
  another. `stop()` differs in a way that matters: burn and nerve send `kill -TERM` on Unix
  first, `llm`'s does not. Four `windows-sys` dependencies carry the same feature list.
- Proposal: a `benches/common` path crate (no `[workspace]`), depended on by the four bench
  workspaces: `memory`, `percentile` (one documented definition), `median`,
  `wait_until_healthy`, `free_port`, `load`, `stop`, `fail`, `report::write_json`. Whichever
  harness changes its percentile definition is a documented behaviour change in
  `docs/PERFORMANCE.md`.

### 2.6 Smaller duplication items

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/api.rs:268-274, 2292-2306` and `src/error.rs:104-122` | medium | S | **(Replaced by the fabric.)** `ErrorResponse` and its constructor exist twice, identically. Keep `error::ErrorResponse`, `pub use` it from `api`. The two paths keep resolving. |
| `src/api/v2.rs:138-157`, `src/api/grpc.rs:172-191` | low | S | **(Replaced by the fabric.)** Model metadata assembled twice; `From<&ModelInfo> for v2::ModelMetadata`, build the protobuf from it. |
| `src/api.rs:1607-1608, 1693-1696`, `src/api/v2.rs:344-379, 856-861`, `src/api/grpc.rs:252-256, 291-294` | low | S | **(Replaced by the fabric.)** Raw f32 codec written five ways; `decode_raw`/`encode_raw` exist, use them. |
| `src/api.rs:1161-1171`, `src/api/v2.rs:119-124` | low | S | **(Replaced by the fabric.)** "Unknown model" message built twice. |
| `src/api/v2.rs:825-826` | low | S | **(Replaced by the fabric.)** `check_batch_len` returns `Result<(), Box<Response>>`; v2 builds a JSON body plus `Utc::now()` only to drop it. Return a reason, build the response in `/predict`. |
| `src/api.rs:1998-2021` | low | S | **(Replaced by the fabric.)** `json_metrics` locks each slot's stats twice; compute totals from the first loop. |
| `src/config.rs:64` and `src/llm/http.rs:144` | low | S | **(Replaced by the fabric.)** Two `OneOrMany` enums (generic and `String`-only). Reuse the generic one. |
| `src/models/mod.rs:302-307`, `build/registry.rs:84-89` | low | S | `normalize_name` exists in the crate and the build script, kept in sync by a test (`models/mod.rs:509-514`). Move to `src/models/name.rs`, `#[path]`-include from the build script, delete the drift test. |
| `src/model.rs:689-697`, `src/tensor.rs:207-215, 256-268` | low | S | Integer range scanned twice per request (`check_inputs`, then `into_int`); `int_range` builds a probe tensor on every call. Check once; cache per device in a `OnceLock`. |
| `src/nerve/fabric.rs:512-514`, `effector.rs:322-324`, `receptor.rs:188-190` | low | S | "nanos since epoch" written three times with three epochs; one `pub(crate) fn nanos_since`. |
| `src/nerve/receptor.rs:713-728, 884-939`, `effector.rs:409-415` | low | S | Reconnect-with-backoff written twice (100 ms vs 200 ms start); a `Backoff` helper. |
| `src/nerve/effector.rs:328-339` vs `neuron.rs:364-378` | low | S | `Effector::refractory` re-implements the gate's refractory with its own state and counters; decide on one or document the difference (removal is a public API change). |
| `src/nerve/fabric.rs:936-947`, `receptor.rs:974-984`, `effector.rs:1108-1114`, `brain.rs:599-621` | low | S | `wait_until` and `fabric_with` test scaffolding written three times; one `#[cfg(test)] mod testing`. |
| `benches/burn/src/models.rs:18-40` vs `examples/serve_burn_model/models.rs:13-35` | low | S | `DigitNet` byte-for-byte twice; `TextModel` (`burn/main.rs:231-271`) and `Sentiment` are the same `BurnModel`. Point at each other in a comment or share through `benches/common`. |
| `benches/*/Cargo.toml` `[profile.release]` ×4 vs `Cargo.toml:208-216` | low | S | Must stay identical; nothing checks. A comment listing the copies, or a CI grep. |

---

## 3. Naming and concept drift

The concept inventories of the five areas were compared. These are the same ideas under
different names across `api`, `model`, `llm` and `nerve`. None of these must change, and most
renames would be public API changes; the ones worth doing now are marked.

### 3.1 One idea, several names

| Idea | Names and where | Note |
|---|---|---|
| A served thing plus bookkeeping | `ServedModel` `api.rs:362`; `Served` `llm/http.rs:45`; `Served` (generated) `models/mod.rs:88`; `Loaded` `api.rs:522`; `Contents` `api.rs:395` | `Contents` says nothing about what it holds; rename to `ServerContents` (private). `llm::Served` to `ServedLlm` (private). **Do now.** |
| Describe a tensor | `TensorSpec` `types.rs:5`; `TensorInfo` `tensor.rs:96`; `IoSpec` `tensor.rs:56`; `StaticIo` `models/mod.rs:23`; `IoDesc`/`Io`/`IoType` `build/registry.rs:34,27,19`; `v2::TensorMetadata` `v2.rs:67`; `pb::TensorMetadata` | `TensorInfo` and `v2::TensorMetadata` have the same three fields and the same `-1` convention: reuse `TensorInfo` (serde output identical, no wire change). `ModelInfo` carries `input_spec` and `inputs[0]` for the same facts (`model.rs:23-28`). |
| Describe a model | `ModelInfo` `model.rs:20`; `ModelSummary` `api.rs:222`; `ModelMetrics` `api.rs:202`; `v2::ModelMetadata` `v2.rs:75`; `Neuron::describe()` `neuron.rs:126`; `Description`/`NeuronDescription` `nerve/metrics.rs:80,88` | Wire types; leave. |
| Counters and snapshots | `ModelStats` `model.rs:37` (live, under a Mutex); `SchedulerStats` `scheduler.rs:98` (live histograms); `EdgeMetrics` `llm/metrics.rs:18` (live); `LlmMetrics` (snapshot); `NeuronMetrics`/`PathwayMetrics`/`ArcMetrics` (snapshots); `fabric::Counters`, `effector::Counters`, `BrainCounters` (live); `Percentiles` (ms, p50/p95/p99) vs `LatencySnapshot` (µs, p50/p99) | Convention to write down: live = `*Counters`, snapshot = `*Metrics`; "Stats" appears only in `model` and `scheduler`. Renaming `SchedulerStats` is a public API change; `ModelStats` is the `/metrics` JSON. Document, do not rename. |
| Batch of work | `BatchConfig { max_batch_size, batch_timeout, workers }` `batching.rs:83`; `Summation { max_batch, window, capacity }` `fabric.rs:38`; `config::Batching` `config.rs:80`; `SchedulerConfig { max_prefill_tokens }` | Overlaps the batching unification; the unification should keep the `BatchConfig` names (public and on the CLI). |
| The way back to the client | `PredictResponse`; `Reply<J>` (`batching.rs:112`, a `Result`); `llm::http::Reply` (`http.rs:494`, id/created/model); `nerve::Reply` (`signal.rs:404`, a oneshot handle); `nerve::Answer`; `wire::Response` | Three unrelated `Reply`s in one crate. `nerve::Reply` to `ReplyHandle` is a public API change; the batching alias is private to its module. Document for now. |
| A request body | `PredictRequest`, `v2::InferRequest`, `pb::ModelInferRequest`, `llm::Input`/`GenerateRequest`/`ChatRequest`, `nerve::Signal`/`Ingest`, `FlatInput`/`NamedInput`/`HostTensor` | `NamedInput(pub Vec<HostTensor>)` holds tensors that are not named (names come from `inputs()` order, `batching.rs:58`). `llm::Input` is ambiguous at crate level (`Prompt` would be clearer). Public API changes; defer. |
| Rate limiting | `access::RateLimit { per_second, burst }` `access.rs:17`; `effector::RateLimit { per_second, burst, max_wait }` `effector.rs:140` | Same name, two types, both public. Alias in user code. Defer. |
| Channel and SSE | `receptor::Channel`/`effector::Channel`; `receptor::Sse`/`effector::Sse`; `Delivery` `llm/http.rs:229` vs `effector::Delivery` `effector.rs:952` | `benches/nerve/src/main.rs:39` already aliases `Channel as ChannelEffector`. Renames are public API changes; defer to 0.11. |
| `Inner`, `Counters`, `Shared` | `fabric::Inner`/`effector::Inner`; `fabric::Counters`/`effector::Counters`; `scheduler::Shared`/`brain::llm::Shared` | Private; rename effector's to `EffectorInner`, `LoopCounters`. **Do now** with the effector split. |
| Why a request ended | `FinishReason` `llm/mod.rs:233`; `Quiet` `neuron.rs:229`; `Outcome` `fabric.rs:223`; HTTP status | Different domains; fine. |
| Request identity | `x-request-id` (`api.rs:55`) vs `x-furnace-signal` (`nerve/http.rs:26`); `session_id: Uuid` (`main.rs:614`); `completion_id()` (`llm/http.rs:861`) | Two per-process seed-plus-counter idioms (`api.rs:897-901`, `llm/http.rs:863`): one `request_id()` helper. |
| Errors | `FurnaceError` (api, model); `EngineError` (llm); `ActError { message, retry }` (nerve); `tonic::Status`; `Result<_, String>` in `tls`, `config`, `v2`, `wire`, `brain` | See section 5. |
| The one limit | `max_concurrent_requests` (`api.rs:281`), `max_concurrent` (`api.rs:604`), `max_concurrent_inferences` (JSON, `api.rs:169`), `furnace_max_concurrent_requests` (Prometheus, `api.rs:1877`) | Rename the private field; the wire names stay. |

### 3.2 Names that are misleading

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/nerve/metrics.rs:46, 153, 279` | low | S | `NeuronMetrics.quiet` is the fourth reason next to `quiet_by_threshold` etc., rendered as `reason="not_interested"` in Prometheus; a JSON reader takes it for a total. Renaming the JSON field is a public API change; at least document. |
| `src/nerve/metrics.rs:297-300`, `fabric.rs:27-34` | low | S | `PathwayKind` is `"reflex"` in JSON and `kind="Reflex"` in Prometheus (`{:?}`). Add `as_str()` as `NeuronKind` has. Behaviour change (label value). |
| `src/model.rs:966-973, 923-938`, `src/tensor.rs:218-226` | low | S | `InvalidDataType` for a non-finite value, `InputValidation {expected:[1000], actual:[1001]}` for a batch overflow, `InputValidation` for a rank mismatch: the HTTP message reads "shape mismatch" for "batch too large". A `BatchSize { max, got }` variant is a behaviour change (`code` field). |
| `src/llm/mock.rs:79-92, 169-173` | low | S | `Live` also counts decode calls that are not alive. Public API in `furnace::llm::mock`; defer. |
| `src/llm/pipeline.rs:18` | low | S | `templates: Option<Environment>` holds exactly one template. Rename `chat_template`; private. |
| `src/tensor.rs:15-19, 116-119` | low | S | `DataType::Int32` inputs travel as `Values::I64`; document that INT32 is a wire label stored widened. `build/registry.rs:11` has `Elem { F32, I64, I32 }` for the same idea; rename to match. |
| `src/nerve/receptor.rs:269` vs `nerve/http.rs:24-26` | low | S | `DEADLINE_HEADER: &str` next to two `HeaderName` consts, and only the latter re-exported. Public API change (type of a const). |
| `src/api.rs:1452-1456, 1204` | low | S | **(Replaced by the fabric.)** `SUPPORTED_REQUEST_TYPES` is a fragment that only reads right inside one `format!`. |

---

## 4. Public API surface

### 4.1 Rate limiting without `ConnectInfo<ClientAddr>` puts every client in one bucket (replaced by the fabric)
- Where: `src/api.rs:787-791`, `src/access.rs:70-74`, `src/api.rs:1041`, `src/lib.rs:51-92`.
- Severity: high. Effort: S.
- Finding: `guard` reads the peer address from the `ConnectInfo<ClientAddr>` extension, which
  exists only when the router is served through `into_make_service_with_connect_info::<ClientAddr>()`
  (`serve()`, line 1041). The documented "mount in your own axum app" path (`src/lib.rs:51-92`,
  `Server::router()`) serves with plain `axum::serve`, so `ip` is `None` and, without API keys,
  every request lands in `Client::Unknown`: one shared budget for the world, silently.
- Proposal: in `router_with_state`, when `rate_limit.is_some()` and `api_keys.is_empty()`, log a
  startup warning that IP limiting needs `ConnectInfo<ClientAddr>`, and say so in the
  `Server::router` doc with the one-line `into_make_service_with_connect_info` example. A
  fallback to `ConnectInfo<SocketAddr>` is a behaviour change worth considering.

### 4.2 Five error variants nothing constructs, four `#[allow(dead_code)]`, one of them stale
- Where: `src/error.rs:46, 64, 71, 75`; variants `ApiError::{InvalidRequest, JsonParsing,
  ModelNotLoaded}` (64-78), `CliError::{MissingArgument, InvalidModelPath}` (89-99).
- Severity: high (house rule). Effort: S.
- Finding: grep over `src/`, `tests/`, `examples/` finds zero constructions of the five
  variants; they feed only the dead `IntoResponse` impl (2.1). `ModelError::InvalidDataType`
  *is* constructed (`model.rs:217, 967`, `tensor.rs:271`), so its `#[allow]` is stale. The four
  attributes are also no-ops: variants of a `pub enum` re-exported from the crate root never
  trigger `dead_code` in a library.
- Proposal: delete the four attributes and the five variants (public API change: `ApiError`
  becomes a one-variant enum, `CliError` loses two). Note in CHANGELOG.

### 4.3 `furnace::nerve::wire` is public
- Where: `src/nerve/mod.rs:73`, `src/nerve/wire.rs:22-27, 231-234`, `benches/nerve/src/main.rs:41`.
- Severity: medium. Effort: S.
- Finding: `pub mod wire` exposes `Url`, `Response`, `Streaming`, `Client` with semver
  obligations, for a client whose own header says it does plain `http://` and "nothing else".
  Only `benches/nerve` uses it from outside, for the raw-call baseline.
- Proposal: `pub(crate) mod wire` with a `#[doc(hidden)]` re-export for the bench. Public API
  change (removal); best before 1.0.

### 4.4 Constructors and trait methods that panic or cannot be stopped (nerve)
- Where: `src/nerve/effector.rs:694` (`Webhook::with_timeout` panics on a non-`http://` URL,
  HTTPS included), `261, 707-708` (`expect("configured before use")` once the value was cloned),
  `src/nerve/fabric.rs:299-304, 341-346` (`FabricBuilder::add`/`build` asserts);
  `src/nerve/receptor.rs:75-77` (`Receptor::start` has no stop signal), `487-496, 560-581,
  699-753, 880-941` (Channel, Timer, Lines threads never joined; `Lines::stdin` and
  `Lines::listen` block forever).
- Severity: medium. Effort: M.
- Finding: all reachable from user code with no `Result`. A Timer with a 1 h period outlives
  its fabric by an hour; `Lines::listen` keeps the port.
- Proposal: `Webhook::new -> Result` and `FabricBuilder::build -> Result` (public API change),
  or `# Panics` sections now (docs only); a stop flag on `Afferent` set from `Drop for Inner`,
  with `start` returning an optional `JoinHandle` kept in `Inner` (public API change to the
  trait unless the handle stays out of the signature).

### 4.5 Smaller public API items

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/api.rs:32`, `src/api/v2.rs:26-33, 60, 67, 75, 274, 283, 297` | low | S | **(Replaced by the fabric.)** Ten `pub` items in the private `mod v2` are unreachable; `pub(super)` (honest) or `pub mod v2` (API addition useful to Rust v2 clients). |
| `src/model.rs:512, 1069-1071` | low | S | `Model::new(Box<dyn BurnModel>, &'static str, u64)` is a third public constructor; `model_type()` has no caller. `pub(crate)` and removal are public API changes. |
| `src/model.rs:278-425, 1102-1112` | medium | M | `DummyModel`, `DummyTensorsModel` (with a test-only run counter) and the `"sample"` fallback ship in the release library; used by tests in seven files and `benches/inference_benchmark.rs`. A `test-models` feature or `pub mod testing` is a public API change. |
| `src/model.rs:1102-1112` | medium | S | `load_model` serves a freshly initialised random MLP for any path whose stem *contains* `"sample"` with an `info!` line, contradicting `DummyModel`'s "never a silent fallback" (280). Match the exact stem and `warn!`. Behaviour change (narrows what is accepted). |
| `src/model.rs:990, 1016`, `src/api.rs:2243-2244` | medium | S | `Model::warm_up` returns `Result<_, String>` and the only caller wraps it back into `InferenceFailed`. Return `crate::Result`. Public API change (signature). |
| `src/llm/mod.rs:252-265` vs `231-238, 268-286`; `src/nerve/brain.rs:274-281` | medium | S | `Event` is exhaustive while `FinishReason`, `EngineError`, `SamplingParams` are `#[non_exhaustive]`; any new event breaks every downstream `match`. Add `#[non_exhaustive]` before 1.0 (public API change). |
| `src/llm/scheduler.rs:73-84` | low | S | `SchedulerConfig` exhaustive with public fields; tests already use `..Default::default()`. |
| `src/llm/pipeline.rs:8`, `mod.rs:121` | low | S | `pub use tokenizers::Tokenizer` makes a tokenizers major bump a furnace breaking change; document the decision. |
| `src/llm/mod.rs:308-311` | low | S | `LlmEngine::scheduler_stats` is a `Scheduled`-only hook on the generic trait; document it as the extension point it is. |
| `src/nerve/signal.rs:506-512`, `neuron.rs:56-58`, `receptor.rs:632-641` | low | S | `Reply::is_open` and `Firing::compute` have no caller; `Lines::line_signal` is `pub` but an internal helper. Removal is a public API change; `Firing::Compute` deserves one test if kept. |
| `src/nerve/neuron.rs:126-133`, `brain.rs:162-172`, `effector.rs:540-542` | low | M | `Neuron::metrics()`/`describe()` return untyped `serde_json::Value`; `EffectorMetrics` is typed and then erased into `NeuronMetrics::extra`. A typed `NeuronExtra` enum is a public API change. |
| `src/batching.rs:157-160` | medium | S | `Batcher::new` `expect`s on thread spawn at server start and on every reload. Overlaps the batching unification. |
| `src/tls/mod.rs:60, 85, 134, 171` | low | M | `TlsConfigs::load`/`reload_if_changed`/`build`/`read_certs` return `Result<_, String>`; a `tls::Error` enum is a public API change. |

---

## 5. Error handling

### 5.1 A `/predict` validation error comes back as HTTP 500
- Where: `src/model.rs:1021-1030` (`forward` maps every error to `String`), `907-913`
  (`predict_many` wraps it in `InferenceFailed`), `749-753` (`run_in_chunks` returns
  `InvalidInput("the model runs batches of N; got M items")`).
- Severity: high. Effort: S (minimal) / covered by 1.2.
- Finding: a batch size that does not fit a fixed-batch ONNX graph is a client error (`400` on
  `/v2/.../infer` for a single request) but a server failure (`500 INFERENCE_FAILED`) on
  `/predict`, with a double-prefixed message ("Inference failed: Model error: Invalid input:
  ..."). Also affects `warm_up` through the same path.
- Proposal: `forward` returns `crate::Result<Vec<f32>>` and the single-request arm keeps the
  error as `infer_many` does (636-639). Behaviour change (500 becomes 400 for that case); a bug
  fix to list in CHANGELOG. Subsumed by 1.2.

### 5.2 `warm_up` of a simple model ignores `fixed_batch`
- Where: `src/model.rs:992-995` vs `997`, `745-754`; `src/api.rs:539-550, 2241-2244`.
- Severity: medium. Effort: S.
- Finding: the simple branch warms up with batch 1; a model with `fixed_batch() == Some(n)`,
  `n > 1`, fails warm-up at every start ("Warm-up inference failed" logged) and every reload
  (which then *fails*). The named branch uses `fixed_batch().unwrap_or(1)`. Only `mnist`
  (`Some(1)`) is in the repository, so it has not bitten yet.
- Proposal: delete the simple branch (1.2).

### 5.3 `Model::infer` on a rank-0 tensor panics, aborting a release server
- Where: `src/model.rs:706-718` (`check_inputs` accepts `shape == []` for a scalar-per-item
  spec), `src/tensor.rs:142-145` (`batch()` returns 1 for an empty shape), `148, 168, 175`
  (`shape[1..]`, `shape[0]` on an empty Vec), `src/nerve/brain.rs:534`.
- Severity: medium. Effort: S.
- Finding: HTTP and gRPC are safe because `request_shape` (`v2.rs:433-435`) always inserts the
  batch dimension, but library users and `ModelNeuron` input closures build `HostTensor`s by
  hand; two such requests in `infer_many` reach `concat_batch` and panic, and with
  `panic = "abort"` (`Cargo.toml:213`) the process dies.
- Proposal: `check_inputs` requires `tensor.shape.len() == spec.shape.len() + 1` ("needs a batch
  dimension"); `batch()` returns 0 for an empty shape. Behaviour change (an input that happened
  to work alone is now rejected).

### 5.4 `Respond` failures from already-answered requests open the breaker for live ones
- Where: `src/nerve/effector.rs:662-666` (`ActError::fatal("nobody is waiting")`), `385-399`
  (the breaker counts it), `128-134` and `236-262` (`Effector::new` applies `Breaker { failures:
  5, cooldown: 10 s }` by default), `src/nerve/http.rs:85-96` (`respond_within` defaults to
  zero, so queued signals are answered `202` at once).
- Severity: high. Effort: S.
- Finding: five brain-answered requests whose clients already got `202` open the circuit and
  every `Respond`, including those for live reflex-arc requests, is dropped for 10 s with a
  `warn!` each. `tests/nerve.rs` never routes the brain to `respond`, so it is not caught.
- Proposal: `Respond` returns `Action::ok()` (or a non-counting outcome) when nobody waits; a test
  with a deliberative pathway ending in `Respond`. Behaviour change (`failed` no longer counts
  late answers).

### 5.5 Smaller error-handling items

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/api.rs:1503, 1531, 1559-1563, 1618, 1637-1642, 1715-1762, 1336, 1371` | low | M | **(Replaced by the fabric.)** Decoders use `Box<Response>` as the error type, so `run_inference` returns `Result<Result<Vec<f32>>, Box<Response>>` and grpc.rs:323-330 reverse-engineers a status from a `Response`. A small `enum PredictError` with `IntoResponse` and `From<&PredictError> for tonic::Status`. No wire change. |
| `src/api/grpc.rs:57`, `src/api.rs:1012-1017, 1067`, `src/config.rs:87-95`, `src/api/v2.rs:114-131, 391-470, 473-541, 792-878` | low | M | **(Replaced by the fabric.)** Stringly errors at the server boundary; the gRPC task's failure is `ApiError::ServerStartup(e)` where `e` is already formatted. Keep v2's `String` messages (they are the wire `error` field) behind a `ValidationError(String)` newtype. |
| `src/error.rs:25-29, 32, 38, 48, 52`; `src/burn_model.rs:103-106, 110-114, 128` | low | M | Five of eight `ModelError` variants carry prose; `burn_model.rs:110/128` format a `serde_json::Error` into `InvalidFormat(String)` and `:103` uses `LoadFailed` for a *save* failure. Payload changes are public API changes. |
| `src/model.rs:640-646, 907-913` | low | S | When a shared forward pass fails, every request in the batch gets `InferenceFailed(e.to_string())`, losing the variant. Make `FurnaceError` `Clone` (`LoadFailed.source` as `Arc<dyn Error>`) and clone the error. |
| `src/weights.rs:22-31, 43` | low | S | A missing file takes the mmap fallback path and logs "Could not memory-map ...; reading it into memory instead" before failing with the real error. Open the file once outside `map_file`. |
| `build.rs:154, 163` | low | S | `input.to_str().unwrap()` inside `catch_unwind` reports a non-UTF-8 model path as "unsupported operator?". Convert before `catch_unwind`, as `out` is (148-151). |
| `build/registry.rs:93-98` | low | S | The keyword list omits the reserved keywords (`final`, `override`, ...), so `final.onnx` generates `pub mod final`, which does not compile. Complete the list or always prefix the module. |
| `src/api.rs:1076-1098, 996-999` | low | S | **(Replaced by the fabric.)** `expect` on signal-handler installation panics in a spawned task; the server keeps running but cannot be stopped gracefully, and the panic is only logged by tokio. Warn and fall back to `pending()`. |
| `src/api.rs:1774-1776, 161` | low | S | **(Replaced by the fabric.)** `count_requests` treats any 503 on the predict routes as `rejected_busy`; correct today, fragile. |
| `src/llm/http.rs:379-387` | low | S | **(Replaced by the fabric.)** `{"type": "text"}` without `text` is reported as "Content of type 'text' is not supported". A separate arm. Message-only change. |
| `src/llm/mod.rs:279-285`, `scheduler.rs:119, 398-404`, `furnace-candle/src/lib.rs:110-145`, `src/nerve/brain.rs:270-273` | low | M | `EngineError::{Busy,Invalid,Failed}(String)`, `Step::Failed(String)`, `guarded -> Result<T, String>`; the HTTP layer maps them correctly, but a model error loses its type at the first boundary. `Failed(BoxError)` is a public API change; low priority. |
| `src/nerve/effector.rs:36-39`, `wire.rs:30`, `brain.rs:269, 291, 296`, `effector.rs:746` | medium | M | `ActError { message: String, retry: bool }`, `Url::parse -> Result<Url, String>`, the brain's `Result<_, String>`; nothing downstream can tell a 429 from a connection refused. A `wire::Error` and an `ActError` enum with `retryable()` is a public API change. |
| `src/nerve/effector.rs:741-746` | low | S | `Webhook` treats 3xx as final (five of them open the breaker); say "no redirects, no HTTPS" on `Webhook` and in `docs/NERVE.md`. |
| `src/tls/provider.rs:521, 712-717, 680-684` | low | S | Any unsupported PKCS#8 curve is reported as "(RSA?)"; `signature_alg_id` has a `_ =>` arm that only agrees with `verify_signature` because the four statics are the only constructors. A two-variant hash enum makes both matches exhaustive. |
| `src/tls/provider.rs:143, 245, 252, 278, 339-352` | low | S | The `expect`s on key length rely on a hand-written `KEY_LEN`; derive it from `KeySizeUser` in the existing const block so they are backed by compile-time checks. |
| `src/main.rs:263` | low | S | **(Replaced by the fabric.)** The `--tls-cert` help string contains a run of 21 spaces that `--help` prints. |

---

## 6. Unsafe code

Two sites in the library, both necessary, both with correct SAFETY comments.

### 6.1 `src/weights.rs:52` `unsafe { Mmap::map(&file)? }`
- Severity: low. Effort: S.
- Finding: the comment (44-51) names the real hazard (the mapping aliases a file another
  process may write or truncate), the operational invariant (weight files are read-only
  deployment artifacts replaced by rename), the Windows behaviour, and claims the Burnpack
  reader bounds-checks every tensor against the mapped length. `memmap2` has no safe mapping
  API and `std::fs::read` would lose the page-cache sharing that is the module's point. The
  bounds-check claim is about `burn-store` internals and could not be verified here.
- Proposal: keep; date the claim ("burn-store 0.22; re-check on upgrade") and keep
  `malformed_file_is_an_error` (96-102) as its regression test.

### 6.2 `src/allocator.rs:99-124` `unsafe impl GlobalAlloc for BlockCache`
- Severity: low. Effort: S.
- Finding: the impl-level comment (96-98) states the invariant and the code upholds it: `take`
  matches exact `size` and `align` (62-64) so a reused block satisfies the requested `Layout`
  and goes back to `System` with an equal layout; `alloc_zeroed` zeroes a block of exactly
  `layout.size()`; `dealloc` never frees a kept block; the default `realloc` composes. The
  `Mutex` inside an allocator is justified at 45-46 (no allocation on lock). Only the binary
  installs it (`src/main.rs:15-16`). The `MAX_CACHED`/`SLOTS` overflow paths (77-82) are
  untested.
- Proposal: keep; add that `Block` stores `ptr` as `usize` only to be `Copy`/`Send`, a
  `debug_assert!(layout.size() >= MIN_SIZE)` in `keep`, and one test for the full-cache path.

The other `unsafe` blocks are in the bench harnesses (Win32 memory and timer calls, each with a
SAFETY comment: `benches/burn/src/main.rs`, `benches/llm/src/main.rs:105, 888, 944`,
`benches/nerve/src/main.rs:1117-1133`, `benches/compare/src/lib.rs`); 2.5 collapses them into
one. `build/registry.rs:97` contains `"unsafe"` only as an entry in the keyword list.

---

## 7. Concurrency

### 7.1 `wire::Client::request` times out the head only; the body read is unbounded
- Where: `src/nerve/wire.rs:285-308` (`request` loops `streaming.next().await` with no
  timeout), `311-321` (`send` wraps only `send_inner`), `src/nerve/effector.rs:736-739, 426-458`.
- Severity: high. Effort: S.
- Finding: a webhook target that answers `200` with a large `Content-Length` and then stalls
  holds the efferent task, the `inflight` counter and one of the two efferent threads' tasks
  forever; `Effector::run` adds no outer timeout. `Process` (865) and the brain do bound their
  waits.
- Proposal: wrap all of `request()` (and the body `write_all`) in the timeout, or give
  `Streaming::next` a per-read timeout. Behaviour change (slow responses fail with `TimedOut`
  and retry).

### 7.2 A loop through a deliberative pathway is not cut by `MAX_HOPS`
- Where: `src/nerve/fabric.rs:614-622` (`route` drops at `depth >= MAX_HOPS`), `788-791`
  (`fire_summed` calls `after_firing(.., 1)` with depth reset), `src/nerve/signal.rs:407-414`
  (`Trace::len()` keeps the true count but is not consulted).
- Severity: medium. Effort: S.
- Finding: a cycle reflex → brain (deliberative) → reflex → brain circulates forever through the
  queue, each pass counted as `reached_brain`.
- Proposal: in `route`, also drop when `signal.trace.len() >= MAX_HOPS`, counting it in `loops`.
  Behaviour change (such signals are dropped). **The fix lands in `fabric.rs`, which the
  batching unification is editing; do it after that merges.**

### 7.3 Chat template rendering and tokenization run on the Tokio runtime inside `generate`
- Where: `src/llm/scheduler.rs:230-231`, `src/llm/pipeline.rs:190-203`, `src/llm/mod.rs:301-302`
  (the trait doc says `generate` must not block the runtime).
- Severity: medium. Effort: M.
- Finding: for a 4k-token prompt with a fancy-regex pre-tokenizer this is milliseconds during
  which that worker delivers no SSE chunks to any stream. Everything else (prefill, decode,
  sampling) is on dedicated `furnace-llm-{i}` threads; detokenization on a per-request relay task.
- Proposal: move encoding to the worker (`Job` carries the `Input`); `TooLong`/`Invalid` then
  arrive after `Started`, so the HTTP layer must wait for the first event before committing to
  `200` (which also fixes "error after 200" for `TooLong`). Behaviour change; alternatively
  `block_in_place` for long prompts only (no API change, but panics on a current-thread runtime).

### 7.4 `catch_unwind` around the model is a no-op in release builds
- Where: `src/llm/scheduler.rs:397-404, 463-479`; `Cargo.toml:213` (`panic = "abort"`);
  `benches/llm/Cargo.toml:36`.
- Severity: medium. Effort: S.
- Finding: the `furnace` binary, the examples and the bench abort on a model panic; the "worker
  and its other sequences live on" promise holds only for library users on the default profile,
  who then keep using a model whose internal state may be half-updated. The batcher's
  `catch_unwind` (`src/batching.rs:255-266`) has the same caveat.
- Proposal: say so in `guarded`'s doc and `docs/LLM.md`; optionally poison the model after a
  caught panic (behaviour change).

### 7.5 The synchronous nerve chain runs on the axum worker thread
- Where: `src/nerve/http.rs:81-84` (`ingest` calls `fabric.stimulate` inline),
  `src/nerve/effector.rs:502-513, 922-935` (`FileAppend` opens and writes under a `Mutex`),
  `src/nerve/brain.rs:318-324, 513-514` (an `Llm` on a reflex pathway spawns a scoped thread and
  joins it; `ModelNeuron::reflex` runs `infer`), `docs/NERVE.md:335-337` (documents the model
  case only).
- Severity: medium. Effort: M.
- Finding: by design for the reflex arc (0.8 µs measured), but blocking Acts block a Tokio
  worker. The file write is not documented as such.
- Proposal: document `FileAppend`'s blocking write; make it return `Action::pending` on
  `spawn_blocking`. Moving `stimulate` itself off the worker is a behaviour change that would
  move the bench's arc numbers; not proposed.

### 7.6 No bound on in-flight TLS handshakes
- Where: `src/tls/mod.rs:277-293, 36-39`.
- Severity: medium. Effort: S.
- Finding: `accept()` spawns a task per accepted TCP connection; only completed handshakes are
  bounded (`ACCEPT_QUEUE` 256). A client opening many connections without a ClientHello holds a
  task and a socket each for `HANDSHAKE_TIMEOUT` (10 s); `--max-concurrent-requests` does not
  cover this stage.
- Proposal: a `Semaphore` (e.g. 1024 permits) acquired before `acceptor.accept`. Behaviour
  change (new limit); document next to `ACCEPT_QUEUE`.

### 7.7 What is right, and worth writing down
- Locks (`src/api.rs:576-597` `RwLock<Arc<Loaded>>` cloned out and never held across `.await`;
  `581, 2198-2248` a tokio `Mutex` only ever `try_lock`ed; `src/access.rs:124-157` sharded
  GCRA; `src/model.rs:441` `Arc<Mutex<ModelStats>>` taken once per forward pass, after it;
  poisoning recovered everywhere with `unwrap_or_else(|e| e.into_inner())`). Atomics: `ready`
  is Release/Acquire (`api.rs:551, 570, 2245`), counters Relaxed, `started` AcqRel. All correct.
- `Model` is immutable after construction except `stats`; reload builds a new `Loaded`, shares
  the old stats `Arc`, swaps under a brief write lock; in-flight requests keep their clone.
- The scheduler's shutdown is graceful (`src/llm/scheduler.rs:214-219, 422-434`: drop stops
  accepting, running sequences finish, threads are not joined); a slow client does not slow the
  worker (`251, 273, 300-303`, unbounded worker→relay channel, bounded relay→HTTP); cancellation
  on disconnect is correct and tested. Document both on `Scheduled`.
- `LlmEngine::generate` is synchronous but both implementations `tokio::spawn`
  (`scheduler.rs:274`, `mock.rs:283`), so it panics outside a runtime; the brain works around it
  with `runtime.enter()` (`brain.rs:293`). State the contract in the trait doc.

### 7.8 Smaller concurrency items

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/api.rs:1343-1349, 1379-1388, 1654-1657`, `src/threads.rs:24` | medium | M | **(Replaced by the fabric.)** With batching off (default) every forward pass is a `spawn_blocking` task that then enters the rayon pool; batcher workers are dedicated threads. Two thread models for one job. Routing the unbatched case through the batcher with `max_batch_size = 1` overlaps the batching unification; excluded from the plan. |
| `src/model.rs:81, 612, 877` | low | S | `record_success` calls `Utc::now()` under the stats lock; for a batch of N the same `elapsed_ms` is recorded N times (documented at 842-843). Take the time before the lock; document that timing fields are per forward pass. |
| `src/telemetry.rs:63-71` | low | S | `Histogram::snapshot` is not atomic across buckets, so `_count` may briefly exceed the bucket sum in a scrape. Note it in the doc. |
| `src/api/grpc.rs:198-209` vs `src/api.rs:727-762` | low | S | **(Replaced by the fabric.)** 401/429 rejections are counted in neither transport's `total_requests`; `docs/API.md:64-65` implies otherwise. Docs fix. |
| `src/nerve/neuron.rs:326, 350-378` | low | M | The habituation table is a `Mutex` on the hot path; `admit` reads `last_fired` and `fired` stores it on separate calls, so two concurrent signals can both pass the refractory check. `compare_exchange` in `admit` when a period is set. |
| `src/nerve/effector.rs:862-865` | low | S | `Process` writes stdin before the timed section; a child that never reads leaves the task stuck. Include the write in the timeout. Also say in the doc that `args` from a brain reach the program as-is (argv, no shell, so no injection). |
| `src/nerve/fabric.rs:347-358, 205-217` | low | M | Each `Fabric` builds two Tokio runtimes plus a thread per deliberative pathway; tests build dozens. `FabricBuilder::efferent(Handle)` to reuse the server's runtime (API addition). After the unification. |
| `src/llm/scheduler.rs:259-269, 428` | low | S | `retain` over the whole queue under the lock on every request; count closed jobs lazily. |
| `src/tls/mod.rs:86-91` | low | S | `reload_if_changed` checks and sets under two separate lock acquisitions; take the write lock once. |
| `src/batching.rs:226-238, 190-193, 204-210` | low | S | `busy` incremented once more than decremented on shutdown; `arriving` leaks when `send` fails. Overlaps the batching unification. |

### 7.9 What the batching unification must not lose
Recorded here for the session doing it. `src/batching.rs` has, and its tests pin: skipping
requests whose reply is closed (284-296, `abandoned_request_is_skipped` :706); running side by
side while `busy + 1 < workers` (299-301, :719); the deadline taken from the *first* request's
`queued_at` (303); waiting only while `arriving > 0` (310-323, :652, :665); carry-over by items,
not by count (332-335, :618); `Message::Wake` on an abandoned `Arrival` (204-210); permits held
until the reply is sent and `catch_unwind` per batch (255-266, :693, :751); named-tensor
requests sharing a batch (:435). The nerve summation worker (`src/nerve/fabric.rs:814-845`) has
none of these today and decrements `waiting` with `Relaxed` where the batcher uses `SeqCst`.
The batching tests themselves are timing-dependent (`settle()` sleeps 10 ms, :563-565; wall-clock
asserts at :644-648, :674-678, :688-689); `Holds` (:378-400) shows the deterministic pattern.

---

## 8. Tests

### 8.1 Paths with no test at all (replaced by the fabric)
- Where: `src/api.rs:1401-1418, 1352-1360, 1391-1398` (504 `INFERENCE_TIMEOUT` and the panic
  500; `ServerConfig::request_timeout` is never set in a test; the gRPC `deadline_exceeded`
  mapping at `grpc.rs:323-330` likewise); `src/api/v2.rs:44-48, 160-166, 40, 95-101`
  (`/v2/models/{model}/ready`, `/versions/{v}/ready`, `/v2/health/ready`); `src/api/grpc.rs:130-162`
  (`server_ready`, `model_ready`, `server_metadata`); `src/api.rs:2175-2182` (`NOT_RELOADABLE`
  is reached in `test_api_keys` but only asserted `!= UNAUTHORIZED`); `src/api.rs:927-1074`
  (`serve()` only through the CLI binary).
- Severity: medium. Effort: M.
- Proposal: a delaying `DummyModel` (the batching tests already have one, `batching.rs:424`) and
  `request_timeout: Some(1 ms)` asserting 504 on `/predict`, v2 and gRPC; a panicking dummy for
  the 500; v2 and gRPC readiness and metadata asserts; an explicit 409.

### 8.2 LLM paths not covered, and timing-dependent assertions
- Where: `tests/llm.rs:1-746`; untested: `Event::Error` after `Started` at the HTTP level
  (`http.rs:656-664`, error payload then `[DONE]`; `collect` returning 500), `prompt` as an
  array of several strings, the `max_prefill_tokens` budget (`scheduler.rs:435-444`),
  `workers > 1`, the `decode_batch` length mismatch (470-478), `Scheduled` drop while
  streaming, `from_files`. Timing: `tests/llm.rs:706-710` asserts `ttft >= 15.0` from a 20 ms
  delay; `603-606` asserts `steps <= 3` depending on when the socket write fails; `538-544` and
  `526-534` poll with 5 s deadlines; mocks pace with `sleep` (`src/llm/mock.rs:200, 209, 308`).
- Severity: medium. Effort: M.
- Proposal: error-mid-stream and `decode_batch`-mismatch tests first; replace `>= 15 ms` with
  "is Some"; drive the mock with a `Notify` so `wait_for` awaits a signal; keep one real-TCP
  disconnect test without a step bound. The step-counted tests (`decodes()` handles, 564-569,
  675-682) are the pattern.

### 8.3 Nerve mechanisms without a direct test; timing in unit tests
- Where: untested: the habituation recovery curve (one point at 6 τ, `neuron.rs:433-436`), the
  breaker closing after a successful trial (`effector.rs:381`), delayed rate-limit admission
  (`effector.rs:498-500`), `Http::respond_within` and the `timeout(wait, receiver)` branch
  (`http.rs:86-96`), `Timer::same_feature`, `Lines::stdin`, `Process::stdin_payload`,
  `FileAppend` reopen after an error, `Firing::Compute`, `fits()` at `p99 == remaining`,
  `LatencyEstimate` staleness between recomputes; `wire.rs`: `Framing::Close`, the retry-once
  path (348-357), `MAX_HEAD`/`MAX_BODY`, IPv6 authority, chunked trailers, a non-numeric
  `Content-Length`, a head without CRLF-CRLF. Timing: `effector.rs:1156-1178` (60 ms sleep for
  a 50 ms cooldown; `(2..=6).contains(&delivered)` from a 1000/s bucket), `fabric.rs:1034-1035,
  1075-1076`, `receptor.rs:1043-1051, 1154-1159`, `tests/nerve.rs:189-197, 256-266`.
- Severity: medium. Effort: M.
- Proposal: `Gate::admit` and the effector's `admit` already take `now`, so the loop tests can
  pass times instead of sleeping; a raw `TcpListener` fixture writing fixed bytes for wire; in
  `tests/nerve.rs` have the brain signal completion through the `Channel` effector instead of
  polling the journal file and `/metrics`.

### 8.4 Timing-dependent integration tests (replaced by the fabric)
- Where: `tests/integration_tests.rs:261-292` (`test_backpressure_returns_503` needs at least one
  503 and one 200 from 6 concurrent requests on one slot), `304-456` (`(1..8).contains(&batches)`
  depends on 4 workers being saturated by 9 requests), `570-577, 625-634` (bounded 60 s polls,
  fine); `src/api.rs:3031-3046`.
- Severity: medium. Effort: M.
- Proposal: `batching.workers = 1` for the batching test; a model that blocks on a barrier until
  N requests are in flight for backpressure.

### 8.5 Smaller test items

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `src/json_fast.rs:166, 183-219, 258-294` | medium | S | Edge cases reviewed and right (`-0`, nesting, trailing garbage, leading `+`/`.5`/`01`), but `1e400`/`1e39` are untested: `fast_float2` returns `inf` where serde_json rejects, so the fast path may accept what serde rejects, which `agree` (243) forbids. Both end as HTTP 400 today with different messages. Add the cases; reject non-finite in `parse_list` if confirmed. |
| `benches/inference_benchmark.rs:56-82`, `src/api.rs:1506-1508` | medium | S | `request_decode/json` benchmarks `serde_json::from_slice`, the fallback, not `json_fast::parse` the server runs first; the module is private (`lib.rs:183`). A `#[doc(hidden)] pub` re-export for the bench. |
| `tests/integration_tests.rs:3, 22-28, 548, 674` | low | S | **(Replaced by the fabric.)** Needs `resnet18` and `mnist` compiled in or the whole file panics; `test_cli_lists_available_models` hard-codes the exact set. Gate with `#![cfg(all(model_resnet18, model_mnist))]` and relax to `contains`. |
| `src/api.rs:2565-2595` | low | S | **(Replaced by the fabric.)** The openapi test checks a hard-coded path list, not the router; `MetricsResponse`'s `llm`/`nerve` fields are not in the spec. Derive the expected paths from one `const ROUTES`. |
| `tests/graph_optimize.rs:87-185`, `build/optimize.rs:154-216` | low | M | Untested branches: a BN parameter shared by two nodes, a Conv output that is a graph output, training outputs, default epsilon, pruning, a weight whose size is not a multiple of channels. One table-driven test. |
| `src/tls/mod.rs:423, 434-441, 528-529` | low | S | `peer_certificates().unwrap()[0]` only; `key_formats` asserts the RSA error contains "ECDSA"; no P-384 or Ed25519 client certificate for mutual TLS. Add one per key type to `generate.sh`. |
| `src/allocator.rs:77-82` | low | S | The full-cache path is untested (6.2). |
| `src/batching.rs:563-565, 644-689` | medium | M | Timing-dependent (7.9). Overlaps the batching unification. |

### 8.6 Tests that never run anywhere
- `furnace-candle/tests/gguf.rs:121, 127, 164, 170` are `#[ignore]`d and need local GGUF files;
  CI runs only the four unit tests, so the adapter's forward pass against a real model is
  verified by hand only. Proposal: a tiny synthetic GGUF built in the test and compared against
  candle-transformers, or a scheduled workflow with a cached small file (medium, L).
- Doc-tests: `.github/workflows/ci.yml:85` runs `--lib --bins --tests`, never `--doc`, so the
  three `no_run` examples on the crate's front page (`src/lib.rs:16, 57, 101`) are never compiled
  (medium, S). See 10.2.

---

## 9. Docs and comments

### 9.1 Stale or wrong

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `README.md:47, 132` | medium | S | `furnace-server = { version = "0.9" }` resolves to `>=0.9, <0.10`, which has no `llm` or `nerve` feature although README.md:105-112 advertises them. `"0.10"`. |
| `README.md:353-360`, `Dockerfile:3` | medium | S | The Docker quick start builds without `models/`, so `--model-name resnet18` fails at run time; the release workflow downloads the files first (`release.yml:263-267`). Add the two `curl` lines. |
| `docs/API.md:58` vs `src/api.rs:1818-1824, 1916-1922` | medium | S | **(Replaced by the fabric.)** Documents `furnace_build_info{version, model, backend}`; the code emits `furnace_build_info{version}` and `furnace_model_info{model, backend}` per model. |
| `docs/API.md:348-355, 451-467` | medium | S | **(Replaced by the fabric.)** Error tables miss `409 NOT_RELOADABLE`, `500 RELOAD_FAILED`, `400 NAMED_TENSORS_REQUIRED`, `500 INTERNAL_ERROR`, the panic `500 INFERENCE_FAILED`, `404 MODEL_NOT_FOUND` on `/predict` for an LLM-only server; `docs/openapi.json:1468` lists none of the auth, reload or nerve codes. |
| `docs/openapi.json:16-1069`, `docs/API.md:16-17` | low | M | **(Replaced by the fabric.)** The `/v1/*` and `/nerve*` routes and `MetricsResponse`'s `llm`/`nerve` fields are absent from the spec (every documented path does exist in code). |
| `docs/NERVE.md:12, 15-16` | medium | S | `version = "0.9"`; "a 300-line HTTP/1.1 client" (wire.rs is 423 production lines). |
| `docs/LLM.md:181` | low | S | "furnace 0.9.0 plus the LLM layer" at v0.10.0. Everything else in LLM.md matches the code. |
| `src/nerve/brain.rs:511-516` | medium | S | "A brain returns Compute, so the fabric counts it" precedes `Firing::Fire(..)`; `NeuronMetrics::computed` never moves for `ModelNeuron` or `Llm`. Fix the comment or return `Compute` (behaviour change). |
| `docs/NERVE.md:217-222` | low | S | Says brains add `generations`, `commands` and latency to Prometheus; `Llm` and `ModelNeuron` implement `metrics()` (JSON) but not `render_prometheus`. Qualify, or implement (behaviour change). |
| `src/nerve/http.rs:124-133` | low | S | The `quiet:not_interested` arm is unreachable (`Outcome::is_quiet()` is `hops == 0`, which only the receptor's gate produces). Delete the arm and say a `Quiet` reflex answers `204` with `done`. |
| `src/nerve/neuron.rs:153-154, 180-183, 279` | low | S | `Habituation::capacity` doc vs the table's rounding up to a multiple of 4. |
| `src/nerve/mod.rs:55-56` | low | S | The example pads with `let _ = Summation::default();` instead of showing `fabric.deliberative(..)` as `tests/nerve.rs:57-70` does. |
| `src/llm/scheduler.rs:8-9` | low | S | A broken line in the module doc. |
| `src/llm/http.rs:620, 643` | low | S | `Streamed::completion_tokens` is write-only (a write counts as a use for rustc). Remove. |
| `src/api.rs:153` | low | S | The `MetricsResponse` doc cites a pull-request number; the house rule says not to carry those. |
| `docs/API.md:38-50` | low | S | **(Replaced by the fabric.)** The `/metrics` JSON example omits `rejected_unauthorized`, `rejected_rate_limited`, `models`. |
| `README.md:276-284` | low | S | `/health`, `/models/{model}`, the `/v1/*` and `/nerve*` routes are absent from the route table. |
| `docs/SERVERLESS.md:5` vs 14; `CHANGELOG.md:10` (`2026-10-21`, a day after the release commit); `DEVELOPMENT.md:84` (`cargo audit` vs `cargo deny`); `src/tls/provider.rs:178` ("the AEAD limits draft" for RFC 8446 §5.5's figure); `Cargo.toml:258` (`[profile.dist]` for a cargo-dist that is not configured) | low | S | Five one-line drifts. |
| `deploy/helm/furnace/Chart.yaml:7`, `deploy/kserve/*.yaml:25,19`, `docs/SERVERLESS.md:15,31,59-62,71`, `benches/burn/src/cold_start.rs:6`, `benches/compare/src/lib.rs:21` | medium | S | `0.10.0` hand-written in eight places; `benches/compare` hard-codes `"furnace 0.10.0, Burn 0.22"` while the other harnesses read Cargo.toml. A CI grep against `Cargo.toml`'s version in the existing `kubernetes` job. |
| `docs/MODELS.md:23-29` | low | S | MobileNetV2, EfficientNet-Lite4, Tiny YOLOv2, YOLOv4 are listed as "builds" with no test or fixture; nothing is wrong, but add a "last checked" column. |

### 9.2 Missing
- Module headers written as `//` instead of `//!` (`src/weights.rs:1-7`, `src/models/mod.rs:1-9`,
  `build/registry.rs:1-4`, `build/optimize.rs:1-16`); `weights.rs` is a public module users are
  pointed at (`lib.rs:42, 80`), so its mmap explanation is invisible in rustdoc. Low, S.
- Roughly 110 `pub` items without a doc comment: ~20 in api/v2/access/telemetry
  (`api.rs:57-68, 136-151, 221-230, 252-274, 441-456, 906-917`; `v2.rs:59-81, 273-300`), ~20 in
  model/tensor/backend/error (`model.rs:512, 561, 565, 793, 1045-1065`; `tensor.rs:116-119, 142,
  228, 241`; `error.rs:11-60`; `ModelStats` fields, which are the `/metrics` JSON), ~25 in llm
  and furnace-candle (`llm/mod.rs:134-264`, `scheduler.rs:38, 169, 204`, `pipeline.rs:124-198`,
  `furnace-candle/src/lib.rs:59, 184`), ~60 in nerve (`signal.rs`, `fabric.rs:441-509`,
  `effector.rs:265-279`, every field of `NeuronMetrics`/`PathwayMetrics` but three), 5 in tls.
  Low, M. `#![warn(missing_docs)]` in `lib.rs` would keep it that way, and is a warning the house
  rule then requires fixing, which is the point.
- `Server::router` does not tell users to serve with `into_make_service_with_connect_info::<ClientAddr>()`
  (4.1). `Scheduled` does not say what drop does (7.7). `Process` does not say that brain-chosen
  `args` reach the program unfiltered (7.8). `Webhook` does not say no HTTPS, no redirects (5.5).

---

## 10. Dependencies and feature flags

### 10.1 House rule: `#[allow]` (nine sites) and a build warning for a documented configuration
- Where and fix:
  - `src/backend.rs:62` `#[allow(unused_mut)]`: `let mut backends` is only mutated under
    `#[cfg(feature)]`. Build the list by shadowing per feature. S.
  - `src/error.rs:46, 64, 71, 75` `#[allow(dead_code)]`: see 4.2. S.
  - `src/models/mod.rs:61` `#[allow(unused_macros)]`: `pub(crate) use built_in_model;` after
    the macro; a macro re-exported as an item is never reported unused. S.
  - `src/models/mod.rs:65` `#[allow(clippy::all, unused)]` on the burn-onnx generated module:
    the cause is upstream codegen. Build once with the attribute removed, record which lints
    fire, fix what `build.rs` can post-process, and narrow the attribute to the exact lints
    with a comment naming the upstream issue until it is fixed there. M; needs a build.
  - `build.rs:6` `#[allow(dead_code)]` on the registry module: split `build/registry.rs` into
    `render.rs` (always used) and `spec.rs` (only with `onnx`), include the second under
    `#[cfg(feature = "onnx")]`. M.
  - `build/optimize.rs:141` `#[allow(clippy::too_many_arguments)]`: a `GraphIndex<'a>` struct
    built once in `fold_conv_batchnorm`. S.
  - `build.rs:48-52, 83-89`: `cargo:warning=onnx feature not enabled` on every build for the
    `default-features = false` configuration the README recommends (`README.md:47, 132`), and
    "not found, no built-in models" for a checkout without `models/`; `rerun-if-changed` on the
    missing directory (`build.rs:30`) reruns the script every build. Print nothing when the
    feature is off; warn about the directory only when `FURNACE_MODELS_DIR` was set explicitly;
    emit the directory `rerun-if-changed` only when it exists. Build-output change only. S.
- Severity: high (house rule). Effort: M in total.

### 10.2 Feature combinations never compiled anywhere
- Where: `Cargo.toml:78-102`; `.github/workflows/ci.yml:74-75, 79, 85, 167`;
  `.github/workflows/bench.yml:100, 108-109`.
- Severity: medium. Effort: S.
- Finding: CI builds default (`onnx`), `onnx + image grpc tls llm nerve`, `--no-default-features
  --features wgpu` (clippy only), and weekly `onnx + grpc tls`, `--no-default-features` (burn
  bench) and `--no-default-features --features llm` (llm bench). Never built: `tls` alone,
  `grpc` alone, `grpc` without `tls`, `nerve` alone (`benches/nerve` builds furnace that way and
  no workflow builds it), `image` alone, `embed-weights` (nowhere), `llm` without `nerve` in
  furnace's own tests, `metal`. `cfg(feature = "llm")` appears 16 times and `nerve` 12 times in
  `src/`, so a `nerve`-only build has code paths no CI job compiles.
- Proposal: a `features` job with `cargo check --all-targets` over `--no-default-features`,
  `--features tls`, `grpc`, `nerve`, `llm`, `image`, `embed-weights`, and `metal` on
  `macos-latest`. Type-checks only, a few minutes.

### 10.3 Dependency hygiene

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `Cargo.toml:116`, `src/api.rs:2313`, `src/api/v2.rs:885`, `src/api/grpc.rs:647` | low | S | `tower = { features = ["limit"] }` in `[dependencies]`, but the only `tower::` paths in `src/` are in test modules, served by the dev-dependency. Delete the entry. |
| `Cargo.toml:151`, `src/api/grpc.rs:60-72` | low | S | `tokio-stream` is attached to `tls` but used only in grpc.rs under `cfg(feature = "tls")`; list it under `grpc`. |
| `Cargo.toml:178`, `Cargo.lock` | medium | S | The `reqwest` dev-dependency keeps its default TLS stack; the lockfile shows `rustls` resolved with `ring` and `aws-lc-rs`, and `aws-lc-sys`, `cmake`, `cc` in the graph, for tests that only call `http://127.0.0.1`. `default-features = false, features = ["json"]` as the three bench harnesses already do. If `cargo tree -i aws-lc-sys` still finds a path afterwards, it is Burn's optional stacks and can be documented. Test-only. |
| `Cargo.toml:182` | low | S | `criterion` with `html_reports` pulls plotters into every `cargo test` build for an 89-line bench. Drop the feature. |
| `Cargo.toml:17-34, 66` | low | S | `package.exclude` lists `/benches/compare/` only and `workspace.exclude` lists `burn` and `compare` only; `cargo publish` ships `benches/burn`, `llm` and `nerve` with their lockfiles (~750 KB). `"/benches/"` in `exclude` (keeping `inference_benchmark.rs`) and all four in `workspace.exclude`. |
| `Cargo.lock` | low | S | Duplicate majors: `getrandom` ×3, `rand` ×3, `sha2`/`hmac`/`aes-gcm`/`digest` ×2, `fancy-regex` ×2, `thiserror` ×2, 60 names in all; nearly all from Burn's unbuilt backends (`tch`, `turso_core`). `multiple-versions = "warn"` is never enforced; a `skip` list for Burn's keeps the output readable. |
| `furnace-candle/Cargo.toml:10-25`, `deny.toml:46-48` | medium | M | candle comes from a personal fork at a pinned revision (Oniguruma-free main plus an unmerged zip-9 fix); the pin is exact and the pure-Rust scan guards it, but the fork can vanish and the crate cannot be published. Track the upstream state in an issue linked from the comment; consider mirroring the revision under the project's own organisation. |
| `Cargo.toml:99`, `src/nerve/effector.rs:915`, `src/nerve/wire.rs:512-513` | low | S | `nerve = ["dep:futures-util"]` says "nothing else"; true for normal deps (`chrono` is unconditional). The nerve unit tests use `tokio_stream::iter` from the dev-dependency that is also an optional `tls` dep: note it so the two entries are not "simplified" apart. |
| `src/tls/provider.rs:66-70, 363, 410, 457` | low | S | `random::<N>()` returns secret bytes through a plain stack array that is not zeroized; `zeroize::Zeroizing` is already in the tree. |
| `build.rs:125, 128-130, 136, 201-204, 152-161` | low | S | The ONNX file is parsed up to four times per model; `graph_io_names` could take the `ModelProto` `optimized_graph` already holds. |
| `build.rs:16-18, 35-37` | low | S | `KNOWN_MODEL_CFGS` must be edited by hand for every test that mentions a model; grep `cfg(model_` in `src/` and `tests/` instead. |
| `build.rs:94` | low | S | `CARGO_FEATURE_EMBED_WEIGHTS` read as an env var where line 9 uses `#[cfg(feature)]`; `cfg!(feature = "embed-weights")` for consistency. |

---

## 11. CI

| Where | Sev | Eff | Finding and proposal |
|---|---|---|---|
| `ci.yml:85` | medium | S | Doc-tests never run (`--lib --bins --tests`); add `cargo test --profile ci --doc --features "image grpc tls llm nerve"` in one job. |
| `ci.yml:67-79`, `bench.yml:108-109` | medium | S | The four bench workspaces are never `fmt`'d or `clippy`'d; `compare` and `nerve` are not even built. Add `fmt --check` and `clippy --all-targets -D warnings` with `--manifest-path` for burn, llm, nerve (ubuntu stable only); `cargo check` for compare. |
| `ci.yml:109-149`, `release.yml:115`, `Dockerfile:18` | medium | S | The pure-Rust scan covers `llm,nerve` and furnace-candle but not the release feature set `grpc tls` or `image`; the pure-Rust claim for TLS rests on nobody re-enabling `ring` transitively. Add `--features grpc,tls` and `--features image` to the loop at ci.yml:133. |
| `release.yml:115`, `bench.yml:100` | medium | S | Release binaries are built without `--locked`, so they may resolve a different graph than CI tested and than the SBOM (release.yml:152-158) describes. Add `--locked`. |
| `ci.yml:257` | low | S | Coverage runs with default features only; tls, grpc, llm, nerve, image never appear in the report. Add the feature list. |
| `ci.yml:267-294` | low | S | `cargo bench --bench inference_benchmark` on every push to main; output is neither uploaded nor compared. Delete, or `--no-run` as a compile check. |
| `ci.yml:46` | low | S | `dtolnay/rust-toolchain@master` in the `test` job; every other job uses `@stable`. Pin. |
| `security.yml:47-59`, `deny.toml:38, 11` | low | S | `dependency-review-action` has `continue-on-error: true` so `fail-on-severity` never blocks; `multiple-versions` and `yanked` only warn. Advisories run daily with no ignores (good). |
| `bench.yml:3-5, 49-55`, `benches/ci-summary.sh:17-45` | low | M | Ratios are computed and uploaded but nothing compares a run with the previous one. When the unification lands, a ratio floor (furnace/hand-written ≥ 0.85 on echo) that fails the job. |
| `ci.yml:36-39`, `benches/*/Cargo.toml` | low | S | MSRV 1.95 is consistent everywhere, but only the root crate is checked on it. |
| `kserve-e2e.yml:7-12`, `deploy/kserve/*.yaml` | low | S | The e2e runs only when `deploy/kserve` changes and against the released image, never the branch. A `workflow_dispatch` input for the image tag and a note in `deploy/README.md`. |
| `Dockerfile:15-28` | low | M | `COPY . .` then build: any source change recompiles Burn, rustls and tonic with fat LTO. Copy manifests and a stub first to cache dependencies. |
| `ci.yml:87-107` | low | M | No job runs the ignored furnace-candle tests or the `real` LLM bench (8.6). |

---

## 12. The plan

Waves are pull-request-sized and land one at a time, in order. They cover only code that
survives the fabric rebuild (see "Scope" at the top); nothing here splits, renames or polishes
the handlers in `src/api.rs`, `src/api/v2.rs`, `src/api/grpc.rs` or `src/llm/http.rs`. Waves are
file-disjoint from each other except where "after wave N" says otherwise. Two gates apply:

- **Batching gate.** No wave touches `src/batching*` or `src/nerve/fabric.rs` until the batching
  unification (branch `claude/busy-murdock-c6b7da`) has merged.
- **Fabric-rebuild gate.** Waves 6 to 8 edit `src/nerve/*`, which the rebuild of the serving
  routes is also changing. Before each of them starts, the rebuild branch's owner decides one of:
  the wave lands first and the rebuild rebases onto it, or the rebuild absorbs the wave's
  findings and the wave is dropped. Waves 1 to 5 do not touch `src/nerve/` and need no such
  decision.

Every wave must pass: `cargo fmt --all -- --check`; `cargo clippy --all-targets -- -D warnings`
for default, `--features nerve` and `--features "image grpc tls llm nerve"`; `cargo test
--features "grpc tls llm nerve"`; `cargo deny check`; and keep `benches/burn` `overhead echo` and
`benches/nerve`'s arc numbers within noise (run only when no `furnace-*-bench` process is
running). Waves marked "no public API change" also diff the public item list before and after
(`cargo doc` or `cargo public-api`) and show it empty. Behaviour and API changes are stated per
wave and go in CHANGELOG under Unreleased.

### Wave 1: House rules, dead code, dependency hygiene
- Files: `src/error.rs`, `src/backend.rs`, `src/models/mod.rs`, `src/models/name.rs` (new),
  `build.rs`, `build/registry.rs` → `build/render.rs` + `build/spec.rs`, `build/optimize.rs`,
  `Cargo.toml`, `deny.toml`.
- Does: removes all nine `#[allow]`s by fixing their causes (10.1), including the registry split
  and `GraphIndex`; measures the lints on the generated module and narrows that one attribute
  with an upstream reference; silences the build-time warnings for documented configurations and
  fixes `rerun-if-changed`; shares `normalize_name` (2.6); deletes the five dead error variants
  and the dead `IntoResponse` impl (4.2, 2.1; nothing in the handlers uses either); drops `tower`
  from `[dependencies]`, moves `tokio-stream` to `grpc`, turns off `reqwest`'s default features
  in dev, drops `html_reports`, fixes both `exclude` lists, adds a `skip` list for Burn's
  duplicates (10.3); the build-script parsing and `KNOWN_MODEL_CFGS` items (10.3) and the
  keyword list (5.5).
- Says: **public API change** (`ApiError`/`CliError` lose variants; `FurnaceError` no longer
  implements `IntoResponse`); build-output change.
- Accepts: zero `#[allow` in `src/` and `build/` except the one narrowed attribute with its
  comment; `cargo build --no-default-features` prints no warning; `cargo tree -e
  normal,build,dev -i aws-lc-sys` finds nothing, or the remaining path is documented; one
  `normalize_name` and no drift test.

### Wave 2: One request pipeline in `Model`
- Files: `src/model.rs`, `src/tensor.rs`, `src/types.rs`, `src/lib.rs` (re-export line only),
  `src/weights.rs`.
- Does: 1.2 (flat methods as adapters over the named path, with the simple-aware error mapping
  so `/predict` codes and messages stay while it exists), 1.3 (`take_inputs`,
  `into_float_flat`), 5.2 (delete the simple warm-up branch), 5.3 (rank check), the integer
  range checked once and `int_range` cached (2.6), `TensorSpec` into `tensor.rs` with
  `pub mod types` kept (1.5), `Utc::now()` before the stats lock (7.8), the mmap-fallback
  message (5.5), clone the error instead of its text for a failed shared pass (5.5), `//!`
  headers and the missing docs on `Model`, `HostTensor`, `ModelStats` (9.2). `ModelNeuron` uses
  the named door only, so the brain is unaffected.
- Says: **behaviour changes**, all bug fixes: a fixed-batch mismatch on `/predict` is 400, not
  500 (5.1); warm-up honours `fixed_batch`; a rank-0 tensor is rejected instead of panicking.
  No public API change. Once `/predict` is retired, the flat `predict*` methods can be removed
  in a later public API change.
- Accepts: a test per behaviour change; the existing `/predict` error-code tests pass unchanged
  (they are not edited); `model.rs` loses at least 150 lines.

### Wave 3: LLM engine side, TLS, access and telemetry
- Files: `src/llm/{mod,scheduler,pipeline,mock}.rs`, `src/tls/mod.rs`, `src/tls/provider.rs`,
  `tests/fixtures/tls/generate.sh` and its fixtures, `src/access.rs`, `src/telemetry.rs`,
  `furnace-candle/src/*` (docs only), `docs/LLM.md`.
- Does: `Scheduled` delegates `name`/`max_context` and `Running` becomes generic over `M` (1.5);
  lazy `retain` of closed jobs (7.8); documents the contracts on `guarded` (no effect under
  `panic = "abort"`, 7.4), `Scheduled` (what drop does, slow readers, 7.7),
  `LlmEngine::generate` (callers inside a runtime, implementations may spawn, 7.7) and
  `scheduler_stats` (4.5); the `scheduler.rs` module-doc line and the `templates` rename (9.1,
  3.2); the BOS explanation onto `TextPipeline::encode` and the other missing docs in `src/llm`
  and `furnace-candle` (9.2); the `docs/LLM.md` version and the panic caveat. TLS: `pub(super)`
  on the provider's items (1.5), `reload_if_changed` under one lock (7.8), `KEY_LEN` from
  `KeySizeUser`, a two-variant hash enum and the PKCS#8 message (5.5), `Zeroizing` for secret
  bytes (10.3), missing docs, client certificates per key type in the mutual-TLS test (8.5).
  `access.rs`/`telemetry.rs`: missing docs and the `Histogram::snapshot` tolerance note (7.8).
  Engine-level tests for the `decode_batch` length mismatch, the `max_prefill_tokens` budget and
  `workers > 1`, written against `Scheduled` directly, not through the OpenAI routes (8.2).
- Says: the PKCS#8 error message changes; nothing else. No public API change.
- Accepts: the three new scheduler tests count steps rather than sleep; the mutual-TLS test runs
  once per key type.

### Wave 4: CI
- Files: `.github/workflows/ci.yml`, `security.yml`, `release.yml`, `bench.yml`,
  `kserve-e2e.yml`, `Dockerfile`.
- Does: a `features` job type-checking `--no-default-features`, `tls`, `grpc`, `nerve`, `llm`,
  `image`, `embed-weights`, and `metal` on macOS (10.2); `cargo test --doc` (8.6); fmt and clippy
  for `benches/burn`, `benches/llm`, `benches/nerve` and `cargo check` for `benches/compare`; the
  pure-Rust scan over `grpc,tls` and `image`; `--locked` for release builds; coverage with the
  feature list; the push-to-main criterion job becomes `--no-run`; the toolchain action pinned;
  a version grep against `Cargo.toml` in the `kubernetes` job; dependency caching in the
  Dockerfile (section 11).
- Says: no library change. If the `features` job finds a combination that does not build, the
  fix goes in this wave and is named in the PR.
- Accepts: every new job is green on the PR; the doc-tests compile the three `lib.rs` examples.

### Wave 5: Surviving docs and `benches/common`
- Files: `README.md` (version, Docker quick start), `CHANGELOG.md` (the 0.10.0 date),
  `DEVELOPMENT.md`, `docs/SERVERLESS.md`, `docs/MODELS.md`, `docs/PERFORMANCE.md`,
  `deploy/README.md`, `benches/common/` (new) and the four harnesses' `Cargo.toml` and
  `src/main.rs` (and `benches/compare/src/lib.rs`), `examples/serve_burn_model/models.rs`
  (comment only).
- Does: the surviving rows of 9.1; `benches/common` with the memory probe, one documented
  percentile, `median`, `wait_until_healthy`, `free_port`, `load`, `stop` (with `TERM` first
  everywhere), `fail` and JSON writing (2.5); the version read from Cargo.toml in
  `benches/compare`; the duplicated `DigitNet`/`TextModel` pointed at each other (2.6). Leaves
  `docs/API.md`, `docs/openapi.json` and the README route table to the rebuild, and
  `docs/NERVE.md` to wave 8.
- Says: **behaviour change** in the harnesses only where a percentile definition changes,
  documented in `docs/PERFORMANCE.md`. No library change.
- Accepts: `cargo publish --dry-run --locked` lists no `benches/*/` file but
  `inference_benchmark.rs`; the four harnesses produce the same numbers within noise before and
  after; one `unsafe` block for the memory probe instead of four.

### Wave 6: Nerve correctness (fabric-rebuild gate)
- Files: `src/nerve/effector.rs`, `src/nerve/wire.rs`, `src/nerve/brain.rs`,
  `src/nerve/http.rs`, `src/nerve/neuron.rs`, `tests/nerve.rs`.
- Does: 5.4 (`Respond` with nobody waiting is not a failure), 7.1 (whole-request timeout in
  `wire`), 7.8 (`Process` stdin write inside the timeout and its `args` doc; refractory
  `compare_exchange`), 5.5 (`Webhook` doc: no HTTPS, no redirects), 9.1 (the `brain.rs` comment,
  the unreachable `quiet:not_interested` arm, the `Habituation::capacity` doc), `# Panics`
  sections on `Webhook::new`, `Effector::retry/breaker/rate_limit/refractory`, `Webhook::header`
  and `FabricBuilder::add/build` (4.4). Tests for 5.4 and 7.1, and `wire` fixtures for
  `Framing::Close`, the retry-once path, the size limits and a truncated head (8.3).
- Says: **behaviour changes**: a late `Respond` no longer counts as `failed`; a stalled webhook
  body now times out and retries. No public API change.
- Accepts: a deliberative pathway ending in `Respond` with `respond_within(ZERO)` keeps answering
  after six queued requests; a fixture server that stalls after the head makes the webhook fail
  with `TimedOut` within its timeout; `benches/nerve` arc numbers within noise.

### Wave 7: Split `effector.rs` and `receptor.rs` (fabric-rebuild gate; after wave 6)
- Files: `src/nerve/effector.rs` → `src/nerve/effector/{mod,respond,webhook,process,
  file_append,channel,sse}.rs`; `src/nerve/receptor.rs` → `src/nerve/receptor/{mod,http,
  channel,timer,lines,sse}.rs`; `src/nerve/mod.rs` (a `#[cfg(test)] mod testing`).
- Does: 1.4; renames the effector's private `Inner`/`Counters` (3.1); one `wait_until` and
  `fabric_with` for the unit tests (2.6); `Backoff` and `nanos_since` helpers (2.6). This is the
  wave most likely to conflict with the rebuild, which adds receptors and effectors; if the
  rebuild lands first, its new kinds go straight into the per-kind files.
- Says: no public API change; no behaviour change.
- Accepts: identical public item list; no file over 500 lines; `git log --follow` works on the
  moved files.

### Wave 8: Fabric and nerve metrics (batching gate and fabric-rebuild gate)
- Files: `src/nerve/fabric.rs`, `src/nerve/neuron.rs`, `src/nerve/signal.rs`,
  `src/nerve/metrics.rs`, `src/nerve/effector/mod.rs`, `src/telemetry.rs`, `src/nerve/mod.rs`,
  `docs/NERVE.md`. After waves 3 and 7.
- Does: 7.2 (`route` also drops at `trace.len() >= MAX_HOPS`); `FnNeuron` to `neuron.rs`,
  `impl Payload` to `signal.rs` (1.5); `PathwayKind::as_str()` (3.2); `telemetry::render_family`
  and `render_labelled`, used by `nerve/metrics.rs` and the effector (2.4; `llm/metrics.rs` is
  left to the rebuild, which replaces its HTTP-side half); the `mod.rs` example showing a real
  deliberative pathway, and the `docs/NERVE.md` items the unification did not already fix
  (version, client size, the brain Prometheus sentence) (9.1); the missing docs in `src/nerve`
  (9.2).
- Says: **behaviour changes**: a signal looping through a queue is dropped and counted in
  `loops`; the Prometheus `kind` label becomes lowercase. No public API change.
- Accepts: a reflex → brain → reflex cycle terminates with `loops == 1`; a snapshot of the
  `furnace_nerve_*` Prometheus text is byte-identical before and after except the `kind` label;
  the unification's tests still pass.

### Wave 9: Tests for surviving code
- Files: the test modules of `src/nerve/effector/*.rs`, `src/nerve/receptor/*.rs`,
  `src/nerve/neuron.rs`, `src/nerve/fabric.rs`, `src/nerve/latency.rs`, `src/allocator.rs`,
  `src/batching*` (coordinated with the unification's owner), `tests/nerve.rs`,
  `tests/graph_optimize.rs`. After waves 7 and 8.
- Does: clock injection instead of sleeps in the nerve loop and gate tests, and the untested
  mechanisms of 8.3 (habituation curve, breaker closing, delayed admission, `respond_within`,
  `Timer::same_feature`, `Process::stdin_payload`, `FileAppend` reopen, `Firing::Compute`, the
  `fits()` boundary, `LatencyEstimate` staleness); `tests/nerve.rs` waits on a `Channel` effector
  instead of polling a file and `/metrics`; the batching tests' `settle()` replaced by holds
  (7.9); the allocator's full-cache path (6.2); the optimizer table test (8.5).
- Says: no library change.
- Accepts: no `sleep` in a test that asserts a count or a state (bounded polls waiting for a
  signal are fine); `cargo test` passes three times in a row while a bench loads the machine.

### 12.1 Checklist for the fabric rebuild
Findings in the replaced handlers that are requirements for the receptors and effectors that take
over. Each points at the section with the anchors.

1. **One error mapping.** Status and `code` from a typed error in one place, keeping today's
   codes (`INPUT_VALIDATION_FAILED`, `INVALID_INPUT_VALUE`, `INFERENCE_FAILED`,
   `INTERNAL_ERROR`, `INFERENCE_TIMEOUT`, ...), not four copies (2.1, 5.5). v2 picks its `code`
   from the status alone today (`src/api/v2.rs:171-175`).
2. **Rate limits need the peer address.** The limiter must get the client IP however the router
   is mounted, or warn when it cannot (4.1).
3. **Timeouts everywhere.** `request_timeout` must cover the LLM routes too (today only tensor
   inference, `src/api.rs:1352, 1391`; `src/llm/http.rs:461, 722-738` waits forever), with an
   idle timeout between stream events, so a stalled engine cannot hold a concurrency slot.
4. **Encode before committing to 200.** Template rendering and tokenization off the runtime, and
   the HTTP status decided after the first engine event, so `TooLong` is a 400 and not an error
   after a 200 (7.3).
5. **Admission once.** Concurrency permit, timing and stats recorded by one path whatever the
   protocol (2.2); 401/429 counted consistently with the docs (7.8).
6. **One decoder per format.** Named tensors decoded once for v2 JSON and gRPC (2.3), one raw
   f32 codec (2.6); `json_fast` either reused with non-finite numbers rejected, or dropped (8.5).
7. **Per-request overhead to avoid.** In today's layer: an INFO `request` span formatting three
   fields per request (`src/api.rs:873-879`), a full `HeaderMap` clone to read one header
   (`src/api.rs:1216`), `Utc::now().to_rfc3339()` and `"success".to_string()` per response
   (`src/api.rs:1298-1300, 2302`), a `Vec<Vec<f32>>` copy of every batch output
   (`src/api.rs:1286-1292`), `inputs.concat()` after parsing (`src/api.rs:1524`),
   `OCTET_STREAM.to_string()` per binary response (`src/api.rs:1705`). The nerve hand-over
   attributes the 11% throughput gap at 16 clients to this layer; a profile should confirm which
   of these dominates.
8. **One thread model.** Unbatched inference is a `spawn_blocking` task entering the rayon pool
   while batched inference runs on dedicated threads (7.8); the fabric's pathways decide this
   once.
9. **Tests the old handlers never had.** 504 on timeout, 500 on a panicking model, v2 and gRPC
   readiness and metadata, 409 `NOT_RELOADABLE`, deterministic saturation instead of racing
   requests (8.1, 8.4); an error event mid-stream and `[DONE]` on the OpenAI routes (8.2).
10. **Library entry points do not panic.** An empty server is an error from `router()` as it is
    from `serve()` (1.5); a failed signal-handler install is logged, not a panic in a task (5.5).
11. **Docs generated from one route list.** `docs/API.md` and `docs/openapi.json` drifted
    (`furnace_build_info` labels, missing error codes, missing `/v1` and `/nerve` routes and
    fields, 9.1); a `ROUTES` constant shared by the router and the spec test prevents it (8.5).
12. **Names on the wire.** One name for the concurrency limit across config, JSON and Prometheus
    (3.1); `quiet` vs `quiet_by_*` in the nerve JSON (3.2); not three unrelated `Reply` types
    (3.1).
13. **TLS accept is bounded.** In-flight handshakes need a limit (7.6), whatever listener the new
    server uses.

### 12.2 Deferred: public API changes for 0.11
Not in the waves because each changes the public surface and deserves its own decision, and
several will be shaped by the rebuild: `pub(crate) mod wire` (4.3); `Webhook::new` and
`FabricBuilder::build` returning `Result`, and a stop signal on `Receptor::start` (4.4);
`#[non_exhaustive]` on `Event` and `SchedulerConfig` (4.5); `Model::warm_up -> crate::Result`,
`Model::new` as `pub(crate)`, and the flat `predict*` methods once `/predict` is gone (4.5,
wave 2); `DummyModel` behind a `test-models` feature and the `"sample"` fallback narrowed (4.5);
renaming `receptor::Channel`/`effector::Channel`, `Sse`, `nerve::Reply`, `access::RateLimit`/
`effector::RateLimit`, `NamedInput`, `llm::Input`, `SchedulerStats`, `Live` (3.1, 3.2); a typed
`Neuron::metrics` (4.5); `ActError`, `wire::Error`, `tls::Error` enums and
`EngineError::Failed(BoxError)` (5.5, 4.5); nerve latency on `telemetry::Histogram` (2.4);
`Firing::Compute` for brains (9.1); `FabricBuilder::efferent(Handle)` (7.8).
From the 12.1 checklist: the concurrency limit's JSON field `max_concurrent_inferences` renamed
to `max_concurrent_requests`, the name the flag, the config and Prometheus use (12.1 item 12;
the private field already took it); `Server::router` and `router_with_models` returning
`Result<Router>` (item 10; today `Server::try_router` gives the error and `router()` serves
nothing instead of panicking); `quiet` vs `quiet_by_*` in the nerve JSON and the three `Reply`
types (item 12, nerve side).

Decided for 0.11. Outside `src/nerve`, done: `Server::router`, `api::router` and
`router_with_models` return `Result<Router>` (`try_router` is gone); `Model::warm_up ->
crate::Result`; the dummies behind `test-models` (`src/model/testing.rs`) and the fallback
only for a path named `sample_model`, with a warning; `max_concurrent_requests` in the JSON;
`#[non_exhaustive]` on `Event` and `SchedulerConfig` (with `with_*` setters);
`EngineError::Failed(BoxError)`; `tls::Error`; the renames `llm::Prompt`,
`SchedulerCounters`, `mock::SharedCount`, `access::ClientRateLimit`, without aliases (no
compatibility shims before 1.0). `NamedInput` no longer exists (the batching unification
removed it). The `src/nerve` items follow in a second change. Not done: `Model::new` stays
public, the flat `predict*` methods stay while `/predict` is served, and a typed
`Neuron::metrics` waits.
