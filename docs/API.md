# HTTP API

All endpoints are on the address given by `--host` and `--port` (default `127.0.0.1:3000`).
The same API as an OpenAPI 3 spec: [openapi.json](openapi.json).

| Endpoint | Purpose |
|---|---|
| `GET /healthz` (alias `/health`) | Liveness: the process is up |
| `GET /readyz` | Readiness: every model's warm-up inference has run |
| `GET /metrics` | Request, error and latency counters |
| `GET /model/info` | Input and output shapes, backend, per-model statistics |
| `POST /predict` | Inference for models with one float tensor in and out: JSON, raw `float32`, or (with `--features image`) a JPEG/PNG |
| `GET /models`, `GET /models/{model}`, `POST /models/{model}/predict` | The same, per model, when serving several |
| `POST /admin/models/{model}/reload` | Reload a model without downtime (with `--admin-token`) |
| `/v2/...` | Inference over the KServe / Triton Open Inference Protocol, for every model, by named tensors |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | LLMs in the OpenAI format (`--features llm`, [LLM.md](LLM.md)) |
| `GET /nerve`, `GET /nerve/effectors/{name}/events`, the fabric's own routes | The nervous system ([NERVE.md](NERVE.md)), which every route above runs on: its description with measured latencies, an SSE effector's commands, and each HTTP receptor's route |

## `GET /healthz` (alias `GET /health`)

```json
{ "status": "healthy", "model_loaded": true, "backend": "cpu", "uptime_seconds": 42, "timestamp": "2026-10-14T03:34:47Z" }
```

## `GET /readyz`

`200 {"status": "ready"}` once the warm-up inference has run, `503 {"status": "warming_up"}`
before. At startup the server runs one all-zero input through the model in the background, so
lazy initialization (weight transforms, buffer pools, the first touch of the memory-mapped
weights) is not paid by a real request. A model with an input whose shape is only fixed per
request has no input to warm up with and is ready at once. Requests are accepted from the start; point a load
balancer's or Kubernetes' readiness probe here and the liveness probe at `/healthz`.

## `GET /metrics`

Server counters for monitoring, as JSON:

```json
{
  "uptime_seconds": 3600,
  "total_requests": 1520, "total_errors": 12, "rejected_busy": 9,
  "rejected_unauthorized": 0, "rejected_rate_limited": 3,
  "in_flight_inferences": 2, "max_concurrent_requests": 100,
  "total_inferences": 1508, "total_batches": 611, "average_batch_size": 2.6, "max_batch_size": 8,
  "average_inference_time_ms": 84.2,
  "last_inference_at": "2026-10-14T05:01:12Z",
  "request_latency_ms": { "p50": 9.1, "p95": 14.2, "p99": 23.8 },
  "inference_latency_ms": { "p50": 8.3, "p95": 12.9, "p99": 21.0 },
  "ready": true,
  "models": [{ "name": "resnet18", "ready": true, "inference_count": 1508, "error_count": 3, "batch_count": 611, "average_batch_size": 2.6, "average_inference_time_ms": 84.2, "inference_latency_ms": { "p50": 8.3, "p95": 12.9, "p99": 21.0 } }],
  "timestamp": "2026-10-14T05:01:13Z"
}
```

With LLMs (`--features llm`) an `llm` array carries their counters ([LLM.md](LLM.md)), and
`nerve` describes the nervous system as `GET /nerve` does.

Prometheus gets the same counters in its text format: a request whose `Accept` header lists
`text/plain` or `application/openmetrics-text` before any JSON type, as Prometheus' scraper
sends, gets `furnace_requests_total`, `furnace_in_flight_requests`, `furnace_batches_total`
and the rest, plus the latency histograms `furnace_request_duration_seconds` and
`furnace_inference_duration_seconds` (buckets from 0.5 ms to 10 s), `furnace_build_info{version}`
and, per model, `furnace_model_info{model, backend}`. `--max-concurrent-requests` is
`max_concurrent_requests` in the JSON and `furnace_max_concurrent_requests` here. Point a
scrape job at `/metrics`; no exporter is needed.

The percentiles in the JSON are estimated from those histograms, so they are accurate to the
bucket they fall in.

`total_requests` and `total_errors` count every inference request (`/predict`,
`/models/{model}/predict`, v2 `infer` over HTTP and gRPC), including those rejected before
inference (400, 413, 503). `rejected_busy` is the 503 subset. Requests refused by the API key or
rate limit check (401, 429) come before any route and are counted only in
`rejected_unauthorized` and `rejected_rate_limited`. `total_inferences` counts
requests that reached the model and `total_batches` the forward passes that served them (see
[Dynamic batching](#dynamic-batching)); `average_batch_size` is items per forward pass.
`in_flight_inferences` includes requests waiting in the batching queue.

## `GET /model/info`

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

## `POST /predict`

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

### Binary input and output

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

### Image input

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

## Dynamic batching

With `--max-batch-size` above 1, concurrent `/predict` requests are merged into shared
forward passes once the server is saturated. Results are the same as unbatched, and each
response still carries only its own outputs, in order.

Batching is **off by default** (`--max-batch-size 1`), because whether it pays depends on the
model. A convolutional network on a CPU runs a batch of 4 in about the time of 2 single passes,
so batching serves more requests under load (ResNet-18: about 10-20% more at 16 clients, with
shorter tail latency). A transformer on a CPU takes nearly 4 times as long for 4, so merging
only makes requests wait (a 4-layer text encoder: about 25% fewer requests and twice the p95).
Turn it on, e.g. `--max-batch-size 8` (in Rust, `Server::new().max_batch_size(8)`), for
models of the first kind and on GPUs;
`benches/burn` measures a model both ways.

Up to `--batch-workers` forward passes (default 4) run at the same time. While a worker is
free, each request gets a pass of its own, started at once: on a CPU, two small passes side
by side use the cores better than one pass over both. Once every worker is busy, requests
queue, and the next worker to finish takes everything queued, up to `--max-batch-size`
items, as one batch. If it finds a single request queued while another is still
being received or decoded, it waits for that one, at most `--batch-timeout-us` (default
2000 µs) after the first was queued. A lone request never waits. A request with several
items counts as that many; one larger than the maximum runs alone.

Each request is validated on its own, so a bad one (wrong size, NaN) gets its own 400 and the
others in its batch are unaffected. `inference_time_ms` includes time spent in the queue.
Queued requests hold one of the `--max-concurrent-requests` slots, and `--request-timeout-ms`
covers queueing and inference; a request that times out while queued is dropped from its
batch. A model's batching is its deliberative pathway in the nervous system: the same engine,
with one worker and a learned window, sums the signals of your own pathways (see
[NERVE.md](NERVE.md#the-server-runs-on-it)).

A batch size that a graph exported for a fixed batch cannot run is the client's error, `400`,
on every route; a model whose forward pass panics answers `500 INFERENCE_FAILED` and goes on
serving (where panics unwind; furnace's release builds abort).

With batching off, every request runs its own pass on a blocking thread, and up to
`--max-concurrent-requests` of them run side by side, competing for the cores, as in a
hand-written server: under heavy load some requests take far longer than others. See
[`PERFORMANCE.md`](PERFORMANCE.md#concurrent-load) for measurements.

## Open Inference Protocol (KServe / Triton v2)

The server also speaks the [Open Inference Protocol](https://github.com/kserve/open-inference-protocol)
over HTTP, so clients written for NVIDIA Triton or KServe work unchanged:

| Endpoint | |
|---|---|
| `GET /v2` | Server metadata (`extensions: ["binary_tensor_data"]`) |
| `GET /v2/health/live`, `GET /v2/health/ready` | Liveness, readiness (same as `/healthz`, `/readyz`) |
| `GET /v2/models/{model}[/versions/1]` | Model metadata: inputs and outputs with datatypes and shapes, `-1` for the batch |
| `GET /v2/models/{model}[/versions/1]/ready` | Model readiness |
| `POST /v2/models/{model}[/versions/1]/infer` | Inference |

`{model}` is the served model's name (`resnet18`) and its only version is `1`. A model with one
`float32` input and output takes one `FP32` tensor named `input` with shape `[batch, ...]` (or
the item shape alone), with `data` flat or nested, and goes through the same batching, limits,
timeout and metrics as `/predict`. Other models take named tensors; see below. Triton's binary tensor data extension, which `tritonclient.http` uses by default, is
supported in both directions.

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:3000")
x = np.random.rand(2, 3, 224, 224).astype(np.float32)
inp = httpclient.InferInput("input", list(x.shape), "FP32")
inp.set_data_from_numpy(x)
result = client.infer("resnet18", [inp])
logits = result.as_numpy("output")     # shape (2, 1000)
```

Errors are `{"error": "..."}` (with Furnace's `code`, `details` and `timestamp` alongside); the
`code` is the one `/predict` gives for the same error (see [Errors](#errors)), and
`INVALID_REQUEST` for a request v2 cannot read.

### Models with several inputs or outputs

A model with several inputs or outputs, integer tensors, or dimensions fixed only per request
takes its tensors by name. `GET /v2/models/{model}` lists them, e.g. for a text classifier:

```json
{ "name": "classifier",
  "inputs":  [ { "name": "input_ids",      "datatype": "INT64", "shape": [-1, 128] },
               { "name": "attention_mask", "datatype": "INT64", "shape": [-1, 128] } ],
  "outputs": [ { "name": "logits",         "datatype": "FP32",  "shape": [-1, 2] } ] }
```

Send every input, in any order, as `FP32`, `INT64` or `INT32` (taken for `INT64` inputs) in JSON
or as binary tensor data. Every output comes back unless `outputs` names some. When every input
has a fixed shape per item, concurrent requests can be batched like `/predict`'s (with
batching turned on, [above](#dynamic-batching)); a model
with an input whose shape each request fixes (a sequence length, say) runs one request per
forward pass. Such models are not served on
`/predict`, which answers `400 NAMED_TENSORS_REQUIRED` pointing here. The CPU backend computes
integers in 32 bits, so an integer input outside the `i32` range gets `400`.

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:3000")
ids = np.array([[101, 7592, 2088, 102] + [0] * 124], dtype=np.int64)
inputs = []
for name, value in [("input_ids", ids), ("attention_mask", (ids != 0).astype(np.int64))]:
    tensor = httpclient.InferInput(name, list(value.shape), "INT64")
    tensor.set_data_from_numpy(value)
    inputs.append(tensor)
logits = client.infer("classifier", inputs).as_numpy("logits")
```

### Over gRPC

Built with `--features grpc`, `--grpc-port 8001` also serves the protocol's gRPC transport
(`inference.GRPCInferenceService`, from the
[KServe proto](https://github.com/kserve/open-inference-protocol) in `proto/`), so
`tritonclient.grpc` and other KServe gRPC clients work unchanged. Inference takes the input as
`raw_input_contents` (what `tritonclient.grpc` sends) or `contents.fp32_contents` and answers
with `raw_output_contents`; it shares the HTTP server's batching, limits, timeout and metrics.
Errors map to gRPC status codes: `NOT_FOUND` for an unknown model, `INVALID_ARGUMENT`,
`RESOURCE_EXHAUSTED` when all slots are busy, `DEADLINE_EXCEEDED` on the timeout. Messages
may be as large as `--max-body-mb`.

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")
inp = grpcclient.InferInput("input", list(x.shape), "FP32")
inp.set_data_from_numpy(x)
logits = client.infer("resnet18", [inp]).as_numpy("output")
```

## Several models

`--model-name resnet18,mnist` (or the option repeated, or `name = ["resnet18", "mnist"]` in the
config file) serves both from one process, sharing the threads and the
`--max-concurrent-requests` limit; each model has its own batching queue. `--model-path` can be
repeated the same way. The first model is the default: `/predict` and `/model/info` use it.

```bash
curl localhost:3000/models
# {"default_model":"resnet18","models":[{"name":"resnet18","ready":true,"input_shape":[3,224,224],...},
#                                       {"name":"mnist","ready":true,"input_shape":[1,28,28],...}]}
curl -X POST localhost:3000/models/mnist/predict -H 'Content-Type: application/json' -d @mnist.json
curl localhost:3000/models/mnist              # as /model/info
```

`/models/{model}/predict` takes everything `/predict` does. The v2 protocol (HTTP and gRPC)
addresses the models by name. `/metrics` reports totals and a `models` list; the Prometheus
format labels the per-model series with `model="..."`.

## Reloading a model

With `--admin-token <TOKEN>` (or `FURNACE_ADMIN_TOKEN`, which keeps it out of the process list;
at least 16 characters), the server accepts:

```bash
curl -X POST localhost:3000/admin/models/resnet18/reload -H "Authorization: Bearer $TOKEN"
# {"model":"resnet18","weights":null,"reload_ms":15.0,"warm_up_ms":14.3,"timestamp":"..."}

# New weights, e.g. after fine-tuning
curl -X POST localhost:3000/admin/models/resnet18/reload -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"weights": "/srv/models/resnet18-v2.bpk"}'
```

The new model is loaded and warmed up next to the serving one, then swapped in; requests already
running finish on the previous model, so none fail or wait. The model must keep its name and
input and output shapes, and its statistics carry over. If loading fails, the previous model
keeps serving and the response says why. `weights` is a `.bpk` file for a built-in model, or
another model path for a `--model-path` model; the path is read on the server.

| Status | `code` | Cause |
|---|---|---|
| 404 | `ADMIN_API_DISABLED` | Started without `--admin-token` |
| 401 | `UNAUTHORIZED` | Missing or wrong `Authorization: Bearer` token |
| 404 | `MODEL_NOT_FOUND` | No such model |
| 400 | `INVALID_REQUEST` | A body that is neither empty nor `{"weights": "<path>"}` |
| 409 | `NOT_RELOADABLE` | The model was served without a way to reload it |
| 409 | `RELOAD_IN_PROGRESS` | The model is being reloaded already |
| 422 | `RELOAD_FAILED` | Loading or warming up failed; the previous model is still serving |
| 500 | `RELOAD_FAILED` | The reload task itself failed; the previous model is still serving |

## LLMs (OpenAI API)

With the `llm` feature, `POST /v1/chat/completions`, `POST /v1/completions` and
`GET /v1/models` serve LLMs in the OpenAI format, streaming included, next to the routes above.
Their requests, streaming format, errors and metrics are in [LLM.md](LLM.md).
`--request-timeout-ms` covers them too: a whole answer that takes longer is a `504`, and a
stream whose model sends nothing for that long ends with an error payload and `[DONE]`.

## TLS

Built with `--features tls` (the release binaries and the Docker image are), the server speaks
HTTPS, and gRPC over TLS on `--grpc-port`, with a PEM certificate and key:

```bash
furnace --tls-cert /etc/furnace/tls.crt --tls-key /etc/furnace/tls.key
curl --cacert ca.crt https://localhost:3000/healthz
```

At most 1024 handshakes run at once per listener; past that the server stops accepting until
one finishes (or times out after 10 seconds), so connections that never send a ClientHello
cannot pile up. `--max-concurrent-requests` counts requests, after the handshake.

Add `--tls-client-ca ca.crt` for mutual TLS: then only clients presenting a certificate signed by
that CA can connect at all, over HTTP and gRPC alike.

```bash
furnace --tls-cert tls.crt --tls-key tls.key --tls-client-ca ca.crt
curl --cacert ca.crt --cert client.crt --key client.key https://localhost:3000/healthz
```

The files are checked every 10 seconds, and a changed certificate is used for new connections
without a restart, so renewals by cert-manager or certbot just work. If the new files do not
load (e.g. caught half-written), the error is logged and the previous certificate stays.

The TLS stack is pure Rust like the rest of Furnace: rustls for the protocol, and a small crypto
provider of our own over [RustCrypto](https://github.com/RustCrypto) primitives,
[rustls-pure](../rustls-pure/README.md), a crate of its own. It is deliberately narrow:

| | Supported |
|---|---|
| Protocol | TLS 1.3 |
| Ciphers | AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305 |
| Key exchange | X25519, P-256, P-384 |
| Keys and client certificates | ECDSA P-256, ECDSA P-384, Ed25519 (PKCS#8 or SEC1 PEM) |

RSA keys are refused with a clear error: the pure-Rust RSA implementation has an unfixed timing
side channel (RUSTSEC-2023-0071). Issue an ECDSA P-256 certificate instead; every CA,
cert-manager (`privateKey.algorithm: ECDSA`) and Let's Encrypt (`--key-type ecdsa`) can. Prefer
P-256 over Ed25519 for the server certificate: Windows' TLS stack, and so clients built on it,
do not accept Ed25519 certificates.

The provider has not had an outside security audit, and neither have the RustCrypto crates
under it. It is checked mechanically instead: its tests run Google's
[Wycheproof](https://github.com/C2SP/wycheproof) test vectors for every algorithm it offers
(AES-GCM, ChaCha20-Poly1305, X25519, ECDH and ECDSA on P-256 and P-384, Ed25519, HMAC and HKDF),
which must accept the valid inputs and refuse the crafted bad ones: forged signatures, points off
the curve, altered tags, low-order keys. Further tests pin what the vectors cannot see: each
record's nonce is unique, a record that is altered, replayed or reordered does not decrypt, the
randomness comes from the operating system, and keys are wiped when dropped. The tag comparisons
happen in the AEAD crates and rustls, in constant time. See
[rustls-pure/src/tests.rs](../rustls-pure/src/tests.rs).

With `tritonclient.grpc`, connect with `ssl=True` and the CA (and, for mutual TLS, the client
key and certificate): `InferenceServerClient(url="host:8001", ssl=True,
root_certificates="ca.crt", private_key="client.key", certificate_chain="client.crt")`.

## API keys and rate limiting

With `--api-key <KEY>` (repeat it for several keys, or set `FURNACE_API_KEYS=key1,key2`, which
keeps them out of the process list; at least 16 characters each), every route except the health
probes (`/healthz`, `/health`, `/readyz`, `/v2/health/live`, `/v2/health/ready`) needs one:

```bash
curl -H "Authorization: Bearer $KEY" localhost:3000/models
curl -H "X-API-Key: $KEY" -X POST localhost:3000/predict -H 'Content-Type: application/json' -d @input.json
```

Over gRPC, send the same as `authorization` or `x-api-key` metadata (with `tritonclient.grpc`,
`client.infer(..., headers={"authorization": f"Bearer {key}"})`). A missing or unknown key gets
`401 UNAUTHORIZED` (gRPC `UNAUTHENTICATED`). Keys are compared in constant time. The admin API
takes `--admin-token` instead.

`--rate-limit <PER_SECOND>` caps each client at that many requests a second on average, with
bursts of up to `--rate-limit-burst` (default: the rate). A client is its API key, or, without
keys, its IP address; behind a proxy every request comes from the proxy's address, so give
clients keys there. Over the limit, requests get `429 RATE_LIMITED` with a `Retry-After`
header (gRPC `RESOURCE_EXHAUSTED`). Health probes are never limited. `/metrics` counts both
rejections (`rejected_unauthorized`, `rejected_rate_limited`).

## Request IDs and logs

Every response, errors included, has an `X-Request-Id` header. Send your own (1-128 visible
ASCII characters, e.g. a trace ID from your gateway) and it comes back unchanged; otherwise the
server generates a UUID. The server's log lines for a request run inside a `request` span with
that ID, its method and its path:

```text
INFO request{id=trace-42 method=POST path=/predict}: Prediction completed in 8.10ms, batch size: 1
```

`--log-level` sets the overall level. `RUST_LOG` takes precedence and can set levels per module,
e.g. `RUST_LOG=info,furnace::batching=debug`. `FURNACE_ENV=production` writes JSON lines, with
the span's fields in each line.

## Errors

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
| 400 | `NAMED_TENSORS_REQUIRED` | `/predict` for a model that takes named tensors; use `/v2/models/{model}/infer` |
| 404 | `MODEL_NOT_FOUND` | `/models/{model}/...` or `/v2/models/{model}/...` names a model the server does not serve, or `/predict` on a server without a tensor model |
| 413 | `INVALID_REQUEST` | Body larger than `--max-body-mb` |
| 401 | `UNAUTHORIZED` | `--api-key` is set and the request has no or an unknown key |
| 429 | `RATE_LIMITED` | Over `--rate-limit`; `Retry-After` says when to retry |
| 503 | `SERVER_BUSY` | All `--max-concurrent-requests` slots in use; retry later |
| 504 | `INFERENCE_TIMEOUT` | Exceeded `--request-timeout-ms` |
| 500 | `INFERENCE_FAILED` | Model error, or a model that panicked |
| 500 | `INTERNAL_ERROR` | Any other server-side failure |

The admin API's codes (`ADMIN_API_DISABLED`, `NOT_RELOADABLE`, `RELOAD_IN_PROGRESS`,
`RELOAD_FAILED`) are in [Reloading a model](#reloading-a-model). Over gRPC the same errors map to
`INVALID_ARGUMENT` (400), `NOT_FOUND` (404), `RESOURCE_EXHAUSTED` (503), `DEADLINE_EXCEEDED`
(504) and `INTERNAL` (500).

## Troubleshooting

**`unknown built-in model` / model missing**: `furnace-burn/models/<name>.onnx` (or
`$FURNACE_MODELS_DIR/<name>.onnx`) wasn't present at build time, or it was skipped: look for
`skipping model '<name>'` warnings in the build output. Download it (see the README's quick
start), then `cargo build --release`. `touch furnace-burn/build.rs` forces regeneration.

**`no model selected`**: the build has several built-in models; pick one with `--model-name`.

**`length limit exceeded` / 413**: raise `--max-body-mb`. A ResNet-18 image is 1.5–3 MB of JSON.

**Lots of `503 SERVER_BUSY`**: increase `--max-concurrent-requests`. Requests waiting to be
batched hold a slot too.

**`Server startup failed: Failed to bind to 127.0.0.1:3000: ...`**: another process has the
port (`Address already in use`; on Windows, `os error 10048`). Stop it, or pick another port
with `--port` (in Rust, `Server::port`). With `--grpc-port`, the same holds for
`Failed to bind gRPC to ...`.

**A feature missing from the build.** Some options need a Cargo feature, and a build without
it says so or refuses the request:

| What you see | Rebuild with |
|---|---|
| `backend not available in this build. Available backends: cpu` | `--features wgpu` (or `metal`, `cuda`) |
| ``Server startup failed: --grpc-port needs a build with the `grpc` feature`` | `--features grpc` |
| ``Server startup failed: --tls-cert needs a build with the `tls` feature`` | `--features tls` |
| `415` for `Content-Type: image/jpeg` or `image/png` | `--features image` |
| Your code does not compile: `furnace::llm`, `nerve::Llm` or `Server::llm` not found | `features = ["llm"]` on `furnace-server` |
| `unknown built-in model` for a model whose `.onnx` is in `furnace-burn/models/` | furnace-cli's default `onnx` feature, without `--no-default-features` |

The release binaries and the Docker image are built with `grpc` and `tls`, not `image`. LLMs
are served from the library only ([LLM.md](LLM.md)).

**The server prints nothing** (from your own program): Furnace logs through `tracing`, which
writes nowhere until a subscriber is installed. Add `tracing-subscriber = "0.3"` to
`Cargo.toml` and call `tracing_subscriber::fmt::init()` first thing in `main`.

**Windows: `link.exe` not found**: install Visual Studio Build Tools with the
"Desktop development with C++" workload.

