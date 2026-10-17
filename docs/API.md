# HTTP API

All endpoints are on the address given by `--host` and `--port` (default `127.0.0.1:3000`).

| Endpoint | Purpose |
|---|---|
| `GET /healthz` (alias `/health`) | Liveness and readiness |
| `GET /metrics` | Request, error and latency counters |
| `GET /model/info` | Input and output shapes, backend, per-model statistics |
| `POST /predict` | Inference: JSON, raw `float32`, or (with `--features image`) a JPEG/PNG |

## `GET /healthz` (alias `GET /health`)

```json
{ "status": "healthy", "model_loaded": true, "backend": "cpu", "uptime_seconds": 42, "timestamp": "2026-10-14T03:34:47Z" }
```

## `GET /metrics`

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
| 413 | `INVALID_REQUEST` | Body larger than `--max-body-mb` |
| 503 | `SERVER_BUSY` | All `--max-concurrent-requests` slots in use; retry later |
| 504 | `INFERENCE_TIMEOUT` | Exceeded `--request-timeout-ms` |
| 500 | `INFERENCE_FAILED` | Model error |

## Troubleshooting

**`unknown built-in model` / model missing**: `models/<name>.onnx` wasn't present at
build time, or it was skipped: look for `skipping model '<name>'` warnings in the build
output. Download it (see the README's quick start), then `cargo build --release`.
`touch build.rs` forces regeneration.

**`no model selected`**: the build has several built-in models; pick one with `--model-name`.

**`length limit exceeded` / 413**: raise `--max-body-mb`. A ResNet-18 image is 1.5–3 MB of JSON.

**Lots of `503 SERVER_BUSY`**: increase `--max-concurrent-requests`, or batch on the client.

**`backend not available in this build`**: rebuild with the matching feature,
e.g. `--features wgpu`.

**Windows: `link.exe` not found**: install Visual Studio Build Tools with the
"Desktop development with C++" workload.

