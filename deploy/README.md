# Deploying Furnace on Kubernetes

## Helm

```bash
helm install furnace ./deploy/helm/furnace
kubectl port-forward svc/furnace 3000
curl localhost:3000/v2/models/resnet18
```

The chart deploys the published image (ResNet-18 and MNIST built in) with the liveness probe on
`/healthz`, the readiness probe on `/readyz` (ready after a warm-up inference), a non-root,
read-only container, and gRPC on port 8001. [`values.yaml`](helm/furnace/values.yaml) covers:

| Value | |
|---|---|
| `models` | Built-in models to serve; the first answers `POST /predict` |
| `apiKeys.existingSecret` | Require API keys, read from a Secret |
| `rateLimit.perSecond` / `burst` | Per-client rate limit |
| `adminToken.existingSecret` | Enable reloading a model without downtime |
| `tls.existingSecret` | HTTPS and gRPC over TLS from a `kubernetes.io/tls` Secret, e.g. from cert-manager (ECDSA keys); renewals are picked up without a restart |
| `tls.clientCaSecret` | Mutual TLS |
| `serviceMonitor.enabled` | Prometheus Operator scraping of `/metrics` |
| `autoscaling.enabled` | Horizontal pod autoscaling on CPU |
| `extraArgs` | Any other `furnace` option |

To serve your own models, build an image from the [Dockerfile](../Dockerfile) with them in
`furnace-burn/models/` and set `image.repository`.

## KServe

[`kserve/servingruntime.yaml`](kserve/servingruntime.yaml) registers Furnace as a
`ClusterServingRuntime` for the Open Inference Protocol over HTTP and gRPC, and
[`kserve/inferenceservice.yaml`](kserve/inferenceservice.yaml) serves ResNet-18 with it.

The model architecture is compiled into the image, and KServe supplies its weights: the
InferenceService's name picks the built-in model, and its `storageUri` is a `<model>.bpk` file.
That is how to roll out fine-tuned weights through KServe without rebuilding the image.

Both manifests are tested in CI ([`kserve-e2e.yml`](../.github/workflows/kserve-e2e.yml)): on a
kind cluster with KServe v0.21 in standard (RawDeployment) mode, the published image serves
ResNet-18 with weights fetched by KServe's storage initializer, and answers a v2 inference
request with 1000 logits. The workflow runs on pull requests that touch `deploy/kserve`, against
the released image the manifests name; to test another build, run it by hand (Actions → KServe
end-to-end → Run workflow) with `image` set to an image the cluster can pull.
