# Scale to zero

Serverless platforms stop a service when it gets no traffic and start it again on the next
request, so the first user waits for the server to start. For a Python inference stack that is
seconds. Furnace is ready about 30 ms after launch and has answered its first inference by
40 ms, which makes scaling to zero practical for models.

## Cold start, measured

ResNet-18, from launching the server to its answers, median of 10 runs on the laptop of
[PERFORMANCE.md](PERFORMANCE.md) (Ryzen 9 7940HS, Windows 11):

| | Accepting requests | Ready (`/readyz` 200, after the warm-up inference) | First inference answered |
|---|---|---|---|
| Furnace binary (`--features "grpc tls"`) | 23 ms | 29 ms | 40 ms |
| Furnace in Docker (`ghcr.io/gilfeather/furnace:0.10.0`) | 404 ms | 427 ms | 458 ms |
| FastAPI + ONNX Runtime ([PERFORMANCE.md](PERFORMANCE.md)) | | 1,761 ms | |

In Docker almost all the time is Docker's own: a container that only prints `furnace --version`
takes 722 ms to create, run and remove on the same machine. What a platform adds depends on the
platform; what Furnace adds is the ~30-40 ms of the first row. The "first inference" column
includes one ResNet-18 forward pass and sending its 0.6 MB input (about 11 ms here).

An earlier Python version of this measurement reported 42 and 67 ms for the binary: its 1 ms
polling sleep sleeps about 15 ms on Windows, and its HTTP client added time of its own.

Reproduce:

```bash
cargo build --release --manifest-path benches/burn/Cargo.toml
benches/burn/target/release/furnace-burn-bench cold-start binary ./target/release/furnace --model-name resnet18
benches/burn/target/release/furnace-burn-bench cold-start docker ghcr.io/gilfeather/furnace:0.10.0
```

The image is 139 MB with ResNet-18's 46 MB of weights; a smaller image pulls faster when a
platform starts on a node that does not have it yet.

## Running with scale to zero

The commands below use each platform's standard options and have not been measured there yet.
Reports of cold starts on real platforms are welcome in an issue.

### Knative / KServe

Knative scales to zero by default. With KServe, set the predictor's minimum replicas to 0 and
use the serving runtime in [`deploy/kserve`](../deploy/kserve):

```yaml
spec:
  predictor:
    minReplicas: 0
```

### Google Cloud Run

Cloud Run starts instances on demand and scales to zero by default. It pulls images from
Artifact Registry, so copy the image there first:

```bash
docker pull ghcr.io/gilfeather/furnace:0.16.2
docker tag ghcr.io/gilfeather/furnace:0.16.2 REGION-docker.pkg.dev/PROJECT/REPO/furnace:0.16.2
docker push REGION-docker.pkg.dev/PROJECT/REPO/furnace:0.16.2
gcloud run deploy furnace --image REGION-docker.pkg.dev/PROJECT/REPO/furnace:0.16.2 \
  --port 3000 --cpu 2 --memory 512Mi --min-instances 0
```

### Fly.io

```toml
# fly.toml
[build]
  image = "ghcr.io/gilfeather/furnace:0.16.2"

[http_service]
  internal_port = 3000
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
```

## Making the first request fast

- **Readiness**: point the platform's startup or readiness check at `/readyz`. It turns 200
  after a warm-up inference, so the first real request does not pay for lazy initialization.
- **Threads**: set `--threads` to the CPUs the instance gets; by default Furnace uses one thread
  per physical core of the machine it sees, which in a container can be more than its share.
- **Weights**: they are memory-mapped, so start-up does not read them all; pages load as the
  warm-up touches them.
