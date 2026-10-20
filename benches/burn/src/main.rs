//! Serving Burn models: what furnace adds over calling `forward` yourself, and how it holds up
//! under concurrent load against the server a Burn user would otherwise write by hand.
//!
//! ```bash
//! cargo build --release
//! target/release/furnace-burn-bench [--models digits,resnet18,encoder] [--clients 1,4,16]
//!     [--backend cpu|wgpu] [--runs N] [--label TEXT] [--json PATH]
//! ```
//!
//! `--runs N` starts every server N times, the servers taking turns, and reports the median of
//! each figure.
//!
//! For each model it measures, in this order:
//!
//! 1. *Direct*: in this process, host data -> tensor -> `forward` -> host data, batch 1 and 4.
//!    That is the work any server has to do, with nothing around it.
//! 2. *furnace*: the model served with furnace's library API (`Model::from_fn`, or a
//!    `BurnModel` for the two-input text model), default settings, in its own process; requests
//!    use the KServe / Triton v2 protocol with binary tensors over HTTP.
//! 3. *hand-written*: the same model behind a minimal axum server in its own process: one route,
//!    raw little-endian tensors in and out, `forward` on `spawn_blocking`. About as little as a
//!    server can do, and what you would write without furnace.
//!
//! Each server is loaded by N clients, each on its own keep-alive connection, sending batch-1
//! requests back to back. `serve KIND MODEL PORT [--backend B]` runs one server on its own.
//!
//! `cold-start binary|docker PATH-OR-IMAGE [ARGS]` times a server from launch to its first
//! answers (see `cold_start.rs`).
//!
//! `overhead MODEL [N]` calls each server's router in this process, without a network, one
//! request at a time, and prints the time per request: the server's own CPU cost around
//! `forward`.

mod cold_start;
mod models;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::routing::{get, post};
use burn::tensor::{Device, Int, Tensor, TensorData};
use furnace::{BackendKind, BurnModel, DataType, HostTensor, IoSpec, Model, Server};
use models::{DigitNet, ResNet18, TextEncoder};
use serde::Serialize;

const DIRECT_RUNS: usize = 50;
/// Longest any one request may take before the run fails
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const WARMUP: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Which {
    Digits,
    Resnet18,
    Encoder,
    /// Returns its input: what is left is the server's own cost
    Echo,
}

impl Which {
    fn parse(name: &str) -> Self {
        match name {
            "digits" => Self::Digits,
            "resnet18" => Self::Resnet18,
            "encoder" => Self::Encoder,
            "echo" => Self::Echo,
            _ => fail(&format!(
                "unknown model {name}: digits, resnet18, encoder or echo"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Digits => "digits",
            Self::Resnet18 => "resnet18",
            Self::Encoder => "encoder",
            Self::Echo => "echo",
        }
    }

    /// Per-item image shape, for the two image models
    fn image_shape(self) -> Option<[usize; 3]> {
        match self {
            Self::Digits => Some([1, 28, 28]),
            Self::Resnet18 => Some([3, 224, 224]),
            Self::Encoder => None,
            Self::Echo => Some([1, 1, 10]),
        }
    }

    fn outputs(self) -> usize {
        match self {
            Self::Digits => 10,
            Self::Resnet18 => 1000,
            Self::Encoder => 2,
            Self::Echo => 10,
        }
    }

    /// Timed requests per client: enough for a few seconds of load at 16 clients
    fn requests_per_client(self) -> usize {
        match self {
            Self::Digits | Self::Echo => 400,
            Self::Resnet18 => 30,
            Self::Encoder => 40,
        }
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2)
}

// ---------------------------------------------------------------------------------------------
// Inputs

/// Deterministic values (SplitMix64), so every run sends the same requests
fn values(count: usize, seed: u64) -> impl Iterator<Item = u64> {
    let mut state = seed;
    (0..count).map(move |_| {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

enum Input {
    Images(Vec<f32>),
    /// Token ids and attention mask, `batch * SEQ` each
    Text(Vec<i64>, Vec<i64>),
}

fn make_input(which: Which, batch: usize) -> Input {
    match which.image_shape() {
        Some(shape) => {
            let n = batch * shape.iter().product::<usize>();
            Input::Images(
                values(n, 1)
                    .map(|v| (v >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                    .collect(),
            )
        }
        None => {
            let n = batch * TextEncoder::SEQ;
            let ids = values(n, 2)
                .map(|v| (v % TextEncoder::VOCAB as u64) as i64)
                .collect();
            // Each sequence is real tokens followed by padding, 96 of 128 used
            let mask = (0..n)
                .map(|i| i64::from(i % TextEncoder::SEQ < 96))
                .collect();
            Input::Text(ids, mask)
        }
    }
}

impl Input {
    /// Little-endian bytes: the images, or the ids followed by the mask
    fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Images(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Text(ids, mask) => ids
                .iter()
                .chain(mask)
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The models

enum Net {
    Digits(DigitNet),
    // Boxed: twice the size of the next largest variant
    Resnet18(Box<ResNet18>),
    Encoder(TextEncoder),
    Echo,
}

impl Net {
    fn new(which: Which, device: &Device) -> Self {
        match which {
            Which::Digits => Self::Digits(DigitNet::new(device)),
            Which::Resnet18 => Self::Resnet18(Box::new(ResNet18::new(device))),
            Which::Encoder => Self::Encoder(TextEncoder::new(device)),
            Which::Echo => Self::Echo,
        }
    }

    /// Host data in, host data out: everything a server has to do around `forward`
    fn run(&self, which: Which, batch: usize, input: Input, device: &Device) -> Vec<f32> {
        let output = match (self, input) {
            (Self::Digits(net), Input::Images(v)) => net.forward(images(which, batch, v, device)),
            (Self::Resnet18(net), Input::Images(v)) => net.forward(images(which, batch, v, device)),
            (Self::Echo, Input::Images(v)) => echo(images(which, batch, v, device)),
            (Self::Encoder(net), Input::Text(ids, mask)) => {
                net.forward(tokens(batch, ids, device), tokens(batch, mask, device))
            }
            _ => unreachable!("input built for this model"),
        };
        output
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("f32 output")
    }
}

fn echo(x: Tensor<4>) -> Tensor<2> {
    x.flatten::<2>(1, 3)
}

fn images(which: Which, batch: usize, values: Vec<f32>, device: &Device) -> Tensor<4> {
    let [c, h, w] = which.image_shape().expect("image model");
    Tensor::from_data(TensorData::new(values, [batch, c, h, w]), device)
}

fn tokens(batch: usize, values: Vec<i64>, device: &Device) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(values, [batch, TextEncoder::SEQ]), device)
}

/// The text model for furnace: two named INT64 inputs, so a `BurnModel` rather than `from_fn`
#[derive(Debug)]
struct TextModel {
    net: TextEncoder,
    backend: BackendKind,
    device: Device,
}

impl BurnModel for TextModel {
    fn name(&self) -> &str {
        "encoder"
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }

    fn inputs(&self) -> Vec<IoSpec> {
        let seq = [Some(TextEncoder::SEQ)];
        vec![
            IoSpec::new("input_ids", DataType::Int64, &seq),
            IoSpec::new("attention_mask", DataType::Int64, &seq),
        ]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("logits", &[2])]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
        let mut inputs = inputs.into_iter();
        let (Some(ids), Some(mask)) = (inputs.next(), inputs.next()) else {
            unreachable!("checked against inputs()");
        };
        let logits = self.net.forward(
            ids.into_int::<2>(&self.device)?,
            mask.into_int::<2>(&self.device)?,
        );
        Ok(vec![HostTensor::from_float(logits)?])
    }
}

fn furnace_model(which: Which, backend: BackendKind) -> Model {
    let device = backend.device();
    let model = match Net::new(which, &device) {
        Net::Digits(net) => Model::from_fn(
            "digits",
            backend,
            &[1, 28, 28],
            &[10],
            move |x: Tensor<4>| net.forward(x),
        ),
        Net::Resnet18(net) => Model::from_fn(
            "resnet18",
            backend,
            &[3, 224, 224],
            &[1000],
            move |x: Tensor<4>| net.forward(x),
        ),
        Net::Echo => Model::from_fn("echo", backend, &[1, 1, 10], &[10], echo),
        Net::Encoder(net) => Ok(Model::custom(TextModel {
            net,
            backend,
            device,
        })),
    };
    model.unwrap_or_else(|e| fail(&format!("model: {e}")))
}

// ---------------------------------------------------------------------------------------------
// Servers

/// The hand-written server's state
struct HandWritten {
    which: Which,
    net: Net,
    device: Device,
}

impl HandWritten {
    fn infer(&self, body: &[u8]) -> Vec<u8> {
        let (input, batch) = match self.which.image_shape() {
            Some(shape) => {
                let values: Vec<f32> = body
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                let batch = values.len() / shape.iter().product::<usize>();
                (Input::Images(values), batch)
            }
            None => {
                let mut ids: Vec<i64> = body
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect();
                let mask = ids.split_off(ids.len() / 2);
                let batch = ids.len() / TextEncoder::SEQ;
                (Input::Text(ids, mask), batch)
            }
        };
        self.net
            .run(self.which, batch, input, &self.device)
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect()
    }
}

async fn hand_written_infer(State(server): State<Arc<HandWritten>>, body: Bytes) -> Vec<u8> {
    tokio::task::spawn_blocking(move || server.infer(&body))
        .await
        .expect("inference panicked")
}

async fn serve(kind: &str, which: Which, port: u16, backend: BackendKind, batching: &[String]) {
    match kind {
        "furnace" => {
            let result = Server::new()
                .host("127.0.0.1")
                .port(port)
                .model(furnace_model(which, backend))
                .configure(|config| {
                    for pair in batching.chunks(2) {
                        let value = pair[1].parse().unwrap_or_else(|_| fail("bad number"));
                        match pair[0].as_str() {
                            "--workers" => config.batching.workers = value,
                            "--max-batch" => config.batching.max_batch_size = value,
                            _ => {}
                        }
                    }
                })
                .serve()
                .await;
            if let Err(e) = result {
                fail(&format!("furnace: {e}"));
            }
        }
        "hand-written" => {
            let device = backend.device();
            let state = Arc::new(HandWritten {
                which,
                net: Net::new(which, &device),
                device,
            });
            let app = Router::new()
                .route("/infer", post(hand_written_infer))
                .route("/healthz", get(|| async { "ok" }))
                .layer(axum::extract::DefaultBodyLimit::max(64 << 20))
                .with_state(state);
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
                .await
                .unwrap_or_else(|e| fail(&format!("bind {port}: {e}")));
            axum::serve(listener, app).await.expect("serve");
        }
        _ => fail(&format!("unknown server {kind}: furnace or hand-written")),
    }
}

// ---------------------------------------------------------------------------------------------
// Clients

/// One request as the client sends it
struct Request {
    url: String,
    headers: Vec<(&'static str, String)>,
    body: Bytes,
    /// Expected response size in bytes, of the output tensor
    output_bytes: usize,
}

fn request(kind: &str, which: Which, port: u16) -> Request {
    let raw = make_input(which, 1).to_bytes();
    let output_bytes = 4 * which.outputs();
    if kind == "furnace /predict" {
        return Request {
            url: format!("http://127.0.0.1:{port}/predict"),
            headers: vec![
                ("content-type", "application/octet-stream".into()),
                ("accept", "application/octet-stream".into()),
            ],
            body: raw.into(),
            output_bytes,
        };
    }
    if kind == "hand-written" || kind.starts_with("fastapi") {
        return Request {
            url: format!("http://127.0.0.1:{port}/infer"),
            headers: vec![("content-type", "application/octet-stream".into())],
            body: raw.into(),
            output_bytes,
        };
    }
    let inputs = match which.image_shape() {
        Some([c, h, w]) => serde_json::json!([{
            "name": "input", "datatype": "FP32", "shape": [1, c, h, w],
            "parameters": {"binary_data_size": raw.len()}
        }]),
        None => serde_json::json!([
            {"name": "input_ids", "datatype": "INT64", "shape": [1, TextEncoder::SEQ],
             "parameters": {"binary_data_size": raw.len() / 2}},
            {"name": "attention_mask", "datatype": "INT64", "shape": [1, TextEncoder::SEQ],
             "parameters": {"binary_data_size": raw.len() / 2}},
        ]),
    };
    let header = serde_json::json!({
        "inputs": inputs,
        "parameters": {"binary_data_output": true}
    })
    .to_string();
    let mut body = header.clone().into_bytes();
    body.extend(raw);
    Request {
        url: format!("http://127.0.0.1:{port}/v2/models/{}/infer", which.name()),
        headers: vec![
            ("content-type", "application/octet-stream".into()),
            ("inference-header-content-length", header.len().to_string()),
        ],
        body: body.into(),
        output_bytes,
    }
}

async fn send(client: &reqwest::Client, req: &Request) -> Duration {
    let mut builder = client.post(&req.url).body(req.body.clone());
    for (name, value) in &req.headers {
        builder = builder.header(*name, value);
    }
    let start = Instant::now();
    let response = builder.send().await.expect("request failed");
    let status = response.status();
    let json_len = response
        .headers()
        .get("inference-header-content-length")
        .and_then(|v| v.to_str().ok()?.parse::<usize>().ok())
        .unwrap_or(0);
    let body = response.bytes().await.expect("response body");
    let elapsed = start.elapsed();
    if !status.is_success() || body.len() - json_len != req.output_bytes {
        fail(&format!(
            "{}: status {status}, {} bytes: {}",
            req.url,
            body.len(),
            String::from_utf8_lossy(&body[..body.len().min(300)])
        ));
    }
    elapsed
}

#[derive(Debug, Clone, Serialize)]
struct Load {
    clients: usize,
    requests_per_second: f64,
    p50_ms: f64,
    p95_ms: f64,
}

async fn load(req: Arc<Request>, clients: usize, per_client: usize) -> Load {
    let barrier = Arc::new(tokio::sync::Barrier::new(clients + 1));
    let tasks: Vec<_> = (0..clients)
        .map(|_| {
            let (req, barrier) = (req.clone(), barrier.clone());
            tokio::spawn(async move {
                let client = reqwest::Client::builder()
                    .pool_max_idle_per_host(1)
                    .timeout(REQUEST_TIMEOUT)
                    .build()
                    .expect("client");
                for _ in 0..WARMUP {
                    send(&client, &req).await;
                }
                barrier.wait().await;
                let mut times = Vec::with_capacity(per_client);
                for _ in 0..per_client {
                    times.push(send(&client, &req).await);
                }
                times
            })
        })
        .collect();
    barrier.wait().await;
    let start = Instant::now();
    let mut times = Vec::new();
    for task in tasks {
        times.extend(task.await.expect("client task"));
    }
    let wall = start.elapsed();
    times.sort();
    let pct = |p: f64| ms(times[((times.len() - 1) as f64 * p).round() as usize]);
    Load {
        clients,
        requests_per_second: times.len() as f64 / wall.as_secs_f64(),
        p50_ms: pct(0.5),
        p95_ms: pct(0.95),
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

// ---------------------------------------------------------------------------------------------
// The benchmark

#[derive(Debug, Serialize)]
struct Direct {
    batch1_ms: f64,
    batch4_ms: f64,
}

#[derive(Debug, Serialize)]
struct ServerResult {
    server: &'static str,
    ready_ms: f64,
    peak_private_mb: f64,
    load: Vec<Load>,
}

#[derive(Debug, Serialize)]
struct ModelResult {
    model: Which,
    direct: Direct,
    servers: Vec<ServerResult>,
}

#[derive(Debug, Serialize)]
struct Report {
    label: String,
    backend: String,
    /// Each server's figures are the medians of this many runs
    runs: usize,
    furnace_version: &'static str,
    models: Vec<ModelResult>,
}

fn direct(which: Which, backend: BackendKind) -> Direct {
    let device = backend.device();
    let net = Net::new(which, &device);
    let runs = match which {
        Which::Digits => 10 * DIRECT_RUNS,
        _ => DIRECT_RUNS,
    };
    let time = |batch: usize| {
        let mut times: Vec<_> = (0..WARMUP + runs)
            .map(|_| {
                let input = make_input(which, batch);
                let start = Instant::now();
                std::hint::black_box(net.run(which, batch, input, &device));
                start.elapsed()
            })
            .skip(WARMUP)
            .collect();
        times.sort();
        ms(times[times.len() / 2])
    };
    Direct {
        batch1_ms: time(1),
        batch4_ms: time(4),
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or_else(|e| fail(&format!("no free port: {e}")))
}

async fn bench_server(
    kind: &'static str,
    which: Which,
    backend: &str,
    clients: &[usize],
    batching: &[String],
) -> ServerResult {
    let port = free_port();
    let exe = std::env::current_exe().expect("own path");
    let start = Instant::now();
    let mut command = match kind {
        "fastapi" | "fastapi x4" => {
            let mut command = std::process::Command::new(python());
            command
                .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/fastapi/echo.py"))
                .args([
                    "--port",
                    &port.to_string(),
                    "--workers",
                    if kind == "fastapi" { "1" } else { "4" },
                ]);
            command
        }
        _ => {
            let mut command = std::process::Command::new(exe);
            command
                .args([
                    "serve",
                    kind,
                    which.name(),
                    &port.to_string(),
                    "--backend",
                    backend,
                ])
                .args(batching);
            command
        }
    };
    let mut child = command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("start server");
    // A server that accepts connections but never answers must fail the run, not hang it
    let health = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("client");
    let url = format!("http://127.0.0.1:{port}/healthz");
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    // The process to measure: the Python interpreter reports its own pid, since on Windows a
    // virtual environment's python.exe only launches it
    let pid = loop {
        // A short TCP probe first: on Windows a refused connection takes about 0.5 s to
        // report, which would swamp the start-up time
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(5)).is_ok()
            && let Ok(r) = health.get(&url).send().await
            && r.status().is_success()
        {
            let body = r.text().await.unwrap_or_default();
            break serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["pid"].as_u64())
                .map_or(child.id(), |pid| pid as u32);
        }
        if start.elapsed() > Duration::from_secs(120) {
            stop(&mut child);
            fail(&format!("{kind} server did not start"));
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    let ready_ms = ms(start.elapsed());
    let req = Arc::new(request(kind, which, port));
    let mut results = Vec::new();
    for &n in clients {
        results.push(load(req.clone(), n, which.requests_per_client()).await);
    }
    // Several uvicorn workers are several processes: one's memory would understate it
    let peak_private_mb = if kind == "fastapi x4" {
        0.0
    } else {
        memory::peak_private_mb(pid)
    };
    stop(&mut child);
    ServerResult {
        server: kind,
        ready_ms,
        peak_private_mb,
        load: results,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    (values[(n - 1) / 2] + values[n / 2]) / 2.0
}

/// One server's runs as one result: the median of each figure on its own
fn median_server(runs: Vec<ServerResult>) -> ServerResult {
    let of = |f: &dyn Fn(&ServerResult) -> f64| median(runs.iter().map(f).collect());
    ServerResult {
        server: runs[0].server,
        ready_ms: of(&|s| s.ready_ms),
        peak_private_mb: of(&|s| s.peak_private_mb),
        load: (0..runs[0].load.len())
            .map(|i| Load {
                clients: runs[0].load[i].clients,
                requests_per_second: of(&|s| s.load[i].requests_per_second),
                p50_ms: of(&|s| s.load[i].p50_ms),
                p95_ms: of(&|s| s.load[i].p95_ms),
            })
            .collect(),
    }
}

/// The Python with FastAPI, uvicorn and numpy: `$FURNACE_BENCH_PYTHON`, else the virtual
/// environment of `benches/compare/fastapi-ort`
fn python() -> std::path::PathBuf {
    std::env::var_os("FURNACE_BENCH_PYTHON").map_or_else(
        || {
            let venv = concat!(env!("CARGO_MANIFEST_DIR"), "/../compare/fastapi-ort/.venv");
            if cfg!(windows) {
                std::path::Path::new(venv).join("Scripts/python.exe")
            } else {
                std::path::Path::new(venv).join("bin/python")
            }
        },
        std::path::PathBuf::from,
    )
}

/// Stop a server with every process it started (uvicorn's workers, Python's launcher)
fn stop(child: &mut std::process::Child) {
    if cfg!(windows) {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    } else {
        // uvicorn stops its workers on SIGTERM
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// furnace's version, from its Cargo.toml (this crate builds furnace from the same checkout)
fn furnace_version() -> &'static str {
    include_str!("../../../Cargo.toml")
        .lines()
        .find_map(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or("?")
}

fn parse_backend(name: &str) -> BackendKind {
    match name {
        "cpu" => BackendKind::Cpu,
        #[cfg(feature = "wgpu")]
        "wgpu" => BackendKind::Wgpu,
        _ => fail(&format!(
            "unknown backend {name} (wgpu needs --features wgpu)"
        )),
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let backend_name = flag("--backend").unwrap_or_else(|| "cpu".into());
    let backend = parse_backend(&backend_name);
    // furnace's batching settings, passed on to its server
    let batching: Vec<String> = ["--workers", "--max-batch"]
        .iter()
        .filter_map(|&name| Some([name.to_string(), flag(name)?]))
        .flatten()
        .collect();

    // `--log LEVEL`: log as furnace's command line does (its default is `info`)
    if let Some(level) = flag("--log") {
        tracing_subscriber::fmt()
            .with_target(false)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_env_filter(tracing_subscriber::EnvFilter::new(level))
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
            .with_writer(std::io::stderr)
            .init();
    }
    if args.first().map(String::as_str) == Some("cold-start") {
        cold_start::main(&args[1..]);
        return;
    }
    if args.first().map(String::as_str) == Some("overhead") {
        let which = Which::parse(args.get(1).map_or("digits", String::as_str));
        let n = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(20_000);
        overhead(which, backend, n, &batching).await;
        return;
    }
    if args.first().map(String::as_str) == Some("serve") {
        let [_, kind, model, port, ..] = args.as_slice() else {
            fail("serve KIND MODEL PORT [--backend B]");
        };
        let port = port.parse().unwrap_or_else(|_| fail("bad port"));
        serve(kind, Which::parse(model), port, backend, &batching).await;
        return;
    }

    let models: Vec<Which> = flag("--models")
        .unwrap_or_else(|| "digits,resnet18,encoder".into())
        .split(',')
        .map(Which::parse)
        .collect();
    let clients: Vec<usize> = flag("--clients")
        .unwrap_or_else(|| "1,4,16".into())
        .split(',')
        .map(|n| n.parse().unwrap_or_else(|_| fail("bad --clients")))
        .collect();
    let runs: usize = flag("--runs").map_or(1, |n| match n.parse() {
        Ok(n) if n > 0 => n,
        _ => fail("bad --runs"),
    });
    let mut report = Report {
        label: flag("--label").unwrap_or_default(),
        backend: backend_name.clone(),
        runs,
        furnace_version: furnace_version(),
        models: Vec::new(),
    };

    for which in models {
        eprintln!("{}: direct", which.name());
        let direct = tokio::task::spawn_blocking(move || direct(which, backend))
            .await
            .expect("direct run");
        let mut kinds = vec!["furnace", "hand-written"];
        // Python serves only a model it can run: the echo
        if which == Which::Echo {
            kinds.extend(["fastapi", "fastapi x4"]);
        }
        // The servers take turns, so a slow stretch of the machine hits each of them alike
        let mut samples: Vec<Vec<ServerResult>> = kinds.iter().map(|_| Vec::new()).collect();
        for run in 1..=runs {
            for (kind, samples) in kinds.iter().zip(&mut samples) {
                // A laptop that has just run a heavy load is slower for a while
                tokio::time::sleep(Duration::from_secs(5)).await;
                eprintln!("{}: {kind} (run {run} of {runs})", which.name());
                samples.push(bench_server(kind, which, &backend_name, &clients, &batching).await);
            }
        }
        report.models.push(ModelResult {
            model: which,
            direct,
            servers: samples.into_iter().map(median_server).collect(),
        });
    }

    print_report(&report);
    if let Some(path) = flag("--json") {
        let json = serde_json::to_string_pretty(&report).expect("json");
        std::fs::write(&path, json).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    }
}

/// Each server's router called in process, one request at a time
async fn overhead(which: Which, backend: BackendKind, n: usize, batching: &[String]) {
    use tower::ServiceExt;

    let furnace = Server::new()
        .model(furnace_model(which, backend))
        .configure(|config| {
            for pair in batching.chunks(2) {
                let value = pair[1].parse().unwrap_or_else(|_| fail("bad number"));
                match pair[0].as_str() {
                    "--workers" => config.batching.workers = value,
                    "--max-batch" => config.batching.max_batch_size = value,
                    _ => {}
                }
            }
        })
        .router();
    let device = backend.device();
    let hand_written = Router::new()
        .route("/infer", post(hand_written_infer))
        .with_state(Arc::new(HandWritten {
            which,
            net: Net::new(which, &device),
            device,
        }));
    let device = backend.device();
    let net = Net::new(which, &device);
    let start = Instant::now();
    for _ in 0..n {
        std::hint::black_box(net.run(which, 1, make_input(which, 1), &device));
    }
    println!(
        "forward only     {:8.1} us",
        start.elapsed().as_secs_f64() * 1e6 / n as f64
    );
    if let Input::Images(values) = make_input(which, 1) {
        let model = furnace_model(which, backend);
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(model.predict_flat(values.clone(), 1).expect("predict"));
        }
        println!(
            "Model::predict   {:8.1} us",
            start.elapsed().as_secs_f64() * 1e6 / n as f64
        );
    }
    let mut routers = vec![("furnace", furnace.clone())];
    if which.image_shape().is_some() {
        routers.push(("furnace /predict", furnace));
    }
    routers.push(("hand-written", hand_written));
    for (kind, router) in routers {
        let req = request(kind, which, 0);
        let path = req.url.trim_start_matches("http://127.0.0.1:0").to_string();
        let call = || {
            let mut builder = axum::http::Request::post(&path);
            for (name, value) in &req.headers {
                builder = builder.header(*name, value);
            }
            let request = builder
                .body(axum::body::Body::from(req.body.clone()))
                .expect("request");
            let router = router.clone();
            async move {
                let response = router.oneshot(request).await.expect("infallible");
                assert!(response.status().is_success(), "{}", response.status());
                axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("body")
            }
        };
        for _ in 0..WARMUP * 20 {
            call().await;
        }
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(call().await);
        }
        println!(
            "{kind:16} {:8.1} us",
            start.elapsed().as_secs_f64() * 1e6 / n as f64
        );
    }
}

fn print_report(report: &Report) {
    println!(
        "furnace {} on {}{}{}
",
        report.furnace_version,
        report.backend,
        if report.runs > 1 {
            format!(", medians of {} runs", report.runs)
        } else {
            String::new()
        },
        if report.label.is_empty() {
            String::new()
        } else {
            format!(" ({})", report.label)
        }
    );
    println!("| Model | Direct, batch 1 / 4 | furnace, 1 client | Added by furnace |");
    println!("|---|---|---|---|");
    for m in &report.models {
        let one = m.servers[0].load.iter().find(|l| l.clients == 1);
        println!(
            "| {} | {:.3} / {:.3} ms | {} | {} |",
            m.model.name(),
            m.direct.batch1_ms,
            m.direct.batch4_ms,
            one.map_or("-".into(), |l| format!("{:.3} ms", l.p50_ms)),
            one.map_or("-".into(), |l| format!(
                "{:+.3} ms",
                l.p50_ms - m.direct.batch1_ms
            )),
        );
    }
    for m in &report.models {
        println!(
            "
{}
",
            m.model.name()
        );
        let names: Vec<_> = m.servers.iter().map(|s| s.server).collect();
        println!("| | {} |", names.join(" | "));
        println!("|---|{}", "---|".repeat(names.len()));
        let cell = |f: &dyn Fn(&ServerResult) -> String| {
            m.servers.iter().map(f).collect::<Vec<_>>().join(" | ")
        };
        println!(
            "| Ready after start | {} |",
            cell(&|s| format!("{:.0} ms", s.ready_ms))
        );
        println!(
            "| Peak private memory | {} |",
            cell(&|s| if s.peak_private_mb > 0.0 {
                format!("{:.0} MB", s.peak_private_mb)
            } else {
                "-".into()
            })
        );
        for (i, load) in m.servers[0].load.iter().enumerate() {
            println!(
                "| {} client(s): req/s, p50 / p95 | {} |",
                load.clients,
                cell(&|s| {
                    let l = &s.load[i];
                    format!(
                        "{:.0}, {:.2} / {:.2} ms",
                        l.requests_per_second, l.p50_ms, l.p95_ms
                    )
                })
            );
        }
    }
}

mod memory {
    /// Peak memory private to a process, in MB (10^6 bytes); 0 where unsupported
    #[cfg(windows)]
    pub fn peak_private_mb(pid: u32) -> f64 {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
        };

        // SAFETY: plain Win32 calls on a handle we open and close here; the counters
        // struct is fully initialized by GetProcessMemoryInfo on success.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid);
            if handle.is_null() {
                return 0.0;
            }
            let mut counters: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
            counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
            let ok = GetProcessMemoryInfo(
                handle,
                (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX)
                    .cast::<PROCESS_MEMORY_COUNTERS>(),
                counters.cb,
            );
            CloseHandle(handle);
            if ok == 0 {
                return 0.0;
            }
            counters.PeakPagefileUsage as f64 / 1e6
        }
    }

    #[cfg(target_os = "linux")]
    pub fn peak_private_mb(pid: u32) -> f64 {
        // No peak private counter on Linux: the current anonymous resident memory
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        status
            .lines()
            .find(|l| l.starts_with("RssAnon:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(0.0, |kb| kb * 1024.0 / 1e6)
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub fn peak_private_mb(_pid: u32) -> f64 {
        0.0
    }
}
