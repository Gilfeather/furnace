//! The LLM server layer alone: what furnace adds to every streamed token, against a hand-written
//! axum server and against FastAPI, the front end of vLLM-style Python servers.
//!
//! ```bash
//! cargo build --release
//! target/release/furnace-llm-bench [--clients 1,16,64,256] [--tokens 64] [--token-ms 20]
//!     [--requests 4] [--servers furnace,scheduled,hand-written,fastapi] [--runs N] [--json PATH]
//! ```
//!
//! `--runs N` starts every server N times, the servers taking turns, and reports the median of
//! each figure.
//!
//! Every server streams the same mock answer to `POST /v1/chat/completions`: `--tokens`
//! OpenAI chat chunks of one token each (`"x"`), one every `--token-ms`, after a role chunk,
//! then the finish chunk and `[DONE]`. The engine does no work, so whatever a stream costs
//! beyond its schedule is the server's.
//!
//! - *furnace*: `Server::llm` with an `LlmEngine` that paces its tokens: token *i* goes out at
//!   *i* × `--token-ms` after the request arrived, so lateness does not add up. Furnace's
//!   HTTP layer, SSE, metrics and admission, with no scheduler (an engine that schedules itself).
//! - *scheduled*: `Server::llm` with furnace's scheduler over a `StepModel` whose decode step
//!   takes `--token-ms` for every running sequence at once (a batched forward pass), through
//!   the text pipeline: sampling over a 257-token vocabulary, detokenization, stop strings.
//!   Each step starts when the last one ends, so its lateness does add up; that is the
//!   scheduler's cost per token.
//! - *hand-written*: axum with `Sse`, paced like *furnace*: one route, the same chunks, about as
//!   little as a server can do.
//! - *fastapi*: FastAPI and uvicorn (one worker) with a `StreamingResponse`, paced the same
//!   way with `asyncio.sleep` (fastapi/stream.py).
//!
//! N clients, each on its own keep-alive connection, send streaming requests back to back.
//! For every token the client records its *lateness*: arrival minus (request sent + its slot
//! in the schedule). The first token's lateness is the time to first token. Server CPU time
//! and peak memory come from the server's process.
//!
//! `serve KIND PORT --tokens N --token-ms MS` runs one server on its own.
//!
//! `real --gguf PATH [--llama-server PATH] [--mistralrs PATH]` serves one GGUF file with
//! furnace + candle, llama.cpp and mistral.rs, and compares them on real tokens (see `real.rs`).
//!
//! `overhead [--tokens N] [--streams N]` calls each Rust server's router in this process, with
//! no interval and no network, one stream at a time, and prints the time per token: the server
//! layer's own CPU cost, which the paced runs above cannot resolve below the timers' millisecond.

mod real;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use furnace::Server;
use furnace::llm::{
    BoxError, EngineError, Event, FinishReason, GenerateRequest, LlmEngine, Logits, Scheduled,
    SchedulerConfig, StepModel, mock,
};
use serde::Serialize;
use tokio::sync::mpsc;

/// Longest a whole stream may take before the run fails
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Streams per client before the measured ones
const WARMUP: usize = 1;

/// The server being measured, stopped by `fail` too, so a failed run leaves no process behind
static SERVER: std::sync::Mutex<Option<std::process::Child>> = std::sync::Mutex::new(None);

fn fail(message: &str) -> ! {
    eprintln!("furnace-llm-bench: {message}");
    stop_server();
    std::process::exit(1)
}

/// The mock answer every server streams
#[derive(Clone, Copy, Debug, Serialize)]
struct Pace {
    tokens: usize,
    token_ms: u64,
}

impl Pace {
    fn interval(self) -> Duration {
        Duration::from_millis(self.token_ms)
    }
}

/// Wait for token `i`'s slot, not at all when it has passed (the first token's, always), as
/// the FastAPI server does
async fn wait_for_slot(start: tokio::time::Instant, pace: Pace, i: usize) {
    let slot = start + pace.interval() * i as u32;
    if slot > tokio::time::Instant::now() {
        tokio::time::sleep_until(slot).await;
    }
}

/// Timers to the millisecond. Windows wakes sleeping threads every 15.6 ms by default, which
/// would turn a 20 ms token interval into 15.6 or 31.2 ms for every server alike; each server
/// process asks for 1 ms (fastapi/stream.py does the same).
fn fine_timers() {
    #[cfg(windows)]
    // SAFETY: a plain Win32 call; the period is released when the process exits
    unsafe {
        windows_sys::Win32::Media::timeBeginPeriod(1);
    }
}

// ---------------------------------------------------------------------------------------------
// The servers

/// An engine of its own, like one that schedules itself: token i at i × interval after the
/// request arrived
#[derive(Debug)]
struct PacedEngine {
    pace: Pace,
}

impl LlmEngine for PacedEngine {
    fn name(&self) -> &str {
        "mock"
    }

    fn max_context(&self) -> usize {
        1 << 20
    }

    fn generate(&self, _request: GenerateRequest) -> Result<mpsc::Receiver<Event>, EngineError> {
        let pace = self.pace;
        let start = tokio::time::Instant::now();
        let (events, receiver) = mpsc::channel(16);
        tokio::spawn(async move {
            if events
                .send(Event::Started { prompt_tokens: 1 })
                .await
                .is_err()
            {
                return;
            }
            for i in 0..pace.tokens {
                tokio::select! {
                    () = wait_for_slot(start, pace, i) => {}
                    () = events.closed() => return,
                }
                if events.send(Event::Text("x".to_string())).await.is_err() {
                    return;
                }
            }
            let _ = events
                .send(Event::Finished {
                    reason: FinishReason::Stop,
                    completion_tokens: pace.tokens,
                })
                .await;
        });
        Ok(receiver)
    }
}

/// A step model whose decode step costs the same for any number of sequences, as a batched
/// forward pass does: `x` until the reply is long enough, then EOS
#[derive(Debug)]
struct BatchedMock {
    pace: Pace,
}

const VOCAB: usize = mock::EOS_ID as usize + 1;

impl BatchedMock {
    /// Logits for the token after `position` generated ones
    fn logits(&self, position: usize) -> Logits {
        let token = if position < self.pace.tokens {
            usize::from(b'x')
        } else {
            mock::EOS_ID as usize
        };
        let mut logits = vec![0.0; VOCAB];
        logits[token] = 10.0;
        Logits(logits)
    }
}

impl StepModel for BatchedMock {
    /// Tokens generated so far
    type Sequence = usize;

    fn name(&self) -> &str {
        "mock"
    }

    fn max_context(&self) -> usize {
        1 << 20
    }

    fn max_sequences(&self) -> usize {
        4096
    }

    fn prefill(&self, _prompt: &[u32]) -> Result<(usize, Logits), BoxError> {
        Ok((0, self.logits(0)))
    }

    fn decode(&self, position: &mut usize, _token: u32) -> Result<Logits, BoxError> {
        *position += 1;
        Ok(self.logits(*position))
    }

    fn decode_batch(
        &self,
        positions: &mut [&mut usize],
        _tokens: &[u32],
    ) -> Vec<Result<Logits, BoxError>> {
        std::thread::sleep(self.pace.interval());
        positions
            .iter_mut()
            .map(|position| {
                **position += 1;
                Ok(self.logits(**position))
            })
            .collect()
    }
}

/// The OpenAI chunk the hand-written server sends, as a careful hand-writer would
#[derive(Serialize)]
struct Chunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: [Choice<'a>; 1],
}

#[derive(Serialize)]
struct Choice<'a> {
    index: u32,
    delta: Delta<'a>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Delta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

fn chunk(
    id: &str,
    role: Option<&'static str>,
    content: Option<&str>,
    finish: Option<&'static str>,
) -> String {
    serde_json::to_string(&Chunk {
        id,
        object: "chat.completion.chunk",
        created: 0,
        model: "mock",
        choices: [Choice {
            index: 0,
            delta: Delta { role, content },
            finish_reason: finish,
        }],
    })
    .expect("chunks serialize")
}

/// The hand-written server: one route, the request body read and ignored, an engine task
/// pacing the same tokens as `PacedEngine` and handing them over a channel, as any server
/// with an engine behind it does, and the chunks written as they come
fn hand_written(pace: Pace) -> Router {
    let stream = move |_body: axum::body::Bytes| async move {
        let start = tokio::time::Instant::now();
        let (tokens, receiver) = mpsc::channel::<String>(16);
        tokio::spawn(async move {
            for i in 0..pace.tokens {
                wait_for_slot(start, pace, i).await;
                if tokens.send("x".to_string()).await.is_err() {
                    return;
                }
            }
        });
        let id = "chatcmpl-hand-written";
        // 0: the role chunk; 1: tokens until the channel ends, then the finish chunk; 2: [DONE]
        let events = futures_util::stream::unfold(
            (0u8, receiver),
            move |(phase, mut receiver)| async move {
                let (data, next) = match phase {
                    0 => (chunk(id, Some("assistant"), Some(""), None), 1),
                    1 => match receiver.recv().await {
                        Some(token) => (chunk(id, None, Some(&token), None), 1),
                        None => (chunk(id, None, None, Some("stop")), 2),
                    },
                    2 => ("[DONE]".to_string(), 3),
                    _ => return None,
                };
                Some((
                    Ok::<_, Infallible>(SseEvent::default().data(data)),
                    (next, receiver),
                ))
            },
        );
        Sse::new(events).keep_alive(KeepAlive::default())
    };
    Router::new()
        .route("/v1/chat/completions", post(stream))
        .route("/healthz", get(health))
}

async fn health() -> Response {
    Json(serde_json::json!({"status": "ok", "pid": std::process::id()})).into_response()
}

/// A server's router
fn router(kind: &str, pace: Pace) -> Router {
    match kind {
        "furnace" => Server::new()
            .configure(|c| c.max_concurrent_requests = 10_000)
            .llm(PacedEngine { pace })
            .router(),
        "scheduled" => {
            let engine = Scheduled::with_config(
                BatchedMock { pace },
                mock::byte_pipeline(),
                SchedulerConfig {
                    max_waiting: 10_000,
                    ..SchedulerConfig::default()
                },
            );
            Server::new()
                .configure(|c| c.max_concurrent_requests = 10_000)
                .llm(engine)
                .router()
        }
        "hand-written" => hand_written(pace),
        other => fail(&format!("unknown server {other}")),
    }
}

async fn serve(kind: &str, port: u16, pace: Pace) {
    fine_timers();
    let app = router(kind, pace);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap_or_else(|e| fail(&format!("cannot bind port {port}: {e}")));
    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| fail(&format!("server: {e}")));
}

/// `overhead`: each Rust server's router called in this process, with tokens as fast as the
/// engine can make them (no interval, no network), one stream at a time: the server's own CPU
/// cost per streamed token, chunk serialization and SSE framing included
async fn overhead(tokens: usize, streams: usize) {
    use tower::ServiceExt;
    let pace = Pace {
        tokens,
        token_ms: 0,
    };
    let body = serde_json::json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "Say x."}],
        "stream": true,
        "max_tokens": tokens,
        "temperature": 0,
    })
    .to_string();
    println!(
        "{tokens} tokens per stream, {streams} streams one after another, in process
"
    );
    println!(
        "| Server | Per token | Per stream |
|---|---|---|"
    );
    for kind in ["furnace", "scheduled", "hand-written"] {
        let app = router(kind, pace);
        let run = |n: usize| {
            let (app, body) = (app.clone(), body.clone());
            async move {
                let start = Instant::now();
                for _ in 0..n {
                    let request = axum::http::Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(body.clone()))
                        .expect("request");
                    let response = app.clone().oneshot(request).await.expect("response");
                    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .expect("body");
                    if bytes.len() < tokens * 20 {
                        fail(&format!("{kind}: a stream of {} bytes", bytes.len()));
                    }
                }
                start.elapsed()
            }
        };
        run(streams / 10 + 1).await;
        let elapsed = run(streams).await;
        let per_stream = elapsed.as_secs_f64() * 1e6 / streams as f64;
        println!(
            "| {kind} | {:.2} µs | {:.1} µs |",
            per_stream / tokens as f64,
            per_stream
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The client

/// One stream: when each token arrived, relative to when the request was sent
struct Stream {
    arrivals: Vec<Duration>,
}

async fn stream(client: &reqwest::Client, url: &str, body: &str) -> Stream {
    let sent = Instant::now();
    let mut response = client
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap_or_else(|e| fail(&format!("request: {e}")));
    if !response.status().is_success() {
        fail(&format!("status {}", response.status()));
    }
    let mut arrivals = Vec::new();
    let mut pending = Vec::new();
    loop {
        let piece = match response.chunk().await {
            Ok(Some(piece)) => piece,
            Ok(None) => break,
            Err(e) => fail(&format!("stream: {e}")),
        };
        let now = sent.elapsed();
        pending.extend_from_slice(&piece);
        // Whole events only: `data: ...\n\n`
        while let Some(end) = pending.windows(2).position(|w| w == b"\n\n") {
            let event: Vec<u8> = pending.drain(..end + 2).collect();
            const TOKEN: &[u8] = b"\"content\":\"x\"";
            if event.windows(TOKEN.len()).any(|w| w == TOKEN) {
                arrivals.push(now);
            }
        }
    }
    Stream { arrivals }
}

#[derive(Debug, Serialize)]
struct Load {
    clients: usize,
    streams: usize,
    tokens_per_second: f64,
    /// The first token's lateness
    ttft_p50_ms: f64,
    ttft_p99_ms: f64,
    /// Every token's lateness
    lateness_p50_ms: f64,
    lateness_p99_ms: f64,
    /// Between two tokens of one stream
    itl_p50_ms: f64,
    itl_p99_ms: f64,
    /// Server CPU time per 1000 tokens streamed
    cpu_ms_per_1k_tokens: f64,
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

async fn load(
    url: Arc<String>,
    body: Arc<String>,
    pace: Pace,
    clients: usize,
    per_client: usize,
    pid: u32,
) -> Load {
    let barrier = Arc::new(tokio::sync::Barrier::new(clients + 1));
    let tasks: Vec<_> = (0..clients)
        .map(|_| {
            let (url, body, barrier) = (url.clone(), body.clone(), barrier.clone());
            tokio::spawn(async move {
                let client = reqwest::Client::builder()
                    .pool_max_idle_per_host(1)
                    .timeout(REQUEST_TIMEOUT)
                    .build()
                    .expect("client");
                for _ in 0..WARMUP {
                    stream(&client, &url, &body).await;
                }
                barrier.wait().await;
                let mut streams = Vec::with_capacity(per_client);
                for _ in 0..per_client {
                    streams.push(stream(&client, &url, &body).await);
                }
                streams
            })
        })
        .collect();
    barrier.wait().await;
    let cpu_before = cpu::seconds(pid);
    let start = Instant::now();
    let mut streams = Vec::new();
    for task in tasks {
        streams.extend(task.await.expect("client task"));
    }
    let wall = start.elapsed();
    let cpu = cpu::seconds(pid) - cpu_before;

    let interval = pace.interval().as_secs_f64() * 1e3;
    let (mut ttft, mut lateness, mut itl) = (Vec::new(), Vec::new(), Vec::new());
    let mut tokens = 0;
    for s in &streams {
        if s.arrivals.len() != pace.tokens {
            fail(&format!(
                "a stream had {} tokens, not {}",
                s.arrivals.len(),
                pace.tokens
            ));
        }
        tokens += s.arrivals.len();
        for (i, arrival) in s.arrivals.iter().enumerate() {
            let at = arrival.as_secs_f64() * 1e3;
            lateness.push(at - interval * i as f64);
            if i == 0 {
                ttft.push(at);
            } else {
                itl.push(at - s.arrivals[i - 1].as_secs_f64() * 1e3);
            }
        }
    }
    Load {
        clients,
        streams: streams.len(),
        tokens_per_second: tokens as f64 / wall.as_secs_f64(),
        ttft_p50_ms: percentile(&mut ttft, 0.5),
        ttft_p99_ms: percentile(&mut ttft, 0.99),
        lateness_p50_ms: percentile(&mut lateness, 0.5),
        lateness_p99_ms: percentile(&mut lateness, 0.99),
        itl_p50_ms: percentile(&mut itl, 0.5),
        itl_p99_ms: percentile(&mut itl, 0.99),
        cpu_ms_per_1k_tokens: cpu * 1e3 / tokens as f64 * 1e3,
    }
}

// ---------------------------------------------------------------------------------------------
// The benchmark

#[derive(Debug, Serialize)]
struct ServerResult {
    server: String,
    peak_private_mb: f64,
    load: Vec<Load>,
}

#[derive(Debug, Serialize)]
struct Report {
    furnace: &'static str,
    pace: Pace,
    requests_per_client: usize,
    /// Each server's figures are the medians of this many runs
    runs: usize,
    servers: Vec<ServerResult>,
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
        server: runs[0].server.clone(),
        peak_private_mb: of(&|s| s.peak_private_mb),
        load: (0..runs[0].load.len())
            .map(|i| Load {
                clients: runs[0].load[i].clients,
                streams: runs[0].load[i].streams,
                tokens_per_second: of(&|s| s.load[i].tokens_per_second),
                ttft_p50_ms: of(&|s| s.load[i].ttft_p50_ms),
                ttft_p99_ms: of(&|s| s.load[i].ttft_p99_ms),
                lateness_p50_ms: of(&|s| s.load[i].lateness_p50_ms),
                lateness_p99_ms: of(&|s| s.load[i].lateness_p99_ms),
                itl_p50_ms: of(&|s| s.load[i].itl_p50_ms),
                itl_p99_ms: of(&|s| s.load[i].itl_p99_ms),
                cpu_ms_per_1k_tokens: of(&|s| s.load[i].cpu_ms_per_1k_tokens),
            })
            .collect(),
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or_else(|e| fail(&format!("no free port: {e}")))
}

/// The Python with FastAPI and uvicorn: `$FURNACE_BENCH_PYTHON`, else the virtual environment
/// of `benches/compare/fastapi-ort`
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

/// Stop the server with every process it started (Python's launcher on Windows)
fn stop_server() {
    let Some(mut child) = SERVER.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    if cfg!(windows) {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

async fn bench_server(
    kind: &str,
    pace: Pace,
    clients: &[usize],
    per_client: usize,
) -> ServerResult {
    let port = free_port();
    let pace_args = [
        "--tokens".to_string(),
        pace.tokens.to_string(),
        "--token-ms".to_string(),
        pace.token_ms.to_string(),
    ];
    let mut command = if kind == "fastapi" {
        let mut command = std::process::Command::new(python());
        command.arg(concat!(env!("CARGO_MANIFEST_DIR"), "/fastapi/stream.py"));
        command.args(["--port", &port.to_string()]);
        command
    } else {
        let mut command = std::process::Command::new(std::env::current_exe().expect("own path"));
        command.args(["serve", kind, &port.to_string()]);
        command
    };
    let child = command
        .args(pace_args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap_or_else(|e| fail(&format!("cannot start {kind}: {e}")));
    let launcher = child.id();
    *SERVER.lock().unwrap_or_else(|e| e.into_inner()) = Some(child);
    let probe = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("client");
    let started = Instant::now();
    // The process to measure: the Python interpreter reports its own pid, since on Windows a
    // virtual environment's python.exe only launches it
    let pid = loop {
        if let Ok(r) = probe
            .get(format!("http://127.0.0.1:{port}/healthz"))
            .send()
            .await
            && r.status().is_success()
        {
            let body: serde_json::Value =
                serde_json::from_slice(&r.bytes().await.unwrap_or_default()).unwrap_or_default();
            break body["pid"].as_u64().map_or(launcher, |pid| pid as u32);
        }
        if started.elapsed() > Duration::from_secs(60) {
            fail(&format!("{kind} server did not start"));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    let url = Arc::new(format!("http://127.0.0.1:{port}/v1/chat/completions"));
    let body = Arc::new(
        serde_json::json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "Say x."}],
            "stream": true,
            "max_tokens": pace.tokens,
            "temperature": 0,
        })
        .to_string(),
    );
    let mut results = Vec::new();
    for &n in clients {
        let result = load(url.clone(), body.clone(), pace, n, per_client, pid).await;
        eprintln!("  {kind:>12} {n:>4} clients: {result:?}");
        results.push(result);
    }
    let peak_private_mb = memory::peak_private_mb(pid);
    stop_server();
    ServerResult {
        server: kind.to_string(),
        peak_private_mb,
        load: results,
    }
}

fn furnace_version() -> &'static str {
    include_str!("../../../Cargo.toml")
        .lines()
        .find_map(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or("?")
}

fn print_report(report: &Report) {
    println!(
        "\n{} tokens per stream, one every {} ms; {} streams per client (furnace {}){}\n",
        report.pace.tokens,
        report.pace.token_ms,
        report.requests_per_client,
        report.furnace,
        if report.runs > 1 {
            format!("; medians of {} runs", report.runs)
        } else {
            String::new()
        }
    );
    println!(
        "| Server | Clients | Tokens/s | TTFT p50 / p99 | Lateness p50 / p99 | ITL p50 / p99 | CPU per 1k tokens |"
    );
    println!("|---|---|---|---|---|---|---|");
    for server in &report.servers {
        for l in &server.load {
            println!(
                "| {} | {} | {:.0} | {:.2} / {:.2} ms | {:.2} / {:.2} ms | {:.2} / {:.2} ms | {:.1} ms |",
                server.server,
                l.clients,
                l.tokens_per_second,
                l.ttft_p50_ms,
                l.ttft_p99_ms,
                l.lateness_p50_ms,
                l.lateness_p99_ms,
                l.itl_p50_ms,
                l.itl_p99_ms,
                l.cpu_ms_per_1k_tokens
            );
        }
    }
    println!("\n| Server | Peak memory (private) |\n|---|---|");
    for server in &report.servers {
        println!("| {} | {:.1} MB |", server.server, server.peak_private_mb);
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
    let number = |name: &str, default: u64| {
        flag(name).map_or(default, |v| {
            v.parse()
                .unwrap_or_else(|_| fail(&format!("{name} takes a number")))
        })
    };
    let pace = Pace {
        tokens: number("--tokens", 64) as usize,
        token_ms: number("--token-ms", 20),
    };
    if args.first().map(String::as_str) == Some("serve-gguf") {
        let (Some(path), Some(port), Some(sequences)) = (
            args.get(1),
            args.get(2).and_then(|p| p.parse().ok()),
            args.get(3).and_then(|s| s.parse().ok()),
        ) else {
            fail("usage: serve-gguf PATH PORT SEQUENCES");
        };
        real::serve_gguf(path.as_ref(), port, sequences).await;
        return;
    }
    if args.first().map(String::as_str) == Some("real") {
        let gguf = flag("--gguf").unwrap_or_else(|| fail("real needs --gguf PATH"));
        let clients = flag("--clients")
            .unwrap_or_else(|| "1,8,32".to_string())
            .split(',')
            .map(|c| {
                c.parse()
                    .unwrap_or_else(|_| fail("--clients takes numbers"))
            })
            .collect();
        real::run(real::Options {
            gguf: gguf.into(),
            llama_server: flag("--llama-server").map(Into::into),
            mistralrs: flag("--mistralrs").map(Into::into),
            clients,
            max_tokens: number("--max-tokens", 128) as usize,
            per_client: number("--requests", 1) as usize,
            json: flag("--json"),
            servers: flag("--servers")
                .unwrap_or_else(|| "furnace,llama.cpp,mistral.rs".to_string())
                .split(',')
                .map(str::to_string)
                .collect(),
        })
        .await;
        return;
    }
    if args.first().map(String::as_str) == Some("overhead") {
        overhead(pace.tokens, number("--streams", 500) as usize).await;
        return;
    }
    if args.first().map(String::as_str) == Some("serve") {
        let (Some(kind), Some(port)) = (args.get(1), args.get(2).and_then(|p| p.parse().ok()))
        else {
            fail("usage: serve KIND PORT [--tokens N] [--token-ms MS]");
        };
        serve(kind, port, pace).await;
        return;
    }
    let clients: Vec<usize> = flag("--clients")
        .unwrap_or_else(|| "1,16,64,256".to_string())
        .split(',')
        .map(|c| {
            c.parse()
                .unwrap_or_else(|_| fail("--clients takes numbers"))
        })
        .collect();
    let servers =
        flag("--servers").unwrap_or_else(|| "furnace,scheduled,hand-written,fastapi".to_string());
    let per_client = number("--requests", 4) as usize;

    let runs = number("--runs", 1) as usize;
    if runs == 0 {
        fail("--runs takes a positive number");
    }

    // The servers take turns, so a slow stretch of the machine hits each of them alike
    let kinds: Vec<&str> = servers.split(',').collect();
    let mut samples: Vec<Vec<ServerResult>> = kinds.iter().map(|_| Vec::new()).collect();
    for run in 1..=runs {
        for (kind, samples) in kinds.iter().zip(&mut samples) {
            eprintln!("{kind} (run {run} of {runs})");
            samples.push(bench_server(kind, pace, &clients, per_client).await);
        }
    }
    let report = Report {
        furnace: furnace_version(),
        pace,
        requests_per_client: per_client,
        runs,
        servers: samples.into_iter().map(median_server).collect(),
    };
    print_report(&report);
    if let Some(path) = flag("--json") {
        let json = serde_json::to_string_pretty(&report).expect("report serializes");
        std::fs::write(&path, json).unwrap_or_else(|e| fail(&format!("cannot write {path}: {e}")));
    }
}

mod cpu {
    /// Processor time a process has used (user and kernel), in seconds; 0 where unsupported
    #[cfg(windows)]
    pub fn seconds(pid: u32) -> f64 {
        use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
        use windows_sys::Win32::System::Threading::{
            GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        let ticks =
            |t: FILETIME| (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) as f64;
        // SAFETY: plain Win32 calls on a handle opened and closed here; the FILETIMEs are
        // written by GetProcessTimes on success.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return 0.0;
            }
            let zero = FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            };
            let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
            let ok = GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user);
            CloseHandle(handle);
            if ok == 0 {
                return 0.0;
            }
            // 100 ns ticks
            (ticks(kernel) + ticks(user)) / 1e7
        }
    }

    #[cfg(target_os = "linux")]
    pub fn seconds(pid: u32) -> f64 {
        // utime and stime, fields 14 and 15 of /proc/PID/stat, in clock ticks (100 per second
        // on every common Linux configuration)
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        let fields: Vec<&str> = stat
            .rsplit_once(')')
            .map_or(Vec::new(), |(_, rest)| rest.split_whitespace().collect());
        let tick = |i: usize| {
            fields
                .get(i)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        (tick(11) + tick(12)) / 100.0
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub fn seconds(_pid: u32) -> f64 {
        0.0
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
        // SAFETY: plain Win32 calls on a handle opened and closed here; the counters struct is
        // fully initialized by GetProcessMemoryInfo on success.
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
