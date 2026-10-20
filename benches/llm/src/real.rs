//! `real`: one GGUF file on the CPU, served by furnace + candle (furnace-candle), by llama.cpp's
//! `llama-server` and by `mistralrs serve`, under the same streaming chat requests.
//!
//! ```bash
//! furnace-llm-bench real --gguf qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!     [--llama-server PATH] [--mistralrs PATH] [--clients 1,8,32] [--max-tokens 128]
//!     [--servers furnace,llama.cpp,mistral.rs]
//! ```
//!
//! Each request asks for a story with greedy sampling and `max_tokens`, so the servers generate
//! about the same tokens; the client counts them from the usage chunk when there is one, else
//! from the content chunks. Prompt caches are off (llama.cpp's `cache_prompt`, mistral.rs's
//! prefix cache), and each prompt begins with its own number, so every request pays its
//! prefill, as furnace's do. Each server runs with its own defaults for threads; each gets as
//! many sequences as the most clients.
//!
//! Before the servers, the model runs in this process with furnace-candle alone (prefill, then
//! greedy decoding, no HTTP and no scheduler): the engine's own speed, for one stream.
//!
//! The numbers compare engines as much as servers: llama.cpp's CPU kernels, mistral.rs's
//! continuous batching and candle's batch-1 decoding differ far more than their HTTP layers do.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use furnace::Server;
use furnace::llm::{Message, StepModel};
use furnace_candle::Gguf;
use serde::Serialize;

use furnace_bench_common::{free_port, memory, percentile, write_json};

use super::{SERVER, cpu, fail, stop_server};

/// Longest a whole stream may take
const TIMEOUT: Duration = Duration::from_secs(600);
const PROMPT: &str = "Write a long story about a lighthouse keeper and a storm.";

pub struct Options {
    pub gguf: PathBuf,
    pub llama_server: Option<PathBuf>,
    pub mistralrs: Option<PathBuf>,
    pub clients: Vec<usize>,
    pub max_tokens: usize,
    pub per_client: usize,
    pub json: Option<String>,
    /// Which of `furnace`, `llama.cpp` and `mistral.rs` to run
    pub servers: Vec<String>,
}

/// `serve-gguf PATH PORT SEQUENCES`: furnace + candle on its own
pub async fn serve_gguf(path: &Path, port: u16, sequences: usize) {
    let engine = Gguf::new(path)
        .name("model")
        .max_sequences(sequences)
        .max_context(2048)
        .load()
        .unwrap_or_else(|e| fail(&format!("loading {}: {e}", path.display())));
    let app = Server::new()
        .configure(|c| c.max_concurrent_requests = 10_000)
        .llm(engine)
        .router()
        .expect("a server with something to serve");
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap_or_else(|e| fail(&format!("cannot bind port {port}: {e}")));
    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| fail(&format!("server: {e}")));
}

#[derive(Debug, Serialize)]
struct InProcess {
    prompt_tokens: usize,
    prefill_ms: f64,
    tokens: usize,
    tokens_per_second: f64,
}

/// furnace-candle alone: prefill, then greedy decoding of `tokens` tokens
fn in_process(path: &Path, tokens: usize) -> InProcess {
    let (model, pipeline) = Gguf::new(path)
        .max_context(2048)
        .load_parts()
        .unwrap_or_else(|e| fail(&format!("loading {}: {e}", path.display())));
    let text = pipeline
        .apply_chat_template(&[Message::new("user", &format!("1. {PROMPT}"))])
        .unwrap_or_else(|e| fail(&e.to_string()));
    let prompt = pipeline
        .encode(&text, false)
        .unwrap_or_else(|e| fail(&e.to_string()));
    let argmax = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u32)
    };
    // Once to warm up, then measured
    let run = || {
        let start = Instant::now();
        let (mut sequence, logits) = model
            .prefill(&prompt)
            .unwrap_or_else(|e| fail(&e.to_string()));
        let prefill = start.elapsed();
        let mut token = argmax(&logits.0);
        let decode = Instant::now();
        for _ in 1..tokens {
            let logits = model
                .decode(&mut sequence, token)
                .unwrap_or_else(|e| fail(&e.to_string()));
            token = argmax(&logits.0);
        }
        (prefill, decode.elapsed())
    };
    run();
    let (prefill, decode) = run();
    InProcess {
        prompt_tokens: prompt.len(),
        prefill_ms: prefill.as_secs_f64() * 1e3,
        tokens,
        tokens_per_second: (tokens - 1) as f64 / decode.as_secs_f64(),
    }
}

/// One stream's chunks: when each piece of content arrived, and the tokens the server counted
struct Stream {
    arrivals: Vec<Duration>,
    completion_tokens: Option<usize>,
    /// The last event, to show when a stream went wrong
    last_event: String,
}

impl Stream {
    fn tokens(&self) -> usize {
        self.completion_tokens.unwrap_or(self.arrivals.len())
    }
}

async fn stream(client: &reqwest::Client, url: &str, body: String) -> Stream {
    let sent = Instant::now();
    let mut response = client
        .post(url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap_or_else(|e| fail(&format!("request: {e}")));
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        fail(&format!("status {status}: {text}"));
    }
    let mut arrivals = Vec::new();
    let mut completion_tokens = None;
    let mut pending = Vec::new();
    let mut last_event = String::new();
    loop {
        let piece = match response.chunk().await {
            Ok(Some(piece)) => piece,
            Ok(None) => break,
            Err(e) => fail(&format!("stream: {e}")),
        };
        let now = sent.elapsed();
        pending.extend_from_slice(&piece);
        while let Some(end) = pending.windows(2).position(|w| w == b"\n\n") {
            let event: Vec<u8> = pending.drain(..end + 2).collect();
            last_event = String::from_utf8_lossy(&event).trim().to_string();
            let Some(data) = event.strip_prefix(b"data: ") else {
                continue;
            };
            let Ok(chunk) = serde_json::from_slice::<serde_json::Value>(data) else {
                continue;
            };
            let content = chunk["choices"][0]["delta"]["content"].as_str();
            if content.is_some_and(|c| !c.is_empty()) {
                arrivals.push(now);
            }
            if let Some(tokens) = chunk["usage"]["completion_tokens"].as_u64() {
                completion_tokens = Some(tokens as usize);
            }
        }
    }
    Stream {
        arrivals,
        completion_tokens,
        last_event,
    }
}

#[derive(Debug, Serialize)]
struct Load {
    clients: usize,
    streams: usize,
    /// Streams that ended with an error, or without any content; the figures below leave them
    /// out
    failed: usize,
    /// The first failure's last event
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<String>,
    /// Tokens per stream, on average
    completion_tokens: f64,
    ttft_p50_ms: f64,
    ttft_p99_ms: f64,
    itl_p50_ms: f64,
    itl_p99_ms: f64,
    /// One stream's decoding speed: tokens after the first over the time after the first
    stream_tokens_per_second: f64,
    /// Every stream's tokens over the wall time
    tokens_per_second: f64,
    /// Server CPU time per token streamed
    cpu_ms_per_token: f64,
}

async fn load(
    url: Arc<String>,
    model: Arc<String>,
    max_tokens: usize,
    clients: usize,
    per_client: usize,
    pid: u32,
) -> Load {
    let body = move |n: usize| {
        serde_json::json!({
            "model": *model,
            "messages": [{"role": "user", "content": format!("{n}. {PROMPT}")}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": max_tokens,
            "temperature": 0,
            "cache_prompt": false,
        })
        .to_string()
    };
    let body = Arc::new(body);
    // A new connection per stream: llama-server closes connections after a stream, and a
    // client reusing one at that moment fails now and then. Connecting costs well under a
    // millisecond, the same for every server.
    let connect = || {
        reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .timeout(TIMEOUT)
            .build()
            .expect("client")
    };
    let call = move |client: reqwest::Client, c: usize, i: usize| {
        let (url, body) = (url.clone(), body(c * 1000 + i));
        async move { stream(&client, &url, body).await }
    };
    let mut cpu_before = 0.0;
    // One stream per client to warm up
    let run = furnace_bench_common::load(clients, 1, per_client, connect, call, || {
        cpu_before = cpu::seconds(pid);
    })
    .await;
    let cpu = cpu::seconds(pid) - cpu_before;
    let (streams, wall) = (run.results, run.wall);

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let (mut ttft, mut itl) = (Vec::new(), Vec::new());
    let (mut decode_tokens, mut decode_seconds) = (0.0, 0.0);
    let mut tokens = 0;
    let (mut failed, mut failure) = (0, None);
    for s in &streams {
        let (Some(first), Some(last)) = (s.arrivals.first(), s.arrivals.last()) else {
            failed += 1;
            failure.get_or_insert_with(|| s.last_event.clone());
            continue;
        };
        tokens += s.tokens();
        ttft.push(ms(*first));
        itl.extend(s.arrivals.windows(2).map(|w| ms(w[1] - w[0])));
        if s.arrivals.len() > 1 {
            // Tokens per content chunk can be more than one; count the stream's tokens
            decode_tokens += (s.tokens() - 1) as f64;
            decode_seconds += (*last - *first).as_secs_f64();
        }
    }
    let served = streams.len() - failed;
    for values in [&mut ttft, &mut itl] {
        values.sort_by(f64::total_cmp);
    }
    let percentile = |values: &[f64], p: f64| {
        if values.is_empty() {
            f64::NAN
        } else {
            percentile(values, p)
        }
    };
    Load {
        clients,
        streams: streams.len(),
        failed,
        failure,
        completion_tokens: tokens as f64 / served as f64,
        ttft_p50_ms: percentile(&ttft, 0.5),
        ttft_p99_ms: percentile(&ttft, 0.99),
        itl_p50_ms: percentile(&itl, 0.5),
        itl_p99_ms: percentile(&itl, 0.99),
        stream_tokens_per_second: decode_tokens / decode_seconds,
        tokens_per_second: tokens as f64 / wall.as_secs_f64(),
        cpu_ms_per_token: cpu * 1e3 / tokens as f64,
    }
}

#[derive(Debug, Serialize)]
struct ServerResult {
    server: &'static str,
    ready_s: f64,
    peak_private_mb: f64,
    load: Vec<Load>,
}

async fn bench(
    kind: &'static str,
    mut command: std::process::Command,
    options: &Options,
) -> ServerResult {
    let port = free_port();
    let sequences = options.clients.iter().copied().max().unwrap_or(1);
    match kind {
        "furnace + candle" => {
            command.args([
                "serve-gguf",
                &options.gguf.to_string_lossy(),
                &port.to_string(),
                &sequences.to_string(),
            ]);
        }
        "llama.cpp" => {
            command.args(["-m", &options.gguf.to_string_lossy()]).args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "-np",
                &sequences.to_string(),
                "-c",
                &(sequences * 2048).to_string(),
                "--no-webui",
            ]);
        }
        _ => {
            let dir = options.gguf.parent().unwrap_or(Path::new("."));
            let file = options
                .gguf
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default();
            command
                .args([
                    "serve",
                    "-p",
                    &port.to_string(),
                    "--host",
                    "127.0.0.1",
                    "--no-ui",
                ])
                .args([
                    "--max-seqs",
                    &sequences.to_string(),
                    "--prefix-cache-n",
                    "0",
                    "--cpu",
                    "-m",
                    &dir.to_string_lossy(),
                    "-f",
                    &file,
                ]);
        }
    }
    let log = std::env::temp_dir().join(format!(
        "furnace-llm-bench-{}.log",
        kind.replace([' ', '+', '.'], "")
    ));
    let child = command
        .stdout(std::process::Stdio::null())
        .stderr(std::fs::File::create(&log).map_or(std::process::Stdio::null(), Into::into))
        .spawn()
        .unwrap_or_else(|e| fail(&format!("cannot start {kind}: {e}")));
    let pid = child.id();
    *SERVER.lock().unwrap_or_else(|e| e.into_inner()) = Some(child);
    let probe = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client");
    let started = Instant::now();
    // Ready once it lists its model; its id is what the requests name
    let model = loop {
        if let Ok(r) = probe
            .get(format!("http://127.0.0.1:{port}/v1/models"))
            .send()
            .await
            && r.status().is_success()
        {
            let body: serde_json::Value =
                serde_json::from_slice(&r.bytes().await.unwrap_or_default()).unwrap_or_default();
            if let Some(id) = body["data"][0]["id"].as_str() {
                break id.to_string();
            }
        }
        if started.elapsed() > Duration::from_secs(300) {
            fail(&format!("{kind} did not start; see {}", log.display()));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let ready_s = started.elapsed().as_secs_f64();
    let url = Arc::new(format!("http://127.0.0.1:{port}/v1/chat/completions"));
    let model = Arc::new(model);
    let mut results = Vec::new();
    for &n in &options.clients {
        let result = load(
            url.clone(),
            model.clone(),
            options.max_tokens,
            n,
            options.per_client,
            pid,
        )
        .await;
        eprintln!("  {kind:>16} {n:>3} clients: {result:?}");
        results.push(result);
    }
    let peak_private_mb = memory::of_process(pid).peak_private_mb;
    stop_server();
    ServerResult {
        server: kind,
        ready_s,
        peak_private_mb,
        load: results,
    }
}

#[derive(Debug, Serialize)]
struct Report {
    gguf: String,
    max_tokens: usize,
    requests_per_client: usize,
    in_process: InProcess,
    servers: Vec<ServerResult>,
}

pub async fn run(options: Options) {
    eprintln!("furnace-candle in process");
    let in_process = in_process(&options.gguf, options.max_tokens);
    eprintln!("  {in_process:?}");
    let wanted = |name: &str| options.servers.iter().any(|s| s == name);
    let mut servers = Vec::new();
    if wanted("furnace") {
        let own = std::env::current_exe().expect("own path");
        let command = std::process::Command::new(own);
        servers.push(bench("furnace + candle", command, &options).await);
    }
    if let Some(path) = options
        .llama_server
        .as_ref()
        .filter(|_| wanted("llama.cpp"))
    {
        servers.push(bench("llama.cpp", std::process::Command::new(path), &options).await);
    }
    if let Some(path) = options.mistralrs.as_ref().filter(|_| wanted("mistral.rs")) {
        servers.push(bench("mistral.rs", std::process::Command::new(path), &options).await);
    }
    let report = Report {
        gguf: options
            .gguf
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        max_tokens: options.max_tokens,
        requests_per_client: options.per_client,
        in_process,
        servers,
    };
    print_report(&report);
    if let Some(path) = &options.json {
        write_json(path, &report);
    }
}

fn print_report(report: &Report) {
    let p = &report.in_process;
    println!(
        "\n{}, up to {} tokens per stream, {} streams per client\n\nfurnace-candle in process, one stream: {} prompt tokens prefilled in {:.0} ms, {:.1} tokens/s\n",
        report.gguf,
        report.max_tokens,
        report.requests_per_client,
        p.prompt_tokens,
        p.prefill_ms,
        p.tokens_per_second
    );
    println!(
        "| Server | Clients | Failed | Tokens per stream | TTFT p50 / p99 | ITL p50 / p99 | Tokens/s per stream | Tokens/s total | CPU per token |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for server in &report.servers {
        for l in &server.load {
            println!(
                "| {} | {} | {} of {} | {:.0} | {:.0} / {:.0} ms | {:.1} / {:.1} ms | {:.1} | {:.1} | {:.0} ms |",
                server.server,
                l.clients,
                l.failed,
                l.streams,
                l.completion_tokens,
                l.ttft_p50_ms,
                l.ttft_p99_ms,
                l.itl_p50_ms,
                l.itl_p99_ms,
                l.stream_tokens_per_second,
                l.tokens_per_second,
                l.cpu_ms_per_token
            );
        }
    }
    for server in &report.servers {
        for l in server.load.iter().filter(|l| l.failed > 0) {
            if let Some(failure) = &l.failure {
                println!(
                    "\n{} at {} clients failed with: {failure}",
                    server.server, l.clients
                );
            }
        }
    }
    println!("\n| Server | Ready after | Peak memory (private) |\n|---|---|---|");
    for server in &report.servers {
        println!(
            "| {} | {:.1} s | {:.0} MB |",
            server.server, server.ready_s, server.peak_private_mb
        );
    }
}
