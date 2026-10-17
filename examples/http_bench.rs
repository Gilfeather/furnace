//! End-to-end HTTP latency of `POST /predict` against a running ResNet-18 server.
//!
//! ```bash
//! cargo run --release --example http_bench -- PORT [MODE[@CLIENTS] ...]
//! ```
//!
//! Modes: `json1`, `json4` (JSON, 1 or 4 images); `bin1`, `bin4` (raw little-endian `f32`
//! with `X-Batch-Size`); append `o` (e.g. `bin4o`) to also request a binary response.
//! `image=PATH` sends a JPEG or PNG file as is (needs a server built with `--features image`).
//!
//! Request bodies are built before timing. Each sample covers sending the request, reading
//! the full response and decoding it, on one keep-alive connection. `io-overhead` is the
//! end-to-end median minus the server-reported inference time.
//!
//! Append `@N` to a mode (e.g. `bin1o@8`) for concurrent load: N clients, each on its own
//! connection, send requests back to back. After a warm-up every client sends
//! `CONCURRENT_RUNS` requests; the report gives throughput over that phase and the latency
//! percentiles of every request in it.

use reqwest::Client;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Barrier;

const N: usize = 3 * 224 * 224;
const RUNS: usize = 30;
const WARMUP: usize = 3;
/// Timed requests per client in `@N` mode
const CONCURRENT_RUNS: usize = 40;

/// Deterministic standard-normal values (SplitMix64 + Box-Muller), so runs are comparable
fn make_images(count: usize) -> Vec<Vec<f32>> {
    let mut state = 0u64;
    let mut next_unit = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Uniform in (0, 1]
        ((z >> 11) + 1) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            (0..N)
                .map(|_| {
                    let (u1, u2) = (next_unit(), next_unit());
                    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
                })
                .collect()
        })
        .collect()
}

fn make_body(mode: &str, images: &[Vec<f32>]) -> Result<(HeaderMap, Vec<u8>), String> {
    let mut headers = HeaderMap::new();
    if let Some(path) = mode.strip_prefix("image=") {
        let content_type = match path.rsplit('.').next().map(str::to_ascii_lowercase) {
            Some(ext) if ext == "png" => "image/png",
            Some(ext) if ext == "jpg" || ext == "jpeg" => "image/jpeg",
            _ => return Err(format!("{path}: expected a .jpg, .jpeg or .png file")),
        };
        let body = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        headers.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
        return Ok((headers, body));
    }

    let (kind, batch, binary_out) = match mode {
        "json1" => ("json", 1, false),
        "json4" => ("json", 4, false),
        "bin1" => ("bin", 1, false),
        "bin4" => ("bin", 4, false),
        "bin1o" => ("bin", 1, true),
        "bin4o" => ("bin", 4, true),
        _ => return Err(format!("unknown mode: {mode}")),
    };
    if binary_out {
        headers.insert(ACCEPT, HeaderValue::from_static("application/octet-stream"));
    }

    if kind == "json" {
        // ~7 significant digits, as a float32-aware JSON encoder would write them
        let rows: Vec<Vec<f64>> = images[..batch]
            .iter()
            .map(|im| {
                im.iter()
                    .map(|v| format!("{v:.6e}").parse().unwrap())
                    .collect()
            })
            .collect();
        let doc = if batch == 1 {
            serde_json::json!({ "input": rows[0] })
        } else {
            serde_json::json!({ "inputs": rows })
        };
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        return Ok((headers, serde_json::to_vec(&doc).unwrap()));
    }

    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    headers.insert("x-batch-size", HeaderValue::from(batch));
    let body = images[..batch]
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok((headers, body))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Upper decile, same method as Python's `statistics.quantiles(n=10)[-1]` (exclusive)
fn p90(sorted: &[f64]) -> f64 {
    quantile(sorted, 0.9)
}

/// Same method as Python's `statistics.quantiles` (exclusive) on sorted values
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() + 1) as f64;
    let lower = (pos.floor() as usize).clamp(1, sorted.len());
    let upper = (lower + 1).min(sorted.len());
    let frac = pos - pos.floor();
    sorted[lower - 1] + (sorted[upper - 1] - sorted[lower - 1]) * frac
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let port: u16 = args
        .next()
        .ok_or("usage: http_bench PORT [MODE ...]")?
        .parse()?;
    let mut modes: Vec<String> = args.collect();
    if modes.is_empty() {
        modes = ["json1", "json4", "bin1", "bin4"]
            .map(String::from)
            .to_vec();
    }

    let images = make_images(4);
    // One connection, reused for every request
    let client = Client::builder().pool_max_idle_per_host(1).build()?;
    let url = format!("http://127.0.0.1:{port}/predict");

    for mode in &modes {
        let (mode, clients) = match mode.rsplit_once('@') {
            Some((mode, n)) => (
                mode,
                Some(n.parse::<usize>().map_err(|e| format!("{n}: {e}"))?),
            ),
            None => (mode.as_str(), None),
        };
        let (headers, payload) = make_body(mode, &images)?;
        // Long image paths would break the column layout
        let label = match mode.strip_prefix("image=") {
            Some(path) => path.rsplit(['/', '\\']).next().unwrap_or(path),
            None => mode,
        };
        let payload = bytes::Bytes::from(payload);
        if let Some(clients) = clients {
            concurrent(&url, label, &headers, &payload, clients).await?;
            continue;
        }
        let mut totals = Vec::with_capacity(RUNS);
        let mut inference = Vec::with_capacity(RUNS);

        for i in 0..WARMUP + RUNS {
            let (elapsed_ms, server_ms) = send(&client, &url, &headers, &payload)
                .await
                .map_err(|e| format!("{mode}: {e}"))?;
            if i >= WARMUP {
                totals.push(elapsed_ms);
                inference.push(server_ms);
            }
        }

        let median_ms = median(&mut totals);
        let p90_ms = p90(&totals);
        let server_ms = median(&mut inference);
        println!(
            "{label:6} body={:5.2}MB  e2e median={median_ms:7.1}ms p90={p90_ms:7.1}ms  \
             server inference={server_ms:6.1}ms  io-overhead={:5.1}ms",
            payload.len() as f64 / 1e6,
            median_ms - server_ms
        );
    }
    Ok(())
}

/// Send one request and decode the response: (end-to-end ms, server-reported inference ms)
async fn send(
    client: &Client,
    url: &str,
    headers: &HeaderMap,
    payload: &bytes::Bytes,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();
    let response = client
        .post(url)
        .headers(headers.clone())
        .body(payload.clone())
        .send()
        .await?;
    let status = response.status();
    let is_json = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("json"));
    let header_ms = response
        .headers()
        .get("x-inference-time-ms")
        .and_then(|v| v.to_str().ok()?.parse::<f64>().ok());
    let raw = response.bytes().await?;
    if !status.is_success() {
        let text = String::from_utf8_lossy(&raw[..raw.len().min(300)]).into_owned();
        return Err(format!("HTTP {status} {text}").into());
    }
    let server_ms = if is_json {
        let doc: serde_json::Value = serde_json::from_slice(&raw)?;
        doc["inference_time_ms"]
            .as_f64()
            .ok_or("missing inference_time_ms")?
    } else {
        let (chunks, _) = raw.as_chunks::<4>();
        let decoded: Vec<f32> = chunks.iter().copied().map(f32::from_le_bytes).collect();
        std::hint::black_box(decoded);
        header_ms.ok_or("missing X-Inference-Time-Ms")?
    };
    Ok((start.elapsed().as_secs_f64() * 1000.0, server_ms))
}

/// `clients` connections sending requests back to back: throughput and latency percentiles
async fn concurrent(
    url: &str,
    label: &str,
    headers: &HeaderMap,
    payload: &bytes::Bytes,
    clients: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let items = headers
        .get("x-batch-size")
        .and_then(|v| v.to_str().ok()?.parse::<usize>().ok())
        .unwrap_or(1);
    // Everyone warms up, then the timed phase starts for all clients at once
    let barrier = Arc::new(Barrier::new(clients + 1));
    let tasks: Vec<_> = (0..clients)
        .map(|_| {
            let (url, headers, payload) = (url.to_string(), headers.clone(), payload.clone());
            let barrier = barrier.clone();
            tokio::spawn(async move {
                let client = Client::builder().pool_max_idle_per_host(1).build()?;
                for _ in 0..WARMUP {
                    send(&client, &url, &headers, &payload).await?;
                }
                barrier.wait().await;
                let mut latencies = Vec::with_capacity(CONCURRENT_RUNS);
                for _ in 0..CONCURRENT_RUNS {
                    latencies.push(send(&client, &url, &headers, &payload).await?.0);
                }
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(latencies)
            })
        })
        .collect();
    barrier.wait().await;
    let start = Instant::now();
    let mut latencies = Vec::with_capacity(clients * CONCURRENT_RUNS);
    for task in tasks {
        latencies.extend(task.await?.map_err(|e| format!("{label}: {e}"))?);
    }
    let wall = start.elapsed().as_secs_f64();
    latencies.sort_by(f64::total_cmp);
    let requests = latencies.len() as f64 / wall;
    println!(
        "{label:6} clients={clients:3}  throughput={requests:6.1} req/s ({:6.1} items/s)  \
         latency p50={:7.1}ms p95={:7.1}ms",
        requests * items as f64,
        quantile(&latencies, 0.5),
        quantile(&latencies, 0.95),
    );
    Ok(())
}
