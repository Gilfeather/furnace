//! Cold start: from launching the server to its first answers, as a scale-to-zero platform
//! sees it.
//!
//! ```bash
//! furnace-burn-bench cold-start binary ./target/release/furnace --model-name resnet18
//! furnace-burn-bench cold-start docker ghcr.io/gilfeather/furnace:0.16.0
//! ```
//!
//! For each run it starts the server, polls `GET /readyz` every millisecond, then sends one
//! ResNet-18 inference (binary `float32`, all zeros) and records:
//!
//! - *listening*: the first HTTP answer of any kind (the process is accepting requests). The
//!   port is probed with 5 ms connects, so a refused connection's delay is not counted.
//! - *ready*: `/readyz` answered 200 (the warm-up inference has run).
//! - *first inference*: the first `POST /predict` answered 200, counted from launch.
//!
//! then stops the server, and reports the median and the range over the runs.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 3995;
const RUNS: usize = 10;
const CONTAINER: &str = "furnace-cold-start";

fn addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], PORT))
}

/// One HTTP/1.1 request on a fresh connection; the response's status code
fn request(method: &str, path: &str, body: &[u8], timeout: Duration) -> Option<u16> {
    let mut stream = TcpStream::connect_timeout(&addr(), timeout).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/octet-stream\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).ok()?;
    stream.write_all(body).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    // "HTTP/1.1 200 OK"
    std::str::from_utf8(response.get(9..12)?).ok()?.parse().ok()
}

/// A connect with a short timeout: on Windows a refused connection takes about half a second
/// to fail, which would otherwise be counted as start-up time
fn port_open() -> bool {
    TcpStream::connect_timeout(&addr(), Duration::from_millis(5)).is_ok()
}

fn launch(kind: &str, target: &str, args: &[String]) -> Child {
    let mut command = if kind == "binary" {
        let mut command = Command::new(target);
        command.args(["--port", &PORT.to_string()]);
        command
    } else {
        let mut command = Command::new("docker");
        command.args(["run", "--rm", "--name", CONTAINER, "-p"]);
        command.arg(format!("{PORT}:3000")).arg(target);
        command
    };
    command
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| super::fail(&format!("{target}: {e}")))
}

fn stop(kind: &str, mut process: Child) {
    if kind == "docker" {
        let _ = Command::new("docker")
            .args(["stop", "-t", "1", CONTAINER])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    furnace_bench_common::stop(&mut process);
}

/// Milliseconds from launch to listening, ready and the first inference
fn run(kind: &str, target: &str, args: &[String]) -> [f64; 3] {
    let input = vec![0u8; 3 * 224 * 224 * 4];
    let start = Instant::now();
    let process = launch(kind, target, args);
    let (mut listening, mut ready) = (None, None);
    while ready.is_none() && start.elapsed() < Duration::from_secs(60) {
        if !port_open() {
            continue;
        }
        if let Some(status) = request("GET", "/readyz", &[], Duration::from_secs(5)) {
            listening = listening.or(Some(start.elapsed()));
            if status == 200 {
                ready = Some(start.elapsed());
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let (Some(listening), Some(ready)) = (listening, ready) else {
        stop(kind, process);
        super::fail("the server never became ready");
    };
    let status = request("POST", "/predict", &input, Duration::from_secs(30));
    let first = start.elapsed();
    stop(kind, process);
    if status != Some(200) {
        super::fail(&format!("POST /predict answered {status:?}"));
    }
    [listening, ready, first].map(|d| d.as_secs_f64() * 1e3)
}

pub fn main(args: &[String]) {
    let [kind, target, rest @ ..] = args else {
        super::fail("cold-start binary|docker PATH-OR-IMAGE [SERVER ARGS...]");
    };
    if kind != "binary" && kind != "docker" {
        super::fail("cold-start binary|docker PATH-OR-IMAGE [SERVER ARGS...]");
    }
    let results: Vec<[f64; 3]> = (0..RUNS).map(|_| run(kind, target, rest)).collect();
    for (i, name) in ["listening", "ready", "first inference"].iter().enumerate() {
        let mut values: Vec<f64> = results.iter().map(|r| r[i]).collect();
        values.sort_by(f64::total_cmp);
        let median = furnace_bench_common::median(values.clone());
        println!(
            "{kind:6} {name:16} median {median:7.1} ms   (min {:.1}, max {:.1}, {RUNS} runs)",
            values[0],
            values[RUNS - 1]
        );
    }
}
