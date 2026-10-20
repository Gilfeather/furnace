//! What the bench harnesses (`benches/burn`, `llm`, `nerve` and `compare`) measure with, in
//! one place, so a figure means the same thing in every table:
//!
//! - [`percentile`] and [`median`]: one definition each, written down below.
//! - [`memory::of_process`]: a process's private and peak memory.
//! - [`free_port`], [`wait_until_healthy`] and [`stop`]: a server in its own process.
//! - [`python`]: the interpreter for the FastAPI comparisons.
//! - [`load()`] (feature `load`): N clients sending back to back.
//! - [`fail`], [`write_json`] and [`furnace_version`]: reporting.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(feature = "load")]
mod load;
pub mod memory;

#[cfg(feature = "load")]
pub use load::{Run, load};

/// Print `message` after the program's name and exit with status 2
pub fn fail(message: &str) -> ! {
    let program = std::env::current_exe()
        .ok()
        .and_then(|p| Some(p.file_stem()?.to_string_lossy().into_owned()))
        .unwrap_or_default();
    eprintln!("{program}: {message}");
    std::process::exit(2)
}

/// The nearest-rank percentile of `sorted`, which must be in ascending order: the smallest
/// sample with at least a fraction `p` of all samples at or below it, `sorted[⌈p·n⌉ − 1]`
/// (the first sample for `p = 0`). It never interpolates, so it is always a measured value,
/// and `p = 1` is the maximum. Every percentile in every harness is this one.
///
/// # Panics
///
/// If `sorted` is empty.
pub fn percentile<T: Copy>(sorted: &[T], p: f64) -> T {
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The median of `values`: the middle one, or the mean of the two middle ones for an even
/// count. Used to combine repeated runs, where each run is already a summary.
///
/// # Panics
///
/// If `values` is empty.
pub fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    (values[(n - 1) / 2] + values[n / 2]) / 2.0
}

/// A port on 127.0.0.1 nobody listens on now
pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or_else(|e| fail(&format!("no free port: {e}")))
}

/// Poll `GET /healthz` on 127.0.0.1:`port` every 5 ms until it answers 2xx, and return the
/// pid of the process that does the work: the `pid` field of the answer's JSON if it has one
/// (on Windows a virtual environment's `python.exe` only launches the interpreter), else
/// `child`'s.
///
/// Each poll connects with a 5 ms timeout first: on Windows a refused connection takes about
/// half a second to report, which would swamp the start-up time. A server that accepts but
/// does not answer within 2 s is polled again rather than waited for.
///
/// Stops `child` and [`fail`]s if it exits first or `timeout` passes.
pub fn wait_until_healthy(name: &str, child: &mut Child, port: u16, timeout: Duration) -> u32 {
    let start = Instant::now();
    loop {
        if let Some(body) = healthz(port) {
            return serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| u32::try_from(v["pid"].as_u64()?).ok())
                .unwrap_or_else(|| child.id());
        }
        if let Ok(Some(status)) = child.try_wait() {
            fail(&format!("{name} exited during start-up: {status}"));
        }
        if start.elapsed() > timeout {
            stop(child);
            fail(&format!("{name} did not start within {timeout:?}"));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// One `GET /healthz` on a fresh connection: the body of a 2xx answer
fn healthz(port: u16) -> Option<String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(5)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .ok()?;
    let head =
        format!("GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    stream.write_all(head.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let response = String::from_utf8_lossy(&response);
    // "HTTP/1.1 200 OK"
    if !response.get(9..10).is_some_and(|class| class == "2") {
        return None;
    }
    let (head, body) = response.split_once("\r\n\r\n")?;
    let chunked = head
        .lines()
        .any(|l| l.eq_ignore_ascii_case("transfer-encoding: chunked"));
    // A chunked body: the first chunk is the whole of a small JSON answer
    let body = if chunked {
        body.split("\r\n").nth(1).unwrap_or_default()
    } else {
        body
    };
    Some(body.to_string())
}

/// Stop a server with every process it started (uvicorn's workers, Python's launcher): on
/// Windows `taskkill /T /F` on its tree, elsewhere `SIGTERM` first, on which uvicorn stops
/// its workers; then kill it and reap it.
pub fn stop(child: &mut Child) {
    let pid = child.id().to_string();
    let _ = if cfg!(windows) {
        Command::new("taskkill")
            .args(["/PID", &pid, "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    } else {
        Command::new("kill").args(["-TERM", &pid]).status()
    };
    let _ = child.kill();
    let _ = child.wait();
}

/// The Python with FastAPI, uvicorn and numpy: `$FURNACE_BENCH_PYTHON`, else the virtual
/// environment of `benches/compare/fastapi-ort`
pub fn python() -> PathBuf {
    std::env::var_os("FURNACE_BENCH_PYTHON").map_or_else(
        || {
            let venv = Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../compare/fastapi-ort/.venv"
            ));
            if cfg!(windows) {
                venv.join("Scripts/python.exe")
            } else {
                venv.join("bin/python")
            }
        },
        PathBuf::from,
    )
}

/// furnace's version, from its Cargo.toml (every harness builds furnace from this checkout)
pub fn furnace_version() -> &'static str {
    manifest_version(include_str!("../../../Cargo.toml"), "version")
}

/// The version a dependency of furnace asks for, from its Cargo.toml (`burn` gives "0.22")
pub fn furnace_dependency(name: &str) -> &'static str {
    manifest_version(include_str!("../../../Cargo.toml"), name)
}

/// `name = "x"` or `name = { version = "x", .. }` on a line of `manifest`; "?" if absent
fn manifest_version(manifest: &'static str, name: &str) -> &'static str {
    manifest
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix(name)?.trim_start().strip_prefix('=')?;
            let rest = rest.trim_start();
            let rest = rest
                .strip_prefix('{')
                .and_then(|r| r.trim_start().strip_prefix("version"))
                .and_then(|r| r.trim_start().strip_prefix('='))
                .map_or(rest, str::trim_start);
            rest.strip_prefix('"')?.split('"').next()
        })
        .unwrap_or("?")
}

/// Write `report` to `path` as pretty JSON, or [`fail`]
pub fn write_json(path: &str, report: &impl serde::Serialize) {
    let json = serde_json::to_string_pretty(report).expect("report serializes");
    std::fs::write(path, json).unwrap_or_else(|e| fail(&format!("cannot write {path}: {e}")));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank() {
        let sorted: Vec<u32> = (1..=10).collect();
        assert_eq!(percentile(&sorted, 0.0), 1);
        assert_eq!(percentile(&sorted, 0.5), 5);
        assert_eq!(percentile(&sorted, 0.95), 10);
        assert_eq!(percentile(&sorted, 1.0), 10);
        assert_eq!(percentile(&[7], 0.99), 7);
        let hundred: Vec<u32> = (1..=100).collect();
        assert_eq!(percentile(&hundred, 0.95), 95);
        assert_eq!(percentile(&hundred, 0.99), 99);
    }

    #[test]
    fn median_of_even_and_odd() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn versions_from_the_manifest() {
        assert_ne!(furnace_version(), "?");
        assert_eq!(manifest_version("burn = \"0.22\"\n", "burn"), "0.22");
        assert_eq!(
            manifest_version(
                "burn = { version = \"0.22\", default-features = false }",
                "burn"
            ),
            "0.22"
        );
        assert_eq!(manifest_version("burn-store = \"0.1\"", "burn"), "?");
        assert_eq!(manifest_version("", "burn"), "?");
    }
}
