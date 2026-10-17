//! FastAPI + ONNX Runtime (`fastapi-ort/server.py`) over HTTP on localhost: the usual way
//! to serve an ONNX model from Python, and what the furnace server competes with. It speaks
//! the same binary protocol and uses the same client as `cmp-http`, so the two rows differ
//! only in the server.
//!
//! Python is only the thing being measured here, never part of furnace. Set it up once,
//! from `benches/compare`:
//!
//! ```text
//! uv venv fastapi-ort/.venv
//! uv pip install -p fastapi-ort/.venv -r fastapi-ort/requirements.txt
//! ```
//!
//! `--python PATH` picks another interpreter. The size column is the virtual
//! environment's packages, without the Python interpreter itself.

use std::path::{Path, PathBuf};
use std::process::Command;

use furnace_compare::{Args, bench_http_server, dir_size_mb};

fn main() {
    let args = Args::parse("python");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fastapi-ort");
    let venv = dir.join(".venv");
    let python = args.extra.clone().map(PathBuf::from).unwrap_or_else(|| {
        if cfg!(windows) {
            venv.join("Scripts/python.exe")
        } else {
            venv.join("bin/python")
        }
    });
    let threads = args.threads;

    let mut report = bench_http_server(&args, "FastAPI + ONNX Runtime (HTTP, binary)", |port| {
        let mut cmd = Command::new(&python);
        cmd.arg(dir.join("server.py"))
            .arg("--model")
            .arg(&args.model)
            .args(["--port", &port.to_string()])
            .args(["--threads", &threads.to_string()]);
        cmd
    });
    if threads > 0 {
        report.threads = threads.to_string();
    }
    report.version = versions(&python);
    report.binary_mb = dir_size_mb(&venv);
    report.notes.extend([
        "memory is the Python server process".to_string(),
        "size is the virtual environment, without the Python interpreter".to_string(),
    ]);
    report.print();
}

/// Python, FastAPI, uvicorn and ONNX Runtime versions, as the server imports them
fn versions(python: &Path) -> String {
    let script = "import sys, fastapi, uvicorn, onnxruntime as o; \
                  print(f'Python {sys.version.split()[0]}, FastAPI {fastapi.__version__}, \
                  uvicorn {uvicorn.__version__}, ONNX Runtime {o.__version__}')";
    Command::new(python)
        .args(["-c", script])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}
