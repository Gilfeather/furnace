//! The furnace server binary over HTTP on localhost, binary (`application/octet-stream`)
//! request and response, one keep-alive connection, one request at a time.
//!
//! `--server PATH` is the furnace executable (default: `target/release/furnace` of the
//! main crate). Startup is the time from spawning it until `/healthz` answers.

use std::path::PathBuf;
use std::process::Command;

use furnace_compare::{Args, bench_http_server, furnace_version};

fn main() {
    let args = Args::parse("server");
    let server = args.extra.clone().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/furnace")
            .with_extension(std::env::consts::EXE_EXTENSION)
    });

    let mut report = bench_http_server(&args, "furnace server (HTTP, binary)", |port| {
        let mut cmd = Command::new(&server);
        cmd.args(["--model-name", "resnet18", "--port", &port.to_string()])
            .args(["--log-level", "warn"]);
        cmd
    });
    report.version = furnace_version();
    report.binary_mb = std::fs::metadata(&server).map_or(0.0, |m| m.len() as f64 / 1e6);
    report
        .notes
        .push("memory and binary are the server process".to_string());
    report.print();
}
