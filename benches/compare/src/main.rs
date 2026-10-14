//! Run every engine binary in its own process and print the results as Markdown.
//!
//! ```text
//! cargo run --release -- [--iters N] [--warmup N] [--only furnace,tract,ort,http]
//!                        [--json results.json] [--server PATH]
//! ```
//!
//! Each engine binary prints one JSON `Report`; run one directly to see it, e.g.
//! `target/release/cmp-ort --threads 1`.

use std::path::{Path, PathBuf};
use std::process::Command;

use furnace_compare::{EXPECTED_TOP1, Report};

struct Engine {
    key: &'static str,
    bin: &'static str,
    args: Vec<String>,
}

fn main() {
    let mut iters = "30".to_string();
    let mut warmup = "5".to_string();
    let mut only: Option<Vec<String>> = None;
    let mut json: Option<PathBuf> = None;
    let mut server: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .unwrap_or_else(|| panic!("missing value for {flag}"));
        match flag.as_str() {
            "--iters" => iters = value,
            "--warmup" => warmup = value,
            "--only" => only = Some(value.split(',').map(str::to_string).collect()),
            "--json" => json = Some(value.into()),
            "--server" => server = Some(value),
            other => panic!("unknown argument {other}"),
        }
    }

    let all_threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let engine = |key, bin, args: &[&str]| Engine {
        key,
        bin,
        args: args.iter().map(|s| s.to_string()).collect(),
    };
    let mut engines = vec![engine("furnace", "cmp-furnace", &["--backend", "cpu"])];
    if cfg!(feature = "wgpu") {
        engines.push(engine("wgpu", "cmp-furnace", &["--backend", "wgpu"]));
    }
    let mut http = engine("http", "cmp-http", &[]);
    if let Some(server) = &server {
        http.args = vec!["--server".to_string(), server.clone()];
    }
    engines.push(http);
    engines.push(engine("tract", "cmp-tract", &["--threads", "1"]));
    let tract_mt = all_threads.to_string();
    engines.push(engine("tract", "cmp-tract", &["--threads", &tract_mt]));
    engines.push(engine("ort", "cmp-ort", &["--threads", "0"]));
    engines.push(engine("ort", "cmp-ort", &["--threads", "1"]));

    let dir: PathBuf = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .map(Path::to_path_buf)
        .expect("exe dir");
    let mut reports = Vec::new();
    for e in engines
        .iter()
        .filter(|e| only.as_ref().is_none_or(|o| o.iter().any(|k| k == e.key)))
    {
        let exe = dir
            .join(e.bin)
            .with_extension(std::env::consts::EXE_EXTENSION);
        eprintln!("running {} {}", e.bin, e.args.join(" "));
        let output = Command::new(&exe)
            .args(&e.args)
            .args(["--iters", &iters, "--warmup", &warmup])
            .output()
            .unwrap_or_else(|err| panic!("cannot run {}: {err}", exe.display()));
        if !output.status.success() {
            eprintln!(
                "{} failed ({}):\n{}",
                e.bin,
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .rfind(|l| l.starts_with('{'))
            .expect("report line");
        let report: Report = serde_json::from_str(line).expect("parse report");
        if report.top1 != EXPECTED_TOP1 {
            eprintln!(
                "WARNING: {} top-1 is {}, expected {EXPECTED_TOP1}",
                report.engine, report.top1
            );
        }
        reports.push(report);
    }

    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&reports).expect("json"))
            .expect("write json");
    }
    print_markdown(&reports);
}

fn print_markdown(reports: &[Report]) {
    let batch = |r: &Report, b: usize| r.batches.iter().find(|x| x.batch == b).cloned();
    println!(
        "| Engine | Threads | Batch 1 median / p95 | Batch 4 median / p95 | img/s (b1 / b4) | Top-1 |"
    );
    println!("|---|---|---|---|---|---|");
    for r in reports {
        let lat = |b| {
            batch(r, b).map_or("–".to_string(), |x| {
                format!("{:.1} / {:.1} ms", x.median_ms, x.p95_ms)
            })
        };
        let tput = |b| batch(r, b).map_or("–".to_string(), |x| format!("{:.1}", x.images_per_s));
        println!(
            "| {} | {} | {} | {} | {} / {} | {} ({:.2}) |",
            r.engine,
            r.threads,
            lat(1),
            lat(4),
            tput(1),
            tput(4),
            r.top1,
            r.top1_logit
        );
    }
    println!();
    println!(
        "| Engine | Threads | Load | First inference | Peak working set | Peak private | Binary |"
    );
    println!("|---|---|---|---|---|---|---|");
    for r in reports {
        println!(
            "| {} | {} | {:.0} ms | {:.0} ms | {:.0} MB | {:.0} MB | {:.1} MB |",
            r.engine,
            r.threads,
            r.load_ms,
            r.first_inference_ms,
            r.peak_working_set_mb,
            r.peak_private_mb,
            r.binary_mb
        );
    }
    let server: Vec<_> = reports
        .iter()
        .flat_map(|r| r.batches.iter().map(move |b| (r, b)))
        .filter_map(|(r, b)| b.server_median_ms.map(|s| (r, b, s)))
        .collect();
    if !server.is_empty() {
        println!();
        for (r, b, s) in server {
            println!(
                "{} batch {}: server-reported inference {:.1} ms, HTTP overhead {:.1} ms",
                r.engine,
                b.batch,
                s,
                b.median_ms - s
            );
        }
    }
    println!();
    for r in reports {
        println!("- {}: {}", r.engine, r.version);
    }
}
