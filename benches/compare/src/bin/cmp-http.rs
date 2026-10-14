//! The furnace server binary over HTTP on localhost, binary (`application/octet-stream`)
//! request and response, one keep-alive connection, one request at a time.
//!
//! `--server PATH` is the furnace executable (default: `target/release/furnace` of the
//! main crate). Startup is the time from spawning it until `/healthz` answers.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use furnace_compare::{
    Args, BatchResult, CLASSES, Report, argmax, batch_input, memory, preprocess,
};

fn main() {
    let args = Args::parse("server");
    let server = args.extra.clone().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/furnace")
            .with_extension(std::env::consts::EXE_EXTENSION)
    });
    let image = preprocess(&args.image);

    // A port nobody listens on, so /healthz can only be answered by our own server
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("free port")
        .port();

    let start = Instant::now();
    let mut child = Command::new(&server)
        .args(["--model-name", "resnet18", "--port", &port.to_string()])
        .args(["--log-level", "warn"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot start {}: {e}", server.display()));
    // Kill the server even if a request below panics
    struct KillOnDrop(std::process::Child);
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(60)))
        .build()
        .into();
    let base = format!("http://127.0.0.1:{port}");
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    loop {
        // Windows retries a refused SYN for ~0.5 s, so probe the port with a short
        // connect timeout before asking /healthz, or startup is overstated
        let listening =
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(10)).is_ok();
        if listening && agent.get(format!("{base}/healthz")).call().is_ok() {
            break;
        }
        if let Some(status) = child.try_wait().expect("try_wait") {
            panic!("furnace exited during startup: {status}");
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "server did not start"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pid = child.id();
    let mut guard = KillOnDrop(child);

    // Returns (logits, server-reported inference ms)
    let predict = |body: &[u8], batch: usize| -> (Vec<f32>, f64) {
        let mut resp = agent
            .post(format!("{base}/predict"))
            .header("Content-Type", "application/octet-stream")
            .header("Accept", "application/octet-stream")
            .header("X-Batch-Size", batch.to_string())
            .send(body)
            .expect("POST /predict");
        let server_ms = resp
            .headers()
            .get("x-inference-time-ms")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN);
        let bytes = resp.body_mut().read_to_vec().expect("read body");
        let logits = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        (logits, server_ms)
    };
    let to_body =
        |floats: &[f32]| -> Vec<u8> { floats.iter().flat_map(|f| f.to_le_bytes()).collect() };

    let one = to_body(&image);
    let start = Instant::now();
    let (logits, _) = predict(&one, 1);
    let first_inference_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(logits.len(), CLASSES);
    let top1 = argmax(&logits);

    let batches = args
        .batches
        .iter()
        .map(|&batch| {
            let body = to_body(&batch_input(&image, batch));
            for _ in 0..args.warmup {
                predict(&body, batch);
            }
            let mut server_ms = Vec::with_capacity(args.iters);
            let mut samples: Vec<Duration> = (0..args.iters)
                .map(|_| {
                    let start = Instant::now();
                    let (out, ms) = predict(&body, batch);
                    let elapsed = start.elapsed();
                    assert_eq!(out.len(), batch * CLASSES);
                    server_ms.push(ms);
                    elapsed
                })
                .collect();
            let mut result = BatchResult::from_samples(batch, &mut samples);
            server_ms.sort_by(f64::total_cmp);
            result.server_median_ms = Some(server_ms[server_ms.len() / 2]);
            result
        })
        .collect();

    assert!(
        guard.0.try_wait().is_ok_and(|s| s.is_none()),
        "furnace server exited during the benchmark"
    );
    let mem = memory::of_process(pid);
    let binary_mb = std::fs::metadata(&server).map_or(0.0, |m| m.len() as f64 / 1e6);
    Report {
        engine: "furnace server (HTTP, binary)".to_string(),
        version: "furnace 0.4.0, Burn 0.22".to_string(),
        threads: "default".to_string(),
        load_ms,
        first_inference_ms,
        top1,
        top1_logit: logits[top1],
        batches,
        peak_working_set_mb: mem.peak_working_set_mb,
        peak_private_mb: mem.peak_private_mb,
        binary_mb,
        notes: vec![
            "load = spawn to first healthy /healthz".to_string(),
            "memory and binary are the server process".to_string(),
        ],
    }
    .print();
}
