//! Shared pieces of the engine comparison: image preprocessing, timing, memory and the
//! JSON report every engine binary prints.
//!
//! Each engine lives in its own binary (`src/bin/cmp-*.rs`) so that startup, memory and
//! binary size are measured for that engine alone. `compare` runs them all and prints a
//! Markdown table.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub const CHANNELS: usize = 3;
pub const SIDE: usize = 224;
pub const INPUT_LEN: usize = CHANNELS * SIDE * SIDE;
pub const CLASSES: usize = 1000;
/// ImageNet class 258 ("Samoyed"), the expected top-1 for PyTorch Hub's `dog.jpg`
pub const EXPECTED_TOP1: usize = 258;
/// What the furnace rows run: the furnace checkout this harness sits in
pub const FURNACE_VERSION: &str = "furnace 0.6.1, Burn 0.22";

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Command-line options shared by all engine binaries
#[derive(Debug, Clone)]
pub struct Args {
    pub model: PathBuf,
    pub image: PathBuf,
    pub warmup: usize,
    pub iters: usize,
    pub batches: Vec<usize>,
    pub threads: usize,
    /// Engine-specific extra option (`--backend`, `--server`)
    pub extra: Option<String>,
}

impl Args {
    /// Parse `--model --image --warmup --iters --batches 1,4 --threads N` plus one engine
    /// specific `--<extra_name> VALUE`.
    pub fn parse(extra_name: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut args = Args {
            model: root.join("../../models/resnet18.onnx"),
            image: root.join("dog.jpg"),
            warmup: 5,
            iters: 30,
            batches: vec![1, 4],
            threads: 0,
            extra: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = || {
                it.next()
                    .unwrap_or_else(|| panic!("missing value for {flag}"))
            };
            match flag.as_str() {
                "--model" => args.model = value().into(),
                "--image" => args.image = value().into(),
                "--warmup" => args.warmup = value().parse().expect("--warmup"),
                "--iters" => args.iters = value().parse().expect("--iters"),
                "--threads" => args.threads = value().parse().expect("--threads"),
                "--batches" => {
                    args.batches = value()
                        .split(',')
                        .map(|b| b.trim().parse().expect("--batches"))
                        .collect()
                }
                f if f.strip_prefix("--") == Some(extra_name) => args.extra = Some(value()),
                other => panic!("unknown argument {other}"),
            }
        }
        args
    }
}

/// Decode an image and turn it into a normalized CHW `f32` tensor, torchvision style:
/// resize the shorter side to 256 (bilinear), center-crop 224, scale to [0, 1], then
/// normalize with the ImageNet mean and standard deviation.
pub fn preprocess(path: &Path) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let scale = 256.0 / w.min(h) as f32;
    let (nw, nh) = (
        (w as f32 * scale).round() as u32,
        (h as f32 * scale).round() as u32,
    );
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    let (left, top) = ((nw - SIDE as u32) / 2, (nh - SIDE as u32) / 2);

    let mut out = vec![0.0f32; INPUT_LEN];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let px = resized.get_pixel(left + x as u32, top + y as u32);
            for c in 0..CHANNELS {
                let v = px[c] as f32 / 255.0;
                out[c * SIDE * SIDE + y * SIDE + x] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    out
}

/// The same image repeated `batch` times, back to back
pub fn batch_input(image: &[f32], batch: usize) -> Vec<f32> {
    image.repeat(batch)
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

/// Latency figures for one batch size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub batch: usize,
    pub iters: usize,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub min_ms: f64,
    /// Images per second at the median latency, one request at a time
    pub images_per_s: f64,
    /// Median server-reported inference time (HTTP only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_median_ms: Option<f64>,
}

impl BatchResult {
    pub fn from_samples(batch: usize, samples: &mut [Duration]) -> Self {
        samples.sort();
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let median_ms = ms(percentile(samples, 0.50));
        BatchResult {
            batch,
            iters: samples.len(),
            median_ms,
            p95_ms: ms(percentile(samples, 0.95)),
            min_ms: ms(samples[0]),
            images_per_s: batch as f64 * 1000.0 / median_ms,
            server_median_ms: None,
        }
    }
}

/// Nearest-rank percentile of sorted samples
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// Run `f` `warmup` times untimed, then `iters` times timed
pub fn time_iters(warmup: usize, iters: usize, mut f: impl FnMut()) -> Vec<Duration> {
    for _ in 0..warmup {
        f();
    }
    (0..iters)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect()
}

/// What each engine binary prints as one JSON line on stdout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub engine: String,
    pub version: String,
    /// Intra-op threads, as configured ("default" means the engine decides)
    pub threads: String,
    /// Reading the model and making it ready to run, without the first inference
    pub load_ms: f64,
    /// First batch-1 inference after loading (includes any lazy initialization)
    pub first_inference_ms: f64,
    pub top1: usize,
    pub top1_logit: f32,
    pub batches: Vec<BatchResult>,
    /// Peak working set (resident memory) of the process doing inference
    pub peak_working_set_mb: f64,
    /// Peak private (non-shared, committed) memory of that process
    pub peak_private_mb: f64,
    /// Size of the executable plus any runtime libraries it loads from its directory
    pub binary_mb: f64,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Report {
    pub fn print(&self) {
        println!("{}", serde_json::to_string(self).expect("serialize report"));
    }
}

/// A full in-process benchmark: load, first inference + accuracy check, then timed runs
/// for each batch size. `run` takes a flat input of `batch` images and returns flat logits.
pub fn bench_in_process<E>(
    args: &Args,
    engine: &str,
    version: &str,
    threads: String,
    load: impl FnOnce() -> E,
    mut run: impl FnMut(&mut E, &[f32], usize) -> Vec<f32>,
) -> Report {
    let image = preprocess(&args.image);

    let start = Instant::now();
    let mut model = load();
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let logits = run(&mut model, &image, 1);
    let first_inference_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(logits.len(), CLASSES, "{engine}: unexpected output length");
    let top1 = argmax(&logits);

    let batches = args
        .batches
        .iter()
        .map(|&batch| {
            let input = batch_input(&image, batch);
            let mut samples = time_iters(args.warmup, args.iters, || {
                let out = run(&mut model, &input, batch);
                assert_eq!(out.len(), batch * CLASSES);
                std::hint::black_box(out);
            });
            BatchResult::from_samples(batch, &mut samples)
        })
        .collect();

    let mem = memory::current_process();
    Report {
        engine: engine.to_string(),
        version: version.to_string(),
        threads,
        load_ms,
        first_inference_ms,
        top1,
        top1_logit: logits[top1],
        batches,
        peak_working_set_mb: mem.peak_working_set_mb,
        peak_private_mb: mem.peak_private_mb,
        binary_mb: binary_size_mb(&[]),
        notes: Vec::new(),
    }
}

/// A full benchmark of an inference server over HTTP on localhost: start it, wait for
/// `/healthz`, then time `POST /predict` with furnace's binary protocol (little-endian f32
/// body, batch size in `X-Batch-Size`, inference time in the `X-Inference-Time-Ms` answer
/// header) on one keep-alive connection, one request at a time.
///
/// `command` builds the server command for a free port. Load is the time from spawning it
/// until `/healthz` answers. Memory is that process's, or the process whose `pid` the
/// `/healthz` JSON names, if any. The caller fills in `version`,
/// `binary_mb` and anything else engine specific.
pub fn bench_http_server(
    args: &Args,
    engine: &str,
    command: impl FnOnce(u16) -> Command,
) -> Report {
    let image = preprocess(&args.image);

    // A port nobody listens on, so /healthz can only be answered by our own server
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("free port")
        .port();

    let start = Instant::now();
    let mut child = command(port)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot start {engine}: {e}"));
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
    let health = loop {
        // Windows retries a refused SYN for ~0.5 s, so probe the port with a short
        // connect timeout before asking /healthz, or startup is overstated
        let listening =
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(10)).is_ok();
        if listening && let Ok(mut resp) = agent.get(format!("{base}/healthz")).call() {
            break resp.body_mut().read_to_string().unwrap_or_default();
        }
        if let Some(status) = child.try_wait().expect("try_wait") {
            panic!("{engine} exited during startup: {status}");
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "{engine} did not start"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    // A server started through a launcher reports the pid that does the work
    let pid = serde_json::from_str::<serde_json::Value>(&health)
        .ok()
        .and_then(|h| h.get("pid")?.as_u64())
        .and_then(|p| u32::try_from(p).ok())
        .unwrap_or_else(|| child.id());
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
    assert_eq!(logits.len(), CLASSES, "{engine}: unexpected output length");
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
        "{engine} exited during the benchmark"
    );
    let mem = memory::of_process(pid);
    Report {
        engine: engine.to_string(),
        version: String::new(),
        threads: "default".to_string(),
        load_ms,
        first_inference_ms,
        top1,
        top1_logit: logits[top1],
        batches,
        peak_working_set_mb: mem.peak_working_set_mb,
        peak_private_mb: mem.peak_private_mb,
        binary_mb: 0.0,
        notes: vec!["load = spawn to first healthy /healthz".to_string()],
    }
}

/// Total size of the files under `dir`, in MB (10^6 bytes)
pub fn dir_size_mb(dir: &Path) -> f64 {
    fn walk(dir: &Path) -> u64 {
        std::fs::read_dir(dir).map_or(0, |entries| {
            entries
                .flatten()
                .map(|e| match e.file_type() {
                    Ok(t) if t.is_dir() => walk(&e.path()),
                    Ok(_) => e.metadata().map_or(0, |m| m.len()),
                    Err(_) => 0,
                })
                .sum()
        })
    }
    walk(dir) as f64 / 1e6
}

/// Size of the running executable plus files next to it whose names contain any of
/// `libs` (e.g. `onnxruntime` for a dynamically loaded runtime)
pub fn binary_size_mb(libs: &[&str]) -> f64 {
    let exe = std::env::current_exe().expect("current_exe");
    let mut bytes = std::fs::metadata(&exe).map_or(0, |m| m.len());
    if !libs.is_empty()
        && let Some(dir) = exe.parent()
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            let is_lib = [".dll", ".so", ".dylib"].iter().any(|e| name.contains(e));
            if is_lib && libs.iter().any(|l| name.contains(l)) {
                bytes += entry.metadata().map_or(0, |m| m.len());
            }
        }
    }
    bytes as f64 / 1e6
}

pub mod memory {
    /// Peak memory of a process, in MB (10^6 bytes)
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Memory {
        pub peak_working_set_mb: f64,
        pub peak_private_mb: f64,
    }

    pub fn current_process() -> Memory {
        of_process(std::process::id())
    }

    #[cfg(windows)]
    pub fn of_process(pid: u32) -> Memory {
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
                return Memory::default();
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
                return Memory::default();
            }
            Memory {
                peak_working_set_mb: counters.PeakWorkingSetSize as f64 / 1e6,
                // Peak commit charge: memory private to this process
                peak_private_mb: counters.PeakPagefileUsage as f64 / 1e6,
            }
        }
    }

    #[cfg(target_os = "linux")]
    pub fn of_process(pid: u32) -> Memory {
        // VmHWM is the peak RSS. Linux has no peak private counter, so report the
        // current anonymous resident memory measured after the benchmark.
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        let kb = |key: &str| {
            status
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        Memory {
            peak_working_set_mb: kb("VmHWM:") * 1024.0 / 1e6,
            peak_private_mb: kb("RssAnon:") * 1024.0 / 1e6,
        }
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub fn of_process(_pid: u32) -> Memory {
        Memory::default()
    }
}
