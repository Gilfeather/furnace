use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use furnace::model::{load_model, load_model_with_config, Model};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn create_test_model() -> Model {
    // Try to load sample model first, fallback to test model
    let sample_path = PathBuf::from("sample_model");
    if sample_path.with_extension("mpk").exists() {
        load_model(&sample_path).expect("Failed to load sample model")
    } else {
        let test_path = PathBuf::from("test_model.mpk");
        load_model(&test_path).expect("Failed to load test model")
    }
}

fn create_optimized_model(backend: Option<&str>, kernel_fusion: bool, autotuning: bool) -> Model {
    let config = furnace::model::ModelConfig {
        backend: backend.map(|s| s.to_string()),
        enable_kernel_fusion: kernel_fusion,
        enable_autotuning: autotuning,
    };

    let sample_path = PathBuf::from("sample_model");
    if sample_path.with_extension("mpk").exists() {
        load_model_with_config(&sample_path, config).expect("Failed to load optimized sample model")
    } else {
        let test_path = PathBuf::from("test_model.mpk");
        load_model_with_config(&test_path, config).expect("Failed to load optimized test model")
    }
}

fn bench_single_inference(c: &mut Criterion) {
    let model = Arc::new(create_test_model());
    let input = vec![0.5f32; 784]; // MNIST-like input

    c.bench_function("single_inference", |b| {
        b.iter(|| {
            let result = model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });
}

fn bench_batch_inference(c: &mut Criterion) {
    let model = Arc::new(create_test_model());
    let single_input = vec![0.5f32; 784];

    let mut group = c.benchmark_group("batch_inference");

    for batch_size in [1, 2, 4, 8, 16, 32].iter() {
        let inputs: Vec<Vec<f32>> = (0..*batch_size).map(|_| single_input.clone()).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let result = model.predict_batch(black_box(inputs.clone()));
                    black_box(result).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_concurrent_inference(c: &mut Criterion) {
    let model = Arc::new(create_test_model());
    let input = vec![0.5f32; 784];

    c.bench_function("concurrent_inference", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let model = model.clone();
                    let input = input.clone();
                    std::thread::spawn(move || model.predict_batch(vec![input]).unwrap())
                })
                .collect();

            for handle in handles {
                black_box(handle.join().unwrap());
            }
        })
    });
}

fn bench_optimization_comparison(c: &mut Criterion) {
    let input = vec![0.5f32; 784];
    let mut group = c.benchmark_group("optimization_comparison");

    // Baseline: no optimizations
    let baseline_model = Arc::new(create_optimized_model(None, false, false));
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let result = baseline_model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });

    // Kernel fusion enabled
    let fusion_model = Arc::new(create_optimized_model(None, true, false));
    group.bench_function("kernel_fusion", |b| {
        b.iter(|| {
            let result = fusion_model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });

    // Autotuning enabled
    let autotuning_model = Arc::new(create_optimized_model(None, false, true));
    group.bench_function("autotuning", |b| {
        b.iter(|| {
            let result = autotuning_model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });

    // Both optimizations enabled
    let optimized_model = Arc::new(create_optimized_model(None, true, true));
    group.bench_function("full_optimization", |b| {
        b.iter(|| {
            let result = optimized_model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });

    group.finish();
}

fn bench_backend_comparison(c: &mut Criterion) {
    let input = vec![0.5f32; 784];
    let mut group = c.benchmark_group("backend_comparison");

    // CPU backend
    let cpu_model = Arc::new(create_optimized_model(Some("cpu"), true, true));
    group.bench_function("cpu_backend", |b| {
        b.iter(|| {
            let result = cpu_model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });

    // Note: GPU backends would be tested here if available
    // For now, they'll fallback to CPU but still test the code path
    let gpu_model = Arc::new(create_optimized_model(Some("wgpu"), true, true));
    group.bench_function("gpu_backend_fallback", |b| {
        b.iter(|| {
            let result = gpu_model.predict_batch(vec![black_box(input.clone())]);
            black_box(result).unwrap()
        })
    });

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let model = Arc::new(create_test_model());
    let mut group = c.benchmark_group("memory_efficiency");

    // Test different batch sizes with correct input size (784 for MNIST-like model)
    let input_size = 784; // Fixed input size for the model
    for batch_size in [1, 4, 8, 16, 32].iter() {
        let inputs: Vec<Vec<f32>> = (0..*batch_size).map(|_| vec![0.5f32; input_size]).collect();

        group.throughput(Throughput::Bytes(
            *batch_size as u64 * input_size as u64 * 4,
        )); // 4 bytes per f32
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let result = model.predict_batch(black_box(inputs.clone()));
                    black_box(result).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_latency_percentiles(c: &mut Criterion) {
    let model = Arc::new(create_test_model());
    let input = vec![0.5f32; 784];

    c.bench_function("latency_measurement", |b| {
        b.iter_custom(|iters| {
            let mut times = Vec::with_capacity(iters as usize);

            for _ in 0..iters {
                let start = Instant::now();
                let result = model.predict_batch(vec![black_box(input.clone())]);
                let duration = start.elapsed();
                black_box(result).unwrap();
                times.push(duration);
            }

            // Calculate statistics
            times.sort();
            let total: std::time::Duration = times.iter().sum();

            // Log percentiles for analysis
            if !times.is_empty() {
                let p50 = times[times.len() / 2];
                let p95 = times[(times.len() * 95) / 100];
                let p99 = times[(times.len() * 99) / 100];

                eprintln!("Latency stats - P50: {p50:?}, P95: {p95:?}, P99: {p99:?}");
            }

            total
        });
    });
}

fn bench_throughput_scaling(c: &mut Criterion) {
    let model = Arc::new(create_test_model());
    let single_input = vec![0.5f32; 784];
    let mut group = c.benchmark_group("throughput_scaling");

    // Test throughput with increasing concurrent requests
    for concurrency in [1, 2, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrency", concurrency),
            concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let handles: Vec<_> = (0..concurrency)
                        .map(|_| {
                            let model = model.clone();
                            let input = single_input.clone();
                            std::thread::spawn(move || model.predict_batch(vec![input]).unwrap())
                        })
                        .collect();

                    for handle in handles {
                        black_box(handle.join().unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_inference,
    bench_batch_inference,
    bench_concurrent_inference,
    bench_optimization_comparison,
    bench_backend_comparison,
    bench_memory_efficiency,
    bench_latency_percentiles,
    bench_throughput_scaling
);
criterion_main!(benches);
