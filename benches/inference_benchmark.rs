use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use furnace::model::{load_model, Model};
use std::path::PathBuf;
use std::sync::Arc;

fn create_test_model() -> Model {
    // Try to load sample model first, fallback to test model
    let sample_path = PathBuf::from("sample_model");
    if sample_path.with_extension("mpk").exists() {
        load_model(&sample_path).expect("Failed to load sample model")
    } else {
        let test_path = PathBuf::from("test_model.burn");
        load_model(&test_path).expect("Failed to load test model")
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

criterion_group!(
    benches,
    bench_single_inference,
    bench_batch_inference,
    bench_concurrent_inference
);
criterion_main!(benches);
