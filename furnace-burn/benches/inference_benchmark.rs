//! Criterion benchmarks.
//!
//! - `resnet18/*`: real ResNet-18 inference through `Model::predict_batch`
//!   (requires the `onnx` feature and `models/resnet18.onnx` at build time)
//! - `mlp/*`: file-format MLP (784 -> 128 -> 10)
//! - `overhead/*`: constant-output dummy model with ResNet-sized I/O, i.e. the cost of
//!   validation and tensor conversion alone, excluding the network itself
//! - `request_decode/*`: turning one ResNet-sized `/predict` body into floats, JSON vs raw
//!   little-endian `f32` (`application/octet-stream`)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use furnace::api::PredictRequest;
use furnace::model::{DummyModel, Model};
use furnace_burn::load_model;
use std::hint::black_box;
use std::path::Path;

const RESNET_INPUT: usize = 3 * 224 * 224;

fn dummy_resnet_shaped() -> Model {
    let dummy = DummyModel::new("dummy_resnet18".to_string(), vec![3, 224, 224], vec![1000]);
    Model::new(Box::new(dummy), "dummy", 0)
}

fn bench_model(c: &mut Criterion, group_name: &str, model: &Model, batch_sizes: &[usize]) {
    let input_size: usize = model.get_info().input_spec.shape.iter().product();
    let mut group = c.benchmark_group(group_name);
    for &batch_size in batch_sizes {
        let batch = vec![vec![0.5f32; input_size]; batch_size];
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch_size), &batch, |b, batch| {
            b.iter(|| black_box(model.predict_batch(black_box(batch.clone())).unwrap()))
        });
    }
    group.finish();
}

fn bench_resnet18(c: &mut Criterion) {
    let Ok(model) = furnace_burn::load_built_in_model("resnet18") else {
        eprintln!("resnet18 not compiled in; skipping (place models/resnet18.onnx and rebuild)");
        return;
    };
    // Warm up allocations / lazy initialization outside the measurement
    model.predict(vec![0.5; RESNET_INPUT]).unwrap();
    bench_model(c, "resnet18", &model, &[1, 4]);
}

fn bench_mlp(c: &mut Criterion) {
    let model = load_model(Path::new("sample_model")).expect("sample MLP");
    bench_model(c, "mlp", &model, &[1, 32]);
}

fn bench_overhead(c: &mut Criterion) {
    bench_model(c, "overhead", &dummy_resnet_shaped(), &[1, 4]);
}

fn bench_request_decode(c: &mut Criterion) {
    // Deterministic, varied values: JSON size and parse cost depend on the digits
    let image: Vec<f32> = (0..RESNET_INPUT)
        .map(|i| ((i * 7919) % 10007) as f32 / 2501.0 - 2.0)
        .collect();
    let json = serde_json::to_vec(&serde_json::json!({ "input": image })).unwrap();
    let binary: Vec<u8> = image.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut group = c.benchmark_group("request_decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("json", |b| {
        b.iter(|| black_box(serde_json::from_slice::<PredictRequest>(black_box(&json)).unwrap()))
    });
    group.bench_function("binary", |b| {
        b.iter(|| {
            let (chunks, _) = black_box(binary.as_slice()).as_chunks::<4>();
            black_box(
                chunks
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect::<Vec<f32>>(),
            )
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_request_decode, bench_overhead, bench_mlp, bench_resnet18
}
criterion_main!(benches);
