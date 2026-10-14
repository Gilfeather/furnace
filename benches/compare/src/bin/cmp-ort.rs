//! ONNX Runtime through the `ort` crate. Not pure Rust: ORT is a C++ library (MLAS
//! kernels), downloaded prebuilt by `ort`'s build script.
//!
//! On Windows the prebuilt runtime is the DirectML build: it loads `DirectML.dll` even on
//! the CPU path, so that file counts towards the binary size.
//!
//! `--threads 0` (default) lets ORT pick its intra-op thread count (physical cores).

use furnace_compare::{Args, CHANNELS, SIDE, bench_in_process, binary_size_mb};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

fn main() {
    let args = Args::parse("unused");
    let threads = args.threads;
    let model_path = args.model.clone();

    let mut report = bench_in_process(
        &args,
        "ONNX Runtime (ort)",
        "ort 2.0.0-rc.14, ONNX Runtime 1.31",
        if threads == 0 {
            "default".to_string()
        } else {
            threads.to_string()
        },
        || {
            let env = ort::init().build().expect("ort environment");
            let mut builder = Session::builder(&env)
                .expect("session builder")
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .expect("opt level");
            if threads > 0 {
                builder = builder.with_intra_threads(threads).expect("threads");
            }
            builder.commit_from_file(&model_path).expect("load onnx")
        },
        |session, input, batch| {
            let tensor =
                TensorRef::from_array_view(([batch, CHANNELS, SIDE, SIDE], input)).expect("tensor");
            let outputs = session.run(ort::inputs![tensor]).expect("ort inference");
            let (_, logits) = outputs[0].try_extract_tensor::<f32>().expect("f32 output");
            logits.to_vec()
        },
    );
    report.binary_mb = binary_size_mb(&["onnxruntime", "directml"]);
    report.print();
}
