//! tract (pure Rust): optimized ONNX plan with a fixed input shape per batch size.
//!
//! tract runs single-threaded by default; `--threads N` (N > 1) installs its rayon
//! executor for matrix products.

use std::path::Path;
use std::sync::Arc;

use furnace_compare::{Args, CHANNELS, SIDE, bench_in_process};
use tract_onnx::prelude::*;

type Plan = Arc<TypedRunnableModel>;

fn load(model: &Path, batch: usize) -> Plan {
    tract_onnx::onnx()
        .model_for_path(model)
        .expect("read onnx")
        .with_input_fact(0, f32::fact([batch, CHANNELS, SIDE, SIDE]).into())
        .expect("input fact")
        .into_optimized()
        .expect("optimize")
        .into_runnable()
        .expect("runnable")
}

fn main() {
    let args = Args::parse("unused");
    if args.threads > 1 {
        tract_linalg::multithread::set_default_executor(
            tract_linalg::multithread::Executor::multithread(args.threads),
        );
    }
    let threads = if args.threads > 1 {
        args.threads.to_string()
    } else {
        "1".to_string()
    };

    // tract specializes the plan for one input shape, so keep one plan per batch size.
    // Load time below is for the batch-1 plan; other plans are built outside the timer.
    let model_path = args.model.clone();
    let extra_batches: Vec<usize> = args.batches.iter().copied().filter(|&b| b != 1).collect();
    let mut report = bench_in_process(
        &args,
        "tract",
        "tract-onnx 0.23.8",
        threads,
        || vec![(1, load(&model_path, 1))],
        |plans, input, batch| {
            if !plans.iter().any(|(b, _)| *b == batch) {
                assert!(extra_batches.contains(&batch));
                plans.push((batch, load(&model_path, batch)));
            }
            let plan = &plans.iter().find(|(b, _)| *b == batch).unwrap().1;
            let tensor = Tensor::from_shape(&[batch, CHANNELS, SIDE, SIDE], input).expect("tensor");
            let out = plan.run(tvec!(tensor.into())).expect("tract inference");
            out[0]
                .try_as_plain_ram()
                .and_then(|v| v.as_slice::<f32>().map(<[f32]>::to_vec))
                .expect("f32 output")
        },
    );
    report
        .notes
        .push("one optimized plan per batch size".to_string());
    report.print();
}
