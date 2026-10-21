//! furnace in process: `Model::predict_flat` on the generated ResNet-18.
//!
//! `--backend cpu` (default, burn-flex) or `--backend wgpu` (needs `--features wgpu`).

use furnace_burn::{BackendKind, load_built_in_model_with_backend};
use furnace_compare::{Args, bench_in_process, furnace_version};

// The same allocator as the furnace server binary
#[global_allocator]
static ALLOCATOR: kamui::BlockCache = kamui::BlockCache::new();

fn main() {
    let args = Args::parse("backend");
    let backend_name = args.extra.clone().unwrap_or_else(|| "cpu".to_string());
    let backend = BackendKind::from_name(&backend_name).expect("unknown backend");

    let mut report = bench_in_process(
        &args,
        &format!("furnace ({backend_name})"),
        &furnace_version(),
        "default".to_string(),
        // Default weight lookup finds resnet18.bpk in the furnace build output directory
        || load_built_in_model_with_backend("resnet18", backend, None).expect("load furnace"),
        |model, input, batch| {
            model
                .predict_flat(input.to_vec(), batch)
                .expect("furnace inference")
        },
    );
    report
        .notes
        .push("weights memory-mapped from resnet18.bpk".to_string());
    report.print();
}
