#[cfg(feature = "onnx")]
use burn_onnx::{LoadStrategy, ModelGen};
#[cfg(feature = "onnx")]
use std::fs;
#[cfg(feature = "onnx")]
use std::path::Path;

fn main() {
    // Rerun if any ONNX files change
    println!("cargo:rerun-if-changed=models/");

    // Declare possible cfg flags
    println!("cargo:rustc-check-cfg=cfg(model_resnet18)");

    // Generate Rust code from ONNX models following the burn-onnx documentation
    #[cfg(feature = "onnx")]
    generate_onnx_models();

    #[cfg(not(feature = "onnx"))]
    println!("cargo:warning=onnx feature not enabled, skipping ONNX model generation");
}

#[cfg(feature = "onnx")]
fn generate_onnx_models() {
    let models_dir = Path::new("models");
    let onnx_files = match fs::read_dir(models_dir) {
        Ok(entries) => entries
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension()?.to_str()? == "onnx").then_some(path)
            })
            .collect::<Vec<_>>(),
        Err(_) => {
            println!(
                "cargo:warning=models/ directory not found, no built-in models will be available"
            );
            return;
        }
    };

    // Keep weights out of the binary unless explicitly requested
    let embed = std::env::var_os("CARGO_FEATURE_EMBED_WEIGHTS").is_some();

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let models_out_dir = format!("{out_dir}/models");
    fs::create_dir_all(&models_out_dir).expect("Failed to create models output directory");

    for onnx_path in onnx_files {
        let model_name = onnx_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        println!("cargo:rerun-if-changed={}", onnx_path.display());

        // Don't fail the entire build if one model can't be converted
        let input = onnx_path.to_str().unwrap().to_string();
        let out = models_out_dir.clone();
        let result = std::panic::catch_unwind(move || {
            ModelGen::new()
                .input(&input)
                .out_dir(&out)
                .load_strategy(if embed {
                    LoadStrategy::Embedded
                } else {
                    LoadStrategy::File
                })
                .run_from_script();
        });

        match result {
            Ok(()) => println!("cargo:rustc-cfg=model_{}", model_name.replace('-', "_")),
            Err(_) => println!(
                "cargo:warning=Failed to generate model '{model_name}' (unsupported ONNX graph), skipping"
            ),
        }
    }
}
