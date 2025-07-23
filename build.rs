#[cfg(feature = "burn-import")]
use burn_import::onnx::ModelGen;
use std::fs;
use std::path::Path;

fn main() {
    // Generate Rust code from ONNX models following Burn documentation
    #[cfg(feature = "burn-import")]
    {
        generate_onnx_models();
    }

    #[cfg(not(feature = "burn-import"))]
    {
        eprintln!("burn-import feature not enabled, skipping ONNX model generation");
    }

    // Rerun if any ONNX files change
    println!("cargo:rerun-if-changed=models/");

    // Declare possible cfg flags
    println!("cargo:rustc-check-cfg=cfg(model_resnet18)");
    println!("cargo:rustc-check-cfg=cfg(model_gptneox_opset18)");
}

#[cfg(feature = "burn-import")]
fn generate_onnx_models() {
    eprintln!("Generating ONNX models following Burn documentation");

    let models_dir = Path::new("models");
    if !models_dir.exists() {
        eprintln!("Models directory not found, skipping ONNX generation");
        return;
    }

    // Find all ONNX files in the models directory
    let onnx_files = match fs::read_dir(models_dir) {
        Ok(entries) => entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()?.to_str()? == "onnx" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("Failed to read models directory: {e}");
            return;
        }
    };

    if onnx_files.is_empty() {
        eprintln!("No ONNX files found in models directory");
        return;
    }

    // Generate Rust code for each ONNX model
    for onnx_path in onnx_files {
        let model_name = onnx_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        eprintln!("Generating model: {model_name}");
        println!("cargo:rerun-if-changed={}", onnx_path.display());

        // Try to generate the model, but don't fail the entire build if one model fails
        match std::panic::catch_unwind(|| {
            let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
            let models_out_dir = format!("{out_dir}/models");
            std::fs::create_dir_all(&models_out_dir)
                .expect("Failed to create models output directory");

            ModelGen::new()
                .input(onnx_path.to_str().unwrap())
                .out_dir(&models_out_dir)
                .run_from_script();
        }) {
            Ok(_) => {
                eprintln!("✅ Model '{model_name}' generated successfully");
                // Tell Cargo that this model was successfully generated
                println!("cargo:rustc-cfg=model_{}", model_name.replace("-", "_"));
            }
            Err(_) => {
                eprintln!("❌ Failed to generate model '{model_name}' - incompatible ONNX format");
                eprintln!("   This model will be skipped. Consider simplifying the ONNX file.");
            }
        }
    }
}
