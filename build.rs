fn main() {
    // Check if burn-import feature is enabled
    let burn_import_enabled = cfg!(feature = "burn-import");

    if burn_import_enabled {
        // Only try to use burn_import if the feature is enabled
        #[cfg(feature = "burn-import")]
        generate_onnx_models();
    } else {
        println!("cargo:warning=burn-import feature not enabled, skipping ONNX model generation");
    }

    // Rerun if any ONNX files change
    println!("cargo:rerun-if-changed=models/");
}

#[cfg(feature = "burn-import")]
fn generate_onnx_models() {
    use burn_import::onnx::ModelGen;
    use std::env;
    use std::path::Path;

    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:warning=OUT_DIR: {}", out_dir);

    // Generate code for existing ONNX models in the models directory
    let models_path = Path::new("models");
    if models_path.exists() {
        println!("cargo:warning=Found models directory");

        for entry in std::fs::read_dir(models_path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                let model_name = path.file_stem().unwrap().to_str().unwrap();

                println!("cargo:rerun-if-changed={}", path.display());
                println!(
                    "cargo:warning=Attempting to generate Rust code for ONNX model: {}",
                    model_name
                );

                // Skip problematic models for now
                if model_name.contains("gptneox") {
                    println!(
                        "cargo:warning=Skipping {} due to known compatibility issues",
                        model_name
                    );
                    continue;
                }

                // Generate the model code with error handling
                match std::panic::catch_unwind(|| {
                    ModelGen::new()
                        .input(path.to_str().unwrap())
                        .out_dir("models/")
                        .run_from_script();
                }) {
                    Ok(_) => {
                        println!(
                            "cargo:warning=Successfully generated code for {}",
                            model_name
                        );
                    }
                    Err(_) => {
                        println!(
                            "cargo:warning=Failed to generate code for {} - skipping",
                            model_name
                        );
                    }
                }
            }
        }
    } else {
        println!("cargo:warning=Models directory not found");
    }
}
