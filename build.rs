// Generates Rust code from every ONNX file in models/ (burn-onnx) and a registry of the
// generated models that src/models/mod.rs includes. Dropping `models/foo.onnx` in and
// rebuilding is all it takes to serve it with `--model-name foo`.

#[path = "build/registry.rs"]
#[allow(dead_code)]
mod registry;

#[cfg(feature = "onnx")]
#[path = "build/optimize.rs"]
mod optimize;

use std::fs;
use std::path::Path;

// Built-in models the source refers to (tests, examples); every generated model also gets
// `cfg(model_<module>)`, declared as it is emitted.
const KNOWN_MODEL_CFGS: &[&str] = &["resnet18", "mnist"];

fn main() {
    println!("cargo:rerun-if-changed=models/");
    println!("cargo:rerun-if-changed=build/registry.rs");
    println!("cargo:rerun-if-changed=build/optimize.rs");
    println!("cargo:rerun-if-env-changed=FURNACE_NO_GRAPH_OPT");
    for name in KNOWN_MODEL_CFGS {
        println!("cargo:rustc-check-cfg=cfg(model_{name})");
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let models_out_dir = Path::new(&out_dir).join("models");
    fs::create_dir_all(&models_out_dir).expect("Failed to create models output directory");

    #[cfg(feature = "onnx")]
    let specs = onnx::generate_models(&models_out_dir);
    #[cfg(not(feature = "onnx"))]
    let specs = {
        println!("cargo:warning=onnx feature not enabled, skipping ONNX model generation");
        Vec::<registry::ModelSpec>::new()
    };

    for spec in &specs {
        println!("cargo:rustc-check-cfg=cfg(model_{})", spec.module);
        println!("cargo:rustc-cfg=model_{}", spec.module);
    }
    fs::write(
        models_out_dir.join("registry.rs"),
        registry::render_registry(&specs),
    )
    .expect("Failed to write model registry");
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::registry::{self, Io, IoType, ModelSpec};
    use burn_onnx::{LoadStrategy, ModelGen};
    use onnx_ir::{ArgType, DType, Message, OnnxGraphBuilder};
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Generate code for every servable model in models/ and return their specs, sorted by name
    pub fn generate_models(out_dir: &Path) -> Vec<ModelSpec> {
        let mut onnx_files = match fs::read_dir("models") {
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
                return Vec::new();
            }
        };
        onnx_files.sort_by(|a, b| a.file_stem().cmp(&b.file_stem()));

        // Keep weights out of the binary unless explicitly requested
        let embed = std::env::var_os("CARGO_FEATURE_EMBED_WEIGHTS").is_some();

        let mut specs = Vec::new();
        for onnx_path in onnx_files {
            println!("cargo:rerun-if-changed={}", onnx_path.display());
            let Some(name) = onnx_path.file_stem().and_then(|s| s.to_str()) else {
                println!(
                    "cargo:warning=skipping {}: file name is not valid UTF-8",
                    onnx_path.display()
                );
                continue;
            };
            match generate_model(&onnx_path, name, out_dir, embed) {
                Ok(spec) => specs.push(spec),
                Err(reason) => println!("cargo:warning=skipping model '{name}': {reason}"),
            }
        }

        let (specs, warnings) = registry::dedup(specs);
        for warning in warnings {
            println!("cargo:warning={warning}");
        }
        specs
    }

    fn generate_model(
        onnx_path: &Path,
        name: &str,
        out_dir: &Path,
        embed: bool,
    ) -> Result<ModelSpec, String> {
        let onnx_path = &optimized_graph(onnx_path, name, out_dir)?;

        // Check the graph's I/O first, so unsupported models are skipped before codegen
        let graph = OnnxGraphBuilder::new()
            .parse_file(onnx_path)
            .map_err(|e| format!("could not parse the ONNX graph: {e}"))?;
        let spec = registry::model_spec(
            name,
            &graph.inputs.iter().map(io).collect::<Vec<_>>(),
            &graph.outputs.iter().map(io).collect::<Vec<_>>(),
        )?;
        drop(graph);

        // burn-onnx panics on graphs it can't convert; don't fail the whole build over one model
        let input: PathBuf = onnx_path.to_path_buf();
        let out = out_dir
            .to_str()
            .ok_or("output path is not valid UTF-8")?
            .to_string();
        std::panic::catch_unwind(move || {
            ModelGen::new()
                .input(input.to_str().unwrap())
                .out_dir(&out)
                .load_strategy(if embed {
                    LoadStrategy::Embedded
                } else {
                    LoadStrategy::File
                })
                .run_from_script();
        })
        .map_err(|_| "burn-onnx could not convert the graph (unsupported operator?)".to_string())?;
        Ok(spec)
    }

    /// Apply build-time graph optimizations and return the ONNX file to generate code from.
    /// Set `FURNACE_NO_GRAPH_OPT=1` to generate from the original graph.
    fn optimized_graph(onnx_path: &Path, name: &str, out_dir: &Path) -> Result<PathBuf, String> {
        if std::env::var_os("FURNACE_NO_GRAPH_OPT").is_some() {
            return Ok(onnx_path.to_path_buf());
        }
        let bytes = fs::read(onnx_path).map_err(|e| format!("could not read the file: {e}"))?;
        let mut model = onnx_ir::ModelProto::parse_from_bytes(&bytes)
            .map_err(|e| format!("could not parse the ONNX graph: {e}"))?;
        let report = super::optimize::fold_conv_batchnorm(&mut model);
        if report.folded == 0 {
            return Ok(onnx_path.to_path_buf());
        }
        // `cargo build -vv` shows these; skips are expected for BN not preceded by Conv
        eprintln!(
            "{name}: folded {} BatchNormalization node(s) into Conv, skipped {}",
            report.folded,
            report.skipped.len()
        );
        for (node, reason) in &report.skipped {
            eprintln!("  {name}: kept BatchNormalization '{node}': {reason}");
        }
        let dir = out_dir.join("optimized");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{name}.onnx"));
        let out = model
            .write_to_bytes()
            .map_err(|e| format!("could not serialize the optimized graph: {e}"))?;
        fs::write(&path, out).map_err(|e| e.to_string())?;
        Ok(path)
    }

    fn io(arg: &onnx_ir::Argument) -> Io {
        let ty = match &arg.ty {
            ArgType::Tensor(t) if t.dtype == DType::F32 => {
                IoType::F32Tensor(t.static_shape.clone().unwrap_or_else(|| vec![None; t.rank]))
            }
            ArgType::Tensor(t) => IoType::Other(format!("{:?} tensor", t.dtype)),
            ArgType::ScalarTensor(dtype) | ArgType::ScalarNative(dtype) => {
                IoType::Other(format!("{dtype:?} scalar"))
            }
            ArgType::Shape(_) => IoType::Other("shape".to_string()),
        };
        Io {
            name: arg.name.clone(),
            ty,
        }
    }
}
