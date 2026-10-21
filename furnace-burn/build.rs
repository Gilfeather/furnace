// Generates Rust code from every ONNX file in models/ (burn-onnx) and a registry of the
// generated models that src/models/mod.rs includes. Dropping `models/foo.onnx` in and
// rebuilding is all it takes to serve it with `--model-name foo`.

#[cfg(feature = "onnx")]
#[path = "build/render.rs"]
mod render;

#[cfg(feature = "onnx")]
#[path = "build/spec.rs"]
mod spec;

#[cfg(feature = "onnx")]
#[path = "build/optimize.rs"]
mod optimize;

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Where the ONNX models to compile in live: `$FURNACE_MODELS_DIR` if set (e.g. for
/// `cargo install`), otherwise `models/` next to this manifest.
fn models_dir() -> std::path::PathBuf {
    match std::env::var_os("FURNACE_MODELS_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("models"),
    }
}

/// The `model_<module>` cfgs the source refers to (tests written for one built-in model),
/// found by scanning `cfg(...)` and `cfg!(...)` in src/, tests/, examples/ and benches/*.rs.
///
/// The scan runs whenever this script does. The sources are not watched, since that would
/// rerun model generation on every edit, so a new `cfg(model_x)` is reported by rustc as an
/// unexpected cfg until the script next runs (`touch build.rs`).
fn referenced_model_cfgs() -> BTreeSet<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let benches = root.join("benches");
    let mut dirs: Vec<_> = ["src", "tests", "examples", "benches"]
        .iter()
        .map(|dir| root.join(dir))
        .collect();
    let mut found = BTreeSet::new();
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for path in entries.filter_map(|entry| Some(entry.ok()?.path())) {
            if path.is_dir() {
                // The harnesses in benches/ are crates of their own
                if dir != benches {
                    dirs.push(path);
                }
            } else if path.extension().is_some_and(|ext| ext == "rs")
                && let Ok(source) = fs::read_to_string(&path)
            {
                cfg_model_names(&source, &mut found);
            }
        }
    }
    found
}

/// Add the modules of the `model_*` names inside each `cfg(...)` or `cfg!(...)` of `source`
fn cfg_model_names(source: &str, found: &mut BTreeSet<String>) {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    for (start, _) in source.match_indices("cfg") {
        if source[..start].ends_with(is_ident) {
            continue;
        }
        let rest = &source[start + "cfg".len()..];
        let Some(rest) = rest.strip_prefix('!').unwrap_or(rest).strip_prefix('(') else {
            continue;
        };
        let mut depth = 1;
        let end = rest
            .char_indices()
            .find(|&(_, c)| {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                depth == 0
            })
            .map_or(rest.len(), |(i, _)| i);
        for word in rest[..end].split(|c: char| !is_ident(c)) {
            if let Some(module) = word.strip_prefix("model_")
                && !module.is_empty()
            {
                found.insert(module.to_string());
            }
        }
    }
}

fn main() {
    let models_dir = models_dir();
    // A path that does not exist counts as changed, which would rerun this script every build
    if models_dir.is_dir() {
        println!("cargo:rerun-if-changed={}", models_dir.display());
    }
    println!("cargo:rerun-if-env-changed=FURNACE_MODELS_DIR");
    println!("cargo:rerun-if-changed=build/render.rs");
    println!("cargo:rerun-if-changed=build/spec.rs");
    println!("cargo:rerun-if-changed=build/optimize.rs");
    println!("cargo:rerun-if-changed=src/models/name.rs");
    println!("cargo:rerun-if-env-changed=FURNACE_NO_GRAPH_OPT");
    for module in referenced_model_cfgs() {
        println!("cargo:rustc-check-cfg=cfg(model_{module})");
    }

    // Without `onnx` there are no built-in models (src/models/mod.rs then declares the empty
    // registry itself), as the library configuration intends
    #[cfg(feature = "onnx")]
    onnx::build_models(&models_dir);
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::render::{self, Elem, ModelSpec};
    use super::spec::{self, Io, IoType};
    use burn_onnx::{LoadStrategy, ModelGen};
    use onnx_ir::{ArgType, DType, Message, ModelProto, OnnxGraphBuilder};
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Generate code for every servable model in `models_dir` and the registry of them, in
    /// `$OUT_DIR/models`
    pub fn build_models(models_dir: &Path) {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_dir = Path::new(&out_dir).join("models");
        fs::create_dir_all(&out_dir).expect("Failed to create models output directory");
        let specs = generate_models(models_dir, &out_dir);
        for spec in &specs {
            println!("cargo:rustc-check-cfg=cfg(model_{})", spec.module);
            println!("cargo:rustc-cfg=model_{}", spec.module);
        }
        fs::write(out_dir.join("registry.rs"), render::render_registry(&specs))
            .expect("Failed to write model registry");
    }

    /// Generate code for every servable model in `models_dir` and return their specs, sorted
    /// by name
    fn generate_models(models_dir: &Path, out_dir: &Path) -> Vec<ModelSpec> {
        let mut onnx_files = match fs::read_dir(models_dir) {
            Ok(entries) => entries
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    (path.extension()?.to_str()? == "onnx").then_some(path)
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                // A checkout without models/ just has no built-in models; a directory named
                // explicitly should exist
                if std::env::var_os("FURNACE_MODELS_DIR").is_some() {
                    println!(
                        "cargo:warning=FURNACE_MODELS_DIR={}: {e}; no built-in models",
                        models_dir.display()
                    );
                }
                return Vec::new();
            }
        };
        onnx_files.sort_by(|a, b| a.file_stem().cmp(&b.file_stem()));

        // Keep weights out of the binary unless explicitly requested
        let embed = cfg!(feature = "embed-weights");

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

        let (specs, warnings) = spec::dedup(specs);
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
        // The protobuf is parsed here once, for the graph optimizations and the I/O names;
        // onnx-ir and burn-onnx then each parse the file to generate from
        let bytes = fs::read(onnx_path).map_err(|e| format!("could not read the file: {e}"))?;
        let mut model = ModelProto::parse_from_bytes(&bytes)
            .map_err(|e| format!("could not parse the ONNX graph: {e}"))?;
        drop(bytes);
        let onnx_path = &optimized_graph(onnx_path, &mut model, name, out_dir)?;
        // onnx-ir turns names into identifiers (`input_ids:0` -> `input_ids_0`); clients use
        // the graph's own, so serve those
        let (input_names, output_names) = graph_io_names(&model);
        drop(model);

        // Check the graph's I/O first, so unsupported models are skipped before codegen
        let graph = OnnxGraphBuilder::new()
            .parse_file(onnx_path)
            .map_err(|e| format!("could not parse the ONNX graph: {e}"))?;
        let mut inputs: Vec<Io> = graph.inputs.iter().map(io).collect();
        let mut outputs: Vec<Io> = graph.outputs.iter().map(io).collect();
        drop(graph);
        for (args, names) in [(&mut inputs, input_names), (&mut outputs, output_names)] {
            if args.len() == names.len() {
                for (arg, name) in args.iter_mut().zip(names) {
                    arg.name = name;
                }
            }
        }
        let spec = spec::model_spec(name, &inputs, &outputs)?;

        // burn-onnx panics on graphs it can't convert; don't fail the whole build over one model
        let input = onnx_path
            .to_str()
            .ok_or("model path is not valid UTF-8")?
            .to_string();
        let out = out_dir
            .to_str()
            .ok_or("output path is not valid UTF-8")?
            .to_string();
        std::panic::catch_unwind(move || {
            ModelGen::new()
                .input(&input)
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

    /// Apply build-time graph optimizations to `model` (`onnx_path`, parsed) and return the
    /// ONNX file to generate code from. Set `FURNACE_NO_GRAPH_OPT=1` to generate from the
    /// original graph.
    fn optimized_graph(
        onnx_path: &Path,
        model: &mut ModelProto,
        name: &str,
        out_dir: &Path,
    ) -> Result<PathBuf, String> {
        if std::env::var_os("FURNACE_NO_GRAPH_OPT").is_some() {
            return Ok(onnx_path.to_path_buf());
        }
        let report = super::optimize::fold_conv_batchnorm(model);
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

    /// The graph's input names (initializers left out, as onnx-ir does) and output names, in
    /// order, as written in the ONNX file
    fn graph_io_names(model: &ModelProto) -> (Vec<String>, Vec<String>) {
        let initializers: std::collections::HashSet<&str> = model
            .graph
            .initializer
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        let inputs = model
            .graph
            .input
            .iter()
            .filter(|i| !initializers.contains(i.name.as_str()))
            .map(|i| i.name.clone())
            .collect();
        let outputs = model.graph.output.iter().map(|o| o.name.clone()).collect();
        (inputs, outputs)
    }

    fn io(arg: &onnx_ir::Argument) -> Io {
        let ty = match &arg.ty {
            ArgType::Tensor(t) => {
                let dims = t.static_shape.clone().unwrap_or_else(|| vec![None; t.rank]);
                match t.dtype {
                    DType::F32 => IoType::Tensor(Elem::F32, dims),
                    DType::I64 => IoType::Tensor(Elem::I64, dims),
                    DType::I32 => IoType::Tensor(Elem::I32, dims),
                    other => IoType::Other(format!("{other:?} tensor")),
                }
            }
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
