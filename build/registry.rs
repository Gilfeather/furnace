// Turning ONNX graph I/O into the built-in model registry.
//
// Used by build.rs (which reads the graph with onnx-ir) and unit-tested from src/models/mod.rs.
// Kept free of onnx-ir types so the tests don't need the build-time dependencies.

use std::collections::HashSet;
use std::fmt::Write;

/// A graph input or output, as far as the registry cares
#[derive(Debug, Clone, PartialEq)]
pub enum IoType {
    /// f32 tensor; `None` dims are symbolic (unknown until runtime)
    F32Tensor(Vec<Option<usize>>),
    /// Anything else, described for the warning (e.g. "i64 tensor", "shape")
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Io {
    pub name: String,
    pub ty: IoType,
}

/// A model that can be served: one f32 input and one f32 output with static per-item shapes
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Name used on the command line and for the weight file (`<name>.bpk`): the ONNX file stem
    pub name: String,
    /// Rust module holding the generated code
    pub module: String,
    /// Per-item shapes, without the batch dimension
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    /// Batch size the graph was exported with, if static. Such graphs often bake it into
    /// reshapes, so the server runs larger batches through them in chunks of this size.
    pub fixed_batch: Option<usize>,
}

/// Normalized form used to match `--model-name` against registry names:
/// case-insensitive, ignoring `-`, `_` and `.` (so `resnet-18` finds `resnet18`).
pub fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '-' | '_' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

/// A valid, non-keyword Rust identifier derived from a file stem
pub fn rust_ident(name: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "as", "async", "await", "box", "break", "const", "continue", "crate", "dyn", "else",
        "enum", "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match",
        "mod", "move", "mut", "pub", "ref", "return", "self", "static", "struct", "super", "trait",
        "true", "try", "type", "unsafe", "use", "where", "while", "yield",
    ];
    let ident: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let needs_prefix = ident.is_empty()
        || ident.starts_with(|c: char| c.is_ascii_digit())
        || ident.chars().all(|c| c == '_')
        || KEYWORDS.contains(&ident.as_str());
    if needs_prefix {
        format!("model_{ident}")
    } else {
        ident
    }
}

/// Check that a graph fits the v1 serving contract and extract its per-item shapes.
///
/// The first dimension of the input and output is the batch; every other dimension must be
/// static. A static batch dimension is kept as [`ModelSpec::fixed_batch`]. Returns a
/// human-readable reason when the model can't be served.
pub fn model_spec(name: &str, inputs: &[Io], outputs: &[Io]) -> Result<ModelSpec, String> {
    let input = single(inputs, "input")?;
    let output = single(outputs, "output")?;
    Ok(ModelSpec {
        name: name.to_string(),
        module: rust_ident(name),
        input_shape: per_item_shape(input, "input")?,
        output_shape: per_item_shape(output, "output")?,
        fixed_batch: match &input.ty {
            IoType::F32Tensor(dims) => dims[0].filter(|&n| n > 0),
            IoType::Other(_) => None,
        },
    })
}

fn single<'a>(args: &'a [Io], what: &str) -> Result<&'a Io, String> {
    match args {
        [arg] => Ok(arg),
        _ => Err(format!(
            "has {} {what}s ({}); only models with exactly one {what} are supported",
            args.len(),
            args.iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn per_item_shape(arg: &Io, what: &str) -> Result<Vec<usize>, String> {
    let dims = match &arg.ty {
        IoType::F32Tensor(dims) => dims,
        IoType::Other(ty) => {
            return Err(format!(
                "{what} '{}' is a {ty}; only f32 tensors are supported",
                arg.name
            ));
        }
    };
    if dims.len() < 2 {
        return Err(format!(
            "{what} '{}' has rank {}; a batch dimension plus at least one more is required",
            arg.name,
            dims.len()
        ));
    }
    dims[1..]
        .iter()
        .map(|d| d.filter(|&d| d > 0))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            format!(
                "{what} '{}' has shape [{}]; every dimension except the first (batch) must be \
                 static",
                arg.name,
                dims.iter()
                    .map(|d| d.map_or("?".to_string(), |d| d.to_string()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// Drop specs whose name collides with an earlier one after normalization.
/// Returns the kept specs and a warning per dropped one.
pub fn dedup(specs: Vec<ModelSpec>) -> (Vec<ModelSpec>, Vec<String>) {
    let mut seen = HashSet::new();
    let mut modules = HashSet::new();
    let mut warnings = Vec::new();
    let mut kept = Vec::new();
    for spec in specs {
        if !seen.insert(normalize_name(&spec.name)) || !modules.insert(spec.module.clone()) {
            warnings.push(format!(
                "skipping model '{}': its name collides with another model in models/",
                spec.name
            ));
            continue;
        }
        kept.push(spec);
    }
    (kept, warnings)
}

/// Source of `$OUT_DIR/models/registry.rs`, included by src/models/mod.rs.
///
/// `built_in_model!` (defined in src/models/mod.rs) expands to a module holding the generated
/// code and a wrapper implementing `BurnModel`; this file only supplies the per-model facts.
pub fn render_registry(specs: &[ModelSpec]) -> String {
    let mut out = String::from("// @generated by build.rs from the ONNX files in models/\n\n");
    for spec in specs {
        let dims = |shape: &[usize]| {
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            out,
            "built_in_model!({module}, {name:?}, {file:?}, [{input}], [{output}], {batch:?});",
            module = spec.module,
            name = spec.name,
            file = format!("/models/{}.rs", spec.name),
            input = dims(&spec.input_shape),
            output = dims(&spec.output_shape),
            batch = spec.fixed_batch,
        )
        .unwrap();
    }
    out.push_str(
        "\n/// Models compiled into this binary, sorted by name\npub static MODELS: &[ModelEntry] = &[",
    );
    for spec in specs {
        write!(out, "\n    {}::ENTRY,", spec.module).unwrap();
    }
    out.push_str(if specs.is_empty() { "];\n" } else { "\n];\n" });
    out
}
