// Turning ONNX graph I/O into the specs of servable models.
//
// Used by build.rs (which reads the graph with onnx-ir, `onnx` feature only) and unit-tested
// from src/models/mod.rs. Kept free of onnx-ir types so the tests don't need the build-time
// dependencies.

use std::collections::HashSet;

use super::render::{Elem, IoDesc, ModelSpec};

// The crate matches `--model-name` with the same function
include!("../src/models/name.rs");

/// A graph input or output, as far as the registry cares
#[derive(Debug, Clone, PartialEq)]
pub enum IoType {
    /// A tensor; `None` dims are symbolic (unknown until runtime)
    Tensor(Elem, Vec<Option<usize>>),
    /// Anything else, described for the warning (e.g. "Bool tensor", "shape")
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Io {
    pub name: String,
    pub ty: IoType,
}

/// A valid, non-keyword Rust identifier derived from a file stem
fn rust_ident(name: &str) -> String {
    // Strict and reserved keywords (2024 edition); the weak ones (`union`, `raw`, `safe`, ...)
    // are valid module names. The identifier is lowercase, so `Self` cannot occur.
    const KEYWORDS: &[&str] = &[
        "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
        "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl",
        "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub",
        "ref", "return", "self", "static", "struct", "super", "trait", "true", "try", "type",
        "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
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

/// Check that a graph can be served and describe its inputs and outputs.
///
/// Every input and output must be an `f32`, `int64` or `int32` tensor whose first dimension is
/// the batch. Other dimensions may be open; requests then fix them. A static batch dimension on
/// the first input is kept as [`ModelSpec::fixed_batch`]. Returns a human-readable reason when
/// the model can't be served.
pub fn model_spec(name: &str, inputs: &[Io], outputs: &[Io]) -> Result<ModelSpec, String> {
    if inputs.is_empty() {
        return Err("has no inputs".to_string());
    }
    if outputs.is_empty() {
        return Err("has no outputs".to_string());
    }
    let describe = |args: &[Io], what| {
        args.iter()
            .map(|arg| describe(arg, what))
            .collect::<Result<Vec<_>, _>>()
    };
    let fixed_batch = match &inputs[0].ty {
        IoType::Tensor(_, dims) => dims.first().copied().flatten().filter(|&n| n > 0),
        IoType::Other(_) => None,
    };
    Ok(ModelSpec {
        name: name.to_string(),
        module: rust_ident(name),
        inputs: describe(inputs, "input")?,
        outputs: describe(outputs, "output")?,
        fixed_batch,
    })
}

fn describe(arg: &Io, what: &str) -> Result<IoDesc, String> {
    let (elem, dims) = match &arg.ty {
        IoType::Tensor(elem, dims) => (*elem, dims),
        IoType::Other(ty) => {
            return Err(format!(
                "{what} '{}' is a {ty}; only f32, int64 and int32 tensors are supported",
                arg.name
            ));
        }
    };
    if dims.is_empty() {
        return Err(format!(
            "{what} '{}' is a scalar; a batch dimension is required",
            arg.name
        ));
    }
    Ok(IoDesc {
        name: arg.name.clone(),
        elem,
        item: dims[1..].iter().map(|d| d.filter(|&d| d > 0)).collect(),
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
