// Turning ONNX graph I/O into the built-in model registry.
//
// Used by build.rs (which reads the graph with onnx-ir) and unit-tested from src/models/mod.rs.
// Kept free of onnx-ir types so the tests don't need the build-time dependencies.

use std::collections::HashSet;
use std::fmt::Write;

/// Element types the server can feed to and read from a model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Elem {
    F32,
    I64,
    I32,
}

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

/// An input or output of a servable model
#[derive(Debug, Clone, PartialEq)]
pub struct IoDesc {
    /// Name in the graph, which is also the argument order of the generated `forward`
    pub name: String,
    pub elem: Elem,
    /// Per-item shape, without the batch dimension; `None` where the graph leaves it open
    pub item: Vec<Option<usize>>,
}

impl IoDesc {
    /// Rank of the full tensor, batch included
    pub fn rank(&self) -> usize {
        self.item.len() + 1
    }

    fn static_item(&self) -> Option<Vec<usize>> {
        self.item.iter().copied().collect()
    }
}

/// A model that can be served: tensors in and out, each with the batch as its first dimension
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Name used on the command line and for the weight file (`<name>.bpk`): the ONNX file stem
    pub name: String,
    /// Rust module holding the generated code
    pub module: String,
    pub inputs: Vec<IoDesc>,
    pub outputs: Vec<IoDesc>,
    /// Batch size the graph was exported with, if static. Such graphs often bake it into
    /// reshapes, so the server runs larger batches through them in chunks of this size.
    pub fixed_batch: Option<usize>,
}

impl ModelSpec {
    /// Per-item input and output shapes if the model takes one `f32` tensor and returns one,
    /// both with fixed shapes: the common case, served with dynamic batching on flat buffers
    pub fn simple(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        match (self.inputs.as_slice(), self.outputs.as_slice()) {
            ([input], [output]) if input.elem == Elem::F32 && output.elem == Elem::F32 => {
                let input = input.static_item().filter(|s| !s.is_empty())?;
                let output = output.static_item().filter(|s| !s.is_empty())?;
                Some((input, output))
            }
            _ => None,
        }
    }
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

/// Source of `$OUT_DIR/models/registry.rs`, included by src/models/mod.rs.
///
/// `built_in_model!` (defined in src/models/mod.rs) expands to a module holding the generated
/// code and a wrapper implementing `BurnModel`. This file supplies the per-model facts, and
/// for each model the code calling its generated `forward`, whose arguments and results follow
/// the graph's inputs and outputs.
pub fn render_registry(specs: &[ModelSpec]) -> String {
    let mut out = String::from("// @generated by build.rs from the ONNX files in models/\n\n");
    if !specs.is_empty() {
        out.push_str("use crate::tensor::HostTensor;\n\n");
    }
    for spec in specs {
        writeln!(
            out,
            "built_in_model!({module}, {name:?}, {file:?}, {batch:?});",
            module = spec.module,
            name = spec.name,
            file = format!("/models/{}.rs", spec.name),
            batch = spec.fixed_batch,
        )
        .unwrap();
        render_io(&mut out, spec, spec.simple().is_some());
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

/// The model's input and output descriptions and `run_generated`, which turns host tensors
/// into the generated `forward`'s arguments and its results back
fn render_io(out: &mut String, spec: &ModelSpec, simple: bool) {
    // A simple model keeps the names `input` and `output`, as before multi-tensor support
    let name = |desc: &IoDesc, fixed: &str| {
        if simple {
            fixed.to_string()
        } else {
            desc.name.clone()
        }
    };
    let list = |descs: &[IoDesc], fixed: &str| {
        descs
            .iter()
            .map(|d| {
                let shape = d
                    .item
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "StaticIo {{ name: {:?}, dtype: DataType::{}, shape: &[{shape}] }}",
                    name(d, fixed),
                    match d.elem {
                        Elem::F32 => "Fp32",
                        Elem::I64 => "Int64",
                        Elem::I32 => "Int32",
                    },
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let kind = |d: &IoDesc| if d.elem == Elem::F32 { "float" } else { "int" };
    let args: Vec<String> = spec
        .inputs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                "        let in{i} = next()?.into_{}::<{}>(&self.device)?;\n",
                kind(d),
                d.rank()
            )
        })
        .collect();
    let call_args = (0..spec.inputs.len())
        .map(|i| format!("in{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let results = (0..spec.outputs.len())
        .map(|i| format!("out{i}"))
        .collect::<Vec<_>>();
    let bind = if results.len() == 1 {
        results[0].clone()
    } else {
        format!("({})", results.join(", "))
    };
    let converted = spec
        .outputs
        .iter()
        .enumerate()
        .map(|(i, d)| format!("HostTensor::from_{}::<{}>(out{i})?", kind(d), d.rank()))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        out,
        "impl {module}::Served {{
    pub(super) const INPUTS: &'static [StaticIo] = &[{inputs}];
    pub(super) const OUTPUTS: &'static [StaticIo] = &[{outputs}];

    pub(super) fn run_generated(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {{
        let mut inputs = inputs.into_iter();
        let mut next = || inputs.next().ok_or_else(|| missing_input({count}));
{args}        let {bind} = self.model.forward({call_args});
        Ok(vec![{converted}])
    }}
}}
",
        module = spec.module,
        inputs = list(&spec.inputs, "input"),
        outputs = list(&spec.outputs, "output"),
        count = spec.inputs.len(),
        args = args.concat(),
    )
    .unwrap();
}
