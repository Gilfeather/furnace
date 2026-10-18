// Built-in models generated from ONNX files in models/ by build.rs (burn-onnx).
//
// build.rs reads each graph's input and output shapes and writes a registry
// (`$OUT_DIR/models/registry.rs`) that invokes `built_in_model!` once per model, so any
// supported ONNX file in models/ is servable without hand-written code.
//
// The model *architecture* is compiled into the binary; the weights live in a separate
// Burnpack file (`<name>.bpk`) loaded at startup, so the server binary stays small no matter
// how large the model is. With the `embed-weights` feature the weights are compiled in instead.

use std::path::{Path, PathBuf};

use crate::backend::BackendKind;
use crate::error::{FurnaceError, ModelError, Result};
use crate::model::BurnModel;
use crate::tensor::{DataType, IoSpec};

/// Loads a built-in model on a backend, from a weight file or embedded weights
type LoadFn = fn(BackendKind, Option<&Path>) -> Result<Box<dyn BurnModel>>;

/// A built-in model's input or output, as the registry records it
#[derive(Debug)]
pub struct StaticIo {
    pub name: &'static str,
    pub dtype: DataType,
    /// Per-item shape; `None` where the graph leaves the dimension open
    pub shape: &'static [Option<usize>],
}

impl StaticIo {
    pub fn spec(&self) -> IoSpec {
        IoSpec {
            name: self.name.to_string(),
            dtype: self.dtype,
            shape: self.shape.to_vec(),
        }
    }
}

/// A built-in model in the registry
#[derive(Debug)]
pub struct ModelEntry {
    /// Name used with `--model-name` and for the weight file: the ONNX file stem
    pub name: &'static str,
    pub inputs: &'static [StaticIo],
    pub outputs: &'static [StaticIo],
    /// Batch size the graph was exported with, if static; larger batches run in chunks of it
    pub fixed_batch: Option<usize>,
    load: LoadFn,
}

/// Error for a generated `run_generated` given fewer inputs than its model takes. Inputs are
/// checked before that, so this only guards against a bug.
pub fn missing_input(expected: usize) -> FurnaceError {
    ModelError::InferenceFailed(format!("the model takes {expected} inputs")).into()
}

/// Defines module `$module` with the burn-onnx code generated for one model and a `Served`
/// wrapper implementing [`BurnModel`]. The registry adds `Served::INPUTS`, `Served::OUTPUTS` and
/// `Served::run_generated`, which calls the generated `forward`.
#[allow(unused_macros)] // builds without models
macro_rules! built_in_model {
    ($module:ident, $name:literal, $file:literal, $fixed_batch:expr) => {
        pub mod $module {
            #[allow(clippy::all, unused)]
            mod generated {
                include!(concat!(env!("OUT_DIR"), $file));
            }
            pub use generated::Model;

            use burn::tensor::Device;
            use std::path::Path;

            use crate::backend::BackendKind;
            use crate::error::{ModelError, Result};
            use crate::model::BurnModel;
            use crate::tensor::{HostTensor, IoSpec};

            pub const ENTRY: super::ModelEntry = super::ModelEntry {
                name: $name,
                inputs: Served::INPUTS,
                outputs: Served::OUTPUTS,
                fixed_batch: $fixed_batch,
                load: |backend, weights| Ok(Box::new(Served::load(backend, weights)?)),
            };

            #[derive(Debug)]
            pub struct Served {
                pub(super) model: Model,
                pub(super) device: Device,
                backend: BackendKind,
            }

            impl Served {
                /// Load the model on `backend`, reading weights from `weights` (a `.bpk` file).
                ///
                /// With the `embed-weights` feature, `weights` may be `None` to use the
                /// compiled-in weights.
                pub fn load(backend: BackendKind, weights: Option<&Path>) -> Result<Self> {
                    let device = backend.device();
                    let model = match weights {
                        Some(path) => {
                            let mut model = Model::new(&device);
                            crate::weights::load_burnpack(&mut model, path)?;
                            model
                        }
                        #[cfg(feature = "embed-weights")]
                        None => Model::from_embedded(&device),
                        #[cfg(not(feature = "embed-weights"))]
                        None => {
                            return Err(ModelError::InvalidArgument {
                                arg: "weights".to_string(),
                                value: $name.to_string(),
                                reason: "no weight file given and weights are not embedded"
                                    .to_string(),
                            }
                            .into());
                        }
                    };
                    Ok(Self {
                        model,
                        device,
                        backend,
                    })
                }
            }

            impl BurnModel for Served {
                fn name(&self) -> &str {
                    $name
                }

                fn backend(&self) -> BackendKind {
                    self.backend
                }

                fn inputs(&self) -> Vec<IoSpec> {
                    Self::INPUTS.iter().map(super::StaticIo::spec).collect()
                }

                fn outputs(&self) -> Vec<IoSpec> {
                    Self::OUTPUTS.iter().map(super::StaticIo::spec).collect()
                }

                fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
                    self.run_generated(inputs)
                }

                fn fixed_batch(&self) -> Option<usize> {
                    $fixed_batch
                }
            }
        }
    };
}

include!(concat!(env!("OUT_DIR"), "/models/registry.rs"));

/// Where to look for a built-in model's weights when `--weights` is not given, in order:
/// next to the executable (release archives), `models/` in the working directory, and the
/// build output directory (so `cargo run` works on the build machine).
pub fn default_weight_candidates(name: &str) -> Vec<PathBuf> {
    let file = format!("{name}.bpk");
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        candidates.push(dir.join(&file));
    }
    candidates.push(Path::new("models").join(&file));
    candidates.push(Path::new(env!("OUT_DIR")).join("models").join(&file));
    candidates
}

/// Resolve the weight file for a built-in model.
///
/// Returns `Ok(None)` when weights are compiled in and no override was given.
pub fn resolve_weights(name: &str, explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        if !path.is_file() {
            return Err(ModelError::FileNotFound(path.to_path_buf()).into());
        }
        return Ok(Some(path.to_path_buf()));
    }
    if cfg!(feature = "embed-weights") {
        return Ok(None);
    }
    let candidates = default_weight_candidates(name);
    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            ModelError::InvalidArgument {
                arg: "weights".to_string(),
                value: name.to_string(),
                reason: format!(
                    "weight file not found; pass --weights <PATH> or place {name}.bpk in one of: {}",
                    candidates
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
            .into()
        })
}

/// A model compiled into the binary (see [`BuiltInModel::available_models`])
#[derive(Debug, Clone, Copy)]
pub struct BuiltInModel(&'static ModelEntry);

impl PartialEq for BuiltInModel {
    fn eq(&self, other: &Self) -> bool {
        self.0.name == other.0.name
    }
}

impl Eq for BuiltInModel {}

impl BuiltInModel {
    /// Look up a built-in model by name, ignoring case, `-`, `_` and `.`
    /// (so `resnet-18` finds `resnet18`)
    pub fn from_name(name: &str) -> Result<Self> {
        let wanted = normalize_name(name);
        MODELS
            .iter()
            .find(|entry| normalize_name(entry.name) == wanted)
            .map(Self)
            .ok_or_else(|| {
                ModelError::InvalidArgument {
                    arg: "model-name".to_string(),
                    value: name.to_string(),
                    reason: format!("unknown built-in model. {}", Self::describe_available()),
                }
                .into()
            })
    }

    /// The model to serve when neither `--model-name` nor `--model-path` is given: the only
    /// built-in model, if there is exactly one
    pub fn default_model() -> Result<Self> {
        match MODELS {
            [entry] => Ok(Self(entry)),
            _ => Err(ModelError::InvalidArgument {
                arg: "model-name".to_string(),
                value: String::new(),
                reason: format!("no model selected. {}", Self::describe_available()),
            }
            .into()),
        }
    }

    /// Names of the built-in models that were generated at build time, sorted
    pub fn available_models() -> Vec<&'static str> {
        MODELS.iter().map(|entry| entry.name).collect()
    }

    /// One-sentence summary of the built-in models, for help text and errors
    pub fn describe_available() -> String {
        if MODELS.is_empty() {
            "No built-in models in this build: place an ONNX file in models/ and rebuild"
                .to_string()
        } else {
            format!("Available models: {}", Self::available_models().join(", "))
        }
    }

    pub fn name(&self) -> &'static str {
        self.0.name
    }

    /// The model's inputs, in the order its graph takes them
    pub fn inputs(&self) -> &'static [StaticIo] {
        self.0.inputs
    }

    /// The model's outputs
    pub fn outputs(&self) -> &'static [StaticIo] {
        self.0.outputs
    }

    /// Batch size the graph was exported with, if static (see [`ModelEntry::fixed_batch`])
    pub fn fixed_batch(&self) -> Option<usize> {
        self.0.fixed_batch
    }

    /// Instantiate the model on the given backend, loading weights from `weights`
    /// (see [`resolve_weights`]).
    pub fn create_model(
        &self,
        backend: BackendKind,
        weights: Option<&Path>,
    ) -> Result<Box<dyn BurnModel>> {
        (self.0.load)(backend, weights)
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '-' | '_' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
#[path = "../../build/registry.rs"]
mod build_registry;

#[cfg(test)]
mod tests {
    use super::build_registry::{
        Elem, Io, IoDesc, IoType, dedup, model_spec, normalize_name as build_normalize,
        render_registry,
    };
    use super::*;

    fn tensor(name: &str, elem: Elem, dims: &[Option<usize>]) -> Io {
        Io {
            name: name.to_string(),
            ty: IoType::Tensor(elem, dims.to_vec()),
        }
    }

    fn image_classifier() -> (Vec<Io>, Vec<Io>) {
        (
            vec![tensor(
                "data",
                Elem::F32,
                &[None, Some(3), Some(224), Some(224)],
            )],
            vec![tensor("logits", Elem::F32, &[Some(1), Some(1000)])],
        )
    }

    /// The shape of bertsquad-12: four int64 inputs, one of rank 1; outputs of open shape
    fn bert() -> (Vec<Io>, Vec<Io>) {
        (
            vec![
                tensor("unique_ids", Elem::I64, &[None]),
                tensor("segment_ids", Elem::I64, &[None, Some(256)]),
                tensor("input_mask", Elem::I64, &[None, Some(256)]),
                tensor("input_ids", Elem::I64, &[None, Some(256)]),
            ],
            vec![
                tensor("start_logits", Elem::F32, &[None, None]),
                tensor("end_logits", Elem::F32, &[None, None]),
                tensor("unique_ids_out", Elem::I64, &[None]),
            ],
        )
    }

    #[test]
    fn spec_from_single_float_io() {
        let (inputs, outputs) = image_classifier();
        let spec = model_spec("resnet18", &inputs, &outputs).unwrap();
        assert_eq!(spec.module, "resnet18");
        assert_eq!(spec.fixed_batch, None);
        assert_eq!(
            spec.inputs,
            [IoDesc {
                name: "data".to_string(),
                elem: Elem::F32,
                item: vec![Some(3), Some(224), Some(224)],
            }]
        );
        assert_eq!(spec.simple(), Some((vec![3, 224, 224], vec![1000])));

        let fixed = [tensor(
            "data",
            Elem::F32,
            &[Some(1), Some(3), Some(224), Some(224)],
        )];
        let spec = model_spec("m", &fixed, &outputs).unwrap();
        assert_eq!(spec.fixed_batch, Some(1));
        assert!(spec.simple().is_some());
    }

    #[test]
    fn spec_for_several_and_integer_tensors() {
        let (inputs, outputs) = bert();
        let spec = model_spec("bertsquad", &inputs, &outputs).unwrap();
        assert_eq!(spec.inputs.len(), 4);
        assert_eq!(spec.inputs[0].item, []);
        assert_eq!(spec.inputs[0].rank(), 1);
        assert_eq!(spec.inputs[3].elem, Elem::I64);
        assert_eq!(spec.outputs[0].item, [None]);
        assert_eq!(spec.simple(), None);

        // Open per-item dimensions are fine; so is a second output
        let (inputs, mut outputs) = image_classifier();
        outputs.push(tensor("boxes", Elem::F32, &[None, None, Some(4)]));
        let spec = model_spec("detector", &inputs, &outputs).unwrap();
        assert_eq!(spec.simple(), None);
        let dynamic = [tensor("data", Elem::F32, &[None, Some(3), None, None])];
        let spec = model_spec("m", &dynamic, &outputs[..1]).unwrap();
        assert_eq!(spec.simple(), None, "open input dims are not simple");

        // int32 inputs become integer tensors too, described as INT32
        let ids = [tensor("ids", Elem::I32, &[None, Some(8)])];
        let spec = model_spec("m", &ids, &outputs[..1]).unwrap();
        assert_eq!(spec.simple(), None);
        let source = render_registry(&[spec]);
        assert!(
            source.contains("let in0 = next()?.into_int::<2>(&self.device)?;"),
            "{source}"
        );
        assert!(source.contains("dtype: DataType::Int32"), "{source}");
    }

    #[test]
    fn spec_rejects_unsupported_graphs() {
        let (inputs, outputs) = image_classifier();
        assert!(
            model_spec("m", &[], &outputs)
                .unwrap_err()
                .contains("no inputs")
        );
        assert!(
            model_spec("m", &inputs, &[])
                .unwrap_err()
                .contains("no outputs")
        );
        let boolean = [Io {
            name: "mask".to_string(),
            ty: IoType::Other("Bool tensor".to_string()),
        }];
        let err = model_spec("m", &boolean, &outputs).unwrap_err();
        assert!(err.contains("'mask' is a Bool tensor"), "{err}");
        let scalar = [tensor("x", Elem::F32, &[])];
        let err = model_spec("m", &inputs, &scalar).unwrap_err();
        assert!(err.contains("scalar"), "{err}");
    }

    #[test]
    fn module_names_are_valid_identifiers() {
        let (inputs, outputs) = image_classifier();
        let module = |name| model_spec(name, &inputs, &outputs).unwrap().module;
        assert_eq!(module("MobileNet-V2.opt"), "mobilenet_v2_opt");
        assert_eq!(module("18layers"), "model_18layers");
        assert_eq!(module("type"), "model_type");
    }

    #[test]
    fn dedup_drops_colliding_names() {
        let (inputs, outputs) = image_classifier();
        let spec = |name| model_spec(name, &inputs, &outputs).unwrap();
        let (kept, warnings) = dedup(vec![spec("resnet18"), spec("ResNet-18"), spec("mnist")]);
        let names: Vec<_> = kept.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["resnet18", "mnist"]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("ResNet-18"));
    }

    #[test]
    fn registry_source() {
        assert!(render_registry(&[]).contains("pub static MODELS: &[ModelEntry] = &[];"));

        let (inputs, outputs) = image_classifier();
        let source = render_registry(&[model_spec("my-net", &inputs, &outputs).unwrap()]);
        assert!(
            source.contains(r#"built_in_model!(my_net, "my-net", "/models/my-net.rs", None);"#),
            "{source}"
        );
        // A simple model keeps the names input and output
        assert!(
            source.contains(r#"name: "input", dtype: DataType::Fp32"#),
            "{source}"
        );
        assert!(
            source.contains("let out0 = self.model.forward(in0);"),
            "{source}"
        );
        assert!(source.contains("my_net::ENTRY,"), "{source}");

        let (inputs, outputs) = bert();
        let source = render_registry(&[model_spec("bert", &inputs, &outputs).unwrap()]);
        assert!(
            source.contains(r#"built_in_model!(bert, "bert", "/models/bert.rs", None);"#),
            "{source}"
        );
        assert!(
            source.contains("let in0 = next()?.into_int::<1>(&self.device)?;"),
            "{source}"
        );
        assert!(
            source.contains("let in3 = next()?.into_int::<2>(&self.device)?;"),
            "{source}"
        );
        assert!(
            source.contains("let (out0, out1, out2) = self.model.forward(in0, in1, in2, in3);"),
            "{source}"
        );
        assert!(
            source.contains(
                "HostTensor::from_float::<2>(out0)?, HostTensor::from_float::<2>(out1)?, HostTensor::from_int::<1>(out2)?"
            ),
            "{source}"
        );
        assert!(
            source.contains(r#"name: "input_ids", dtype: DataType::Int64, shape: &[Some(256)]"#),
            "{source}"
        );
    }

    #[test]
    fn name_normalization_matches_build_script() {
        for name in ["ResNet-18", "mobilenet_v2.opt", "MNIST"] {
            assert_eq!(normalize_name(name), build_normalize(name));
        }
    }

    #[test]
    fn unknown_model_lists_available_ones() {
        let err = BuiltInModel::from_name("no-such-model")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown built-in model"), "{err}");
        assert!(err.contains(&BuiltInModel::describe_available()), "{err}");
    }

    #[test]
    fn registry_is_sorted_and_consistent() {
        let names = BuiltInModel::available_models();
        assert!(names.is_sorted(), "{names:?}");
        for name in names {
            let model = BuiltInModel::from_name(name).unwrap();
            assert_eq!(model.name(), name);
            assert!(!model.inputs().is_empty());
            assert!(!model.outputs().is_empty());
        }
    }

    #[cfg(model_resnet18)]
    #[test]
    fn resnet18_is_registered_with_aliases() {
        let model = BuiltInModel::from_name("ResNet-18").unwrap();
        assert_eq!(model.name(), "resnet18");
        assert_eq!(model.inputs()[0].name, "input");
        assert_eq!(model.inputs()[0].shape, [Some(3), Some(224), Some(224)]);
        assert_eq!(model.outputs()[0].shape, [Some(1000)]);
    }

    #[cfg(model_mnist)]
    #[test]
    fn mnist_is_registered() {
        let model = BuiltInModel::from_name("mnist").unwrap();
        assert_eq!(model.inputs()[0].shape, [Some(1), Some(28), Some(28)]);
        assert_eq!(model.outputs()[0].shape, [Some(10)]);
        // mnist-8 is exported with batch 1 and reshapes to [1, 256] inside the graph
        assert_eq!(model.fixed_batch(), Some(1));
    }
}
