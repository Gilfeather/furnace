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
use crate::error::{ModelError, Result};
use crate::model::BurnModel;

/// Loads a built-in model on a backend, from a weight file or embedded weights
type LoadFn = fn(BackendKind, Option<&Path>) -> Result<Box<dyn BurnModel>>;

/// A built-in model in the registry
#[derive(Debug)]
pub struct ModelEntry {
    /// Name used with `--model-name` and for the weight file: the ONNX file stem
    pub name: &'static str,
    /// Per-item input shape (without the batch dimension)
    pub input_shape: &'static [usize],
    /// Per-item output shape (without the batch dimension)
    pub output_shape: &'static [usize],
    /// Batch size the graph was exported with, if static; larger batches run in chunks of it
    pub fixed_batch: Option<usize>,
    load: LoadFn,
}

/// Defines module `$module` with the burn-onnx code generated for one model and a `Served`
/// wrapper that adapts its `forward` to the flattened `[batch, features]` [`BurnModel`] API.
#[allow(unused_macros)] // builds without models
macro_rules! built_in_model {
    (
        $module:ident, $name:literal, $file:literal,
        [$($in_dim:literal),+], [$($out_dim:literal),+], $fixed_batch:expr
    ) => {
        pub mod $module {
            #[allow(clippy::all, unused)]
            mod generated {
                include!(concat!(env!("OUT_DIR"), $file));
            }
            pub use generated::Model;

            use burn::tensor::{Device, Tensor};
            use std::path::Path;

            use crate::backend::BackendKind;
            use crate::error::{ModelError, Result};
            use crate::model::BurnModel;

            const INPUT_SHAPE: &[usize] = &[$($in_dim),+];
            const OUTPUT_SHAPE: &[usize] = &[$($out_dim),+];
            const INPUT_SIZE: usize = 1 $(* $in_dim)+;
            const OUTPUT_SIZE: usize = 1 $(* $out_dim)+;
            const FIXED_BATCH: Option<usize> = $fixed_batch;

            pub const ENTRY: super::ModelEntry = super::ModelEntry {
                name: $name,
                input_shape: INPUT_SHAPE,
                output_shape: OUTPUT_SHAPE,
                fixed_batch: FIXED_BATCH,
                load: |backend, weights| Ok(Box::new(Served::load(backend, weights)?)),
            };

            #[derive(Debug)]
            pub struct Served {
                model: Model,
                device: Device,
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

                fn forward(&self, input: Tensor<2>, batch_size: usize) -> Tensor<2> {
                    let output = self.model.forward(input.reshape([batch_size, $($in_dim),+]));
                    output.reshape([batch_size, OUTPUT_SIZE])
                }
            }

            impl BurnModel for Served {
                fn predict(&self, input: Tensor<2>) -> Result<Tensor<2>> {
                    let [batch_size, input_size] = input.dims();
                    if input_size != INPUT_SIZE {
                        return Err(ModelError::InputValidation {
                            expected: INPUT_SHAPE.to_vec(),
                            actual: vec![input_size],
                        }
                        .into());
                    }
                    match FIXED_BATCH {
                        // Exported with a static batch size, which reshapes inside the graph may
                        // depend on: feed it batches of exactly that size
                        Some(chunk) if chunk != batch_size => {
                            if batch_size % chunk != 0 {
                                return Err(ModelError::InputValidation {
                                    expected: vec![chunk],
                                    actual: vec![batch_size],
                                }
                                .into());
                            }
                            let outputs = input
                                .split(chunk, 0)
                                .into_iter()
                                .map(|part| self.forward(part, chunk))
                                .collect();
                            Ok(Tensor::cat(outputs, 0))
                        }
                        _ => Ok(self.forward(input, batch_size)),
                    }
                }

                fn device(&self) -> &Device {
                    &self.device
                }

                fn get_input_shape(&self) -> &[usize] {
                    INPUT_SHAPE
                }

                fn get_output_shape(&self) -> &[usize] {
                    OUTPUT_SHAPE
                }

                fn get_name(&self) -> &str {
                    $name
                }

                fn backend(&self) -> BackendKind {
                    self.backend
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

    /// Per-item input shape (without the batch dimension)
    pub fn input_shape(&self) -> &'static [usize] {
        self.0.input_shape
    }

    /// Per-item output shape (without the batch dimension)
    pub fn output_shape(&self) -> &'static [usize] {
        self.0.output_shape
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
    use super::build_registry::{Io, IoType, ModelSpec, dedup, model_spec, render_registry};
    use super::*;

    fn f32_io(name: &str, dims: &[Option<usize>]) -> Io {
        Io {
            name: name.to_string(),
            ty: IoType::F32Tensor(dims.to_vec()),
        }
    }

    fn image_classifier() -> (Vec<Io>, Vec<Io>) {
        (
            vec![f32_io("data", &[None, Some(3), Some(224), Some(224)])],
            vec![f32_io("logits", &[Some(1), Some(1000)])],
        )
    }

    #[test]
    fn spec_from_single_float_io() {
        let (inputs, outputs) = image_classifier();
        let spec = model_spec("resnet18", &inputs, &outputs).unwrap();
        assert_eq!(
            spec,
            ModelSpec {
                name: "resnet18".to_string(),
                module: "resnet18".to_string(),
                input_shape: vec![3, 224, 224],
                output_shape: vec![1000],
                fixed_batch: None,
            }
        );

        let fixed = [f32_io("data", &[Some(1), Some(3), Some(224), Some(224)])];
        let spec = model_spec("m", &fixed, &outputs).unwrap();
        assert_eq!(spec.fixed_batch, Some(1));
        assert_eq!(spec.input_shape, [3, 224, 224]);
    }

    #[test]
    fn spec_rejects_unsupported_graphs() {
        let (inputs, outputs) = image_classifier();
        let two_inputs = [inputs.clone(), inputs.clone()].concat();
        let err = model_spec("m", &two_inputs, &outputs).unwrap_err();
        assert!(err.contains("2 inputs (data, data)"), "{err}");
        assert!(
            model_spec("m", &inputs, &[])
                .unwrap_err()
                .contains("0 outputs")
        );

        let int_input = [Io {
            name: "ids".to_string(),
            ty: IoType::Other("I64 tensor".to_string()),
        }];
        let err = model_spec("m", &int_input, &outputs).unwrap_err();
        assert!(err.contains("'ids' is a I64 tensor"), "{err}");

        let dynamic = [f32_io("data", &[None, Some(3), None, None])];
        let err = model_spec("m", &dynamic, &outputs).unwrap_err();
        assert!(err.contains("[?, 3, ?, ?]"), "{err}");

        let rank1 = [f32_io("x", &[None])];
        let err = model_spec("m", &inputs, &rank1).unwrap_err();
        assert!(err.contains("rank 1"), "{err}");
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
            source.contains(
                r#"built_in_model!(my_net, "my-net", "/models/my-net.rs", [3, 224, 224], [1000], None);"#
            ),
            "{source}"
        );
        assert!(source.contains("my_net::ENTRY,"), "{source}");
    }

    #[test]
    fn name_normalization_matches_build_script() {
        for name in ["ResNet-18", "mobilenet_v2.opt", "MNIST"] {
            assert_eq!(normalize_name(name), build_registry::normalize_name(name));
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
            assert!(!model.input_shape().is_empty());
            assert!(!model.output_shape().is_empty());
        }
    }

    #[cfg(model_resnet18)]
    #[test]
    fn resnet18_is_registered_with_aliases() {
        let model = BuiltInModel::from_name("ResNet-18").unwrap();
        assert_eq!(model.name(), "resnet18");
        assert_eq!(model.input_shape(), [3, 224, 224]);
        assert_eq!(model.output_shape(), [1000]);
    }

    #[cfg(model_mnist)]
    #[test]
    fn mnist_is_registered() {
        let model = BuiltInModel::from_name("mnist").unwrap();
        assert_eq!(model.input_shape(), [1, 28, 28]);
        assert_eq!(model.output_shape(), [10]);
        // mnist-8 is exported with batch 1 and reshapes to [1, 256] inside the graph
        assert_eq!(model.fixed_batch(), Some(1));
    }
}
