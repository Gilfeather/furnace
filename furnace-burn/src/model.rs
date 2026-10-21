//! Burn models as furnace serves them: [`BurnModel`], the models this crate ships (a forward
//! function, the file-based MLP, the built-in ONNX models) and the functions that load them.

use std::path::{Path, PathBuf};

use burn::tensor::{Device, Tensor};
use furnace::error::{ModelError, Result};
use furnace::{HostTensor, IoSpec, Model, TensorModel};
use tracing::{info, warn};

use crate::BackendKind;
use crate::mlp::{BurnModelContainer, create_sample_model};
use crate::models::{BuiltInModel, resolve_weights};
use crate::tensor::{HostTensorExt, int_range, into_float_flat, take_inputs};

/// A Burn model the server can run: what it takes, what it returns, and how to run it.
///
/// Implement this for your own Burn model to serve it with every feature of the server (named
/// tensors over the v2 protocol, metrics, reloads, ...), through [`custom`]. A model that takes
/// one float tensor and returns one needs none of this: see [`from_fn`].
///
/// ```ignore
/// use furnace_burn::{BackendKind, BurnModel, HostTensorExt};
///
/// #[derive(Debug)]
/// struct Classifier { net: MyNet, device: Device }
///
/// impl BurnModel for Classifier {
///     fn name(&self) -> &str { "classifier" }
///     fn backend(&self) -> BackendKind { BackendKind::Cpu }
///     fn inputs(&self) -> Vec<IoSpec> {
///         vec![IoSpec::new("input_ids", DataType::Int64, &[Some(128)]),
///              IoSpec::new("attention_mask", DataType::Int64, &[Some(128)])]
///     }
///     fn outputs(&self) -> Vec<IoSpec> { vec![IoSpec::fp32("logits", &[2])] }
///     fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
///         let [ids, mask] = <[HostTensor; 2]>::try_from(inputs).unwrap();
///         let logits = self.net.forward(ids.into_int::<2>(&self.device)?,
///                                       mask.into_int::<2>(&self.device)?);
///         Ok(vec![HostTensor::from_float(logits)?])
///     }
/// }
/// ```
pub trait BurnModel: Send + Sync + std::fmt::Debug {
    /// The name clients use: `/v2/models/{name}`, `/models/{name}/predict`
    fn name(&self) -> &str;

    /// The backend the model runs on, as reported by `/model/info`
    fn backend(&self) -> BackendKind;

    /// The inputs, in the order [`BurnModel::run`] takes them
    fn inputs(&self) -> Vec<IoSpec>;

    /// The outputs, in the order [`BurnModel::run`] returns them
    fn outputs(&self) -> Vec<IoSpec>;

    /// Run one batch. The server has already checked the inputs against
    /// [`BurnModel::inputs`]: one tensor each, in that order, with the batch as the first
    /// dimension, every fixed dimension as declared, and the same batch size throughout.
    /// Blocking; the server calls it off the async runtime.
    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>>;

    /// The batch size the model was built for, if it takes no other: larger batches are then
    /// run in chunks of it
    fn fixed_batch(&self) -> Option<usize> {
        None
    }
}

/// A [`BurnModel`] as the server runs it: its backend's name, and the range of its backend's
/// integers, which the server checks integer inputs against
#[derive(Debug)]
struct Burn(Box<dyn BurnModel>);

impl TensorModel for Burn {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn backend(&self) -> &str {
        self.0.backend().as_str()
    }

    fn inputs(&self) -> Vec<IoSpec> {
        self.0.inputs()
    }

    fn outputs(&self) -> Vec<IoSpec> {
        self.0.outputs()
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        self.0.run(inputs)
    }

    fn fixed_batch(&self) -> Option<usize> {
        self.0.fixed_batch()
    }

    fn int_range(&self) -> (i64, i64) {
        int_range(&self.0.backend().device())
    }
}

/// Wrap a Burn model; `model_type` and `model_size_bytes` are reported by `/model/info`
fn served(model: Box<dyn BurnModel>, model_type: &'static str, model_size_bytes: u64) -> Model {
    Model::new(Box::new(Burn(model)), model_type, model_size_bytes)
}

/// Serve your own Burn model (see [`BurnModel`])
pub fn custom(model: impl BurnModel + 'static) -> Model {
    served(Box::new(model), "burn", 0)
}

/// Serve a Burn forward function that takes one float tensor of rank `DI` (batch first) and
/// returns one of rank `DO`, e.g. a `Module`'s `forward`. `input_shape` and `output_shape` are
/// the per-item shapes, without the batch dimension. Such a model also serves `/predict` and
/// batches concurrent requests together.
///
/// ```ignore
/// let device = BackendKind::Cpu.device();
/// let net = MyNet::new(&device); // load weights with furnace_burn::weights::load_burnpack
/// let model = furnace_burn::from_fn("digits", BackendKind::Cpu, &[1, 28, 28], &[10],
///     move |x: Tensor<4>| net.forward(x))?;
/// ```
pub fn from_fn<const DI: usize, const DO: usize, F>(
    name: &str,
    backend: BackendKind,
    input_shape: &[usize],
    output_shape: &[usize],
    forward: F,
) -> Result<Model>
where
    F: Fn(Tensor<DI>) -> Tensor<DO> + Send + Sync + 'static,
{
    for (what, shape, rank) in [("input", input_shape, DI), ("output", output_shape, DO)] {
        if shape.len() + 1 != rank {
            return Err(ModelError::InvalidArgument {
                arg: format!("{what}_shape"),
                value: format!("{shape:?}"),
                reason: format!(
                    "the forward function's {what} has rank {rank}, so the per-item shape \
                     needs {} dimensions (the batch comes first)",
                    rank.saturating_sub(1)
                ),
            }
            .into());
        }
    }
    Ok(custom(FnModel {
        name: name.to_string(),
        backend,
        device: backend.device(),
        input: IoSpec::fp32("input", input_shape),
        output: IoSpec::fp32("output", output_shape),
        forward,
    }))
}

/// [`from_fn`]: a forward function from one float tensor to one
struct FnModel<F, const DI: usize, const DO: usize> {
    name: String,
    backend: BackendKind,
    device: Device,
    input: IoSpec,
    output: IoSpec,
    forward: F,
}

impl<F, const DI: usize, const DO: usize> std::fmt::Debug for FnModel<F, DI, DO> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnModel")
            .field("name", &self.name)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl<F, const DI: usize, const DO: usize> BurnModel for FnModel<F, DI, DO>
where
    F: Fn(Tensor<DI>) -> Tensor<DO> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![self.input.clone()]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![self.output.clone()]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let [input] = take_inputs(inputs)?;
        let output = (self.forward)(input.into_float::<DI>(&self.device)?);
        Ok(vec![HostTensor::from_float(output)?])
    }
}

/// File-based MLP model (`.bpk` weights + `.json` metadata)
#[derive(Debug)]
pub struct MlpModel {
    container: BurnModelContainer,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl MlpModel {
    /// Serve a loaded MLP container, with the shapes its metadata declares
    pub fn new(container: BurnModelContainer) -> Self {
        let input_shape = container.input_shape();
        let output_shape = container.output_shape();
        Self {
            container,
            input_shape,
            output_shape,
        }
    }
}

impl BurnModel for MlpModel {
    fn name(&self) -> &str {
        &self.container.metadata.name
    }

    fn backend(&self) -> BackendKind {
        self.container.backend
    }

    fn inputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("input", &self.input_shape)]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("output", &self.output_shape)]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let [input] = take_inputs(inputs)?;
        let mut shape = vec![input.batch()];
        shape.extend_from_slice(&self.output_shape);
        let output = self
            .container
            .predict(into_float_flat(input, self.container.device())?);
        Ok(vec![HostTensor {
            shape,
            values: HostTensor::from_float(output)?.values,
        }])
    }
}

/// The file name `load_model` replaces with a sample MLP with random weights when no such
/// file exists
const SAMPLE_MODEL: &str = "sample_model";

/// Load a file-based MLP model from `<path>.bpk` + `<path>.json` on the CPU backend.
///
/// A path named `sample_model` with no files serves an MLP with random weights, with a warning
/// (`cargo run -p furnace-cli --bin create_sample_model` writes real files there); any other
/// missing path is an error.
pub fn load_model(path: impl AsRef<Path>) -> Result<Model> {
    load_model_with_backend(path, BackendKind::Cpu)
}

/// Load a file-based MLP model from `<path>.bpk` + `<path>.json`
pub fn load_model_with_backend(path: impl AsRef<Path>, backend: BackendKind) -> Result<Model> {
    let path = path.as_ref();
    info!("Loading model from: {:?}", path);

    // Accept `model`, `model.bpk` or `model.json`
    let base: PathBuf = match path.extension().and_then(|e| e.to_str()) {
        Some("bpk") | Some("json") => path.with_extension(""),
        _ => path.to_path_buf(),
    };
    let weights_path = base.with_extension("bpk");

    // `sample_model` (what `create_sample_model` writes) without files on disk yields a freshly
    // initialized MLP with random weights; any other missing path is an error
    if !weights_path.exists() {
        if base.file_name().and_then(|s| s.to_str()) == Some(SAMPLE_MODEL) {
            warn!(
                "{} not found: serving a sample MLP with random weights",
                weights_path.display()
            );
            let container = create_sample_model()?;
            return Ok(served(Box::new(MlpModel::new(container)), "mlp", 0));
        }
        return Err(ModelError::FileNotFound(weights_path).into());
    }

    let model_size_bytes = std::fs::metadata(&weights_path)
        .map_err(|e| ModelError::LoadFailed {
            path: weights_path.clone(),
            source: Box::new(e),
        })?
        .len();

    let container = BurnModelContainer::load(&base, backend)?;
    Ok(served(
        Box::new(MlpModel::new(container)),
        "mlp",
        model_size_bytes,
    ))
}

/// Load a built-in (compiled-in) model by name on the CPU backend, with default weight lookup
pub fn load_built_in_model(model_name: &str) -> Result<Model> {
    load_built_in_model_with_backend(model_name, BackendKind::Cpu, None)
}

/// Load a built-in model by name on the given backend.
///
/// `weights` overrides where the `.bpk` weight file is read from; see
/// [`crate::models::resolve_weights`] for the default lookup.
pub fn load_built_in_model_with_backend(
    model_name: &str,
    backend: BackendKind,
    weights: Option<&Path>,
) -> Result<Model> {
    let built_in = BuiltInModel::from_name(model_name)?;
    let weights = resolve_weights(built_in.name(), weights)?;
    match &weights {
        Some(path) => info!(
            "Loading built-in model {} ({} backend) with weights from {:?}",
            model_name, backend, path
        ),
        None => info!(
            "Loading built-in model {} ({} backend) with embedded weights",
            model_name, backend
        ),
    }
    let model_size_bytes = weights
        .as_deref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(0, |m| m.len());
    let inner = built_in.create_model(backend, weights.as_deref())?;
    Ok(served(inner, "onnx", model_size_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use furnace::FurnaceError;

    #[test]
    fn test_model_loading_invalid_path() {
        let result = load_model(Path::new("nonexistent_model"));
        assert!(matches!(
            result,
            Err(FurnaceError::Model(ModelError::FileNotFound(_)))
        ));
    }

    #[test]
    fn test_real_mlp_inference_is_not_constant() {
        let model = load_model(Path::new("sample_model")).unwrap();
        assert_eq!(model.get_info().model_type, "mlp");
        assert_eq!(model.get_info().backend, "cpu");
        let out = model.predict(vec![0.3; 784]).unwrap();
        assert_eq!(out.len(), 10);
        assert!(out.iter().any(|&v| v != out[0]));
    }

    #[test]
    fn only_sample_model_itself_falls_back_to_random_weights() {
        for missing in ["my_sample_model", "samples/net", "sample"] {
            let result = load_model(Path::new(missing));
            assert!(
                matches!(
                    result,
                    Err(FurnaceError::Model(ModelError::FileNotFound(_)))
                ),
                "{missing}: {result:?}"
            );
        }
    }

    #[test]
    fn a_forward_function_is_served_with_its_shapes_checked() {
        let model = from_fn("double", BackendKind::Cpu, &[2], &[2], |x: Tensor<2>| {
            x * 2.0
        })
        .unwrap();
        assert_eq!(model.get_info().model_type, "burn");
        assert_eq!(model.predict(vec![1.0, 2.5]).unwrap(), [2.0, 5.0]);
        let err = from_fn("bad", BackendKind::Cpu, &[2, 2], &[2], |x: Tensor<2>| x).unwrap_err();
        assert!(err.to_string().contains("input_shape"), "{err}");
    }

    #[cfg(model_resnet18)]
    #[test]
    fn test_built_in_model_missing_weights_file() {
        let result = load_built_in_model_with_backend(
            "resnet18",
            BackendKind::Cpu,
            Some(Path::new("does/not/exist.bpk")),
        );
        assert!(matches!(
            result,
            Err(FurnaceError::Model(ModelError::FileNotFound(_)))
        ));
    }

    #[cfg(model_resnet18)]
    #[test]
    fn test_resnet18_weights_from_explicit_path() {
        // Copy the generated weights elsewhere to prove the path override is honored
        let dir = tempfile::tempdir().unwrap();
        let src = Path::new(env!("OUT_DIR")).join("models/resnet18.bpk");
        if !src.exists() {
            return; // built with embed-weights: no separate file
        }
        let dst = dir.path().join("custom.bpk");
        std::fs::copy(&src, &dst).unwrap();

        let model =
            load_built_in_model_with_backend("resnet18", BackendKind::Cpu, Some(&dst)).unwrap();
        assert_eq!(
            model.get_info().model_size_bytes,
            std::fs::metadata(&dst).unwrap().len()
        );
        assert_eq!(model.predict(vec![0.5; 3 * 224 * 224]).unwrap().len(), 1000);
    }

    #[cfg(model_resnet18)]
    #[test]
    fn test_resnet18_inference() {
        let model = load_built_in_model("resnet18").unwrap();
        assert_eq!(model.get_info().input_spec.shape, vec![3, 224, 224]);
        let out = model.predict(vec![0.5; 3 * 224 * 224]).unwrap();
        assert_eq!(out.len(), 1000);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[cfg(model_mnist)]
    #[test]
    fn test_mnist_inference() {
        let model = load_built_in_model("mnist").unwrap();
        assert_eq!(model.get_info().input_spec.shape, vec![1, 28, 28]);
        assert_eq!(model.get_info().output_spec.shape, vec![10]);
        let outputs = model
            .predict_batch(vec![vec![0.0; 28 * 28], vec![255.0; 28 * 28]])
            .unwrap();
        assert!(outputs.iter().all(|o| o.len() == 10));
        assert_ne!(outputs[0], outputs[1]);
    }
}
