// Built-in models generated from ONNX files in models/ by build.rs (burn-onnx).
//
// The model *architecture* is compiled into the binary; the weights live in a separate
// Burnpack file (`<name>.bpk`) loaded at startup, so the server binary stays small no matter
// how large the model is. With the `embed-weights` feature the weights are compiled in instead.

#[cfg(model_resnet18)]
#[allow(clippy::all, unused)]
pub mod resnet18 {
    include!(concat!(env!("OUT_DIR"), "/models/resnet18.rs"));
}

#[cfg(model_resnet18)]
use burn::tensor::{Device, Tensor};
use std::path::{Path, PathBuf};

use crate::backend::BackendKind;
use crate::error::{ModelError, Result};
use crate::model::BurnModel;

/// ResNet-18 image classifier (ImageNet, 1000 classes)
#[cfg(model_resnet18)]
#[derive(Debug)]
pub struct ResNet18 {
    model: resnet18::Model,
    device: Device,
    backend: BackendKind,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

#[cfg(model_resnet18)]
impl ResNet18 {
    /// Load ResNet-18 on `backend`, reading weights from `weights` (a `.bpk` file).
    ///
    /// With the `embed-weights` feature, `weights` is `None` and the compiled-in weights are used.
    pub fn load(backend: BackendKind, weights: Option<&Path>) -> Result<Self> {
        let device = backend.device();
        let model = match weights {
            Some(path) => load_burnpack(resnet18::Model::new(&device), path)?,
            #[cfg(feature = "embed-weights")]
            None => resnet18::Model::from_embedded(&device),
            #[cfg(not(feature = "embed-weights"))]
            None => unreachable!("weights are resolved before loading"),
        };
        Ok(Self {
            model,
            device,
            backend,
            input_shape: vec![3, 224, 224],
            output_shape: vec![1000],
        })
    }
}

/// Load Burnpack weights into a freshly constructed module, reporting errors instead of
/// panicking (release builds use `panic = "abort"`).
#[cfg(model_resnet18)]
fn load_burnpack<M: burn_store::ModuleSnapshot>(mut model: M, path: &Path) -> Result<M> {
    let mut store = burn_store::BurnpackStore::from_file(path).auto_extension(false);
    model
        .load_from(&mut store)
        .map_err(|e| ModelError::LoadFailed {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;
    Ok(model)
}

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

#[cfg(model_resnet18)]
impl BurnModel for ResNet18 {
    fn predict(&self, input: Tensor<2>) -> Result<Tensor<2>> {
        let [batch_size, input_size] = input.dims();
        let expected: usize = self.input_shape.iter().product();
        if input_size != expected {
            return Err(ModelError::InputValidation {
                expected: self.input_shape.clone(),
                actual: vec![input_size],
            }
            .into());
        }

        // Flattened [batch, C*H*W] -> NCHW image tensor
        let images = input.reshape([
            batch_size,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ]);
        Ok(self.model.forward(images))
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn get_name(&self) -> &str {
        "resnet18"
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

/// Models compiled into the binary
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInModel {
    ResNet18,
}

impl BuiltInModel {
    /// Look up a built-in model by name
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "resnet18" | "resnet-18" => Ok(Self::ResNet18),
            _ => Err(ModelError::InvalidArgument {
                arg: "model-name".to_string(),
                value: name.to_string(),
                reason: format!(
                    "unknown built-in model. Available models: {}",
                    Self::available_models().join(", ")
                ),
            }
            .into()),
        }
    }

    /// Built-in models that were successfully generated at build time
    pub fn available_models() -> Vec<&'static str> {
        vec![
            #[cfg(model_resnet18)]
            "resnet18",
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ResNet18 => "resnet18",
        }
    }

    /// Instantiate the model on the given backend, loading weights from `weights`
    /// (see [`resolve_weights`]).
    pub fn create_model(
        &self,
        backend: BackendKind,
        weights: Option<&Path>,
    ) -> Result<Box<dyn BurnModel>> {
        match self {
            Self::ResNet18 => {
                #[cfg(model_resnet18)]
                {
                    Ok(Box::new(ResNet18::load(backend, weights)?))
                }
                #[cfg(not(model_resnet18))]
                {
                    let _ = (backend, weights);
                    Err(ModelError::InvalidFormat(
                        "resnet18 was not compiled into this binary. \
                         Place models/resnet18.onnx and rebuild with the `onnx` feature"
                            .to_string(),
                    )
                    .into())
                }
            }
        }
    }
}
