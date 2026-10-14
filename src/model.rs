use burn::tensor::{Device, Tensor, TensorData};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use tracing::{debug, info};

use crate::backend::BackendKind;
use crate::burn_model::{BurnModelContainer, create_sample_model};
use crate::error::{ModelError, Result};
use crate::models::{BuiltInModel, resolve_weights};
use crate::types::TensorSpec;

/// Hard upper bound on the number of items in one inference batch
pub const MAX_BATCH_SIZE: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub input_spec: TensorSpec,
    pub output_spec: TensorSpec,
    pub model_type: String,
    pub backend: String,
    pub created_at: String,
    pub model_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub inference_count: u64,
    pub total_inference_time_ms: f64,
    pub last_inference_time: Option<DateTime<Utc>>,
    pub average_inference_time_ms: f64,
    pub error_count: u64,
    pub success_count: u64,
    pub min_inference_time_ms: Option<f64>,
    pub max_inference_time_ms: Option<f64>,
}

impl ModelStats {
    fn new() -> Self {
        Self {
            inference_count: 0,
            total_inference_time_ms: 0.0,
            last_inference_time: None,
            average_inference_time_ms: 0.0,
            error_count: 0,
            success_count: 0,
            min_inference_time_ms: None,
            max_inference_time_ms: None,
        }
    }

    fn record_success(&mut self, elapsed_ms: f64) {
        self.inference_count += 1;
        self.success_count += 1;
        self.total_inference_time_ms += elapsed_ms;
        self.average_inference_time_ms = self.total_inference_time_ms / self.success_count as f64;
        self.min_inference_time_ms = Some(
            self.min_inference_time_ms
                .map_or(elapsed_ms, |m| m.min(elapsed_ms)),
        );
        self.max_inference_time_ms = Some(
            self.max_inference_time_ms
                .map_or(elapsed_ms, |m| m.max(elapsed_ms)),
        );
        self.last_inference_time = Some(Utc::now());
    }

    fn record_error(&mut self) {
        self.inference_count += 1;
        self.error_count += 1;
    }
}

/// A model that can run inference on flattened `[batch, features]` input.
pub trait BurnModel: Send + Sync + std::fmt::Debug {
    fn predict(&self, input: Tensor<2>) -> Result<Tensor<2>>;
    /// Device the model's weights live on; inputs are created here.
    fn device(&self) -> &Device;
    /// Per-item input shape (without the batch dimension)
    fn get_input_shape(&self) -> &[usize];
    /// Per-item output shape (without the batch dimension)
    fn get_output_shape(&self) -> &[usize];
    fn get_name(&self) -> &str;
    fn backend(&self) -> BackendKind;
}

/// File-based MLP model (`.bpk` weights + `.json` metadata)
#[derive(Debug)]
pub struct MlpModel {
    container: BurnModelContainer,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl MlpModel {
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
    fn predict(&self, input: Tensor<2>) -> Result<Tensor<2>> {
        Ok(self.container.predict(input))
    }

    fn device(&self) -> &Device {
        self.container.device()
    }

    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn get_name(&self) -> &str {
        &self.container.metadata.name
    }

    fn backend(&self) -> BackendKind {
        self.container.backend
    }
}

/// Constant-output model for tests and benchmarking server overhead.
///
/// Never used as a silent fallback: if a real model fails to load, loading fails.
#[derive(Debug)]
pub struct DummyModel {
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    device: Device,
}

impl DummyModel {
    pub fn new(name: String, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            name,
            input_shape,
            output_shape,
            device: BackendKind::Cpu.device(),
        }
    }
}

impl BurnModel for DummyModel {
    fn predict(&self, input: Tensor<2>) -> Result<Tensor<2>> {
        let [batch_size, _] = input.dims();
        let output_size = self.output_shape.iter().product::<usize>();
        Ok(Tensor::from_data(
            TensorData::new(
                vec![0.5f32; batch_size * output_size],
                [batch_size, output_size],
            ),
            &self.device,
        ))
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
        &self.name
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }
}

/// A loaded model plus its metadata and runtime statistics
#[derive(Debug)]
pub struct Model {
    inner: Box<dyn BurnModel>,
    info: ModelInfo,
    model_type: &'static str,
    stats: Mutex<ModelStats>,
}

impl Model {
    pub fn new(inner: Box<dyn BurnModel>, model_type: &'static str, model_size_bytes: u64) -> Self {
        let info = ModelInfo {
            name: inner.get_name().to_string(),
            version: "1.0.0".to_string(),
            input_spec: TensorSpec {
                shape: inner.get_input_shape().to_vec(),
                dtype: "float32".to_string(),
                min_value: None,
                max_value: None,
            },
            output_spec: TensorSpec {
                shape: inner.get_output_shape().to_vec(),
                dtype: "float32".to_string(),
                min_value: None,
                max_value: None,
            },
            model_type: model_type.to_string(),
            backend: inner.backend().to_string(),
            created_at: Utc::now().to_rfc3339(),
            model_size_bytes,
        };

        Self {
            inner,
            info,
            model_type,
            stats: Mutex::new(ModelStats::new()),
        }
    }

    pub fn predict(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        self.predict_batch(vec![input])
            .map(|mut batch| batch.pop().expect("batch of one yields one output"))
    }

    /// Run inference on a batch of flattened inputs.
    ///
    /// This is CPU/GPU-bound and blocking; call it from a blocking context
    /// (e.g. `tokio::task::spawn_blocking`) inside async code.
    pub fn predict_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let result = self.run_batch(inputs);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        match &result {
            Ok(outputs) => {
                stats.record_success(elapsed_ms);
                debug!(
                    "Batch inference completed in {:.2}ms, batch size: {}, backend: {}",
                    elapsed_ms,
                    outputs.len(),
                    self.info.backend
                );
            }
            Err(_) => stats.record_error(),
        }
        result
    }

    fn run_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Err(ModelError::InputValidation {
                expected: vec![1],
                actual: vec![0],
            }
            .into());
        }

        let batch_size = inputs.len();
        if batch_size > MAX_BATCH_SIZE {
            return Err(ModelError::InputValidation {
                expected: vec![MAX_BATCH_SIZE],
                actual: vec![batch_size],
            }
            .into());
        }

        let input_size: usize = self.info.input_spec.shape.iter().product();
        let mut flattened = Vec::with_capacity(batch_size * input_size);
        for (i, input) in inputs.into_iter().enumerate() {
            self.validate_input_shape(&input)?;
            if let Some(j) = input.iter().position(|x| !x.is_finite()) {
                return Err(ModelError::InvalidDataType(format!(
                    "Invalid value at batch[{i}][{j}]: {} (not finite)",
                    input[j]
                ))
                .into());
            }
            flattened.extend_from_slice(&input);
        }

        let input_tensor = Tensor::from_data(
            TensorData::new(flattened, [batch_size, input_size]),
            self.inner.device(),
        );

        let output = self
            .inner
            .predict(input_tensor)
            .map_err(|e| ModelError::InferenceFailed(e.to_string()))?;

        let output_size: usize = self.info.output_spec.shape.iter().product();
        let [out_batch, out_size] = output.dims();
        if out_batch != batch_size || out_size != output_size {
            return Err(ModelError::InferenceFailed(format!(
                "model returned shape [{out_batch}, {out_size}], expected [{batch_size}, {output_size}]"
            ))
            .into());
        }

        let flat: Vec<f32> = output
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| ModelError::InferenceFailed(format!("{e:?}")))?;

        Ok(flat
            .chunks_exact(output_size)
            .map(<[f32]>::to_vec)
            .collect())
    }

    pub fn get_info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn get_stats(&self) -> ModelStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn validate_input_shape(&self, input: &[f32]) -> Result<()> {
        let expected_size: usize = self.info.input_spec.shape.iter().product();
        if input.len() != expected_size {
            return Err(ModelError::InputValidation {
                expected: self.info.input_spec.shape.clone(),
                actual: vec![input.len()],
            }
            .into());
        }
        Ok(())
    }

    pub fn get_backend_info(&self) -> String {
        self.info.backend.clone()
    }

    pub fn model_type(&self) -> &'static str {
        self.model_type
    }
}

/// Load a file-based MLP model from `<path>.bpk` + `<path>.json` on the CPU backend
pub fn load_model(path: &Path) -> Result<Model> {
    load_model_with_backend(path, BackendKind::Cpu)
}

/// Load a file-based MLP model from `<path>.bpk` + `<path>.json`
pub fn load_model_with_backend(path: &Path, backend: BackendKind) -> Result<Model> {
    info!("Loading model from: {:?}", path);

    // Accept `model`, `model.bpk` or `model.json`
    let base: PathBuf = match path.extension().and_then(|e| e.to_str()) {
        Some("bpk") | Some("json") => path.with_extension(""),
        _ => path.to_path_buf(),
    };
    let weights_path = base.with_extension("bpk");

    // A `sample` path without files on disk yields a freshly initialized sample model
    if !weights_path.exists() {
        let is_sample = base
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.contains("sample"));
        if is_sample {
            info!("Creating sample MLP model");
            let container = create_sample_model()?;
            return Ok(Model::new(Box::new(MlpModel::new(container)), "mlp", 0));
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
    Ok(Model::new(
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
    Ok(Model::new(inner, "onnx", model_size_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_model() -> Model {
        let dummy = DummyModel::new("test".to_string(), vec![784], vec![10]);
        Model::new(Box::new(dummy), "dummy", 0)
    }

    #[test]
    fn test_model_info() {
        let model = get_test_model();
        let info = model.get_info();
        assert_eq!(info.model_type, "dummy");
        assert_eq!(info.backend, "cpu");
        assert_eq!(info.input_spec.shape, vec![784]);
        assert_eq!(info.output_spec.shape, vec![10]);
        assert_eq!(info.input_spec.dtype, "float32");
    }

    #[test]
    fn test_model_loading_invalid_path() {
        let result = load_model(Path::new("nonexistent_model"));
        assert!(matches!(
            result,
            Err(crate::error::FurnaceError::Model(ModelError::FileNotFound(
                _
            )))
        ));
    }

    #[test]
    fn test_inference_with_valid_input() {
        let model = get_test_model();
        let output = model.predict(vec![0.5; 784]).unwrap();
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_batch_inference_splits_outputs() {
        let model = get_test_model();
        let outputs = model.predict_batch(vec![vec![0.5; 784]; 3]).unwrap();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|o| o.len() == 10));
    }

    #[test]
    fn test_inference_with_invalid_shape() {
        let model = get_test_model();
        let result = model.predict(vec![0.5; 100]);
        assert!(matches!(
            result,
            Err(crate::error::FurnaceError::Model(
                ModelError::InputValidation { .. }
            ))
        ));
    }

    #[test]
    fn test_inference_rejects_non_finite() {
        let model = get_test_model();
        let mut input = vec![0.5; 784];
        input[42] = f32::NAN;
        assert!(matches!(
            model.predict(input),
            Err(crate::error::FurnaceError::Model(
                ModelError::InvalidDataType(_)
            ))
        ));
    }

    #[test]
    fn test_model_stats() {
        let model = get_test_model();
        let stats = model.get_stats();
        assert_eq!(stats.inference_count, 0);
        assert!(stats.last_inference_time.is_none());

        model.predict(vec![0.5; 784]).unwrap();
        let _ = model.predict(vec![0.5; 1]);

        let stats = model.get_stats();
        assert_eq!(stats.inference_count, 2);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.error_count, 1);
        assert!(stats.last_inference_time.is_some());
        assert!(stats.min_inference_time_ms.is_some());
    }

    #[test]
    fn test_real_mlp_inference_is_not_constant() {
        let model = load_model(Path::new("sample_model")).unwrap();
        let out = model.predict(vec![0.3; 784]).unwrap();
        assert_eq!(out.len(), 10);
        assert!(out.iter().any(|&v| v != out[0]));
    }

    #[test]
    fn test_built_in_model_missing_weights_file() {
        let result = load_built_in_model_with_backend(
            "resnet18",
            BackendKind::Cpu,
            Some(Path::new("does/not/exist.bpk")),
        );
        assert!(matches!(
            result,
            Err(crate::error::FurnaceError::Model(ModelError::FileNotFound(
                _
            )))
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
}
