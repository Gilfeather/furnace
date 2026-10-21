use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Device, Tensor},
};
use burn_store::{BurnpackStore, ModuleSnapshot};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

use furnace::error::{ModelError, Result as FurnaceResult};

use crate::BackendKind;

/// Simple MLP model configuration
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

/// Simple MLP model implementation
#[derive(Module, Debug)]
pub struct Mlp {
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
}

impl Mlp {
    /// Create a new MLP model
    pub fn new(config: &MlpConfig, device: &Device) -> Self {
        Self {
            linear1: LinearConfig::new(config.input_size, config.hidden_size).init(device),
            linear2: LinearConfig::new(config.hidden_size, config.output_size).init(device),
            activation: Relu::new(),
        }
    }

    /// Forward pass
    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

/// Model metadata stored alongside the model weights
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BurnModelMetadata {
    pub name: String,
    pub version: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub model_type: String,
    pub created_at: String,
}

/// Container for the complete model with metadata.
///
/// On disk a model is a pair of files: `<name>.bpk` (Burnpack weights) and
/// `<name>.json` (metadata).
#[derive(Debug)]
pub struct BurnModelContainer {
    pub model: Mlp,
    pub metadata: BurnModelMetadata,
    pub backend: BackendKind,
    device: Device,
}

impl BurnModelContainer {
    /// Create a new, randomly initialized model container
    pub fn new(config: MlpConfig, name: String, backend: BackendKind) -> Self {
        let device = backend.device();
        let model = Mlp::new(&config, &device);

        let metadata = BurnModelMetadata {
            name,
            version: "1.0.0".to_string(),
            input_size: config.input_size,
            hidden_size: config.hidden_size,
            output_size: config.output_size,
            model_type: "mlp".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        Self {
            model,
            metadata,
            backend,
            device,
        }
    }

    /// Save model weights (`.bpk`) and metadata (`.json`)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> FurnaceResult<()> {
        let weights_path = path.as_ref().with_extension("bpk");
        let mut store = BurnpackStore::from_file(&weights_path).overwrite(true);
        self.model
            .save_into(&mut store)
            .map_err(|e| ModelError::LoadFailed {
                path: weights_path.clone(),
                source: Box::new(e),
            })?;

        let metadata_path = path.as_ref().with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to serialize metadata: {e}")))?;
        std::fs::write(&metadata_path, metadata_json).map_err(|e| ModelError::LoadFailed {
            path: metadata_path,
            source: Box::new(e),
        })?;

        Ok(())
    }

    /// Load model weights (`.bpk`) and metadata (`.json`) onto the given backend
    pub fn load<P: AsRef<Path>>(path: P, backend: BackendKind) -> FurnaceResult<Self> {
        let metadata_path = path.as_ref().with_extension("json");
        let metadata_content =
            std::fs::read_to_string(&metadata_path).map_err(|e| ModelError::LoadFailed {
                path: metadata_path.clone(),
                source: Box::new(e),
            })?;
        let metadata: BurnModelMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to parse metadata: {e}")))?;

        let config = MlpConfig::new(
            metadata.input_size,
            metadata.hidden_size,
            metadata.output_size,
        );

        let device = backend.device();
        let mut model = Mlp::new(&config, &device);

        let weights_path = path.as_ref().with_extension("bpk");
        if !weights_path.exists() {
            return Err(ModelError::FileNotFound(weights_path).into());
        }
        crate::weights::load_burnpack(&mut model, &weights_path)?;

        info!(
            "Loaded MLP model '{}' from {:?} on {} backend",
            metadata.name, weights_path, backend
        );

        Ok(Self {
            model,
            metadata,
            backend,
            device,
        })
    }

    /// Run inference
    pub fn predict(&self, input: Tensor<2>) -> Tensor<2> {
        self.model.forward(input)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get input shape
    pub fn input_shape(&self) -> Vec<usize> {
        vec![self.metadata.input_size]
    }

    /// Get output shape
    pub fn output_shape(&self) -> Vec<usize> {
        vec![self.metadata.output_size]
    }
}

/// Create a sample model for testing
pub fn create_sample_model() -> FurnaceResult<BurnModelContainer> {
    let config = MlpConfig::new(784, 128, 10); // MNIST-like model
    Ok(BurnModelContainer::new(
        config,
        "sample_mnist_model".to_string(),
        BackendKind::Cpu,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use tempfile::tempdir;

    #[test]
    fn test_model_creation() {
        let container = create_sample_model().unwrap();
        assert_eq!(container.metadata.input_size, 784);
        assert_eq!(container.metadata.output_size, 10);
        assert_eq!(container.metadata.model_type, "mlp");
    }

    #[test]
    fn test_model_forward() {
        let container = create_sample_model().unwrap();
        let input = Tensor::from_data(
            TensorData::new(vec![0.1f32; 784], [1, 784]),
            container.device(),
        );

        let output = container.predict(input);
        assert_eq!(output.dims(), [1, 10]);
    }

    #[test]
    fn test_model_save_load_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test_model");

        let original = create_sample_model().unwrap();
        original.save(&model_path).unwrap();
        assert!(model_path.with_extension("bpk").exists());

        let loaded = BurnModelContainer::load(&model_path, BackendKind::Cpu).unwrap();
        assert_eq!(loaded.metadata.name, original.metadata.name);
        assert_eq!(loaded.metadata.input_size, original.metadata.input_size);
        assert_eq!(loaded.metadata.output_size, original.metadata.output_size);

        // Weights must survive the roundtrip: identical input gives identical output
        let data = TensorData::new(vec![0.1f32; 784], [1, 784]);
        let a = original
            .predict(Tensor::from_data(data.clone(), original.device()))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let b = loaded
            .predict(Tensor::from_data(data, loaded.device()))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(a, b);
    }
}
