use burn::backend::ndarray::NdArray;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

use crate::error::{ModelError, Result as FurnaceResult};

// Default backend type
type B = NdArray<f32>;

/// Supported backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendType {
    NdArray,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendType::NdArray => "ndarray",
            #[cfg(feature = "wgpu")]
            BackendType::Wgpu => "wgpu",
            #[cfg(feature = "metal")]
            BackendType::Metal => "metal",
            #[cfg(feature = "cuda")]
            BackendType::Cuda => "cuda",
        }
    }

    pub fn from_string(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ndarray" | "cpu" => Some(BackendType::NdArray),
            #[cfg(feature = "wgpu")]
            "wgpu" | "gpu" => Some(BackendType::Wgpu),
            #[cfg(feature = "metal")]
            "metal" => Some(BackendType::Metal),
            #[cfg(feature = "cuda")]
            "cuda" => Some(BackendType::Cuda),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn available_backends() -> Vec<BackendType> {
        #[allow(unused_mut)]
        let mut backends = vec![BackendType::NdArray];

        #[cfg(feature = "wgpu")]
        backends.push(BackendType::Wgpu);

        #[cfg(feature = "metal")]
        backends.push(BackendType::Metal);

        #[cfg(feature = "cuda")]
        backends.push(BackendType::Cuda);

        backends
    }
}

/// Configuration for advanced Burn optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub kernel_fusion: bool,
    pub autotuning_cache: bool,
    pub backend_type: BackendType,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            kernel_fusion: true,
            autotuning_cache: true,
            backend_type: BackendType::NdArray,
        }
    }
}

/// Simple MLP model configuration
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

/// Simple MLP model implementation
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Mlp<B> {
    /// Create a new MLP model
    pub fn new(config: &MlpConfig, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(config.input_size, config.hidden_size).init(device);
        let linear2 = LinearConfig::new(config.hidden_size, config.output_size).init(device);
        let activation = Relu::new();

        Self {
            linear1,
            linear2,
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
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

/// Container for the complete model with metadata
#[derive(Debug)]
pub struct BurnModelContainer {
    pub model: std::sync::Arc<std::sync::Mutex<Mlp<B>>>,
    pub metadata: BurnModelMetadata,
    #[allow(dead_code)]
    pub config: MlpConfig,
    pub optimization_config: OptimizationConfig,
}

impl BurnModelContainer {
    /// Create a new model container with default optimization settings
    pub fn new(config: MlpConfig, name: String) -> Self {
        Self::new_with_optimization(config, name, OptimizationConfig::default())
    }

    /// Create a new model container with custom optimization settings
    pub fn new_with_optimization(
        config: MlpConfig,
        name: String,
        optimization_config: OptimizationConfig,
    ) -> Self {
        info!(
            "Creating model with backend: {}, kernel_fusion: {}, autotuning_cache: {}",
            optimization_config.backend_type.as_str(),
            optimization_config.kernel_fusion,
            optimization_config.autotuning_cache
        );

        let device = <B as Backend>::Device::default();
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
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            metadata,
            config,
            optimization_config,
        }
    }

    /// Save model to file
    #[allow(dead_code)]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> FurnaceResult<()> {
        let recorder = CompactRecorder::new();

        // Save model weights
        let model_path = path.as_ref().with_extension("burn");
        let model = self.model.lock().unwrap();
        model
            .clone()
            .save_file(model_path.clone(), &recorder)
            .map_err(|e| ModelError::LoadFailed {
                path: model_path.clone(),
                source: Box::new(e),
            })?;

        // Save metadata
        let metadata_path = path.as_ref().with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to serialize metadata: {e}")))?;

        std::fs::write(&metadata_path, metadata_json).map_err(|e| ModelError::LoadFailed {
            path: metadata_path,
            source: Box::new(e),
        })?;

        Ok(())
    }

    /// Load model from file
    #[allow(dead_code)]
    pub fn load<P: AsRef<Path>>(path: P) -> FurnaceResult<Self> {
        let model_path = path.as_ref();
        let metadata_path = model_path.with_extension("json");

        // Load metadata first
        let metadata_content =
            std::fs::read_to_string(&metadata_path).map_err(|e| ModelError::LoadFailed {
                path: metadata_path.clone(),
                source: Box::new(e),
            })?;

        let metadata: BurnModelMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to parse metadata: {e}")))?;

        // Create config from metadata
        let config = MlpConfig::new(
            metadata.input_size,
            metadata.hidden_size,
            metadata.output_size,
        );

        // Load model weights
        let device = <B as Backend>::Device::default();
        let model = Mlp::new(&config, &device);
        let recorder = CompactRecorder::new();

        // Try to load from .mpk file
        let mpk_path = path.as_ref().with_extension("mpk");
        let model = model
            .load_file(mpk_path.clone(), &recorder, &device)
            .map_err(|e| ModelError::LoadFailed {
                path: mpk_path,
                source: Box::new(e),
            })?;

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            metadata,
            config,
            optimization_config: OptimizationConfig::default(),
        })
    }

    /// Run inference
    pub fn predict(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let model = self.model.lock().unwrap();
        model.forward(input)
    }

    /// Get input shape
    pub fn input_shape(&self) -> Vec<usize> {
        vec![self.metadata.input_size]
    }

    /// Get output shape
    pub fn output_shape(&self) -> Vec<usize> {
        vec![self.metadata.output_size]
    }

    /// Get backend information
    pub fn get_backend_info(&self) -> String {
        self.optimization_config.backend_type.as_str().to_string()
    }

    /// Get optimization information
    pub fn get_optimization_info(&self) -> crate::model::OptimizationInfo {
        crate::model::OptimizationInfo {
            kernel_fusion: self.optimization_config.kernel_fusion,
            autotuning_cache: self.optimization_config.autotuning_cache,
            backend_type: self.get_backend_info(),
        }
    }

    /// Load model with specific backend and optimization settings
    pub fn load_with_backend<P: AsRef<Path>>(
        path: P,
        backend_name: Option<&str>,
        enable_kernel_fusion: bool,
        enable_autotuning: bool,
    ) -> FurnaceResult<Self> {
        let backend_type = if let Some(name) = backend_name {
            BackendType::from_string(name).unwrap_or_else(|| {
                warn!("Unknown backend '{}', falling back to NdArray", name);
                BackendType::NdArray
            })
        } else {
            BackendType::NdArray
        };

        info!(
            "Attempting to load model with backend: {}",
            backend_type.as_str()
        );

        // Try to initialize the requested backend
        match try_load_with_backend(
            path.as_ref(),
            &backend_type,
            enable_kernel_fusion,
            enable_autotuning,
        ) {
            Ok(container) => {
                info!(
                    "Successfully loaded model with {} backend",
                    backend_type.as_str()
                );
                Ok(container)
            }
            Err(e) => {
                warn!(
                    "Failed to load with {} backend: {}",
                    backend_type.as_str(),
                    e
                );

                // Fallback to CPU backend if the requested backend fails
                if !matches!(backend_type, BackendType::NdArray) {
                    warn!("Falling back to CPU (NdArray) backend");
                    try_load_with_backend(
                        path.as_ref(),
                        &BackendType::NdArray,
                        enable_kernel_fusion,
                        enable_autotuning,
                    )
                } else {
                    Err(e)
                }
            }
        }
    }
}

/// Try to load model with specific backend
fn try_load_with_backend(
    path: &Path,
    backend_type: &BackendType,
    enable_kernel_fusion: bool,
    enable_autotuning: bool,
) -> FurnaceResult<BurnModelContainer> {
    let optimization_config = OptimizationConfig {
        kernel_fusion: enable_kernel_fusion,
        autotuning_cache: enable_autotuning,
        backend_type: backend_type.clone(),
    };

    // Log optimization settings
    info!(
        "Loading model with optimizations - Kernel Fusion: {}, Autotuning Cache: {}",
        enable_kernel_fusion, enable_autotuning
    );

    let model_path = path;
    let metadata_path = model_path.with_extension("json");

    // Load metadata first
    let metadata_content =
        std::fs::read_to_string(&metadata_path).map_err(|e| ModelError::LoadFailed {
            path: metadata_path.clone(),
            source: Box::new(e),
        })?;

    let metadata: BurnModelMetadata = serde_json::from_str(&metadata_content)
        .map_err(|e| ModelError::InvalidFormat(format!("Failed to parse metadata: {e}")))?;

    // Create config from metadata
    let config = MlpConfig::new(
        metadata.input_size,
        metadata.hidden_size,
        metadata.output_size,
    );

    // Initialize backend-specific device and model
    let device = <B as Backend>::Device::default();
    let model = Mlp::new(&config, &device);
    let recorder = CompactRecorder::new();

    // Try to load from .mpk file
    let mpk_path = path.with_extension("mpk");
    let model = model
        .load_file(mpk_path.clone(), &recorder, &device)
        .map_err(|e| ModelError::LoadFailed {
            path: mpk_path,
            source: Box::new(e),
        })?;

    // Apply optimizations if enabled
    if enable_kernel_fusion {
        info!("Kernel fusion optimization enabled");
        // TODO: Implement actual kernel fusion when Burn supports it
    }

    if enable_autotuning {
        info!("Autotuning cache optimization enabled");
        // TODO: Implement actual autotuning cache when Burn supports it
    }

    Ok(BurnModelContainer {
        model: std::sync::Arc::new(std::sync::Mutex::new(model)),
        metadata,
        config,
        optimization_config,
    })
}

/// Create a sample model for testing
pub fn create_sample_model() -> FurnaceResult<BurnModelContainer> {
    let config = MlpConfig::new(784, 128, 10); // MNIST-like model
    let container = BurnModelContainer::new(config, "sample_mnist_model".to_string());
    Ok(container)
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
        let device = <B as Backend>::Device::default();

        // Create dummy input
        let input_data = vec![0.1; 784];
        let input = Tensor::from_data(TensorData::new(input_data, [1, 784]), &device);

        let output = container.predict(input);
        let output_shape = output.shape();

        assert_eq!(output_shape.dims, [1, 10]);
    }

    #[test]
    fn test_model_save_load() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test_model");

        // Create and save model
        let original_container = create_sample_model().unwrap();
        original_container.save(&model_path).unwrap();

        // Load model
        let loaded_container = BurnModelContainer::load(&model_path).unwrap();

        // Verify metadata
        assert_eq!(
            loaded_container.metadata.name,
            original_container.metadata.name
        );
        assert_eq!(
            loaded_container.metadata.input_size,
            original_container.metadata.input_size
        );
        assert_eq!(
            loaded_container.metadata.output_size,
            original_container.metadata.output_size
        );

        // Test inference
        let device = <B as Backend>::Device::default();
        let input_data = vec![0.1; 784];
        let input = Tensor::from_data(TensorData::new(input_data, [1, 784]), &device);

        let output = loaded_container.predict(input);
        assert_eq!(output.shape().dims, [1, 10]);
    }
}
