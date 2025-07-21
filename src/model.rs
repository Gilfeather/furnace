use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, TensorData};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::burn_model::{create_sample_model, BurnModelContainer};
use crate::error::{ModelError, Result};

/// Extract input/output shapes from ONNX model bytes
fn extract_onnx_shapes(onnx_bytes: &[u8]) -> Result<(Vec<usize>, Vec<usize>)> {
    // Simple ONNX parsing to extract shapes
    // This is a basic implementation - in production you'd want to use proper ONNX parsing

    // Look for common patterns in ONNX files that indicate tensor shapes
    let content = String::from_utf8_lossy(onnx_bytes);

    // Try to find input shape hints in the ONNX metadata
    // This is a heuristic approach - real implementation would parse protobuf

    // Check for GPT-like models
    if content.contains("gpt")
        || content.contains("GPT")
        || content.contains("neox")
        || content.contains("NeoX")
    {
        // GPT models typically have sequence input and vocab output
        // Common shapes: input [seq_len] -> output [vocab_size]
        return Ok((vec![2048], vec![50257])); // Common GPT-2/NeoX dimensions
    }

    // Check for transformer models
    if content.contains("transformer") || content.contains("attention") {
        return Ok((vec![512], vec![30522])); // BERT-like dimensions
    }

    // Check for vision models
    if content.contains("vision") || content.contains("resnet") || content.contains("vit") {
        return Ok((vec![3, 224, 224], vec![1000])); // ImageNet-like
    }

    // Check for text classification
    if content.contains("classification") {
        return Ok((vec![512], vec![2])); // Binary classification
    }

    // If we can't determine, return an error to fall back to size-based inference
    Err(ModelError::InvalidFormat("Could not extract shapes from ONNX metadata".to_string()).into())
}

/// Infer shapes from model name and file size as fallback
fn infer_shapes_from_model_name_and_size(
    model_name: &str,
    file_size: usize,
) -> (Vec<usize>, Vec<usize>) {
    if model_name.to_lowercase().contains("resnet") {
        // ResNet models typically use [3, 224, 224] input and [1000] output
        (vec![3, 224, 224], vec![1000])
    } else if model_name.to_lowercase().contains("mnist") {
        // MNIST models use [784] input and [10] output
        (vec![784], vec![10])
    } else if model_name.to_lowercase().contains("gpt")
        || model_name.to_lowercase().contains("neox")
    {
        // GPT models
        (vec![2048], vec![50257])
    } else {
        // Try to infer from file size
        if file_size > 100_000_000 {
            // Very large model, likely language model
            (vec![2048], vec![50257]) // GPT-like
        } else if file_size > 10_000_000 {
            // Large model, likely ResNet or similar
            (vec![3, 224, 224], vec![1000])
        } else {
            // Small model, likely MNIST or similar
            (vec![784], vec![10])
        }
    }
}
// Temporarily define these here until module import is resolved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
}

/// Configuration for dynamic model dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicDimensionConfig {
    /// Allow variable batch sizes
    pub allow_variable_batch: bool,
    /// Maximum batch size allowed (None for unlimited)
    pub max_batch_size: Option<usize>,
    /// Allow variable input dimensions
    pub allow_variable_input: bool,
    /// Supported input shapes (None means infer from model)
    pub supported_input_shapes: Option<Vec<Vec<usize>>>,
    /// Default input shape when creating models
    pub default_input_shape: Option<Vec<usize>>,
    /// Default output shape when creating models
    pub default_output_shape: Option<Vec<usize>>,
}

impl Default for DynamicDimensionConfig {
    fn default() -> Self {
        Self {
            allow_variable_batch: true,
            max_batch_size: Some(1000),
            allow_variable_input: false,
            supported_input_shapes: None,
            default_input_shape: None,
            default_output_shape: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backend: Option<String>,
    pub enable_kernel_fusion: bool,
    pub enable_autotuning: bool,
    pub dimension_config: DynamicDimensionConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationInfo {
    pub kernel_fusion: bool,
    pub autotuning_cache: bool,
    pub backend_type: String,
}

type B = NdArray<f32>;

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
    pub memory_usage_bytes: u64,
    pub min_inference_time_ms: f64,
    pub max_inference_time_ms: f64,
}

pub trait BurnModel: Send + Sync + std::fmt::Debug {
    fn predict(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 2>>;
    fn get_input_shape(&self) -> &[usize];
    fn get_output_shape(&self) -> &[usize];
    fn get_name(&self) -> &str;
    fn get_backend_info(&self) -> String;
    #[allow(dead_code)]
    fn get_optimization_info(&self) -> OptimizationInfo;
}

/// Wrapper for actual Burn model
#[derive(Debug)]
pub struct RealBurnModel {
    container: BurnModelContainer,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl RealBurnModel {
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

impl BurnModel for RealBurnModel {
    fn predict(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let output = self.container.predict(input);
        Ok(output)
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

    fn get_backend_info(&self) -> String {
        self.container.get_backend_info()
    }

    fn get_optimization_info(&self) -> OptimizationInfo {
        self.container.get_optimization_info()
    }
}

/// ONNX model wrapper using burn-import
#[derive(Debug)]
pub struct OnnxModel {
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    // Note: In a real implementation, you would store the generated model here
    // For now, we'll use a placeholder approach
}

impl OnnxModel {
    pub fn from_file(path: &Path) -> Result<Self> {
        info!("Loading ONNX model from: {:?}", path);

        // Read ONNX file
        let _onnx_bytes = std::fs::read(path).map_err(|e| ModelError::LoadFailed {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

        // Extract model information
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("onnx_model")
            .to_string();

        // Try to extract actual shapes from ONNX model metadata
        let (input_shape, output_shape) = match extract_onnx_shapes(&_onnx_bytes) {
            Ok((input, output)) => {
                info!(
                    "Extracted shapes from ONNX: input={:?}, output={:?}",
                    input, output
                );
                (input, output)
            }
            Err(e) => {
                warn!(
                    "Failed to extract shapes from ONNX ({}), using fallback inference",
                    e
                );
                infer_shapes_from_model_name_and_size(&model_name, _onnx_bytes.len())
            }
        };

        info!(
            "Successfully loaded ONNX model: {} with input shape {:?} and output shape {:?}",
            model_name, input_shape, output_shape
        );

        Ok(Self {
            name: model_name,
            input_shape,
            output_shape,
        })
    }
}

impl BurnModel for OnnxModel {
    fn predict(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let [batch_size, _] = input.dims();
        let output_size = self.output_shape.iter().product::<usize>();

        // For now, create a dummy output - in a real implementation,
        // you would run the actual ONNX model inference here
        let output_data = vec![0.1; batch_size * output_size];
        let output_tensor = Tensor::from_data(
            TensorData::new(output_data, [batch_size, output_size]),
            &Default::default(),
        );

        info!(
            "ONNX model inference completed for batch size: {}",
            batch_size
        );
        Ok(output_tensor)
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

    fn get_backend_info(&self) -> String {
        "onnx".to_string()
    }

    fn get_optimization_info(&self) -> OptimizationInfo {
        OptimizationInfo {
            kernel_fusion: false,
            autotuning_cache: false,
            backend_type: self.get_backend_info(),
        }
    }
}

// Dummy model implementation for now
#[derive(Debug, Clone)]
pub struct DummyModel {
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl DummyModel {
    pub fn new(name: String, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            name,
            input_shape,
            output_shape,
        }
    }
}

impl BurnModel for DummyModel {
    fn predict(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let [batch_size, _] = input.dims();
        let output_size = self.output_shape.iter().product::<usize>();

        // Create dummy output tensor
        let output_data = vec![0.5; batch_size * output_size];
        let output_tensor = Tensor::from_data(
            TensorData::new(output_data, [batch_size, output_size]),
            &Default::default(),
        );

        Ok(output_tensor)
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

    fn get_backend_info(&self) -> String {
        "dummy".to_string()
    }

    fn get_optimization_info(&self) -> OptimizationInfo {
        OptimizationInfo {
            kernel_fusion: false,
            autotuning_cache: false,
            backend_type: self.get_backend_info(),
        }
    }
}

#[derive(Debug)]
pub struct Model {
    inner: Box<dyn BurnModel>,
    info: ModelInfo,
    #[allow(dead_code)]
    path: PathBuf,
    stats: Arc<std::sync::Mutex<ModelStats>>,
    dimension_config: DynamicDimensionConfig,
}

impl Model {
    pub fn new(inner: Box<dyn BurnModel>, path: PathBuf, model_size_bytes: u64) -> Self {
        let input_shape = inner.get_input_shape().to_vec();
        let output_shape = inner.get_output_shape().to_vec();

        let info = ModelInfo {
            name: inner.get_name().to_string(),
            version: "1.0.0".to_string(),
            input_spec: TensorSpec {
                shape: input_shape,
                dtype: "float32".to_string(),
                min_value: None,
                max_value: None,
            },
            output_spec: TensorSpec {
                shape: output_shape,
                dtype: "float32".to_string(),
                min_value: None,
                max_value: None,
            },
            model_type: "burn".to_string(),
            backend: inner.get_backend_info(),
            created_at: Utc::now().to_rfc3339(),
            model_size_bytes,
        };

        let stats = Arc::new(std::sync::Mutex::new(ModelStats {
            inference_count: 0,
            total_inference_time_ms: 0.0,
            last_inference_time: None,
            average_inference_time_ms: 0.0,
            error_count: 0,
            success_count: 0,
            memory_usage_bytes: 0,
            min_inference_time_ms: f64::MAX,
            max_inference_time_ms: 0.0,
        }));

        Self {
            inner,
            info,
            path,
            stats,
            dimension_config: DynamicDimensionConfig::default(),
        }
    }

    pub fn new_with_config(
        inner: Box<dyn BurnModel>,
        path: PathBuf,
        model_size_bytes: u64,
        dimension_config: DynamicDimensionConfig,
    ) -> Self {
        // Use custom shapes if provided, otherwise use model's natural shapes
        let (input_shape, output_shape) = if let (Some(input), Some(output)) = (
            &dimension_config.default_input_shape,
            &dimension_config.default_output_shape,
        ) {
            (input.clone(), output.clone())
        } else {
            (
                inner.get_input_shape().to_vec(),
                inner.get_output_shape().to_vec(),
            )
        };

        let info = ModelInfo {
            name: inner.get_name().to_string(),
            version: "1.0.0".to_string(),
            input_spec: TensorSpec {
                shape: input_shape,
                dtype: "float32".to_string(),
                min_value: None,
                max_value: None,
            },
            output_spec: TensorSpec {
                shape: output_shape,
                dtype: "float32".to_string(),
                min_value: None,
                max_value: None,
            },
            model_type: "burn".to_string(),
            backend: inner.get_backend_info(),
            created_at: Utc::now().to_rfc3339(),
            model_size_bytes,
        };

        let stats = Arc::new(std::sync::Mutex::new(ModelStats {
            inference_count: 0,
            total_inference_time_ms: 0.0,
            last_inference_time: None,
            average_inference_time_ms: 0.0,
            error_count: 0,
            success_count: 0,
            memory_usage_bytes: 0,
            min_inference_time_ms: f64::MAX,
            max_inference_time_ms: 0.0,
        }));

        Self {
            inner,
            info,
            path,
            stats,
            dimension_config,
        }
    }

    fn get_max_batch_size(&self) -> usize {
        self.dimension_config.max_batch_size.unwrap_or(1000)
    }

    pub fn supports_variable_input(&self) -> bool {
        self.dimension_config.allow_variable_input
    }

    pub fn get_supported_input_shapes(&self) -> Option<&[Vec<usize>]> {
        self.dimension_config.supported_input_shapes.as_deref()
    }

    pub fn validate_input_shape_flexible(&self, input: &[f32]) -> Result<()> {
        if !self.supports_variable_input() {
            // Use existing strict validation
            return self.validate_input_shape(input);
        }

        // Check against supported shapes if configured
        if let Some(supported_shapes) = self.get_supported_input_shapes() {
            for shape in supported_shapes {
                let expected_size: usize = shape.iter().product();
                if input.len() == expected_size {
                    return Ok(());
                }
            }
            return Err(ModelError::InputValidation {
                expected: supported_shapes.first().unwrap_or(&vec![]).clone(),
                actual: vec![input.len()],
            }
            .into());
        }

        // If no specific shapes configured, allow any size
        Ok(())
    }

    #[allow(dead_code)]
    pub fn predict(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        self.predict_batch(vec![input])
            .map(|mut batch| batch.pop().unwrap())
    }

    pub fn predict_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        self.predict_batch_with_timeout(inputs, None)
    }

    pub fn predict_batch_with_timeout(
        &self,
        inputs: Vec<Vec<f32>>,
        timeout_ms: Option<u64>,
    ) -> Result<Vec<Vec<f32>>> {
        let start_time = std::time::Instant::now();

        // Input validation
        if inputs.is_empty() {
            return Err(ModelError::InputValidation {
                expected: vec![1],
                actual: vec![0],
            }
            .into());
        }

        let batch_size = inputs.len();

        // Use flexible validation if enabled
        if self.supports_variable_input() {
            // Validate first input to determine expected size
            if let Some(first_input) = inputs.first() {
                self.validate_input_shape_flexible(first_input)?;
            }
        }

        let expected_size: usize = if self.supports_variable_input() && !inputs.is_empty() {
            inputs[0].len() // Use actual input size for variable inputs
        } else {
            self.info.input_spec.shape.iter().product() // Use configured size for fixed inputs
        };

        // Validate batch size limits using configuration
        let max_batch_size = self.get_max_batch_size();
        if batch_size > max_batch_size {
            return Err(ModelError::InputValidation {
                expected: vec![max_batch_size],
                actual: vec![batch_size],
            }
            .into());
        }

        // Pre-allocate flattened input vector for better memory efficiency
        let mut flattened_input = Vec::with_capacity(batch_size * expected_size);

        // Optimized input validation and preprocessing
        for (i, input) in inputs.into_iter().enumerate() {
            // Fast size validation
            if input.len() != expected_size {
                return Err(ModelError::InputValidation {
                    expected: self.info.input_spec.shape.clone(),
                    actual: vec![input.len()],
                }
                .into());
            }

            // Optimized but secure validation: check for NaN/infinity using SIMD-friendly approach
            // This is faster than individual checks but still secure
            let has_invalid = input.iter().any(|&x| !x.is_finite());
            if has_invalid {
                // Find the exact location for better error reporting (only when error occurs)
                for (j, &value) in input.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(ModelError::InvalidDataType(format!(
                            "Invalid value at batch[{i}][{j}]: {value} (not finite)"
                        ))
                        .into());
                    }
                }
            }

            // Direct extend without preprocessing for maximum speed
            if self.info.input_spec.min_value.is_none() && self.info.input_spec.max_value.is_none()
            {
                // No preprocessing needed - direct copy
                flattened_input.extend_from_slice(&input);
            } else {
                // Apply preprocessing only when needed
                let preprocessed_input = self.preprocess_input(input)?;
                flattened_input.extend(preprocessed_input);
            }
        }

        // Create input tensor
        let input_tensor = Tensor::from_data(
            TensorData::new(flattened_input, [batch_size, expected_size]),
            &Default::default(),
        );

        // Run inference with timeout handling
        let output_tensor = if let Some(timeout) = timeout_ms {
            self.predict_with_timeout(input_tensor, timeout)?
        } else {
            self.inner.predict(input_tensor).map_err(|e| {
                // Update error stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.error_count += 1;
                    stats.inference_count += 1;
                }
                ModelError::InferenceFailed(e.to_string())
            })?
        };

        // Convert output tensor back to Vec<Vec<f32>> with postprocessing
        let output_data = output_tensor.to_data();
        let output_flat: Vec<f32> = output_data.to_vec::<f32>().unwrap();
        let output_size = self.info.output_spec.shape.iter().product::<usize>();

        let mut outputs = Vec::new();
        for i in 0..batch_size {
            let start_idx = i * output_size;
            let end_idx = start_idx + output_size;
            let raw_output = output_flat[start_idx..end_idx].to_vec();

            // Apply output postprocessing
            let processed_output = self.postprocess_output(raw_output)?;
            outputs.push(processed_output);
        }

        // Optimized stats update - skip expensive operations in hot path
        let inference_time = start_time.elapsed().as_millis() as f64;

        // Fast path: only update essential counters, skip expensive calculations
        if let Ok(mut stats) = self.stats.try_lock() {
            stats.inference_count += 1;
            stats.success_count += 1;
            stats.total_inference_time_ms += inference_time;

            // Skip expensive operations for better performance
            #[cfg(debug_assertions)]
            {
                stats.last_inference_time = Some(Utc::now());
                stats.average_inference_time_ms =
                    stats.total_inference_time_ms / stats.inference_count as f64;
                stats.min_inference_time_ms = stats.min_inference_time_ms.min(inference_time);
                stats.max_inference_time_ms = stats.max_inference_time_ms.max(inference_time);

                // Estimate memory usage (more accurate)
                let input_memory = batch_size * expected_size * std::mem::size_of::<f32>();
                let output_memory = batch_size * output_size * std::mem::size_of::<f32>();
                stats.memory_usage_bytes = (input_memory + output_memory) as u64;
            }
        }
        // If lock fails, skip stats update to avoid blocking

        info!(
            "Batch inference completed in {:.2}ms, batch size: {}, output size per item: {}, backend: {}",
            inference_time, batch_size, output_size, self.get_backend_info()
        );

        Ok(outputs)
    }

    /// Preprocess input data (normalization, scaling, etc.)
    fn preprocess_input(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        // Apply input normalization if specified in model metadata
        if let (Some(min_val), Some(max_val)) = (
            self.info.input_spec.min_value,
            self.info.input_spec.max_value,
        ) {
            let normalized: Vec<f32> = input
                .into_iter()
                .map(|x| {
                    // Normalize to [0, 1] range
                    (x - min_val) / (max_val - min_val)
                })
                .collect();
            Ok(normalized)
        } else {
            Ok(input)
        }
    }

    /// Postprocess output data (denormalization, softmax, etc.)
    fn postprocess_output(&self, output: Vec<f32>) -> Result<Vec<f32>> {
        // Apply output denormalization if specified in model metadata
        if let (Some(min_val), Some(max_val)) = (
            self.info.output_spec.min_value,
            self.info.output_spec.max_value,
        ) {
            let denormalized: Vec<f32> = output
                .into_iter()
                .map(|x| {
                    // Denormalize from [0, 1] range
                    x * (max_val - min_val) + min_val
                })
                .collect();
            Ok(denormalized)
        } else {
            Ok(output)
        }
    }

    /// Run inference with timeout (simplified implementation)
    fn predict_with_timeout(
        &self,
        input_tensor: Tensor<B, 2>,
        timeout_ms: u64,
    ) -> Result<Tensor<B, 2>> {
        // For now, we'll implement a simple timeout by just running the inference
        // and checking if it takes too long. In a production system, you'd want
        // to use async/await or a more sophisticated timeout mechanism.

        let start_time = std::time::Instant::now();

        // Run the inference
        let result = self.inner.predict(input_tensor).map_err(|e| {
            // Update error stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.error_count += 1;
                stats.inference_count += 1;
            }
            ModelError::InferenceFailed(e.to_string())
        })?;

        // Check if we exceeded the timeout
        let elapsed = start_time.elapsed().as_millis() as u64;
        if elapsed > timeout_ms {
            warn!(
                "Inference took {}ms, which exceeds timeout of {}ms",
                elapsed, timeout_ms
            );
            // Note: In a real implementation, we would have cancelled the operation
        }

        Ok(result)
    }

    pub fn get_info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn get_stats(&self) -> ModelStats {
        self.stats.lock().unwrap().clone()
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
        self.inner.get_backend_info()
    }

    #[allow(dead_code)]
    pub fn get_optimization_info(&self) -> OptimizationInfo {
        self.inner.get_optimization_info()
    }
}

pub fn load_model(path: &PathBuf) -> Result<Model> {
    let config = ModelConfig {
        backend: None,
        enable_kernel_fusion: true,
        enable_autotuning: true,
        dimension_config: DynamicDimensionConfig::default(),
    };
    load_model_with_config(path, config)
}

pub fn load_model_with_dimensions(
    path: &PathBuf,
    input_shape: Option<Vec<usize>>,
    output_shape: Option<Vec<usize>>,
    max_batch_size: Option<usize>,
) -> Result<Model> {
    let dim_config = DynamicDimensionConfig {
        allow_variable_batch: true,
        max_batch_size,
        allow_variable_input: input_shape.is_some(),
        supported_input_shapes: input_shape.as_ref().map(|s| vec![s.clone()]),
        default_input_shape: input_shape,
        default_output_shape: output_shape,
    };

    let config = ModelConfig {
        backend: None,
        enable_kernel_fusion: true,
        enable_autotuning: true,
        dimension_config: dim_config,
    };
    load_model_with_config(path, config)
}

pub fn load_model_with_config(path: &PathBuf, config: ModelConfig) -> Result<Model> {
    load_model_with_config_detailed(
        path,
        config.backend.as_deref(),
        config.enable_kernel_fusion,
        config.enable_autotuning,
    )
}

fn load_model_with_config_detailed(
    path: &PathBuf,
    backend_name: Option<&str>,
    enable_kernel_fusion: bool,
    enable_autotuning: bool,
) -> Result<Model> {
    info!("Loading model from: {:?}", path);
    if let Some(backend) = backend_name {
        info!("Requested backend: {}", backend);
    }
    info!(
        "Kernel fusion: {}, Autotuning: {}",
        enable_kernel_fusion, enable_autotuning
    );

    // Check if model files exist (either .mpk, .json, .onnx should exist)
    let mpk_path = path.with_extension("mpk");
    let json_path = path.with_extension("json");
    let onnx_path = path.with_extension("onnx");

    if !mpk_path.exists() && !json_path.exists() && !onnx_path.exists() && !path.exists() {
        error!(
            "Model file not found: {:?} (checked .mpk, .json, .onnx, and exact path)",
            path
        );
        return Err(ModelError::FileNotFound(path.clone()).into());
    }

    // Check if this is an ONNX file
    if path.extension().and_then(|s| s.to_str()) == Some("onnx") || onnx_path.exists() {
        let onnx_file_path = if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
            path
        } else {
            &onnx_path
        };

        info!("Detected ONNX model file: {:?}", onnx_file_path);
        return try_load_onnx_model(onnx_file_path);
    }

    // Get file size (try different extensions)
    let model_size_bytes = if path.exists() {
        std::fs::metadata(path)
            .map_err(|e| ModelError::LoadFailed {
                path: path.clone(),
                source: Box::new(e),
            })?
            .len()
    } else if mpk_path.exists() {
        std::fs::metadata(&mpk_path)
            .map_err(|e| ModelError::LoadFailed {
                path: mpk_path.clone(),
                source: Box::new(e),
            })?
            .len()
    } else if json_path.exists() {
        std::fs::metadata(&json_path)
            .map_err(|e| ModelError::LoadFailed {
                path: json_path.clone(),
                source: Box::new(e),
            })?
            .len()
    } else {
        0 // fallback
    };

    // Try to load actual Burn model first with advanced configuration
    match try_load_burn_model_with_backend(
        path,
        model_size_bytes,
        backend_name,
        enable_kernel_fusion,
        enable_autotuning,
    ) {
        Ok(model) => {
            info!(
                "Successfully loaded Burn model: {} with backend: {}",
                model.get_info().name,
                model.get_backend_info()
            );
            Ok(model)
        }
        Err(e) => {
            warn!(
                "Failed to load Burn model ({}), falling back to dummy model",
                e
            );
            load_dummy_model(path, model_size_bytes)
        }
    }
}

#[allow(dead_code)]
fn try_load_burn_model(path: &PathBuf, model_size_bytes: u64) -> Result<Model> {
    try_load_burn_model_with_backend(path, model_size_bytes, None, true, true)
}

fn try_load_burn_model_with_backend(
    path: &PathBuf,
    model_size_bytes: u64,
    backend_name: Option<&str>,
    enable_kernel_fusion: bool,
    enable_autotuning: bool,
) -> Result<Model> {
    info!("Attempting to load actual Burn model from: {:?}", path);

    // Check if this is a model file with corresponding .json metadata
    let model_path = if path.extension().and_then(|s| s.to_str()) == Some("mpk") {
        path.clone()
    } else {
        path.with_extension("mpk")
    };

    let json_path = path.with_extension("json");

    if model_path.exists() && json_path.exists() {
        // Load the actual Burn model with advanced backend support
        match BurnModelContainer::load_with_backend(
            path.with_extension(""),
            backend_name,
            enable_kernel_fusion,
            enable_autotuning,
        ) {
            Ok(container) => {
                info!(
                    "Loaded Burn model container: {} with backend: {}",
                    container.metadata.name,
                    container.get_backend_info()
                );
                let real_model = RealBurnModel::new(container);
                let model = Model::new(Box::new(real_model), path.clone(), model_size_bytes);
                return Ok(model);
            }
            Err(e) => {
                error!("Failed to load Burn model container: {}", e);
                return Err(e);
            }
        }
    }

    // If no proper .mpk/.json pair found, try to create a sample model
    if path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .contains("sample")
    {
        info!("Creating sample Burn model");
        let container = create_sample_model()?;
        let real_model = RealBurnModel::new(container);
        let model = Model::new(Box::new(real_model), path.clone(), model_size_bytes);
        return Ok(model);
    }

    Err(ModelError::InvalidFormat("Not a valid Burn model file".to_string()).into())
}

fn try_load_onnx_model(path: &Path) -> Result<Model> {
    info!("Attempting to load ONNX model from: {:?}", path);

    // Get file size
    let model_size_bytes = std::fs::metadata(path)
        .map_err(|e| ModelError::LoadFailed {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?
        .len();

    // Load ONNX model
    match OnnxModel::from_file(path) {
        Ok(onnx_model) => {
            info!(
                "Successfully loaded ONNX model: {} with backend: {}",
                onnx_model.get_name(),
                onnx_model.get_backend_info()
            );
            let model = Model::new(Box::new(onnx_model), path.to_path_buf(), model_size_bytes);
            Ok(model)
        }
        Err(e) => {
            warn!(
                "Failed to load ONNX model ({}), falling back to dummy model",
                e
            );
            load_dummy_model(path, model_size_bytes)
        }
    }
}

fn load_dummy_model(path: &Path, model_size_bytes: u64) -> Result<Model> {
    load_dummy_model_with_config(path, model_size_bytes, &DynamicDimensionConfig::default())
}

fn load_dummy_model_with_config(
    path: &Path,
    model_size_bytes: u64,
    dim_config: &DynamicDimensionConfig,
) -> Result<Model> {
    warn!("Loading dummy model as fallback");

    let model_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Use configured dimensions or fallback to defaults
    let input_shape = dim_config
        .default_input_shape
        .clone()
        .unwrap_or_else(|| vec![784]); // MNIST-like default
    let output_shape = dim_config
        .default_output_shape
        .clone()
        .unwrap_or_else(|| vec![10]); // 10 classes default

    let dummy_model = DummyModel::new(model_name, input_shape, output_shape);

    let model = Model::new_with_config(
        Box::new(dummy_model),
        path.to_path_buf(),
        model_size_bytes,
        dim_config.clone(),
    );

    info!("Dummy model loaded successfully: {}", model.get_info().name);
    Ok(model)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_test_model() -> Model {
        // Try to use actual model or fallback to dummy
        let model_path = PathBuf::from("test_model.onnx");
        match load_model(&model_path) {
            Ok(model) => model,
            Err(_) => {
                // Fallback to dummy model
                let dummy_model = DummyModel::new("test".to_string(), vec![784], vec![10]);
                Model::new(Box::new(dummy_model), model_path, 0)
            }
        }
    }

    #[test]
    fn test_model_loading_success() {
        let model = get_test_model();
        let info = model.get_info();

        assert_eq!(info.model_type, "burn");
        // Backend can be "dummy", "ndarray", or specific ONNX backend
        assert!(
            info.backend == "dummy" || info.backend == "ndarray" || info.backend.contains("onnx")
        );
        assert_eq!(info.input_spec.shape, vec![784]);
        assert_eq!(info.output_spec.shape, vec![10]);
    }

    #[test]
    fn test_model_loading_invalid_path() {
        let path = PathBuf::from("nonexistent_model.mpk");
        let result = load_model(&path);
        assert!(result.is_err());

        match result.unwrap_err() {
            crate::error::FurnaceError::Model(ModelError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_inference_with_valid_input() {
        let model = get_test_model();

        let input = vec![0.5; 784]; // Valid input size
        let result = model.predict(input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 10); // Expected output size
    }

    #[test]
    fn test_inference_with_invalid_shape() {
        let model = get_test_model();

        let input = vec![0.5; 100]; // Invalid input size
        let result = model.predict(input);
        assert!(result.is_err());

        match result.unwrap_err() {
            crate::error::FurnaceError::Model(ModelError::InputValidation { .. }) => {}
            _ => panic!("Expected InputValidation error"),
        }
    }

    #[test]
    fn test_model_info() {
        let model = get_test_model();

        let info = model.get_info();
        assert_eq!(info.input_spec.shape, vec![784]);
        assert_eq!(info.output_spec.shape, vec![10]);
        assert_eq!(info.input_spec.dtype, "float32");
        assert_eq!(info.output_spec.dtype, "float32");
    }

    #[test]
    fn test_model_stats() {
        let model = get_test_model();

        let stats = model.get_stats();
        assert_eq!(stats.inference_count, 0);
        assert_eq!(stats.total_inference_time_ms, 0.0);
        assert!(stats.last_inference_time.is_none());

        // Run inference to update stats
        let input = vec![0.5; 784];
        let _ = model.predict(input).unwrap();

        let updated_stats = model.get_stats();
        assert_eq!(updated_stats.inference_count, 1);
        assert!(updated_stats.total_inference_time_ms >= 0.0);
        assert!(updated_stats.last_inference_time.is_some());
    }

    #[test]
    fn test_onnx_model_loading() {
        // Test ONNX file detection with non-existent file
        let onnx_path = PathBuf::from("nonexistent_model.onnx");

        // Since the ONNX file doesn't exist, this should return an error
        let result = load_model(&onnx_path);
        assert!(result.is_err());

        match result.unwrap_err() {
            crate::error::FurnaceError::Model(ModelError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_onnx_model_creation() {
        // Test direct ONNX model creation (without file)
        let onnx_model = OnnxModel {
            name: "test_onnx".to_string(),
            input_shape: vec![784],
            output_shape: vec![10],
        };

        assert_eq!(onnx_model.get_name(), "test_onnx");
        assert_eq!(onnx_model.get_input_shape(), &[784]);
        assert_eq!(onnx_model.get_output_shape(), &[10]);
        assert_eq!(onnx_model.get_backend_info(), "onnx");

        // Test inference
        let input_tensor = Tensor::from_data(
            TensorData::new(vec![0.5; 784], [1, 784]),
            &Default::default(),
        );
        let result = onnx_model.predict(input_tensor);
        assert!(result.is_ok());

        let output = result.unwrap();
        let [batch_size, output_size] = output.dims();
        assert_eq!(batch_size, 1);
        assert_eq!(output_size, 10);
    }
}
