use serde::{Deserialize, Serialize};

/// Standardized API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub data: Option<T>,
    pub status: String,
    pub message: Option<String>,
    pub timestamp: String,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            data: Some(data),
            status: "success".to_string(),
            message: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn success_with_message(data: T, message: String) -> Self {
        Self {
            data: Some(data),
            status: "success".to_string(),
            message: Some(message),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn error(message: String) -> ApiResponse<()> {
        ApiResponse {
            data: None,
            status: "error".to_string(),
            message: Some(message),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Request structure for prediction endpoint
#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub input: Vec<f32>,
    #[serde(default)]
    pub reshape: Option<Vec<usize>>,
    #[serde(default)]
    pub batch_size: Option<usize>,
}

/// Response structure for prediction endpoint
#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub output: Vec<f32>,
    pub inference_time_ms: f64,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// Response structure for health check endpoint
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub uptime_seconds: u64,
    pub backend: Option<String>,
}

/// Response structure for model info endpoint
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    pub name: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub model_type: String,
    pub backend: String,
    pub model_size_bytes: Option<u64>,
    pub optimization_enabled: OptimizationInfo,
}

/// Information about enabled optimizations
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationInfo {
    pub kernel_fusion: bool,
    pub autotuning_cache: bool,
    pub backend_type: String,
}

/// Tensor specification for model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
}

/// Model metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub input_spec: TensorSpec,
    pub output_spec: TensorSpec,
    pub created_at: String,
    pub model_size_bytes: u64,
}
