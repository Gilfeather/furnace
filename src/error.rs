use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FurnaceError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("API error: {0}")]
    Api(#[from] ApiError),

    #[error("CLI error: {0}")]
    Cli(#[from] CliError),
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Failed to load model from {path}: {source}")]
    LoadFailed {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Model file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Input validation failed: expected shape {expected:?}, got {actual:?}")]
    InputValidation {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[allow(dead_code)]
    #[error("Invalid input data type: {0}")]
    InvalidDataType(String),
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[allow(dead_code)]
    #[error("Invalid request format: {0}")]
    InvalidRequest(String),

    #[error("Server startup failed: {0}")]
    ServerStartup(String),

    #[allow(dead_code)]
    #[error("JSON parsing error: {0}")]
    JsonParsing(String),

    #[allow(dead_code)]
    #[error("Model not loaded")]
    ModelNotLoaded,
}

#[derive(Debug, Error)]
pub enum CliError {
    #[allow(dead_code)]
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Missing required argument: {0}")]
    MissingArgument(String),

    #[error("Invalid port number: {0}")]
    InvalidPort(String),

    #[error("Invalid host address: {0}")]
    InvalidHost(String),
}

pub type Result<T> = std::result::Result<T, FurnaceError>;

/// Structured error response for HTTP API
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: String,
}

impl ErrorResponse {
    pub fn new(error: String, code: String, details: Option<serde_json::Value>) -> Self {
        Self {
            error,
            code,
            details,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Convert FurnaceError to HTTP response
impl IntoResponse for FurnaceError {
    fn into_response(self) -> Response {
        let (status, error_code, message, details) = match &self {
            FurnaceError::Model(model_err) => match model_err {
                ModelError::LoadFailed { path, .. } => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "MODEL_LOAD_FAILED",
                    format!("Failed to load model from {}", path.display()),
                    Some(serde_json::json!({ "path": path })),
                ),
                ModelError::InvalidFormat(msg) => (
                    StatusCode::BAD_REQUEST,
                    "INVALID_MODEL_FORMAT",
                    msg.clone(),
                    None,
                ),
                ModelError::FileNotFound(path) => (
                    StatusCode::NOT_FOUND,
                    "MODEL_FILE_NOT_FOUND",
                    format!("Model file not found: {}", path.display()),
                    Some(serde_json::json!({ "path": path })),
                ),
                ModelError::InferenceFailed(msg) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "INFERENCE_FAILED",
                    msg.clone(),
                    None,
                ),
                ModelError::InputValidation { expected, actual } => (
                    StatusCode::BAD_REQUEST,
                    "INPUT_VALIDATION_FAILED",
                    format!(
                        "Input shape mismatch: expected {:?}, got {:?}",
                        expected, actual
                    ),
                    Some(serde_json::json!({
                        "expected_shape": expected,
                        "actual_shape": actual
                    })),
                ),
                ModelError::InvalidDataType(msg) => (
                    StatusCode::BAD_REQUEST,
                    "INVALID_DATA_TYPE",
                    msg.clone(),
                    None,
                ),
            },
            FurnaceError::Api(api_err) => match api_err {
                ApiError::InvalidRequest(msg) => (
                    StatusCode::BAD_REQUEST,
                    "INVALID_REQUEST",
                    msg.clone(),
                    None,
                ),
                ApiError::ServerStartup(msg) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "SERVER_STARTUP_FAILED",
                    msg.clone(),
                    None,
                ),
                ApiError::JsonParsing(msg) => (
                    StatusCode::BAD_REQUEST,
                    "JSON_PARSING_ERROR",
                    msg.clone(),
                    None,
                ),
                ApiError::ModelNotLoaded => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "MODEL_NOT_LOADED",
                    "Model is not loaded".to_string(),
                    None,
                ),
            },
            FurnaceError::Cli(cli_err) => match cli_err {
                CliError::InvalidArgument(msg) => (
                    StatusCode::BAD_REQUEST,
                    "INVALID_ARGUMENT",
                    msg.clone(),
                    None,
                ),
                CliError::MissingArgument(msg) => (
                    StatusCode::BAD_REQUEST,
                    "MISSING_ARGUMENT",
                    msg.clone(),
                    None,
                ),
                CliError::InvalidPort(msg) => {
                    (StatusCode::BAD_REQUEST, "INVALID_PORT", msg.clone(), None)
                }
                CliError::InvalidHost(msg) => {
                    (StatusCode::BAD_REQUEST, "INVALID_HOST", msg.clone(), None)
                }
            },
        };

        let error_response = ErrorResponse::new(message, error_code.to_string(), details);

        (status, Json(error_response)).into_response()
    }
}
