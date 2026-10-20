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

impl FurnaceError {
    /// Whether the request is at fault (its inputs do not fit the model), rather than the
    /// server: a 400 rather than a 500
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::Model(
                ModelError::InputValidation { .. }
                    | ModelError::InvalidDataType(_)
                    | ModelError::InvalidInput(_)
                    | ModelError::InvalidArgument { .. }
            )
        )
    }
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

    #[error("Invalid input data type: {0}")]
    InvalidDataType(String),

    /// A named input tensor that does not fit the model (multi-tensor models)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid argument '{arg}': {value} - {reason}")]
    InvalidArgument {
        arg: String,
        value: String,
        reason: String,
    },
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Server startup failed: {0}")]
    ServerStartup(String),
}

#[derive(Debug, Error)]
pub enum CliError {
    #[error("Invalid argument '{arg}': {value} - {reason}")]
    InvalidArgument {
        arg: String,
        value: String,
        reason: String,
    },

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
