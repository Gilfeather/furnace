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
