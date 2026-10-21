use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum FurnaceError {
    Model(ModelError),

    Api(ApiError),

    /// A nervous system that cannot be built, from [`Server::nerve`](crate::Server::nerve) or
    /// a receptor or effector's constructor
    Nerve(crate::nerve::BuildError),
}

impl FurnaceError {
    /// Whether the request is at fault (its inputs do not fit the model), rather than the
    /// server: a 400 rather than a 500
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::Model(
                ModelError::InputValidation { .. }
                    | ModelError::EmptyBatch
                    | ModelError::BatchTooLarge { .. }
                    | ModelError::InvalidDataType(_)
                    | ModelError::InvalidInput(_)
                    | ModelError::InvalidArgument { .. }
            )
        )
    }
}

// Each variant is its inner error, message and source alike
impl fmt::Display for FurnaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => fmt::Display::fmt(e, f),
            Self::Api(e) => fmt::Display::fmt(e, f),
            Self::Nerve(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl std::error::Error for FurnaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => e.source(),
            Self::Api(e) => e.source(),
            Self::Nerve(e) => e.source(),
        }
    }
}

impl From<ModelError> for FurnaceError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<ApiError> for FurnaceError {
    fn from(e: ApiError) -> Self {
        Self::Api(e)
    }
}

impl From<crate::nerve::BuildError> for FurnaceError {
    fn from(e: crate::nerve::BuildError) -> Self {
        Self::Nerve(e)
    }
}

#[derive(Debug)]
pub enum ModelError {
    LoadFailed {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    InvalidFormat(String),

    FileNotFound(PathBuf),

    InferenceFailed(String),

    /// A simple model's input with the wrong number of values
    InputValidation {
        /// Items in the request
        batch: usize,
        /// One item's shape
        item: Vec<usize>,
        /// Values received
        got: usize,
    },

    /// A request with no item
    EmptyBatch,

    /// A request with more items than one request may carry
    BatchTooLarge {
        size: usize,
        max: usize,
    },

    InvalidDataType(String),

    /// A named input tensor that does not fit the model (multi-tensor models)
    InvalidInput(String),

    InvalidArgument {
        arg: String,
        value: String,
        reason: String,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadFailed { path, source } => {
                write!(f, "Failed to load model from {}: {source}", path.display())
            }
            Self::InvalidFormat(m) => write!(f, "Invalid model format: {m}"),
            Self::FileNotFound(path) => write!(f, "Model file not found: {}", path.display()),
            Self::InferenceFailed(m) => write!(f, "Inference failed: {m}"),
            Self::InputValidation { batch, item, got } => write!(
                f,
                "expected {} values (batch {batch} × {item:?}), got {got}",
                batch * item.iter().product::<usize>()
            ),
            Self::EmptyBatch => {
                f.write_str("a request needs at least one item, got an empty batch")
            }
            Self::BatchTooLarge { size, max } => write!(
                f,
                "batch size {size} exceeds the limit of {max} items per request"
            ),
            Self::InvalidDataType(m) => write!(f, "Invalid input data type: {m}"),
            Self::InvalidInput(m) => write!(f, "Invalid input: {m}"),
            Self::InvalidArgument { arg, value, reason } => {
                write!(f, "Invalid argument '{arg}': {value} - {reason}")
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LoadFailed { source, .. } => Some(&**source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ApiError {
    ServerStartup(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerStartup(m) => write!(f, "Server startup failed: {m}"),
        }
    }
}

impl std::error::Error for ApiError {}

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
            timestamp: crate::time::rfc3339_now(),
        }
    }
}
