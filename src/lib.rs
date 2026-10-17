pub mod allocator;
pub mod api;
pub mod backend;
pub mod batching;
pub mod burn_model;
pub mod config;
pub mod error;
pub mod model;
pub mod models;
#[cfg(feature = "image")]
pub mod preprocess;
pub mod telemetry;
pub mod threads;
pub mod types;
pub mod weights;

pub use api::{ServerConfig, start_server, start_server_with_config};
pub use backend::BackendKind;
pub use error::{ApiError, CliError, FurnaceError, ModelError, Result};
pub use model::{Model, ModelInfo, load_built_in_model, load_model};
pub use types::TensorSpec;
