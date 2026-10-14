use serde::{Deserialize, Serialize};

/// Tensor specification for model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
}
