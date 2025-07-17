pub mod containers;
pub mod controller;
pub mod load_generator;
pub mod metrics;
pub mod reports;

pub use containers::{ContainerManager, ServerType};
pub use controller::BenchmarkController;
pub use load_generator::{LoadGenerator, LoadPattern};
pub use metrics::{LatencyStats, MetricsCollector, ThroughputStats};
pub use reports::ReportGenerator;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("Container management error: {0}")]
    Container(#[from] bollard::errors::Error),

    #[error("Load generation error: {message}")]
    LoadGeneration { message: String },

    #[error("Metrics collection error: {message}")]
    Metrics { message: String },

    #[error("Report generation error: {message}")]
    Report { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Environment validation error: {message}")]
    Environment { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Plotting error: {0}")]
    Plotting(String),
}

pub type Result<T> = std::result::Result<T, BenchmarkError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub servers: Vec<ServerConfig>,
    pub models: Vec<ModelConfig>,
    pub load_patterns: Vec<LoadPattern>,
    pub duration: Duration,
    pub warmup_duration: Duration,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub server_type: ServerType,
    pub image: String,
    pub port: u16,
    pub environment: HashMap<String, String>,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub size_category: ModelSize,
    pub model_type: ModelType,
    pub input_shape: Vec<usize>,
    pub formats: HashMap<ServerType, String>, // Path to model file for each server
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_limit: String,
    pub memory_limit: String,
    pub cpu_request: Option<String>,
    pub memory_request: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSize {
    Small,  // < 10MB
    Medium, // 10-100MB
    Large,  // > 100MB
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    ImageClassification,
    TextClassification,
    Regression,
    ObjectDetection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub timestamp: DateTime<Utc>,
    pub config: BenchmarkConfig,
    pub server_results: HashMap<String, ServerResults>,
    pub comparison_analysis: ComparisonAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerResults {
    pub server_id: String,
    pub server_type: ServerType,
    pub latency_stats: LatencyStats,
    pub throughput_stats: ThroughputStats,
    pub resource_stats: crate::benchmark::metrics::ResourceStats,
    pub error_stats: ErrorStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub error_rate: f64,
    pub timeout_count: u64,
    pub connection_errors: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonAnalysis {
    pub performance_ranking: Vec<PerformanceRank>,
    pub relative_improvements: HashMap<String, RelativeImprovement>,
    pub statistical_significance: HashMap<String, StatisticalTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRank {
    pub server_id: String,
    pub rank: usize,
    pub score: f64,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeImprovement {
    pub baseline_server: String,
    pub comparison_server: String,
    pub latency_improvement: f64,
    pub throughput_improvement: f64,
    pub memory_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_type: String,
    pub p_value: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
}
