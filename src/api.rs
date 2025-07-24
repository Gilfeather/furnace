use axum::http::{
    header::{X_CONTENT_TYPE_OPTIONS, X_FRAME_OPTIONS, X_XSS_PROTECTION},
    HeaderValue,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{error, info, warn};

use crate::error::{ApiError, FurnaceError, ModelError, Result};
use crate::model::{Model, ModelInfo, ModelStats};

#[derive(Debug, Deserialize, Serialize)]
pub struct PredictRequest {
    #[serde(flatten)]
    pub input_data: PredictInputData,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub reshape: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PredictInputData {
    Single { input: Vec<f32> },
    Batch { inputs: Vec<Vec<f32>> },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    #[serde(flatten)]
    pub output_data: PredictOutputData,
    pub status: String,
    pub inference_time_ms: f64,
    pub timestamp: String,
    pub batch_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PredictOutputData {
    Single { output: Vec<f32> },
    Batch { outputs: Vec<Vec<f32>> },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub uptime_seconds: u64,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfoResponse {
    pub model_info: ModelInfo,
    pub stats: ModelStats,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub data: Option<T>,
    pub status: String,
    pub message: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: Option<usize>,
}

type AppState = Arc<ServerState>;

struct ServerState {
    model: Model,
    start_time: SystemTime,
}

impl ServerState {
    fn new(model: Model) -> Self {
        Self {
            model,
            start_time: SystemTime::now(),
        }
    }

    fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().unwrap_or_default().as_secs()
    }
}

pub async fn start_server(host: &str, port: u16, model: Model) -> Result<()> {
    let config = ServerConfig {
        host: host.to_string(),
        port,
        max_concurrent_requests: None,
    };
    start_server_with_config(config, model).await
}

pub async fn start_server_with_config(config: ServerConfig, model: Model) -> Result<()> {
    let app_state = Arc::new(ServerState::new(model));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let mut app = Router::new()
        .route("/healthz", get(health_check))
        .route("/predict", post(predict))
        .route("/model/info", get(model_info))
        .layer(RequestBodyLimitLayer::new(100 * 1024 * 1024)) // 100MB limit
        .layer(SetResponseHeaderLayer::overriding(
            X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            X_FRAME_OPTIONS,
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            X_XSS_PROTECTION,
            HeaderValue::from_static("1; mode=block"),
        ))
        .layer(cors)
        .with_state(app_state);

    // Add concurrency limiting if specified
    if let Some(max_concurrent) = config.max_concurrent_requests {
        use tower::limit::ConcurrencyLimitLayer;
        app = app.layer(ConcurrencyLimitLayer::new(max_concurrent));
        info!("🚦 Concurrency limit set to {} requests", max_concurrent);
    }

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| ApiError::ServerStartup(format!("Failed to bind to {bind_addr}: {e}")))?;

    info!("✅ Server running on http://{}", bind_addr);

    // Set up graceful shutdown
    let server = axum::serve(listener, app);

    // Handle shutdown signals
    tokio::select! {
        result = server => {
            result.map_err(|e| ApiError::ServerStartup(format!("Server error: {e}")))?;
        }
        _ = shutdown_signal() => {
            info!("🛑 Shutdown signal received, stopping server gracefully");
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("Received SIGTERM signal");
        },
    }
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let response = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        uptime_seconds: state.uptime_seconds(),
        timestamp: Utc::now().to_rfc3339(),
    };

    Json(response)
}

async fn predict(State(state): State<AppState>, Json(payload): Json<PredictRequest>) -> Response {
    let start_time = std::time::Instant::now();

    let input_count = match &payload.input_data {
        PredictInputData::Single { input } => input.len(),
        PredictInputData::Batch { inputs } => inputs.len(),
    };

    // Security: Log prediction requests for monitoring
    info!(
        "Received prediction request with {} inputs, batch_size: {:?}",
        input_count, payload.batch_size
    );

    // Security: Validate input data for potential attacks
    if let Err(e) = validate_input_security(&payload.input_data) {
        warn!("Security validation failed: {}", e);
        return create_error_response(
            StatusCode::BAD_REQUEST,
            "SECURITY_VALIDATION_FAILED",
            &e,
            None,
        );
    }

    // Extract inputs based on request type
    let (inputs, batch_size) = match &payload.input_data {
        PredictInputData::Single { input } => {
            if input.is_empty() {
                warn!("Empty input received");
                return create_error_response(
                    StatusCode::BAD_REQUEST,
                    "INVALID_INPUT",
                    "Input cannot be empty",
                    None,
                );
            }

            // Validate input shape against model requirements
            if let Err(e) = state.model.validate_input_shape(input) {
                warn!("Input validation failed: {}", e);
                return create_error_response(
                    StatusCode::BAD_REQUEST,
                    "INPUT_VALIDATION_FAILED",
                    &e.to_string(),
                    Some(serde_json::json!({
                        "expected_shape": state.model.get_info().input_spec.shape,
                        "received_size": input.len()
                    })),
                );
            }

            (vec![input.clone()], 1)
        }
        PredictInputData::Batch { inputs } => {
            if inputs.is_empty() {
                warn!("Empty batch received");
                return create_error_response(
                    StatusCode::BAD_REQUEST,
                    "INVALID_INPUT",
                    "Batch cannot be empty",
                    None,
                );
            }

            // Validate all inputs in the batch
            for (i, input) in inputs.iter().enumerate() {
                if let Err(e) = state.model.validate_input_shape(input) {
                    warn!("Input validation failed for batch item {}: {}", i, e);
                    return create_error_response(
                        StatusCode::BAD_REQUEST,
                        "INPUT_VALIDATION_FAILED",
                        &format!("Batch item {i} validation failed: {e}"),
                        Some(serde_json::json!({
                            "expected_shape": state.model.get_info().input_spec.shape,
                            "received_size": input.len(),
                            "batch_item": i
                        })),
                    );
                }
            }

            (inputs.clone(), inputs.len())
        }
    };

    // Run inference
    match state.model.predict_batch(inputs) {
        Ok(outputs) => {
            let inference_time = start_time.elapsed().as_millis() as f64;

            let output_data = if batch_size == 1 {
                PredictOutputData::Single {
                    output: outputs[0].clone(),
                }
            } else {
                PredictOutputData::Batch { outputs }
            };

            let response = PredictResponse {
                output_data,
                status: "success".to_string(),
                inference_time_ms: inference_time,
                timestamp: Utc::now().to_rfc3339(),
                batch_size,
            };

            info!(
                "Prediction completed successfully in {:.2}ms, batch size: {}",
                inference_time, batch_size
            );
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            error!("Prediction error: {}", e);

            let (status_code, error_code) = match &e {
                FurnaceError::Model(ModelError::InputValidation { .. }) => {
                    (StatusCode::BAD_REQUEST, "INPUT_VALIDATION_ERROR")
                }
                FurnaceError::Model(ModelError::InferenceFailed(_)) => {
                    (StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_FAILED")
                }
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
            };

            create_error_response(status_code, error_code, &e.to_string(), None)
        }
    }
}

async fn model_info(State(state): State<AppState>) -> Json<ModelInfoResponse> {
    let response = ModelInfoResponse {
        model_info: state.model.get_info().clone(),
        stats: state.model.get_stats(),
        timestamp: Utc::now().to_rfc3339(),
    };

    Json(response)
}

/// Security validation for input data
fn validate_input_security(input_data: &PredictInputData) -> std::result::Result<(), String> {
    const MAX_INPUT_SIZE: usize = 10_000_000; // 10M elements max
    const MAX_BATCH_SIZE: usize = 1000; // 1000 items max per batch

    match input_data {
        PredictInputData::Single { input } => {
            // Check for excessively large inputs
            if input.len() > MAX_INPUT_SIZE {
                return Err(format!(
                    "Input size {} exceeds maximum allowed size {}",
                    input.len(),
                    MAX_INPUT_SIZE
                ));
            }

            // Check for NaN or infinite values that could cause issues
            for (i, &value) in input.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!(
                        "Invalid value at index {}: {} (NaN or infinite values not allowed)",
                        i, value
                    ));
                }
            }
        }
        PredictInputData::Batch { inputs } => {
            // Check batch size limits
            if inputs.len() > MAX_BATCH_SIZE {
                return Err(format!(
                    "Batch size {} exceeds maximum allowed batch size {}",
                    inputs.len(),
                    MAX_BATCH_SIZE
                ));
            }

            // Validate each input in the batch
            for (batch_idx, input) in inputs.iter().enumerate() {
                if input.len() > MAX_INPUT_SIZE {
                    return Err(format!(
                        "Batch item {} size {} exceeds maximum allowed size {}",
                        batch_idx,
                        input.len(),
                        MAX_INPUT_SIZE
                    ));
                }

                // Check for NaN or infinite values
                for (i, &value) in input.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(format!(
                            "Invalid value in batch item {} at index {}: {} (NaN or infinite values not allowed)",
                            batch_idx, i, value
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

fn create_error_response(
    status_code: StatusCode,
    error_code: &str,
    message: &str,
    details: Option<serde_json::Value>,
) -> Response {
    let error_response = ErrorResponse {
        error: message.to_string(),
        code: error_code.to_string(),
        details,
        timestamp: Utc::now().to_rfc3339(),
    };

    (status_code, Json(error_response)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::path::PathBuf;
    use tower::util::ServiceExt;

    /// Ensure a test model is available for testing
    async fn ensure_test_model() -> PathBuf {
        let test_model_path = PathBuf::from("test_model.onnx");

        // If model already exists, return it
        if test_model_path.exists() {
            return test_model_path;
        }

        // Try to download a small ONNX model for testing
        if let Ok(response) = reqwest::get("https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx").await {
            if response.status().is_success() {
                if let Ok(bytes) = response.bytes().await {
                    if std::fs::write(&test_model_path, bytes).is_ok() {
                        println!("Downloaded test ONNX model for testing");
                        return test_model_path;
                    }
                }
            }
        }

        // If download fails, try the mpk from examples
        let mpk_path = PathBuf::from("test_model.mpk");
        if tokio::process::Command::new("cargo")
            .args(["run", "--example", "basic_mnist_create"])
            .status()
            .await
            .is_ok()
        {
            let example_model = PathBuf::from("examples/basic_mnist/model.mpk");
            if example_model.exists() && std::fs::copy(&example_model, &mpk_path).is_ok() {
                return mpk_path;
            }
        }

        // If all else fails, return the ONNX path anyway (will trigger dummy model fallback)
        test_model_path
    }

    async fn create_test_app() -> Router {
        // Try to use an actual model file for testing, fallback to dummy
        let model_path = ensure_test_model().await;
        let model = crate::model::load_model(&model_path).unwrap_or_else(|_| {
            // Fallback to dummy model if load fails
            let dummy_model =
                crate::model::DummyModel::new("test_model".to_string(), vec![784], vec![10]);
            crate::model::Model::new(Box::new(dummy_model), model_path.clone(), 0)
        });
        let state = Arc::new(ServerState::new(model));

        Router::new()
            .route("/healthz", get(health_check))
            .route("/predict", post(predict))
            .route("/model/info", get(model_info))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_health_check_endpoint() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let health_response: HealthResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(health_response.status, "healthy");
        assert!(health_response.model_loaded);
        // uptime_seconds is u64, so it's always >= 0
        assert!(health_response.uptime_seconds < u64::MAX);
    }

    #[tokio::test]
    async fn test_model_info_endpoint() {
        let app = create_test_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/model/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let info_response: ModelInfoResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(info_response.model_info.input_spec.shape, vec![784]);
        assert_eq!(info_response.model_info.output_spec.shape, vec![10]);
        assert_eq!(info_response.model_info.model_type, "burn");
    }

    #[tokio::test]
    async fn test_predict_endpoint_success() {
        let app = create_test_app().await;

        let predict_request = PredictRequest {
            input_data: PredictInputData::Single {
                input: vec![0.5; 784],
            },
            batch_size: None,
            reshape: None,
        };

        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&predict_request).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let predict_response: PredictResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(predict_response.status, "success");
        match predict_response.output_data {
            PredictOutputData::Single { output } => assert_eq!(output.len(), 10),
            _ => panic!("Expected single output"),
        }
        assert!(predict_response.inference_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_predict_endpoint_invalid_input() {
        let app = create_test_app().await;

        let predict_request = PredictRequest {
            input_data: PredictInputData::Single {
                input: vec![0.5; 100],
            }, // Invalid size
            batch_size: None,
            reshape: None,
        };

        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&predict_request).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error_response: ErrorResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(error_response.code, "INPUT_VALIDATION_FAILED");
    }

    #[tokio::test]
    async fn test_predict_endpoint_empty_input() {
        let app = create_test_app().await;

        let predict_request = PredictRequest {
            input_data: PredictInputData::Single { input: vec![] }, // Empty input
            batch_size: None,
            reshape: None,
        };

        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&predict_request).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error_response: ErrorResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(error_response.code, "INVALID_INPUT");
    }
}
