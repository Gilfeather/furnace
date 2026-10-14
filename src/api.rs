use axum::http::{
    HeaderValue,
    header::{X_CONTENT_TYPE_OPTIONS, X_FRAME_OPTIONS},
};
use axum::{
    Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::signal;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{error, info, warn};

use crate::error::{ApiError, FurnaceError, ModelError, Result};
use crate::model::{MAX_BATCH_SIZE, Model, ModelInfo, ModelStats};

#[derive(Debug, Deserialize, Serialize)]
pub struct PredictRequest {
    #[serde(flatten)]
    pub input_data: PredictInputData,
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
    pub backend: String,
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

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Maximum number of inferences running at once; extra requests get 503
    pub max_concurrent_requests: usize,
    /// Maximum request body size in bytes
    pub max_body_bytes: usize,
    /// Per-request inference timeout; exceeded requests get 504
    pub request_timeout: Option<Duration>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            max_concurrent_requests: 100,
            // A single 224x224 RGB image is ~150k floats, i.e. 1-3 MB of JSON
            max_body_bytes: 100 * 1024 * 1024,
            request_timeout: None,
        }
    }
}

type AppState = Arc<ServerState>;

struct ServerState {
    model: Arc<Model>,
    start_time: SystemTime,
    inference_slots: Arc<Semaphore>,
    request_timeout: Option<Duration>,
}

impl ServerState {
    fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().unwrap_or_default().as_secs()
    }
}

/// Build the application router with all middleware applied
pub fn router(model: Model, config: &ServerConfig) -> Router {
    let state = Arc::new(ServerState {
        model: Arc::new(model),
        start_time: SystemTime::now(),
        inference_slots: Arc::new(Semaphore::new(config.max_concurrent_requests)),
        request_timeout: config.request_timeout,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/healthz", get(health_check))
        .route("/health", get(health_check))
        .route("/predict", post(predict))
        .route("/model/info", get(model_info))
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .layer(SetResponseHeaderLayer::overriding(
            X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            X_FRAME_OPTIONS,
            HeaderValue::from_static("DENY"),
        ))
        .layer(cors)
        .with_state(state)
}

pub async fn start_server(host: &str, port: u16, model: Model) -> Result<()> {
    let config = ServerConfig {
        host: host.to_string(),
        port,
        ..ServerConfig::default()
    };
    start_server_with_config(config, model).await
}

pub async fn start_server_with_config(config: ServerConfig, model: Model) -> Result<()> {
    let app = router(model, &config);
    info!(
        "🚦 Concurrency limit set to {} inferences",
        config.max_concurrent_requests
    );

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| ApiError::ServerStartup(format!("Failed to bind to {bind_addr}: {e}")))?;

    info!("✅ Server running on http://{}", bind_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| ApiError::ServerStartup(format!("Server error: {e}")))?;

    info!("🛑 Server stopped gracefully");
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
        _ = ctrl_c => info!("Received Ctrl+C signal"),
        _ = terminate => info!("Received SIGTERM signal"),
    }
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        backend: state.model.get_backend_info(),
        uptime_seconds: state.uptime_seconds(),
        timestamp: Utc::now().to_rfc3339(),
    })
}

async fn predict(
    State(state): State<AppState>,
    payload: std::result::Result<Json<PredictRequest>, JsonRejection>,
) -> Response {
    let start_time = Instant::now();

    let Json(payload) = match payload {
        Ok(p) => p,
        Err(rejection) => {
            warn!("Rejected request body: {}", rejection.body_text());
            return create_error_response(
                rejection.status(),
                "INVALID_REQUEST",
                &rejection.body_text(),
                None,
            );
        }
    };

    let (inputs, single) = match payload.input_data {
        PredictInputData::Single { input } => (vec![input], true),
        PredictInputData::Batch { inputs } => (inputs, false),
    };

    if let Err(response) = validate_inputs(&state.model, &inputs) {
        return *response;
    }
    let batch_size = inputs.len();

    // Backpressure: reject immediately instead of queueing when all slots are busy
    let permit = match state.inference_slots.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            warn!("All inference slots busy, rejecting request");
            return create_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "SERVER_BUSY",
                "Too many concurrent requests, retry later",
                None,
            );
        }
    };

    // Inference is CPU/GPU-bound; keep it off the async runtime threads
    let model = state.model.clone();
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        model.predict_batch(inputs)
    });

    let joined = match state.request_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, task).await {
            Ok(joined) => joined,
            Err(_) => {
                warn!("Inference exceeded timeout of {:?}", timeout);
                return create_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    "INFERENCE_TIMEOUT",
                    &format!("Inference exceeded timeout of {}ms", timeout.as_millis()),
                    None,
                );
            }
        },
        None => task.await,
    };

    let result = match joined {
        Ok(result) => result,
        Err(e) => {
            error!("Inference task failed: {}", e);
            return create_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INFERENCE_FAILED",
                "Inference task panicked",
                None,
            );
        }
    };

    match result {
        Ok(mut outputs) => {
            let inference_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            let output_data = if single {
                PredictOutputData::Single {
                    output: outputs.pop().unwrap_or_default(),
                }
            } else {
                PredictOutputData::Batch { outputs }
            };

            info!(
                "Prediction completed in {:.2}ms, batch size: {}",
                inference_time_ms, batch_size
            );
            (
                StatusCode::OK,
                Json(PredictResponse {
                    output_data,
                    status: "success".to_string(),
                    inference_time_ms,
                    timestamp: Utc::now().to_rfc3339(),
                    batch_size,
                }),
            )
                .into_response()
        }
        Err(e) => {
            error!("Prediction error: {}", e);
            let (status_code, error_code) = match &e {
                FurnaceError::Model(ModelError::InputValidation { .. }) => {
                    (StatusCode::BAD_REQUEST, "INPUT_VALIDATION_FAILED")
                }
                FurnaceError::Model(ModelError::InvalidDataType(_)) => {
                    (StatusCode::BAD_REQUEST, "INVALID_INPUT_VALUE")
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

/// Cheap request-level checks done before taking an inference slot
fn validate_inputs(model: &Model, inputs: &[Vec<f32>]) -> std::result::Result<(), Box<Response>> {
    if inputs.is_empty() || inputs.iter().any(Vec::is_empty) {
        return Err(Box::new(create_error_response(
            StatusCode::BAD_REQUEST,
            "INVALID_INPUT",
            "Input cannot be empty",
            None,
        )));
    }

    if inputs.len() > MAX_BATCH_SIZE {
        return Err(Box::new(create_error_response(
            StatusCode::BAD_REQUEST,
            "BATCH_TOO_LARGE",
            &format!(
                "Batch size {} exceeds maximum allowed batch size {MAX_BATCH_SIZE}",
                inputs.len()
            ),
            None,
        )));
    }

    for (i, input) in inputs.iter().enumerate() {
        if let Err(e) = model.validate_input_shape(input) {
            return Err(Box::new(create_error_response(
                StatusCode::BAD_REQUEST,
                "INPUT_VALIDATION_FAILED",
                &e.to_string(),
                Some(serde_json::json!({
                    "expected_shape": model.get_info().input_spec.shape,
                    "received_size": input.len(),
                    "batch_item": i
                })),
            )));
        }
    }

    Ok(())
}

async fn model_info(State(state): State<AppState>) -> Json<ModelInfoResponse> {
    Json(ModelInfoResponse {
        model_info: state.model.get_info().clone(),
        stats: state.model.get_stats(),
        timestamp: Utc::now().to_rfc3339(),
    })
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
    use crate::model::DummyModel;
    use axum::{body::Body, http::Request};
    use tower::util::ServiceExt;

    fn dummy_model() -> Model {
        let dummy = DummyModel::new("test_model".to_string(), vec![784], vec![10]);
        Model::new(Box::new(dummy), "dummy", 0)
    }

    fn test_app() -> Router {
        router(dummy_model(), &ServerConfig::default())
    }

    async fn send(app: Router, request: Request<Body>) -> (StatusCode, axum::body::Bytes) {
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, body)
    }

    fn predict_request(body: String) -> Request<Body> {
        Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn test_health_check_endpoints() {
        for path in ["/healthz", "/health"] {
            let request = Request::builder().uri(path).body(Body::empty()).unwrap();
            let (status, body) = send(test_app(), request).await;
            assert_eq!(status, StatusCode::OK);
            let health: HealthResponse = serde_json::from_slice(&body).unwrap();
            assert_eq!(health.status, "healthy");
            assert!(health.model_loaded);
            assert_eq!(health.backend, "cpu");
        }
    }

    #[tokio::test]
    async fn test_security_headers_present() {
        let request = Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        let response = test_app().oneshot(request).await.unwrap();
        assert_eq!(response.headers()[X_CONTENT_TYPE_OPTIONS], "nosniff");
        assert_eq!(response.headers()[X_FRAME_OPTIONS], "DENY");
    }

    #[tokio::test]
    async fn test_model_info_endpoint() {
        let request = Request::builder()
            .uri("/model/info")
            .body(Body::empty())
            .unwrap();
        let (status, body) = send(test_app(), request).await;
        assert_eq!(status, StatusCode::OK);
        let info: ModelInfoResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(info.model_info.input_spec.shape, vec![784]);
        assert_eq!(info.model_info.output_spec.shape, vec![10]);
        assert_eq!(info.model_info.model_type, "dummy");
    }

    #[tokio::test]
    async fn test_predict_single() {
        let body = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        let (status, body) = send(test_app(), predict_request(body)).await;
        assert_eq!(status, StatusCode::OK);
        let response: PredictResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.status, "success");
        assert_eq!(response.batch_size, 1);
        match response.output_data {
            PredictOutputData::Single { output } => assert_eq!(output.len(), 10),
            _ => panic!("Expected single output"),
        }
    }

    #[tokio::test]
    async fn test_predict_batch() {
        let body = serde_json::json!({ "inputs": vec![vec![0.5; 784]; 3] }).to_string();
        let (status, body) = send(test_app(), predict_request(body)).await;
        assert_eq!(status, StatusCode::OK);
        let response: PredictResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.batch_size, 3);
        match response.output_data {
            PredictOutputData::Batch { outputs } => assert_eq!(outputs.len(), 3),
            _ => panic!("Expected batch output"),
        }
    }

    #[tokio::test]
    async fn test_predict_accepts_body_larger_than_axum_default() {
        // ResNet-sized payloads exceed axum's 2 MB default body limit
        let dummy = DummyModel::new("big".to_string(), vec![3, 224, 224], vec![1000]);
        let app = router(
            Model::new(Box::new(dummy), "dummy", 0),
            &ServerConfig::default(),
        );
        let body =
            serde_json::json!({ "inputs": vec![vec![0.123_456_7f32; 150_528]; 2] }).to_string();
        assert!(body.len() > 2 * 1024 * 1024);
        let (status, _) = send(app, predict_request(body)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_predict_body_limit_enforced() {
        let config = ServerConfig {
            max_body_bytes: 1024,
            ..ServerConfig::default()
        };
        let app = router(dummy_model(), &config);
        let body = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        let (status, _) = send(app, predict_request(body)).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_predict_invalid_shape() {
        let body = serde_json::json!({ "input": vec![0.5; 100] }).to_string();
        let (status, body) = send(test_app(), predict_request(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "INPUT_VALIDATION_FAILED");
    }

    #[tokio::test]
    async fn test_predict_empty_input() {
        let body = serde_json::json!({ "input": [] }).to_string();
        let (status, body) = send(test_app(), predict_request(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "INVALID_INPUT");
    }

    #[tokio::test]
    async fn test_predict_malformed_json() {
        let (status, body) = send(test_app(), predict_request("{not json".to_string())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "INVALID_REQUEST");
    }

    #[tokio::test]
    async fn test_predict_rejects_when_saturated() {
        let config = ServerConfig {
            max_concurrent_requests: 0,
            ..ServerConfig::default()
        };
        let app = router(dummy_model(), &config);
        let body = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        let (status, body) = send(app, predict_request(body)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "SERVER_BUSY");
    }
}
