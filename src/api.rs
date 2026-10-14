use axum::http::{
    HeaderMap, HeaderName, HeaderValue,
    header::{ACCEPT, CONTENT_TYPE, X_CONTENT_TYPE_OPTIONS, X_FRAME_OPTIONS},
};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use chrono::Utc;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};
use tokio::signal;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{error, info, warn};

use crate::error::{ApiError, FurnaceError, ModelError, Result};
use crate::model::{MAX_BATCH_SIZE, Model, ModelInfo, ModelStats};
#[cfg(feature = "image")]
use crate::preprocess::{ImageKind, image_input_size, preprocess_image};

/// Media type of the raw little-endian `f32` input/output format
pub const OCTET_STREAM: &str = "application/octet-stream";
/// Request: number of items in a binary body (omit for a single item).
/// Response: number of items in a binary body.
pub const BATCH_SIZE_HEADER: HeaderName = HeaderName::from_static("x-batch-size");
/// Binary response: per-item output shape, comma separated (e.g. `1000`)
pub const OUTPUT_SHAPE_HEADER: HeaderName = HeaderName::from_static("x-output-shape");
/// Binary response: what `inference_time_ms` holds in the JSON response
pub const INFERENCE_TIME_HEADER: HeaderName = HeaderName::from_static("x-inference-time-ms");

#[derive(Debug, Serialize)]
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

// Hand-written so the float arrays are parsed straight into `Vec`s. The derived
// `flatten` + `untagged` impl first buffers the whole document into serde's generic
// `Content` tree and then parses it a second time, which doubles the cost on
// image-sized payloads.
impl<'de> Deserialize<'de> for PredictRequest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Input,
            Inputs,
            #[serde(other)]
            Other,
        }

        struct RequestVisitor;

        impl<'de> Visitor<'de> for RequestVisitor {
            type Value = PredictRequest;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object with an `input` or `inputs` field")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<PredictRequest, A::Error> {
                let mut input_data = None;
                while let Some(field) = map.next_key()? {
                    let value = match field {
                        Field::Input => PredictInputData::Single {
                            input: map.next_value()?,
                        },
                        Field::Inputs => PredictInputData::Batch {
                            inputs: map.next_value()?,
                        },
                        Field::Other => {
                            map.next_value::<IgnoredAny>()?;
                            continue;
                        }
                    };
                    if input_data.replace(value).is_some() {
                        return Err(de::Error::custom(
                            "expected exactly one of `input` or `inputs`",
                        ));
                    }
                }
                input_data
                    .map(|input_data| PredictRequest { input_data })
                    .ok_or_else(|| de::Error::custom("missing field `input` or `inputs`"))
            }
        }

        deserializer.deserialize_map(RequestVisitor)
    }
}

/// A decoded `/predict` body: `batch_size` items back to back in one buffer
struct PredictInput {
    data: Vec<f32>,
    batch_size: usize,
    /// Answer with `output` instead of `outputs`
    single: bool,
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

/// `GET /metrics`: server-level counters for monitoring (originally contributed in #32)
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub uptime_seconds: u64,
    /// `/predict` requests received, including ones rejected before inference
    pub total_requests: u64,
    /// `/predict` requests answered with a 4xx/5xx status
    pub total_errors: u64,
    /// Requests rejected with `503 SERVER_BUSY`
    pub rejected_busy: u64,
    /// Inferences currently running
    pub in_flight_inferences: usize,
    pub max_concurrent_inferences: usize,
    /// Batches that reached the model (successful or failed)
    pub total_inferences: u64,
    pub average_inference_time_ms: f64,
    /// When the last successful inference finished
    pub last_inference_at: Option<String>,
    pub timestamp: String,
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
    max_concurrent: usize,
    request_timeout: Option<Duration>,
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    rejected_busy: AtomicU64,
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
        max_concurrent: config.max_concurrent_requests,
        request_timeout: config.request_timeout,
        total_requests: AtomicU64::new(0),
        total_errors: AtomicU64::new(0),
        rejected_busy: AtomicU64::new(0),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .expose_headers([
            BATCH_SIZE_HEADER,
            OUTPUT_SHAPE_HEADER,
            INFERENCE_TIME_HEADER,
        ]);

    Router::new()
        .route("/healthz", get(health_check))
        .route("/health", get(health_check))
        .route(
            "/predict",
            post(predict).layer(middleware::from_fn_with_state(
                state.clone(),
                count_requests,
            )),
        )
        .route("/model/info", get(model_info))
        .route("/metrics", get(metrics))
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

async fn predict(State(state): State<AppState>, request: Request) -> Response {
    // Pick the decoder before buffering the body so unsupported types fail fast
    let format = match RequestFormat::from_headers(request.headers()) {
        Some(format) => format,
        None => {
            return create_error_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "INVALID_REQUEST",
                &format!("Expected request with `Content-Type: {SUPPORTED_REQUEST_TYPES}`"),
                None,
            );
        }
    };
    #[cfg(feature = "image")]
    if let RequestFormat::Image(_) = format
        && let Err(response) = image_size(&state.model)
    {
        return *response;
    }
    let binary_response = accepts_octet_stream(request.headers());
    let headers = request.headers().clone();

    let body = match Bytes::from_request(request, &state).await {
        Ok(body) => body,
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

    let decoded = match format {
        RequestFormat::Json => decode_json(&state.model, &body),
        RequestFormat::Binary => decode_binary(&state.model, &headers, &body),
        #[cfg(feature = "image")]
        RequestFormat::Image(kind) => decode_image(&state, kind, body.clone()).await,
    };
    // Free the raw body before inference; only the decoded floats are needed now
    drop(body);
    let PredictInput {
        data,
        batch_size,
        single,
    } = match decoded {
        Ok(input) => input,
        Err(response) => return *response,
    };
    // `inference_time_ms` excludes receiving and decoding the body
    let start_time = Instant::now();

    // Backpressure: reject immediately instead of queueing when all slots are busy
    let permit = match state.inference_slots.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            warn!("All inference slots busy, rejecting request");
            return busy_response();
        }
    };

    // Inference is CPU/GPU-bound; keep it off the async runtime threads
    let model = state.model.clone();
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        model.predict_flat(data, batch_size)
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
        Ok(output) => {
            let inference_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            info!(
                "Prediction completed in {:.2}ms, batch size: {}",
                inference_time_ms, batch_size
            );
            if binary_response {
                return binary_output_response(
                    &output,
                    batch_size,
                    &state.model.get_info().output_spec.shape,
                    inference_time_ms,
                );
            }

            let output_data = if single {
                PredictOutputData::Single { output }
            } else {
                let output_size = output.len() / batch_size;
                PredictOutputData::Batch {
                    outputs: output
                        .chunks_exact(output_size)
                        .map(<[f32]>::to_vec)
                        .collect(),
                }
            };
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

fn busy_response() -> Response {
    create_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "SERVER_BUSY",
        "Too many concurrent requests, retry later",
        None,
    )
}

#[cfg(not(feature = "image"))]
const SUPPORTED_REQUEST_TYPES: &str = "application/json` or `application/octet-stream";
#[cfg(feature = "image")]
const SUPPORTED_REQUEST_TYPES: &str =
    "application/json`, `application/octet-stream`, `image/jpeg` or `image/png";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestFormat {
    Json,
    /// Raw little-endian `f32`, see [`decode_binary`]
    Binary,
    /// One encoded image, see [`decode_image`]
    #[cfg(feature = "image")]
    Image(ImageKind),
}

impl RequestFormat {
    fn from_headers(headers: &HeaderMap) -> Option<Self> {
        let content_type = headers.get(CONTENT_TYPE)?.to_str().ok()?;
        let essence = media_type_essence(content_type);
        if essence.eq_ignore_ascii_case("application/json")
            || essence.to_ascii_lowercase().ends_with("+json")
        {
            return Some(Self::Json);
        }
        if essence.eq_ignore_ascii_case(OCTET_STREAM) {
            return Some(Self::Binary);
        }
        #[cfg(feature = "image")]
        if let Some(kind) = ImageKind::from_media_type(essence) {
            return Some(Self::Image(kind));
        }
        None
    }
}

/// `type/subtype` without parameters such as `; charset=utf-8`
fn media_type_essence(value: &str) -> &str {
    value.split(';').next().unwrap_or_default().trim()
}

/// Whether the client asked for a raw `f32` response with `Accept: application/octet-stream`
fn accepts_octet_stream(headers: &HeaderMap) -> bool {
    headers
        .get_all(ACCEPT)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|item| media_type_essence(item).eq_ignore_ascii_case(OCTET_STREAM))
}

fn decode_json(model: &Model, body: &[u8]) -> std::result::Result<PredictInput, Box<Response>> {
    // serde_json directly rather than axum's `Json`, which also tracks the path of
    // every element for error messages
    let request: PredictRequest = serde_json::from_slice(body).map_err(|e| {
        let (status, message) = match e.classify() {
            serde_json::error::Category::Data => (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Failed to deserialize the JSON body into the target type: {e}"),
            ),
            _ => (
                StatusCode::BAD_REQUEST,
                format!("Failed to parse the request body as JSON: {e}"),
            ),
        };
        warn!("Rejected request body: {}", message);
        Box::new(create_error_response(
            status,
            "INVALID_REQUEST",
            &message,
            None,
        ))
    })?;

    match request.input_data {
        PredictInputData::Single { input } => {
            validate_inputs(model, std::slice::from_ref(&input))?;
            Ok(PredictInput {
                data: input,
                batch_size: 1,
                single: true,
            })
        }
        PredictInputData::Batch { inputs } => {
            validate_inputs(model, &inputs)?;
            Ok(PredictInput {
                batch_size: inputs.len(),
                data: inputs.concat(),
                single: false,
            })
        }
    }
}

/// Decode a raw body of `batch_size × input_size` little-endian `f32` values.
///
/// Without an `X-Batch-Size` header the body is one item and the response uses
/// `output`; with it the response uses `outputs`, mirroring `input` / `inputs`.
fn decode_binary(
    model: &Model,
    headers: &HeaderMap,
    body: &[u8],
) -> std::result::Result<PredictInput, Box<Response>> {
    let batch_header = headers.get(BATCH_SIZE_HEADER);
    let batch_size = match batch_header {
        None => 1,
        Some(value) => match value.to_str().ok().and_then(|s| s.trim().parse().ok()) {
            Some(n) => n,
            None => {
                return Err(Box::new(create_error_response(
                    StatusCode::BAD_REQUEST,
                    "INVALID_REQUEST",
                    "X-Batch-Size must be a non-negative integer",
                    None,
                )));
            }
        },
    };

    check_batch_len(batch_size)?;
    if body.is_empty() {
        return Err(empty_input_response());
    }

    let shape = &model.get_info().input_spec.shape;
    let input_size: usize = shape.iter().product();
    let expected_bytes = batch_size * input_size * size_of::<f32>();
    if body.len() != expected_bytes {
        return Err(Box::new(create_error_response(
            StatusCode::BAD_REQUEST,
            "INPUT_VALIDATION_FAILED",
            &format!(
                "Expected {expected_bytes} bytes ({batch_size} item(s) of shape {shape:?} as \
                 little-endian f32), got {}",
                body.len()
            ),
            Some(serde_json::json!({
                "expected_shape": shape,
                "batch_size": batch_size,
                "expected_bytes": expected_bytes,
                "received_bytes": body.len()
            })),
        )));
    }

    // Length was checked above, so there is no remainder
    let (chunks, _) = body.as_chunks::<4>();
    let data = chunks.iter().copied().map(f32::from_le_bytes).collect();
    Ok(PredictInput {
        data,
        batch_size,
        single: batch_header.is_none(),
    })
}

/// `(H, W)` of a model that takes `[3, H, W]` input, or a 415 for any other model
#[cfg(feature = "image")]
fn image_size(model: &Model) -> std::result::Result<(u32, u32), Box<Response>> {
    let shape = &model.get_info().input_spec.shape;
    image_input_size(shape).ok_or_else(|| {
        Box::new(create_error_response(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "INVALID_REQUEST",
            &format!(
                "Image input needs a model with input shape [3, H, W]; this model takes \
                 {shape:?}. Send JSON or `{OCTET_STREAM}` instead"
            ),
            Some(serde_json::json!({ "expected_shape": shape })),
        ))
    })
}

/// Decode one JPEG or PNG and apply ImageNet preprocessing (see [`crate::preprocess`]).
///
/// The response uses `output`, as for `{"input": ...}`.
#[cfg(feature = "image")]
async fn decode_image(
    state: &ServerState,
    kind: ImageKind,
    body: Bytes,
) -> std::result::Result<PredictInput, Box<Response>> {
    let (height, width) = image_size(&state.model)?;
    if body.is_empty() {
        return Err(empty_input_response());
    }

    // Decoding can allocate far more than the body size, so it shares the inference
    // slots: at most `max_concurrent_requests` decodes or inferences run at once
    let Ok(permit) = state.inference_slots.clone().try_acquire_owned() else {
        warn!("All inference slots busy, rejecting image request");
        return Err(Box::new(busy_response()));
    };
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        preprocess_image(&body, kind, height, width)
    });

    match task.await {
        Ok(Ok(data)) => Ok(PredictInput {
            data,
            batch_size: 1,
            single: true,
        }),
        Ok(Err(e)) => {
            warn!("Rejected image body: {}", e);
            Err(Box::new(create_error_response(
                StatusCode::BAD_REQUEST,
                "INVALID_IMAGE",
                &format!("Failed to decode the image: {e}"),
                None,
            )))
        }
        Err(e) => {
            error!("Image decoding task failed: {}", e);
            Err(Box::new(create_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "Image decoding task panicked",
                None,
            )))
        }
    }
}

/// Raw little-endian `f32` outputs, `batch_size × output_size` values
fn binary_output_response(
    output: &[f32],
    batch_size: usize,
    output_shape: &[usize],
    inference_time_ms: f64,
) -> Response {
    let mut body = Vec::with_capacity(size_of_val(output));
    for value in output {
        body.extend_from_slice(&value.to_le_bytes());
    }
    let shape = output_shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    (
        StatusCode::OK,
        [
            (CONTENT_TYPE, OCTET_STREAM.to_string()),
            (BATCH_SIZE_HEADER, batch_size.to_string()),
            (OUTPUT_SHAPE_HEADER, shape),
            (INFERENCE_TIME_HEADER, format!("{inference_time_ms:.3}")),
        ],
        body,
    )
        .into_response()
}

fn empty_input_response() -> Box<Response> {
    Box::new(create_error_response(
        StatusCode::BAD_REQUEST,
        "INVALID_INPUT",
        "Input cannot be empty",
        None,
    ))
}

fn check_batch_len(batch_size: usize) -> std::result::Result<(), Box<Response>> {
    if batch_size == 0 {
        return Err(empty_input_response());
    }
    if batch_size > MAX_BATCH_SIZE {
        return Err(Box::new(create_error_response(
            StatusCode::BAD_REQUEST,
            "BATCH_TOO_LARGE",
            &format!("Batch size {batch_size} exceeds maximum allowed batch size {MAX_BATCH_SIZE}"),
            None,
        )));
    }
    Ok(())
}

/// Cheap request-level checks done before taking an inference slot
fn validate_inputs(model: &Model, inputs: &[Vec<f32>]) -> std::result::Result<(), Box<Response>> {
    check_batch_len(inputs.len())?;
    if inputs.iter().any(Vec::is_empty) {
        return Err(empty_input_response());
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

/// Count every `/predict` request and its outcome, including rejections by the body limit
async fn count_requests(State(state): State<AppState>, request: Request, next: Next) -> Response {
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    let response = next.run(request).await;
    let status = response.status();
    if status.is_client_error() || status.is_server_error() {
        state.total_errors.fetch_add(1, Ordering::Relaxed);
    }
    if status == StatusCode::SERVICE_UNAVAILABLE {
        state.rejected_busy.fetch_add(1, Ordering::Relaxed);
    }
    response
}

async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let stats = state.model.get_stats();
    Json(MetricsResponse {
        uptime_seconds: state.uptime_seconds(),
        total_requests: state.total_requests.load(Ordering::Relaxed),
        total_errors: state.total_errors.load(Ordering::Relaxed),
        rejected_busy: state.rejected_busy.load(Ordering::Relaxed),
        in_flight_inferences: state
            .max_concurrent
            .saturating_sub(state.inference_slots.available_permits()),
        max_concurrent_inferences: state.max_concurrent,
        total_inferences: stats.inference_count,
        average_inference_time_ms: stats.average_inference_time_ms,
        last_inference_at: stats.last_inference_time.map(|t| t.to_rfc3339()),
        timestamp: Utc::now().to_rfc3339(),
    })
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

    async fn get_metrics(app: Router) -> MetricsResponse {
        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let (status, body) = send(app, request).await;
        assert_eq!(status, StatusCode::OK);
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn test_metrics_counts_requests_and_errors() {
        let app = test_app();
        let initial = get_metrics(app.clone()).await;
        assert_eq!(initial.total_requests, 0);
        assert_eq!(initial.max_concurrent_inferences, 100);
        assert!(initial.last_inference_at.is_none());

        let ok = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        let (status, _) = send(app.clone(), predict_request(ok)).await;
        assert_eq!(status, StatusCode::OK);
        let bad = serde_json::json!({ "input": vec![0.5; 3] }).to_string();
        let (status, _) = send(app.clone(), predict_request(bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let m = get_metrics(app).await;
        assert_eq!(m.total_requests, 2);
        assert_eq!(m.total_errors, 1);
        assert_eq!(m.rejected_busy, 0);
        assert_eq!(m.total_inferences, 1);
        assert_eq!(m.in_flight_inferences, 0);
        assert!(m.last_inference_at.is_some());
    }

    #[tokio::test]
    async fn test_metrics_counts_busy_and_body_limit_rejections() {
        let config = ServerConfig {
            max_concurrent_requests: 0,
            max_body_bytes: 64,
            ..ServerConfig::default()
        };
        let app = router(dummy_model(), &config);
        let small = serde_json::json!({ "input": [0.5] }).to_string();
        let (status, _) = send(app.clone(), predict_request(small)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let big = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        let (status, _) = send(app.clone(), predict_request(big)).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

        let m = get_metrics(app).await;
        assert_eq!(m.total_requests, 2);
        assert_eq!(m.total_errors, 2);
        assert_eq!(m.total_inferences, 0);
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
    fn binary_request(values: &[f32], batch_size: Option<&str>) -> Request<Body> {
        let body: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut builder = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", OCTET_STREAM);
        if let Some(n) = batch_size {
            builder = builder.header(BATCH_SIZE_HEADER, n);
        }
        builder.body(Body::from(body)).unwrap()
    }

    async fn error_code(app: Router, request: Request<Body>) -> (StatusCode, String) {
        let (status, body) = send(app, request).await;
        let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
        (status, error.code)
    }

    #[test]
    fn test_predict_request_deserialize() {
        let single: PredictRequest =
            serde_json::from_str(r#"{"input": [1, 2.5, -3e-2], "extra": {"a": [1]}}"#).unwrap();
        match &single.input_data {
            PredictInputData::Single { input } => assert_eq!(input, &vec![1.0, 2.5, -0.03]),
            _ => panic!("Expected single input"),
        }

        let batch: PredictRequest = serde_json::from_str(r#"{"inputs": [[1], [2, 3]]}"#).unwrap();
        match batch.input_data {
            PredictInputData::Batch { inputs } => {
                assert_eq!(inputs, vec![vec![1.0], vec![2.0, 3.0]])
            }
            _ => panic!("Expected batch input"),
        }

        for invalid in [
            r#"{}"#,
            r#"{"input": [1], "inputs": [[1]]}"#,
            r#"{"input": [1], "input": [2]}"#,
            r#"{"input": "nope"}"#,
            r#"[1, 2]"#,
        ] {
            assert!(
                serde_json::from_str::<PredictRequest>(invalid).is_err(),
                "{invalid}"
            );
        }

        // Serialization is unchanged
        let round_trip = serde_json::to_value(&single).unwrap();
        assert_eq!(round_trip["input"], serde_json::json!([1.0, 2.5, -0.03f32]));
    }

    #[tokio::test]
    async fn test_predict_json_errors() {
        // Valid JSON, wrong structure: 422, like axum's `Json` extractor
        let body = serde_json::json!({ "data": [1, 2] }).to_string();
        let (status, code) = error_code(test_app(), predict_request(body)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(code, "INVALID_REQUEST");

        // Content type parameters are allowed
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json; charset=utf-8")
            .body(Body::from(
                serde_json::json!({ "input": vec![0.5; 784] }).to_string(),
            ))
            .unwrap();
        let (status, _) = send(test_app(), request).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_predict_unsupported_content_type() {
        for content_type in [Some("text/plain"), None] {
            let mut builder = Request::builder().uri("/predict").method("POST");
            if let Some(content_type) = content_type {
                builder = builder.header("content-type", content_type);
            }
            let request = builder.body(Body::from("{}")).unwrap();
            let (status, code) = error_code(test_app(), request).await;
            assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
            assert_eq!(code, "INVALID_REQUEST");
        }
    }

    #[tokio::test]
    async fn test_predict_binary_single() {
        let (status, body) = send(test_app(), binary_request(&[0.5; 784], None)).await;
        assert_eq!(status, StatusCode::OK);
        let response: PredictResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.batch_size, 1);
        match response.output_data {
            PredictOutputData::Single { output } => assert_eq!(output.len(), 10),
            _ => panic!("Expected single output"),
        }
    }

    #[tokio::test]
    async fn test_predict_binary_batch() {
        let request = binary_request(&[0.5; 784 * 3], Some("3"));
        let (status, body) = send(test_app(), request).await;
        assert_eq!(status, StatusCode::OK);
        let response: PredictResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.batch_size, 3);
        match response.output_data {
            PredictOutputData::Batch { outputs } => {
                assert_eq!(outputs.len(), 3);
                assert!(outputs.iter().all(|o| o.len() == 10));
            }
            _ => panic!("Expected batch output"),
        }

        // An explicit batch of one still answers with `outputs`
        let (status, body) = send(test_app(), binary_request(&[0.5; 784], Some("1"))).await;
        assert_eq!(status, StatusCode::OK);
        let response: PredictResponse = serde_json::from_slice(&body).unwrap();
        assert!(matches!(
            response.output_data,
            PredictOutputData::Batch { .. }
        ));
    }

    #[tokio::test]
    async fn test_predict_binary_response() {
        let mut request = binary_request(&[0.5; 784 * 2], Some("2"));
        request
            .headers_mut()
            .insert(ACCEPT, HeaderValue::from_static(OCTET_STREAM));
        let response = test_app().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let headers = response.headers().clone();
        assert_eq!(headers[CONTENT_TYPE], OCTET_STREAM);
        assert_eq!(headers[BATCH_SIZE_HEADER], "2");
        assert_eq!(headers[OUTPUT_SHAPE_HEADER], "10");
        let time: f64 = headers[INFERENCE_TIME_HEADER]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!(time >= 0.0);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.len(), 2 * 10 * 4);
        let (values, _) = body.as_chunks::<4>();
        assert!(values.iter().all(|v| f32::from_le_bytes(*v) == 0.5));

        // JSON requests can ask for a binary response too
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json")
            .header("accept", "text/html, application/octet-stream;q=0.9")
            .body(Body::from(
                serde_json::json!({ "input": vec![0.5; 784] }).to_string(),
            ))
            .unwrap();
        let response = test_app().oneshot(request).await.unwrap();
        assert_eq!(response.headers()[CONTENT_TYPE], OCTET_STREAM);
        assert_eq!(response.headers()[BATCH_SIZE_HEADER], "1");
    }

    #[tokio::test]
    async fn test_predict_binary_errors() {
        let cases = [
            // Body length doesn't match the model input
            (
                binary_request(&[0.5; 100], None),
                StatusCode::BAD_REQUEST,
                "INPUT_VALIDATION_FAILED",
            ),
            // Body length doesn't match X-Batch-Size
            (
                binary_request(&[0.5; 784 * 2], Some("3")),
                StatusCode::BAD_REQUEST,
                "INPUT_VALIDATION_FAILED",
            ),
            (
                binary_request(&[0.5; 784], Some("abc")),
                StatusCode::BAD_REQUEST,
                "INVALID_REQUEST",
            ),
            (
                binary_request(&[0.5; 784], Some("0")),
                StatusCode::BAD_REQUEST,
                "INVALID_INPUT",
            ),
            (
                binary_request(&[], None),
                StatusCode::BAD_REQUEST,
                "INVALID_INPUT",
            ),
            (
                binary_request(&[0.5; 784], Some("1001")),
                StatusCode::BAD_REQUEST,
                "BATCH_TOO_LARGE",
            ),
        ];
        for (request, expected_status, expected_code) in cases {
            let (status, code) = error_code(test_app(), request).await;
            assert_eq!((status, code.as_str()), (expected_status, expected_code));
        }

        // Odd byte counts are rejected, not truncated
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", OCTET_STREAM)
            .body(Body::from(vec![0u8; 784 * 4 + 1]))
            .unwrap();
        let (status, code) = error_code(test_app(), request).await;
        assert_eq!(
            (status, code.as_str()),
            (StatusCode::BAD_REQUEST, "INPUT_VALIDATION_FAILED")
        );

        // Non-finite values are caught by the model, as for JSON
        let mut values = vec![0.5; 784];
        values[7] = f32::INFINITY;
        let (status, code) = error_code(test_app(), binary_request(&values, None)).await;
        assert_eq!(
            (status, code.as_str()),
            (StatusCode::BAD_REQUEST, "INVALID_INPUT_VALUE")
        );
    }

    #[tokio::test]
    async fn test_predict_binary_body_limit_enforced() {
        let config = ServerConfig {
            max_body_bytes: 1024,
            ..ServerConfig::default()
        };
        let app = router(dummy_model(), &config);
        let (status, code) = error_code(app, binary_request(&[0.5; 784], None)).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(code, "INVALID_REQUEST");
    }

    fn image_request(content_type: &str, body: Vec<u8>) -> Request<Body> {
        Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap()
    }

    #[cfg(not(feature = "image"))]
    #[tokio::test]
    async fn test_predict_image_needs_feature() {
        let request = image_request("image/png", vec![0x89, b'P', b'N', b'G']);
        let (status, code) = error_code(test_app(), request).await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(code, "INVALID_REQUEST");
    }

    #[cfg(feature = "image")]
    mod image_input {
        use super::*;
        use image::{ImageEncoder, RgbImage, codecs::png::PngEncoder};

        fn image_model_app(config: &ServerConfig) -> Router {
            let dummy = DummyModel::new("rgb".to_string(), vec![3, 16, 16], vec![10]);
            router(Model::new(Box::new(dummy), "dummy", 0), config)
        }

        fn png(width: u32, height: u32) -> Vec<u8> {
            let image = RgbImage::from_fn(width, height, |x, y| {
                image::Rgb([(x * 7) as u8, (y * 5) as u8, 128])
            });
            let mut bytes = Vec::new();
            PngEncoder::new(&mut bytes)
                .write_image(
                    image.as_raw(),
                    width,
                    height,
                    image::ExtendedColorType::Rgb8,
                )
                .unwrap();
            bytes
        }

        #[tokio::test]
        async fn test_predict_png() {
            let app = image_model_app(&ServerConfig::default());
            let request = image_request("image/png", png(40, 30));
            let (status, body) = send(app, request).await;
            assert_eq!(status, StatusCode::OK);
            let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(response["batch_size"], 1);
            assert_eq!(response["output"].as_array().unwrap().len(), 10);
        }

        #[tokio::test]
        async fn test_predict_image_wrong_model_shape() {
            // The default test model takes [784]
            let (status, body) = send(test_app(), image_request("image/png", png(8, 8))).await;
            assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
            let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
            assert_eq!(error.code, "INVALID_REQUEST");
            assert_eq!(
                error.details.unwrap()["expected_shape"],
                serde_json::json!([784])
            );
        }

        #[tokio::test]
        async fn test_predict_invalid_image() {
            let app = image_model_app(&ServerConfig::default());
            for (content_type, body) in [
                ("image/jpeg", b"not a jpeg".to_vec()),
                // PNG bytes labelled as JPEG
                ("image/jpeg", png(8, 8)),
            ] {
                let request = image_request(content_type, body);
                let (status, code) = error_code(app.clone(), request).await;
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(code, "INVALID_IMAGE");
            }

            let (status, code) = error_code(app, image_request("image/png", Vec::new())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(code, "INVALID_INPUT");
        }

        #[tokio::test]
        async fn test_predict_image_body_limit_enforced() {
            let config = ServerConfig {
                max_body_bytes: 1024,
                ..ServerConfig::default()
            };
            let app = image_model_app(&config);
            let body = png(256, 256);
            assert!(body.len() > 1024);
            let (status, code) = error_code(app, image_request("image/png", body)).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
            assert_eq!(code, "INVALID_REQUEST");
        }
    }
}
