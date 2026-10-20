use axum::http::{
    HeaderMap, HeaderName, HeaderValue, Method,
    header::{
        ACCEPT, ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS,
        ACCESS_CONTROL_ALLOW_ORIGIN, ACCESS_CONTROL_EXPOSE_HEADERS, ACCESS_CONTROL_REQUEST_METHOD,
        CONTENT_TYPE, ORIGIN, VARY, X_CONTENT_TYPE_OPTIONS, X_FRAME_OPTIONS,
    },
};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Path, Request, State},
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};
use tokio::signal;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{Instrument, debug, error, info, info_span, warn};

#[cfg(feature = "grpc")]
pub mod grpc;
mod v2;

use crate::access::{Access, RateLimit, Rejection, constant_time_eq, presented_key};
pub use crate::batching::BatchConfig;
use crate::error::{ApiError, FurnaceError, ModelError, Result};
use crate::model::{MAX_BATCH_SIZE, Model, ModelInfo, ModelStats};
use crate::nerve::Announced;
use crate::nerve::serve::{Refused, ServedModel as ServedNeuron, Serving};
#[cfg(feature = "image")]
use crate::preprocess::{ImageKind, image_input_size, preprocess_image};
use crate::telemetry::{Histogram, Percentiles};
use crate::tensor::{HostTensor, Values};

/// Media type of the raw little-endian `f32` input/output format
pub const OCTET_STREAM: &str = "application/octet-stream";
/// Request: number of items in a binary body (omit for a single item).
/// Response: number of items in a binary body.
pub const BATCH_SIZE_HEADER: HeaderName = HeaderName::from_static("x-batch-size");
/// Binary response: per-item output shape, comma separated (e.g. `1000`)
pub const OUTPUT_SHAPE_HEADER: HeaderName = HeaderName::from_static("x-output-shape");
/// Binary response: what `inference_time_ms` holds in the JSON response
pub const INFERENCE_TIME_HEADER: HeaderName = HeaderName::from_static("x-inference-time-ms");
/// Request and response: an ID for the request, taken from the client (up to 128 visible ASCII
/// characters) or generated, attached to every log line of the request
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

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
    /// Requests rejected with `401 UNAUTHORIZED` for a missing or unknown API key
    pub rejected_unauthorized: u64,
    /// Requests rejected with `429 RATE_LIMITED`
    pub rejected_rate_limited: u64,
    /// Requests queued for or running inference
    pub in_flight_inferences: usize,
    pub max_concurrent_inferences: usize,
    /// Requests that reached the model (successful or failed)
    pub total_inferences: u64,
    /// Forward passes run; with dynamic batching one pass serves several requests
    pub total_batches: u64,
    /// Items per forward pass, on average
    pub average_batch_size: f64,
    /// `--max-batch-size`; 1 means dynamic batching is off
    pub max_batch_size: usize,
    pub average_inference_time_ms: f64,
    /// When the last successful inference finished
    pub last_inference_at: Option<String>,
    /// `/predict` latency from arrival to response, estimated percentiles
    pub request_latency_ms: Percentiles,
    /// Successful inference latency, queueing included, estimated percentiles
    pub inference_latency_ms: Percentiles,
    /// Whether every model's warm-up inference has run (see `GET /readyz`)
    pub ready: bool,
    /// The same per model; the fields above are totals over them
    pub models: Vec<ModelMetrics>,
    /// The LLMs (feature `llm`), with their own counters; absent without any
    #[cfg(feature = "llm")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub llm: Vec<crate::llm::LlmMetrics>,
    /// The nervous system: the served models' pathways, and the neurons added with
    /// `Server::nerve`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nerve: Option<crate::nerve::NerveMetrics>,
    pub timestamp: String,
}

/// Per-model part of `GET /metrics`
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub name: String,
    pub ready: bool,
    pub inference_count: u64,
    pub error_count: u64,
    pub batch_count: u64,
    pub average_batch_size: f64,
    pub average_inference_time_ms: f64,
    pub inference_latency_ms: Percentiles,
}

/// `GET /models`
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// The model `/predict` and `/model/info` use; empty when the server serves only LLMs
    pub default_model: String,
    pub models: Vec<ModelSummary>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSummary {
    pub name: String,
    pub ready: bool,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub backend: String,
    /// Whether `POST /admin/models/{model}/reload` can reload it
    pub reloadable: bool,
}

/// `POST /admin/models/{model}/reload`
#[derive(Debug, Serialize, Deserialize)]
pub struct ReloadResponse {
    pub model: String,
    /// The weight file given in the request, if any
    pub weights: Option<String>,
    /// Loading, warming up and switching to the new model
    pub reload_ms: f64,
    pub warm_up_ms: f64,
    pub timestamp: String,
}

/// `GET /readyz`
#[derive(Debug, Serialize, Deserialize)]
pub struct ReadinessResponse {
    /// `ready`, or `warming_up` before the warm-up inference has run
    pub status: String,
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
    /// Dynamic batching of concurrent requests
    pub batching: BatchConfig,
    /// Port for the Open Inference Protocol over gRPC (feature `grpc`); `None` serves HTTP only
    pub grpc_port: Option<u16>,
    /// Enables `POST /admin/models/{model}/reload` for clients presenting this bearer token;
    /// `None` leaves the admin API off
    pub admin_token: Option<String>,
    /// When not empty, every route but the health probes needs one of these keys, as
    /// `Authorization: Bearer <key>` or `X-API-Key: <key>`
    pub api_keys: Vec<String>,
    /// Per-client rate limit; clients are their API key, or their IP address without keys
    pub rate_limit: Option<RateLimit>,
    /// Serve HTTPS (and gRPC over TLS) instead of plain text; needs the `tls` feature
    pub tls: Option<TlsSettings>,
}

/// `--tls-cert`, `--tls-key` and `--tls-client-ca`
#[derive(Debug, Clone, PartialEq)]
pub struct TlsSettings {
    /// PEM certificate chain, the server's certificate first
    pub cert: std::path::PathBuf,
    /// PEM private key (ECDSA P-256 / P-384 or Ed25519)
    pub key: std::path::PathBuf,
    /// PEM CA certificates; when set, clients must present a certificate they signed
    pub client_ca: Option<std::path::PathBuf>,
}

/// The peer address of a connection, plain or TLS, for the rate limit
#[derive(Debug, Clone, Copy)]
pub struct ClientAddr(pub std::net::SocketAddr);

impl
    axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, tokio::net::TcpListener>>
    for ClientAddr
{
    fn connect_info(stream: axum::serve::IncomingStream<'_, tokio::net::TcpListener>) -> Self {
        Self(*stream.remote_addr())
    }
}

#[cfg(feature = "tls")]
impl
    axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, crate::tls::TlsListener>>
    for ClientAddr
{
    fn connect_info(stream: axum::serve::IncomingStream<'_, crate::tls::TlsListener>) -> Self {
        Self(*stream.remote_addr())
    }
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
            batching: BatchConfig::default(),
            grpc_port: None,
            admin_token: None,
            api_keys: Vec::new(),
            rate_limit: None,
            tls: None,
        }
    }
}

type AppState = Arc<ServerState>;

/// Loads a model again for `POST /admin/models/{model}/reload`, from the given weight file or,
/// with `None`, from where it was loaded before. Blocking.
pub type Reloader = Arc<dyn Fn(Option<&std::path::Path>) -> Result<Model> + Send + Sync>;

/// A model to serve, and how to reload it (`None`: it cannot be reloaded)
pub struct ServedModel {
    pub model: Model,
    pub reload: Option<Reloader>,
}

impl From<Model> for ServedModel {
    fn from(model: Model) -> Self {
        Self {
            model,
            reload: None,
        }
    }
}

/// Serve your own models from Rust, with everything the `furnace` binary does: dynamic
/// batching, the Open Inference Protocol over HTTP (and gRPC), metrics, readiness, reloads, API
/// keys, rate limits and TLS, all configured through [`ServerConfig`].
///
/// ```ignore
/// let model = Model::from_fn("digits", BackendKind::Cpu, &[1, 28, 28], &[10],
///     move |x: Tensor<4>| net.forward(x))?;
/// Server::new().port(8080).model(model).serve().await?;
/// ```
///
/// [`Server::router`] gives the HTTP API as an axum [`Router`] instead, to mount in your own
/// application.
pub struct Server {
    config: ServerConfig,
    contents: Contents,
}

/// What a server serves
#[derive(Default)]
struct Contents {
    models: Vec<ServedModel>,
    #[cfg(feature = "llm")]
    llms: Vec<Arc<dyn crate::llm::LlmEngine>>,
    nerve: Option<crate::nerve::FabricBuilder>,
}

impl Contents {
    fn is_empty(&self) -> bool {
        #[cfg(feature = "llm")]
        if !self.llms.is_empty() {
            return false;
        }
        if self.nerve.is_some() {
            return false;
        }
        self.models.is_empty()
    }
}

impl From<Vec<ServedModel>> for Contents {
    fn from(models: Vec<ServedModel>) -> Self {
        Self {
            models,
            #[cfg(feature = "llm")]
            llms: Vec::new(),
            nerve: None,
        }
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

impl Server {
    /// A server with [`ServerConfig::default`]: `127.0.0.1:3000`, batching on
    pub fn new() -> Self {
        Self::with_config(ServerConfig::default())
    }

    pub fn with_config(config: ServerConfig) -> Self {
        Self {
            config,
            contents: Contents::default(),
        }
    }

    pub fn host(mut self, host: &str) -> Self {
        self.config.host = host.to_string();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Change any other setting
    pub fn configure(mut self, change: impl FnOnce(&mut ServerConfig)) -> Self {
        change(&mut self.config);
        self
    }

    /// Serve `model`; the first model added is the one `/predict` uses
    pub fn model(mut self, model: Model) -> Self {
        self.contents.models.push(model.into());
        self
    }

    /// Serve an LLM (feature `llm`) on `POST /v1/chat/completions`, `POST /v1/completions` and
    /// `GET /v1/models`, next to the other models: a [`Scheduled`](crate::llm::Scheduled) step
    /// model, or any [`LlmEngine`](crate::llm::LlmEngine). A request without `model` goes to the
    /// first LLM added.
    #[cfg(feature = "llm")]
    pub fn llm(mut self, engine: impl crate::llm::LlmEngine) -> Self {
        self.contents.llms.push(Arc::new(engine));
        self
    }

    /// Add neurons of your own to the server's nervous system. The server wires its models
    /// into the same fabric (an entrance `in:{model}`, a neuron `model:{model}` and the
    /// `respond` effector), so your pathways and its requests share one system. The HTTP
    /// receptors become routes, the receptors start with the server, `GET /nerve` describes
    /// every neuron, pathway and measured latency, `GET /nerve/effectors/{name}/events`
    /// streams an SSE effector's commands, and `/metrics` carries it all. A server may serve
    /// your neurons alone.
    pub fn nerve(mut self, fabric: crate::nerve::FabricBuilder) -> Self {
        self.contents.nerve = Some(fabric);
        self
    }

    /// Serve `model` and let `POST /admin/models/{name}/reload` replace it with what `reload`
    /// returns (given the request's `weights` path, if any). Needs an admin token in the config.
    pub fn reloadable_model(
        mut self,
        model: Model,
        reload: impl Fn(Option<&std::path::Path>) -> Result<Model> + Send + Sync + 'static,
    ) -> Self {
        self.contents.models.push(ServedModel {
            model,
            reload: Some(Arc::new(reload)),
        });
        self
    }

    /// The HTTP API as an axum router, to serve or mount yourself. Panics without a model.
    pub fn router(self) -> Router {
        router_with_state(server_state(self.contents, &self.config), &self.config)
    }

    /// Serve until Ctrl-C / SIGTERM
    pub async fn serve(self) -> Result<()> {
        if self.contents.is_empty() {
            return Err(ApiError::ServerStartup(
                "no model to serve; add one with Server::model".to_string(),
            )
            .into());
        }
        serve(self.config, self.contents).await
    }
}

/// What a reload replaces at once: the model and its readiness. Its pathway in the nervous
/// system stays; the model's neuron reads the current one.
struct Loaded {
    model: Arc<Model>,
    /// Set once the warm-up inference has run
    ready: Arc<AtomicBool>,
}

impl Loaded {
    /// Serve `model`, warming it up on a background thread (`/readyz` waits for it)
    fn start(model: Model) -> Arc<Self> {
        let loaded = Self::new(model);
        let model = loaded.model.clone();
        let ready = loaded.ready.clone();
        std::thread::spawn(move || {
            match model.warm_up() {
                Ok(Some(elapsed)) => info!(
                    "🔥 Warm-up inference of {} took {:.1}ms",
                    model.get_info().name,
                    elapsed.as_secs_f64() * 1e3
                ),
                Ok(None) => info!(
                    "No warm-up for {}: an input's shape is only fixed per request",
                    model.get_info().name
                ),
                Err(e) => warn!("Warm-up inference failed: {e}"),
            }
            ready.store(true, Ordering::Release);
        });
        Arc::new(loaded)
    }

    fn new(model: Model) -> Self {
        Self {
            model: Arc::new(model),
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

/// One served model, by name. Its [`Loaded`] is swapped as a whole on reload; requests already
/// holding the old one finish on it, and it is dropped with the last of them.
struct ModelSlot {
    name: String,
    current: std::sync::RwLock<Arc<Loaded>>,
    reload: Option<Reloader>,
    /// One reload at a time
    reloading: tokio::sync::Mutex<()>,
    /// Successful inferences of this model, from decoded input to output (queueing included)
    inference_latency: Histogram,
}

impl ModelSlot {
    fn get(&self) -> Arc<Loaded> {
        self.current
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn replace(&self, loaded: Arc<Loaded>) {
        *self.current.write().unwrap_or_else(|e| e.into_inner()) = loaded;
    }
}

struct ServerState {
    /// The served models; the first is the default for `/predict` and `/model/info`
    models: Vec<Arc<ModelSlot>>,
    start_time: SystemTime,
    inference_slots: Arc<Semaphore>,
    max_concurrent: usize,
    request_timeout: Option<Duration>,
    max_batch_size: usize,
    admin_token: Option<String>,
    access: Access,
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    rejected_busy: AtomicU64,
    rejected_unauthorized: AtomicU64,
    rejected_rate_limited: AtomicU64,
    /// Inference requests from arrival to response, every status
    request_latency: Histogram,
    /// Successful inferences of every model, from decoded input to output (queueing included)
    inference_latency: Histogram,
    /// The LLMs, when there are any
    #[cfg(feature = "llm")]
    llm: Option<Arc<crate::llm::LlmState>>,
    /// The nervous system: every model's pathway, and the neurons added with `Server::nerve`
    serving: Serving,
}

impl ServerState {
    fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().unwrap_or_default().as_secs()
    }

    /// `None` when the server serves only LLMs
    fn default_slot(&self) -> Option<&Arc<ModelSlot>> {
        self.models.first()
    }

    fn slot(&self, name: &str) -> Option<&Arc<ModelSlot>> {
        self.models.iter().find(|slot| slot.name == name)
    }

    /// Whether every model has run its warm-up
    fn ready(&self) -> bool {
        self.models.iter().all(|slot| slot.get().ready())
    }

    fn model_names(&self) -> Vec<&str> {
        self.models.iter().map(|slot| slot.name.as_str()).collect()
    }

    /// Record a successful inference of `slot`'s model
    fn observe_inference(&self, slot: &ModelSlot, elapsed: Duration) {
        slot.inference_latency.observe(elapsed);
        self.inference_latency.observe(elapsed);
    }
}

/// Build the application router with all middleware applied
pub fn router(model: Model, config: &ServerConfig) -> Router {
    router_with_models(vec![model.into()], config)
}

/// [`router`] for several models; the first is the default for `/predict`
pub fn router_with_models(models: Vec<ServedModel>, config: &ServerConfig) -> Router {
    router_with_state(server_state(models.into(), config), config)
}

/// The shared state of the HTTP and gRPC servers; also starts the models' warm-ups
fn server_state(contents: Contents, config: &ServerConfig) -> AppState {
    assert!(!contents.is_empty(), "at least one model to serve");
    let batching = config.batching;
    let inference_slots = Arc::new(Semaphore::new(config.max_concurrent_requests));
    #[cfg(feature = "llm")]
    let llm = (!contents.llms.is_empty()).then(|| {
        let others = contents
            .models
            .iter()
            .map(|served| served.model.get_info().name.clone())
            .collect();
        Arc::new(crate::llm::LlmState::new(
            contents.llms,
            others,
            inference_slots.clone(),
        ))
    });
    let models = contents
        .models
        .into_iter()
        .map(|served| {
            Arc::new(ModelSlot {
                name: served.model.get_info().name.clone(),
                current: std::sync::RwLock::new(Loaded::start(served.model)),
                reload: served.reload,
                reloading: tokio::sync::Mutex::new(()),
                inference_latency: Histogram::new(),
            })
        })
        .collect::<Vec<Arc<ModelSlot>>>();
    let serving = Serving::build(
        contents.nerve.unwrap_or_default(),
        models
            .iter()
            .map(|slot| {
                let model = slot.get().model.clone();
                let current = slot.clone();
                ServedNeuron {
                    name: slot.name.clone(),
                    batches: model.is_simple() || model.batches_named(),
                    source: Arc::new(move || current.get().model.clone()),
                }
            })
            .collect(),
        batching,
        config.max_concurrent_requests,
    );
    Arc::new(ServerState {
        models,
        start_time: SystemTime::now(),
        inference_slots,
        max_concurrent: config.max_concurrent_requests,
        request_timeout: config.request_timeout,
        max_batch_size: batching.max_batch_size.max(1),
        admin_token: config.admin_token.clone(),
        access: Access::new(&config.api_keys, config.rate_limit),
        total_requests: AtomicU64::new(0),
        total_errors: AtomicU64::new(0),
        rejected_busy: AtomicU64::new(0),
        rejected_unauthorized: AtomicU64::new(0),
        rejected_rate_limited: AtomicU64::new(0),
        request_latency: Histogram::new(),
        inference_latency: Histogram::new(),
        #[cfg(feature = "llm")]
        llm,
        serving,
    })
}

fn router_with_state(state: AppState, config: &ServerConfig) -> Router {
    let router = Router::new()
        .route("/healthz", get(health_check))
        .route("/health", get(health_check))
        .route("/readyz", get(readiness))
        .route(
            "/predict",
            post(predict).layer(middleware::from_fn_with_state(
                state.clone(),
                count_requests,
            )),
        )
        .route(
            "/models/{model}/predict",
            post(predict_named).layer(middleware::from_fn_with_state(
                state.clone(),
                count_requests,
            )),
        )
        .route("/model/info", get(model_info))
        .route("/models", get(list_models))
        .route("/models/{model}", get(model_info_named))
        .route("/admin/models/{model}/reload", post(reload_model))
        .route("/metrics", get(metrics))
        .merge(v2::routes(state.clone()));
    #[cfg(feature = "llm")]
    let router = match &state.llm {
        Some(llm) => router.merge(crate::llm::routes(llm.clone())),
        None => router,
    };
    let router = router.merge(crate::nerve::http::routes(state.serving.fabric().clone()));
    // Without API keys or a rate limit every request is admitted: skip the check altogether
    let router = if state.access.is_open() {
        router
    } else {
        router.layer(middleware::from_fn_with_state(state.clone(), guard))
    };
    router
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .layer(middleware::from_fn(edge))
        .with_state(state)
}

/// Health probes, open to everyone and never rate limited
const PUBLIC_PATHS: &[&str] = &[
    "/healthz",
    "/health",
    "/readyz",
    "/v2/health/live",
    "/v2/health/ready",
];

/// API keys and the rate limit (see [`crate::access`]). The admin API checks its own token.
async fn guard(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let path = request.uri().path();
    if PUBLIC_PATHS.contains(&path) {
        return next.run(request).await;
    }
    let verdict = {
        let header = |name: &str| request.headers().get(name).and_then(|v| v.to_str().ok());
        let key = presented_key(header("authorization"), header(API_KEY_HEADER));
        let ip = request
            .extensions()
            .get::<axum::extract::ConnectInfo<ClientAddr>>()
            .map(|info| info.0.0.ip());
        state.access.admit(key, ip, !path.starts_with("/admin/"))
    };
    match verdict {
        Ok(()) => next.run(request).await,
        Err(rejection) => rejection_response(&state, &rejection),
    }
}

/// Header carrying an API key, as an alternative to `Authorization: Bearer`
const API_KEY_HEADER: &str = "x-api-key";

fn rejection_response(state: &ServerState, rejection: &Rejection) -> Response {
    match rejection {
        Rejection::Unauthorized => {
            state.rejected_unauthorized.fetch_add(1, Ordering::Relaxed);
            let mut response = create_error_response(
                StatusCode::UNAUTHORIZED,
                "UNAUTHORIZED",
                "Missing or unknown API key; send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'",
                None,
            );
            response.headers_mut().insert(
                axum::http::header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer"),
            );
            response
        }
        Rejection::RateLimited(retry) => {
            state.rejected_rate_limited.fetch_add(1, Ordering::Relaxed);
            let seconds = retry.as_secs() + u64::from(retry.subsec_nanos() > 0);
            let mut response = create_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                &format!(
                    "Rate limit exceeded; retry in {} ms",
                    retry.as_millis().max(1)
                ),
                Some(serde_json::json!({ "retry_after_ms": retry.as_millis() })),
            );
            response
                .headers_mut()
                .insert(axum::http::header::RETRY_AFTER, HeaderValue::from(seconds));
            response
        }
    }
}

/// Response headers exposed to browser scripts (CORS)
const EXPOSED_HEADERS: &str = "x-batch-size, x-output-shape, x-inference-time-ms, x-request-id";

/// What every request goes through, in one layer: an ID (the client's `X-Request-Id` when it is
/// a reasonable token, else a new UUID) and a `request` span carrying it, CORS open to any
/// origin, and security headers. The ID is echoed in the response.
async fn edge(request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(&REQUEST_ID_HEADER)
        .filter(|id| {
            let id = id.as_bytes();
            !id.is_empty() && id.len() <= 128 && id.iter().all(u8::is_ascii_graphic)
        })
        .cloned()
        .unwrap_or_else(new_request_id);
    let preflight = request.method() == Method::OPTIONS
        && request.headers().contains_key(ORIGIN)
        && request
            .headers()
            .contains_key(ACCESS_CONTROL_REQUEST_METHOD);
    let mut response = if preflight {
        let mut response = StatusCode::OK.into_response();
        let headers = response.headers_mut();
        let any = HeaderValue::from_static("*");
        headers.insert(ACCESS_CONTROL_ALLOW_METHODS, any.clone());
        headers.insert(ACCESS_CONTROL_ALLOW_HEADERS, any);
        headers.insert(
            VARY,
            HeaderValue::from_static(
                "origin, access-control-request-method, access-control-request-headers",
            ),
        );
        response
    } else {
        let span = info_span!(
            "request",
            id = id.to_str().unwrap_or_default(),
            method = %request.method(),
            path = %request.uri().path()
        );
        next.run(request).instrument(span).await
    };
    let headers = response.headers_mut();
    headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    headers.insert(
        ACCESS_CONTROL_EXPOSE_HEADERS,
        HeaderValue::from_static(EXPOSED_HEADERS),
    );
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    headers.insert(X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(REQUEST_ID_HEADER, id);
    response
}

/// A new request ID in UUID v4 format: a random per-process value plus a counter, so IDs are
/// unique without asking the OS for randomness on every request
fn new_request_id() -> HeaderValue {
    static SEED: std::sync::LazyLock<u128> =
        std::sync::LazyLock::new(|| uuid::Uuid::new_v4().as_u128());
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let bits = *SEED ^ u128::from(NEXT.fetch_add(1, Ordering::Relaxed));
    let id = uuid::Builder::from_random_bytes(bits.to_be_bytes()).into_uuid();
    let mut buffer = uuid::Uuid::encode_buffer();
    HeaderValue::from_str(id.hyphenated().encode_lower(&mut buffer))
        .expect("a UUID is a valid header value")
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
    start_server_with_models(config, vec![model.into()]).await
}

/// [`start_server_with_config`] for several models; the first is the default for `/predict`
pub async fn start_server_with_models(
    config: ServerConfig,
    models: Vec<ServedModel>,
) -> Result<()> {
    serve(config, models.into()).await
}

async fn serve(config: ServerConfig, contents: Contents) -> Result<()> {
    let state = server_state(contents, &config);
    #[cfg(not(feature = "llm"))]
    let names = state.model_names();
    #[cfg(feature = "llm")]
    let names: Vec<&str> = state
        .model_names()
        .into_iter()
        .chain(state.llm.iter().flat_map(|llm| llm.names()))
        .collect();
    info!("📚 Serving {}", names.join(", "));
    let app = router_with_state(state.clone(), &config);
    info!(
        "🚦 Concurrency limit set to {} inferences",
        config.max_concurrent_requests
    );
    if config.batching.enabled() {
        info!(
            "📦 Dynamic batching: up to {} items per batch, waiting at most {}µs, {} worker(s)",
            config.batching.max_batch_size,
            config.batching.batch_timeout.as_micros(),
            config.batching.workers
        );
    } else {
        info!("📦 Dynamic batching off");
    }

    #[cfg(feature = "tls")]
    let tls = match &config.tls {
        Some(settings) => {
            let configs = crate::tls::TlsConfigs::load(settings.clone())
                .map_err(|e| ApiError::ServerStartup(format!("TLS: {e}")))?;
            configs.watch();
            info!("🔐 TLS 1.3 with {}", crate::tls::describe(settings));
            Some(configs)
        }
        None => None,
    };
    #[cfg(not(feature = "tls"))]
    if config.tls.is_some() {
        return Err(ApiError::ServerStartup(
            "--tls-cert needs a build with the `tls` feature".to_string(),
        )
        .into());
    }

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| ApiError::ServerStartup(format!("Failed to bind to {bind_addr}: {e}")))?;

    let scheme = if config.tls.is_some() {
        "https"
    } else {
        "http"
    };
    info!("✅ Server running on {scheme}://{bind_addr}");

    // One signal stops both servers
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let wait_for_stop = move |mut stopped: tokio::sync::watch::Receiver<bool>| async move {
        let _ = stopped.wait_for(|&stop| stop).await;
    };
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = stop.send(true);
    });

    #[cfg(feature = "grpc")]
    let grpc = config.grpc_port.map(|port| {
        let addr = format!("{}:{port}", config.host);
        let state = state.clone();
        let max_message_bytes = config.max_body_bytes;
        let shutdown = wait_for_stop(stopped.clone());
        #[cfg(feature = "tls")]
        let tls = tls.clone();
        #[cfg(not(feature = "tls"))]
        let tls = None;
        tokio::spawn(async move {
            let addr = tokio::net::lookup_host(&addr)
                .await
                .ok()
                .and_then(|mut addrs| addrs.next())
                .ok_or_else(|| format!("cannot resolve {addr}"))?;
            grpc::serve(state, addr, max_message_bytes, tls, shutdown).await
        })
    });
    #[cfg(not(feature = "grpc"))]
    if config.grpc_port.is_some() {
        return Err(ApiError::ServerStartup(
            "--grpc-port needs a build with the `grpc` feature".to_string(),
        )
        .into());
    }

    if !config.api_keys.is_empty() {
        info!(
            "🔑 API keys required ({} configured)",
            config.api_keys.len()
        );
    }
    if let Some(limit) = config.rate_limit {
        info!(
            "⏱️ Rate limit: {} requests/s per client, bursts of {}",
            limit.per_second, limit.burst
        );
    }

    let make_service = app.into_make_service_with_connect_info::<ClientAddr>();
    #[cfg(feature = "tls")]
    let served = match tls {
        Some(configs) => {
            let listener = crate::tls::TlsListener::new(listener, configs)
                .map_err(|e| ApiError::ServerStartup(format!("TLS listener: {e}")))?;
            axum::serve(listener, make_service)
                .with_graceful_shutdown(wait_for_stop(stopped))
                .await
        }
        None => {
            axum::serve(listener, make_service)
                .with_graceful_shutdown(wait_for_stop(stopped))
                .await
        }
    };
    #[cfg(not(feature = "tls"))]
    let served = axum::serve(listener, make_service)
        .with_graceful_shutdown(wait_for_stop(stopped))
        .await;
    served.map_err(|e| ApiError::ServerStartup(format!("Server error: {e}")))?;

    #[cfg(feature = "grpc")]
    if let Some(grpc) = grpc {
        match grpc.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(ApiError::ServerStartup(e).into()),
            Err(e) => return Err(ApiError::ServerStartup(format!("gRPC task failed: {e}")).into()),
        }
    }

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
        backend: state.default_slot().map_or_else(
            || "none".to_string(),
            |slot| slot.get().model.get_backend_info(),
        ),
        uptime_seconds: state.uptime_seconds(),
        timestamp: Utc::now().to_rfc3339(),
    })
}

/// `GET /readyz`: 200 once the warm-up inference has run, 503 before. Unlike `/healthz`
/// (alive), this tells a load balancer when to start sending traffic.
async fn readiness(State(state): State<AppState>) -> Response {
    let ready = state.ready();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(ReadinessResponse {
            status: if ready { "ready" } else { "warming_up" }.to_string(),
            timestamp: Utc::now().to_rfc3339(),
        }),
    )
        .into_response()
}

async fn predict(State(state): State<AppState>, request: Request) -> Response {
    let Some(slot) = state.default_slot().cloned() else {
        return no_tensor_model_response();
    };
    predict_with(&state, &slot, request).await
}

/// For `/predict` and `/model/info` on a server that serves only LLMs
fn no_tensor_model_response() -> Response {
    create_error_response(
        StatusCode::NOT_FOUND,
        "MODEL_NOT_FOUND",
        "This server serves no tensor model, only LLMs on /v1/chat/completions",
        None,
    )
}

/// `POST /models/{model}/predict`: `/predict` for any served model
async fn predict_named(
    State(state): State<AppState>,
    Path(name): Path<String>,
    request: Request,
) -> Response {
    let Some(slot) = state.slot(&name).cloned() else {
        return unknown_model_response(&state, &name);
    };
    predict_with(&state, &slot, request).await
}

fn unknown_model_response(state: &ServerState, name: &str) -> Response {
    create_error_response(
        StatusCode::NOT_FOUND,
        "MODEL_NOT_FOUND",
        &format!(
            "Unknown model '{name}'; this server serves {}",
            state.model_names().join(", ")
        ),
        Some(serde_json::json!({ "models": state.model_names() })),
    )
}

async fn predict_with(state: &AppState, slot: &ModelSlot, request: Request) -> Response {
    // The model as of now: a reload during this request does not affect it
    let loaded = slot.get();
    if !loaded.model.is_simple() {
        let info = loaded.model.get_info();
        return create_error_response(
            StatusCode::BAD_REQUEST,
            "NAMED_TENSORS_REQUIRED",
            &format!(
                "Model '{}' takes named tensors ({}); send them to POST /v2/models/{}/infer",
                info.name,
                info.inputs
                    .iter()
                    .map(|t| format!("{} {}", t.name, t.datatype))
                    .collect::<Vec<_>>()
                    .join(", "),
                info.name
            ),
            None,
        );
    }
    // Announce the request on its pathway, so a batch about to start can wait for it while
    // its body is received and decoded
    let announced = state.serving.announce(&slot.name);
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
        && let Err(response) = image_size(&loaded.model)
    {
        return *response;
    }
    let binary_response = accepts_octet_stream(request.headers());
    let headers = request.headers().clone();

    let body = match Bytes::from_request(request, state).await {
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
        RequestFormat::Json => decode_json(&loaded.model, &body),
        RequestFormat::Binary => decode_binary(&loaded.model, &headers, &body),
        #[cfg(feature = "image")]
        RequestFormat::Image(kind) => decode_image(state, &loaded.model, kind, body.clone()).await,
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

    let result = match run_inference(state, &loaded, announced, data, batch_size, permit).await {
        Ok(result) => result,
        Err(response) => return *response,
    };

    match result {
        Ok(output) => {
            let elapsed = start_time.elapsed();
            state.observe_inference(slot, elapsed);
            let inference_time_ms = elapsed.as_secs_f64() * 1000.0;
            // Per request, so debug: at info it would cost more than the server's own work
            debug!(
                "Prediction completed in {:.2}ms, batch size: {}",
                inference_time_ms, batch_size
            );
            if binary_response {
                return binary_output_response(
                    &output,
                    batch_size,
                    &loaded.model.get_info().output_spec.shape,
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
                FurnaceError::Model(
                    ModelError::InvalidDataType(_) | ModelError::InvalidInput(_),
                ) => (StatusCode::BAD_REQUEST, "INVALID_INPUT_VALUE"),
                FurnaceError::Model(ModelError::InferenceFailed(_)) => {
                    (StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_FAILED")
                }
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
            };
            create_error_response(status_code, error_code, &e.to_string(), None)
        }
    }
}

/// Run one request of flat `f32` inputs through its model's pathway (see [`run_general`]).
///
/// The outer error is a finished response (busy, timeout, a lost request); the inner result is
/// the model's answer for this request.
async fn run_inference(
    state: &ServerState,
    loaded: &Loaded,
    announced: Option<Announced<'_>>,
    data: Vec<f32>,
    batch_size: usize,
    permit: OwnedSemaphorePermit,
) -> std::result::Result<Result<Vec<f32>>, Box<Response>> {
    let mut shape = Vec::with_capacity(loaded.model.get_info().input_spec.shape.len() + 1);
    shape.push(batch_size);
    shape.extend_from_slice(&loaded.model.get_info().input_spec.shape);
    let input = HostTensor {
        shape,
        values: Values::F32(data),
    };
    let outputs = run_general(state, loaded, announced, vec![input], permit).await?;
    Ok(outputs.and_then(|outputs| match outputs.into_iter().next() {
        Some(HostTensor {
            values: Values::F32(values),
            ..
        }) => Ok(values),
        _ => Err(ModelError::InferenceFailed("the model returned no f32 output".to_string()).into()),
    }))
}

/// Run one request of named tensors through its model's pathway in the nervous system:
/// summed with concurrent requests when batching is on, on a blocking thread of its own
/// otherwise; under the timeout, holding its concurrency slot until it is answered.
async fn run_general(
    state: &ServerState,
    loaded: &Loaded,
    announced: Option<Announced<'_>>,
    inputs: Vec<HostTensor>,
    permit: OwnedSemaphorePermit,
) -> std::result::Result<Result<Vec<HostTensor>>, Box<Response>> {
    let task = state
        .serving
        .infer(&loaded.model.get_info().name, inputs, announced);
    let answered = match state.request_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, task).await {
            Ok(answered) => answered,
            Err(_) => return Err(Box::new(timeout_response(timeout))),
        },
        None => task.await,
    };
    // Free the slot before the response is made, so a client that reads /metrics right after
    // its response never sees its own request still in flight
    drop(permit);
    match answered {
        Ok(result) => Ok(result),
        Err(Refused::Busy) => Err(Box::new(busy_response())),
        Err(Refused::Lost) => Err(Box::new(panicked_response())),
    }
}

fn timeout_response(timeout: Duration) -> Response {
    warn!("Inference exceeded timeout of {:?}", timeout);
    create_error_response(
        StatusCode::GATEWAY_TIMEOUT,
        "INFERENCE_TIMEOUT",
        &format!("Inference exceeded timeout of {}ms", timeout.as_millis()),
        None,
    )
}

fn panicked_response() -> Response {
    create_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "INFERENCE_FAILED",
        "Inference task panicked",
        None,
    )
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
    // The usual bodies, `{"input": [...]}` and `{"inputs": [[...], ...]}`, take a faster
    // parser; anything else, including every malformed body, goes through serde_json
    let input_data = match crate::json_fast::parse(body) {
        Some(crate::json_fast::FastInput::Single(input)) => PredictInputData::Single { input },
        Some(crate::json_fast::FastInput::Batch(inputs)) => PredictInputData::Batch { inputs },
        None => decode_json_serde(body)?.input_data,
    };
    match input_data {
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

fn decode_json_serde(body: &[u8]) -> std::result::Result<PredictRequest, Box<Response>> {
    // serde_json directly rather than axum's `Json`, which also tracks the path of
    // every element for error messages
    serde_json::from_slice(body).map_err(|e| {
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
    })
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
    model: &Model,
    kind: ImageKind,
    body: Bytes,
) -> std::result::Result<PredictInput, Box<Response>> {
    let (height, width) = image_size(model)?;
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
    let start = Instant::now();
    let response = next.run(request).await;
    state.request_latency.observe(start.elapsed());
    let status = response.status();
    if status.is_client_error() || status.is_server_error() {
        state.total_errors.fetch_add(1, Ordering::Relaxed);
    }
    if status == StatusCode::SERVICE_UNAVAILABLE {
        state.rejected_busy.fetch_add(1, Ordering::Relaxed);
    }
    response
}

/// `GET /metrics`: JSON by default, or the Prometheus text format for a scraper (a request
/// whose `Accept` lists `text/plain` or `application/openmetrics-text` before any JSON type)
async fn metrics(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if wants_prometheus(&headers) {
        return (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
            )],
            prometheus_metrics(&state),
        )
            .into_response();
    }
    Json(json_metrics(&state)).into_response()
}

/// Whether `Accept` lists `text/plain` or `application/openmetrics-text` before (or without)
/// JSON or a wildcard
fn wants_prometheus(headers: &HeaderMap) -> bool {
    let Some(accept) = headers.get(ACCEPT).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    for part in accept.split(',') {
        let media = media_type_essence(part.trim());
        if media == "application/json" || media == "*/*" {
            return false;
        }
        if media == "text/plain" || media == "application/openmetrics-text" {
            return true;
        }
    }
    false
}

fn prometheus_metrics(state: &ServerState) -> String {
    use crate::telemetry::{label, render_value};
    use std::fmt::Write as _;
    let mut out = String::with_capacity(8192);
    let _ = writeln!(out, "# HELP furnace_build_info Furnace version");
    let _ = writeln!(out, "# TYPE furnace_build_info gauge");
    let _ = writeln!(
        out,
        "furnace_build_info{{version=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION")
    );
    let gauge = "gauge";
    let counter = "counter";
    let values: [(&str, &str, &str, u64); 10] = [
        (
            "furnace_uptime_seconds",
            gauge,
            "Seconds since the server started",
            state.uptime_seconds(),
        ),
        (
            "furnace_ready",
            gauge,
            "1 once every model's warm-up inference has run",
            u64::from(state.ready()),
        ),
        (
            "furnace_requests_total",
            counter,
            "/predict requests received, including rejected ones",
            state.total_requests.load(Ordering::Relaxed),
        ),
        (
            "furnace_request_errors_total",
            counter,
            "/predict requests answered with a 4xx or 5xx status",
            state.total_errors.load(Ordering::Relaxed),
        ),
        (
            "furnace_requests_rejected_busy_total",
            counter,
            "Requests rejected with 503 SERVER_BUSY",
            state.rejected_busy.load(Ordering::Relaxed),
        ),
        (
            "furnace_requests_rejected_unauthorized_total",
            counter,
            "Requests rejected with 401 for a missing or unknown API key",
            state.rejected_unauthorized.load(Ordering::Relaxed),
        ),
        (
            "furnace_requests_rejected_rate_limited_total",
            counter,
            "Requests rejected with 429 by the rate limit",
            state.rejected_rate_limited.load(Ordering::Relaxed),
        ),
        (
            "furnace_in_flight_requests",
            gauge,
            "Requests queued for or running inference",
            in_flight(state) as u64,
        ),
        (
            "furnace_max_concurrent_requests",
            gauge,
            "Most requests queued for or running inference at once",
            state.max_concurrent as u64,
        ),
        (
            "furnace_max_batch_size",
            gauge,
            "Most items per batched forward pass; 1 means batching is off",
            state.max_batch_size as u64,
        ),
    ];
    for (name, kind, help, value) in values {
        render_value(&mut out, name, kind, help, value);
    }
    state.request_latency.render(
        &mut out,
        "furnace_request_duration_seconds",
        "Inference request latency from arrival to response",
    );
    #[cfg(feature = "llm")]
    if let Some(llm) = &state.llm {
        llm.render_prometheus(&mut out);
    }
    state.serving.fabric().render_prometheus(&mut out);

    // Per model, labelled `model`
    let models: Vec<_> = state
        .models
        .iter()
        .map(|slot| {
            let loaded = slot.get();
            (slot, loaded.model.get_stats(), loaded)
        })
        .collect();
    type Field = fn(&ModelStats, &Loaded) -> String;
    let families: [(&str, &str, &str, Field); 7] = [
        (
            "furnace_model_info",
            gauge,
            "Served models and their backends",
            |_, _| "1".to_string(),
        ),
        (
            "furnace_model_ready",
            gauge,
            "1 once the model's warm-up inference has run",
            |_, l| u8::from(l.ready()).to_string(),
        ),
        (
            "furnace_inferences_total",
            counter,
            "Requests that reached the model",
            |s, _| s.inference_count.to_string(),
        ),
        (
            "furnace_inference_errors_total",
            counter,
            "Requests the model failed",
            |s, _| s.error_count.to_string(),
        ),
        (
            "furnace_batches_total",
            counter,
            "Forward passes run",
            |s, _| s.batch_count.to_string(),
        ),
        (
            "furnace_batch_items_total",
            counter,
            "Items processed by forward passes",
            |s, _| s.batched_items.to_string(),
        ),
        (
            "furnace_model_size_bytes",
            gauge,
            "Size of the model weights",
            |_, l| l.model.get_info().model_size_bytes.to_string(),
        ),
    ];
    for (name, kind, help, field) in families {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for (slot, stats, loaded) in &models {
            let mut labels = format!("model=\"{}\"", label(&slot.name));
            if name == "furnace_model_info" {
                let _ = write!(
                    labels,
                    ",backend=\"{}\"",
                    label(&loaded.model.get_info().backend)
                );
            }
            let _ = writeln!(out, "{name}{{{labels}}} {}", field(stats, loaded));
        }
    }
    let name = "furnace_inference_duration_seconds";
    let _ = writeln!(
        out,
        "# HELP {name} Successful inference latency from decoded input to output, queueing included"
    );
    let _ = writeln!(out, "# TYPE {name} histogram");
    for (slot, _, _) in &models {
        slot.inference_latency.render_series(
            &mut out,
            name,
            &format!("model=\"{}\"", label(&slot.name)),
        );
    }
    out
}

fn in_flight(state: &ServerState) -> usize {
    state
        .max_concurrent
        .saturating_sub(state.inference_slots.available_permits())
}

fn json_metrics(state: &ServerState) -> MetricsResponse {
    let models: Vec<ModelMetrics> = state
        .models
        .iter()
        .map(|slot| {
            let loaded = slot.get();
            let stats = loaded.model.get_stats();
            ModelMetrics {
                name: slot.name.clone(),
                ready: loaded.ready(),
                inference_count: stats.inference_count,
                error_count: stats.error_count,
                batch_count: stats.batch_count,
                average_batch_size: average_batch_size(&stats),
                average_inference_time_ms: stats.average_inference_time_ms,
                inference_latency_ms: slot.inference_latency.percentiles_ms(),
            }
        })
        .collect();
    // Totals over the models
    let stats: Vec<ModelStats> = state
        .models
        .iter()
        .map(|slot| slot.get().model.get_stats())
        .collect();
    let success_count: u64 = stats.iter().map(|s| s.success_count).sum();
    let total_inference_time_ms: f64 = stats.iter().map(|s| s.total_inference_time_ms).sum();
    let stats = ModelStats {
        inference_count: stats.iter().map(|s| s.inference_count).sum(),
        total_inference_time_ms,
        last_inference_time: stats.iter().filter_map(|s| s.last_inference_time).max(),
        average_inference_time_ms: if success_count == 0 {
            0.0
        } else {
            total_inference_time_ms / success_count as f64
        },
        error_count: stats.iter().map(|s| s.error_count).sum(),
        success_count,
        min_inference_time_ms: stats
            .iter()
            .filter_map(|s| s.min_inference_time_ms)
            .reduce(f64::min),
        max_inference_time_ms: stats
            .iter()
            .filter_map(|s| s.max_inference_time_ms)
            .reduce(f64::max),
        batch_count: stats.iter().map(|s| s.batch_count).sum(),
        batched_items: stats.iter().map(|s| s.batched_items).sum(),
    };
    MetricsResponse {
        uptime_seconds: state.uptime_seconds(),
        total_requests: state.total_requests.load(Ordering::Relaxed),
        total_errors: state.total_errors.load(Ordering::Relaxed),
        rejected_busy: state.rejected_busy.load(Ordering::Relaxed),
        rejected_unauthorized: state.rejected_unauthorized.load(Ordering::Relaxed),
        rejected_rate_limited: state.rejected_rate_limited.load(Ordering::Relaxed),
        in_flight_inferences: in_flight(state),
        max_concurrent_inferences: state.max_concurrent,
        total_inferences: stats.inference_count,
        total_batches: stats.batch_count,
        average_batch_size: average_batch_size(&stats),
        max_batch_size: state.max_batch_size,
        average_inference_time_ms: stats.average_inference_time_ms,
        last_inference_at: stats.last_inference_time.map(|t| t.to_rfc3339()),
        request_latency_ms: state.request_latency.percentiles_ms(),
        inference_latency_ms: state.inference_latency.percentiles_ms(),
        ready: state.ready(),
        models,
        #[cfg(feature = "llm")]
        llm: state
            .llm
            .as_ref()
            .map(|llm| llm.json_metrics())
            .unwrap_or_default(),
        nerve: Some(state.serving.fabric().metrics()),
        timestamp: Utc::now().to_rfc3339(),
    }
}

fn average_batch_size(stats: &ModelStats) -> f64 {
    if stats.batch_count == 0 {
        0.0
    } else {
        stats.batched_items as f64 / stats.batch_count as f64
    }
}

fn model_info_response(slot: &ModelSlot) -> ModelInfoResponse {
    let loaded = slot.get();
    ModelInfoResponse {
        model_info: loaded.model.get_info().clone(),
        stats: loaded.model.get_stats(),
        timestamp: Utc::now().to_rfc3339(),
    }
}

/// `GET /model/info`: the default model
async fn model_info(State(state): State<AppState>) -> Response {
    match state.default_slot() {
        Some(slot) => Json(model_info_response(slot)).into_response(),
        None => no_tensor_model_response(),
    }
}

/// `GET /models/{model}`
async fn model_info_named(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    match state.slot(&name) {
        Some(slot) => Json(model_info_response(slot)).into_response(),
        None => unknown_model_response(&state, &name),
    }
}

/// `GET /models`: every served model
async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let models = state
        .models
        .iter()
        .map(|slot| {
            let loaded = slot.get();
            let info = loaded.model.get_info();
            ModelSummary {
                name: slot.name.clone(),
                ready: loaded.ready(),
                input_shape: info.input_spec.shape.clone(),
                output_shape: info.output_spec.shape.clone(),
                backend: info.backend.clone(),
                reloadable: slot.reload.is_some(),
            }
        })
        .collect();
    Json(ModelsResponse {
        default_model: state
            .default_slot()
            .map(|slot| slot.name.clone())
            .unwrap_or_default(),
        models,
    })
}

#[derive(Debug, Default, Deserialize)]
struct ReloadRequest {
    /// New weight file; by default the model reloads from where it was loaded before
    weights: Option<std::path::PathBuf>,
}

/// `POST /admin/models/{model}/reload` (bearer token `--admin-token`): load the model again,
/// warm it up, and switch to it. Requests in flight finish on the previous one.
async fn reload_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(token) = &state.admin_token else {
        return create_error_response(
            StatusCode::NOT_FOUND,
            "ADMIN_API_DISABLED",
            "The admin API is off; start the server with --admin-token to enable it",
            None,
        );
    };
    if !authorized(&headers, token) {
        let mut response = create_error_response(
            StatusCode::UNAUTHORIZED,
            "UNAUTHORIZED",
            "Missing or wrong bearer token",
            None,
        );
        response.headers_mut().insert(
            axum::http::header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer"),
        );
        return response;
    }
    let Some(slot) = state.slot(&name).cloned() else {
        return unknown_model_response(&state, &name);
    };
    let Some(reload) = slot.reload.clone() else {
        return create_error_response(
            StatusCode::CONFLICT,
            "NOT_RELOADABLE",
            &format!("Model '{name}' cannot be reloaded"),
            None,
        );
    };
    let request: ReloadRequest = if body.iter().all(u8::is_ascii_whitespace) {
        ReloadRequest::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(request) => request,
            Err(e) => {
                return create_error_response(
                    StatusCode::BAD_REQUEST,
                    "INVALID_REQUEST",
                    &format!("Expected {{\"weights\": \"<path>\"}} or an empty body: {e}"),
                    None,
                );
            }
        }
    };
    let Ok(_reloading) = slot.reloading.try_lock() else {
        return create_error_response(
            StatusCode::CONFLICT,
            "RELOAD_IN_PROGRESS",
            &format!("Model '{name}' is already being reloaded"),
            None,
        );
    };

    let previous = slot.get();
    let weights = request.weights.clone();
    let start = Instant::now();
    let task = tokio::task::spawn_blocking(move || -> Result<(Loaded, f64)> {
        let mut model = reload(weights.as_deref())?;
        let previous_info = previous.model.get_info();
        let info = model.get_info();
        if info.name != previous_info.name {
            return Err(ModelError::InvalidFormat(format!(
                "The reloaded model is named '{}', not '{}'",
                info.name, previous_info.name
            ))
            .into());
        }
        if info.inputs != previous_info.inputs || info.outputs != previous_info.outputs {
            let describe = |tensors: &[crate::tensor::TensorInfo]| {
                tensors
                    .iter()
                    .map(|t| format!("{} {} {:?}", t.name, t.datatype, t.shape))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(ModelError::InvalidFormat(format!(
                "The reloaded model takes [{}] -> [{}], the served one [{}] -> [{}]",
                describe(&info.inputs),
                describe(&info.outputs),
                describe(&previous_info.inputs),
                describe(&previous_info.outputs)
            ))
            .into());
        }
        model.adopt_stats(&previous.model);
        let loaded = Loaded::new(model);
        let warm_up = loaded
            .model
            .warm_up()
            .map_err(ModelError::InferenceFailed)?;
        loaded.ready.store(true, Ordering::Release);
        Ok((loaded, warm_up.map_or(0.0, |d| d.as_secs_f64() * 1e3)))
    });
    match task.await {
        Ok(Ok((loaded, warm_up_ms))) => {
            slot.replace(Arc::new(loaded));
            let reload_ms = start.elapsed().as_secs_f64() * 1e3;
            info!("🔄 Reloaded model {name} in {reload_ms:.1}ms (warm-up {warm_up_ms:.1}ms)");
            Json(ReloadResponse {
                model: name,
                weights: request.weights.map(|p| p.display().to_string()),
                reload_ms,
                warm_up_ms,
                timestamp: Utc::now().to_rfc3339(),
            })
            .into_response()
        }
        Ok(Err(e)) => {
            warn!("Reloading model {name} failed, still serving the previous one: {e}");
            create_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "RELOAD_FAILED",
                &format!("Reload failed, still serving the previous model: {e}"),
                None,
            )
        }
        Err(e) => create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "RELOAD_FAILED",
            &format!("Reload task failed, still serving the previous model: {e}"),
            None,
        ),
    }
}

/// Whether `Authorization` is `Bearer <token>`, compared in constant time
fn authorized(headers: &HeaderMap, token: &str) -> bool {
    let Some(given) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_eq(given.as_bytes(), token.as_bytes())
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
pub(crate) mod tests {
    use super::*;
    use crate::model::DummyModel;
    use axum::{body::Body, http::Request};
    use tower::util::ServiceExt;

    pub(super) fn dummy_model() -> Model {
        let dummy = DummyModel::new("test_model".to_string(), vec![784], vec![10]);
        Model::new(Box::new(dummy), "dummy", 0)
    }

    pub(super) fn test_app() -> Router {
        router(dummy_model(), &ServerConfig::default())
    }

    #[cfg(feature = "grpc")]
    pub(super) fn test_app_state() -> AppState {
        server_state(
            vec![ServedModel::from(dummy_model())].into(),
            &ServerConfig::default(),
        )
    }

    pub(super) async fn send(
        app: Router,
        request: Request<Body>,
    ) -> (StatusCode, axum::body::Bytes) {
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
        assert_eq!(m.total_batches, 1);
        assert_eq!(m.average_batch_size, 1.0);
        // Batching is off by default
        assert_eq!(m.max_batch_size, 1);
        assert_eq!(m.in_flight_inferences, 0);
        assert!(m.last_inference_at.is_some());
    }

    async fn concurrent_predictions(config: &ServerConfig) -> MetricsResponse {
        let app = router(dummy_model(), config);
        let mut tasks = Vec::new();
        for i in 0..12 {
            let body = if i % 3 == 0 {
                serde_json::json!({ "inputs": vec![vec![0.5; 784]; 2] })
            } else if i == 4 {
                serde_json::json!({ "input": vec![0.5; 3] })
            } else {
                serde_json::json!({ "input": vec![0.5; 784] })
            };
            let app = app.clone();
            tasks.push(tokio::spawn(async move {
                send(app, predict_request(body.to_string())).await
            }));
        }
        for (i, task) in tasks.into_iter().enumerate() {
            let (status, body) = task.await.unwrap();
            if i == 4 {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                continue;
            }
            assert_eq!(status, StatusCode::OK);
            let response: PredictResponse = serde_json::from_slice(&body).unwrap();
            let expected = if i % 3 == 0 { 2 } else { 1 };
            assert_eq!(response.batch_size, expected);
            match response.output_data {
                PredictOutputData::Batch { outputs } => assert_eq!(outputs.len(), 2),
                PredictOutputData::Single { output } => assert_eq!(output.len(), 10),
            }
        }
        get_metrics(app).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_predictions_with_and_without_batching() {
        let batched = concurrent_predictions(&ServerConfig::default()).await;
        let unbatched = concurrent_predictions(&ServerConfig {
            batching: BatchConfig {
                max_batch_size: 1,
                ..BatchConfig::default()
            },
            ..ServerConfig::default()
        })
        .await;
        for m in [&batched, &unbatched] {
            // The request with the wrong shape is rejected before reaching the model
            assert_eq!(m.total_requests, 12);
            assert_eq!(m.total_errors, 1);
            assert_eq!(m.total_inferences, 11);
            assert!((1..=11).contains(&m.total_batches), "{m:?}");
            assert_eq!(m.in_flight_inferences, 0);
        }
        assert_eq!(unbatched.total_batches, 11);
        assert_eq!(unbatched.average_batch_size, 15.0 / 11.0);
        assert_eq!(unbatched.max_batch_size, 1);
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
    async fn test_cors_open_to_any_origin() {
        // A browser's preflight is answered without reaching the routes
        let preflight = Request::builder()
            .uri("/predict")
            .method("OPTIONS")
            .header(ORIGIN, "https://example.com")
            .header(ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header("access-control-request-headers", "content-type")
            .body(Body::empty())
            .unwrap();
        let response = test_app().oneshot(preflight).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let headers = response.headers();
        assert_eq!(headers[ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        assert_eq!(headers[ACCESS_CONTROL_ALLOW_METHODS], "*");
        assert_eq!(headers[ACCESS_CONTROL_ALLOW_HEADERS], "*");
        assert!(headers.contains_key(REQUEST_ID_HEADER));

        // Responses let scripts read furnace's own headers
        let request = Request::builder()
            .uri("/healthz")
            .header(ORIGIN, "https://example.com")
            .body(Body::empty())
            .unwrap();
        let response = test_app().oneshot(request).await.unwrap();
        assert_eq!(response.headers()[ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        let exposed = response.headers()[ACCESS_CONTROL_EXPOSE_HEADERS]
            .to_str()
            .unwrap();
        for name in [
            BATCH_SIZE_HEADER,
            OUTPUT_SHAPE_HEADER,
            INFERENCE_TIME_HEADER,
            REQUEST_ID_HEADER,
        ] {
            assert!(exposed.contains(name.as_str()), "{exposed}");
        }
    }

    /// Every field of `value` is in the `name` schema of docs/openapi.json, and every field the
    /// schema requires is in `value`
    fn assert_matches_schema(spec: &serde_json::Value, name: &str, value: &serde_json::Value) {
        let schema = &spec["components"]["schemas"][name];
        let properties = schema["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("no schema {name}"));
        let fields = value
            .as_object()
            .unwrap_or_else(|| panic!("{name}: not an object"));
        for field in fields.keys() {
            assert!(
                properties.contains_key(field),
                "{name}: the server sends `{field}`, the spec does not list it"
            );
        }
        for required in schema["required"].as_array().into_iter().flatten() {
            let required = required.as_str().unwrap();
            assert!(
                fields.contains_key(required),
                "{name}: the spec requires `{required}`, the server does not send it"
            );
        }
    }

    /// docs/openapi.json describes the routes and the responses the server actually sends
    #[tokio::test]
    async fn test_openapi_spec_matches_server() {
        let spec: serde_json::Value =
            serde_json::from_str(include_str!("../docs/openapi.json")).unwrap();
        assert_eq!(spec["info"]["version"], env!("CARGO_PKG_VERSION"));
        let paths: std::collections::BTreeSet<&str> = spec["paths"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            paths,
            [
                "/health",
                "/healthz",
                "/metrics",
                "/admin/models/{model}/reload",
                "/model/info",
                "/models",
                "/models/{model}",
                "/models/{model}/predict",
                "/predict",
                "/readyz",
                "/v2",
                "/v2/health/live",
                "/v2/health/ready",
                "/v2/models/{model}",
                "/v2/models/{model}/infer",
                "/v2/models/{model}/ready",
                "/v2/models/{model}/versions/{version}",
                "/v2/models/{model}/versions/{version}/infer",
                "/v2/models/{model}/versions/{version}/ready"
            ]
            .into()
        );

        let app = test_app();
        let get = |path: &'static str| {
            let app = app.clone();
            let request = Request::builder().uri(path).body(Body::empty()).unwrap();
            async move {
                let (status, body) = send(app, request).await;
                assert_eq!(status, StatusCode::OK, "{path}");
                serde_json::from_slice::<serde_json::Value>(&body).unwrap()
            }
        };
        for path in ["/healthz", "/health"] {
            assert_matches_schema(&spec, "HealthResponse", &get(path).await);
        }

        let single = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        let (status, body) = send(app.clone(), predict_request(single)).await;
        assert_eq!(status, StatusCode::OK);
        let single: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_matches_schema(&spec, "PredictResponse", &single);
        let batch = serde_json::json!({ "inputs": vec![vec![0.5; 784]; 2] }).to_string();
        let (status, body) = send(app.clone(), predict_request(batch)).await;
        assert_eq!(status, StatusCode::OK);
        let batch: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_matches_schema(&spec, "PredictResponse", &batch);

        let bad = serde_json::json!({ "input": vec![0.5; 3] }).to_string();
        let (status, body) = send(app.clone(), predict_request(bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_matches_schema(&spec, "ErrorResponse", &error);

        let metrics = get("/metrics").await;
        assert_matches_schema(&spec, "MetricsResponse", &metrics);
        assert_matches_schema(&spec, "Percentiles", &metrics["request_latency_ms"]);
        wait_until_ready(&app).await;
        assert_matches_schema(&spec, "ReadinessResponse", &get("/readyz").await);

        assert_matches_schema(&spec, "V2ServerMetadata", &get("/v2").await);
        let model = get("/v2/models/test_model").await;
        assert_matches_schema(&spec, "V2ModelMetadata", &model);
        assert_matches_schema(&spec, "V2TensorMetadata", &model["inputs"][0]);
        let infer = Request::builder()
            .uri("/v2/models/test_model/infer")
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "id": "a",
                    "inputs": [{ "name": "input", "shape": [1, 784], "datatype": "FP32", "data": vec![0.5; 784] }],
                })
                .to_string(),
            ))
            .unwrap();
        let (status, body) = send(app.clone(), infer).await;
        assert_eq!(status, StatusCode::OK);
        let infer: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_matches_schema(&spec, "V2InferResponse", &infer);
        assert_matches_schema(&spec, "V2OutputTensor", &infer["outputs"][0]);
        let info = get("/model/info").await;
        assert_matches_schema(&spec, "ModelInfoResponse", &info);
        assert_matches_schema(&spec, "ModelInfo", &info["model_info"]);
        assert_matches_schema(&spec, "ModelStats", &info["stats"]);
        assert_matches_schema(&spec, "TensorSpec", &info["model_info"]["input_spec"]);
        assert_matches_schema(&spec, "TensorInfo", &info["model_info"]["inputs"][0]);
        assert_matches_schema(&spec, "ModelInfoResponse", &get("/models/test_model").await);
        assert_matches_schema(&spec, "ModelMetrics", &metrics["models"][0]);
        let models = get("/models").await;
        assert_matches_schema(&spec, "ModelsResponse", &models);
        assert_matches_schema(&spec, "ModelSummary", &models["models"][0]);

        let app = multi_app(Some(TOKEN));
        let (status, body) = send(app.clone(), reload_request("a", Some(TOKEN), "")).await;
        assert_eq!(status, StatusCode::OK);
        let reload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_matches_schema(&spec, "ReloadResponse", &reload);
    }

    const TOKEN: &str = "0123456789abcdef-token";
    const API_KEY: &str = "api-key-0123456789";

    fn guarded_app(api_keys: &[&str], rate_limit: Option<RateLimit>) -> Router {
        let config = ServerConfig {
            api_keys: api_keys.iter().map(|k| k.to_string()).collect(),
            rate_limit,
            admin_token: Some(TOKEN.to_string()),
            ..ServerConfig::default()
        };
        router(dummy_model(), &config)
    }

    fn with_header(mut request: Request<Body>, name: &'static str, value: &str) -> Request<Body> {
        request
            .headers_mut()
            .insert(name, HeaderValue::from_str(value).unwrap());
        request
    }

    fn ok_predict() -> Request<Body> {
        predict_request(serde_json::json!({ "input": vec![0.5; 784] }).to_string())
    }

    #[tokio::test]
    async fn test_api_keys() {
        let app = guarded_app(&[API_KEY, "another-key-0123456789"], None);
        let response = app.clone().oneshot(ok_predict()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()["www-authenticate"], "Bearer");
        let wrong = with_header(ok_predict(), "authorization", "Bearer api-key-012345678");
        assert_eq!(send(app.clone(), wrong).await.0, StatusCode::UNAUTHORIZED);

        let bearer = with_header(ok_predict(), "authorization", &format!("Bearer {API_KEY}"));
        assert_eq!(send(app.clone(), bearer).await.0, StatusCode::OK);
        let header = with_header(ok_predict(), "x-api-key", "another-key-0123456789");
        assert_eq!(send(app.clone(), header).await.0, StatusCode::OK);

        // Probes stay open; everything else is closed
        for path in ["/healthz", "/health", "/readyz", "/v2/health/live"] {
            let request = Request::builder().uri(path).body(Body::empty()).unwrap();
            let status = send(app.clone(), request).await.0;
            assert_ne!(status, StatusCode::UNAUTHORIZED, "{path}");
        }
        for path in [
            "/metrics",
            "/model/info",
            "/models",
            "/v2",
            "/v2/models/test_model",
        ] {
            let request = Request::builder().uri(path).body(Body::empty()).unwrap();
            assert_eq!(
                send(app.clone(), request).await.0,
                StatusCode::UNAUTHORIZED,
                "{path}"
            );
        }
        // The admin API takes its own token, not an API key
        let reload = Request::builder()
            .uri("/admin/models/test_model/reload")
            .method("POST")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::empty())
            .unwrap();
        assert_ne!(send(app.clone(), reload).await.0, StatusCode::UNAUTHORIZED);

        let request = Request::builder()
            .uri("/metrics")
            .header("x-api-key", API_KEY)
            .body(Body::empty())
            .unwrap();
        let (_, body) = send(app, request).await;
        let metrics: MetricsResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(metrics.rejected_unauthorized, 7);
    }

    #[tokio::test]
    async fn test_rate_limit() {
        let limit = RateLimit {
            per_second: 0.001,
            burst: 2,
        };
        let app = guarded_app(&[], Some(limit));
        for _ in 0..2 {
            assert_eq!(send(app.clone(), ok_predict()).await.0, StatusCode::OK);
        }
        let response = app.clone().oneshot(ok_predict()).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        // One request every 1000 s
        let retry: u64 = response.headers()["retry-after"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((990..=1000).contains(&retry), "{retry}");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["code"], "RATE_LIMITED");
        // Probes are not limited
        let request = Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(app.clone(), request).await.0, StatusCode::OK);

        // With keys, each key has its own budget
        let app = guarded_app(&[API_KEY, "another-key-0123456789"], Some(limit));
        let auth = |key: &str| with_header(ok_predict(), "x-api-key", key);
        for _ in 0..2 {
            assert_eq!(send(app.clone(), auth(API_KEY)).await.0, StatusCode::OK);
        }
        assert_eq!(
            send(app.clone(), auth(API_KEY)).await.0,
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            send(app.clone(), auth("another-key-0123456789")).await.0,
            StatusCode::OK
        );
    }

    fn named_model(name: &str) -> Model {
        let dummy = DummyModel::new(name.to_string(), vec![784], vec![10]);
        Model::new(Box::new(dummy), "dummy", 0)
    }

    /// Models "a" (the default) and "b"; reloading with weights "bad" fails, "renamed" loads a
    /// model with another name
    fn multi_app(admin_token: Option<&str>) -> Router {
        let models = ["a", "b"]
            .map(|name| {
                let name = name.to_string();
                ServedModel {
                    model: named_model(&name),
                    reload: Some(Arc::new(
                        move |weights: Option<&std::path::Path>| match weights
                            .and_then(|w| w.to_str())
                        {
                            Some("bad") => {
                                Err(ModelError::InvalidFormat("bad weights".to_string()).into())
                            }
                            Some("renamed") => Ok(named_model("other")),
                            _ => Ok(named_model(&name)),
                        },
                    )),
                }
            })
            .into();
        let config = ServerConfig {
            admin_token: admin_token.map(str::to_string),
            ..ServerConfig::default()
        };
        router_with_models(models, &config)
    }

    fn reload_request(model: &str, token: Option<&str>, body: &str) -> Request<Body> {
        let mut request = Request::builder()
            .uri(format!("/admin/models/{model}/reload"))
            .method("POST")
            .header(CONTENT_TYPE, "application/json");
        if let Some(token) = token {
            request = request.header("authorization", format!("Bearer {token}"));
        }
        request.body(Body::from(body.to_string())).unwrap()
    }

    fn named_predict_request(model: &str) -> Request<Body> {
        Request::builder()
            .uri(format!("/models/{model}/predict"))
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({ "input": vec![0.5; 784] }).to_string(),
            ))
            .unwrap()
    }

    async fn get_json(app: &Router, path: &str) -> (StatusCode, serde_json::Value) {
        let request = Request::builder().uri(path).body(Body::empty()).unwrap();
        let (status, body) = send(app.clone(), request).await;
        (status, serde_json::from_slice(&body).unwrap_or_default())
    }

    #[tokio::test]
    async fn test_several_models_side_by_side() {
        let app = multi_app(None);
        let ok = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        // `/predict` serves the first model, "a"; the per-model counts below tell them apart
        let (status, _) = send(app.clone(), predict_request(ok)).await;
        assert_eq!(status, StatusCode::OK);
        for _ in 0..2 {
            let (status, _) = send(app.clone(), named_predict_request("b")).await;
            assert_eq!(status, StatusCode::OK);
        }
        let (status, body) = send(app.clone(), named_predict_request("c")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["code"], "MODEL_NOT_FOUND");

        let (status, models) = get_json(&app, "/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(models["default_model"], "a");
        assert_eq!(models["models"].as_array().unwrap().len(), 2);
        assert_eq!(models["models"][1]["name"], "b");
        assert_eq!(models["models"][1]["reloadable"], true);
        let (status, info) = get_json(&app, "/models/b").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(info["model_info"]["name"], "b");
        assert_eq!(info["stats"]["inference_count"], 2);
        let (status, _) = get_json(&app, "/models/c").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // The v2 protocol addresses the models by name too
        let infer = |model: &str| {
            Request::builder()
                .uri(format!("/v2/models/{model}/infer"))
                .method("POST")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "inputs": [{ "name": "input", "shape": [1, 784], "datatype": "FP32", "data": vec![0.5; 784] }],
                    })
                    .to_string(),
                ))
                .unwrap()
        };
        let (status, body) = send(app.clone(), infer("b")).await;
        assert_eq!(status, StatusCode::OK);
        let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(response["model_name"], "b");
        let (status, body) = send(app.clone(), infer("c")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let error = String::from_utf8_lossy(&body);
        assert!(error.contains("a, b"), "{error}");

        let metrics = get_metrics(app.clone()).await;
        assert_eq!(metrics.total_inferences, 4);
        let per_model: Vec<_> = metrics
            .models
            .iter()
            .map(|m| (m.name.as_str(), m.inference_count))
            .collect();
        assert_eq!(per_model, [("a", 1), ("b", 3)]);

        let request = Request::builder()
            .uri("/metrics")
            .header(ACCEPT, "text/plain")
            .body(Body::empty())
            .unwrap();
        let (_, body) = send(app.clone(), request).await;
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains("furnace_inferences_total{model=\"b\"} 3\n"),
            "{text}"
        );
        assert!(
            text.contains("furnace_inference_duration_seconds_count{model=\"a\"} 1\n"),
            "{text}"
        );
        assert!(
            text.contains("furnace_model_info{model=\"a\",backend="),
            "{text}"
        );
        // One HELP / TYPE per family, as Prometheus requires
        assert_eq!(text.matches("# TYPE furnace_inferences_total ").count(), 1);
        assert_eq!(
            text.matches("# TYPE furnace_inference_duration_seconds ")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn test_reload_needs_the_admin_token() {
        let (status, body) = send(multi_app(None), reload_request("a", Some(TOKEN), "")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(String::from_utf8_lossy(&body).contains("ADMIN_API_DISABLED"));

        let app = multi_app(Some(TOKEN));
        for token in [None, Some("wrong"), Some("0123456789abcdef-tokeN")] {
            let response = app
                .clone()
                .oneshot(reload_request("a", token, ""))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{token:?}");
            assert_eq!(response.headers()["www-authenticate"], "Bearer");
        }
        let (status, _) = send(app.clone(), reload_request("c", Some(TOKEN), "")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_reload_swaps_the_model_and_keeps_its_stats() {
        let app = multi_app(Some(TOKEN));
        let (status, _) = send(app.clone(), named_predict_request("b")).await;
        assert_eq!(status, StatusCode::OK);

        let (status, body) = send(app.clone(), reload_request("b", Some(TOKEN), "")).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let reload: ReloadResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(reload.model, "b");
        assert!(reload.reload_ms >= reload.warm_up_ms);

        // The new model is ready at once and carries the old one's counts on
        let (_, models) = get_json(&app, "/models").await;
        assert_eq!(models["models"][1]["ready"], true);
        let (status, _) = send(app.clone(), named_predict_request("b")).await;
        assert_eq!(status, StatusCode::OK);
        let (_, info) = get_json(&app, "/models/b").await;
        assert_eq!(info["stats"]["inference_count"], 2);

        // A failed reload leaves the previous model serving
        for (body, code) in [
            (r#"{"weights": "bad"}"#, StatusCode::UNPROCESSABLE_ENTITY),
            (
                r#"{"weights": "renamed"}"#,
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            ("not json", StatusCode::BAD_REQUEST),
        ] {
            let (status, _) = send(app.clone(), reload_request("b", Some(TOKEN), body)).await;
            assert_eq!(status, code, "{body}");
            let (status, _) = send(app.clone(), named_predict_request("b")).await;
            assert_eq!(status, StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn test_requests_during_a_reload_succeed() {
        let app = multi_app(Some(TOKEN));
        let requests: Vec<_> = (0..64)
            .map(|_| tokio::spawn(send(app.clone(), named_predict_request("a"))))
            .collect();
        let reloads: Vec<_> = (0..4)
            .map(|_| tokio::spawn(send(app.clone(), reload_request("a", Some(TOKEN), ""))))
            .collect();
        for request in requests {
            assert_eq!(request.await.unwrap().0, StatusCode::OK);
        }
        // Reloads of one model run one at a time; any overlapping one is told so
        let mut reloaded = 0;
        for reload in reloads {
            match reload.await.unwrap().0 {
                StatusCode::OK => reloaded += 1,
                status => assert_eq!(status, StatusCode::CONFLICT),
            }
        }
        assert!(reloaded >= 1);
        let metrics = get_metrics(app).await;
        assert_eq!(metrics.models[0].inference_count, 64);
    }

    /// Poll `/readyz` until the warm-up has run
    async fn wait_until_ready(app: &Router) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let request = Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap();
            let (status, _) = send(app.clone(), request).await;
            if status == StatusCode::OK {
                return;
            }
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert!(Instant::now() < deadline, "never became ready");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn test_readiness_after_warm_up() {
        let app = test_app();
        wait_until_ready(&app).await;
        // The warm-up does not count as an inference
        let metrics = get_metrics(app).await;
        assert_eq!(metrics.total_inferences, 0);
        assert!(metrics.ready);
    }

    #[tokio::test]
    async fn test_metrics_latency_percentiles_and_prometheus_format() {
        let app = test_app();
        let ok = serde_json::json!({ "input": vec![0.5; 784] }).to_string();
        for _ in 0..3 {
            let (status, _) = send(app.clone(), predict_request(ok.clone())).await;
            assert_eq!(status, StatusCode::OK);
        }
        let m = get_metrics(app.clone()).await;
        assert!(m.request_latency_ms.p50.is_some());
        assert!(m.inference_latency_ms.p99.is_some());

        // What Prometheus sends
        let request = Request::builder()
            .uri("/metrics")
            .header(
                ACCEPT,
                "application/openmetrics-text;version=1.0.0;q=0.5,text/plain;version=0.0.4;q=0.4,*/*;q=0.1",
            )
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()[CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/plain")
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("furnace_requests_total 3\n"), "{text}");
        assert!(
            text.contains("furnace_inferences_total{model=\"test_model\"} 3\n"),
            "{text}"
        );
        assert!(
            text.contains("furnace_request_duration_seconds_count 3\n"),
            "{text}"
        );
        assert!(text.contains("furnace_build_info{version="), "{text}");

        // A browser or curl gets JSON
        let request = Request::builder()
            .uri("/metrics")
            .header(ACCEPT, "*/*")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert!(
            response.headers()[CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json")
        );
    }

    #[tokio::test]
    async fn test_request_id_echoed_or_generated() {
        let with_id = |id: &str| {
            Request::builder()
                .uri("/healthz")
                .header(REQUEST_ID_HEADER, id)
                .body(Body::empty())
                .unwrap()
        };
        // The client's ID comes back
        let response = test_app().oneshot(with_id("trace-42")).await.unwrap();
        assert_eq!(response.headers()[REQUEST_ID_HEADER], "trace-42");

        // Without one, or with one that is too long or has spaces, a UUID is generated
        let plain = Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        for request in [plain, with_id(&"x".repeat(129)), with_id("a b")] {
            let response = test_app().oneshot(request).await.unwrap();
            let id = response.headers()[REQUEST_ID_HEADER].to_str().unwrap();
            let uuid = uuid::Uuid::parse_str(id).unwrap_or_else(|_| panic!("{id}"));
            assert_eq!(uuid.get_version(), Some(uuid::Version::Random), "{id}");
        }

        // Generated IDs do not repeat
        let ids: std::collections::HashSet<_> = (0..1000).map(|_| new_request_id()).collect();
        assert_eq!(ids.len(), 1000);

        // Errors carry it too
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header("content-type", "application/json")
            .header(REQUEST_ID_HEADER, "bad-1")
            .body(Body::from("{"))
            .unwrap();
        let response = test_app().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[REQUEST_ID_HEADER], "bad-1");
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
