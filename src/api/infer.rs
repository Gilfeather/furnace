//! What every inference route shares, whatever its protocol: admission (the concurrency slot,
//! the timeout and the latency of a success), the one mapping from an error to an HTTP status
//! and `code`, and the raw little-endian codec.

use super::*;
use crate::tensor::DataType;

macro_rules! codes {
    ($($(#[doc = $doc:literal])* $(#[cfg($cfg:meta)])? $name:ident = $code:literal,)*) => {
        /// The `code` of an error response, what a client can match on. `docs/API.md` and
        /// `docs/openapi.json` list the same set (a test checks).
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Code {
            $($(#[doc = $doc])* $(#[cfg($cfg)])? $name,)*
        }

        impl Code {
            /// Every code
            #[cfg(test)]
            pub(crate) const ALL: &[Code] = &[$($(#[cfg($cfg)])? Code::$name,)*];

            pub(crate) fn as_str(self) -> &'static str {
                match self {
                    $($(#[cfg($cfg)])? Code::$name => $code,)*
                }
            }
        }
    };
}

codes! {
    /// A request that is malformed before it reaches the model
    InvalidRequest = "INVALID_REQUEST",
    /// An image that fails to decode
    #[cfg(feature = "image")]
    InvalidImage = "INVALID_IMAGE",
    /// An empty input
    InvalidInput = "INVALID_INPUT",
    /// Values that do not fit the model's input shape
    InputValidationFailed = "INPUT_VALIDATION_FAILED",
    /// Values the model refuses: NaN, infinite, out of range
    InvalidInputValue = "INVALID_INPUT_VALUE",
    /// More items than a request may carry
    BatchTooLarge = "BATCH_TOO_LARGE",
    /// `/predict` for a model that takes named tensors
    NamedTensorsRequired = "NAMED_TENSORS_REQUIRED",
    /// No such model, or no tensor model at all
    ModelNotFound = "MODEL_NOT_FOUND",
    /// A missing or unknown API key or admin token
    Unauthorized = "UNAUTHORIZED",
    /// Over the per-client rate limit
    RateLimited = "RATE_LIMITED",
    /// Every concurrency slot is taken
    ServerBusy = "SERVER_BUSY",
    /// The model took longer than the request timeout
    InferenceTimeout = "INFERENCE_TIMEOUT",
    /// The model failed or panicked
    InferenceFailed = "INFERENCE_FAILED",
    /// Anything else the server got wrong
    InternalError = "INTERNAL_ERROR",
    /// The admin API is off
    AdminApiDisabled = "ADMIN_API_DISABLED",
    /// A reload of a model served without a reloader
    NotReloadable = "NOT_RELOADABLE",
    /// A reload while one runs
    ReloadInProgress = "RELOAD_IN_PROGRESS",
    /// A reload that failed; the previous model still serves
    ReloadFailed = "RELOAD_FAILED",
}

/// The status and `code` of a model's error: 400 when the request is at fault, 500 when the
/// model or the server is
pub(crate) fn model_error_status(e: &FurnaceError) -> (StatusCode, Code) {
    match e {
        FurnaceError::Model(ModelError::InputValidation { .. }) => {
            (StatusCode::BAD_REQUEST, Code::InputValidationFailed)
        }
        FurnaceError::Model(ModelError::EmptyBatch) => {
            (StatusCode::BAD_REQUEST, Code::InvalidInput)
        }
        FurnaceError::Model(ModelError::BatchTooLarge { .. }) => {
            (StatusCode::BAD_REQUEST, Code::BatchTooLarge)
        }
        e if e.is_client_error() => (StatusCode::BAD_REQUEST, Code::InvalidInputValue),
        FurnaceError::Model(ModelError::InferenceFailed(_)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Code::InferenceFailed)
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, Code::InternalError),
    }
}

/// Why an inference request got no output
#[derive(Debug)]
pub(crate) enum InferError {
    /// Every concurrency slot is taken
    Busy,
    /// The model did not answer within the request timeout
    Timeout(Duration),
    /// The model panicked, or its pathway dropped the request
    Lost,
    /// The model refused or failed the request
    Model(FurnaceError),
}

/// Marks a response as a refusal for want of a concurrency slot, for the request counters
#[derive(Debug, Clone, Copy)]
pub(crate) struct Busy;

impl InferError {
    pub(crate) fn status(&self) -> (StatusCode, Code) {
        match self {
            Self::Busy => (StatusCode::SERVICE_UNAVAILABLE, Code::ServerBusy),
            Self::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, Code::InferenceTimeout),
            Self::Lost => (StatusCode::INTERNAL_SERVER_ERROR, Code::InferenceFailed),
            Self::Model(e) => model_error_status(e),
        }
    }

    pub(crate) fn message(&self) -> String {
        match self {
            Self::Busy => "Too many concurrent requests, retry later".to_string(),
            Self::Timeout(timeout) => {
                format!("Inference exceeded timeout of {}ms", timeout.as_millis())
            }
            Self::Lost => "The inference ended without an answer (see the server log)".to_string(),
            Self::Model(e) => e.to_string(),
        }
    }

    /// The JSON error response, the same for `/predict` and v2
    pub(crate) fn into_response(self) -> Response {
        match &self {
            Self::Busy => warn!("All inference slots busy, rejecting request"),
            Self::Timeout(timeout) => warn!("Inference exceeded timeout of {timeout:?}"),
            Self::Lost => error!("Prediction error: {}", self.message()),
            // The request is at fault, not the server
            Self::Model(e) if e.is_client_error() => debug!("Rejected input: {e}"),
            Self::Model(e) => error!("Prediction error: {e}"),
        }
        let (status, code) = self.status();
        let mut response = create_error_response(status, code, &self.message(), None);
        if matches!(self, Self::Busy) {
            response.extensions_mut().insert(Busy);
        }
        response
    }

    /// The gRPC status, with the code that matches the HTTP status
    #[cfg(feature = "grpc")]
    pub(crate) fn into_status(self) -> tonic::Status {
        grpc_status(self.status().0, self.message())
    }
}

impl From<Refused> for InferError {
    fn from(refused: Refused) -> Self {
        match refused {
            Refused::Busy => Self::Busy,
            Refused::Lost => Self::Lost,
        }
    }
}

/// The gRPC status for an HTTP status
#[cfg(feature = "grpc")]
pub(crate) fn grpc_status(status: StatusCode, message: String) -> tonic::Status {
    match status {
        StatusCode::BAD_REQUEST => tonic::Status::invalid_argument(message),
        StatusCode::NOT_FOUND => tonic::Status::not_found(message),
        StatusCode::SERVICE_UNAVAILABLE => tonic::Status::resource_exhausted(message),
        StatusCode::GATEWAY_TIMEOUT => tonic::Status::deadline_exceeded(message),
        _ => tonic::Status::internal(message),
    }
}

impl ServerState {
    /// Run one request of named tensors through its model's pathway in the nervous system,
    /// the one way in for every protocol: take a concurrency slot or refuse at once, wait for
    /// the answer under the timeout (summed with concurrent requests when batching is on, on a
    /// blocking thread of its own otherwise), free the slot, and record a success's latency.
    pub(super) async fn infer(
        &self,
        slot: &ModelSlot,
        loaded: &Loaded,
        announced: Option<Announced<'_>>,
        inputs: Vec<HostTensor>,
    ) -> std::result::Result<Vec<HostTensor>, InferError> {
        let start = Instant::now();
        let permit = self
            .inference_slots
            .try_acquire()
            .map_err(|_| InferError::Busy)?;
        let task = self
            .serving
            .infer(&loaded.model.get_info().name, inputs, announced);
        let answered = match self.request_timeout {
            Some(timeout) => tokio::time::timeout(timeout, task)
                .await
                .map_err(|_| InferError::Timeout(timeout))?,
            None => task.await,
        };
        // Free the slot before the response is made, so a client that reads /metrics right
        // after its response never sees its own request still in flight
        drop(permit);
        let outputs = answered?.map_err(InferError::Model)?;
        let elapsed = start.elapsed();
        slot.inference_latency.observe(elapsed);
        self.inference_latency.observe(elapsed);
        Ok(outputs)
    }

    /// [`infer`](Self::infer) for a simple model: `batch_size` items of flat `f32` input, back
    /// to back, and the output the same way
    pub(super) async fn infer_flat(
        &self,
        slot: &ModelSlot,
        loaded: &Loaded,
        announced: Option<Announced<'_>>,
        data: Vec<f32>,
        batch_size: usize,
    ) -> std::result::Result<Vec<f32>, InferError> {
        let item = &loaded.model.get_info().input_spec.shape;
        let mut shape = Vec::with_capacity(item.len() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(item);
        let input = HostTensor {
            shape,
            values: Values::F32(data),
        };
        let outputs = self.infer(slot, loaded, announced, vec![input]).await?;
        match outputs.into_iter().next() {
            Some(HostTensor {
                values: Values::F32(values),
                ..
            }) => Ok(values),
            _ => Err(InferError::Model(
                ModelError::InferenceFailed("the model returned no f32 output".to_string()).into(),
            )),
        }
    }

    /// Count an inference request, whatever its protocol: received, failed, refused for want
    /// of a slot, and its latency from arrival to response. Requests the access check refused
    /// (401, 429) are counted by their own counters only.
    pub(super) fn count_request(&self, elapsed: Duration, failed: bool, busy: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.request_latency.observe(elapsed);
        if failed {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }
        if busy {
            self.rejected_busy.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Raw little-endian `f32` values; a trailing partial value is ignored (callers check lengths)
pub(crate) fn f32s_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// `f32` values as raw little-endian bytes
pub(crate) fn f32s_to_le(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    f32s_to_le_into(values, &mut bytes);
    bytes
}

/// [`f32s_to_le`], appended to `out`
pub(crate) fn f32s_to_le_into(values: &[f32], out: &mut Vec<u8>) {
    out.reserve(size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

/// Raw little-endian tensor bytes as values: `INT32` widens to `i64`
pub(crate) fn decode_raw(datatype: DataType, bytes: &[u8]) -> Values {
    match datatype {
        DataType::Fp32 => Values::F32(f32s_from_le(bytes)),
        DataType::Int64 => Values::I64(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect(),
        ),
        DataType::Int32 => Values::I64(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i64::from(i32::from_le_bytes(*b)))
                .collect(),
        ),
    }
}

/// Values as raw little-endian bytes
#[cfg(feature = "grpc")]
pub(crate) fn encode_raw(values: &Values) -> Vec<u8> {
    let mut bytes = Vec::new();
    encode_raw_into(values, &mut bytes);
    bytes
}

/// [`encode_raw`], appended to `out`
pub(crate) fn encode_raw_into(values: &Values, out: &mut Vec<u8>) {
    match values {
        Values::F32(v) => f32s_to_le_into(v, out),
        Values::I64(v) => {
            out.reserve(size_of_val(v.as_slice()));
            for value in v {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
}

/// How many bytes [`encode_raw`] makes of `values`
pub(crate) fn raw_len(values: &Values) -> usize {
    match values {
        Values::F32(v) => size_of_val(v.as_slice()),
        Values::I64(v) => size_of_val(v.as_slice()),
    }
}

/// The wire datatype of an output's values
pub(crate) fn values_datatype(values: &Values) -> DataType {
    match values {
        Values::F32(_) => DataType::Fp32,
        Values::I64(_) => DataType::Int64,
    }
}
