//! The KServe / Triton "Open Inference Protocol" (v2) over HTTP, so clients written for Triton or
//! KServe can talk to Furnace unchanged:
//! <https://github.com/kserve/open-inference-protocol/blob/main/specification/protocol/inference_rest.md>.
//!
//! Supported: server and model metadata, liveness and readiness, and inference with `FP32`,
//! `INT64` and `INT32` tensors, as JSON (flat or nested `data`) or with Triton's binary tensor
//! data extension (the default of `tritonclient.http`). The first dimension of a tensor's shape
//! is the batch.
//!
//! A simple model (one `f32` tensor in, one out) has one input named `input` and one output
//! named `output`, and goes through the same path as `POST /predict`: dynamic batching, the
//! concurrency limit, the timeout and the metrics. Any other model takes its graph's inputs by
//! name and returns its outputs (all, or those requested), one request per forward pass.

use crate::tensor::{DataType, HostTensor, Values};

use std::fmt;

use axum::extract::Path;
use axum::http::header::CONTENT_LENGTH;
use serde::de::{DeserializeSeed, SeqAccess};

use super::*;

/// Name of the model's input tensor in v2 metadata and requests
pub const INPUT_NAME: &str = "input";
/// Name of the model's output tensor
pub const OUTPUT_NAME: &str = "output";
/// The only model version
pub const MODEL_VERSION: &str = "1";
/// Header of the binary tensor data extension: length of the JSON part of the body
pub const INFERENCE_HEADER_CONTENT_LENGTH: HeaderName =
    HeaderName::from_static("inference-header-content-length");

/// The v2 routes, wired to the shared server state
pub(super) fn routes(state: AppState) -> Router<AppState> {
    let infer_route = post(infer).layer(middleware::from_fn_with_state(state, count_requests));
    Router::new()
        .route("/v2", get(server_metadata))
        .route("/v2/health/live", get(live))
        .route("/v2/health/ready", get(ready))
        .route("/v2/models/{model}", get(model_metadata))
        .route("/v2/models/{model}/versions/{version}", get(model_metadata))
        .route("/v2/models/{model}/ready", get(model_ready))
        .route(
            "/v2/models/{model}/versions/{version}/ready",
            get(model_ready),
        )
        .route("/v2/models/{model}/infer", infer_route.clone())
        .route("/v2/models/{model}/versions/{version}/infer", infer_route)
}

#[derive(Debug, Deserialize)]
struct ModelPath {
    model: String,
    version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerMetadata {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: String,
    /// `-1` for the batch dimension
    pub shape: Vec<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

async fn server_metadata() -> Json<ServerMetadata> {
    Json(ServerMetadata {
        name: "furnace".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        extensions: vec!["binary_tensor_data".to_string()],
    })
}

async fn live() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(state): State<AppState>) -> StatusCode {
    if state.ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// The model the path names, or the v2 error response for an unknown model or version
fn check_model(
    state: &ServerState,
    path: &ModelPath,
) -> std::result::Result<Arc<ModelSlot>, Box<Response>> {
    resolve_model(state, &path.model, path.version.as_deref().unwrap_or(""))
        .map_err(|message| Box::new(v2_error(StatusCode::NOT_FOUND, &message)))
}

/// The served model named `model` (an empty `version` means any), or why there is none.
/// Shared with the gRPC transport.
pub(super) fn resolve_model(
    state: &ServerState,
    model: &str,
    version: &str,
) -> std::result::Result<Arc<ModelSlot>, String> {
    let Some(slot) = state.slot(model) else {
        return Err(format!(
            "Unknown model '{model}'; this server serves {}",
            state.model_names().join(", ")
        ));
    };
    if !version.is_empty() && version != MODEL_VERSION {
        return Err(format!(
            "Unknown version '{version}' of model '{model}'; the only version is {MODEL_VERSION}"
        ));
    }
    Ok(slot.clone())
}

async fn model_metadata(State(state): State<AppState>, Path(path): Path<ModelPath>) -> Response {
    let slot = match check_model(&state, &path) {
        Ok(slot) => slot,
        Err(response) => return *response,
    };
    let loaded = slot.get();
    let info = loaded.model.get_info();
    let metadata = |tensors: &[crate::tensor::TensorInfo]| {
        tensors
            .iter()
            .map(|t| TensorMetadata {
                name: t.name.clone(),
                datatype: t.datatype.clone(),
                shape: t.shape.clone(),
            })
            .collect()
    };
    Json(ModelMetadata {
        name: info.name.clone(),
        versions: vec![MODEL_VERSION.to_string()],
        platform: format!("furnace_{}", info.model_type),
        inputs: metadata(&info.inputs),
        outputs: metadata(&info.outputs),
    })
    .into_response()
}

async fn model_ready(State(state): State<AppState>, Path(path): Path<ModelPath>) -> Response {
    match check_model(&state, &path) {
        Ok(slot) if slot.get().ready() => StatusCode::OK.into_response(),
        Ok(_) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Err(response) => *response,
    }
}

/// A v2 error: `{"error": "..."}` (plus Furnace's `code`, `details` and `timestamp`, which v2
/// clients ignore)
fn v2_error(status: StatusCode, message: &str) -> Response {
    let code = match status {
        StatusCode::NOT_FOUND => "NOT_FOUND",
        StatusCode::SERVICE_UNAVAILABLE => "SERVER_BUSY",
        _ => "INVALID_REQUEST",
    };
    create_error_response(status, code, message, None)
}

#[derive(Debug, Deserialize)]
struct InferRequest {
    id: Option<String>,
    #[serde(default)]
    parameters: Parameters,
    inputs: Vec<InferInput>,
    outputs: Option<Vec<RequestedOutput>>,
}

#[derive(Debug, Default, Deserialize)]
struct Parameters {
    /// Binary tensor data: the size of this input's raw bytes after the JSON
    binary_data_size: Option<usize>,
    /// Binary tensor data: return outputs as raw bytes (per request or per output)
    binary_data_output: Option<bool>,
    binary_data: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct InferInput {
    name: String,
    shape: Vec<i64>,
    datatype: String,
    #[serde(default)]
    parameters: Parameters,
    /// Decoded once the input's datatype is known
    data: Option<Box<serde_json::value::RawValue>>,
}

#[derive(Debug, Deserialize)]
struct RequestedOutput {
    name: String,
    #[serde(default)]
    parameters: Parameters,
}

/// Tensor data as `f32`, flattened in row-major order from a flat or nested JSON array without
/// building an intermediate `serde_json::Value`
#[derive(Debug, Default)]
struct TensorData32(Vec<f32>);

impl<'de> Deserialize<'de> for TensorData32 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let mut values = Vec::new();
        deserializer.deserialize_any(Flatten(&mut values))?;
        Ok(TensorData32(values))
    }
}

/// Appends every number of a (possibly nested) array to the vector
struct Flatten<'a>(&'a mut Vec<f32>);

impl<'de> DeserializeSeed<'de> for Flatten<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Flatten<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number or an array of numbers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        if let Some(hint) = seq.size_hint() {
            self.0.reserve(hint);
        }
        while seq.next_element_seed(Flatten(self.0))?.is_some() {}
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<(), E> {
        self.0.push(v as f32);
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<(), E> {
        self.0.push(v as f32);
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<(), E> {
        self.0.push(v as f32);
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferResponse {
    pub model_name: String,
    pub model_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub outputs: Vec<OutputTensor>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<usize>,
    /// Present unless the output is sent as binary tensor data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<OutputData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// An output's values in JSON
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OutputData {
    I64(Vec<i64>),
    F32(Vec<f32>),
}

/// Integer tensor data from a flat or nested JSON array; non-integers are an error
struct FlattenInts<'a>(&'a mut Vec<i64>);

impl<'de> DeserializeSeed<'de> for FlattenInts<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for FlattenInts<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an integer or an array of integers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        if let Some(hint) = seq.size_hint() {
            self.0.reserve(hint);
        }
        while seq.next_element_seed(FlattenInts(self.0))?.is_some() {}
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<(), E> {
        self.0.push(v);
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<(), E> {
        self.0
            .push(i64::try_from(v).map_err(|_| E::custom(format!("{v} is too large")))?);
        Ok(())
    }
}

/// Raw little-endian tensor bytes as values: `INT32` widens to `i64`
pub(super) fn decode_raw(datatype: DataType, bytes: &[u8]) -> Values {
    match datatype {
        DataType::Fp32 => Values::F32(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        ),
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
pub(super) fn encode_raw(values: &Values) -> Vec<u8> {
    match values {
        Values::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Values::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// The wire datatype of an output's values
pub(super) fn values_datatype(values: &Values) -> DataType {
    match values {
        Values::F32(_) => DataType::Fp32,
        Values::I64(_) => DataType::Int64,
    }
}

/// Where a request's tensor matches the model's input: its index, and the datatype it was sent
/// as. Shared with the gRPC transport.
pub(super) fn match_input(
    model: &Model,
    name: &str,
    datatype: &str,
) -> std::result::Result<(usize, DataType), String> {
    let inputs = model.inputs();
    let Some(index) = inputs.iter().position(|spec| spec.name == name) else {
        return Err(format!(
            "Unknown input '{name}'; the model takes {}",
            inputs
                .iter()
                .map(|spec| format!("{} ({})", spec.name, spec.dtype.v2_name()))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    };
    let spec = &inputs[index];
    let wire = DataType::from_v2_name(datatype)
        .filter(|wire| match spec.dtype {
            DataType::Fp32 => *wire == DataType::Fp32,
            DataType::Int64 | DataType::Int32 => wire.is_int(),
        })
        .ok_or_else(|| {
            format!(
                "Input '{name}' is {datatype}; the model takes {}",
                spec.dtype.v2_name()
            )
        })?;
    Ok((index, wire))
}

/// A request tensor's full shape, adding a batch of one when it was left out. Shared with the
/// gRPC transport.
pub(super) fn request_shape(
    model: &Model,
    index: usize,
    shape: &[i64],
) -> std::result::Result<Vec<usize>, String> {
    let mut dims: Vec<usize> = shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| format!("Invalid shape {shape:?}")))
        .collect::<std::result::Result<_, _>>()?;
    if dims.len() == model.inputs()[index].shape.len() {
        dims.insert(0, 1);
    }
    Ok(dims)
}

/// Requested outputs as (index into the model's outputs, sent as binary); all of them when the
/// request names none. Shared with the gRPC transport.
pub(super) fn select_outputs<'a>(
    model: &Model,
    requested: impl Iterator<Item = (&'a str, bool)>,
    all_binary: bool,
) -> std::result::Result<Vec<(usize, bool)>, String> {
    let outputs = model.outputs();
    let selected: Vec<(usize, bool)> = requested
        .map(|(name, binary)| {
            outputs
                .iter()
                .position(|spec| spec.name == name)
                .map(|index| (index, binary || all_binary))
                .ok_or_else(|| {
                    format!(
                        "Unknown output '{name}'; the model returns {}",
                        outputs
                            .iter()
                            .map(|spec| spec.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
        })
        .collect::<std::result::Result<_, _>>()?;
    Ok(if selected.is_empty() {
        (0..outputs.len()).map(|i| (i, all_binary)).collect()
    } else {
        selected
    })
}

/// The request's tensors, in the order of the model's inputs
fn decode_named_inputs(
    model: &Model,
    request: &mut InferRequest,
    binary: &[u8],
) -> std::result::Result<Vec<HostTensor>, String> {
    let mut tensors: Vec<Option<HostTensor>> = vec![None; model.inputs().len()];
    // Binary data of the inputs follows the JSON in the order the inputs are listed
    let mut offset = 0;
    for input in &mut request.inputs {
        let (index, wire) = match_input(model, &input.name, &input.datatype)?;
        if tensors[index].is_some() {
            return Err(format!("Input '{}' is given twice", input.name));
        }
        let shape = request_shape(model, index, &input.shape)?;
        let elements: usize = shape.iter().product();
        let values = match (input.parameters.binary_data_size, input.data.take()) {
            (Some(size), _) => {
                let end = offset + size;
                if size != elements * wire.size() || end > binary.len() {
                    return Err(format!(
                        "Input '{}': binary_data_size is {size} bytes; shape {:?} of {} needs {}                          and {} are left in the body",
                        input.name,
                        input.shape,
                        wire.v2_name(),
                        elements * wire.size(),
                        binary.len() - offset.min(binary.len())
                    ));
                }
                let values = decode_raw(wire, &binary[offset..end]);
                offset = end;
                values
            }
            (None, Some(raw)) => {
                let values = if wire == DataType::Fp32 {
                    serde_json::from_str::<TensorData32>(raw.get()).map(|d| Values::F32(d.0))
                } else {
                    let mut ints = Vec::new();
                    let mut deserializer = serde_json::Deserializer::from_str(raw.get());
                    FlattenInts(&mut ints)
                        .deserialize(&mut deserializer)
                        .map(|()| Values::I64(ints))
                };
                let values =
                    values.map_err(|e| format!("Input '{}': invalid data: {e}", input.name))?;
                if values.len() != elements {
                    return Err(format!(
                        "Input '{}' has shape {:?} ({elements} elements) but {} values",
                        input.name,
                        input.shape,
                        values.len()
                    ));
                }
                values
            }
            (None, None) => {
                return Err(format!(
                    "Input '{}' has neither data nor binary_data_size",
                    input.name
                ));
            }
        };
        tensors[index] = Some(HostTensor { shape, values });
    }
    tensors
        .into_iter()
        .zip(model.inputs())
        .map(|(tensor, spec)| tensor.ok_or_else(|| format!("Missing input '{}'", spec.name)))
        .collect()
}

/// Inference for a model of named tensors
async fn infer_named(
    state: &ServerState,
    slot: &ModelSlot,
    loaded: &Loaded,
    arrival: Option<crate::batching::Arrival<'_, crate::batching::NamedInput>>,
    mut request: InferRequest,
    binary: &[u8],
) -> Response {
    let model = &loaded.model;
    let inputs = match decode_named_inputs(model, &mut request, binary) {
        Ok(inputs) => inputs,
        Err(message) => return v2_error(StatusCode::BAD_REQUEST, &message),
    };
    let all_binary = request.parameters.binary_data_output == Some(true);
    let requested = request
        .outputs
        .iter()
        .flatten()
        .map(|o| (o.name.as_str(), o.parameters.binary_data == Some(true)));
    let selected = match select_outputs(model, requested, all_binary) {
        Ok(selected) => selected,
        Err(message) => return v2_error(StatusCode::BAD_REQUEST, &message),
    };

    let start_time = Instant::now();
    let permit = match state.inference_slots.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            warn!("All inference slots busy, rejecting request");
            return busy_response();
        }
    };
    let mut outputs = match run_general(state, loaded, arrival, inputs, permit).await {
        Ok(Ok(outputs)) => outputs,
        Ok(Err(e)) => {
            let status = match e {
                FurnaceError::Model(
                    ModelError::InputValidation { .. }
                    | ModelError::InvalidDataType(_)
                    | ModelError::InvalidInput(_),
                ) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return v2_error(status, &e.to_string());
        }
        Err(response) => return *response,
    };
    state.observe_inference(slot, start_time.elapsed());

    let mut raw = Vec::new();
    let mut tensors = Vec::with_capacity(selected.len());
    for (index, binary) in selected {
        let output = std::mem::replace(
            &mut outputs[index],
            HostTensor {
                shape: Vec::new(),
                values: Values::F32(Vec::new()),
            },
        );
        let datatype = values_datatype(&output.values).v2_name().to_string();
        let name = model.outputs()[index].name.clone();
        let (data, parameters) = if binary {
            let bytes = encode_raw(&output.values);
            let size = bytes.len();
            raw.extend(bytes);
            (None, Some(serde_json::json!({ "binary_data_size": size })))
        } else {
            let data = match output.values {
                Values::F32(v) => OutputData::F32(v),
                Values::I64(v) => OutputData::I64(v),
            };
            (Some(data), None)
        };
        tensors.push(OutputTensor {
            name,
            datatype,
            shape: output.shape,
            data,
            parameters,
        });
    }
    let response = InferResponse {
        model_name: model.get_info().name.clone(),
        model_version: MODEL_VERSION.to_string(),
        id: request.id.clone(),
        outputs: tensors,
    };
    if raw.is_empty() {
        return Json(response).into_response();
    }
    binary_response(&response, raw)
}

/// The binary tensor data extension's response: the JSON header, then the raw output bytes
fn binary_response(header: &InferResponse, raw: Vec<u8>) -> Response {
    let mut body = serde_json::to_vec(header).expect("serializable response");
    let json_len = body.len();
    body.extend(raw);
    let total = body.len();
    (
        [
            (CONTENT_TYPE, HeaderValue::from_static(OCTET_STREAM)),
            (INFERENCE_HEADER_CONTENT_LENGTH, HeaderValue::from(json_len)),
            (CONTENT_LENGTH, HeaderValue::from(total)),
        ],
        body,
    )
        .into_response()
}

async fn infer(
    State(state): State<AppState>,
    Path(path): Path<ModelPath>,
    request: Request,
) -> Response {
    let slot = match check_model(&state, &path) {
        Ok(slot) => slot,
        Err(response) => return *response,
    };
    let loaded = slot.get();
    // As in `/predict`: a batch about to start may wait for this request's body
    let arrival = loaded.batcher.as_ref().map(Batcher::arrival);
    let named_arrival = loaded.named_batcher.as_ref().map(Batcher::arrival);
    let header_len = match request.headers().get(&INFERENCE_HEADER_CONTENT_LENGTH) {
        None => None,
        Some(value) => match value
            .to_str()
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            Some(n) => Some(n),
            None => {
                return v2_error(
                    StatusCode::BAD_REQUEST,
                    "Inference-Header-Content-Length must be a non-negative integer",
                );
            }
        },
    };
    let body = match Bytes::from_request(request, &state).await {
        Ok(body) => body,
        Err(rejection) => return v2_error(rejection.status(), &rejection.body_text()),
    };
    let (json, binary) = match header_len {
        None => (&body[..], &[][..]),
        Some(n) if n <= body.len() => body.split_at(n),
        Some(n) => {
            return v2_error(
                StatusCode::BAD_REQUEST,
                &format!(
                    "Inference-Header-Content-Length is {n} but the body has {} bytes",
                    body.len()
                ),
            );
        }
    };

    let mut request: InferRequest = match serde_json::from_slice(json) {
        Ok(request) => request,
        Err(e) => {
            return v2_error(
                StatusCode::BAD_REQUEST,
                &format!("Invalid inference request: {e}"),
            );
        }
    };
    if !loaded.model.is_simple() {
        return infer_named(&state, &slot, &loaded, named_arrival, request, binary).await;
    }
    let (data, batch_size) = match decode_input(&loaded.model, &mut request, binary) {
        Ok(input) => input,
        Err(message) => return v2_error(StatusCode::BAD_REQUEST, &message),
    };
    if let Some(outputs) = &request.outputs
        && let Some(unknown) = outputs.iter().find(|o| o.name != OUTPUT_NAME)
    {
        return v2_error(
            StatusCode::BAD_REQUEST,
            &format!(
                "Unknown output '{}'; the model's output is '{OUTPUT_NAME}'",
                unknown.name
            ),
        );
    }
    let binary_output = request.parameters.binary_data_output == Some(true)
        || request
            .outputs
            .iter()
            .flatten()
            .any(|o| o.parameters.binary_data == Some(true));

    let start_time = Instant::now();
    let permit = match state.inference_slots.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            warn!("All inference slots busy, rejecting request");
            return busy_response();
        }
    };
    let output = match run_inference(&state, &loaded, arrival, data, batch_size, permit).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            let status = match e {
                FurnaceError::Model(
                    ModelError::InputValidation { .. }
                    | ModelError::InvalidDataType(_)
                    | ModelError::InvalidInput(_),
                ) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return v2_error(status, &e.to_string());
        }
        Err(response) => return *response,
    };
    state.observe_inference(&slot, start_time.elapsed());

    let info = loaded.model.get_info();
    let shape: Vec<usize> = std::iter::once(batch_size)
        .chain(info.output_spec.shape.iter().copied())
        .collect();
    let response = |data, parameters| InferResponse {
        model_name: info.name.clone(),
        model_version: MODEL_VERSION.to_string(),
        id: request.id.clone(),
        outputs: vec![OutputTensor {
            name: OUTPUT_NAME.to_string(),
            datatype: "FP32".to_string(),
            shape: shape.clone(),
            data,
            parameters,
        }],
    };
    if !binary_output {
        return Json(response(Some(OutputData::F32(output)), None)).into_response();
    }

    // Binary tensor data: the JSON header, then the output's raw little-endian bytes
    let raw = encode_raw(&Values::F32(output));
    let header = response(
        None,
        Some(serde_json::json!({ "binary_data_size": raw.len() })),
    );
    binary_response(&header, raw)
}

/// Check an input tensor's name, datatype and shape against the model; returns its batch size.
/// The shape is `[batch, ...item shape]`, or the item shape alone for one item. Shared with the
/// gRPC transport.
pub(super) fn check_input(
    model: &Model,
    name: &str,
    datatype: &str,
    shape: &[i64],
) -> std::result::Result<usize, String> {
    if name != INPUT_NAME {
        return Err(format!(
            "Unknown input '{name}'; the model's input is '{INPUT_NAME}'"
        ));
    }
    if datatype != "FP32" {
        return Err(format!(
            "Input datatype {datatype} is not supported; the model takes FP32"
        ));
    }
    let item_shape = &model.get_info().input_spec.shape;
    let dims: Vec<usize> = shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| format!("Invalid shape {shape:?}")))
        .collect::<std::result::Result<_, _>>()?;
    let batch_size = if dims.len() == item_shape.len() + 1 && dims[1..] == item_shape[..] {
        dims[0]
    } else if dims == *item_shape {
        1
    } else {
        let mut expected = vec!["N".to_string()];
        expected.extend(item_shape.iter().map(ToString::to_string));
        return Err(format!(
            "Input shape {shape:?} does not match the model's [{}]",
            expected.join(", ")
        ));
    };
    check_batch_len(batch_size)
        .map_err(|_| format!("Batch size {batch_size} is outside 1..={MAX_BATCH_SIZE}"))?;
    Ok(batch_size)
}

/// The request's one `FP32` input as flat values and its batch size
fn decode_input(
    model: &Model,
    request: &mut InferRequest,
    binary: &[u8],
) -> std::result::Result<(Vec<f32>, usize), String> {
    let [input] = request.inputs.as_mut_slice() else {
        return Err(format!(
            "Expected exactly one input ('{INPUT_NAME}'), got {}",
            request.inputs.len()
        ));
    };
    let batch_size = check_input(model, &input.name, &input.datatype, &input.shape)?;
    let item_size: usize = model.get_info().input_spec.shape.iter().product();
    let elements = batch_size * item_size;

    let data = match (input.parameters.binary_data_size, input.data.take()) {
        (Some(size), _) => {
            if size != elements * size_of::<f32>() || size > binary.len() {
                return Err(format!(
                    "binary_data_size is {size} bytes; shape {:?} needs {} and the body holds {}",
                    input.shape,
                    elements * size_of::<f32>(),
                    binary.len()
                ));
            }
            binary[..size]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
        (None, Some(raw)) => {
            let TensorData32(values) =
                serde_json::from_str(raw.get()).map_err(|e| format!("Invalid input data: {e}"))?;
            if values.len() != elements {
                return Err(format!(
                    "Input shape {:?} has {elements} elements, data has {}",
                    input.shape,
                    values.len()
                ));
            }
            values
        }
        (None, None) => return Err("Input has neither data nor binary_data_size".to_string()),
    };
    Ok((data, batch_size))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{send, test_app};
    use super::*;
    use axum::body::Body;
    use tower::util::ServiceExt;

    const MODEL: &str = "test_model";

    fn get(path: &str) -> Request<Body> {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    }

    fn infer_json(body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .uri(format!("/v2/models/{MODEL}/infer"))
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn values(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i % 97) as f32 * 0.01 + seed).sin())
            .collect()
    }

    /// The /predict output for the same items, to compare against
    async fn predict_outputs(app: &Router, items: &[Vec<f32>]) -> Vec<f32> {
        let body = serde_json::json!({ "inputs": items }).to_string();
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let (status, body) = send(app.clone(), request).await;
        assert_eq!(status, StatusCode::OK);
        let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
        response["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|o| {
                o.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect()
    }

    #[tokio::test]
    async fn metadata_and_health() {
        let app = test_app();
        let (status, body) = send(app.clone(), get("/v2")).await;
        assert_eq!(status, StatusCode::OK);
        let server: ServerMetadata = serde_json::from_slice(&body).unwrap();
        assert_eq!(server.name, "furnace");
        assert!(
            server
                .extensions
                .contains(&"binary_tensor_data".to_string())
        );

        assert_eq!(
            send(app.clone(), get("/v2/health/live")).await.0,
            StatusCode::OK
        );

        for path in [
            format!("/v2/models/{MODEL}"),
            format!("/v2/models/{MODEL}/versions/1"),
        ] {
            let (status, body) = send(app.clone(), get(&path)).await;
            assert_eq!(status, StatusCode::OK, "{path}");
            let model: ModelMetadata = serde_json::from_slice(&body).unwrap();
            assert_eq!(model.name, MODEL);
            assert_eq!(model.versions, vec!["1"]);
            assert_eq!(model.inputs[0].name, "input");
            assert_eq!(model.inputs[0].datatype, "FP32");
            assert_eq!(model.inputs[0].shape, vec![-1, 784]);
            assert_eq!(model.outputs[0].shape, vec![-1, 10]);
        }

        for path in ["/v2/models/other", "/v2/models/test_model/versions/2"] {
            let (status, body) = send(app.clone(), get(path)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(error["error"].is_string());
        }
    }

    #[tokio::test]
    async fn json_inference_matches_predict() {
        let app = test_app();
        let items = vec![values(784, 0.1), values(784, 0.7)];
        let expected = predict_outputs(&app, &items).await;

        // Flat data with an explicit batch, nested data, and versioned path
        let flat: Vec<f32> = items.concat();
        let bodies = [
            serde_json::json!({
                "id": "req-1",
                "inputs": [{ "name": "input", "shape": [2, 784], "datatype": "FP32", "data": flat }],
            }),
            serde_json::json!({
                "inputs": [{ "name": "input", "shape": [2, 784], "datatype": "FP32", "data": items }],
                "outputs": [{ "name": "output" }],
            }),
        ];
        for body in bodies {
            let (status, response) = send(app.clone(), infer_json(body)).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8_lossy(&response)
            );
            let response: InferResponse = serde_json::from_slice(&response).unwrap();
            assert_eq!(response.model_name, MODEL);
            assert_eq!(response.model_version, "1");
            let output = &response.outputs[0];
            assert_eq!(output.name, "output");
            assert_eq!(output.datatype, "FP32");
            assert_eq!(output.shape, vec![2, 10]);
            assert_eq!(output.data, Some(OutputData::F32(expected.clone())));
        }

        // A single item may leave out the batch dimension
        let body = serde_json::json!({
            "inputs": [{ "name": "input", "shape": [784], "datatype": "FP32", "data": items[0] }],
        });
        let (status, response) = send(app.clone(), infer_json(body)).await;
        assert_eq!(status, StatusCode::OK);
        let response: InferResponse = serde_json::from_slice(&response).unwrap();
        assert_eq!(response.outputs[0].shape, vec![1, 10]);
        assert_eq!(
            response.outputs[0].data,
            Some(OutputData::F32(expected[..10].to_vec()))
        );
    }

    #[tokio::test]
    async fn binary_tensor_data_in_and_out() {
        let app = test_app();
        let items = vec![values(784, 0.3), values(784, 0.9), values(784, 0.5)];
        let expected = predict_outputs(&app, &items).await;

        let raw: Vec<u8> = items
            .concat()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let header = serde_json::json!({
            "inputs": [{
                "name": "input", "shape": [3, 784], "datatype": "FP32",
                "parameters": { "binary_data_size": raw.len() },
            }],
            "outputs": [{ "name": "output", "parameters": { "binary_data": true } }],
        })
        .to_string();
        let mut body = header.clone().into_bytes();
        body.extend_from_slice(&raw);
        let request = Request::builder()
            .uri(format!("/v2/models/{MODEL}/infer"))
            .method("POST")
            .header(CONTENT_TYPE, OCTET_STREAM)
            .header(INFERENCE_HEADER_CONTENT_LENGTH, header.len())
            .body(Body::from(body))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json_len: usize = response.headers()[&INFERENCE_HEADER_CONTENT_LENGTH]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let (json, raw) = body.split_at(json_len);
        let header: InferResponse = serde_json::from_slice(json).unwrap();
        let output = &header.outputs[0];
        assert_eq!(output.shape, vec![3, 10]);
        assert!(output.data.is_none());
        assert_eq!(
            output.parameters.as_ref().unwrap()["binary_data_size"],
            raw.len()
        );
        let values: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(values, expected);
    }

    fn tensors_app() -> Router {
        let model = Model::new(
            Box::new(crate::model::DummyTensorsModel::new(None)),
            "dummy",
            0,
        );
        crate::api::router(model, &ServerConfig::default())
    }

    fn infer_at(model: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .uri(format!("/v2/models/{model}/infer"))
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn named_tensors_metadata_and_json() {
        let app = tensors_app();
        let (status, body) = send(app.clone(), get("/v2/models/tensors")).await;
        assert_eq!(status, StatusCode::OK);
        let model: ModelMetadata = serde_json::from_slice(&body).unwrap();
        let described: Vec<_> = model
            .inputs
            .iter()
            .chain(&model.outputs)
            .map(|t| (t.name.as_str(), t.datatype.as_str(), t.shape.clone()))
            .collect();
        assert_eq!(
            described,
            [
                ("ids", "INT64", vec![-1, 4]),
                ("scale", "FP32", vec![-1]),
                ("scores", "FP32", vec![-1, 4]),
                ("count", "INT64", vec![-1]),
            ]
        );

        // Inputs in any order, nested or flat; INT32 is taken for an INT64 input
        let body = serde_json::json!({
            "id": "q1",
            "inputs": [
                { "name": "scale", "shape": [2], "datatype": "FP32", "data": [0.5, 2.0] },
                { "name": "ids", "shape": [2, 4], "datatype": "INT32", "data": [[1, 2, 3, 4], [5, 6, 7, 8]] },
            ],
        });
        let (status, body) = send(app.clone(), infer_at("tensors", body)).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let response: InferResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.id.as_deref(), Some("q1"));
        let scores = &response.outputs[0];
        assert_eq!(
            (scores.name.as_str(), scores.datatype.as_str()),
            ("scores", "FP32")
        );
        assert_eq!(scores.shape, [2, 4]);
        assert_eq!(
            scores.data,
            Some(OutputData::F32(vec![
                0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0
            ]))
        );
        let count = &response.outputs[1];
        assert_eq!(
            (count.name.as_str(), count.datatype.as_str()),
            ("count", "INT64")
        );
        assert_eq!(count.data, Some(OutputData::I64(vec![10, 26])));

        // Only the outputs asked for, in that order
        let body = serde_json::json!({
            "inputs": [
                { "name": "ids", "shape": [1, 4], "datatype": "INT64", "data": [1, 1, 1, 1] },
                { "name": "scale", "shape": [1], "datatype": "FP32", "data": [3.0] },
            ],
            "outputs": [{ "name": "count" }],
        });
        let (status, body) = send(app.clone(), infer_at("tensors", body)).await;
        assert_eq!(status, StatusCode::OK);
        let response: InferResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].data, Some(OutputData::I64(vec![4])));

        // /predict cannot take named tensors and says where to send them
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"input": [1, 2, 3, 4]}"#))
            .unwrap();
        let (status, body) = send(app.clone(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error = String::from_utf8_lossy(&body);
        assert!(error.contains("/v2/models/tensors/infer"), "{error}");
    }

    #[tokio::test]
    async fn concurrent_named_tensor_requests_are_answered_each_their_own() {
        let app = tensors_app();
        let request = |id: i64| {
            infer_at(
                "tensors",
                serde_json::json!({
                    "inputs": [
                        { "name": "ids", "shape": [1, 4], "datatype": "INT64", "data": [id, id, id, id] },
                        { "name": "scale", "shape": [1], "datatype": "FP32", "data": [1.0] },
                    ],
                    "outputs": [{ "name": "count" }],
                }),
            )
        };
        let tasks: Vec<_> = (1..=32)
            .map(|id| {
                let app = app.clone();
                tokio::spawn(async move { (id, send(app, request(id)).await) })
            })
            .collect();
        for task in tasks {
            let (id, (status, body)) = task.await.unwrap();
            assert_eq!(status, StatusCode::OK);
            let response: InferResponse = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                response.outputs[0].data,
                Some(OutputData::I64(vec![4 * id]))
            );
        }
    }

    #[tokio::test]
    async fn named_tensors_binary_in_and_out() {
        let app = tensors_app();
        let ids: Vec<u8> = [1i64, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let scale = 2.0f32.to_le_bytes();
        let header = serde_json::json!({
            "inputs": [
                { "name": "ids", "shape": [1, 4], "datatype": "INT64",
                  "parameters": { "binary_data_size": ids.len() } },
                { "name": "scale", "shape": [1], "datatype": "FP32",
                  "parameters": { "binary_data_size": 4 } },
            ],
            "outputs": [
                { "name": "scores", "parameters": { "binary_data": true } },
                { "name": "count" },
            ],
        })
        .to_string();
        let mut body = header.clone().into_bytes();
        body.extend_from_slice(&ids);
        body.extend_from_slice(&scale);
        let request = Request::builder()
            .uri("/v2/models/tensors/infer")
            .method("POST")
            .header(CONTENT_TYPE, OCTET_STREAM)
            .header(INFERENCE_HEADER_CONTENT_LENGTH, header.len())
            .body(Body::from(body))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json_len: usize = response.headers()[&INFERENCE_HEADER_CONTENT_LENGTH]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let (json, raw) = body.split_at(json_len);
        let header: InferResponse = serde_json::from_slice(json).unwrap();
        // scores as raw bytes, count as JSON
        assert_eq!(header.outputs[0].data, None);
        assert_eq!(raw.len(), 16);
        assert_eq!(
            decode_raw(DataType::Fp32, raw),
            Values::F32(vec![2.0, 4.0, 6.0, 8.0])
        );
        assert_eq!(header.outputs[1].data, Some(OutputData::I64(vec![10])));
    }

    #[tokio::test]
    async fn named_tensors_invalid_requests() {
        let app = tensors_app();
        let ids = serde_json::json!({ "name": "ids", "shape": [1, 4], "datatype": "INT64", "data": [1, 2, 3, 4] });
        let scale =
            serde_json::json!({ "name": "scale", "shape": [1], "datatype": "FP32", "data": [1.0] });
        let cases = [
            (
                serde_json::json!({ "inputs": [ids] }),
                "Missing input 'scale'",
            ),
            (
                serde_json::json!({ "inputs": [ids, scale, { "name": "extra", "shape": [1], "datatype": "FP32", "data": [1] }] }),
                "Unknown input 'extra'",
            ),
            (
                serde_json::json!({ "inputs": [ids, ids, scale] }),
                "given twice",
            ),
            (
                serde_json::json!({ "inputs": [{ "name": "ids", "shape": [1, 4], "datatype": "FP32", "data": [1, 2, 3, 4] }, scale] }),
                "Input 'ids' is FP32; the model takes INT64",
            ),
            (
                serde_json::json!({ "inputs": [{ "name": "ids", "shape": [1, 4], "datatype": "INT64", "data": [1.5, 2, 3, 4] }, scale] }),
                "invalid data",
            ),
            (
                serde_json::json!({ "inputs": [{ "name": "ids", "shape": [1, 3], "datatype": "INT64", "data": [1, 2, 3] }, scale] }),
                "'ids' has shape [1, 3]",
            ),
            (
                serde_json::json!({ "inputs": [ids, scale], "outputs": [{ "name": "logits" }] }),
                "Unknown output 'logits'",
            ),
        ];
        for (body, expected) in cases {
            let (status, response) = send(app.clone(), infer_at("tensors", body.clone())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            let error: serde_json::Value = serde_json::from_slice(&response).unwrap();
            let message = error["error"].as_str().unwrap();
            assert!(
                message.contains(expected),
                "{expected:?} not in {message:?}"
            );
        }
    }

    #[tokio::test]
    async fn invalid_requests_get_400_with_an_error() {
        let app = test_app();
        let input = |name: &str,
                     shape: serde_json::Value,
                     datatype: &str,
                     data: serde_json::Value| {
            serde_json::json!({
                "inputs": [{ "name": name, "shape": shape, "datatype": datatype, "data": data }],
            })
        };
        let cases = [
            input(
                "input",
                serde_json::json!([1, 784]),
                "INT64",
                serde_json::json!(vec![1; 784]),
            ),
            input(
                "pixels",
                serde_json::json!([1, 784]),
                "FP32",
                serde_json::json!(vec![0.5; 784]),
            ),
            input(
                "input",
                serde_json::json!([1, 783]),
                "FP32",
                serde_json::json!(vec![0.5; 783]),
            ),
            input(
                "input",
                serde_json::json!([2, 784]),
                "FP32",
                serde_json::json!(vec![0.5; 784]),
            ),
            input(
                "input",
                serde_json::json!([1, 784]),
                "FP32",
                serde_json::json!("x"),
            ),
            serde_json::json!({ "inputs": [] }),
            serde_json::json!({
                "inputs": [{ "name": "input", "shape": [1, 784], "datatype": "FP32", "data": vec![0.5; 784] }],
                "outputs": [{ "name": "logits" }],
            }),
        ];
        for body in cases {
            let (status, response) = send(app.clone(), infer_json(body.clone())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            let error: serde_json::Value = serde_json::from_slice(&response).unwrap();
            assert!(error["error"].is_string(), "{error}");
        }
        // Unknown model
        let request = Request::builder()
            .uri("/v2/models/other/infer")
            .method("POST")
            .body(Body::from("{}"))
            .unwrap();
        assert_eq!(send(app.clone(), request).await.0, StatusCode::NOT_FOUND);
    }
}
