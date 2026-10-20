//! Models as the server runs them.
//!
//! [`BurnModel`] is the one trait every model implements, however it was defined
//! ([`Model::from_fn`], [`Model::custom`], a built-in ONNX model). [`Model`] wraps one with its
//! metadata and statistics and runs requests through a single pipeline: check each request's
//! tensors against the model's inputs, join the valid ones along the batch dimension, run them
//! (in chunks of a fixed batch size if the graph has one), check the outputs and split them
//! back. The flat `Vec<f32>` methods (`predict*`) are thin adapters over that pipeline.

use burn::tensor::Tensor;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use tracing::info;

use crate::backend::BackendKind;
use crate::burn_model::{BurnModelContainer, create_sample_model};
use crate::error::{FurnaceError, ModelError, Result};
use crate::models::{BuiltInModel, resolve_weights};
use crate::tensor::{DataType, HostTensor, IoSpec, TensorInfo, TensorSpec, Values};

mod kinds;

use kinds::FnModel;
pub use kinds::{DummyModel, DummyTensorsModel, MlpModel};

/// Hard upper bound on the number of items in one inference batch
pub const MAX_BATCH_SIZE: usize = 1000;

/// What `GET /model/info` reports about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The name clients use: `/v2/models/{name}`, `/models/{name}/predict`
    pub name: String,
    pub version: String,
    /// The first input. For a model with several, see `inputs`
    pub input_spec: TensorSpec,
    /// The first output. For a model with several, see `outputs`
    pub output_spec: TensorSpec,
    /// Every input, with the names `/v2/models/{model}/infer` takes
    pub inputs: Vec<TensorInfo>,
    /// Every output, in the order the model returns them
    pub outputs: Vec<TensorInfo>,
    /// How the model was defined: `burn`, `onnx`, `mlp`, ...
    pub model_type: String,
    /// The backend it runs on, e.g. `cpu`
    pub backend: String,
    /// When it was loaded, RFC 3339
    pub created_at: String,
    /// Size of its weight file; 0 when there is none
    pub model_size_bytes: u64,
}

/// A model's request counters, as `/metrics` reports them.
///
/// Counts are per request; timings are per forward pass, so requests batched together each
/// record the time of the pass they shared.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// Requests answered, successfully or not
    pub inference_count: u64,
    /// Sum of the forward-pass times recorded for successful requests
    pub total_inference_time_ms: f64,
    /// When the last successful request finished
    pub last_inference_time: Option<DateTime<Utc>>,
    /// `total_inference_time_ms / success_count`
    pub average_inference_time_ms: f64,
    /// Requests that failed, whether their inputs were invalid or the model failed
    pub error_count: u64,
    /// Requests that got their outputs
    pub success_count: u64,
    /// Fastest forward pass a successful request was part of
    pub min_inference_time_ms: Option<f64>,
    /// Slowest forward pass a successful request was part of
    pub max_inference_time_ms: Option<f64>,
    /// Forward passes run. Below `success_count` when requests were batched together
    pub batch_count: u64,
    /// Items processed by those forward passes
    pub batched_items: u64,
}

impl ModelStats {
    fn new() -> Self {
        Self {
            inference_count: 0,
            total_inference_time_ms: 0.0,
            last_inference_time: None,
            average_inference_time_ms: 0.0,
            error_count: 0,
            success_count: 0,
            min_inference_time_ms: None,
            max_inference_time_ms: None,
            batch_count: 0,
            batched_items: 0,
        }
    }

    /// `now` is read before the lock is taken, so the clock is not read while holding it
    fn record_success(&mut self, elapsed_ms: f64, now: DateTime<Utc>) {
        self.inference_count += 1;
        self.success_count += 1;
        self.total_inference_time_ms += elapsed_ms;
        self.average_inference_time_ms = self.total_inference_time_ms / self.success_count as f64;
        self.min_inference_time_ms = Some(
            self.min_inference_time_ms
                .map_or(elapsed_ms, |m| m.min(elapsed_ms)),
        );
        self.max_inference_time_ms = Some(
            self.max_inference_time_ms
                .map_or(elapsed_ms, |m| m.max(elapsed_ms)),
        );
        self.last_inference_time = Some(now);
    }

    fn record_error(&mut self) {
        self.inference_count += 1;
        self.error_count += 1;
    }

    fn record_batch(&mut self, items: usize) {
        self.batch_count += 1;
        self.batched_items += items as u64;
    }
}

/// One request's input for [`Model::predict_many`]: `batch_size` items back to back
#[derive(Debug)]
pub struct FlatInput {
    /// The items' values, row-major, `batch_size * input size` of them
    pub data: Vec<f32>,
    /// Items in `data`
    pub batch_size: usize,
}

/// A model the server can run: what it takes, what it returns, and how to run it.
///
/// Implement this for your own Burn model to serve it with every feature of the server (named
/// tensors over the v2 protocol, metrics, reloads, ...). A model that takes one float tensor and
/// returns one needs none of this: see [`Model::from_fn`].
///
/// ```ignore
/// #[derive(Debug)]
/// struct Classifier { net: MyNet, device: Device }
///
/// impl BurnModel for Classifier {
///     fn name(&self) -> &str { "classifier" }
///     fn backend(&self) -> BackendKind { BackendKind::Cpu }
///     fn inputs(&self) -> Vec<IoSpec> {
///         vec![IoSpec::new("input_ids", DataType::Int64, &[Some(128)]),
///              IoSpec::new("attention_mask", DataType::Int64, &[Some(128)])]
///     }
///     fn outputs(&self) -> Vec<IoSpec> { vec![IoSpec::fp32("logits", &[2])] }
///     fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
///         let [ids, mask] = <[HostTensor; 2]>::try_from(inputs).unwrap();
///         let logits = self.net.forward(ids.into_int::<2>(&self.device)?,
///                                       mask.into_int::<2>(&self.device)?);
///         Ok(vec![HostTensor::from_float(logits)?])
///     }
/// }
/// ```
pub trait BurnModel: Send + Sync + std::fmt::Debug {
    /// The name clients use: `/v2/models/{name}`, `/models/{name}/predict`
    fn name(&self) -> &str;

    /// The backend the model runs on, as reported by `/model/info`
    fn backend(&self) -> BackendKind;

    /// The inputs, in the order [`BurnModel::run`] takes them
    fn inputs(&self) -> Vec<IoSpec>;

    /// The outputs, in the order [`BurnModel::run`] returns them
    fn outputs(&self) -> Vec<IoSpec>;

    /// Run one batch. The server has already checked the inputs against
    /// [`BurnModel::inputs`]: one tensor each, in that order, with the batch as the first
    /// dimension, every fixed dimension as declared, and the same batch size throughout.
    /// Blocking; the server calls it off the async runtime.
    fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>>;

    /// The batch size the model was built for, if it takes no other: larger batches are then
    /// run in chunks of it
    fn fixed_batch(&self) -> Option<usize> {
        None
    }
}

/// A loaded model plus its metadata and runtime statistics: what the server serves.
///
/// Build one with [`Model::from_fn`], [`Model::custom`] or the `load_*` functions. Requests
/// run through [`Model::infer`] / [`Model::infer_many`] (named tensors), or the flat
/// [`Model::predict`] family for a model of one `f32` tensor in and one out. All of them block;
/// call them off the async runtime.
#[derive(Debug)]
pub struct Model {
    inner: Box<dyn BurnModel>,
    info: ModelInfo,
    model_type: &'static str,
    inputs: Vec<IoSpec>,
    outputs: Vec<IoSpec>,
    /// One `f32` tensor in and one out, both with fixed per-item shapes
    simple: bool,
    /// The range of the backend's integers, for integer inputs
    int_range: (i64, i64),
    /// Shared with the model this one replaced on a reload, so requests still running on the
    /// old one are counted too
    stats: std::sync::Arc<Mutex<ModelStats>>,
}

/// Which entry point a request came in through. It decides only how invalid input is reported:
/// the flat `predict*` methods keep the error variants and messages they have always returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Door {
    Named,
    Flat,
}

/// The legacy single-tensor description: per-item shape (open dimensions as 0) and dtype
fn tensor_spec(spec: Option<&IoSpec>) -> TensorSpec {
    TensorSpec {
        shape: spec
            .map(|s| s.shape.iter().map(|d| d.unwrap_or(0)).collect())
            .unwrap_or_default(),
        dtype: match spec.map(|s| s.dtype) {
            Some(DataType::Int64) => "int64",
            Some(DataType::Int32) => "int32",
            _ => "float32",
        }
        .to_string(),
        min_value: None,
        max_value: None,
    }
}

impl Model {
    /// Serve your own model (see [`BurnModel`])
    pub fn custom(model: impl BurnModel + 'static) -> Self {
        Self::new(Box::new(model), "burn", 0)
    }

    /// Serve a Burn forward function that takes one float tensor of rank `DI` (batch first) and
    /// returns one of rank `DO`, e.g. a `Module`'s `forward`. `input_shape` and `output_shape` are
    /// the per-item shapes, without the batch dimension. Such a model also serves `/predict`
    /// and batches concurrent requests together.
    ///
    /// ```ignore
    /// let device = BackendKind::Cpu.device();
    /// let net = MyNet::new(&device); // load weights with furnace::weights::load_burnpack
    /// let model = Model::from_fn("digits", BackendKind::Cpu, &[1, 28, 28], &[10],
    ///     move |x: Tensor<4>| net.forward(x))?;
    /// ```
    pub fn from_fn<const DI: usize, const DO: usize, F>(
        name: &str,
        backend: BackendKind,
        input_shape: &[usize],
        output_shape: &[usize],
        forward: F,
    ) -> Result<Self>
    where
        F: Fn(Tensor<DI>) -> Tensor<DO> + Send + Sync + 'static,
    {
        for (what, shape, rank) in [("input", input_shape, DI), ("output", output_shape, DO)] {
            if shape.len() + 1 != rank {
                return Err(ModelError::InvalidArgument {
                    arg: format!("{what}_shape"),
                    value: format!("{shape:?}"),
                    reason: format!(
                        "the forward function's {what} has rank {rank}, so the per-item shape \
                         needs {} dimensions (the batch comes first)",
                        rank.saturating_sub(1)
                    ),
                }
                .into());
            }
        }
        Ok(Self::custom(FnModel {
            name: name.to_string(),
            backend,
            device: backend.device(),
            input: IoSpec::fp32("input", input_shape),
            output: IoSpec::fp32("output", output_shape),
            forward,
        }))
    }

    /// Wrap a model; `model_type` and `model_size_bytes` are reported by `/model/info`
    pub fn new(inner: Box<dyn BurnModel>, model_type: &'static str, model_size_bytes: u64) -> Self {
        let inputs = inner.inputs();
        let outputs = inner.outputs();
        let int_range = crate::tensor::int_range(&inner.backend().device());
        let simple = match (inputs.as_slice(), outputs.as_slice()) {
            ([input], [output]) => [input, output].iter().all(|spec| {
                spec.dtype == DataType::Fp32
                    && spec.static_shape().is_some_and(|shape| !shape.is_empty())
            }),
            _ => false,
        };
        let info = ModelInfo {
            name: inner.name().to_string(),
            version: "1.0.0".to_string(),
            input_spec: tensor_spec(inputs.first()),
            output_spec: tensor_spec(outputs.first()),
            inputs: inputs.iter().map(TensorInfo::from).collect(),
            outputs: outputs.iter().map(TensorInfo::from).collect(),
            model_type: model_type.to_string(),
            backend: inner.backend().to_string(),
            created_at: Utc::now().to_rfc3339(),
            model_size_bytes,
        };

        Self {
            inner,
            info,
            model_type,
            int_range,
            inputs,
            outputs,
            simple,
            stats: std::sync::Arc::new(Mutex::new(ModelStats::new())),
        }
    }

    /// Whether concurrent requests of named tensors can be joined into one batch: every input
    /// has a fixed per-item shape. (Simple models batch on the flat path instead.)
    pub fn batches_named(&self) -> bool {
        !self.simple && self.inputs.iter().all(|spec| spec.static_shape().is_some())
    }

    /// Whether the model takes one `f32` tensor and returns one, with fixed per-item shapes.
    /// Such models serve `/predict` and batch concurrent requests together; the others take
    /// named tensors through [`Model::infer`].
    pub fn is_simple(&self) -> bool {
        self.simple
    }

    /// The inputs, in the order [`Model::infer`] takes them
    pub fn inputs(&self) -> &[IoSpec] {
        &self.inputs
    }

    /// The outputs, in the order [`Model::infer`] returns them
    pub fn outputs(&self) -> &[IoSpec] {
        &self.outputs
    }

    /// Run one request of named tensors: `inputs` in the order of [`Model::inputs`], each with
    /// the batch as its first dimension. Returns the outputs in the order of [`Model::outputs`].
    /// Blocking, like [`Model::predict_batch`].
    pub fn infer(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        self.infer_many(vec![inputs])
            .pop()
            .expect("one request yields one result")
    }

    /// Several requests of named tensors as one forward pass, joined along the batch
    /// dimension, the outputs split back. Each request is checked on its own, so an invalid one
    /// gets its own error and the rest still run. Blocking, like [`Model::predict_batch`].
    pub fn infer_many(&self, requests: Vec<Vec<HostTensor>>) -> Vec<Result<Vec<HostTensor>>> {
        self.run_requests(requests.into_iter().map(Ok).collect(), Door::Named)
    }

    /// The pipeline behind every entry point. A request that arrives as an error (one the flat
    /// adapters found) is counted and answered with it.
    fn run_requests(
        &self,
        requests: Vec<Result<Vec<HostTensor>>>,
        door: Door,
    ) -> Vec<Result<Vec<HostTensor>>> {
        let mut results: Vec<Option<Result<Vec<HostTensor>>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut valid = Vec::with_capacity(requests.len());
        let mut errors = 0;
        for (i, inputs) in requests.into_iter().enumerate() {
            let checked = inputs.and_then(|inputs| {
                self.check_inputs(&inputs, door)
                    .map(|batch| (inputs, batch))
            });
            match checked {
                Ok((inputs, batch)) => valid.push((i, inputs, batch)),
                Err(e) => {
                    results[i] = Some(Err(e));
                    errors += 1;
                }
            }
        }

        let start = Instant::now();
        let total: usize = valid.iter().map(|(_, _, batch)| batch).sum();
        let batches: Vec<(usize, usize)> = valid.iter().map(|(i, _, b)| (*i, *b)).collect();
        let output = match valid.len() {
            0 => None,
            1 => {
                let (_, inputs, _) = valid.pop().expect("one valid request");
                Some(self.run_checked(inputs, total))
            }
            _ => Some(
                join_requests(valid.into_iter().map(|(_, inputs, _)| inputs).collect())
                    .and_then(|inputs| self.run_checked(inputs, total)),
            ),
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let now = Utc::now();

        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..errors {
            stats.record_error();
        }
        if let Some(output) = output {
            stats.record_batch(total);
            match output {
                Ok(outputs) if batches.len() == 1 => {
                    stats.record_success(elapsed_ms, now);
                    results[batches[0].0] = Some(Ok(outputs));
                }
                Ok(outputs) => {
                    let mut offset = 0;
                    for (i, batch) in batches {
                        let mine = outputs
                            .iter()
                            .map(|output| output.slice_batch(offset, offset + batch))
                            .collect();
                        offset += batch;
                        stats.record_success(elapsed_ms, now);
                        results[i] = Some(Ok(mine));
                    }
                }
                // A lone request keeps its error; a shared failure is reported to each
                Err(e) if batches.len() == 1 => {
                    stats.record_error();
                    results[batches[0].0] = Some(Err(e));
                }
                Err(e) => {
                    for (i, _) in batches {
                        stats.record_error();
                        results[i] = Some(Err(duplicate(&e)));
                    }
                }
            }
        }
        results
            .into_iter()
            .map(|r| r.expect("every request has a result"))
            .collect()
    }

    fn run_checked(&self, inputs: Vec<HostTensor>, batch: usize) -> Result<Vec<HostTensor>> {
        let outputs = self.run_in_chunks(inputs, batch)?;
        self.check_outputs(&outputs, batch)?;
        Ok(outputs)
    }

    /// The batch size, if the inputs fit the model
    fn check_inputs(&self, inputs: &[HostTensor], door: Door) -> Result<usize> {
        let invalid =
            |message: String| -> FurnaceError { ModelError::InvalidInput(message).into() };
        if inputs.len() != self.inputs.len() {
            return Err(invalid(format!(
                "expected {} inputs ({}), got {}",
                self.inputs.len(),
                self.inputs
                    .iter()
                    .map(|i| i.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                inputs.len()
            )));
        }
        let mut batch = None;
        for (spec, tensor) in self.inputs.iter().zip(inputs) {
            let name = &spec.name;
            match (&tensor.values, spec.dtype.is_int()) {
                (Values::F32(values), false) => {
                    if let Some(pos) = values.iter().position(|v| !v.is_finite()) {
                        if door == Door::Flat {
                            let per_item: usize = tensor.shape.iter().skip(1).product();
                            return Err(ModelError::InvalidDataType(format!(
                                "Invalid value at batch[{}][{}]: {} (not finite)",
                                pos / per_item,
                                pos % per_item,
                                values[pos]
                            ))
                            .into());
                        }
                        return Err(invalid(format!(
                            "'{name}' has a non-finite value at index {pos}"
                        )));
                    }
                }
                (Values::I64(values), true) => {
                    let (min, max) = self.int_range;
                    if let Some(pos) = values.iter().position(|v| !(min..=max).contains(v)) {
                        return Err(invalid(format!(
                            "'{name}' has {} at index {pos}, outside {min}..={max}, the range \
                             of this backend's integers",
                            values[pos]
                        )));
                    }
                }
                _ => {
                    return Err(invalid(format!(
                        "'{name}' must be {}",
                        spec.dtype.v2_name()
                    )));
                }
            }
            // The batch dimension comes first: a rank-0 tensor has none, and joining or splitting
            // it would index past its shape
            if tensor.shape.is_empty() {
                return Err(invalid(format!(
                    "'{name}' has no batch dimension: its shape is [], the model takes {:?}",
                    spec.v2_shape()
                )));
            }
            let item = &tensor.shape[1..];
            let fits = item.len() == spec.shape.len()
                && item
                    .iter()
                    .zip(&spec.shape)
                    .all(|(&d, want)| want.is_none_or(|w| w == d) && d > 0);
            if !fits {
                return Err(invalid(format!(
                    "'{name}' has shape {:?}; the model takes {:?} (-1: any)",
                    tensor.shape,
                    spec.v2_shape()
                )));
            }
            let expected_len: usize = tensor.shape.iter().product();
            if tensor.values.len() != expected_len {
                return Err(invalid(format!(
                    "'{name}' has shape {:?} ({expected_len} values) but {} values",
                    tensor.shape,
                    tensor.values.len()
                )));
            }
            match batch {
                None => batch = Some(tensor.batch()),
                Some(b) if b != tensor.batch() => {
                    return Err(invalid(format!(
                        "inputs disagree on the batch size: {b} and {} ('{name}')",
                        tensor.batch()
                    )));
                }
                Some(_) => {}
            }
        }
        let batch = batch.unwrap_or(0);
        Self::check_batch_size(batch)?;
        Ok(batch)
    }

    /// Run the inputs, in chunks of the graph's fixed batch size if it has one
    fn run_in_chunks(&self, inputs: Vec<HostTensor>, batch: usize) -> Result<Vec<HostTensor>> {
        let chunk = match self.inner.fixed_batch() {
            Some(chunk) if chunk != batch => chunk,
            _ => return self.inner.run(inputs),
        };
        if !batch.is_multiple_of(chunk) {
            return Err(ModelError::InvalidInput(format!(
                "the model runs batches of {chunk}; got {batch} items"
            ))
            .into());
        }
        let mut per_output: Vec<Vec<HostTensor>> = Vec::new();
        for start in (0..batch).step_by(chunk) {
            let part = inputs
                .iter()
                .map(|t| t.slice_batch(start, start + chunk))
                .collect();
            for (i, output) in self.inner.run(part)?.into_iter().enumerate() {
                if per_output.len() <= i {
                    per_output.push(Vec::new());
                }
                per_output[i].push(output);
            }
        }
        per_output
            .into_iter()
            .map(HostTensor::concat_batch)
            .collect()
    }

    fn check_outputs(&self, outputs: &[HostTensor], batch: usize) -> Result<()> {
        if outputs.len() != self.outputs.len() {
            return Err(ModelError::InferenceFailed(format!(
                "model returned {} outputs, expected {}",
                outputs.len(),
                self.outputs.len()
            ))
            .into());
        }
        if let Some(output) = outputs.iter().find(|o| o.batch() != batch) {
            return Err(ModelError::InferenceFailed(format!(
                "model returned shape {:?} for a batch of {batch}",
                output.shape
            ))
            .into());
        }
        Ok(())
    }

    /// Run one item of a simple model (see [`Model::is_simple`]): its flattened input in, its
    /// flattened output back. Blocking, like [`Model::predict_batch`].
    pub fn predict(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        self.predict_batch(vec![input])
            .map(|mut batch| batch.pop().expect("batch of one yields one output"))
    }

    /// Run inference on a batch of flattened inputs.
    ///
    /// This is CPU/GPU-bound and blocking; call it from a blocking context
    /// (e.g. `tokio::task::spawn_blocking`) inside async code.
    pub fn predict_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        let batch_size = inputs.len();
        let flattened = match self.flatten_batch(inputs) {
            Ok(flattened) => flattened,
            Err(e) => {
                self.stats
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .record_error();
                return Err(e);
            }
        };

        let output_size: usize = self.info.output_spec.shape.iter().product();
        Ok(self
            .predict_flat(flattened, batch_size)?
            .chunks_exact(output_size)
            .map(<[f32]>::to_vec)
            .collect())
    }

    /// Run inference on `batch_size` items stored back to back in one row-major buffer.
    ///
    /// Returns the outputs the same way: `batch_size * output_size` values. This avoids
    /// per-item allocations and copies, so it is the fastest entry point for large inputs.
    /// Blocking, like [`Model::predict_batch`].
    pub fn predict_flat(&self, input: Vec<f32>, batch_size: usize) -> Result<Vec<f32>> {
        self.predict_many(vec![FlatInput {
            data: input,
            batch_size,
        }])
        .pop()
        .expect("one request yields one result")
    }

    /// Run several independent requests as one forward pass and split the outputs back.
    ///
    /// Each request is validated on its own, so an invalid one gets its own error and
    /// the rest still run. Results come back in request order, each holding that
    /// request's `batch_size * output_size` values. The requests together must not
    /// exceed [`MAX_BATCH_SIZE`] items. Statistics count every request, plus one batch
    /// per forward pass. Blocking, like [`Model::predict_batch`].
    pub fn predict_many(&self, requests: Vec<FlatInput>) -> Vec<Result<Vec<f32>>> {
        let batches: Vec<usize> = requests.iter().map(|r| r.batch_size).collect();
        let requests = requests.into_iter().map(|r| self.flat_request(r)).collect();
        self.run_requests(requests, Door::Flat)
            .into_iter()
            .zip(batches)
            .map(|(result, batch)| result.and_then(|outputs| self.flat_output(outputs, batch)))
            .collect()
    }

    /// A flat request as the one tensor of a simple model, without copying its buffer. The
    /// batch size and length are checked here, first, as the flat methods always have.
    fn flat_request(&self, FlatInput { data, batch_size }: FlatInput) -> Result<Vec<HostTensor>> {
        Self::check_batch_size(batch_size)?;
        let item = &self.info.input_spec.shape;
        if data.len() != batch_size * item.iter().product::<usize>() {
            let mut expected = vec![batch_size];
            expected.extend_from_slice(item);
            return Err(ModelError::InputValidation {
                expected,
                actual: vec![data.len()],
            }
            .into());
        }
        let mut shape = Vec::with_capacity(item.len() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(item);
        Ok(vec![HostTensor {
            shape,
            values: Values::F32(data),
        }])
    }

    /// The flat values of a simple model's one output
    fn flat_output(&self, outputs: Vec<HostTensor>, batch_size: usize) -> Result<Vec<f32>> {
        let output_size: usize = self.info.output_spec.shape.iter().product();
        match outputs.into_iter().next() {
            Some(HostTensor {
                values: Values::F32(values),
                ..
            }) if values.len() == batch_size * output_size => Ok(values),
            Some(output) => Err(ModelError::InferenceFailed(format!(
                "model returned shape {:?}, expected [{batch_size}, ..{:?}]",
                output.shape, self.info.output_spec.shape
            ))
            .into()),
            None => Err(ModelError::InferenceFailed("model returned no output".to_string()).into()),
        }
    }

    fn check_batch_size(batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            return Err(ModelError::InputValidation {
                expected: vec![1],
                actual: vec![0],
            }
            .into());
        }
        if batch_size > MAX_BATCH_SIZE {
            return Err(ModelError::InputValidation {
                expected: vec![MAX_BATCH_SIZE],
                actual: vec![batch_size],
            }
            .into());
        }
        Ok(())
    }

    fn flatten_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        Self::check_batch_size(inputs.len())?;
        let input_size: usize = self.info.input_spec.shape.iter().product();
        let mut flattened = Vec::with_capacity(inputs.len() * input_size);
        for input in inputs {
            self.validate_input_shape(&input)?;
            flattened.extend_from_slice(&input);
        }
        Ok(flattened)
    }

    /// Continue `previous`'s statistics, so a reloaded model's counters do not restart at zero.
    /// The two share them from now on: requests still running on `previous` count as well.
    pub fn adopt_stats(&mut self, previous: &Model) {
        self.stats = previous.stats.clone();
    }

    /// Run one all-zero input through the model without touching the statistics, so the first
    /// real request does not pay for lazy initialization (weight transforms, buffer pools,
    /// page faults on the memory-mapped weights). Returns how long it took.
    ///
    /// A model with an input whose shape is only fixed per request cannot be warmed up without
    /// guessing it, and is skipped (`Ok(None)`).
    pub fn warm_up(&self) -> std::result::Result<Option<std::time::Duration>, String> {
        let start = Instant::now();
        // A graph with a fixed batch size runs only that many items at once
        let batch = self.inner.fixed_batch().unwrap_or(1);
        let Some(inputs) = self
            .inputs
            .iter()
            .map(|spec| {
                let mut shape = vec![batch];
                shape.extend(spec.static_shape()?);
                let len = shape.iter().product();
                let values = if spec.dtype.is_int() {
                    Values::I64(vec![0; len])
                } else {
                    Values::F32(vec![0.0; len])
                };
                Some(HostTensor { shape, values })
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        self.inner.run(inputs).map_err(|e| e.to_string())?;
        Ok(Some(start.elapsed()))
    }

    /// Name, shapes, backend and the rest of what `GET /model/info` reports
    pub fn get_info(&self) -> &ModelInfo {
        &self.info
    }

    /// A snapshot of the request counters
    pub fn get_stats(&self) -> ModelStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Whether `input` has one item's worth of values for a simple model's input
    pub fn validate_input_shape(&self, input: &[f32]) -> Result<()> {
        let expected_size: usize = self.info.input_spec.shape.iter().product();
        if input.len() != expected_size {
            return Err(ModelError::InputValidation {
                expected: self.info.input_spec.shape.clone(),
                actual: vec![input.len()],
            }
            .into());
        }
        Ok(())
    }

    /// The backend's name, as in [`ModelInfo::backend`]
    pub fn get_backend_info(&self) -> String {
        self.info.backend.clone()
    }

    /// How the model was defined, as in [`ModelInfo::model_type`]
    pub fn model_type(&self) -> &'static str {
        self.model_type
    }
}

/// Requests' tensors joined along the batch dimension, input by input
fn join_requests(requests: Vec<Vec<HostTensor>>) -> Result<Vec<HostTensor>> {
    let count = requests.first().map_or(0, Vec::len);
    let mut columns: Vec<Vec<HostTensor>> = (0..count).map(|_| Vec::new()).collect();
    for inputs in requests {
        for (column, tensor) in columns.iter_mut().zip(inputs) {
            column.push(tensor);
        }
    }
    columns.into_iter().map(HostTensor::concat_batch).collect()
}

/// The error of a forward pass, for each request that shared it. `FurnaceError` is not `Clone`
/// (`LoadFailed` boxes its source), so the variant is rebuilt with the source kept as text.
fn duplicate(error: &FurnaceError) -> FurnaceError {
    let FurnaceError::Model(error) = error else {
        return ModelError::InferenceFailed(error.to_string()).into();
    };
    match error {
        ModelError::LoadFailed { path, source } => ModelError::LoadFailed {
            path: path.clone(),
            source: source.to_string().into(),
        },
        ModelError::InvalidFormat(m) => ModelError::InvalidFormat(m.clone()),
        ModelError::FileNotFound(path) => ModelError::FileNotFound(path.clone()),
        ModelError::InferenceFailed(m) => ModelError::InferenceFailed(m.clone()),
        ModelError::InputValidation { expected, actual } => ModelError::InputValidation {
            expected: expected.clone(),
            actual: actual.clone(),
        },
        ModelError::InvalidDataType(m) => ModelError::InvalidDataType(m.clone()),
        ModelError::InvalidInput(m) => ModelError::InvalidInput(m.clone()),
        ModelError::InvalidArgument { arg, value, reason } => ModelError::InvalidArgument {
            arg: arg.clone(),
            value: value.clone(),
            reason: reason.clone(),
        },
    }
    .into()
}

/// Load a file-based MLP model from `<path>.bpk` + `<path>.json` on the CPU backend
pub fn load_model(path: &Path) -> Result<Model> {
    load_model_with_backend(path, BackendKind::Cpu)
}

/// Load a file-based MLP model from `<path>.bpk` + `<path>.json`
pub fn load_model_with_backend(path: &Path, backend: BackendKind) -> Result<Model> {
    info!("Loading model from: {:?}", path);

    // Accept `model`, `model.bpk` or `model.json`
    let base: PathBuf = match path.extension().and_then(|e| e.to_str()) {
        Some("bpk") | Some("json") => path.with_extension(""),
        _ => path.to_path_buf(),
    };
    let weights_path = base.with_extension("bpk");

    // A `sample` path without files on disk yields a freshly initialized sample model
    if !weights_path.exists() {
        let is_sample = base
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.contains("sample"));
        if is_sample {
            info!("Creating sample MLP model");
            let container = create_sample_model()?;
            return Ok(Model::new(Box::new(MlpModel::new(container)), "mlp", 0));
        }
        return Err(ModelError::FileNotFound(weights_path).into());
    }

    let model_size_bytes = std::fs::metadata(&weights_path)
        .map_err(|e| ModelError::LoadFailed {
            path: weights_path.clone(),
            source: Box::new(e),
        })?
        .len();

    let container = BurnModelContainer::load(&base, backend)?;
    Ok(Model::new(
        Box::new(MlpModel::new(container)),
        "mlp",
        model_size_bytes,
    ))
}

/// Load a built-in (compiled-in) model by name on the CPU backend, with default weight lookup
pub fn load_built_in_model(model_name: &str) -> Result<Model> {
    load_built_in_model_with_backend(model_name, BackendKind::Cpu, None)
}

/// Load a built-in model by name on the given backend.
///
/// `weights` overrides where the `.bpk` weight file is read from; see
/// [`crate::models::resolve_weights`] for the default lookup.
pub fn load_built_in_model_with_backend(
    model_name: &str,
    backend: BackendKind,
    weights: Option<&Path>,
) -> Result<Model> {
    let built_in = BuiltInModel::from_name(model_name)?;
    let weights = resolve_weights(built_in.name(), weights)?;
    match &weights {
        Some(path) => info!(
            "Loading built-in model {} ({} backend) with weights from {:?}",
            model_name, backend, path
        ),
        None => info!(
            "Loading built-in model {} ({} backend) with embedded weights",
            model_name, backend
        ),
    }
    let model_size_bytes = weights
        .as_deref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(0, |m| m.len());
    let inner = built_in.create_model(backend, weights.as_deref())?;
    Ok(Model::new(inner, "onnx", model_size_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_model() -> Model {
        let dummy = DummyModel::new("test".to_string(), vec![784], vec![10]);
        Model::new(Box::new(dummy), "dummy", 0)
    }

    fn tensors(ids: Vec<i64>, scale: Vec<f32>) -> Vec<HostTensor> {
        vec![
            HostTensor {
                shape: vec![scale.len(), 4],
                values: Values::I64(ids),
            },
            HostTensor {
                shape: vec![scale.len()],
                values: Values::F32(scale),
            },
        ]
    }

    #[test]
    fn named_tensors_in_and_out() {
        let model = Model::new(Box::new(DummyTensorsModel::new(None)), "dummy", 0);
        assert!(!model.is_simple());
        let info = model.get_info();
        assert_eq!(info.inputs[0].datatype, "INT64");
        assert_eq!(info.inputs[0].shape, [-1, 4]);
        assert_eq!(info.inputs[1].shape, [-1]);
        assert_eq!(info.input_spec.dtype, "int64");

        let outputs = model
            .infer(tensors(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![0.5, 2.0]))
            .unwrap();
        assert_eq!(outputs[0].shape, [2, 4]);
        assert_eq!(
            outputs[0].values,
            Values::F32(vec![0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0])
        );
        assert_eq!(outputs[1].values, Values::I64(vec![10, 26]));
        assert_eq!(model.get_stats().success_count, 1);
        // A model whose inputs are all fixed per item is warmed up
        assert!(model.warm_up().unwrap().is_some());
    }

    #[test]
    fn named_tensors_are_checked() {
        let model = Model::new(Box::new(DummyTensorsModel::new(None)), "dummy", 0);
        let check = |inputs: Vec<HostTensor>, expected: &str| {
            let err = model.infer(inputs).unwrap_err().to_string();
            assert!(err.contains(expected), "{expected:?} not in {err:?}");
        };
        let mut inputs = tensors(vec![1, 2, 3, 4], vec![1.0]);
        inputs.pop();
        check(inputs, "expected 2 inputs (ids, scale)");
        let mut swapped = tensors(vec![1, 2, 3, 4], vec![1.0]);
        swapped.swap(0, 1);
        check(swapped, "'ids' must be INT64");
        let mut short = tensors(vec![1, 2, 3], vec![1.0]);
        short[0].shape = vec![1, 3];
        check(short, "'ids' has shape [1, 3]");
        check(
            tensors(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1.0]),
            "'ids' has shape [1, 4] (4 values) but 8 values",
        );
        let mut disagree = tensors(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1.0, 2.0]);
        disagree[1] = HostTensor {
            shape: vec![1],
            values: Values::F32(vec![1.0]),
        };
        check(disagree, "disagree on the batch size");
        check(
            tensors(vec![1, 2, 3, 4], vec![f32::NAN]),
            "'scale' has a non-finite value",
        );
        assert_eq!(model.get_stats().error_count, 6);
    }

    #[test]
    fn requests_of_named_tensors_share_a_forward_pass() {
        let model = Model::new(Box::new(DummyTensorsModel::new(None)), "dummy", 0);
        assert!(model.batches_named());
        let mut bad_shape = tensors(vec![1, 2, 3, 4], vec![1.0]);
        bad_shape[0].shape = vec![2, 2];
        let results = model.infer_many(vec![
            tensors(vec![1, 2, 3, 4], vec![1.0]),
            bad_shape,
            tensors(vec![1, 1, 1, 1, 2, 2, 2, 2], vec![2.0, 0.5]),
            tensors(vec![i64::from(i32::MAX) + 1, 0, 0, 0], vec![1.0]),
        ]);
        // The two valid requests ran as one batch of three items and got their own rows back
        let first = results[0].as_ref().unwrap();
        assert_eq!(first[0].shape, [1, 4]);
        assert_eq!(first[1].values, Values::I64(vec![10]));
        let third = results[2].as_ref().unwrap();
        assert_eq!(third[0].shape, [2, 4]);
        assert_eq!(
            third[0].values,
            Values::F32(vec![2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        );
        assert_eq!(third[1].values, Values::I64(vec![4, 8]));
        // The invalid ones got their own errors, before reaching the model
        assert!(
            results[1]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("'ids'")
        );
        let range = results[3].as_ref().unwrap_err().to_string();
        assert!(range.contains("outside"), "{range}");
        let stats = model.get_stats();
        assert_eq!((stats.batch_count, stats.batched_items), (1, 3));
        assert_eq!((stats.success_count, stats.error_count), (2, 2));
    }

    #[test]
    fn fixed_batch_graphs_run_in_chunks() {
        let model = Model::new(Box::new(DummyTensorsModel::new(Some(2))), "dummy", 0);
        let outputs = model
            .infer(tensors((1..=24).collect(), vec![1.0; 6]))
            .unwrap();
        assert_eq!(outputs[0].shape, [6, 4]);
        assert_eq!(outputs[1].values, Values::I64(vec![10, 26, 42, 58, 74, 90]));
        let err = model
            .infer(tensors((1..=12).collect(), vec![1.0; 3]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("runs batches of 2"), "{err}");
        // Warming up runs the batch the graph was built for, not one item
        assert!(model.warm_up().unwrap().is_some());
    }

    /// A simple model exported for batches of two only
    #[derive(Debug)]
    struct PairsOnly(DummyModel);

    impl BurnModel for PairsOnly {
        fn name(&self) -> &str {
            self.0.name()
        }
        fn backend(&self) -> BackendKind {
            self.0.backend()
        }
        fn inputs(&self) -> Vec<IoSpec> {
            self.0.inputs()
        }
        fn outputs(&self) -> Vec<IoSpec> {
            self.0.outputs()
        }
        fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
            self.0.run(inputs)
        }
        fn fixed_batch(&self) -> Option<usize> {
            Some(2)
        }
    }

    #[test]
    fn a_batch_the_graph_cannot_run_is_the_clients_error() {
        let dummy = DummyModel::new("pairs".to_string(), vec![3], vec![2]);
        let model = Model::new(Box::new(PairsOnly(dummy)), "dummy", 0);
        let err = model
            .predict_many(vec![FlatInput {
                data: vec![1.0; 9],
                batch_size: 3,
            }])
            .pop()
            .unwrap()
            .unwrap_err();
        assert!(err.is_client_error(), "{err:?}");
        assert!(!err.to_string().contains("Inference failed"), "{err}");
        assert!(err.to_string().contains("runs batches of 2"), "{err}");
        let pairs = model.predict_many(vec![FlatInput {
            data: vec![1.0; 12],
            batch_size: 4,
        }]);
        assert_eq!(pairs[0].as_ref().unwrap().len(), 8);
        assert!(model.warm_up().unwrap().is_some());
    }

    #[test]
    fn a_shared_failure_keeps_its_error_for_every_request() {
        // Three items for a graph of pairs: the joined pass fails, and each request gets the
        // client error itself rather than its text as an inference failure
        let model = Model::new(Box::new(DummyTensorsModel::new(Some(2))), "dummy", 0);
        let results = model.infer_many(vec![
            tensors(vec![1, 2, 3, 4], vec![1.0]),
            tensors(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1.0, 2.0]),
        ]);
        for result in results {
            let err = result.unwrap_err();
            assert!(
                matches!(
                    err,
                    crate::error::FurnaceError::Model(ModelError::InvalidInput(_))
                ),
                "{err:?}"
            );
            assert!(err.to_string().contains("runs batches of 2"), "{err}");
        }
        assert_eq!(model.get_stats().error_count, 2);
    }

    #[test]
    fn an_input_without_a_batch_dimension_is_rejected() {
        let model = Model::new(Box::new(DummyTensorsModel::new(None)), "dummy", 0);
        let mut inputs = tensors(vec![1, 2, 3, 4], vec![1.0]);
        inputs[1].shape = vec![];
        assert_eq!(inputs[1].batch(), 0);
        let err = model.infer(inputs).unwrap_err().to_string();
        assert!(err.contains("'scale' has no batch dimension"), "{err}");
    }

    #[test]
    fn test_model_info() {
        let model = get_test_model();
        let info = model.get_info();
        assert_eq!(info.model_type, "dummy");
        assert_eq!(info.backend, "cpu");
        assert_eq!(info.input_spec.shape, vec![784]);
        assert_eq!(info.output_spec.shape, vec![10]);
        assert_eq!(info.input_spec.dtype, "float32");
    }

    #[test]
    fn test_model_loading_invalid_path() {
        let result = load_model(Path::new("nonexistent_model"));
        assert!(matches!(
            result,
            Err(crate::error::FurnaceError::Model(ModelError::FileNotFound(
                _
            )))
        ));
    }

    #[test]
    fn test_inference_with_valid_input() {
        let model = get_test_model();
        let output = model.predict(vec![0.5; 784]).unwrap();
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_batch_inference_splits_outputs() {
        let model = get_test_model();
        let outputs = model.predict_batch(vec![vec![0.5; 784]; 3]).unwrap();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|o| o.len() == 10));
    }

    #[test]
    fn test_predict_flat_matches_batch() {
        let model = get_test_model();
        let flat = model.predict_flat(vec![0.5; 784 * 3], 3).unwrap();
        let batch = model.predict_batch(vec![vec![0.5; 784]; 3]).unwrap();
        assert_eq!(flat, batch.concat());
        assert_eq!(model.get_stats().success_count, 2);
    }

    #[test]
    fn test_predict_flat_rejects_bad_input() {
        let model = get_test_model();
        // Length not a multiple of the batch
        assert!(matches!(
            model.predict_flat(vec![0.5; 784 * 2 + 1], 2),
            Err(crate::error::FurnaceError::Model(
                ModelError::InputValidation { .. }
            ))
        ));
        assert!(model.predict_flat(vec![], 0).is_err());
        assert!(
            model
                .predict_flat(vec![0.5; 784], MAX_BATCH_SIZE + 1)
                .is_err()
        );

        // Non-finite values report the batch item and index
        let mut input = vec![0.5; 784 * 2];
        input[784 + 5] = f32::NAN;
        let err = model.predict_flat(input, 2).unwrap_err().to_string();
        assert!(err.contains("batch[1][5]"), "{err}");
        assert_eq!(model.get_stats().error_count, 4);
    }

    #[test]
    fn test_inference_with_invalid_shape() {
        let model = get_test_model();
        let result = model.predict(vec![0.5; 100]);
        assert!(matches!(
            result,
            Err(crate::error::FurnaceError::Model(
                ModelError::InputValidation { .. }
            ))
        ));
    }

    #[test]
    fn test_inference_rejects_non_finite() {
        let model = get_test_model();
        let mut input = vec![0.5; 784];
        input[42] = f32::NAN;
        assert!(matches!(
            model.predict(input),
            Err(crate::error::FurnaceError::Model(
                ModelError::InvalidDataType(_)
            ))
        ));
    }

    #[test]
    fn a_reloaded_model_shares_its_predecessors_stats() {
        let old = get_test_model();
        old.predict(vec![0.5; 784]).unwrap();
        let mut new = get_test_model();
        new.adopt_stats(&old);
        // A request that was still running on the old model when it was replaced
        old.predict(vec![0.5; 784]).unwrap();
        new.predict(vec![0.5; 784]).unwrap();
        assert_eq!(new.get_stats().success_count, 3);
        assert_eq!(old.get_stats().success_count, 3);
    }

    #[test]
    fn test_model_stats() {
        let model = get_test_model();
        let stats = model.get_stats();
        assert_eq!(stats.inference_count, 0);
        assert!(stats.last_inference_time.is_none());

        model.predict(vec![0.5; 784]).unwrap();
        let _ = model.predict(vec![0.5; 1]);

        let stats = model.get_stats();
        assert_eq!(stats.inference_count, 2);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.error_count, 1);
        assert!(stats.last_inference_time.is_some());
        assert!(stats.min_inference_time_ms.is_some());
    }

    #[test]
    fn test_real_mlp_inference_is_not_constant() {
        let model = load_model(Path::new("sample_model")).unwrap();
        let out = model.predict(vec![0.3; 784]).unwrap();
        assert_eq!(out.len(), 10);
        assert!(out.iter().any(|&v| v != out[0]));
    }

    #[cfg(model_resnet18)]
    #[test]
    fn test_built_in_model_missing_weights_file() {
        let result = load_built_in_model_with_backend(
            "resnet18",
            BackendKind::Cpu,
            Some(Path::new("does/not/exist.bpk")),
        );
        assert!(matches!(
            result,
            Err(crate::error::FurnaceError::Model(ModelError::FileNotFound(
                _
            )))
        ));
    }

    #[cfg(model_resnet18)]
    #[test]
    fn test_resnet18_weights_from_explicit_path() {
        // Copy the generated weights elsewhere to prove the path override is honored
        let dir = tempfile::tempdir().unwrap();
        let src = Path::new(env!("OUT_DIR")).join("models/resnet18.bpk");
        if !src.exists() {
            return; // built with embed-weights: no separate file
        }
        let dst = dir.path().join("custom.bpk");
        std::fs::copy(&src, &dst).unwrap();

        let model =
            load_built_in_model_with_backend("resnet18", BackendKind::Cpu, Some(&dst)).unwrap();
        assert_eq!(
            model.get_info().model_size_bytes,
            std::fs::metadata(&dst).unwrap().len()
        );
        assert_eq!(model.predict(vec![0.5; 3 * 224 * 224]).unwrap().len(), 1000);
    }

    #[cfg(model_resnet18)]
    #[test]
    fn test_resnet18_inference() {
        let model = load_built_in_model("resnet18").unwrap();
        assert_eq!(model.get_info().input_spec.shape, vec![3, 224, 224]);
        let out = model.predict(vec![0.5; 3 * 224 * 224]).unwrap();
        assert_eq!(out.len(), 1000);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[cfg(model_mnist)]
    #[test]
    fn test_mnist_inference() {
        let model = load_built_in_model("mnist").unwrap();
        assert_eq!(model.get_info().input_spec.shape, vec![1, 28, 28]);
        assert_eq!(model.get_info().output_spec.shape, vec![10]);
        let outputs = model
            .predict_batch(vec![vec![0.0; 28 * 28], vec![255.0; 28 * 28]])
            .unwrap();
        assert!(outputs.iter().all(|o| o.len() == 10));
        assert_ne!(outputs[0], outputs[1]);
    }
}
