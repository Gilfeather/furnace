//! The Open Inference Protocol over gRPC (feature `grpc`, `--grpc-port`): the same service as
//! KServe's and Triton's `inference.GRPCInferenceService`, so their gRPC clients work unchanged.
//!
//! Inference takes one `FP32` input named `input`, as `raw_input_contents` (what
//! `tritonclient.grpc` sends) or `contents.fp32_contents`, and answers with
//! `raw_output_contents`. It goes through the same batching, concurrency limit, timeout and
//! metrics as the HTTP API.

use std::net::SocketAddr;

use tonic::{Request as GrpcRequest, Response as GrpcResponse, Status};

use super::v2::{INPUT_NAME, MODEL_VERSION, OUTPUT_NAME, check_input, resolve_model};
use super::*;

/// Types and service generated from proto/open_inference_grpc.proto
pub mod pb {
    tonic::include_proto!("inference");
}

use pb::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use pb::model_infer_response::InferOutputTensor;
use pb::model_metadata_response::TensorMetadata;

/// The gRPC service, sharing the HTTP server's state
pub struct Service {
    state: AppState,
}

/// The gRPC server for `state`, accepting messages up to `max_message_bytes`
pub(super) fn server(
    state: AppState,
    max_message_bytes: usize,
) -> GrpcInferenceServiceServer<Service> {
    GrpcInferenceServiceServer::new(Service { state })
        .max_decoding_message_size(max_message_bytes)
        .max_encoding_message_size(max_message_bytes)
}

/// Serve gRPC on `addr` until `shutdown` resolves
pub(super) async fn serve(
    state: AppState,
    addr: SocketAddr,
    max_message_bytes: usize,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::result::Result<(), String> {
    info!("✅ gRPC (Open Inference Protocol) on {addr}");
    tonic::transport::Server::builder()
        .add_service(server(state, max_message_bytes))
        .serve_with_shutdown(addr, shutdown)
        .await
        .map_err(|e| format!("gRPC server error: {e}"))
}

#[tonic::async_trait]
impl GrpcInferenceService for Service {
    async fn server_live(
        &self,
        _: GrpcRequest<pb::ServerLiveRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ServerLiveResponse>, Status> {
        Ok(GrpcResponse::new(pb::ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _: GrpcRequest<pb::ServerReadyRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ServerReadyResponse>, Status> {
        Ok(GrpcResponse::new(pb::ServerReadyResponse {
            ready: self.state.ready(),
        }))
    }

    async fn model_ready(
        &self,
        request: GrpcRequest<pb::ModelReadyRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ModelReadyResponse>, Status> {
        let request = request.into_inner();
        let slot = resolve_model(&self.state, &request.name, &request.version)
            .map_err(Status::not_found)?;
        Ok(GrpcResponse::new(pb::ModelReadyResponse {
            ready: slot.get().ready(),
        }))
    }

    async fn server_metadata(
        &self,
        _: GrpcRequest<pb::ServerMetadataRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ServerMetadataResponse>, Status> {
        Ok(GrpcResponse::new(pb::ServerMetadataResponse {
            name: "furnace".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: Vec::new(),
        }))
    }

    async fn model_metadata(
        &self,
        request: GrpcRequest<pb::ModelMetadataRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ModelMetadataResponse>, Status> {
        let request = request.into_inner();
        let slot = resolve_model(&self.state, &request.name, &request.version)
            .map_err(Status::not_found)?;
        let loaded = slot.get();
        let info = loaded.model.get_info();
        let with_batch = |shape: &[usize]| {
            std::iter::once(-1)
                .chain(shape.iter().map(|&d| d as i64))
                .collect()
        };
        Ok(GrpcResponse::new(pb::ModelMetadataResponse {
            name: info.name.clone(),
            versions: vec![MODEL_VERSION.to_string()],
            platform: format!("furnace_{}", info.model_type),
            inputs: vec![TensorMetadata {
                name: INPUT_NAME.to_string(),
                datatype: "FP32".to_string(),
                shape: with_batch(&info.input_spec.shape),
            }],
            outputs: vec![TensorMetadata {
                name: OUTPUT_NAME.to_string(),
                datatype: "FP32".to_string(),
                shape: with_batch(&info.output_spec.shape),
            }],
            properties: Default::default(),
        }))
    }

    async fn model_infer(
        &self,
        request: GrpcRequest<pb::ModelInferRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ModelInferResponse>, Status> {
        let state = &self.state;
        let start = Instant::now();
        state.total_requests.fetch_add(1, Ordering::Relaxed);
        let result = infer(state, request.into_inner()).await;
        state.request_latency.observe(start.elapsed());
        if let Err(status) = &result {
            state.total_errors.fetch_add(1, Ordering::Relaxed);
            if status.code() == tonic::Code::ResourceExhausted {
                state.rejected_busy.fetch_add(1, Ordering::Relaxed);
            }
        }
        result.map(GrpcResponse::new)
    }
}

async fn infer(
    state: &ServerState,
    mut request: pb::ModelInferRequest,
) -> std::result::Result<pb::ModelInferResponse, Status> {
    let slot = resolve_model(state, &request.model_name, &request.model_version)
        .map_err(Status::not_found)?;
    let loaded = slot.get();
    let arrival = loaded.batcher.as_ref().map(Batcher::arrival);
    let [input] = request.inputs.as_mut_slice() else {
        return Err(Status::invalid_argument(format!(
            "Expected exactly one input ('{INPUT_NAME}'), got {}",
            request.inputs.len()
        )));
    };
    let batch_size = check_input(&loaded.model, &input.name, &input.datatype, &input.shape)
        .map_err(Status::invalid_argument)?;
    if let Some(unknown) = request.outputs.iter().find(|o| o.name != OUTPUT_NAME) {
        return Err(Status::invalid_argument(format!(
            "Unknown output '{}'; the model's output is '{OUTPUT_NAME}'",
            unknown.name
        )));
    }
    let item_size: usize = loaded.model.get_info().input_spec.shape.iter().product();
    let elements = batch_size * item_size;
    let data = match request.raw_input_contents.first() {
        Some(raw) => {
            if raw.len() != elements * size_of::<f32>() {
                return Err(Status::invalid_argument(format!(
                    "raw_input_contents has {} bytes; shape {:?} needs {}",
                    raw.len(),
                    input.shape,
                    elements * size_of::<f32>()
                )));
            }
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
        None => {
            let values = input
                .contents
                .take()
                .map(|contents| contents.fp32_contents)
                .unwrap_or_default();
            if values.len() != elements {
                return Err(Status::invalid_argument(format!(
                    "Input shape {:?} has {elements} elements, fp32_contents has {}",
                    input.shape,
                    values.len()
                )));
            }
            values
        }
    };

    let start = Instant::now();
    let permit = state
        .inference_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| Status::resource_exhausted("Too many concurrent requests, retry later"))?;
    let output = match run_inference(state, &loaded, arrival, data, batch_size, permit).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return Err(match e {
                FurnaceError::Model(
                    ModelError::InputValidation { .. } | ModelError::InvalidDataType(_),
                ) => Status::invalid_argument(e.to_string()),
                _ => Status::internal(e.to_string()),
            });
        }
        Err(response) if response.status() == StatusCode::GATEWAY_TIMEOUT => {
            return Err(Status::deadline_exceeded("Inference exceeded the timeout"));
        }
        Err(_) => return Err(Status::internal("Inference failed")),
    };
    state.observe_inference(&slot, start.elapsed());

    let info = loaded.model.get_info();
    let shape = std::iter::once(batch_size as i64)
        .chain(info.output_spec.shape.iter().map(|&d| d as i64))
        .collect();
    let mut raw = Vec::with_capacity(output.len() * size_of::<f32>());
    for value in &output {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    Ok(pb::ModelInferResponse {
        model_name: info.name.clone(),
        model_version: MODEL_VERSION.to_string(),
        id: request.id,
        parameters: Default::default(),
        outputs: vec![InferOutputTensor {
            name: OUTPUT_NAME.to_string(),
            datatype: "FP32".to_string(),
            shape,
            parameters: Default::default(),
            contents: None,
        }],
        raw_output_contents: vec![raw],
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::test_app_state;
    use super::pb::grpc_inference_service_client::GrpcInferenceServiceClient;
    use super::pb::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
    use super::pb::{InferTensorContents, ModelInferRequest};
    use super::*;

    /// A gRPC server for the test model on a free port, and a client for it
    async fn client() -> GrpcInferenceServiceClient<tonic::transport::Channel> {
        let state = test_app_state();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(server(state, 64 << 20))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        GrpcInferenceServiceClient::connect(format!("http://{addr}"))
            .await
            .unwrap()
    }

    fn request(
        shape: Vec<i64>,
        raw: Option<Vec<f32>>,
        fp32: Option<Vec<f32>>,
    ) -> ModelInferRequest {
        ModelInferRequest {
            model_name: "test_model".to_string(),
            model_version: String::new(),
            id: "req-7".to_string(),
            parameters: Default::default(),
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: "FP32".to_string(),
                shape,
                parameters: Default::default(),
                contents: fp32.map(|v| InferTensorContents {
                    fp32_contents: v,
                    ..Default::default()
                }),
            }],
            outputs: vec![InferRequestedOutputTensor {
                name: "output".to_string(),
                parameters: Default::default(),
            }],
            raw_input_contents: raw
                .map(|v| vec![v.iter().flat_map(|x| x.to_le_bytes()).collect()])
                .unwrap_or_default(),
        }
    }

    fn floats(raw: &[u8]) -> Vec<f32> {
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    #[tokio::test]
    async fn health_metadata_and_inference() {
        let mut client = client().await;
        assert!(
            client
                .server_live(pb::ServerLiveRequest {})
                .await
                .unwrap()
                .into_inner()
                .live
        );
        let model = client
            .model_metadata(pb::ModelMetadataRequest {
                name: "test_model".to_string(),
                version: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(model.inputs[0].shape, vec![-1, 784]);
        assert_eq!(model.outputs[0].shape, vec![-1, 10]);

        let x: Vec<f32> = (0..2 * 784).map(|i| (i as f32 * 0.01).sin()).collect();
        let raw = client
            .model_infer(request(vec![2, 784], Some(x.clone()), None))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(raw.id, "req-7");
        assert_eq!(raw.outputs[0].shape, vec![2, 10]);
        let from_raw = floats(&raw.raw_output_contents[0]);
        assert_eq!(from_raw.len(), 20);

        // fp32_contents gives the same answer
        let typed = client
            .model_infer(request(vec![2, 784], None, Some(x)))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(floats(&typed.raw_output_contents[0]), from_raw);
    }

    #[tokio::test]
    async fn errors_map_to_grpc_status_codes() {
        let mut client = client().await;
        let unknown = client
            .model_metadata(pb::ModelMetadataRequest {
                name: "other".to_string(),
                version: String::new(),
            })
            .await
            .unwrap_err();
        assert_eq!(unknown.code(), tonic::Code::NotFound);

        let short = client
            .model_infer(request(vec![1, 784], Some(vec![0.5; 100]), None))
            .await
            .unwrap_err();
        assert_eq!(short.code(), tonic::Code::InvalidArgument);
        let shape = client
            .model_infer(request(vec![1, 783], Some(vec![0.5; 783]), None))
            .await
            .unwrap_err();
        assert_eq!(shape.code(), tonic::Code::InvalidArgument);
    }
}
