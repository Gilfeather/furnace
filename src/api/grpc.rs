//! The Open Inference Protocol over gRPC (feature `grpc`, `--grpc-port`): the same service as
//! KServe's and Triton's `inference.GRPCInferenceService`, so their gRPC clients work unchanged.
//!
//! Inference takes one `FP32` input named `input`, as `raw_input_contents` (what
//! `tritonclient.grpc` sends) or `contents.fp32_contents`, and answers with
//! `raw_output_contents`. It goes through the same batching, concurrency limit, timeout and
//! metrics as the HTTP API.

use std::net::SocketAddr;

use tonic::{Request as GrpcRequest, Response as GrpcResponse, Status};

use super::infer::{InferError, encode_raw, grpc_status, values_datatype};
use super::v2::{
    INPUT_NAME, MODEL_VERSION, OUTPUT_NAME, RequestTensor, Source, assemble_inputs, check_input,
    metadata_of, resolve_model, select_outputs,
};
use super::*;
use crate::tensor::{DataType, Values};

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

/// TLS configurations for gRPC, when the `tls` feature is on
#[cfg(feature = "tls")]
pub(super) type GrpcTls = Option<Arc<crate::tls::TlsConfigs>>;
#[cfg(not(feature = "tls"))]
pub(super) type GrpcTls = Option<std::convert::Infallible>;

/// Serve gRPC on `addr`, over TLS when `tls` is given, until `shutdown` resolves
pub(super) async fn serve(
    state: AppState,
    addr: SocketAddr,
    max_message_bytes: usize,
    tls: GrpcTls,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::result::Result<(), String> {
    let service = server(state, max_message_bytes);
    match tls {
        #[cfg(feature = "tls")]
        Some(configs) => {
            use tokio_stream::StreamExt as _;
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| format!("Failed to bind gRPC to {addr}: {e}"))?;
            info!("✅ gRPC (Open Inference Protocol) over TLS on {addr}");
            let connections = tokio_stream::wrappers::ReceiverStream::new(crate::tls::accept(
                listener,
                configs,
                crate::tls::Protocols::Grpc,
            ))
            .map(Ok::<_, std::io::Error>);
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(connections, shutdown)
                .await
        }
        #[cfg(not(feature = "tls"))]
        Some(never) => match never {},
        None => {
            info!("✅ gRPC (Open Inference Protocol) on {addr}");
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_shutdown(addr, shutdown)
                .await
        }
    }
    .map_err(|e| format!("gRPC server error: {e}"))
}

impl Service {
    /// API keys and the rate limit, as for HTTP: `authorization: Bearer <key>` or
    /// `x-api-key: <key>` metadata
    fn admit<T>(&self, request: &GrpcRequest<T>) -> std::result::Result<(), Status> {
        let metadata = |name| request.metadata().get(name).and_then(|v| v.to_str().ok());
        let key = presented_key(metadata("authorization"), metadata(API_KEY_HEADER));
        let ip = request.remote_addr().map(|addr| addr.ip());
        self.state
            .access
            .admit(key, ip, true)
            .map_err(|rejection| match rejection {
                Rejection::Unauthorized => {
                    self.state
                        .rejected_unauthorized
                        .fetch_add(1, Ordering::Relaxed);
                    Status::unauthenticated("Missing or unknown API key")
                }
                Rejection::RateLimited(retry) => {
                    self.state
                        .rejected_rate_limited
                        .fetch_add(1, Ordering::Relaxed);
                    Status::resource_exhausted(format!(
                        "Rate limit exceeded; retry in {} ms",
                        retry.as_millis().max(1)
                    ))
                }
            })
    }

    /// The model a request names, or `NOT_FOUND`
    fn model(&self, name: &str, version: &str) -> std::result::Result<Arc<ModelSlot>, Status> {
        resolve_model(&self.state, name, version)
            .map_err(|message| grpc_status(StatusCode::NOT_FOUND, message))
    }
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
        self.admit(&request)?;
        let request = request.into_inner();
        let slot = self.model(&request.name, &request.version)?;
        Ok(GrpcResponse::new(pb::ModelReadyResponse {
            ready: slot.get().ready(),
        }))
    }

    async fn server_metadata(
        &self,
        request: GrpcRequest<pb::ServerMetadataRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ServerMetadataResponse>, Status> {
        self.admit(&request)?;
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
        self.admit(&request)?;
        let request = request.into_inner();
        let slot = self.model(&request.name, &request.version)?;
        let metadata = metadata_of(slot.get().model.get_info());
        let tensors = |tensors: Vec<super::v2::TensorMetadata>| {
            tensors
                .into_iter()
                .map(|t| TensorMetadata {
                    name: t.name,
                    datatype: t.datatype,
                    shape: t.shape,
                })
                .collect()
        };
        Ok(GrpcResponse::new(pb::ModelMetadataResponse {
            name: metadata.name,
            versions: metadata.versions,
            platform: metadata.platform,
            inputs: tensors(metadata.inputs),
            outputs: tensors(metadata.outputs),
            properties: Default::default(),
        }))
    }

    async fn model_infer(
        &self,
        request: GrpcRequest<pb::ModelInferRequest>,
    ) -> std::result::Result<GrpcResponse<pb::ModelInferResponse>, Status> {
        self.admit(&request)?;
        let state = &self.state;
        let start = Instant::now();
        let result = infer(state, request.into_inner()).await;
        state.count_request(
            start.elapsed(),
            result.is_err(),
            matches!(result, Err(Failed::Infer(InferError::Busy))),
        );
        result.map(GrpcResponse::new).map_err(Failed::into_status)
    }
}

/// Why an inference over gRPC got no answer: a request that does not fit, or the shared
/// inference path's refusal
#[derive(Debug)]
enum Failed {
    Request(Status),
    Infer(InferError),
}

impl Failed {
    fn into_status(self) -> Status {
        match self {
            Self::Request(status) => status,
            Self::Infer(e) => e.into_status(),
        }
    }
}

impl From<InferError> for Failed {
    fn from(e: InferError) -> Self {
        Self::Infer(e)
    }
}

/// A request that does not fit the model
fn invalid(message: String) -> Failed {
    Failed::Request(grpc_status(StatusCode::BAD_REQUEST, message))
}

async fn infer(
    state: &ServerState,
    mut request: pb::ModelInferRequest,
) -> std::result::Result<pb::ModelInferResponse, Failed> {
    let slot = resolve_model(state, &request.model_name, &request.model_version)
        .map_err(|message| Failed::Request(grpc_status(StatusCode::NOT_FOUND, message)))?;
    let loaded = slot.get();
    let announced = state.serving.announce(&slot.name);
    if !loaded.model.is_simple() {
        return infer_named(state, &slot, &loaded, announced, request).await;
    }
    let [input] = request.inputs.as_mut_slice() else {
        return Err(invalid(format!(
            "Expected exactly one input ('{INPUT_NAME}'), got {}",
            request.inputs.len()
        )));
    };
    let batch_size =
        check_input(&loaded.model, &input.name, &input.datatype, &input.shape).map_err(invalid)?;
    if let Some(unknown) = request.outputs.iter().find(|o| o.name != OUTPUT_NAME) {
        return Err(invalid(format!(
            "Unknown output '{}'; the model's output is '{OUTPUT_NAME}'",
            unknown.name
        )));
    }
    let item_size: usize = loaded.model.get_info().input_spec.shape.iter().product();
    let elements = batch_size * item_size;
    let data = match request.raw_input_contents.first() {
        Some(raw) => {
            if raw.len() != elements * size_of::<f32>() {
                return Err(invalid(format!(
                    "Input '{INPUT_NAME}' has {} bytes of raw data; shape {:?} of FP32 needs {}",
                    raw.len(),
                    input.shape,
                    elements * size_of::<f32>()
                )));
            }
            f32s_from_le(raw)
        }
        None => {
            let values = input
                .contents
                .take()
                .map(|contents| contents.fp32_contents)
                .unwrap_or_default();
            if values.len() != elements {
                return Err(invalid(format!(
                    "Input shape {:?} has {elements} elements, fp32_contents has {}",
                    input.shape,
                    values.len()
                )));
            }
            values
        }
    };

    let output = state
        .infer_flat(&slot, &loaded, announced, data, batch_size)
        .await?;
    let info = loaded.model.get_info();
    let shape = std::iter::once(batch_size as i64)
        .chain(info.output_spec.shape.iter().map(|&d| d as i64))
        .collect();
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
        raw_output_contents: vec![f32s_to_le(&output)],
    })
}

/// Inference for a model of named tensors: `raw_input_contents` (one per input, as
/// `tritonclient.grpc` sends them) or typed `contents`; outputs as `raw_output_contents`
async fn infer_named(
    state: &ServerState,
    slot: &ModelSlot,
    loaded: &Loaded,
    announced: Option<crate::nerve::Announced<'_>>,
    mut request: pb::ModelInferRequest,
) -> std::result::Result<pb::ModelInferResponse, Failed> {
    let model = &loaded.model;
    let raw_inputs = std::mem::take(&mut request.raw_input_contents);
    if !raw_inputs.is_empty() && raw_inputs.len() != request.inputs.len() {
        return Err(invalid(format!(
            "{} raw_input_contents for {} inputs",
            raw_inputs.len(),
            request.inputs.len()
        )));
    }
    let tensors = request.inputs.iter_mut().enumerate().map(|(i, input)| {
        let source = match raw_inputs.get(i) {
            Some(raw) => Source::Raw(raw),
            None => {
                let contents = input.contents.take().unwrap_or_default();
                // The typed field that matches the datatype; a mismatch is refused by name
                Source::Values(match DataType::from_v2_name(&input.datatype) {
                    Some(DataType::Int64) => Values::I64(contents.int64_contents),
                    Some(DataType::Int32) => {
                        Values::I64(contents.int_contents.into_iter().map(i64::from).collect())
                    }
                    _ => Values::F32(contents.fp32_contents),
                })
            }
        };
        RequestTensor {
            name: &input.name,
            datatype: &input.datatype,
            shape: &input.shape,
            source,
        }
    });
    let inputs = assemble_inputs(model, tensors).map_err(invalid)?;
    let selected = select_outputs(
        model,
        request.outputs.iter().map(|o| (o.name.as_str(), true)),
        true,
    )
    .map_err(invalid)?;

    let outputs = state.infer(slot, loaded, announced, inputs).await?;
    let mut tensors = Vec::with_capacity(selected.len());
    let mut raw = Vec::with_capacity(selected.len());
    for (index, _) in selected {
        let output = &outputs[index];
        tensors.push(InferOutputTensor {
            name: model.outputs()[index].name.clone(),
            datatype: values_datatype(&output.values).v2_name().to_string(),
            shape: output.shape.iter().map(|&d| d as i64).collect(),
            parameters: Default::default(),
            contents: None,
        });
        raw.push(encode_raw(&output.values));
    }
    Ok(pb::ModelInferResponse {
        model_name: model.get_info().name.clone(),
        model_version: MODEL_VERSION.to_string(),
        id: request.id,
        parameters: Default::default(),
        outputs: tensors,
        raw_output_contents: raw,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tests::test_app_state;
    use super::pb::grpc_inference_service_client::GrpcInferenceServiceClient;
    use super::pb::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
    use super::pb::{InferTensorContents, ModelInferRequest};
    use super::*;

    /// A gRPC server for the test model on a free port, and a client for it
    async fn client() -> GrpcInferenceServiceClient<tonic::transport::Channel> {
        client_for(test_app_state()).await
    }

    pub(crate) async fn client_for(
        state: AppState,
    ) -> GrpcInferenceServiceClient<tonic::transport::Channel> {
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

    #[tokio::test]
    async fn api_keys_and_rate_limit() {
        let config = ServerConfig {
            api_keys: vec!["grpc-key-0123456789".to_string()],
            rate_limit: Some(ClientRateLimit {
                per_second: 0.001,
                burst: 2,
            }),
            ..ServerConfig::default()
        };
        let state = server_state(
            vec![ServedModel::from(super::super::tests::dummy_model())].into(),
            &config,
        );
        let mut client = client_for(state).await;
        let with_key = |mut request: tonic::Request<ModelInferRequest>| {
            request.metadata_mut().insert(
                "authorization",
                "Bearer grpc-key-0123456789".parse().unwrap(),
            );
            request
        };
        let infer = || tonic::Request::new(request(vec![1, 784], Some(vec![0.5; 784]), None));

        // Health needs no key
        let live = client.server_live(pb::ServerLiveRequest {}).await.unwrap();
        assert!(live.into_inner().live);
        let status = client.model_infer(infer()).await.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        // Two at once, then the third is over the limit
        for _ in 0..2 {
            client.model_infer(with_key(infer())).await.unwrap();
        }
        let status = client.model_infer(with_key(infer())).await.unwrap_err();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
        assert!(
            status.message().contains("Rate limit"),
            "{}",
            status.message()
        );
    }

    #[tokio::test]
    async fn named_tensors() {
        let model = Model::new(
            Box::new(crate::model::DummyTensorsModel::new(None)),
            "dummy",
            0,
        );
        let mut client = client_for(server_state(
            vec![ServedModel::from(model)].into(),
            &ServerConfig::default(),
        ))
        .await;
        let metadata = client
            .model_metadata(pb::ModelMetadataRequest {
                name: "tensors".to_string(),
                version: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(metadata.inputs[0].name, "ids");
        assert_eq!(metadata.inputs[0].datatype, "INT64");
        assert_eq!(metadata.outputs[1].shape, [-1]);

        let tensor = |name: &str, datatype: &str, shape: Vec<i64>, contents| InferInputTensor {
            name: name.to_string(),
            datatype: datatype.to_string(),
            shape,
            parameters: Default::default(),
            contents,
        };
        // Raw contents, as tritonclient.grpc sends them
        let ids: Vec<u8> = [1i64, 2, 3, 4, 5, 6, 7, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let scale: Vec<u8> = [0.5f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let request = ModelInferRequest {
            model_name: "tensors".to_string(),
            model_version: String::new(),
            id: "g1".to_string(),
            parameters: Default::default(),
            inputs: vec![
                tensor("ids", "INT64", vec![2, 4], None),
                tensor("scale", "FP32", vec![2], None),
            ],
            outputs: Vec::new(),
            raw_input_contents: vec![ids, scale],
        };
        let response = client.model_infer(request).await.unwrap().into_inner();
        assert_eq!(response.outputs.len(), 2);
        assert_eq!(response.outputs[1].datatype, "INT64");
        assert_eq!(
            super::super::infer::decode_raw(DataType::Int64, &response.raw_output_contents[1]),
            Values::I64(vec![10, 26])
        );

        // Typed contents, with INT32 ids and only one output asked for
        let request = ModelInferRequest {
            model_name: "tensors".to_string(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            inputs: vec![
                tensor(
                    "ids",
                    "INT32",
                    vec![1, 4],
                    Some(InferTensorContents {
                        int_contents: vec![1, 1, 1, 1],
                        ..Default::default()
                    }),
                ),
                tensor(
                    "scale",
                    "FP32",
                    vec![1],
                    Some(InferTensorContents {
                        fp32_contents: vec![3.0],
                        ..Default::default()
                    }),
                ),
            ],
            outputs: vec![InferRequestedOutputTensor {
                name: "scores".to_string(),
                parameters: Default::default(),
            }],
            raw_input_contents: Vec::new(),
        };
        let response = client.model_infer(request).await.unwrap().into_inner();
        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].shape, [1, 4]);
        assert_eq!(
            super::super::infer::decode_raw(DataType::Fp32, &response.raw_output_contents[0]),
            Values::F32(vec![3.0; 4])
        );

        let missing = ModelInferRequest {
            model_name: "tensors".to_string(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            inputs: vec![tensor("ids", "INT64", vec![1, 4], None)],
            outputs: Vec::new(),
            raw_input_contents: vec![vec![0; 32]],
        };
        let status = client.model_infer(missing).await.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("Missing input 'scale'"));
    }

    /// The gRPC service over the pure-Rust TLS, with mutual TLS, reached by a tonic client
    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn over_tls() {
        use crate::tls::{Protocols, TlsConfigs, accept, tests::client_config, tests::settings};
        use tokio_stream::StreamExt as _;

        let configs = TlsConfigs::load(settings("server-p256", true)).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connections =
            tokio_stream::wrappers::ReceiverStream::new(accept(listener, configs, Protocols::Grpc))
                .map(Ok::<_, std::io::Error>);
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(server(test_app_state(), 64 << 20))
                .serve_with_incoming(connections),
        );

        let connect = |certificate: Option<&'static str>| async move {
            let tls = tokio_rustls::TlsConnector::from(Arc::new(client_config(certificate, b"h2")));
            let connector = tower::service_fn(move |_: axum::http::Uri| {
                let tls = tls.clone();
                async move {
                    let tcp = tokio::net::TcpStream::connect(addr).await?;
                    let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
                    let stream = tls.connect(name, tcp).await?;
                    Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
                }
            });
            let channel = tonic::transport::Endpoint::from_static("http://localhost")
                .connect_with_connector(connector)
                .await?;
            Ok::<_, tonic::transport::Error>(GrpcInferenceServiceClient::new(channel))
        };

        let mut client = connect(Some("client")).await.unwrap();
        let response = client
            .model_infer(request(vec![1, 784], Some(vec![0.5; 784]), None))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.outputs[0].shape, [1, 10]);

        // Without a client certificate the call fails, whether at connect or on the first call
        let refused = match connect(None).await {
            Err(_) => true,
            Ok(mut client) => client.server_live(pb::ServerLiveRequest {}).await.is_err(),
        };
        assert!(refused);
    }

    pub(crate) fn request(
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
