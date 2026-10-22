//! Property tests of the request decoders: random valid and broken bodies through the router,
//! for `/predict` (JSON and raw `f32`) and the Open Inference Protocol (one tensor and named
//! tensors). A broken body is the client's error, a `4xx`; it never panics the handler and
//! never answers `5xx`.

use std::sync::OnceLock;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use proptest::prelude::*;
use serde_json::{Value, json};

use super::tests::{dummy_model, router, send};
use super::v2::INFERENCE_HEADER_CONTENT_LENGTH;
use super::{BATCH_SIZE_HEADER, ServerConfig};
use crate::model::{DummyTensorsModel, Model};

/// The dummy model's item: 784 values in, 10 out
const ITEM: usize = 784;

/// One runtime and one server for every case: `/predict` and `/v2/models/test_model` serve
/// the simple dummy model, `/v2/models/tensors` the model of named tensors
fn server() -> &'static (tokio::runtime::Runtime, Router, Router) {
    static SERVER: OnceLock<(tokio::runtime::Runtime, Router, Router)> = OnceLock::new();
    SERVER.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (simple, tensors) = runtime.block_on(async {
            let tensors = Model::new(Box::new(DummyTensorsModel::new(None)), "dummy", 0);
            (
                router(dummy_model(), &ServerConfig::default()).unwrap(),
                router(tensors, &ServerConfig::default()).unwrap(),
            )
        });
        (runtime, simple, tensors)
    })
}

/// The status of `request` on `app`, failing the case on a server error
fn status_of(app: &Router, request: Request<Body>) -> Result<StatusCode, TestCaseError> {
    let (runtime, _, _) = server();
    let (status, body) = runtime.block_on(send(app.clone(), request));
    prop_assert!(
        status.is_success() || status.is_client_error(),
        "{status}: {}",
        String::from_utf8_lossy(&body)
    );
    Ok(status)
}

fn post(path: &str, content_type: Option<&str>) -> axum::http::request::Builder {
    let mut builder = Request::builder().uri(path).method("POST");
    if let Some(content_type) = content_type {
        builder = builder.header("content-type", content_type);
    }
    builder
}

/// A small number, as a JSON value of one of its forms
fn small_number() -> impl Strategy<Value = Value> {
    prop_oneof![
        (-1000i64..1000).prop_map(Value::from),
        (-1000.0f64..1000.0).prop_map(Value::from),
    ]
}

/// Any JSON value, of the shapes a request is made of
fn any_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        small_number(),
        any::<f64>().prop_map(|x| json!(x)),
        any::<i64>().prop_map(Value::from),
        any::<u64>().prop_map(Value::from),
        "[a-z]{0,6}".prop_map(Value::from),
    ];
    leaf.prop_recursive(4, 64, 8, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..8).prop_map(Value::from),
            prop::collection::vec(
                (
                    prop::sample::select(vec!["input", "inputs", "data", "shape", "x"]),
                    inner
                ),
                0..4
            )
            .prop_map(|fields| Value::Object(
                fields
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect()
            )),
        ]
    })
}

/// A valid `/predict` body: one item or a batch of a few
fn valid_predict() -> impl Strategy<Value = Value> {
    let item = || prop::collection::vec(-100.0f32..100.0, ITEM..=ITEM);
    prop_oneof![
        item().prop_map(|input| json!({ "input": input })),
        prop::collection::vec(item(), 1..4).prop_map(|inputs| json!({ "inputs": inputs })),
    ]
}

/// `bytes` with a few bytes removed, replaced or inserted, and maybe cut short
fn mutated(bytes: Vec<u8>) -> impl Strategy<Value = Vec<u8>> {
    (
        prop::collection::vec(
            (
                any::<prop::sample::Index>(),
                0u8..3,
                prop::sample::select(b"[]{},:\"-.e019 ".to_vec()),
            ),
            1..4,
        ),
        any::<Option<prop::sample::Index>>(),
    )
        .prop_map(move |(edits, cut)| {
            let mut bytes = bytes.clone();
            for (at, kind, byte) in edits {
                let i = at.index(bytes.len() + 1);
                match kind {
                    0 if i < bytes.len() => {
                        bytes.remove(i);
                    }
                    1 if i < bytes.len() => bytes[i] = byte,
                    _ => bytes.insert(i, byte),
                }
            }
            if let Some(cut) = cut {
                bytes.truncate(cut.index(bytes.len() + 1));
            }
            bytes
        })
}

fn json_content_type() -> impl Strategy<Value = Option<&'static str>> {
    prop::sample::select(vec![
        Some("application/json"),
        Some("application/json; charset=utf-8"),
        Some("application/vnd.api+json"),
        Some("text/plain"),
        Some("application/x-www-form-urlencoded"),
        None,
    ])
}

/// An `X-Batch-Size` header, well-formed or not
fn batch_header() -> impl Strategy<Value = Option<String>> {
    prop_oneof![
        Just(None),
        (0usize..5).prop_map(|n| Some(n.to_string())),
        prop::sample::select(vec![
            " 2 ",
            "-1",
            "1.5",
            "abc",
            "",
            "1001",
            "1000",
            "18446744073709551616",
        ])
        .prop_map(|s| Some(s.to_string())),
    ]
}

/// `items` items of raw `f32`, some bytes added or taken away
fn raw_body() -> impl Strategy<Value = (usize, Vec<u8>)> {
    (
        0usize..4,
        -6isize..6,
        prop_oneof![
            (-100.0f32..100.0).prop_map(f32::to_le_bytes),
            any::<u32>().prop_map(u32::to_le_bytes),
        ],
    )
        .prop_map(|(items, delta, value)| {
            let len = (items * ITEM * 4).saturating_add_signed(delta);
            (items, value.iter().copied().cycle().take(len).collect())
        })
}

/// A tensor of a v2 request, often wrong in its name, datatype, shape or data
fn v2_tensor() -> impl Strategy<Value = (Value, Option<Vec<u8>>)> {
    let dim = prop_oneof![
        4 => 0i64..6,
        1 => Just(-1i64),
        1 => Just(ITEM as i64),
        2 => Just(1i64 << 33),
        1 => Just(i64::MAX),
    ];
    // Mostly a name and datatype one of the models takes, so the shape and data are checked
    let name_and_type = prop_oneof![
        3 => prop::sample::select(vec![("ids", "INT64"), ("scale", "FP32"), ("input", "FP32")]),
        1 => (
            prop::sample::select(vec!["ids", "scale", "input", "x"]),
            prop::sample::select(vec!["INT64", "INT32", "FP32", "FP64", "BYTES", ""]),
        ),
    ];
    (
        name_and_type,
        prop::collection::vec(dim, 0..4),
        prop_oneof![
            // JSON data: a flat or nested list of small numbers
            prop::collection::vec(small_number(), 0..12).prop_map(|v| (Some(json!(v)), None)),
            prop::collection::vec(prop::collection::vec(small_number(), 0..5), 0..4)
                .prop_map(|v| (Some(json!(v)), None)),
            any_json().prop_map(|v| (Some(v), None)),
            // Binary data, a size of its own
            prop::collection::vec(any::<u8>(), 0..40).prop_map(|b| (None, Some(b))),
        ],
        any::<bool>(),
    )
        .prop_map(|((name, datatype), shape, (data, binary), lie)| {
            let mut tensor = json!({ "name": name, "datatype": datatype, "shape": shape });
            match (data, binary) {
                (Some(data), _) => {
                    tensor["data"] = data;
                    (tensor, None)
                }
                (None, Some(binary)) => {
                    // Sometimes a size the body does not have
                    let size = if lie {
                        binary.len() * 3 + 1
                    } else {
                        binary.len()
                    };
                    tensor["parameters"] = json!({ "binary_data_size": size });
                    (tensor, Some(binary))
                }
                (None, None) => (tensor, None),
            }
        })
}

/// A v2 request body: the JSON, then any binary data, with the header that tells them apart
/// (sometimes a wrong one)
fn v2_request() -> impl Strategy<Value = (Vec<u8>, Option<String>)> {
    (
        prop::collection::vec(v2_tensor(), 0..4),
        any::<bool>(),
        prop_oneof![
            4 => Just(None),
            1 => (0usize..100_000).prop_map(Some),
            1 => Just(Some(usize::MAX)),
        ],
    )
        .prop_map(|(tensors, binary_output, header_override)| {
            let mut binary = Vec::new();
            let inputs: Vec<Value> = tensors
                .into_iter()
                .map(|(tensor, data)| {
                    binary.extend(data.unwrap_or_default());
                    tensor
                })
                .collect();
            let mut body = serde_json::to_vec(&json!({
                "inputs": inputs,
                "parameters": { "binary_data_output": binary_output },
            }))
            .unwrap();
            let json_len = body.len();
            let header = match header_override {
                Some(len) => Some(len.to_string()),
                None => (!binary.is_empty()).then(|| json_len.to_string()),
            };
            body.extend(binary);
            (body, header)
        })
}

fn v2_post(model: &str, body: Vec<u8>, header: Option<String>) -> Request<Body> {
    let mut builder = post(
        &format!("/v2/models/{model}/infer"),
        Some(if header.is_some() {
            "application/octet-stream"
        } else {
            "application/json"
        }),
    );
    if let Some(header) = header {
        builder = builder.header(INFERENCE_HEADER_CONTENT_LENGTH, header);
    }
    builder.body(Body::from(body)).unwrap()
}

proptest! {
    // At most 128 cases; PROPTEST_CASES lowers it
    #![proptest_config(ProptestConfig::with_cases(
        ProptestConfig::default().cases.min(128)
    ))]

    #[test]
    fn valid_predict_bodies_are_answered(body in valid_predict()) {
        let (_, simple, _) = server();
        let request = post("/predict", Some("application/json"))
            .body(Body::from(body.to_string()))
            .unwrap();
        prop_assert_eq!(status_of(simple, request)?, StatusCode::OK);
    }

    #[test]
    fn broken_predict_json_is_a_client_error(
        body in valid_predict().prop_flat_map(|v| mutated(v.to_string().into_bytes())),
        content_type in json_content_type(),
    ) {
        let (_, simple, _) = server();
        let request = post("/predict", content_type).body(Body::from(body)).unwrap();
        status_of(simple, request)?;
    }

    #[test]
    fn any_json_on_predict_is_a_client_error(body in any_json()) {
        let (_, simple, _) = server();
        let request = post("/predict", Some("application/json"))
            .body(Body::from(body.to_string()))
            .unwrap();
        prop_assert!(status_of(simple, request)?.is_client_error());
    }

    #[test]
    fn raw_f32_bodies_are_answered_only_when_whole(
        (items, body) in raw_body(),
        header in batch_header(),
    ) {
        let (_, simple, _) = server();
        let len = body.len();
        let finite = body
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&b| f32::from_le_bytes(b).is_finite());
        let mut builder = post("/predict", Some("application/octet-stream"));
        if let Some(header) = &header {
            builder = builder.header(BATCH_SIZE_HEADER, header.as_str());
        }
        let status = status_of(simple, builder.body(Body::from(body)).unwrap())?;
        let batch = match &header {
            None => Some(1),
            Some(h) => h.trim().parse::<usize>().ok(),
        };
        let whole = batch.is_some_and(|b| (1..=1000).contains(&b) && len == b * ITEM * 4);
        prop_assert_eq!(status == StatusCode::OK, whole && finite, "{} items, {} bytes", items, len);
    }

    #[test]
    fn v2_requests_for_one_tensor_never_fail_the_server((body, header) in v2_request()) {
        let (_, simple, _) = server();
        status_of(simple, v2_post("test_model", body, header))?;
    }

    #[test]
    fn v2_requests_of_named_tensors_never_fail_the_server((body, header) in v2_request()) {
        let (_, _, tensors) = server();
        status_of(tensors, v2_post("tensors", body, header))?;
    }

    #[test]
    fn broken_v2_json_is_a_client_error(
        body in v2_request().prop_flat_map(|(body, _)| mutated(body)),
    ) {
        let (_, simple, tensors) = server();
        for (app, model) in [(simple, "test_model"), (tensors, "tensors")] {
            prop_assert!(status_of(app, v2_post(model, body.clone(), None))?.is_client_error());
        }
    }
}

/// Shapes whose element count overflows `usize`: refused as the client's error, not a panic
/// (debug builds) or a wrapped count that lets the tensor through (release builds)
#[test]
fn shapes_too_large_to_count_are_refused() {
    let (runtime, _, tensors) = server();
    let huge = 1i64 << 33;
    for (ids, scale) in [
        (json!([huge, huge, 4]), json!([1])),
        (json!([0, huge, huge]), json!([0])),
        (json!([1, 4]), json!([huge, huge])),
    ] {
        let body = json!({ "inputs": [
            { "name": "ids", "datatype": "INT64", "shape": ids, "data": [] },
            { "name": "scale", "datatype": "FP32", "shape": scale, "data": [] },
        ]});
        let request = v2_post("tensors", body.to_string().into_bytes(), None);
        let (status, body) = runtime.block_on(send(tensors.clone(), request));
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{}",
            String::from_utf8_lossy(&body)
        );
    }
}
