//! The fabric on the server: receptors mounted as routes, `GET /nerve` describing neurons,
//! pathways and measured latencies, and `GET /nerve/effectors/{name}/events` for the SSE
//! effectors. `Server::nerve` merges these into the server's router, and `/metrics` carries
//! the fabric's metrics.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header::CONTENT_TYPE};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};

use super::fabric::{Fabric, Outcome};
use super::neuron::Quiet;
use super::receptor::{HttpMethod, Ingest, Receptor};
use super::signal::{NeuronId, Reply};

/// What became of the request's signal, on responses the chain did not answer itself
pub const OUTCOME_HEADER: HeaderName = HeaderName::from_static("x-furnace-nerve");
/// The signal's id
pub const SIGNAL_HEADER: HeaderName = HeaderName::from_static("x-furnace-signal");

struct Mounted {
    fabric: Fabric,
    id: NeuronId,
    receptor: Arc<dyn Receptor>,
}

/// The fabric's routes; starts the receptors
pub(crate) fn routes<S: Clone + Send + Sync + 'static>(fabric: Fabric) -> Router<S> {
    fabric.start();
    let mut router = Router::new()
        .route("/nerve", get(describe))
        .route("/nerve/effectors/{name}/events", get(events))
        .with_state(fabric.clone());
    for (id, receptor) in fabric.receptors() {
        let Some(mount) = receptor.mount() else {
            continue;
        };
        let mounted = Arc::new(Mounted {
            fabric: fabric.clone(),
            id: *id,
            receptor: receptor.clone(),
        });
        let handler = match mount.method {
            HttpMethod::Post => post(ingest),
            HttpMethod::Get => get(ingest),
        };
        router = router.merge(
            Router::new()
                .route(&mount.path, handler)
                .with_state(mounted),
        );
    }
    router
}

async fn describe(State(fabric): State<Fabric>) -> Json<super::metrics::Description> {
    Json(fabric.describe())
}

/// One request, one signal: the chain answers through a `Respond` effector, or the request
/// is told what happened
async fn ingest(State(mounted): State<Arc<Mounted>>, headers: HeaderMap, body: Bytes) -> Response {
    let content_type = headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok());
    let Some(signal) = mounted.receptor.ingest(Ingest {
        body,
        content_type,
        headers: &headers,
    }) else {
        return (StatusCode::BAD_REQUEST, "the receptor ignored this request").into_response();
    };
    let (reply, mut receiver) = Reply::new();
    let id = signal.id;
    let wait = mounted.receptor.respond_within();
    // The synchronous chain runs here, on this thread, before the next line
    let outcome = mounted
        .fabric
        .stimulate(mounted.id, signal.with_reply(reply));
    let answer = match receiver.try_recv() {
        Ok(answer) => Some(answer),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            if outcome.queued > 0 && !wait.is_zero() =>
        {
            tokio::time::timeout(wait, receiver)
                .await
                .ok()
                .and_then(Result::ok)
        }
        Err(_) => None,
    };
    let mut response = match answer {
        Some(answer) => (
            StatusCode::from_u16(answer.status).unwrap_or(StatusCode::OK),
            [(
                CONTENT_TYPE,
                HeaderValue::from_str(&answer.content_type)
                    .unwrap_or(HeaderValue::from_static("application/octet-stream")),
            )],
            answer.body,
        )
            .into_response(),
        None => {
            let status = if outcome.queued > 0 {
                StatusCode::ACCEPTED
            } else {
                StatusCode::NO_CONTENT
            };
            status.into_response()
        }
    };
    let headers = response.headers_mut();
    headers.insert(OUTCOME_HEADER, describe_outcome(&outcome));
    headers.insert(SIGNAL_HEADER, HeaderValue::from(id.0));
    response
}

/// `quiet:habituation`, `queued`, `acted:2`, `done`
fn describe_outcome(outcome: &Outcome) -> HeaderValue {
    if outcome.is_quiet() {
        return HeaderValue::from_static(match outcome.quiet {
            Some((_, Quiet::Threshold)) => "quiet:threshold",
            Some((_, Quiet::Habituation)) => "quiet:habituation",
            Some((_, Quiet::Refractory)) => "quiet:refractory",
            Some((_, Quiet::NotInterested)) => "quiet:not_interested",
            None => "quiet",
        });
    }
    match (
        outcome.queued,
        outcome.acted,
        outcome.unrouted,
        outcome.no_pathway,
    ) {
        (0, 0, 0, 0) => return HeaderValue::from_static("done"),
        (1.., 0, 0, 0) => return HeaderValue::from_static("queued"),
        (0, 1, 0, 0) => return HeaderValue::from_static("acted:1"),
        (0, 2, 0, 0) => return HeaderValue::from_static("acted:2"),
        _ => {}
    }
    let mut parts = Vec::new();
    if outcome.queued > 0 {
        parts.push("queued".to_string());
    }
    if outcome.acted > 0 {
        parts.push(format!("acted:{}", outcome.acted));
    }
    if outcome.unrouted > 0 {
        parts.push(format!("unrouted:{}", outcome.unrouted));
    }
    if outcome.no_pathway > 0 {
        parts.push(format!("no-pathway:{}", outcome.no_pathway));
    }
    HeaderValue::from_str(&parts.join(",")).unwrap_or(HeaderValue::from_static("done"))
}

/// Subscribe to an SSE effector's commands
async fn events(State(fabric): State<Fabric>, Path(name): Path<String>) -> Response {
    let Some(id) = fabric.neuron(&name) else {
        return (StatusCode::NOT_FOUND, format!("no neuron named '{name}'")).into_response();
    };
    let Some(receiver) = fabric.events(id) else {
        return (
            StatusCode::NOT_FOUND,
            format!("'{name}' does not publish events"),
        )
            .into_response();
    };
    let stream = futures_util::stream::unfold(receiver, |mut receiver| async move {
        loop {
            match receiver.recv().await {
                Ok(bytes) => {
                    let event = SseEvent::default().data(&*String::from_utf8_lossy(&bytes));
                    return Some((Ok::<_, Infallible>(event), receiver));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}
