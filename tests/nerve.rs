//! The nervous system on the server: receptors as routes, reflex arcs answering requests,
//! habituation at the receptor, the brain behind a deliberative pathway, `/nerve` and the
//! metrics.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode};
use furnace::Server;
use furnace::nerve::effector::{Channel, Delivery, Effector, FileAppend, Respond, Sse};
use furnace::nerve::receptor::Http;
use furnace::nerve::{
    Fabric, Firing, FnNeuron, Habituation, Mechanisms, NeuronKind, OUTCOME_HEADER, Summation,
};
use futures_util::StreamExt;
use tower::ServiceExt;

struct Demo {
    app: Router,
    alarms: mpsc::Receiver<Delivery>,
    journal: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

/// POST /events → habituating receptor → reflex classifier → `ALERT` lines: the alarm
/// channel and the response, at once; other lines: the brain, which journals them and
/// pushes a command to SSE subscribers
fn demo() -> Demo {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("journal.log");
    let mut b = Fabric::builder();
    let events = b.receptor(
        Http::new("events", "/events").text(),
        Mechanisms::NONE
            .threshold(0.6)
            .habituation(Habituation::new(Duration::from_secs(60))),
    );
    let classify = b.neuron(
        FnNeuron::reflex("classify", |signal, _| {
            let text = signal.payload.as_text().unwrap_or_default();
            if text.contains("ALERT") {
                Firing::Fire(vec![
                    signal.command("alarm", serde_json::json!({"text": text})),
                    signal.command(
                        "respond",
                        serde_json::json!({"status": 200, "body": "alarm raised"}),
                    ),
                ])
            } else {
                Firing::one(signal)
            }
        }),
        Mechanisms::NONE,
    );
    let brain = b.neuron(
        FnNeuron::reflex("brain", |signal, _| {
            let text = signal.payload.as_text().unwrap_or_default().to_string();
            Firing::Fire(vec![
                signal.command(
                    "journal",
                    serde_json::json!({"line": format!("seen: {text}")}),
                ),
                signal.command("push", serde_json::json!({"text": text})),
            ])
        })
        .with_kind(NeuronKind::Brain),
        Mechanisms::NONE,
    );
    let (alarm, alarms) = Channel::new("alarm", 16);
    let alarm = b.neuron(Effector::new(alarm), Mechanisms::NONE);
    b.neuron(Effector::new(Respond::new("respond")), Mechanisms::NONE);
    b.neuron(
        Effector::new(FileAppend::new("journal", &journal)),
        Mechanisms::NONE,
    );
    b.neuron(Effector::new(Sse::new("push", 16)), Mechanisms::NONE);
    b.reflex(events, classify, 1.0);
    b.reflex(classify, alarm, 1.0);
    b.deliberative(
        classify,
        brain,
        2.0,
        Summation::default().window(Duration::from_millis(1)),
    );
    let app = Server::new().nerve(b.build()).router();
    Demo {
        app,
        alarms,
        journal,
        _dir: dir,
    }
}

async fn post(app: &Router, path: &str, body: &str) -> (StatusCode, String, Bytes) {
    let request = Request::post(path)
        .header("content-type", "text/plain")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let outcome = response
        .headers()
        .get(OUTCOME_HEADER)
        .map(|v| v.to_str().unwrap().to_string())
        .unwrap_or_default();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, outcome, body)
}

async fn get_json(app: &Router, path: &str, accept: &str) -> (StatusCode, String) {
    let request = Request::get(path)
        .header("accept", accept)
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

#[tokio::test]
async fn reflex_arcs_answer_and_repeats_die_at_the_receptor() {
    let demo = demo();
    let (status, outcome, body) = post(&demo.app, "/events", "ALERT: smoke").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, Bytes::from("alarm raised"));
    assert_eq!(outcome, "acted:2");
    let alarm = demo.alarms.try_recv().unwrap();
    assert_eq!(alarm.command.args["text"], "ALERT: smoke");
    assert!(!alarm.signal.trace.was_queued());

    // The same stimulus again: habituated at the receptor, nothing fires
    let (status, outcome, body) = post(&demo.app, "/events", "ALERT: smoke").await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(outcome, "quiet:habituation");
    assert!(body.is_empty());
    assert!(demo.alarms.try_recv().is_err());

    // A different alert is new
    let (status, _, _) = post(&demo.app, "/events", "ALERT: flood").await;
    assert_eq!(status, StatusCode::OK);
    assert!(demo.alarms.try_recv().is_ok());
}

#[tokio::test]
async fn novel_signals_go_to_the_brain_and_commands_reach_effectors() {
    let demo = demo();
    // Subscribe to the SSE effector before the command is pushed
    let request = Request::get("/nerve/effectors/push/events")
        .body(Body::empty())
        .unwrap();
    let response = demo.app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let mut stream = response.into_body().into_data_stream();

    let (status, outcome, _) = post(&demo.app, "/events", "the door opened").await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(outcome, "queued");

    let chunk = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("an event within 10 s")
        .unwrap()
        .unwrap();
    let text = String::from_utf8(chunk.to_vec()).unwrap();
    let data = text
        .lines()
        .find_map(|l| l.strip_prefix("data: "))
        .expect("a data line");
    let event: serde_json::Value = serde_json::from_str(data).unwrap();
    assert_eq!(event["effector"], "push");
    assert_eq!(event["args"]["text"], "the door opened");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let journal = std::fs::read_to_string(&demo.journal).unwrap_or_default();
        if journal.contains("seen: the door opened") {
            break;
        }
        assert!(Instant::now() < deadline, "journal: {journal:?}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Unknown effectors in a chain are not routes
    let request = Request::get("/nerve/effectors/nobody/events")
        .body(Body::empty())
        .unwrap();
    let response = demo.app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn nerve_and_metrics_describe_the_fabric() {
    let demo = demo();
    post(&demo.app, "/events", "ALERT: one").await;
    post(&demo.app, "/events", "ALERT: one").await;
    post(&demo.app, "/events", "quiet night").await;

    let (status, body) = get_json(&demo.app, "/nerve", "application/json").await;
    assert_eq!(status, StatusCode::OK);
    let nerve: serde_json::Value = serde_json::from_str(&body).unwrap();
    let names: Vec<&str> = nerve["neurons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "events", "classify", "brain", "alarm", "respond", "journal", "push"
        ]
    );
    assert_eq!(nerve["neurons"][0]["kind"], "receptor");
    let threshold = nerve["neurons"][0]["mechanisms"]["threshold"]
        .as_f64()
        .unwrap();
    assert!((threshold - 0.6).abs() < 1e-6, "{threshold}");
    assert_eq!(nerve["neurons"][0]["counters"]["quiet_by_habituation"], 1);
    assert_eq!(nerve["neurons"][0]["describe"]["route"], "POST /EVENTS");
    let pathways = nerve["pathways"].as_array().unwrap();
    assert_eq!(pathways.len(), 3);
    assert_eq!(pathways[0]["kind"], "reflex");
    assert_eq!(pathways[0]["signals"], 2);
    assert!(pathways[0]["latency"]["p50_us"].as_f64().unwrap() > 0.0);
    assert_eq!(
        (&pathways[1]["to"], &pathways[1]["signals"]),
        (&serde_json::json!("alarm"), &serde_json::json!(1))
    );
    assert_eq!(pathways[2]["kind"], "deliberative");
    assert_eq!(pathways[2]["signals"], 1);
    // The reflex arc events → alarm was measured, stimulus to effector call
    let arcs = nerve["arcs"].as_array().unwrap();
    assert!(
        arcs.iter()
            .any(|a| a["from"] == "events" && a["to"] == "alarm" && a["latency"]["count"] == 1),
        "{arcs:?}"
    );

    // The brain fires on its worker, a moment after the request was accepted
    let deadline = Instant::now() + Duration::from_secs(10);
    let metrics: serde_json::Value = loop {
        let (status, body) = get_json(&demo.app, "/metrics", "application/json").await;
        assert_eq!(status, StatusCode::OK);
        let metrics: serde_json::Value = serde_json::from_str(&body).unwrap();
        if metrics["nerve"]["reached_brain"] == 1 {
            break metrics;
        }
        assert!(Instant::now() < deadline, "{}", metrics["nerve"]);
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    let nerve = &metrics["nerve"];
    assert_eq!(nerve["received"], 3);
    assert!((nerve["brain_fraction"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9);
    let alarm = nerve["neurons"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == "alarm")
        .unwrap();
    assert_eq!(alarm["extra"]["acted"], 1);
    assert_eq!(alarm["extra"]["breaker"], "closed");

    let (status, text) = get_json(&demo.app, "/metrics", "text/plain").await;
    assert_eq!(status, StatusCode::OK);
    for needle in [
        "furnace_nerve_signals_received_total 3\n",
        "furnace_nerve_brain_fraction 0.333",
        "furnace_nerve_neuron_quiet_total{neuron=\"events\",reason=\"habituation\"} 1\n",
        "furnace_nerve_pathway_latency_seconds{from=\"events\",to=\"classify\",quantile=\"0.99\"}",
        "furnace_nerve_arc_latency_seconds{from=\"events\",to=\"alarm\",quantile=\"0.5\"}",
        "furnace_nerve_effector_acted_total{neuron=\"alarm\"} 1\n",
        "# TYPE furnace_nerve_effector_acted_total counter\n",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
}

#[tokio::test]
async fn a_server_may_serve_a_fabric_alone() {
    let demo = demo();
    let (status, body) = get_json(&demo.app, "/models", "application/json").await;
    assert_eq!(status, StatusCode::OK);
    let models: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(models["default_model"], "");
    let (status, _) = get_json(&demo.app, "/readyz", "application/json").await;
    assert_eq!(status, StatusCode::OK);
}
