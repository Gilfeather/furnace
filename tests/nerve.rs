//! The nervous system on the server: receptors as routes, reflex arcs answering requests,
//! habituation at the receptor, the brain behind a deliberative pathway, `/nerve` and the
//! metrics.

use std::sync::mpsc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode};
use furnace::Server;
use furnace::nerve::effector::{ChannelOut, Delivery, Effector, Respond, SseOut};
use furnace::nerve::receptor::Http;
use furnace::nerve::{
    Fabric, Firing, FnNeuron, Habituation, Mechanisms, NeuronKind, OUTCOME_HEADER, Summation,
};
use futures_util::StreamExt;
use tower::ServiceExt;

struct Demo {
    app: Router,
    alarms: mpsc::Receiver<Delivery>,
    journal: mpsc::Receiver<Delivery>,
}

/// The next command `channel` carries; fails the test after 10 seconds
fn next(channel: &mpsc::Receiver<Delivery>) -> Delivery {
    channel
        .recv_timeout(Duration::from_secs(10))
        .expect("a command within 10 s")
}

/// POST /events → habituating receptor → reflex classifier → `ALERT` lines: the alarm
/// channel and the response, at once; other lines: the brain, which journals them and
/// pushes a command to SSE subscribers
fn demo() -> Demo {
    let mut b = Fabric::builder();
    let events = b.receptor(Http::new("events", "/events").text());
    b.gate(
        events,
        Mechanisms::NONE
            .threshold(0.6)
            .habituation(Habituation::new(Duration::from_secs(60))),
    );
    let classify = b.neuron(FnNeuron::reflex("classify", |signal, _| {
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
    }));
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
    );
    let (alarm, alarms) = ChannelOut::new("alarm", 16);
    let alarm = b.effector(alarm);
    b.effector(Respond::new("respond"));
    let (journal, journal_lines) = ChannelOut::new("journal", 16);
    let journal = b.effector(journal);
    let push = b.effector(SseOut::new("push", 16));
    b.spine(brain)
        .action(
            journal,
            "Write a line to the journal.",
            serde_json::json!({"line": "text"}),
        )
        .action(
            push,
            "Push to the subscribers.",
            serde_json::json!({"text": "text"}),
        );
    b.reflex(events, classify);
    b.reflex(classify, alarm);
    b.deliberative(classify, brain)
        .summation(Summation::default().window(Duration::from_millis(1)));
    let app = Server::new().nerve(b).router().unwrap();
    Demo {
        app,
        alarms,
        journal: journal_lines,
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

    let journal = next(&demo.journal);
    assert_eq!(journal.command.args["line"], "seen: the door opened");
    assert!(journal.signal.trace.reached_brain());

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
    assert!(pathways[0]["latency_ms"]["p50"].as_f64().unwrap() > 0.0);
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
            .any(|a| a["from"] == "events" && a["to"] == "alarm" && a["latency_ms"]["count"] == 1),
        "{arcs:?}"
    );

    // The brain fires on its worker, a moment after the request was accepted: once its
    // command is journaled, it has been counted
    next(&demo.journal);
    let (status, body) = get_json(&demo.app, "/metrics", "application/json").await;
    assert_eq!(status, StatusCode::OK);
    let metrics: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(metrics["nerve"]["reached_brain"], 1);
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
        "# TYPE furnace_nerve_pathway_latency_seconds histogram\n",
        "furnace_nerve_pathway_latency_seconds_bucket{from=\"events\",to=\"classify\",le=\"+Inf\"} ",
        "furnace_nerve_arc_latency_seconds_count{from=\"events\",to=\"alarm\"} 1\n",
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

/// Two receptors into one deliberative pathway ending in `Respond`: `/later` does not wait for
/// the answer, `/now` does. Every answer is also announced on the channel returned, after
/// `Respond` has acted on it.
fn queued_answers() -> (Router, Effector, mpsc::Receiver<Delivery>) {
    let mut b = Fabric::builder();
    let later = b.receptor(Http::new("later", "/later").text());
    let now = b.receptor(
        Http::new("now", "/now")
            .text()
            .respond_within(Duration::from_secs(5)),
    );
    let answer = b.neuron(FnNeuron::reflex("answer", |signal, _| {
        let text = signal.payload.as_text().unwrap_or_default().to_string();
        Firing::Fire(vec![
            signal.command(
                "respond",
                serde_json::json!({"status": 200, "body": format!("answered {text}")}),
            ),
            signal.command("answered", serde_json::json!(text)),
        ])
    }));
    let respond = Effector::new(Respond::new("respond"));
    let id = b.neuron(respond.clone());
    for receptor in [later, now] {
        b.deliberative(receptor, answer);
    }
    let (answered, answers) = ChannelOut::new("answered", 16);
    let answered = b.effector(answered);
    b.reflex(answer, id);
    b.reflex(answer, answered);
    (Server::new().nerve(b).router().unwrap(), respond, answers)
}

#[tokio::test]
async fn answers_nobody_waits_for_do_not_open_the_breaker() {
    let (app, respond, answers) = queued_answers();
    for i in 0..6 {
        let (status, outcome, _) = post(&app, "/later", &format!("{i}")).await;
        assert_eq!((status, outcome.as_str()), (StatusCode::ACCEPTED, "queued"));
    }
    for _ in 0..6 {
        next(&answers);
    }
    let metrics = respond.metrics();
    assert_eq!((metrics.acted, metrics.failed), (6, 0));
    // Six answers nobody waited for, and the next one waited for still arrives
    let (status, _, body) = post(&app, "/now", "7").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, Bytes::from("answered 7"));
}

#[tokio::test]
async fn an_answer_later_than_respond_within_is_not_waited_for() {
    let mut b = Fabric::builder();
    let soon = b.receptor(
        Http::new("soon", "/soon")
            .text()
            .respond_within(Duration::from_millis(1)),
    );
    // The answer is held until the test lets it go, long after the request gave up waiting
    let (release, released) = mpsc::channel::<()>();
    let released = std::sync::Mutex::new(released);
    let answer = b.neuron(FnNeuron::reflex("answer", move |signal, _| {
        released
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .expect("released within 10 s");
        Firing::Fire(vec![
            signal.command(
                "respond",
                serde_json::json!({"status": 200, "body": "late"}),
            ),
            signal.command("answered", serde_json::Value::Null),
        ])
    }));
    let respond = Effector::new(Respond::new("respond"));
    let id = b.neuron(respond.clone());
    let (answered, answers) = ChannelOut::new("answered", 16);
    let answered = b.effector(answered);
    b.deliberative(soon, answer);
    b.reflex(answer, id);
    b.reflex(answer, answered);
    let app = Server::new().nerve(b).router().unwrap();

    let (status, outcome, body) = post(&app, "/soon", "x").await;
    assert_eq!((status, outcome.as_str()), (StatusCode::ACCEPTED, "queued"));
    assert!(body.is_empty());
    // The late answer goes to nobody, and counts as acted, not failed
    release.send(()).unwrap();
    next(&answers);
    let metrics = respond.metrics();
    assert_eq!((metrics.acted, metrics.failed), (1, 0));
}
