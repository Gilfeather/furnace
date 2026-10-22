//! The motor hierarchy's tests. The motor side runs on each test's runtime with its clock
//! paused: delays and deadlines pass at once and in order, and the tests wait on what the
//! fabric sends them, never on the wall clock.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::*;
use crate::nerve::effector::{Act, ActError, ActResult, Action, Effector, Retry};
use crate::nerve::fabric::{BuildError, Fabric, FabricBuilder, Nerve};
use crate::nerve::neuron::{FnNeuron, NeuronKind};
use crate::nerve::signal::{Command, Signal, Urgency};
use crate::nerve::testing::{command, pass};

/// A fabric whose motor side runs on this test's runtime
fn builder() -> FabricBuilder {
    Fabric::builder().efferent(tokio::runtime::Handle::current())
}

/// A body part's effector: logs each command (`name args`) and its end (`name done`),
/// fails its first `fail` commands, and answers `value` after `delay`
struct Part {
    name: &'static str,
    log: mpsc::UnboundedSender<String>,
    fail: AtomicUsize,
    delay: Duration,
    value: Value,
}

fn part(name: &'static str, log: &mpsc::UnboundedSender<String>) -> Part {
    Part {
        name,
        log: log.clone(),
        fail: AtomicUsize::new(0),
        delay: Duration::ZERO,
        value: Value::Null,
    }
}

impl Part {
    fn failing(self, n: usize) -> Self {
        Self {
            fail: AtomicUsize::new(n),
            ..self
        }
    }

    fn delay(self, delay: Duration) -> Self {
        Self { delay, ..self }
    }

    fn value(self, value: Value) -> Self {
        Self { value, ..self }
    }

    /// With one try only: the program above does its own retrying
    fn once(self) -> Effector {
        Effector::new(self).no_retry()
    }
}

impl Act for Part {
    fn name(&self) -> &str {
        self.name
    }

    fn act(&self, command: &Command, _: &Signal) -> Action {
        let _ = self.log.send(format!("{} {}", self.name, command.args));
        if self
            .fail
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            return Action::Done(Err(ActError::fatal("slipped")));
        }
        if self.delay.is_zero() {
            return Action::value(self.value.clone());
        }
        let (name, log, delay, value) =
            (self.name, self.log.clone(), self.delay, self.value.clone());
        Action::pending(async move {
            tokio::time::sleep(delay).await;
            let _ = log.send(format!("{name} done"));
            Ok(value)
        })
    }
}

/// An effector whose act is `f` of the command's arguments
struct Body<F> {
    name: &'static str,
    f: F,
}

impl<F: Fn(&Value) -> ActResult + Send + Sync + 'static> Act for Body<F> {
    fn name(&self) -> &str {
        self.name
    }

    fn act(&self, command: &Command, _: &Signal) -> Action {
        Action::Done((self.f)(&command.args))
    }
}

/// A neuron that hands the test the payload of every signal reaching it: wired after a
/// motor neuron, it hears the reports a brain would
fn recorder(name: &str) -> (impl Neuron, mpsc::UnboundedReceiver<Value>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let neuron = FnNeuron::reflex(name, move |signal: Signal, _| {
        let _ = sender.send(signal.payload.to_json().unwrap_or_default());
        Firing::Quiet
    });
    (neuron, receiver)
}

/// Everything logged so far
fn drain(log: &mut mpsc::UnboundedReceiver<String>) -> Vec<String> {
    std::iter::from_fn(|| log.try_recv().ok()).collect()
}

fn motor_metrics(fabric: &Fabric, name: &str) -> Value {
    fabric
        .metrics()
        .neurons
        .into_iter()
        .find(|n| n.name == name)
        .map(|n| n.extra)
        .unwrap_or_default()
}

#[tokio::test(start_paused = true)]
async fn a_command_to_a_motor_neuron_arrives_and_its_outcome_goes_up() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let hand = b.motor("hand", |m| async move {
        let force = m.args()["force"].clone();
        m.command("fingers", json!({"force": force})).await
    });
    b.effector(part("fingers", &log).value(json!({"pressure": 0.7})).once());
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(hand, brain);
    let fabric = b.build().unwrap();

    // Addressed to the junction, not to an effector: it is taken, not unrouted
    let outcome = fabric.stimulate(r, command("hand", json!({"force": 0.7})));
    assert_eq!((outcome.acted, outcome.unrouted), (1, 0));
    // The effector's measured value comes back through the hand, and up to the brain
    assert_eq!(
        reports.recv().await.unwrap(),
        json!({"action": "hand", "status": "done", "result": {"pressure": 0.7}})
    );
    assert_eq!(drain(&mut parts), [r#"fingers {"force":0.7}"#]);
    assert_eq!(fabric.metrics().unrouted, 0);
    assert_eq!(motor_metrics(&fabric, "hand")["done"], 1);
}

#[tokio::test(start_paused = true)]
async fn a_command_breaks_down_through_several_levels() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let arm = b.motor("arm", |m| async move {
        m.command("shoulder", json!({"angle": 30})).await?;
        m.command("hand", json!({"curl": 0.5})).await
    });
    b.motor("hand", |m| async move {
        let curl = m.args()["curl"].clone();
        let fingers =
            ["thumb", "index"].map(|f| Command::new(&format!("finger:{f}"), json!({"curl": curl})));
        Ok(json!({"fingers": m.together(fingers).await?}))
    });
    for finger in ["thumb", "index"] {
        b.motor(&format!("finger:{finger}"), move |m| async move {
            m.command(&format!("{finger}_joint"), m.args().clone())
                .await
        });
    }
    b.effector(part("shoulder", &log).once());
    b.effector(part("thumb_joint", &log).value(json!(1)).once());
    b.effector(part("index_joint", &log).value(json!(2)).once());
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(arm, brain);
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("arm", Value::Null));
    assert_eq!(
        reports.recv().await.unwrap(),
        json!({"action": "arm", "status": "done", "result": {"fingers": [1, 2]}})
    );
    assert_eq!(
        drain(&mut parts),
        [
            r#"shoulder {"angle":30}"#,
            r#"thumb_joint {"curl":0.5}"#,
            r#"index_joint {"curl":0.5}"#,
        ]
    );
    // The hand and the fingers answered their commanders, not the brain
    assert!(reports.try_recv().is_err());
    for name in ["arm", "hand", "finger:thumb", "finger:index"] {
        assert_eq!(motor_metrics(&fabric, name)["done"], 1, "{name}");
    }
}

#[tokio::test(start_paused = true)]
async fn together_runs_at_once_and_await_in_turn() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let dress = b.motor("dress", |m| async move {
        let started = tokio::time::Instant::now();
        m.together([
            Command::new("left_sleeve", Value::Null),
            Command::new("right_sleeve", Value::Null),
        ])
        .await?;
        let sleeves = started.elapsed();
        m.command("zip", Value::Null).await?;
        let ms = |d: Duration| d.as_millis() as u64;
        Ok(json!({"sleeves_ms": ms(sleeves), "all_ms": ms(started.elapsed())}))
    });
    let ms = Duration::from_millis;
    b.effector(part("left_sleeve", &log).delay(ms(30)).once());
    b.effector(part("right_sleeve", &log).delay(ms(20)).once());
    b.effector(part("zip", &log).delay(ms(10)).once());
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(dress, brain);
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("dress", Value::Null));
    // Both sleeves at once (30 ms, not 50), then the zip
    assert_eq!(
        reports.recv().await.unwrap()["result"],
        json!({"sleeves_ms": 30, "all_ms": 40})
    );
    assert_eq!(
        drain(&mut parts),
        [
            "left_sleeve null",
            "right_sleeve null",
            "right_sleeve done",
            "left_sleeve done",
            "zip null",
            "zip done",
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn a_program_waits_for_a_receptor_signal() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let eyes = b.neuron(pass("eyes"));
    let fetch = b.motor("fetch", |m| async move {
        let mut eyes = m.sense("eyes")?;
        // Looking from here on
        m.command("ready", Value::Null).await?;
        let coat = eyes
            .until(|s| s.payload.as_text().is_some_and(|t| t.contains("coat")))
            .await
            .ok_or(MotionError::Cancelled)?;
        let place = coat.payload.as_text().unwrap_or_default().to_string();
        m.command("arm", json!({"reach": place})).await
    });
    b.effector(part("ready", &log).once());
    b.effector(part("arm", &log).once());
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(fetch, brain);
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("fetch", Value::Null));
    assert_eq!(parts.recv().await.unwrap(), "ready null");
    let seen = |text: &str| Signal::new(crate::nerve::Feature::NONE, Payload::text(text));
    fabric.stimulate(eyes, seen("a chair"));
    fabric.stimulate(eyes, seen("a coat on the hook"));
    assert_eq!(reports.recv().await.unwrap()["status"], "done");
    assert_eq!(drain(&mut parts), [r#"arm {"reach":"a coat on the hook"}"#]);
}

#[tokio::test(start_paused = true)]
async fn the_deadline_stops_a_program_and_a_step_can_time_out() {
    let mut b = builder();
    let r = b.neuron(pass("in"));
    b.neuron(pass("eyes"));
    // Looks forever: only the command's deadline ends it
    let stare = b.motor("stare", |m| async move {
        let mut eyes = m.sense("eyes")?;
        eyes.next().await;
        Ok(Value::Null)
    });
    // Looks for a while, then does without
    let glance = b.motor("glance", |m| async move {
        let mut eyes = m.sense("eyes")?;
        let looked = m
            .within(Duration::from_millis(100), async {
                eyes.next().await.ok_or(MotionError::Cancelled)
            })
            .await;
        match looked {
            Err(MotionError::TimedOut) => Ok(json!("did without")),
            other => Ok(json!(format!("{other:?}"))),
        }
    });
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(stare, brain);
    b.reflex(glance, brain);
    let fabric = b.build().unwrap();

    let due = Urgency::within(Duration::from_secs(5));
    fabric.stimulate(r, command("stare", Value::Null).with_urgency(due));
    let report = reports.recv().await.unwrap();
    assert_eq!(report["action"], "stare");
    assert_eq!(report["status"], "timed_out");
    assert_eq!(motor_metrics(&fabric, "stare")["timed_out"], 1);

    fabric.stimulate(r, command("glance", Value::Null));
    assert_eq!(
        reports.recv().await.unwrap(),
        json!({"action": "glance", "status": "done", "result": "did without"})
    );
}

#[tokio::test(start_paused = true)]
async fn a_program_retries_and_adjusts_by_what_it_senses() {
    let calls = std::sync::Arc::new(AtomicUsize::new(0));
    let mut b = builder();
    let r = b.neuron(pass("in"));
    // The grip slips once, then reports the pressure it reached: the force asked for
    let grip = {
        let calls = calls.clone();
        Body {
            name: "grip",
            f: move |args: &Value| {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(ActError::fatal("slipped"));
                }
                Ok(json!({"pressure": args["force"]}))
            },
        }
    };
    b.effector(Effector::new(grip).no_retry());
    // Holds at a pressure of 60 or more: tries again after a slip, squeezes harder when it
    // is too weak, and never asks the brain
    let hold = b.motor("hold", |m| async move {
        let (mut force, mut slips) = (30, 0);
        loop {
            match m.command("grip", json!({"force": force})).await {
                Ok(felt) if felt["pressure"].as_u64() >= Some(60) => {
                    return Ok(json!({"force": force, "slips": slips}));
                }
                Ok(_) => force += 20,
                Err(_) if slips < 3 => slips += 1,
                Err(e) => return Err(e),
            }
        }
    });
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(hold, brain);
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("hold", Value::Null));
    assert_eq!(
        reports.recv().await.unwrap()["result"],
        json!({"force": 70, "slips": 1})
    );
    // A slip at 30, then 30, 50 and 70
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(fabric.metrics().reached_brain, 0);
}

#[tokio::test(start_paused = true)]
async fn a_failure_goes_up_to_whoever_commanded() {
    let (log, _parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let arm = b.motor(
        "arm",
        |m| async move { m.command("hand", Value::Null).await },
    );
    b.motor(
        "hand",
        |m| async move { m.command("finger", Value::Null).await },
    );
    b.motor("finger", |m| async move {
        m.command("joint", Value::Null).await
    });
    b.effector(part("joint", &log).failing(usize::MAX).once());
    let lost = b.motor(
        "lost",
        |m| async move { m.command("nobody", Value::Null).await },
    );
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(arm, brain);
    b.reflex(lost, brain);
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("arm", Value::Null));
    assert_eq!(
        reports.recv().await.unwrap(),
        json!({"action": "arm", "status": "failed", "reason": "joint: slipped"})
    );
    for name in ["arm", "hand", "finger"] {
        assert_eq!(motor_metrics(&fabric, name)["failed"], 1, "{name}");
    }

    fabric.stimulate(r, command("lost", Value::Null));
    assert_eq!(
        reports.recv().await.unwrap()["reason"],
        "no effector or motor neuron named 'nobody'"
    );
    assert_eq!(fabric.metrics().unrouted, 1);
}

#[tokio::test(start_paused = true)]
async fn stopping_a_program_stops_everything_below_it() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let dress = b.motor("dress", |m| async move {
        m.together([
            Command::new("arm", Value::Null),
            Command::new("sleeve", Value::Null),
        ])
        .await?;
        Ok(Value::Null)
    });
    b.motor(
        "arm",
        |m| async move { m.command("elbow", Value::Null).await },
    );
    // The elbow keeps failing in a way worth retrying, a second apart
    let elbow_log = log.clone();
    let stuck = Body {
        name: "elbow",
        f: move |_: &Value| {
            let _ = elbow_log.send("elbow".to_string());
            Err(ActError::retry("stuck"))
        },
    };
    let elbow = Effector::new(stuck).retry(Retry {
        attempts: 100,
        backoff: Duration::from_secs(1),
        max_backoff: Duration::from_secs(1),
    });
    b.effector(elbow.clone());
    b.effector(part("sleeve", &log).delay(Duration::from_secs(3600)).once());
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(dress, brain);
    let fabric = b.build().unwrap();

    assert!(!fabric.stop("dress"), "nothing runs yet");
    fabric.stimulate(r, command("dress", Value::Null));
    let mut tried = [parts.recv().await.unwrap(), parts.recv().await.unwrap()];
    tried.sort();
    assert_eq!(tried, ["elbow", "sleeve null"]);
    // "Stop": the dressing ends, and with it the arm and the elbow's retries
    assert!(fabric.stop("dress"));
    assert_eq!(
        reports.recv().await.unwrap(),
        json!({"action": "dress", "status": "cancelled", "reason": "stopped"})
    );
    // Long enough, in paused time, for a hundred retries
    tokio::time::sleep(Duration::from_secs(200)).await;
    assert_eq!(motor_metrics(&fabric, "arm")["cancelled"], 1);
    let m = elbow.metrics();
    assert_eq!((m.inflight, m.failed), (0, 0));
    assert!(m.retries <= 1, "{} retries", m.retries);
    assert!(drain(&mut parts).is_empty());

    // A newer command to the same motor neuron preempts the one running
    fabric.stimulate(r, command("dress", Value::Null));
    fabric.stimulate(r, command("dress", Value::Null));
    assert_eq!(reports.recv().await.unwrap()["status"], "cancelled");
    assert_eq!(motor_metrics(&fabric, "dress")["running"], 1);
    assert!(fabric.stop("dress"));
}

/// A brain that answers every question with the coat, and takes reports in silence
fn wardrobe_brain(name: &str) -> impl Neuron {
    FnNeuron::reflex(name, |signal: Signal, _| {
        let question = signal.payload.as_text()?.ends_with('?');
        question.then(|| signal.command("pick", json!({"item": "coat"})))
    })
    .with_kind(NeuronKind::Brain)
}

/// A program that asks which clothes to wear and puts on what it was told, or a scarf if
/// asking failed
async fn choose(m: Motion) -> Result<Value, MotionError> {
    let item = match m.ask(Payload::text("cold: what should I wear?")).await {
        Ok(answers) => answers
            .iter()
            .find_map(|a| a.payload.as_command().map(|c| c.args["item"].clone()))
            .unwrap_or_default(),
        Err(_) => json!("scarf"),
    };
    m.command("wear", json!({"item": item})).await?;
    Ok(item)
}

#[tokio::test(start_paused = true)]
async fn a_program_asks_the_brain_and_goes_on_with_the_answer() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let dress = b.motor("dress", choose);
    let brain = b.neuron(wardrobe_brain("brain"));
    b.reflex(dress, brain);
    b.effector(part("wear", &log).once());
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("dress", Value::Null));
    assert_eq!(parts.recv().await.unwrap(), r#"wear {"item":"coat"}"#);
    // The brain's answer went back to the program, not on to an effector named `pick`
    assert_eq!(fabric.metrics().unrouted, 0);
}

#[tokio::test(start_paused = true)]
async fn a_question_takes_the_pathway_its_deadline_allows() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let dress = b.motor("dress", choose);
    let brain = b.neuron(wardrobe_brain("brain"));
    // A quick reflex that always says the jumper
    let habit = b.neuron(FnNeuron::reflex("habit", |signal: Signal, _| {
        signal.command("pick", json!({"item": "jumper"}))
    }));
    b.reflex(dress, brain).weight(2.0);
    b.reflex(dress, habit);
    b.effector(part("wear", &log).once());
    let fabric = b.build().unwrap();
    // The brain has been measured at a second
    fabric.seed_latency(0, Duration::from_secs(1));

    // Time to think: the heavier pathway, the brain
    fabric.stimulate(r, command("dress", Value::Null));
    assert_eq!(parts.recv().await.unwrap(), r#"wear {"item":"coat"}"#);
    // Half a second: the brain cannot answer in time, the habit can
    let due = || Urgency::within(Duration::from_millis(500));
    fabric.stimulate(r, command("dress", Value::Null).with_urgency(due()));
    assert_eq!(parts.recv().await.unwrap(), r#"wear {"item":"jumper"}"#);
    // Neither in time: the question fails, and the program does without
    fabric.seed_latency(1, Duration::from_secs(1));
    fabric.stimulate(r, command("dress", Value::Null).with_urgency(due()));
    assert_eq!(parts.recv().await.unwrap(), r#"wear {"item":"scarf"}"#);
}

#[tokio::test(start_paused = true)]
async fn a_question_waits_in_a_deliberative_queue_for_the_brain() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let dress = b.motor("dress", choose);
    let brain = b.neuron(wardrobe_brain("brain"));
    b.deliberative(dress, brain);
    b.effector(part("wear", &log).once());
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("dress", Value::Null));
    assert_eq!(parts.recv().await.unwrap(), r#"wear {"item":"coat"}"#);
    assert_eq!(fabric.metrics().unrouted, 0);
}

#[tokio::test(start_paused = true)]
async fn a_brain_and_a_body_part_answering_each_other_forever_are_cut() {
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let again = b.motor("again", |_| async { Ok(Value::Null) });
    b.action(again, "Do it again.", Value::Null);
    // Whatever it hears, it says "again"
    let brain = b.neuron(
        FnNeuron::reflex("brain", |signal: Signal, _| {
            signal.command("again", Value::Null)
        })
        .with_kind(NeuronKind::Brain),
    );
    b.reflex(again, brain);
    let fabric = b.build().unwrap();

    fabric.stimulate(r, command("again", Value::Null));
    for _ in 0..10_000 {
        if fabric.0.loops.load(Ordering::SeqCst) > 0 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(fabric.0.loops.load(Ordering::SeqCst), 1);
    let started = motor_metrics(&fabric, "again")["started"].clone();
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    // Cut: nothing more happens
    assert_eq!(motor_metrics(&fabric, "again")["started"], started);
    assert!(started.as_u64().unwrap() < 32, "{started}");
}

#[test]
fn a_brain_sees_the_actions_and_not_the_parts_below_them() {
    let mut b = Fabric::builder();
    let dress = b.motor("get_dressed", |_| async { Ok(Value::Null) });
    b.motor("hand", |_| async { Ok(Value::Null) });
    let (log, _parts) = mpsc::unbounded_channel();
    b.effector(part("elbow", &log).once());
    b.action(
        dress,
        "Put on clothes for the weather.",
        json!({"warmth": "0 to 1"}),
    );
    let fabric = b.build().unwrap();
    assert_eq!(
        fabric.actions(),
        [ActionSpec {
            name: "get_dressed".into(),
            description: "Put on clothes for the weather.".into(),
            args: json!({"warmth": "0 to 1"}),
        }]
    );
    let described = fabric.describe();
    let action = |name: &str| {
        described
            .neurons
            .iter()
            .find(|n| n.name == name)
            .and_then(|n| n.action.clone())
    };
    assert!(action("get_dressed").is_some());
    assert!(action("hand").is_none());

    // Only motor neurons and effectors take commands
    let mut b = Fabric::builder();
    let reflex = b.neuron(pass("in"));
    b.action(reflex, "Not an action.", Value::Null);
    assert!(matches!(b.build(), Err(BuildError::NotAnAction(name)) if name == "in"));
}

#[cfg(feature = "llm")]
#[test]
fn an_llm_brain_is_prompted_with_the_actions_only() {
    use crate::llm::{EngineError, Event, GenerateRequest, LlmEngine, Prompt};
    use crate::nerve::brain::Llm;
    use std::sync::{Arc, Mutex};

    /// Keeps the requests and answers nothing
    #[derive(Debug, Clone, Default)]
    struct Prompts(Arc<Mutex<Vec<GenerateRequest>>>);

    impl LlmEngine for Prompts {
        fn name(&self) -> &str {
            "prompts"
        }
        fn max_context(&self) -> usize {
            4096
        }
        fn generate(&self, request: GenerateRequest) -> Result<mpsc::Receiver<Event>, EngineError> {
            self.0.lock().unwrap().push(request);
            Ok(mpsc::channel(1).1)
        }
    }

    let prompts = Prompts::default();
    let mut b = Fabric::builder();
    let r = b.neuron(pass("in"));
    let brain = b.neuron(Llm::new("brain", prompts.clone()));
    b.reflex(r, brain);
    let dress = b.motor("get_dressed", |_| async { Ok(Value::Null) });
    let stop = b.motor("stop", |m| async move {
        m.stop("get_dressed");
        Ok(Value::Null)
    });
    b.motor("hand", |_| async { Ok(Value::Null) });
    let (log, _parts) = mpsc::unbounded_channel();
    b.effector(part("elbow", &log).once());
    b.action(
        dress,
        "Put on clothes for the weather.",
        json!({"warmth": "0 to 1"}),
    );
    b.action(stop, "Stop getting dressed.", Value::Null);
    let fabric = b.build().unwrap();

    fabric.stimulate(
        r,
        Signal::new(crate::nerve::Feature::NONE, Payload::text("cold")),
    );
    let requests = prompts.0.lock().unwrap();
    let Prompt::Chat(messages) = &requests[0].input else {
        panic!("a chat");
    };
    let system = &messages[0].content;
    assert!(
        system.contains(
            "- get_dressed: Put on clothes for the weather. Args: {\"warmth\":\"0 to 1\"}\n\
             - stop: Stop getting dressed.\n"
        ),
        "{system}"
    );
    assert!(
        !system.contains("hand") && !system.contains("elbow"),
        "{system}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_flooded_nerve_holds_up_only_itself_and_lets_the_urgent_go_first() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let hour = Duration::from_secs(3600);
    b.nerve(Nerve::new("right_arm").in_flight(1), |b| {
        b.effector(part("right", &log).delay(hour).once());
    });
    b.nerve("left_arm", |b| {
        b.effector(part("left", &log).once());
    });
    let fabric = b.build().unwrap();

    // The right arm's one place is taken for an hour; the next two wait in its lane
    assert_eq!(fabric.stimulate(r, command("right", json!(1))).acted, 1);
    assert_eq!(fabric.stimulate(r, command("right", json!(2))).queued, 1);
    let urgent = Urgency::within(Duration::from_secs(36_000));
    let third = command("right", json!(3)).with_urgency(urgent);
    assert_eq!(fabric.stimulate(r, third).queued, 1);
    // The left arm is not held up
    assert_eq!(fabric.stimulate(r, command("left", json!(1))).acted, 1);
    assert_eq!(parts.recv().await.unwrap(), "right 1");
    assert_eq!(parts.recv().await.unwrap(), "left 1");
    let lanes = fabric.metrics().nerves;
    assert_eq!((lanes[0].in_flight, lanes[0].waiting), (1, 2));
    assert_eq!((lanes[1].in_flight, lanes[1].waiting), (0, 0));
    // As places free up, the urgent one goes first, then the one that came before it
    let mut order = Vec::new();
    for _ in 0..4 {
        order.push(parts.recv().await.unwrap());
    }
    assert_eq!(order, ["right done", "right 3", "right done", "right 2"]);
    assert_eq!(fabric.metrics().nerves[0].waited, 2);
}

#[tokio::test(start_paused = true)]
async fn inside_a_nerve_a_part_s_own_commands_never_wait() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let hand = b.nerve(Nerve::new("right_arm").in_flight(1), |b| {
        b.effector(part("knuckle", &log).delay(Duration::from_millis(5)).once());
        b.motor("finger", |m| async move {
            m.command("knuckle", Value::Null).await
        });
        b.motor("hand", |m| async move {
            m.together([
                Command::new("finger", Value::Null),
                Command::new("knuckle", Value::Null),
            ])
            .await?;
            Ok(json!("closed"))
        })
    });
    let (brain, mut reports) = recorder("brain");
    let brain = b.neuron(brain);
    b.reflex(hand, brain);
    let fabric = b.build().unwrap();

    // One place in the nerve, taken by the hand: its finger and knuckle go on all the same
    fabric.stimulate(r, command("hand", Value::Null));
    assert_eq!(reports.recv().await.unwrap()["result"], "closed");
    assert_eq!(drain(&mut parts).len(), 4);
    let lane = &fabric.metrics().nerves[0];
    assert_eq!((lane.in_flight, lane.waited), (0, 0));
    assert_eq!(
        fabric.describe().neurons[1].nerve.as_deref(),
        Some("right_arm")
    );
}

#[tokio::test(start_paused = true)]
async fn a_brain_reaches_its_actions_and_not_the_parts_below_them() {
    let (log, mut parts) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let brain = b.neuron(
        FnNeuron::reflex("brain", |signal: Signal, _| {
            vec![
                signal.command("elbow", Value::Null),
                signal.command("wave", Value::Null),
            ]
        })
        .with_kind(NeuronKind::Brain),
    );
    b.reflex(r, brain);
    b.effector(part("elbow", &log).once());
    let wave = b.effector(part("wave", &log).once());
    b.spine(brain).action(wave, "Wave.", Value::Null);
    let fabric = b.build().unwrap();

    let outcome = fabric.stimulate(
        r,
        Signal::new(crate::nerve::Feature::NONE, Payload::text("hi")),
    );
    assert_eq!((outcome.acted, outcome.unrouted), (1, 1));
    assert_eq!(drain(&mut parts), ["wave null"]);
}

#[tokio::test(start_paused = true)]
async fn the_spine_carries_senses_up_and_actions_down_and_back() {
    let (sender, mut heard) = mpsc::unbounded_channel();
    let mut b = builder();
    let r = b.neuron(pass("in"));
    let eyes = b.neuron(pass("eyes"));
    let brain = b.neuron(
        FnNeuron::reflex("brain", move |signal: Signal, _| {
            let _ = sender.send(
                signal
                    .payload
                    .to_json()
                    .unwrap_or_else(|| json!(signal.payload.as_text().unwrap_or_default())),
            );
            Firing::Quiet
        })
        .with_kind(NeuronKind::Brain),
    );
    let dress = b.motor("dress", |_| async { Ok(json!("dressed")) });
    b.spine(brain)
        .ascend(eyes)
        .action(dress, "Get dressed.", Value::Null);
    let fabric = b.build().unwrap();

    // Up: what the eyes see reaches the brain
    fabric.stimulate(
        eyes,
        Signal::new(crate::nerve::Feature::NONE, Payload::text("a coat")),
    );
    assert_eq!(heard.recv().await.unwrap(), "a coat");
    // Down and back: the brain's action, and how it ended
    fabric.stimulate(r, command("dress", Value::Null));
    assert_eq!(
        heard.recv().await.unwrap(),
        json!({"action": "dress", "status": "done", "result": "dressed"})
    );
    assert_eq!(fabric.actions()[0].name, "dress");
}
