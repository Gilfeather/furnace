//! "It is cold: get dressed", from the brain down to the knuckles and back:
//! `cargo run --example get_dressed`. No hardware and no model file: the thermometer, the
//! eyes, the joints and the brain are stand-ins that print what they do.
//!
//! 1. The thermometer reads 4 °C. The brain (a stand-in for an LLM) knows two actions,
//!    `get_dressed` and `stop`, and nothing of arms or fingers: it answers `get_dressed`.
//! 2. `get_dressed` looks for a coat in what the eyes stream. There is none, so it asks the
//!    brain a small question (what to wear instead) and waits for the answer.
//! 3. It breaks the answer down: `reach` moves the right arm's joints, `grasp` closes three
//!    fingers at once and grips; the first grip slips, so the hand tries again and squeezes
//!    harder by the pressure the fingertips report, without the brain.
//! 4. Both sleeves go on at once. `get_dressed` is done, and the brain hears so.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use furnace::nerve::effector::{Act, ActError, Action};
use furnace::nerve::prelude::*;
use furnace::nerve::{Cx, Feature};
use serde_json::{Value, json};

/// A line of the story: milliseconds since the start, indented by how deep in the body
fn log(depth: usize, line: impl std::fmt::Display) {
    static START: OnceLock<Instant> = OnceLock::new();
    let ms = START.get_or_init(Instant::now).elapsed().as_millis();
    println!("{ms:>5} ms  {}{line}", "    ".repeat(depth));
}

/// A joint: turns to the angle it is told in 20 ms, and says where it is
struct Joint(String);

impl Act for Joint {
    fn name(&self) -> &str {
        &self.0
    }

    fn act(&self, command: &Command, _: &Signal) -> Action {
        let angle = command.args["angle"].clone();
        log(4, format!("[{}] turns to {angle}°", self.0));
        Action::pending(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(json!({"angle": angle}))
        })
    }
}

/// The fingers' grip: squeezes with the force it is told and reports the pressure the
/// fingertips feel. The first grip slips.
#[derive(Default)]
struct Grip(AtomicUsize);

impl Act for Grip {
    fn name(&self) -> &str {
        "grip"
    }

    fn act(&self, command: &Command, _: &Signal) -> Action {
        if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
            log(4, "[grip] slipped");
            return Action::Done(Err(ActError::fatal("slipped")));
        }
        let force = command.args["force"].as_f64().unwrap_or(0.0);
        log(4, format!("[grip] fingertips feel {force:.1}"));
        Action::value(json!({"pressure": force}))
    }
}

/// The brain: what an LLM would do with these signals, written out. It sees the actions and
/// the signals; it never sees a joint.
fn brain(signal: Signal, cx: &Cx<'_>, done: &std::sync::mpsc::Sender<Value>) -> Option<Signal> {
    if let Some(text) = signal.payload.as_text() {
        if text.ends_with('?') {
            log(0, "brain: \"the jacket, on the hook\"");
            return Some(signal.command("pick", json!({"item": "jacket", "at": "hook"})));
        }
        let actions: Vec<&str> = cx.actions().iter().map(|a| a.name.as_str()).collect();
        log(0, format!("brain: {text}; I can {actions:?}: get_dressed"));
        return Some(signal.command("get_dressed", json!({"warmth": 0.9})));
    }
    // How an action ended
    let report = signal.payload.to_json()?;
    log(0, format!("brain: heard {report}"));
    let _ = done.send(report);
    None
}

/// Find something warm, asking the brain if nothing is in sight; reach, grasp, put it on
async fn get_dressed(m: Motion) -> Result<Value, MotionError> {
    let mut eyes = m.sense("eyes")?;
    log(1, "get_dressed: looking for a coat");
    let mut seen: Vec<String> = Vec::new();
    let looked = m
        .within(Duration::from_millis(300), async {
            while let Some(signal) = eyes.next().await {
                let what = signal.payload.as_text().unwrap_or_default().to_string();
                if what.contains("coat") {
                    return Ok(what);
                }
                if !seen.contains(&what) {
                    log(2, format!("eyes: {what}"));
                    seen.push(what);
                }
            }
            Err(MotionError::Cancelled)
        })
        .await;
    let (item, at) = match looked {
        Ok(coat) => ("coat".to_string(), coat),
        Err(MotionError::TimedOut) => {
            let question = format!(
                "No coat in sight, only: {}. What do I wear?",
                seen.join(", ")
            );
            log(1, format!("get_dressed: asks the brain \"{question}\""));
            let answers = m.ask(Payload::text(&question)).await?;
            let choice = answers
                .iter()
                .find_map(|a| a.payload.as_command())
                .ok_or_else(|| MotionError::failed("the brain had no answer"))?;
            let text = |key: &str| choice.args[key].as_str().unwrap_or_default().to_string();
            (text("item"), text("at"))
        }
        Err(e) => return Err(e),
    };
    log(1, format!("get_dressed: wear the {item} ({at})"));
    m.command("reach", json!({"at": at})).await?;
    m.command("grasp", json!({"item": item})).await?;
    log(1, "get_dressed: both sleeves at once");
    m.together([
        Command::new("sleeve:left", Value::Null),
        Command::new("sleeve:right", Value::Null),
    ])
    .await?;
    Ok(json!({"wearing": item}))
}

/// The right arm to where the thing is: the shoulder, then the elbow
async fn reach(m: Motion) -> Result<Value, MotionError> {
    log(
        2,
        format!("reach: to the {}", m.args()["at"].as_str().unwrap_or("?")),
    );
    m.command("shoulder:right", json!({"angle": 40})).await?;
    m.command("elbow:right", json!({"angle": 90})).await
}

/// Close three fingers at once, then grip; a slip is tried again, a light grip squeezed
/// harder, by what the fingertips report: a loop that never goes up to the brain
async fn grasp(m: Motion) -> Result<Value, MotionError> {
    let mut force = 0.4;
    for attempt in 1..=3 {
        log(2, format!("grasp: attempt {attempt}, force {force:.1}"));
        let fingers = ["thumb", "index", "middle"]
            .map(|f| Command::new(&format!("finger:{f}"), json!({"curl": 70})));
        m.together(fingers).await?;
        match m.command("grip", json!({"force": force})).await {
            Ok(felt) => {
                let pressure = felt["pressure"].as_f64().unwrap_or(0.0);
                if pressure >= 0.6 {
                    log(2, "grasp: holding it");
                    return Ok(felt);
                }
                log(2, "grasp: too light, squeezing harder");
                force += 0.3;
            }
            Err(e) => log(2, format!("grasp: {e}, again")),
        }
    }
    Err(MotionError::failed("could not get a grip"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ms = Duration::from_millis;
    let mut b = Fabric::builder();
    let (thermometer, temperature) = ChannelIn::new("thermometer", 8);
    let (eyes, sight) = ChannelIn::new("eyes", 64);
    let thermometer = b.receptor(thermometer);
    b.receptor(eyes);

    // The brain, behind deliberative pathways: from the thermometer, and from get_dressed,
    // whose questions and report travel on it
    let (done, finished) = std::sync::mpsc::channel();
    let brain = b.neuron(
        FnNeuron::reflex("brain", move |signal, cx| brain(signal, cx, &done))
            .with_kind(NeuronKind::Brain),
    );
    b.deliberative(thermometer, brain);
    let dress = b.motor("get_dressed", get_dressed);
    b.deliberative(dress, brain);

    // What the brain may call, and nothing below it
    b.action(
        dress,
        "Put on something warm, finding it first.",
        json!({"warmth": "0 to 1"}),
    );
    let stop = b.motor("stop", |m| async move {
        Ok(json!({"stopped": m.stop("get_dressed")}))
    });
    b.action(stop, "Stop getting dressed.", Value::Null);

    // The body: motor neurons for its parts, effectors for its joints and grip
    b.motor("reach", reach);
    b.motor("grasp", grasp);
    for finger in ["thumb", "index", "middle"] {
        b.motor(&format!("finger:{finger}"), move |m| async move {
            let curl = m.args()["curl"].clone();
            m.command(&format!("knuckle:{finger}"), json!({"angle": curl}))
                .await
        });
        b.effector(Joint(format!("knuckle:{finger}")));
    }
    for side in ["left", "right"] {
        b.motor(&format!("sleeve:{side}"), move |m| async move {
            log(2, format!("sleeve:{side}: arm in"));
            m.command(&format!("shoulder:{side}"), json!({"angle": 120}))
                .await?;
            m.command(&format!("elbow:{side}"), json!({"angle": 30}))
                .await
        });
        b.effector(Joint(format!("shoulder:{side}")));
        b.effector(Joint(format!("elbow:{side}")));
    }
    b.effector(Grip::default());

    let fabric = b.build()?;
    fabric.start();

    // The eyes stream what they see, every 50 ms
    std::thread::spawn(move || {
        loop {
            for seen in ["a scarf on the chair", "a jacket on the hook"] {
                let signal = sight.signal(Feature::sketch(seen.as_bytes()), Payload::text(seen));
                if sight.emit(signal).is_none() {
                    return;
                }
            }
            std::thread::sleep(ms(50));
        }
    });

    log(0, "thermometer: 4 °C");
    temperature.emit(temperature.signal(Feature::NONE, Payload::text("4 °C")));
    finished.recv_timeout(Duration::from_secs(10))?;

    println!();
    for neuron in fabric.metrics().neurons {
        if neuron.kind == NeuronKind::Motor {
            println!("{:>14}: {}", neuron.name, neuron.extra);
        }
    }
    println!("Reached the brain: {}", fabric.metrics().reached_brain);
    Ok(())
}
