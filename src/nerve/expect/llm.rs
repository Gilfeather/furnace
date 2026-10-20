//! The LLM adapter for expectations (feature `llm`): an [`Llm`] brain that knows the
//! expectation language, sees each surprise with its context and the rules already installed
//! for it, and answers with commands and `expect` commands.
//!
//! Nothing here is new machinery: the system prompt teaches the language, the user message
//! renders the surprise, and the answer is parsed as commands as for any [`Llm`] brain, so an
//! `{"effector": "expect", "args": {...}}` reaches the `expect` effector like any command, with
//! its scope limits and budgets. Feedback goes the same way: a shadow check asks whether an
//! absorbed signal should have been absorbed, a violation whether it was a false alarm, and the
//! prompt says which `verdict` answers that.
//!
//! ```no_run
//! # fn engine() -> std::sync::Arc<dyn furnace::llm::LlmEngine> { unimplemented!() }
//! use furnace::nerve::expect::llm::brain;
//! use furnace::nerve::{Fabric, Mechanisms};
//!
//! let mut b = Fabric::builder();
//! let brain = b.neuron(brain("brain", engine()), Mechanisms::NONE);
//! # let _ = brain;
//! ```

use std::fmt::Write as _;
use std::sync::Arc;

use serde_json::Value;

use crate::llm::LlmEngine;
use crate::nerve::brain::{Llm, signal_text};
use crate::nerve::signal::Signal;

/// The system prompt of an expectation-writing brain; `{effectors}` is replaced by the
/// effectors' names
pub const SYSTEM_PROMPT: &str = r#"You are the brain of a nervous system. You do not see the whole stream: rules you wrote earlier, called expectations, run next to the sensors and absorb what fits them. You only see surprises: a signal no rule covers (unexpected), a signal that broke a rule (violation), something a rule expected that did not come (absence), and now and then a signal a rule absorbed, so you can check it (shadow).

You act through these effectors: {effectors}. Answer with JSON only: one command object {"effector": "<name>", "args": {...}}, a JSON array of them, or [] when nothing is needed.

Write expectations so that what is normal stops reaching you. Command the effector "expect" with {"add": <expectation>} (or a list), {"replace": <expectation>}, {"retire": "<id>"} or {"verdict": ...}. An expectation is a JSON object:
- "id": a short name of your own
- "kind" and its parameters, one of:
  - "match": a matching signal arrives (absorb it, or act on it)
  - "rate" with "window_ms", "min", "max": matching signals per window stay within bounds
  - "every" with "period_ms", "tolerance_ms": one matching signal at least every period
  - "range" with "field", "min", "max": a numeric field stays within bounds
  - "then" with "then" (a selector) and "within_ms": a match is followed by another within the time
  - "never": no matching signal at all
- "selector": {"template": "<the template of the surprise>", "fields": {"<field>": "<value>"}}. Copy the template exactly as given. An absorbing expectation must name the template of the surprise it answers, or be narrower.
- optional "on_match": {"act": [commands]} to act without being asked; "absorb" by default
- optional "on_violation": {"act": [commands]} to act at once and still be told; "escalate" by default
- optional "learn": {"<parameter>": [low, high]} to let the sensors tune a parameter within bounds
- optional "provenance": why you wrote it

Only write an expectation for what you are sure is normal. Never absorb errors, failures or security events."#;

/// A brain over `engine` that writes expectations: [`SYSTEM_PROMPT`] and [`surprise_prompt`],
/// greedy, at most 512 tokens per answer
pub fn brain(name: &str, engine: Arc<dyn LlmEngine>) -> Llm {
    Llm::new(name, engine)
        .system(SYSTEM_PROMPT)
        .user(surprise_prompt)
        .params(crate::llm::SamplingParams::greedy().with_max_tokens(512))
}

/// The user message for a signal: a surprise rendered with its context and the expectations
/// installed for its template, and what kind of answer it wants; any other signal as its text
pub fn surprise_prompt(signal: &Signal) -> String {
    let Some(surprise) = signal
        .payload
        .to_json()
        .and_then(|v| v.get("surprise").cloned())
    else {
        return signal_text(signal);
    };
    let reason = surprise["reason"].as_str().unwrap_or("unexpected");
    let observation = &surprise["observation"];
    let template = observation["template"].as_str();
    let mut out = String::new();
    let _ = writeln!(out, "Surprise: {reason}");
    if let Some(text) = observation["text"].as_str() {
        let _ = writeln!(out, "Signal: {text}");
    }
    if let Some(template) = template {
        let _ = writeln!(out, "Template: {template}");
    }
    if let Some(fields) = observation["fields"].as_object()
        && !fields.is_empty()
    {
        let fields: Vec<String> = fields
            .iter()
            .map(|(k, v)| format!("{k}={}", super::language::text(v)))
            .collect();
        let _ = writeln!(out, "Fields: {}", fields.join(" "));
    }
    for broken in surprise["broken"].as_array().into_iter().flatten() {
        let _ = writeln!(
            out,
            "Broken: \"{}\" ({}): observed {}, expected {}",
            broken["expectation"].as_str().unwrap_or_default(),
            broken["kind"].as_str().unwrap_or_default(),
            broken["observed"],
            broken["expected"],
        );
    }
    if let Some(context) = surprise["context"].as_array()
        && !context.is_empty()
    {
        let _ = writeln!(out, "Recent signals of this template, oldest first:");
        for line in context.iter().filter_map(Value::as_str) {
            let _ = writeln!(out, "  {line}");
        }
    }
    let installed: Vec<&Value> = surprise["expectations"]
        .as_array()
        .into_iter()
        .flatten()
        .collect();
    if installed.is_empty() {
        let _ = writeln!(out, "No expectation covers this template yet.");
    } else {
        let _ = writeln!(out, "Expectations for this template now:");
        for e in installed {
            let _ = writeln!(out, "  {e}");
        }
    }
    let ask = match reason {
        "shadow" => {
            let id = surprise["absorbed_by"].as_str().unwrap_or_default();
            format!(
                "Expectation \"{id}\" absorbed this signal. If it should have reached you, answer \
                 {{\"effector\": \"expect\", \"args\": {{\"verdict\": {{\"id\": \"{id}\", \"correct\": false}}}}}}; \
                 otherwise answer []."
            )
        }
        "violation" | "absence" => {
            let id = surprise["broken"][0]["expectation"]
                .as_str()
                .unwrap_or_default();
            format!(
                "Act if this needs action. If it is normal after all, answer \
                 {{\"effector\": \"expect\", \"args\": {{\"verdict\": {{\"id\": \"{id}\", \"normal\": true}}}}}} \
                 (and replace the expectation if its structure is wrong)."
            )
        }
        _ => "Act if this needs action. If signals like this are normal, add an expectation for \
              this template so they stop reaching you."
            .to_string(),
    };
    let _ = writeln!(out, "{ask}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use crate::llm::{EngineError, Event, FinishReason, GenerateRequest, Prompt};
    use crate::nerve::effector::Effector;
    use crate::nerve::expect::{ExpectConfig, Expectations};
    use crate::nerve::{Feature, Firing, FnNeuron, Mechanisms, Payload, Summation};

    /// An engine that answers each prompt with what `reply` makes of it, and keeps the prompts
    #[derive(Debug)]
    struct Scripted {
        prompts: Mutex<Vec<String>>,
        calls: AtomicUsize,
    }

    impl Scripted {
        fn reply(prompt: &str) -> String {
            let template = prompt
                .lines()
                .find_map(|l| l.strip_prefix("Template: "))
                .unwrap_or_default();
            if prompt.starts_with("Surprise: unexpected") && template.contains("INFO") {
                serde_json::json!([{"effector": "expect", "args": {"add": {
                    "id": "routine", "kind": "match",
                    "selector": {"template": template},
                    "provenance": "routine INFO line"
                }}}])
                .to_string()
            } else {
                "[]".to_string()
            }
        }
    }

    impl LlmEngine for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }

        fn max_context(&self) -> usize {
            1 << 20
        }

        fn generate(
            &self,
            request: GenerateRequest,
        ) -> Result<tokio::sync::mpsc::Receiver<Event>, EngineError> {
            let Prompt::Chat(messages) = request.input else {
                panic!("a chat");
            };
            let user = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            assert!(
                messages[0].content.contains("\"never\""),
                "the language is taught"
            );
            let reply = Self::reply(&user);
            self.prompts.lock().unwrap().push(user);
            self.calls.fetch_add(1, Ordering::SeqCst);
            let (events, receiver) = tokio::sync::mpsc::channel(4);
            tokio::spawn(async move {
                let _ = events.send(Event::Started { prompt_tokens: 1 }).await;
                let _ = events.send(Event::Text(reply)).await;
                let _ = events
                    .send(Event::Finished {
                        reason: FinishReason::Stop,
                        completion_tokens: 1,
                    })
                    .await;
            });
            Ok(receiver)
        }
    }

    #[test]
    fn an_llm_brain_writes_an_expectation_and_stops_being_asked() {
        let engine = Arc::new(Scripted {
            prompts: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        });
        let expectations =
            Expectations::new("logs-expect", ExpectConfig::new().shadow(0.0).young(0));
        let mut b = crate::nerve::Fabric::builder();
        let logs = b.neuron(
            FnNeuron::reflex("logs", |s, _| Firing::one(s)),
            Mechanisms::NONE,
        );
        let expect = b.receptor(expectations.clone(), Mechanisms::NONE);
        b.reflex(logs, expect, 1.0);
        b.neuron(
            Effector::new(expectations.effector("expect")),
            Mechanisms::NONE,
        );
        let llm = b.neuron(brain("brain", engine.clone()), Mechanisms::NONE);
        b.deliberative(expect, llm, 1.0, Summation::none());
        let fabric = b.build();
        let line = |text: &str| {
            fabric.stimulate(logs, Signal::new(Feature::NONE, Payload::text(text)));
        };
        let wait = |done: &dyn Fn() -> bool| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !done() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert!(done());
        };
        line("2026-10-21 12:00:01 INFO request 17 served in 12 ms");
        wait(&|| expectations.ids() == ["routine"]);
        for i in 0..20 {
            line(&format!(
                "2026-10-21 12:00:{i:02} INFO request {i} served in {i} ms"
            ));
        }
        line("2026-10-21 12:01:00 ERROR disk 3 failed");
        wait(&|| engine.calls.load(Ordering::SeqCst) == 2);
        let prompts = engine.prompts.lock().unwrap();
        assert!(
            prompts[0].contains("Template: <*> <*> INFO request <*> served in <*> ms"),
            "{}",
            prompts[0]
        );
        assert!(prompts[0].contains("No expectation covers this template yet."));
        assert!(prompts[1].starts_with("Surprise: unexpected"));
        assert!(prompts[1].contains("ERROR disk"));
        let m = expectations.snapshot();
        assert_eq!((m.absorbed, m.unexpected), (20, 2));
    }

    #[test]
    fn the_prompt_asks_for_the_verdict_a_surprise_needs() {
        let shadow = Signal::new(
            Feature::NONE,
            Payload::json(serde_json::json!({"surprise": {
                "reason": "shadow", "absorbed_by": "quiet-disk", "at": "x",
                "observation": {"template": "disk <*> ok", "fields": {"v0": "3"}, "text": "disk 3 ok"},
                "context": ["disk 1 ok", "disk 2 ok"],
                "expectations": [{"id": "quiet-disk", "kind": "match"}]
            }})),
        );
        let prompt = surprise_prompt(&shadow);
        assert!(prompt.contains("Fields: v0=3"));
        assert!(prompt.contains("  disk 2 ok"));
        assert!(prompt.contains(r#""verdict": {"id": "quiet-disk", "correct": false}"#));
        let absence = Signal::new(
            Feature::NONE,
            Payload::json(serde_json::json!({"surprise": {
                "reason": "absence", "at": "x", "context": [], "expectations": [],
                "broken": [{"expectation": "hb", "kind": "every",
                    "observed": {"silent_ms": 9000}, "expected": {"period_ms": 5000}}]
            }})),
        );
        let prompt = surprise_prompt(&absence);
        assert!(prompt.contains(r#"Broken: "hb" (every): observed {"silent_ms":9000}"#));
        assert!(prompt.contains(r#""verdict": {"id": "hb", "normal": true}"#));
        // Not a surprise: its text
        assert_eq!(
            surprise_prompt(&Signal::new(Feature::NONE, Payload::text("hello"))),
            "hello"
        );
    }
}
