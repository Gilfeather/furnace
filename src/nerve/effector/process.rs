//! Run a local command

use std::time::Duration;

use bytes::Bytes;

use super::{Act, ActError, Action};
use crate::nerve::signal::{Command, Signal};

/// Run a local command. The program and its fixed arguments are set here; the command's
/// `args` adds `{"args": [...], "stdin": "..."}` (or is itself an array of arguments). The
/// process inherits stdout; its stderr is kept for the error message. A non-zero exit is a
/// final failure; a program that cannot be started is retried. Writing stdin and running
/// both count against the timeout, so a program that never reads its input is killed too.
///
/// The arguments, from a brain's command too, reach the program as they are: as argv, with
/// no shell in between, so they cannot inject a command, but the program sees whatever the
/// brain chose. Fix the arguments here and pass only data on stdin when that matters.
pub struct Process {
    name: String,
    program: String,
    base_args: Vec<String>,
    timeout: Duration,
    /// Feed the signal's payload on stdin when the command gives none
    stdin_payload: bool,
}

impl Process {
    /// Run `program`, with no fixed arguments and a 30 s timeout
    pub fn new(name: &str, program: &str) -> Self {
        Self {
            name: name.to_string(),
            program: program.to_string(),
            base_args: Vec::new(),
            timeout: Duration::from_secs(30),
            stdin_payload: false,
        }
    }

    /// Fixed arguments, before any the command adds
    pub fn args(mut self, args: &[&str]) -> Self {
        self.base_args = args.iter().map(|a| a.to_string()).collect();
        self
    }

    /// Bound on each run, stdin included; a run that outlasts it is killed and retried
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Give the process the signal's payload on stdin
    pub fn stdin_payload(mut self) -> Self {
        self.stdin_payload = true;
        self
    }
}

impl Act for Process {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let strings = |v: &serde_json::Value| -> Vec<String> {
            v.as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| match x {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut args = self.base_args.clone();
        let mut stdin: Option<Bytes> = None;
        match &command.args {
            serde_json::Value::Array(_) => args.extend(strings(&command.args)),
            serde_json::Value::Object(object) => {
                if let Some(extra) = object.get("args") {
                    args.extend(strings(extra));
                }
                if let Some(serde_json::Value::String(input)) = object.get("stdin") {
                    stdin = Some(Bytes::from(input.clone()));
                }
            }
            _ => {}
        }
        if stdin.is_none() && self.stdin_payload {
            stdin = Some(signal.payload.to_bytes());
        }
        let program = self.program.clone();
        let timeout = self.timeout;
        Action::pending(async move {
            use tokio::io::AsyncWriteExt;
            let mut child = tokio::process::Command::new(&program)
                .args(&args)
                .stdin(if stdin.is_some() {
                    std::process::Stdio::piped()
                } else {
                    std::process::Stdio::null()
                })
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .map_err(|e| ActError::retry(format!("{program}: {e}")))?;
            let pipe = child.stdin.take();
            let run = async move {
                if let (Some(input), Some(mut pipe)) = (stdin, pipe) {
                    // A program may exit without reading it all; its exit status says how it went
                    let _ = pipe.write_all(&input).await;
                }
                child.wait_with_output().await
            };
            match tokio::time::timeout(timeout, run).await {
                Ok(Ok(output)) if output.status.success() => Ok(()),
                Ok(Ok(output)) => Err(ActError::fatal(format!(
                    "{program} exited with {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr).trim()
                ))),
                Ok(Err(e)) => Err(ActError::retry(format!("{program}: {e}"))),
                Err(_) => Err(ActError::Timeout(program)),
            }
        })
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"process": self.program, "args": self.base_args})
    }
}

#[cfg(test)]
mod tests {
    use super::super::Effector;
    use super::*;
    use crate::nerve::testing::{command, fabric_with, wait_until};

    #[tokio::test]
    async fn process_runs_a_command_and_reports_failures() {
        let (program, ok_args, bad_args): (&str, &[&str], &[&str]) = if cfg!(windows) {
            ("cmd", &["/C", "exit", "0"], &["/C", "exit", "3"])
        } else {
            ("sh", &["-c", "exit 0"], &["-c", "exit 3"])
        };
        let effector = Effector::new(Process::new("cmd", program)).no_retry();
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(r, command("cmd", serde_json::json!({"args": ok_args})));
        fabric.stimulate(r, command("cmd", serde_json::json!(bad_args)));
        tokio::task::spawn_blocking({
            let effector = effector.clone();
            move || {
                wait_until(|| {
                    effector.metrics().inflight == 0
                        && effector.metrics().acted + effector.metrics().failed == 2
                })
            }
        })
        .await
        .unwrap();
        let m = effector.metrics();
        assert_eq!((m.acted, m.failed), (1, 1));
        drop(fabric);
    }

    /// Run `process` once for `command` and `signal`, as the effector's loop would
    async fn run(process: &Process, command: &Command, signal: &Signal) -> Result<(), ActError> {
        match process.act(command, signal) {
            Action::Done(result) => result,
            Action::Pending(future) => future.await,
        }
    }

    #[tokio::test]
    async fn stdin_payload_feeds_the_signal_and_the_command_s_stdin_wins() {
        // Succeeds only if its input contains `hello`
        let process = if cfg!(windows) {
            Process::new("check", "findstr").args(&["hello"])
        } else {
            Process::new("check", "grep").args(&["-q", "hello"])
        }
        .stdin_payload();
        let signal = |text: &str| {
            Signal::new(
                crate::nerve::signal::Feature::NONE,
                crate::nerve::signal::Payload::bytes(text.as_bytes().to_vec()),
            )
        };
        let plain = Command::new("check", serde_json::Value::Null);
        assert_eq!(run(&process, &plain, &signal("hello\n")).await, Ok(()));
        let err = run(&process, &plain, &signal("goodbye\n"))
            .await
            .unwrap_err();
        assert!(!err.retryable(), "{err}");
        // A command that gives its own stdin is fed that, not the payload
        let own = Command::new("check", serde_json::json!({"stdin": "hello\n"}));
        assert_eq!(run(&process, &own, &signal("goodbye\n")).await, Ok(()));
        // Without stdin_payload the process gets no input
        let plain_process = if cfg!(windows) {
            Process::new("check", "findstr").args(&["hello"])
        } else {
            Process::new("check", "grep").args(&["-q", "hello"])
        };
        assert!(
            run(&plain_process, &plain, &signal("hello\n"))
                .await
                .is_err()
        );
    }
}
