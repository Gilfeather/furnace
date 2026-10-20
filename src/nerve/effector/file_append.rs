//! Append to a file, on a thread of its own

use std::sync::{OnceLock, mpsc};

use super::{Act, ActError, Action};
use crate::nerve::signal::{Command, Signal};

/// Append a line per command to a file: `args.line` as it is if it is a string, else one
/// JSON object `{"at": ..., "signal": ..., "args": ...}`.
///
/// The writing happens on a thread of its own, started with the first command, so a slow disk
/// never holds up the thread that stimulated the fabric; the lines still go out in the order
/// the commands came. The file is opened on the first line and reopened after an error.
pub struct FileAppend {
    name: String,
    path: std::path::PathBuf,
    writer: OnceLock<mpsc::Sender<Line>>,
}

/// A line for the writer thread, and where to say how writing it went
type Line = (String, tokio::sync::oneshot::Sender<std::io::Result<()>>);

impl FileAppend {
    pub fn new(name: &str, path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            name: name.to_string(),
            path: path.into(),
            writer: OnceLock::new(),
        }
    }

    /// The writer thread's inbox, starting it on first use
    fn writer(&self) -> std::io::Result<&mpsc::Sender<Line>> {
        if let Some(writer) = self.writer.get() {
            return Ok(writer);
        }
        let (sender, lines) = mpsc::channel::<Line>();
        let path = self.path.clone();
        std::thread::Builder::new()
            .name(format!("furnace-file-{}", self.name))
            .spawn(move || write_lines(&path, &lines))?;
        // Two first commands at once start two threads; the loser's ends with its sender
        Ok(self.writer.get_or_init(|| sender))
    }
}

/// The writer thread: until the [`FileAppend`] is dropped
fn write_lines(path: &std::path::Path, lines: &mpsc::Receiver<Line>) {
    use std::io::Write as _;
    let mut file: Option<std::fs::File> = None;
    for (line, done) in lines {
        let result = (|| {
            let open = match &mut file {
                Some(open) => open,
                None => file.insert(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)?,
                ),
            };
            open.write_all(line.as_bytes())
        })();
        if result.is_err() {
            file = None;
        }
        // Nobody may be waiting any more (the fabric is gone)
        let _ = done.send(result);
    }
}

impl Act for FileAppend {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&self, command: &Command, signal: &Signal) -> Action {
        let mut line = match command.args.get("line").and_then(|l| l.as_str()) {
            Some(line) => line.to_string(),
            None => serde_json::json!({
                "at": chrono::Utc::now().to_rfc3339(),
                "signal": signal.id.0,
                "args": command.args,
            })
            .to_string(),
        };
        line.push('\n');
        let path = self.path.display().to_string();
        let (done, written) = tokio::sync::oneshot::channel();
        let sent = self
            .writer()
            .map_err(|e| format!("{path}: no writer thread: {e}"))
            .and_then(|writer| {
                writer
                    .send((line, done))
                    .map_err(|_| format!("{path}: the writer thread is gone"))
            });
        if let Err(message) = sent {
            return Action::Done(Err(ActError::retry(message)));
        }
        Action::pending(async move {
            match written.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(ActError::retry(format!("{path}: {e}"))),
                Err(_) => Err(ActError::retry(format!(
                    "{path}: the writer thread is gone"
                ))),
            }
        })
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"file": self.path.display().to_string()})
    }
}

#[cfg(test)]
mod tests {
    use super::super::Effector;
    use super::*;
    use crate::nerve::testing::{command, fabric_with};
    use std::time::{Duration, Instant};

    #[test]
    fn file_append_writes_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.log");
        let effector = Effector::new(FileAppend::new("journal", &path));
        let (fabric, r) = fabric_with(effector.clone());
        fabric.stimulate(
            r,
            command("journal", serde_json::json!({"line": "raw line"})),
        );
        fabric.stimulate(r, command("journal", serde_json::json!({"k": 1})));
        // The lines are written on the effector's own thread
        let deadline = Instant::now() + Duration::from_secs(5);
        while effector.metrics().acted < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "raw line");
        let json: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(json["args"]["k"], 1);
        assert_eq!(effector.metrics().acted, 2);
    }
}
