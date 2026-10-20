//! A line stream, from stdin, a TCP connection or a TCP port

use std::io::BufRead;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, warn};

use super::{Afferent, Backoff, RateMeter, Receptor, Source, UrgencyRule, kinds};
use crate::nerve::neuron::{Cx, Firing, Neuron, NeuronKind};
use crate::nerve::signal::{Feature, Payload, Signal};

enum LineSource {
    Stdin,
    Connect(String),
    Listen(String),
}

/// A stream of text lines: one signal per line, the feature a sketch of the line, the
/// payload the line as text. From stdin, from a TCP connection to a host, or from every
/// connection to a TCP port this receptor listens on. Reading runs on threads of its own.
///
/// With the fabric, the listener closes its port and reconnecting stops. A read in progress
/// (stdin, or a connection whose peer sends nothing) ends with its stream: its next line, or
/// the stream's end, finds the fabric gone.
pub struct Lines {
    source: Source,
    from: LineSource,
}

impl Lines {
    /// Read lines from the process's stdin
    pub fn stdin(name: &str) -> Self {
        Self {
            source: Source::new(name),
            from: LineSource::Stdin,
        }
    }

    /// Connect to `addr` (`host:port`) and read lines, reconnecting with backoff when it drops
    pub fn connect(name: &str, addr: &str) -> Self {
        Self {
            source: Source::new(name),
            from: LineSource::Connect(addr.to_string()),
        }
    }

    /// Listen on `addr` and read lines from every connection
    pub fn listen(name: &str, addr: &str) -> Self {
        Self {
            source: Source::new(name),
            from: LineSource::Listen(addr.to_string()),
        }
    }

    /// How each line's urgency is stamped
    pub fn urgency(mut self, rule: UrgencyRule) -> Self {
        self.source.rule = rule;
        self
    }

    /// Make the signal for one line
    pub(crate) fn line_signal(rule: UrgencyRule, meter: &RateMeter, line: &str) -> Signal {
        let now = Instant::now();
        let mut signal = Signal::new(
            Feature::sketch_as(kinds::TEXT, line.as_bytes()),
            Payload::text(line),
        );
        signal.at = now;
        signal.urgency = rule.stamp(meter.tick(now), now);
        signal
    }
}

/// Read lines from `reader` into the fabric until it ends or the fabric is gone; `true` if
/// the reader ended
fn pump_lines(
    reader: impl std::io::Read,
    afferent: &Afferent,
    rule: UrgencyRule,
    meter: &RateMeter,
) -> bool {
    let reader = std::io::BufReader::new(reader);
    for line in reader.lines() {
        let Ok(line) = line else {
            return true;
        };
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if afferent
            .emit(Lines::line_signal(rule, meter, line))
            .is_none()
        {
            return false;
        }
    }
    true
}

impl Neuron for Lines {
    fn name(&self) -> &str {
        &self.source.name
    }

    fn kind(&self) -> NeuronKind {
        NeuronKind::Receptor
    }

    fn stimulate(&self, signal: Signal, _cx: &Cx<'_>) -> Firing {
        Firing::one(signal)
    }

    fn describe(&self) -> serde_json::Value {
        let source = match &self.from {
            LineSource::Stdin => "stdin".to_string(),
            LineSource::Connect(addr) => format!("tcp connect {addr}"),
            LineSource::Listen(addr) => format!("tcp listen {addr}"),
        };
        self.source.describe(serde_json::json!({"source": source}))
    }
}

impl Receptor for Lines {
    fn start(&self, afferent: Afferent) {
        let _ = self.source.afferent.set(afferent.clone());
        let rule = self.source.rule;
        let name = self.source.name.clone();
        let spawn = |f: Box<dyn FnOnce() + Send>| {
            std::thread::Builder::new()
                .name(format!("furnace-receptor-{name}"))
                .spawn(f)
                .expect("failed to spawn a receptor thread");
        };
        match &self.from {
            LineSource::Stdin => spawn(Box::new(move || {
                pump_lines(std::io::stdin().lock(), &afferent, rule, &RateMeter::new());
            })),
            LineSource::Connect(addr) => {
                let addr = addr.clone();
                spawn(Box::new(move || {
                    let meter = RateMeter::new();
                    let mut backoff = Backoff::new(Duration::from_millis(100));
                    loop {
                        match std::net::TcpStream::connect(&addr) {
                            Ok(stream) => {
                                backoff.reset();
                                if !pump_lines(stream, &afferent, rule, &meter) {
                                    return;
                                }
                            }
                            Err(e) => debug!("nerve: lines receptor: connect {addr}: {e}"),
                        }
                        if afferent.fabric().is_none() || !afferent.sleep(backoff.next()) {
                            return;
                        }
                    }
                }));
            }
            LineSource::Listen(addr) => {
                let listener = match std::net::TcpListener::bind(addr) {
                    Ok(listener) => listener,
                    Err(e) => {
                        warn!("nerve: lines receptor cannot listen on {addr}: {e}");
                        return;
                    }
                };
                // Accepting on the fabric's runtime, so that the port closes with the fabric
                let Some(runtime) = afferent.runtime() else {
                    return;
                };
                let listener = match listener.set_nonblocking(true).and_then(|()| {
                    let _entered = runtime.enter();
                    tokio::net::TcpListener::from_std(listener)
                }) {
                    Ok(listener) => listener,
                    Err(e) => {
                        warn!("nerve: lines receptor cannot listen on {addr}: {e}");
                        return;
                    }
                };
                runtime.spawn(async move {
                    let meter = Arc::new(RateMeter::new());
                    loop {
                        let accepted = tokio::select! {
                            accepted = listener.accept() => accepted,
                            () = afferent.stopped() => return,
                        };
                        let Ok(stream) = accepted.and_then(|(s, _)| s.into_std()) else {
                            continue;
                        };
                        let (afferent, meter) = (afferent.clone(), meter.clone());
                        std::thread::spawn(move || {
                            if stream.set_nonblocking(false).is_ok() {
                                pump_lines(stream, &afferent, rule, &meter);
                            }
                        });
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::testing::{fabric_with_receptor, wait_until};

    #[test]
    fn a_listening_receptor_closes_its_port_with_the_fabric() {
        let addr = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let (fabric, _r, seen) = fabric_with_receptor(Lines::listen("in", &addr.to_string()));
        fabric.start();
        let mut client = std::net::TcpStream::connect(addr).unwrap();
        std::io::Write::write_all(&mut client, b"hello\n").unwrap();
        wait_until(|| seen.lock().unwrap().len() == 1);
        drop(fabric);
        wait_until(|| std::net::TcpListener::bind(addr).is_ok());
    }

    #[test]
    fn lines_receptor_reads_a_tcp_stream() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            use std::io::Write;
            let (mut stream, _) = listener.accept().unwrap();
            stream.write_all(b"first line\r\n\nsecond line\n").unwrap();
        });
        let (_fabric, _r, seen) = fabric_with_receptor(Lines::connect("lines", &addr));
        wait_until(|| seen.lock().unwrap().len() >= 2);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].payload.as_text(), Some("first line"));
        assert_eq!(seen[1].payload.as_text(), Some("second line"));
        assert_eq!(seen[0].feature.kind, kinds::TEXT);
    }

    #[test]
    fn lines_receptor_listens() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let addr = format!("127.0.0.1:{port}");
        let (_fabric, _r, seen) = fabric_with_receptor(Lines::listen("lines", &addr));
        use std::io::Write;
        let mut stream = std::net::TcpStream::connect(&addr).unwrap();
        stream.write_all(b"hello\n").unwrap();
        wait_until(|| !seen.lock().unwrap().is_empty());
        assert_eq!(seen.lock().unwrap()[0].payload.as_text(), Some("hello"));
    }
}
