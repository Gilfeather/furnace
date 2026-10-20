//! What the fabric measures: per neuron fired and quiet (and why), per pathway latency,
//! `brain_fraction`, stimulus-to-action per reflex arc. Rendered as JSON for `/metrics` and
//! `/nerve`, and in the Prometheus text format.

use std::fmt::Write as _;
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use super::fabric::{Fabric, PathwayKind};
use super::latency::LatencySnapshot;
use super::neuron::NeuronKind;
use super::signal::NeuronId;
use crate::telemetry::{label, render_value};

/// The fabric's metrics (the `nerve` part of `GET /metrics`)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NerveMetrics {
    /// Signals that entered through receptors
    pub received: u64,
    /// Signals a brain neuron saw
    pub reached_brain: u64,
    /// `reached_brain / received`
    pub brain_fraction: f64,
    /// Commands addressed to a name no effector has
    pub unrouted: u64,
    /// Signals with no pathway fast enough for their deadline
    pub no_eligible_pathway: u64,
    /// Signals waiting in deliberative queues now
    pub waiting: u64,
    pub neurons: Vec<NeuronMetrics>,
    pub pathways: Vec<PathwayMetrics>,
    /// Stimulus-to-action latency of each reflex arc (receptor → effector, never queued)
    pub arcs: Vec<ArcMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronMetrics {
    pub name: String,
    pub kind: NeuronKind,
    pub received: u64,
    pub fired: u64,
    pub quiet_by_threshold: u64,
    pub quiet_by_habituation: u64,
    pub quiet_by_refractory: u64,
    pub quiet: u64,
    /// Blocking computations run (`Firing::Compute`)
    pub computed: u64,
    /// Whatever the neuron adds (an effector's local loop, a brain's model)
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub extra: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathwayMetrics {
    pub from: String,
    pub to: String,
    pub kind: PathwayKind,
    pub weight: f32,
    pub signals: u64,
    pub ineligible: u64,
    pub overflow: u64,
    pub latency: LatencySnapshot,
    /// The summation window a deliberative pathway waits now
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_us: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArcMetrics {
    /// The receptor
    pub from: String,
    /// The effector
    pub to: String,
    pub latency: LatencySnapshot,
}

/// `GET /nerve`: the neurons, their mechanisms, the pathways and the measured latencies
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Description {
    pub neurons: Vec<NeuronDescription>,
    pub pathways: Vec<PathwayMetrics>,
    pub arcs: Vec<ArcMetrics>,
    pub brain_fraction: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronDescription {
    pub id: NeuronId,
    pub name: String,
    pub kind: NeuronKind,
    pub mechanisms: serde_json::Value,
    /// Ids of the pathways leaving it
    pub out: Vec<usize>,
    pub counters: NeuronMetrics,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub describe: serde_json::Value,
}

impl Fabric {
    pub fn metrics(&self) -> NerveMetrics {
        let inner = &self.0;
        NerveMetrics {
            received: inner.received.load(Ordering::Relaxed),
            reached_brain: inner.reached_brain.load(Ordering::Relaxed),
            brain_fraction: self.brain_fraction(),
            unrouted: inner.unrouted.load(Ordering::Relaxed),
            no_eligible_pathway: inner.no_eligible.load(Ordering::Relaxed),
            waiting: self.waiting(),
            neurons: (0..inner.cells.len())
                .map(|i| self.neuron_metrics(i))
                .collect(),
            pathways: self.pathway_metrics(),
            arcs: self.arc_metrics(),
        }
    }

    pub fn describe(&self) -> Description {
        let inner = &self.0;
        Description {
            neurons: inner
                .cells
                .iter()
                .enumerate()
                .map(|(i, cell)| NeuronDescription {
                    id: NeuronId(i as u32),
                    name: cell.neuron.name().to_string(),
                    kind: cell.kind,
                    mechanisms: cell.gate.describe(),
                    out: cell.out.iter().map(|p| p.index()).collect(),
                    counters: self.neuron_metrics(i),
                    describe: cell.neuron.describe(),
                })
                .collect(),
            pathways: self.pathway_metrics(),
            arcs: self.arc_metrics(),
            brain_fraction: self.brain_fraction(),
        }
    }

    fn neuron_metrics(&self, index: usize) -> NeuronMetrics {
        let cell = &self.0.cells[index];
        let c = &cell.counters;
        let load = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        NeuronMetrics {
            name: cell.neuron.name().to_string(),
            kind: cell.kind,
            received: load(&c.received),
            fired: load(&c.fired),
            quiet_by_threshold: load(&c.quiet_threshold),
            quiet_by_habituation: load(&c.quiet_habituation),
            quiet_by_refractory: load(&c.quiet_refractory),
            quiet: load(&c.quiet_not_interested),
            computed: load(&c.computed),
            extra: cell.neuron.metrics(),
        }
    }

    fn pathway_metrics(&self) -> Vec<PathwayMetrics> {
        self.0
            .pathways
            .iter()
            .map(|p| PathwayMetrics {
                from: self.name(p.from).to_string(),
                to: self.name(p.to).to_string(),
                kind: p.kind,
                weight: p.weight,
                signals: p.signals.load(Ordering::Relaxed),
                ineligible: p.ineligible.load(Ordering::Relaxed),
                overflow: p.overflow.load(Ordering::Relaxed),
                latency: p.latency.snapshot(),
                window_us: p.window().map(|w| w.as_secs_f64() * 1e6),
            })
            .collect()
    }

    fn arc_metrics(&self) -> Vec<ArcMetrics> {
        let mut arcs = Vec::new();
        for cell in &self.0.cells {
            for (origin, arc) in cell.arcs.iter().enumerate() {
                if let Some(latency) = arc.get() {
                    arcs.push(ArcMetrics {
                        from: self.0.cells[origin].neuron.name().to_string(),
                        to: cell.neuron.name().to_string(),
                        latency: latency.snapshot(),
                    });
                }
            }
        }
        arcs
    }

    /// Append the fabric's metrics in the Prometheus text format
    pub fn render_prometheus(&self, out: &mut String) {
        let m = self.metrics();
        render_fabric(&m, out);
        if m.neurons.iter().any(|n| n.kind == NeuronKind::Effector) {
            super::effector::render_prometheus_help(out);
        }
        for (i, cell) in self.0.cells.iter().enumerate() {
            cell.neuron
                .render_prometheus(out, &format!("neuron=\"{}\"", label(&m.neurons[i].name)));
        }
    }
}

/// The fabric's own families: its counters, the neurons', the pathways' and the arcs'
fn render_fabric(m: &NerveMetrics, out: &mut String) {
    let counter = "counter";
    let gauge = "gauge";
    render_value(
        out,
        "furnace_nerve_signals_received_total",
        counter,
        "Signals that entered through receptors",
        m.received,
    );
    render_value(
        out,
        "furnace_nerve_signals_brain_total",
        counter,
        "Signals a brain neuron saw",
        m.reached_brain,
    );
    render_value(
        out,
        "furnace_nerve_brain_fraction",
        gauge,
        "Signals that reached a brain over signals received",
        m.brain_fraction,
    );
    render_value(
        out,
        "furnace_nerve_commands_unrouted_total",
        counter,
        "Commands addressed to a name no effector has",
        m.unrouted,
    );
    render_value(
        out,
        "furnace_nerve_signals_no_pathway_total",
        counter,
        "Signals with no pathway fast enough for their deadline",
        m.no_eligible_pathway,
    );
    render_value(
        out,
        "furnace_nerve_signals_waiting",
        gauge,
        "Signals waiting in deliberative queues",
        m.waiting,
    );

    let _ = writeln!(
        out,
        "# HELP furnace_nerve_neuron_received_total Signals that reached the neuron's gate\n\
             # TYPE furnace_nerve_neuron_received_total counter"
    );
    for n in &m.neurons {
        let _ = writeln!(
            out,
            "furnace_nerve_neuron_received_total{{neuron=\"{}\",kind=\"{}\"}} {}",
            label(&n.name),
            n.kind.as_str(),
            n.received
        );
    }
    let _ = writeln!(
        out,
        "# HELP furnace_nerve_neuron_fired_total Times the neuron fired\n\
             # TYPE furnace_nerve_neuron_fired_total counter"
    );
    for n in &m.neurons {
        let _ = writeln!(
            out,
            "furnace_nerve_neuron_fired_total{{neuron=\"{}\",kind=\"{}\"}} {}",
            label(&n.name),
            n.kind.as_str(),
            n.fired
        );
    }
    let _ = writeln!(
        out,
        "# HELP furnace_nerve_neuron_quiet_total Signals that died at the neuron, by reason\n\
             # TYPE furnace_nerve_neuron_quiet_total counter"
    );
    for n in &m.neurons {
        for (reason, value) in [
            ("threshold", n.quiet_by_threshold),
            ("habituation", n.quiet_by_habituation),
            ("refractory", n.quiet_by_refractory),
            ("not_interested", n.quiet),
        ] {
            let _ = writeln!(
                out,
                "furnace_nerve_neuron_quiet_total{{neuron=\"{}\",reason=\"{reason}\"}} {value}",
                label(&n.name)
            );
        }
    }

    let _ = writeln!(
        out,
        "# HELP furnace_nerve_pathway_signals_total Signals that took the pathway\n\
             # TYPE furnace_nerve_pathway_signals_total counter"
    );
    for p in &m.pathways {
        let _ = writeln!(
            out,
            "furnace_nerve_pathway_signals_total{{from=\"{}\",to=\"{}\",kind=\"{:?}\"}} {}",
            label(&p.from),
            label(&p.to),
            p.kind,
            p.signals
        );
    }
    let _ = writeln!(
        out,
        "# HELP furnace_nerve_pathway_latency_seconds Measured signal-in to signal-out on the pathway\n\
             # TYPE furnace_nerve_pathway_latency_seconds gauge"
    );
    for p in &m.pathways {
        render_quantiles(
            out,
            "furnace_nerve_pathway_latency_seconds",
            &p.from,
            &p.to,
            &p.latency,
        );
    }
    let _ = writeln!(
        out,
        "# HELP furnace_nerve_arc_latency_seconds Stimulus to effector call on each reflex arc\n\
             # TYPE furnace_nerve_arc_latency_seconds gauge"
    );
    for a in &m.arcs {
        render_quantiles(
            out,
            "furnace_nerve_arc_latency_seconds",
            &a.from,
            &a.to,
            &a.latency,
        );
    }
}

fn render_quantiles(out: &mut String, name: &str, from: &str, to: &str, l: &LatencySnapshot) {
    for (q, value) in [("0.5", l.p50_us), ("0.99", l.p99_us)] {
        if let Some(us) = value {
            let _ = writeln!(
                out,
                "{name}{{from=\"{}\",to=\"{}\",quantile=\"{q}\"}} {}",
                label(from),
                label(to),
                us / 1e6
            );
        }
    }
    let _ = writeln!(
        out,
        "{name}_count{{from=\"{}\",to=\"{}\"}} {}",
        label(from),
        label(to),
        l.count
    );
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::nerve::effector::{Channel, Effector};
    use crate::nerve::fabric::{FnNeuron, Summation};
    use crate::nerve::neuron::{Firing, Mechanisms};

    fn neuron(name: &str, kind: NeuronKind, base: u64) -> NeuronMetrics {
        NeuronMetrics {
            name: name.to_string(),
            kind,
            received: base,
            fired: base + 1,
            quiet_by_threshold: base + 2,
            quiet_by_habituation: base + 3,
            quiet_by_refractory: base + 4,
            quiet: base + 5,
            computed: 0,
            extra: serde_json::Value::Null,
        }
    }

    fn latency(count: u64, p50_us: Option<f64>, p99_us: Option<f64>) -> LatencySnapshot {
        LatencySnapshot {
            count,
            p50_us,
            p99_us,
            mean_us: None,
            max_us: None,
        }
    }

    /// Every family the fabric writes, with label values that need escaping and latencies
    /// with and without quantiles
    #[test]
    fn the_fabric_families_render_as_before() {
        let metrics = NerveMetrics {
            received: 10,
            reached_brain: 2,
            brain_fraction: 0.2,
            unrouted: 1,
            no_eligible_pathway: 3,
            waiting: 4,
            neurons: vec![
                neuron("in", NeuronKind::Receptor, 10),
                neuron("cl\"assify\\", NeuronKind::Reflex, 20),
                neuron("brain", NeuronKind::Brain, 30),
            ],
            pathways: vec![
                PathwayMetrics {
                    from: "in".into(),
                    to: "cl\"assify\\".into(),
                    kind: PathwayKind::Reflex,
                    weight: 1.0,
                    signals: 7,
                    ineligible: 0,
                    overflow: 0,
                    latency: latency(7, Some(12.5), Some(40.0)),
                    window_us: None,
                },
                PathwayMetrics {
                    from: "cl\"assify\\".into(),
                    to: "brain".into(),
                    kind: PathwayKind::Deliberative,
                    weight: 0.5,
                    signals: 2,
                    ineligible: 1,
                    overflow: 0,
                    latency: latency(0, None, None),
                    window_us: Some(250.0),
                },
            ],
            arcs: vec![ArcMetrics {
                from: "in".into(),
                to: "brain".into(),
                latency: latency(3, Some(1500.0), None),
            }],
        };
        let mut out = String::new();
        render_fabric(&metrics, &mut out);
        assert_eq!(out, FABRIC_FAMILIES, "\n{out}");
    }

    /// A built fabric with an effector: the effector families after the fabric's
    #[test]
    fn a_fabric_with_an_effector_renders_as_before() {
        let mut b = Fabric::builder();
        let input = b.neuron(
            FnNeuron::reflex("in", |s, _| Firing::one(s)).with_kind(NeuronKind::Receptor),
            Mechanisms::NONE,
        );
        let brain = b.neuron(
            FnNeuron::reflex("brain", |s, _| Firing::one(s)).with_kind(NeuronKind::Brain),
            Mechanisms::NONE,
        );
        let (channel, _rx) = Channel::new("out", 1);
        let out = b.neuron(Effector::new(channel), Mechanisms::NONE);
        b.reflex(input, out, 1.0).deliberative(
            input,
            brain,
            0.5,
            Summation {
                window: Some(Duration::from_millis(1)),
                ..Summation::default()
            },
        );
        let fabric = b.build();
        let mut text = String::new();
        fabric.render_prometheus(&mut text);
        assert_eq!(text, WITH_EFFECTOR, "\n{text}");
    }

    const FABRIC_FAMILIES: &str = r#"# HELP furnace_nerve_signals_received_total Signals that entered through receptors
# TYPE furnace_nerve_signals_received_total counter
furnace_nerve_signals_received_total 10
# HELP furnace_nerve_signals_brain_total Signals a brain neuron saw
# TYPE furnace_nerve_signals_brain_total counter
furnace_nerve_signals_brain_total 2
# HELP furnace_nerve_brain_fraction Signals that reached a brain over signals received
# TYPE furnace_nerve_brain_fraction gauge
furnace_nerve_brain_fraction 0.2
# HELP furnace_nerve_commands_unrouted_total Commands addressed to a name no effector has
# TYPE furnace_nerve_commands_unrouted_total counter
furnace_nerve_commands_unrouted_total 1
# HELP furnace_nerve_signals_no_pathway_total Signals with no pathway fast enough for their deadline
# TYPE furnace_nerve_signals_no_pathway_total counter
furnace_nerve_signals_no_pathway_total 3
# HELP furnace_nerve_signals_waiting Signals waiting in deliberative queues
# TYPE furnace_nerve_signals_waiting gauge
furnace_nerve_signals_waiting 4
# HELP furnace_nerve_neuron_received_total Signals that reached the neuron's gate
# TYPE furnace_nerve_neuron_received_total counter
furnace_nerve_neuron_received_total{neuron="in",kind="receptor"} 10
furnace_nerve_neuron_received_total{neuron="cl\"assify\\",kind="reflex"} 20
furnace_nerve_neuron_received_total{neuron="brain",kind="brain"} 30
# HELP furnace_nerve_neuron_fired_total Times the neuron fired
# TYPE furnace_nerve_neuron_fired_total counter
furnace_nerve_neuron_fired_total{neuron="in",kind="receptor"} 11
furnace_nerve_neuron_fired_total{neuron="cl\"assify\\",kind="reflex"} 21
furnace_nerve_neuron_fired_total{neuron="brain",kind="brain"} 31
# HELP furnace_nerve_neuron_quiet_total Signals that died at the neuron, by reason
# TYPE furnace_nerve_neuron_quiet_total counter
furnace_nerve_neuron_quiet_total{neuron="in",reason="threshold"} 12
furnace_nerve_neuron_quiet_total{neuron="in",reason="habituation"} 13
furnace_nerve_neuron_quiet_total{neuron="in",reason="refractory"} 14
furnace_nerve_neuron_quiet_total{neuron="in",reason="not_interested"} 15
furnace_nerve_neuron_quiet_total{neuron="cl\"assify\\",reason="threshold"} 22
furnace_nerve_neuron_quiet_total{neuron="cl\"assify\\",reason="habituation"} 23
furnace_nerve_neuron_quiet_total{neuron="cl\"assify\\",reason="refractory"} 24
furnace_nerve_neuron_quiet_total{neuron="cl\"assify\\",reason="not_interested"} 25
furnace_nerve_neuron_quiet_total{neuron="brain",reason="threshold"} 32
furnace_nerve_neuron_quiet_total{neuron="brain",reason="habituation"} 33
furnace_nerve_neuron_quiet_total{neuron="brain",reason="refractory"} 34
furnace_nerve_neuron_quiet_total{neuron="brain",reason="not_interested"} 35
# HELP furnace_nerve_pathway_signals_total Signals that took the pathway
# TYPE furnace_nerve_pathway_signals_total counter
furnace_nerve_pathway_signals_total{from="in",to="cl\"assify\\",kind="Reflex"} 7
furnace_nerve_pathway_signals_total{from="cl\"assify\\",to="brain",kind="Deliberative"} 2
# HELP furnace_nerve_pathway_latency_seconds Measured signal-in to signal-out on the pathway
# TYPE furnace_nerve_pathway_latency_seconds gauge
furnace_nerve_pathway_latency_seconds{from="in",to="cl\"assify\\",quantile="0.5"} 0.0000125
furnace_nerve_pathway_latency_seconds{from="in",to="cl\"assify\\",quantile="0.99"} 0.00004
furnace_nerve_pathway_latency_seconds_count{from="in",to="cl\"assify\\"} 7
furnace_nerve_pathway_latency_seconds_count{from="cl\"assify\\",to="brain"} 0
# HELP furnace_nerve_arc_latency_seconds Stimulus to effector call on each reflex arc
# TYPE furnace_nerve_arc_latency_seconds gauge
furnace_nerve_arc_latency_seconds{from="in",to="brain",quantile="0.5"} 0.0015
furnace_nerve_arc_latency_seconds_count{from="in",to="brain"} 3
"#;

    const WITH_EFFECTOR: &str = r#"# HELP furnace_nerve_signals_received_total Signals that entered through receptors
# TYPE furnace_nerve_signals_received_total counter
furnace_nerve_signals_received_total 0
# HELP furnace_nerve_signals_brain_total Signals a brain neuron saw
# TYPE furnace_nerve_signals_brain_total counter
furnace_nerve_signals_brain_total 0
# HELP furnace_nerve_brain_fraction Signals that reached a brain over signals received
# TYPE furnace_nerve_brain_fraction gauge
furnace_nerve_brain_fraction 0
# HELP furnace_nerve_commands_unrouted_total Commands addressed to a name no effector has
# TYPE furnace_nerve_commands_unrouted_total counter
furnace_nerve_commands_unrouted_total 0
# HELP furnace_nerve_signals_no_pathway_total Signals with no pathway fast enough for their deadline
# TYPE furnace_nerve_signals_no_pathway_total counter
furnace_nerve_signals_no_pathway_total 0
# HELP furnace_nerve_signals_waiting Signals waiting in deliberative queues
# TYPE furnace_nerve_signals_waiting gauge
furnace_nerve_signals_waiting 0
# HELP furnace_nerve_neuron_received_total Signals that reached the neuron's gate
# TYPE furnace_nerve_neuron_received_total counter
furnace_nerve_neuron_received_total{neuron="in",kind="receptor"} 0
furnace_nerve_neuron_received_total{neuron="brain",kind="brain"} 0
furnace_nerve_neuron_received_total{neuron="out",kind="effector"} 0
# HELP furnace_nerve_neuron_fired_total Times the neuron fired
# TYPE furnace_nerve_neuron_fired_total counter
furnace_nerve_neuron_fired_total{neuron="in",kind="receptor"} 0
furnace_nerve_neuron_fired_total{neuron="brain",kind="brain"} 0
furnace_nerve_neuron_fired_total{neuron="out",kind="effector"} 0
# HELP furnace_nerve_neuron_quiet_total Signals that died at the neuron, by reason
# TYPE furnace_nerve_neuron_quiet_total counter
furnace_nerve_neuron_quiet_total{neuron="in",reason="threshold"} 0
furnace_nerve_neuron_quiet_total{neuron="in",reason="habituation"} 0
furnace_nerve_neuron_quiet_total{neuron="in",reason="refractory"} 0
furnace_nerve_neuron_quiet_total{neuron="in",reason="not_interested"} 0
furnace_nerve_neuron_quiet_total{neuron="brain",reason="threshold"} 0
furnace_nerve_neuron_quiet_total{neuron="brain",reason="habituation"} 0
furnace_nerve_neuron_quiet_total{neuron="brain",reason="refractory"} 0
furnace_nerve_neuron_quiet_total{neuron="brain",reason="not_interested"} 0
furnace_nerve_neuron_quiet_total{neuron="out",reason="threshold"} 0
furnace_nerve_neuron_quiet_total{neuron="out",reason="habituation"} 0
furnace_nerve_neuron_quiet_total{neuron="out",reason="refractory"} 0
furnace_nerve_neuron_quiet_total{neuron="out",reason="not_interested"} 0
# HELP furnace_nerve_pathway_signals_total Signals that took the pathway
# TYPE furnace_nerve_pathway_signals_total counter
furnace_nerve_pathway_signals_total{from="in",to="out",kind="Reflex"} 0
furnace_nerve_pathway_signals_total{from="in",to="brain",kind="Deliberative"} 0
# HELP furnace_nerve_pathway_latency_seconds Measured signal-in to signal-out on the pathway
# TYPE furnace_nerve_pathway_latency_seconds gauge
furnace_nerve_pathway_latency_seconds_count{from="in",to="out"} 0
furnace_nerve_pathway_latency_seconds_count{from="in",to="brain"} 0
# HELP furnace_nerve_arc_latency_seconds Stimulus to effector call on each reflex arc
# TYPE furnace_nerve_arc_latency_seconds gauge
# HELP furnace_nerve_effector_acted_total Commands the effector carried out
# TYPE furnace_nerve_effector_acted_total counter
# HELP furnace_nerve_effector_failed_total Commands that failed after their retries
# TYPE furnace_nerve_effector_failed_total counter
# HELP furnace_nerve_effector_retries_total Retries
# TYPE furnace_nerve_effector_retries_total counter
# HELP furnace_nerve_effector_dropped_open_total Commands dropped while the circuit was open
# TYPE furnace_nerve_effector_dropped_open_total counter
# HELP furnace_nerve_effector_dropped_rate_total Commands dropped by the rate limit
# TYPE furnace_nerve_effector_dropped_rate_total counter
# HELP furnace_nerve_effector_quiet_refractory_total Commands ignored in the refractory period
# TYPE furnace_nerve_effector_quiet_refractory_total counter
# HELP furnace_nerve_effector_inflight Commands in progress
# TYPE furnace_nerve_effector_inflight gauge
# HELP furnace_nerve_effector_circuit_open 1 while the effector's circuit breaker is open or half open
# TYPE furnace_nerve_effector_circuit_open gauge
furnace_nerve_effector_acted_total{neuron="out"} 0
furnace_nerve_effector_failed_total{neuron="out"} 0
furnace_nerve_effector_retries_total{neuron="out"} 0
furnace_nerve_effector_dropped_open_total{neuron="out"} 0
furnace_nerve_effector_dropped_rate_total{neuron="out"} 0
furnace_nerve_effector_quiet_refractory_total{neuron="out"} 0
furnace_nerve_effector_inflight{neuron="out"} 0
furnace_nerve_effector_circuit_open{neuron="out"} 0
"#;
}
