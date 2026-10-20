//! What the fabric measures: per neuron fired and quiet (and why), per pathway latency,
//! `brain_fraction`, stimulus-to-action per reflex arc. Rendered as JSON for `/metrics` and
//! `/nerve`, and in the Prometheus text format.

use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use super::fabric::{Fabric, PathwayKind};
use super::latency::LatencySnapshot;
use super::neuron::{NeuronKind, Quiet};
use super::signal::NeuronId;
use crate::telemetry::{label, render_family, render_labelled, render_value};

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
    /// Each neuron's counters, by id
    pub neurons: Vec<NeuronMetrics>,
    /// Each pathway's counters and latency, by id
    pub pathways: Vec<PathwayMetrics>,
    /// Stimulus-to-action latency of each reflex arc (receptor → effector, never queued)
    pub arcs: Vec<ArcMetrics>,
}

/// One neuron's counters (in `GET /metrics` and `GET /nerve`)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronMetrics {
    /// The neuron's name
    pub name: String,
    /// What kind of neuron
    pub kind: NeuronKind,
    /// Signals that reached the neuron's gate
    pub received: u64,
    /// Times the neuron fired
    pub fired: u64,
    /// Signals below the threshold as they arrived
    pub quiet_by_threshold: u64,
    /// Signals attenuated below the threshold by habituation
    pub quiet_by_habituation: u64,
    /// Signals ignored because the neuron fired too recently
    pub quiet_by_refractory: u64,
    /// Signals that passed the gate but the neuron was not interested in (`Firing::Quiet`),
    /// the fourth reason next to the `quiet_by_*` ones (`reason="not_interested"` in
    /// Prometheus)
    pub quiet_not_interested: u64,
    /// Firings that came from blocking work (`Firing::Compute`): every firing of a brain
    /// (`brain::Llm`, `brain::ModelNeuron`), one per signal of a summed window
    pub computed: u64,
    /// Whatever the neuron adds (an effector's local loop, a brain's model)
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub extra: serde_json::Value,
}

/// One pathway's counters and latency (in `GET /metrics` and `GET /nerve`)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathwayMetrics {
    /// The name of the neuron that fires into it
    pub from: String,
    /// The name of the neuron it stimulates
    pub to: String,
    /// Reflex or deliberative
    pub kind: PathwayKind,
    /// Priority among eligible pathways from the same neuron
    pub weight: f32,
    /// Signals that took the pathway
    pub signals: u64,
    /// Signals that could not take it: too slow for their deadline
    pub ineligible: u64,
    /// Signals dropped because the queue was full
    pub overflow: u64,
    /// Signal-in to signal-out on this hop, in milliseconds
    pub latency_ms: LatencySnapshot,
    /// The summation window a deliberative pathway waits now
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_us: Option<f64>,
}

/// Stimulus-to-action latency of one reflex arc
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArcMetrics {
    /// The receptor
    pub from: String,
    /// The effector
    pub to: String,
    /// Stimulus-to-action, from the receptor's stimulus to the effector, in milliseconds
    pub latency_ms: LatencySnapshot,
}

/// `GET /nerve`: the neurons, their mechanisms, the pathways and the measured latencies
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Description {
    /// Each neuron with its mechanisms and counters, by id
    pub neurons: Vec<NeuronDescription>,
    /// Each pathway's counters and latency, by id
    pub pathways: Vec<PathwayMetrics>,
    /// Stimulus-to-action latency of each reflex arc
    pub arcs: Vec<ArcMetrics>,
    /// Signals that reached a brain over signals received
    pub brain_fraction: f64,
}

/// One neuron in `GET /nerve`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronDescription {
    /// Its index in the fabric
    pub id: NeuronId,
    /// The neuron's name
    pub name: String,
    /// What kind of neuron
    pub kind: NeuronKind,
    /// Its threshold, habituation and refractory period
    pub mechanisms: serde_json::Value,
    /// Ids of the pathways leaving it
    pub out: Vec<usize>,
    /// The same counters as in `GET /metrics`
    pub counters: NeuronMetrics,
    /// Whatever the neuron adds about itself ([`Neuron::describe`](super::Neuron::describe))
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub describe: serde_json::Value,
}

impl Fabric {
    /// The fabric's counters, pathways and arcs, for `GET /metrics`
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

    /// Neurons, mechanisms, pathways and latencies, for `GET /nerve`
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
            quiet_not_interested: load(&c.quiet_not_interested),
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
                latency_ms: p.latency.snapshot(),
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
                        latency_ms: latency.snapshot(),
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
        self.render_latencies(out);
        if m.neurons.iter().any(|n| n.kind == NeuronKind::Effector) {
            super::effector::render_prometheus_help(out);
        }
        for (i, cell) in self.0.cells.iter().enumerate() {
            cell.neuron
                .render_prometheus(out, &format!("neuron=\"{}\"", label(&m.neurons[i].name)));
        }
    }
}

impl Fabric {
    /// The pathways' and the reflex arcs' latencies, as histograms of every measurement
    fn render_latencies(&self, out: &mut String) {
        let name = "furnace_nerve_pathway_latency_seconds";
        render_family(
            out,
            name,
            "histogram",
            "Measured signal-in to signal-out on the pathway",
        );
        for p in &self.0.pathways {
            let labels = between(self.name(p.from), self.name(p.to));
            p.latency.histogram().render_series(out, name, &labels);
        }
        let name = "furnace_nerve_arc_latency_seconds";
        render_family(
            out,
            name,
            "histogram",
            "Stimulus to effector call on each reflex arc",
        );
        for cell in &self.0.cells {
            for (origin, arc) in cell.arcs.iter().enumerate() {
                if let Some(latency) = arc.get() {
                    let labels = between(self.0.cells[origin].neuron.name(), cell.neuron.name());
                    latency.histogram().render_series(out, name, &labels);
                }
            }
        }
    }
}

/// The fabric's own families: its counters, the neurons' and the pathways' (the latencies
/// are [`Fabric::render_latencies`]')
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

    let neuron =
        |n: &NeuronMetrics| format!("neuron=\"{}\",kind=\"{}\"", label(&n.name), n.kind.as_str());
    render_family(
        out,
        "furnace_nerve_neuron_received_total",
        counter,
        "Signals that reached the neuron's gate",
    );
    for n in &m.neurons {
        render_labelled(
            out,
            "furnace_nerve_neuron_received_total",
            &neuron(n),
            n.received,
        );
    }
    render_family(
        out,
        "furnace_nerve_neuron_fired_total",
        counter,
        "Times the neuron fired",
    );
    for n in &m.neurons {
        render_labelled(out, "furnace_nerve_neuron_fired_total", &neuron(n), n.fired);
    }
    render_family(
        out,
        "furnace_nerve_neuron_quiet_total",
        counter,
        "Signals that died at the neuron, by reason",
    );
    for n in &m.neurons {
        for (reason, value) in [
            (Quiet::Threshold, n.quiet_by_threshold),
            (Quiet::Habituation, n.quiet_by_habituation),
            (Quiet::Refractory, n.quiet_by_refractory),
            (Quiet::NotInterested, n.quiet_not_interested),
        ] {
            let labels = format!(
                "neuron=\"{}\",reason=\"{}\"",
                label(&n.name),
                reason.as_str()
            );
            render_labelled(out, "furnace_nerve_neuron_quiet_total", &labels, value);
        }
    }

    render_family(
        out,
        "furnace_nerve_pathway_signals_total",
        counter,
        "Signals that took the pathway",
    );
    for p in &m.pathways {
        let labels = format!("{},kind=\"{}\"", between(&p.from, &p.to), p.kind.as_str());
        render_labelled(
            out,
            "furnace_nerve_pathway_signals_total",
            &labels,
            p.signals,
        );
    }
}

/// The `from` and `to` labels of a pathway or an arc
fn between(from: &str, to: &str) -> String {
    format!("from=\"{}\",to=\"{}\"", label(from), label(to))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::nerve::effector::{ChannelOut, Effector};
    use crate::nerve::fabric::Summation;
    use crate::nerve::neuron::{Firing, FnNeuron, Mechanisms};

    fn neuron(name: &str, kind: NeuronKind, base: u64) -> NeuronMetrics {
        NeuronMetrics {
            name: name.to_string(),
            kind,
            received: base,
            fired: base + 1,
            quiet_by_threshold: base + 2,
            quiet_by_habituation: base + 3,
            quiet_by_refractory: base + 4,
            quiet_not_interested: base + 5,
            computed: 0,
            extra: serde_json::Value::Null,
        }
    }

    fn latency(count: u64) -> LatencySnapshot {
        LatencySnapshot {
            count,
            ..LatencySnapshot::default()
        }
    }

    /// Every counter family the fabric writes, with label values that need escaping
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
                    latency_ms: latency(7),
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
                    latency_ms: latency(0),
                    window_us: Some(250.0),
                },
            ],
            arcs: vec![ArcMetrics {
                from: "in".into(),
                to: "brain".into(),
                latency_ms: latency(3),
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
        let (channel, _rx) = ChannelOut::new("out", 1);
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
        let fabric = b.build().unwrap();
        let mut text = String::new();
        fabric.render_prometheus(&mut text);
        let latencies = [
            "# HELP furnace_nerve_pathway_latency_seconds Measured signal-in to signal-out on \
             the pathway\n# TYPE furnace_nerve_pathway_latency_seconds histogram\n"
                .to_string(),
            empty_histogram(
                "furnace_nerve_pathway_latency_seconds",
                r#"from="in",to="out""#,
            ),
            empty_histogram(
                "furnace_nerve_pathway_latency_seconds",
                r#"from="in",to="brain""#,
            ),
            "# HELP furnace_nerve_arc_latency_seconds Stimulus to effector call on each reflex \
             arc\n# TYPE furnace_nerve_arc_latency_seconds histogram\n"
                .to_string(),
        ]
        .concat();
        let expected = WITH_EFFECTOR.replace("LATENCIES\n", &latencies);
        assert_eq!(text, expected, "\n{text}");
    }

    /// The series of a histogram with no observations, in the fine buckets
    fn empty_histogram(name: &str, labels: &str) -> String {
        let mut out = String::new();
        for le in crate::telemetry::FINE_LATENCY_BUCKETS
            .iter()
            .map(f64::to_string)
            .chain(["+Inf".to_string()])
        {
            out.push_str(&format!("{name}_bucket{{{labels},le=\"{le}\"}} 0\n"));
        }
        out.push_str(&format!(
            "{name}_sum{{{labels}}} 0\n{name}_count{{{labels}}} 0\n"
        ));
        out
    }

    #[test]
    fn a_measured_pathway_renders_its_histogram() {
        let mut b = Fabric::builder();
        let input = b.neuron(
            FnNeuron::reflex("in", |s, _| Firing::one(s)).with_kind(NeuronKind::Receptor),
            Mechanisms::NONE,
        );
        let sink = b.neuron(
            FnNeuron::reflex("sink", |_, _| Firing::Quiet),
            Mechanisms::NONE,
        );
        b.reflex(input, sink, 1.0);
        let fabric = b.build().unwrap();
        fabric.0.pathways[0]
            .latency
            .observe(Duration::from_micros(3));
        let mut text = String::new();
        fabric.render_prometheus(&mut text);
        let name = "furnace_nerve_pathway_latency_seconds";
        let labels = r#"from="in",to="sink""#;
        for line in [
            format!("{name}_bucket{{{labels},le=\"0.0000025\"}} 0"),
            format!("{name}_bucket{{{labels},le=\"0.000005\"}} 1"),
            format!("{name}_bucket{{{labels},le=\"+Inf\"}} 1"),
            format!("{name}_count{{{labels}}} 1"),
        ] {
            assert!(text.lines().any(|l| l == line), "{line}\n{text}");
        }
        let json = serde_json::to_value(fabric.metrics()).unwrap();
        let latency = &json["pathways"][0]["latency_ms"];
        assert_eq!(latency["count"], 1);
        assert!(latency["p50"].as_f64().unwrap() <= 0.005, "{latency}");
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
furnace_nerve_pathway_signals_total{from="in",to="cl\"assify\\",kind="reflex"} 7
furnace_nerve_pathway_signals_total{from="cl\"assify\\",to="brain",kind="deliberative"} 2
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
furnace_nerve_pathway_signals_total{from="in",to="out",kind="reflex"} 0
furnace_nerve_pathway_signals_total{from="in",to="brain",kind="deliberative"} 0
LATENCIES
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
