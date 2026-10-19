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
        for (i, cell) in self.0.cells.iter().enumerate() {
            cell.neuron
                .render_prometheus(out, &format!("neuron=\"{}\"", label(&m.neurons[i].name)));
        }
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
