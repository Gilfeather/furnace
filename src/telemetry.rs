//! Latency histograms and the Prometheus text format for `GET /metrics`.
//!
//! The histograms are lock-free (one atomic counter per bucket), cheap enough to update on
//! every request, and give both the Prometheus `_bucket`/`_sum`/`_count` series and estimated
//! percentiles for the JSON metrics.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;

/// Bucket upper bounds in seconds, from 0.5 ms to 10 s
pub const LATENCY_BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0,
];

/// A latency histogram with fixed buckets ([`LATENCY_BUCKETS`] plus `+Inf`)
pub struct Histogram {
    /// Observations per bucket (not cumulative); the last one is `+Inf`
    counts: Vec<AtomicU64>,
    sum_nanos: AtomicU64,
}

/// Estimated percentiles of a histogram, in milliseconds (`None` before the first observation)
#[derive(Debug, Clone, Copy, Default, Serialize, serde::Deserialize, PartialEq)]
pub struct Percentiles {
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            counts: (0..=LATENCY_BUCKETS.len())
                .map(|_| AtomicU64::new(0))
                .collect(),
            sum_nanos: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        let bucket = LATENCY_BUCKETS
            .iter()
            .position(|&bound| seconds <= bound)
            .unwrap_or(LATENCY_BUCKETS.len());
        self.counts[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_nanos.fetch_add(
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn snapshot(&self) -> (Vec<u64>, f64) {
        let counts = self
            .counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let sum = self.sum_nanos.load(Ordering::Relaxed) as f64 / 1e9;
        (counts, sum)
    }

    /// The `q` quantile (0..=1) in seconds, interpolated linearly within its bucket. Values in
    /// the `+Inf` bucket are reported as the largest finite bound.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        let (counts, _) = self.snapshot();
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return None;
        }
        let rank = q.clamp(0.0, 1.0) * total as f64;
        let mut below = 0u64;
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 && (below + count) as f64 >= rank {
                let Some(&upper) = LATENCY_BUCKETS.get(i) else {
                    return LATENCY_BUCKETS.last().copied();
                };
                let lower = if i == 0 { 0.0 } else { LATENCY_BUCKETS[i - 1] };
                let fraction = ((rank - below as f64) / count as f64).clamp(0.0, 1.0);
                return Some(lower + (upper - lower) * fraction);
            }
            below += count;
        }
        LATENCY_BUCKETS.last().copied()
    }

    pub fn percentiles_ms(&self) -> Percentiles {
        let ms = |q| self.quantile(q).map(|s| s * 1000.0);
        Percentiles {
            p50: ms(0.50),
            p95: ms(0.95),
            p99: ms(0.99),
        }
    }

    /// Append this histogram as Prometheus `_bucket`, `_sum` and `_count` series
    pub fn render(&self, out: &mut String, name: &str, help: &str) {
        let (counts, sum) = self.snapshot();
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} histogram");
        let mut cumulative = 0;
        for (i, count) in counts.iter().enumerate() {
            cumulative += count;
            let le = LATENCY_BUCKETS
                .get(i)
                .map_or_else(|| "+Inf".to_string(), |b| b.to_string());
            let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {cumulative}");
        }
        let _ = writeln!(out, "{name}_sum {sum}");
        let _ = writeln!(out, "{name}_count {cumulative}");
    }
}

/// Append one counter or gauge in the Prometheus text format
pub fn render_value(out: &mut String, name: &str, kind: &str, help: &str, value: impl ToString) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
    let _ = writeln!(out, "{name} {}", value.to_string());
}

/// Escape a Prometheus label value
pub fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram_has_no_percentiles() {
        assert_eq!(Histogram::new().percentiles_ms(), Percentiles::default());
    }

    #[test]
    fn quantiles_fall_in_the_right_buckets() {
        let h = Histogram::new();
        // 90 fast requests (~8 ms) and 10 slow ones (~200 ms)
        for _ in 0..90 {
            h.observe(Duration::from_millis(8));
        }
        for _ in 0..10 {
            h.observe(Duration::from_millis(200));
        }
        let p = h.percentiles_ms();
        let p50 = p.p50.unwrap();
        assert!((7.5..=10.0).contains(&p50), "{p50}");
        let p99 = p.p99.unwrap();
        assert!((100.0..=250.0).contains(&p99), "{p99}");
    }

    #[test]
    fn prometheus_series_are_cumulative() {
        let h = Histogram::new();
        h.observe(Duration::from_micros(300));
        h.observe(Duration::from_millis(3));
        h.observe(Duration::from_secs(30));
        let mut out = String::new();
        h.render(&mut out, "x_seconds", "test");
        assert!(out.contains("x_seconds_bucket{le=\"0.0005\"} 1\n"), "{out}");
        assert!(out.contains("x_seconds_bucket{le=\"0.005\"} 2\n"), "{out}");
        assert!(out.contains("x_seconds_bucket{le=\"10\"} 2\n"), "{out}");
        assert!(out.contains("x_seconds_bucket{le=\"+Inf\"} 3\n"), "{out}");
        assert!(out.contains("x_seconds_count 3\n"), "{out}");
        assert!(out.contains("# TYPE x_seconds histogram\n"), "{out}");
    }

    #[test]
    fn labels_are_escaped() {
        assert_eq!(label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
