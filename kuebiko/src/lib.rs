//! Latency histograms and the Prometheus text format, with no dependencies.
//!
//! Named after Kuebiko, the scarecrow god of Japanese myth who cannot walk yet knows everything
//! under heaven, as a metric stands in one place and sees every request.
//!
//! The histograms are lock-free (one atomic counter per bucket), cheap enough to update on
//! every request, and give both the Prometheus `_bucket`/`_sum`/`_count` series and estimated
//! percentiles. The `render_*` functions append counters, gauges and labelled samples to a
//! `String` in the [Prometheus text format], for a `GET /metrics` handler to return as is.
//! [furnace](https://github.com/Gilfeather/furnace) serves its metrics with them.
//!
//! ```
//! use std::time::Duration;
//! use kuebiko::{Histogram, label, render_family, render_labelled, render_value};
//!
//! let latency = Histogram::new();
//! latency.observe(Duration::from_millis(8));
//! latency.observe(Duration::from_millis(12));
//! assert!(latency.percentiles_ms().p50.is_some());
//!
//! let mut out = String::new();
//! latency.render(&mut out, "app_request_duration_seconds", "Time to answer a request");
//! render_value(&mut out, "app_up", "gauge", "Whether the app is up", 1);
//! render_family(&mut out, "app_requests_total", "counter", "Requests by model");
//! render_labelled(&mut out, "app_requests_total", &format!("model=\"{}\"", label("a\"b")), 2);
//! assert!(out.contains("app_request_duration_seconds_count 2\n"));
//! assert!(out.contains("app_requests_total{model=\"a\\\"b\"} 2\n"));
//! ```
//!
//! Feature `serde`: `Serialize` and `Deserialize` for [`Percentiles`].
//!
//! [Prometheus text format]: https://prometheus.io/docs/instrumenting/exposition_formats/

use std::fmt::{Display, Write as _};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Bucket upper bounds in seconds, from 0.5 ms to 10 s
pub const LATENCY_BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0,
];

/// Bucket upper bounds in seconds, from 1 µs to 10 s, for the nervous system's hops, which
/// take microseconds
pub const FINE_LATENCY_BUCKETS: &[f64] = &[
    0.000_001,
    0.000_002_5,
    0.000_005,
    0.000_01,
    0.000_025,
    0.000_05,
    0.000_1,
    0.000_25,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
];

/// A latency histogram with fixed buckets (by default [`LATENCY_BUCKETS`], plus `+Inf`)
pub struct Histogram {
    /// Upper bounds in seconds, ascending
    bounds: &'static [f64],
    /// Observations per bucket (not cumulative); the last one is `+Inf`
    counts: Vec<AtomicU64>,
    sum_nanos: AtomicU64,
}

/// Estimated percentiles of a histogram, in milliseconds (`None` before the first observation)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Percentiles {
    /// The median
    pub p50: Option<f64>,
    /// The 95th percentile
    pub p95: Option<f64>,
    /// The 99th percentile
    pub p99: Option<f64>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// An empty histogram with [`LATENCY_BUCKETS`]
    pub fn new() -> Self {
        Self::with_buckets(LATENCY_BUCKETS)
    }

    /// An empty histogram with these upper bounds in seconds, ascending (such as
    /// [`FINE_LATENCY_BUCKETS`]), plus `+Inf`
    pub fn with_buckets(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_nanos: AtomicU64::new(0),
        }
    }

    /// Observations so far
    pub fn count(&self) -> u64 {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Count one observation
    pub fn observe(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        let bucket = self
            .bounds
            .iter()
            .position(|&bound| seconds <= bound)
            .unwrap_or(self.bounds.len());
        self.counts[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum_nanos.fetch_add(
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// The bucket counts and the sum in seconds.
    ///
    /// Not atomic across the counters: each is read on its own. An observation touches one
    /// bucket, and `_count` is the bucket total, so the buckets and `_count` always agree; but
    /// an observation in flight can be in its bucket before its time is in the sum, so in one
    /// scrape `_sum` may lag `_count` by the observations being recorded. The next scrape has
    /// them.
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
                let Some(&upper) = self.bounds.get(i) else {
                    return self.bounds.last().copied();
                };
                let lower = if i == 0 { 0.0 } else { self.bounds[i - 1] };
                let fraction = ((rank - below as f64) / count as f64).clamp(0.0, 1.0);
                return Some(lower + (upper - lower) * fraction);
            }
            below += count;
        }
        self.bounds.last().copied()
    }

    /// The median, 95th and 99th percentiles in milliseconds, as [`quantile`](Self::quantile)
    /// estimates them
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
        render_family(out, name, "histogram", help);
        self.render_series(out, name, "");
    }

    /// Append the series alone, each with `labels` (e.g. `model="a"`, or empty), for a
    /// histogram family whose `HELP` and `TYPE` lines were written already
    pub fn render_series(&self, out: &mut String, name: &str, labels: &str) {
        let (counts, sum) = self.snapshot();
        let sep = if labels.is_empty() { "" } else { "," };
        let mut cumulative = 0;
        for (i, count) in counts.iter().enumerate() {
            cumulative += count;
            let le = self
                .bounds
                .get(i)
                .map_or_else(|| "+Inf".to_string(), |b| b.to_string());
            let _ = writeln!(
                out,
                "{name}_bucket{{{labels}{sep}le=\"{le}\"}} {cumulative}"
            );
        }
        render_labelled(out, &format!("{name}_sum"), labels, sum);
        render_labelled(out, &format!("{name}_count"), labels, cumulative);
    }
}

/// Append one counter or gauge in the Prometheus text format
pub fn render_value(out: &mut String, name: &str, kind: &str, help: &str, value: impl ToString) {
    render_family(out, name, kind, help);
    render_labelled(out, name, "", value.to_string());
}

/// Append the `HELP` and `TYPE` lines of a family (`kind` is `counter`, `gauge`, ...), for
/// the samples [`render_labelled`] writes after them
pub fn render_family(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

/// Append one sample, `name{labels} value`; `labels` is written as given (`a="x",b="y"`,
/// values escaped with [`label`]), and an empty one leaves out the braces
pub fn render_labelled(out: &mut String, name: &str, labels: &str, value: impl Display) {
    if labels.is_empty() {
        let _ = writeln!(out, "{name} {value}");
    } else {
        let _ = writeln!(out, "{name}{{{labels}}} {value}");
    }
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

    mod properties {
        use super::*;
        use proptest::prelude::*;

        /// Latencies up to a minute, most of them within the buckets
        fn latencies() -> impl Strategy<Value = Vec<Duration>> {
            prop::collection::vec(
                prop_oneof![
                    (0u64..20_000_000_000).prop_map(Duration::from_nanos),
                    (0u64..60_000_000_000).prop_map(Duration::from_nanos),
                    prop::sample::select(LATENCY_BUCKETS.to_vec())
                        .prop_map(Duration::from_secs_f64),
                ],
                0..200,
            )
        }

        /// The value of `name{labels} value` in `out`
        fn sample(out: &str, series: &str) -> f64 {
            out.lines()
                .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
                .unwrap_or_else(|| panic!("no {series} in {out}"))
                .parse()
                .unwrap()
        }

        proptest! {
            /// Each bucket counts the observations at or under its bound, the series are
            /// cumulative and end at the count, and the sum is the observations' sum
            #[test]
            fn the_buckets_and_the_sum_add_up(latencies in latencies(), fine in any::<bool>()) {
                let bounds = if fine { FINE_LATENCY_BUCKETS } else { LATENCY_BUCKETS };
                let h = Histogram::with_buckets(bounds);
                for &latency in &latencies {
                    h.observe(latency);
                }
                prop_assert_eq!(h.count(), latencies.len() as u64);
                let mut out = String::new();
                h.render_series(&mut out, "x", "m=\"a\"");
                let mut previous = 0.0;
                for &bound in bounds {
                    let expected = latencies.iter().filter(|l| l.as_secs_f64() <= bound).count();
                    let got = sample(&out, &format!("x_bucket{{m=\"a\",le=\"{bound}\"}}"));
                    prop_assert_eq!(got, expected as f64, "le={}", bound);
                    prop_assert!(got >= previous);
                    previous = got;
                }
                let count = latencies.len() as f64;
                prop_assert_eq!(sample(&out, "x_bucket{m=\"a\",le=\"+Inf\"}"), count);
                prop_assert_eq!(sample(&out, "x_count{m=\"a\"}"), count);
                let sum: u64 = latencies.iter().map(|l| l.as_nanos() as u64).sum();
                let rendered = sample(&out, "x_sum{m=\"a\"}");
                prop_assert!(
                    (rendered - sum as f64 / 1e9).abs() <= 1e-9 * (1.0 + rendered),
                    "{} vs {}", rendered, sum
                );
            }

            /// Quantiles exist once there is an observation, grow with `q`, and stay within the
            /// buckets; the bucket of each quantile holds at least that share of observations
            #[test]
            fn quantiles_are_ordered_and_within_the_buckets(
                latencies in latencies(),
                mut qs in prop::collection::vec(0.0f64..=1.0, 1..8),
            ) {
                let h = Histogram::new();
                for &latency in &latencies {
                    h.observe(latency);
                }
                qs.sort_by(f64::total_cmp);
                let values: Vec<Option<f64>> = qs.iter().map(|&q| h.quantile(q)).collect();
                if latencies.is_empty() {
                    prop_assert!(values.iter().all(Option::is_none));
                    prop_assert_eq!(h.percentiles_ms(), Percentiles::default());
                    return Ok(());
                }
                let values: Vec<f64> = values.into_iter().map(Option::unwrap).collect();
                let top = *LATENCY_BUCKETS.last().unwrap();
                for (&q, &value) in qs.iter().zip(&values) {
                    prop_assert!((0.0..=top).contains(&value), "q={} gives {}", q, value);
                    // Under the top bound, at least a `q` share of the observations is at or
                    // under the bucket's upper bound; the top bound stands for `+Inf` too
                    if value < top {
                        let bound = LATENCY_BUCKETS.iter().copied().find(|&b| value <= b);
                        let bound = bound.unwrap_or(top);
                        let under = latencies.iter().filter(|l| l.as_secs_f64() <= bound);
                        prop_assert!(
                            under.count() as f64 >= q * latencies.len() as f64 - 1e-9,
                            "q={} value={} bound={}", q, value, bound
                        );
                    }
                }
                prop_assert!(values.windows(2).all(|w| w[0] <= w[1]), "{:?} for {:?}", values, qs);
                // Out-of-range and NaN quantiles are answered too
                prop_assert!(h.quantile(-1.0).is_some() && h.quantile(2.0).is_some());
                prop_assert!(h.quantile(f64::NAN).is_some());
            }

            /// Observations from several threads at once are all counted
            #[test]
            fn concurrent_observations_are_all_counted(
                per_thread in prop::collection::vec(latencies(), 1..5)
            ) {
                let h = Histogram::new();
                std::thread::scope(|scope| {
                    for latencies in &per_thread {
                        let h = &h;
                        scope.spawn(move || latencies.iter().for_each(|&l| h.observe(l)));
                    }
                });
                let total: usize = per_thread.iter().map(Vec::len).sum();
                prop_assert_eq!(h.count(), total as u64);
            }

            /// An escaped label has no raw quote or newline left, and reads back as it was
            #[test]
            fn labels_read_back_as_they_were(value in "[ -~\n\"\\\\é]{0,40}") {
                let escaped = label(&value);
                prop_assert!(!escaped.contains('\n'));
                let mut unescaped = String::new();
                let mut chars = escaped.chars();
                while let Some(c) = chars.next() {
                    if c == '"' {
                        prop_assert!(false, "an unescaped quote in {}", escaped);
                    }
                    if c != '\\' {
                        unescaped.push(c);
                        continue;
                    }
                    match chars.next() {
                        Some('\\') => unescaped.push('\\'),
                        Some('"') => unescaped.push('"'),
                        Some('n') => unescaped.push('\n'),
                        other => prop_assert!(false, "a bad escape {:?} in {}", other, escaped),
                    }
                }
                prop_assert_eq!(unescaped, value);
            }
        }
    }
}
