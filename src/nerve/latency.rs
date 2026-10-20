//! Measured latencies: a rolling window of recent samples with cached p50 and p99, cheap to
//! update on every signal and cheap to read on every pathway selection, and a
//! [`Histogram`] of every sample for the metrics, in the shape the rest of `/metrics` uses.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::telemetry::{FINE_LATENCY_BUCKETS, Histogram, Percentiles};

/// Samples kept; the percentiles are of the most recent ones
const WINDOW: usize = 128;
/// Percentiles are recomputed every this many samples (and after each of the first few)
const EVERY: u64 = 16;
/// The value of a percentile that was never computed
const UNKNOWN: u64 = u64::MAX;

/// A latency estimate, in nanoseconds, from a rolling window of measurements. Never
/// configured, only measured. The window's percentiles steer pathway selection; the
/// histogram of every measurement is what the metrics report.
pub struct LatencyEstimate {
    recent: Mutex<Ring>,
    histogram: Histogram,
    p50: AtomicU64,
    p99: AtomicU64,
    count: AtomicU64,
    sum: AtomicU64,
    max: AtomicU64,
}

struct Ring {
    samples: [u64; WINDOW],
    next: usize,
    len: usize,
}

/// A snapshot for metrics, in milliseconds as the rest of `/metrics`: the percentiles are
/// the histogram's estimates over every measurement (see [`Histogram::percentiles_ms`])
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencySnapshot {
    /// Measurements so far
    pub count: u64,
    /// The median and the 95th and 99th percentiles; `None` before the first measurement
    #[serde(flatten)]
    pub percentiles: Percentiles,
    /// Mean of every measurement; `None` before the first
    pub mean: Option<f64>,
    /// Largest measurement so far; `None` before the first
    pub max: Option<f64>,
}

impl Default for LatencyEstimate {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyEstimate {
    /// An estimate with no measurements yet
    pub fn new() -> Self {
        Self {
            recent: Mutex::new(Ring {
                samples: [0; WINDOW],
                next: 0,
                len: 0,
            }),
            histogram: Histogram::with_buckets(FINE_LATENCY_BUCKETS),
            p50: AtomicU64::new(UNKNOWN),
            p99: AtomicU64::new(UNKNOWN),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            max: AtomicU64::new(0),
        }
    }

    /// Record one measurement
    pub fn observe(&self, elapsed: Duration) {
        self.histogram.observe(elapsed);
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let count = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        self.sum.fetch_add(nanos, Ordering::Relaxed);
        self.max.fetch_max(nanos, Ordering::Relaxed);
        let mut ring = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        let next = ring.next;
        ring.samples[next] = nanos;
        ring.next = (next + 1) % WINDOW;
        ring.len = (ring.len + 1).min(WINDOW);
        if count <= 4 || count.is_multiple_of(EVERY) {
            // On the stack, and only the two order statistics: no allocation, no full sort
            let len = ring.len;
            let mut samples = ring.samples;
            drop(ring);
            let samples = &mut samples[..len];
            let rank = |q: f64| ((len - 1) as f64 * q).round() as usize;
            let (p50, p99) = (rank(0.50), rank(0.99));
            let (below, &mut at_p99, _) = samples.select_nth_unstable(p99);
            let at_p50 = if p50 == p99 {
                at_p99
            } else {
                *below.select_nth_unstable(p50).1
            };
            self.p50.store(at_p50, Ordering::Relaxed);
            self.p99.store(at_p99, Ordering::Relaxed);
        }
    }

    fn get(&self, cell: &AtomicU64) -> Option<Duration> {
        match cell.load(Ordering::Relaxed) {
            UNKNOWN => None,
            nanos => Some(Duration::from_nanos(nanos)),
        }
    }

    /// `None` until the first measurement
    pub fn p50(&self) -> Option<Duration> {
        self.get(&self.p50)
    }

    /// `None` until the first measurement
    pub fn p99(&self) -> Option<Duration> {
        self.get(&self.p99)
    }

    /// Measurements so far
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Every measurement, in [`FINE_LATENCY_BUCKETS`], for the Prometheus histograms
    pub fn histogram(&self) -> &Histogram {
        &self.histogram
    }

    /// Count, percentiles, mean and max, in milliseconds
    pub fn snapshot(&self) -> LatencySnapshot {
        let count = self.count();
        LatencySnapshot {
            count,
            percentiles: self.histogram.percentiles_ms(),
            mean: (count > 0).then(|| self.sum.load(Ordering::Relaxed) as f64 / count as f64 / 1e6),
            max: (count > 0).then(|| self.max.load(Ordering::Relaxed) as f64 / 1e6),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_before_the_first_sample_then_tracks_the_window() {
        let l = LatencyEstimate::new();
        assert_eq!(l.p99(), None);
        l.observe(Duration::from_micros(10));
        assert_eq!(l.p50(), Some(Duration::from_micros(10)));
        assert_eq!(l.p99(), Some(Duration::from_micros(10)));
        for _ in 0..200 {
            l.observe(Duration::from_micros(5));
        }
        // The 10 µs sample has left the window of 128
        assert_eq!(l.p99(), Some(Duration::from_micros(5)));
        // The histogram keeps every sample: 200 at 5 µs, one at 10 µs (its own bucket)
        let snapshot = l.snapshot();
        assert_eq!(snapshot.count, 201);
        assert_eq!(l.histogram().count(), 201);
        let p50 = snapshot.percentiles.p50.unwrap();
        assert!((0.0025..=0.005).contains(&p50), "{p50}");
        let p99 = snapshot.percentiles.p99.unwrap();
        assert!((0.0025..=0.005).contains(&p99), "{p99}");
        assert_eq!(snapshot.max, Some(0.01));
        let mean = snapshot.mean.unwrap();
        assert!((0.005..0.0051).contains(&mean), "{mean}");
    }

    #[test]
    fn p99_sees_the_slow_tail() {
        let l = LatencyEstimate::new();
        for i in 0..128 {
            l.observe(Duration::from_micros(if i % 50 == 0 { 1000 } else { 10 }));
        }
        assert_eq!(l.p50(), Some(Duration::from_micros(10)));
        assert_eq!(l.p99(), Some(Duration::from_micros(1000)));
    }
}
