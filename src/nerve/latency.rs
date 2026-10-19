//! Measured latencies: a rolling window of recent samples with cached p50 and p99, cheap to
//! update on every signal and cheap to read on every pathway selection.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Samples kept; the percentiles are of the most recent ones
const WINDOW: usize = 128;
/// Percentiles are recomputed every this many samples (and after each of the first few)
const EVERY: u64 = 16;
/// The value of a percentile that was never computed
const UNKNOWN: u64 = u64::MAX;

/// A latency estimate, in nanoseconds, from a rolling window of measurements. Never
/// configured, only measured.
pub struct LatencyEstimate {
    recent: Mutex<Ring>,
    p50: AtomicU64,
    p99: AtomicU64,
    count: AtomicU64,
    sum: AtomicU64,
    max: AtomicU64,
}

struct Ring {
    samples: [u32; WINDOW],
    next: usize,
    len: usize,
}

/// A snapshot for metrics, in microseconds
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencySnapshot {
    pub count: u64,
    pub p50_us: Option<f64>,
    pub p99_us: Option<f64>,
    pub mean_us: Option<f64>,
    pub max_us: Option<f64>,
}

impl Default for LatencyEstimate {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyEstimate {
    pub fn new() -> Self {
        Self {
            recent: Mutex::new(Ring {
                samples: [0; WINDOW],
                next: 0,
                len: 0,
            }),
            p50: AtomicU64::new(UNKNOWN),
            p99: AtomicU64::new(UNKNOWN),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            max: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let sample = u32::try_from(nanos).unwrap_or(u32::MAX);
        let count = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        self.sum.fetch_add(nanos, Ordering::Relaxed);
        self.max.fetch_max(nanos, Ordering::Relaxed);
        let mut ring = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        let next = ring.next;
        ring.samples[next] = sample;
        ring.next = (next + 1) % WINDOW;
        ring.len = (ring.len + 1).min(WINDOW);
        if count <= 4 || count.is_multiple_of(EVERY) {
            let mut sorted = ring.samples[..ring.len].to_vec();
            drop(ring);
            sorted.sort_unstable();
            let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
            self.p50.store(u64::from(at(0.50)), Ordering::Relaxed);
            self.p99.store(u64::from(at(0.99)), Ordering::Relaxed);
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

    pub fn p99(&self) -> Option<Duration> {
        self.get(&self.p99)
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> LatencySnapshot {
        let count = self.count();
        let us = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1e6);
        LatencySnapshot {
            count,
            p50_us: us(self.p50()),
            p99_us: us(self.p99()),
            mean_us: (count > 0)
                .then(|| self.sum.load(Ordering::Relaxed) as f64 / count as f64 / 1e3),
            max_us: (count > 0).then(|| self.max.load(Ordering::Relaxed) as f64 / 1e3),
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
        let snapshot = l.snapshot();
        assert_eq!(snapshot.count, 201);
        assert_eq!(snapshot.p99_us, Some(5.0));
        assert_eq!(snapshot.max_us, Some(10.0));
        let mean = snapshot.mean_us.unwrap();
        assert!((5.0..5.1).contains(&mean), "{mean}");
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
