//! Dynamic batching: concurrent requests share forward passes.
//!
//! A few workers run a model's forward passes, one batch each at a time, and requests wait for
//! them in a shared queue. While some worker is idle, every request gets a forward pass of its
//! own, started at once. Once all workers are busy, the queue fills, and the next worker to
//! finish takes everything queued, up to `max_batch_size` items, as one batch. If the queue
//! runs dry before the batch is full while another request is on its way in (its body is still
//! being received or decoded), the worker waits for it, but never longer than `batch_timeout`
//! after the first request was queued. A lone request therefore never waits.
//!
//! This order suits CPUs: one small pass leaves cores idle (between layers, and in layers
//! too small to split across every thread), and a second pass running alongside uses them,
//! which gains more than merging the two into one batch. Past a few passes at once,
//! batching wins: more passes only compete for the same cores, and with no limit every
//! request slows down together.
//!
//! The algorithm is the batching engine's, the crate amate. The server runs it on its
//! models' deliberative pathways in the nervous system (`nerve::serve`); [`BatchConfig`] is how
//! those pathways sum requests.

use std::time::Duration;

/// How requests are grouped into batches.
///
/// Batching is off by default: every request runs its own forward pass at once, as in a
/// hand-written server. Whether merging requests pays depends on the model. A convolutional
/// network on a CPU (or most models on a GPU) runs a batch of 4 in about the time of 2 single
/// passes, and serves more requests with batching on; a transformer on a CPU takes nearly 4
/// times as long for 4, and merging only makes requests wait. Turn it on with
/// `max_batch_size` above 1 for models of the first kind (`benches/burn` measures both).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchConfig {
    /// Most items in one forward pass; 1 (the default) turns batching off. A request with
    /// more items than this still runs, alone.
    pub max_batch_size: usize,
    /// Longest wait for requests still being received, counted from when the first
    /// request of the batch was queued
    pub batch_timeout: Duration,
    /// Forward passes that may run at the same time; requests are merged into batches
    /// only while all of them are busy
    pub workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            batch_timeout: Duration::from_millis(2),
            workers: 4,
        }
    }
}

impl BatchConfig {
    pub fn enabled(&self) -> bool {
        self.max_batch_size > 1
    }
}
