//! Dynamic batching: concurrent requests share forward passes.
//!
//! A batcher runs one kind of job: flat `f32` inputs ([`FlatInput`], models with one float
//! tensor in and out) or named tensors ([`NamedInput`], models whose inputs all have fixed
//! per-item shapes, so that requests can be joined along the batch dimension).
//!
//! A few worker threads run the model's forward passes, one batch each at a time, and
//! requests wait for them in a shared queue. While some worker is idle, every request gets
//! a forward pass of its own, started at once. Once all workers are busy, the queue fills,
//! and the next worker to finish takes everything queued, up to `max_batch_size` items, as
//! one batch. If the queue runs dry before the batch is full while another request is on
//! its way in (its body is still being received or decoded), the worker waits for it, but
//! never longer than `batch_timeout` after the first request was queued. A lone request
//! therefore never waits.
//!
//! This order suits CPUs: one small pass leaves cores idle (between layers, and in layers
//! too small to split across every thread), and a second pass running alongside uses them,
//! which gains more than merging the two into one batch. Past a few passes at once,
//! batching wins: more passes only compete for the same cores, and with no limit every
//! request slows down together.
//!
//! The queue, the workers and the window are the [`engine`]'s, shared with the nervous
//! system's deliberative pathways; this module is the engine run over a [`Model`] with a
//! reply for every request.

pub(crate) mod engine;

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, oneshot};
use tracing::debug;

use self::engine::{Batching, Engine, Queued};
use crate::error::Result;
use crate::model::{FlatInput, Model};
use crate::tensor::HostTensor;

/// A request a [`Batcher`] can run together with others of its kind
pub trait BatchJob: Send + Sized + 'static {
    /// What the request receives back
    type Output: Send + 'static;

    /// Items in the request: its share of a batch
    fn items(&self) -> usize;

    /// Run the requests as one forward pass, each getting its own result
    fn run_many(model: &Model, jobs: Vec<Self>) -> Vec<Result<Self::Output>>;
}

impl BatchJob for FlatInput {
    type Output = Vec<f32>;

    fn items(&self) -> usize {
        self.batch_size
    }

    fn run_many(model: &Model, jobs: Vec<Self>) -> Vec<Result<Vec<f32>>> {
        model.predict_many(jobs)
    }
}

/// One request's named tensors, in the order of the model's inputs
#[derive(Debug)]
pub struct NamedInput(pub Vec<HostTensor>);

impl BatchJob for NamedInput {
    type Output = Vec<HostTensor>;

    fn items(&self) -> usize {
        self.0.first().map_or(0, HostTensor::batch)
    }

    fn run_many(model: &Model, jobs: Vec<Self>) -> Vec<Result<Vec<HostTensor>>> {
        model.infer_many(jobs.into_iter().map(|job| job.0).collect())
    }
}

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

/// What a queued request receives: its outputs, or its own error
pub type Reply<J = FlatInput> = Result<<J as BatchJob>::Output>;

struct Pending<J: BatchJob> {
    input: J,
    reply: oneshot::Sender<Reply<J>>,
    /// Released once the request is answered, so queued requests count against
    /// `--max-concurrent-requests` like running ones
    permit: OwnedSemaphorePermit,
}

/// Forward passes of `J` on a model: a batch is one pass, and each request gets its reply
struct Passes<J: BatchJob> {
    model: Arc<Model>,
    batch_timeout: Duration,
    job: PhantomData<fn() -> J>,
}

impl<J: BatchJob> Batching for Passes<J> {
    type Item = Pending<J>;

    fn weight(&self, pending: &Pending<J>) -> usize {
        pending.input.items()
    }

    /// The client may have timed out and stopped listening
    fn wanted(&self, pending: &Pending<J>) -> bool {
        !pending.reply.is_closed()
    }

    fn window(&self) -> Duration {
        self.batch_timeout
    }

    fn run(&self, batch: Vec<Queued<Pending<J>>>) {
        let items: usize = batch.iter().map(|q| q.item.input.items()).sum();
        debug!(
            "Running a batch of {} request(s), {} item(s)",
            batch.len(),
            items
        );
        let (inputs, replies): (Vec<_>, Vec<_>) = batch
            .into_iter()
            .map(|q| (q.item.input, (q.item.reply, q.item.permit)))
            .unzip();
        // Should the model panic, the replies are dropped with this frame, failing only
        // this batch's requests
        let results = J::run_many(&self.model, inputs);
        for ((reply, permit), result) in replies.into_iter().zip(results) {
            // Free the slot before answering, so a client that reads /metrics right after
            // its response never sees its own request still in flight
            drop(permit);
            // The request may have timed out and stopped listening
            let _ = reply.send(result);
        }
    }
}

/// Handle to the batching workers; they stop when the last handle is dropped
pub struct Batcher<J: BatchJob = FlatInput> {
    engine: Engine<Pending<J>>,
}

impl<J: BatchJob> Batcher<J> {
    pub fn new(model: Arc<Model>, config: BatchConfig) -> Self {
        let passes = Passes {
            model,
            batch_timeout: config.batch_timeout,
            job: PhantomData,
        };
        let engine = Engine::start(
            passes,
            engine::Config {
                max_batch: config.max_batch_size,
                workers: config.workers,
                capacity: None,
                waits_unannounced: false,
            },
            |i| format!("furnace-batcher-{i}"),
        );
        Self { engine }
    }

    /// Announce a request that will probably be submitted soon. Hold the guard while
    /// receiving and decoding its body, then pass it to [`Batcher::submit`].
    pub fn arrival(&self) -> Arrival<'_, J> {
        Arrival(self.engine.arrival())
    }

    /// Queue a request; the receiver yields its result, or an error if the batch it ran
    /// in panicked.
    pub fn submit(
        &self,
        arrival: Arrival<'_, J>,
        input: J,
        permit: OwnedSemaphorePermit,
    ) -> oneshot::Receiver<Reply<J>> {
        let (reply, receiver) = oneshot::channel();
        let pending = Pending {
            input,
            reply,
            permit,
        };
        // The queue has no bound, so this fails only once the workers are gone; the
        // receiver then reports the request as failed
        let _ = self.engine.submit(Some(arrival.0), pending);
        receiver
    }
}

/// See [`Batcher::arrival`]
pub struct Arrival<'a, J: BatchJob = FlatInput>(engine::Arrival<'a, Pending<J>>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use crate::error::Result;
    use crate::model::BurnModel;
    use crate::tensor::{HostTensor, IoSpec, Values};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::time::Instant;
    use tokio::sync::Semaphore;

    const WIDTH: usize = 4;

    /// Doubles its input and records every batch size it sees, and the most passes that ran
    /// at once. It can be slowed down or held, and it panics on an input of -1, so tests can
    /// check what was batched with what.
    #[derive(Debug)]
    struct Doubler {
        shape: Vec<usize>,
        batches: Arc<Mutex<Vec<usize>>>,
        delay: Duration,
        running: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        holds: Arc<Holds>,
    }

    /// Input values whose passes wait until released, so a test can keep workers busy for
    /// exactly as long as it needs without depending on timing
    #[derive(Debug, Default)]
    struct Holds {
        held: Mutex<Vec<f32>>,
        released: Condvar,
    }

    impl Holds {
        fn hold(&self, value: f32) {
            self.held.lock().unwrap().push(value);
        }

        fn release(&self, value: f32) {
            self.held.lock().unwrap().retain(|&v| v != value);
            self.released.notify_all();
        }

        /// Block while any of the values is held
        fn wait(&self, values: &[f32]) {
            let mut held = self.held.lock().unwrap();
            while values.iter().any(|v| held.contains(v)) {
                held = self.released.wait(held).unwrap();
            }
        }
    }

    impl BurnModel for Doubler {
        fn name(&self) -> &str {
            "doubler"
        }
        fn backend(&self) -> BackendKind {
            BackendKind::Cpu
        }
        fn inputs(&self) -> Vec<IoSpec> {
            vec![IoSpec::fp32("input", &self.shape)]
        }
        fn outputs(&self) -> Vec<IoSpec> {
            vec![IoSpec::fp32("output", &self.shape)]
        }
        fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
            let input = inputs.into_iter().next().unwrap();
            self.batches.lock().unwrap().push(input.batch());
            let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            let Values::F32(values) = input.values else {
                panic!("f32 input");
            };
            self.holds.wait(&values);
            std::thread::sleep(self.delay);
            self.running.fetch_sub(1, Ordering::SeqCst);
            assert!(!values.contains(&-1.0), "poisoned input");
            Ok(vec![HostTensor {
                shape: input.shape,
                values: Values::F32(values.iter().map(|v| v * 2.0).collect()),
            }])
        }
    }

    #[test]
    fn named_tensor_requests_arriving_together_share_a_batch() {
        let model = Arc::new(Model::new(
            Box::new(crate::model::DummyTensorsModel::new(None)),
            "dummy",
            0,
        ));
        let batcher: Batcher<NamedInput> = Batcher::new(
            model.clone(),
            BatchConfig {
                max_batch_size: 8,
                batch_timeout: Duration::from_secs(5),
                workers: 1,
            },
        );
        let request = |id: i64| {
            vec![
                HostTensor {
                    shape: vec![1, 4],
                    values: Values::I64(vec![id; 4]),
                },
                HostTensor {
                    shape: vec![1],
                    values: Values::F32(vec![1.0]),
                },
            ]
        };
        let slots = Arc::new(Semaphore::new(8));
        // All three are announced before any is queued, so the worker waits for them all
        let arrivals: Vec<_> = (0..3).map(|_| batcher.arrival()).collect();
        let receivers: Vec<_> = arrivals
            .into_iter()
            .zip(1..)
            .map(|(arrival, id)| {
                let permit = slots.clone().try_acquire_owned().unwrap();
                batcher.submit(arrival, NamedInput(request(id)), permit)
            })
            .collect();
        for (receiver, id) in receivers.into_iter().zip(1i64..) {
            let outputs = receiver.blocking_recv().unwrap().unwrap();
            assert_eq!(outputs[1].values, Values::I64(vec![4 * id]));
        }
        let stats = model.get_stats();
        assert_eq!((stats.batch_count, stats.batched_items), (1, 3));
    }

    struct Setup {
        batcher: Batcher,
        model: Arc<Model>,
        batches: Arc<Mutex<Vec<usize>>>,
        running: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        holds: Arc<Holds>,
        slots: Arc<Semaphore>,
    }

    fn setup(max_batch_size: usize, batch_timeout: Duration, delay: Duration) -> Setup {
        setup_with_workers(1, max_batch_size, batch_timeout, delay)
    }

    fn setup_with_workers(
        workers: usize,
        max_batch_size: usize,
        batch_timeout: Duration,
        delay: Duration,
    ) -> Setup {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let holds = Arc::new(Holds::default());
        let model = Arc::new(Model::new(
            Box::new(Doubler {
                shape: vec![WIDTH],
                batches: batches.clone(),
                delay,
                running: running.clone(),
                peak: peak.clone(),
                holds: holds.clone(),
            }),
            "test",
            0,
        ));
        let config = BatchConfig {
            max_batch_size,
            batch_timeout,
            workers,
        };
        Setup {
            batcher: Batcher::new(model.clone(), config),
            model,
            batches,
            running,
            peak,
            holds,
            slots: Arc::new(Semaphore::new(1000)),
        }
    }

    impl Setup {
        /// Submit one request whose items are filled with the given values
        fn submit(&self, items: &[f32]) -> oneshot::Receiver<Reply> {
            self.submit_raw(
                items.iter().flat_map(|&v| [v; WIDTH]).collect(),
                items.len(),
                self.batcher.arrival(),
            )
        }

        fn submit_raw(
            &self,
            data: Vec<f32>,
            batch_size: usize,
            arrival: Arrival<'_>,
        ) -> oneshot::Receiver<Reply> {
            let permit = self.slots.clone().try_acquire_owned().unwrap();
            self.batcher
                .submit(arrival, FlatInput { data, batch_size }, permit)
        }

        fn batches(&self) -> Vec<usize> {
            self.batches.lock().unwrap().clone()
        }
    }

    fn expected(items: &[f32]) -> Vec<f32> {
        items.iter().flat_map(|&v| [2.0 * v; WIDTH]).collect()
    }

    /// Let the worker pick up what was submitted so far
    async fn settle() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    #[tokio::test]
    async fn concurrent_requests_get_their_own_outputs_in_order() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        // The first request occupies the worker while the rest queue up behind it
        let first = s.submit(&[100.0]);
        settle().await;
        let requests: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let i = i as f32;
                if i == 2.0 {
                    vec![i, i + 0.5, i + 0.25] // a client-side batch of 3
                } else {
                    vec![i]
                }
            })
            .collect();
        let receivers: Vec<_> = requests.iter().map(|r| s.submit(r)).collect();

        assert_eq!(first.await.unwrap().unwrap(), expected(&[100.0]));
        for (request, receiver) in requests.iter().zip(receivers) {
            assert_eq!(receiver.await.unwrap().unwrap(), expected(request));
        }
        // 1 alone, then 8 items fill the next batch
        assert_eq!(s.batches(), vec![1, 8]);
        let stats = s.model.get_stats();
        assert_eq!(stats.success_count, 7);
        assert_eq!(stats.batch_count, 2);
        assert_eq!(stats.batched_items, 9);
    }

    #[tokio::test]
    async fn invalid_request_fails_alone() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        let first = s.submit(&[1.0]);
        settle().await;
        let good = s.submit(&[2.0]);
        let not_finite = s.submit(&[f32::NAN]);
        let wrong_length = s.submit_raw(vec![1.0; WIDTH + 1], 1, s.batcher.arrival());
        let also_good = s.submit(&[3.0]);

        assert_eq!(first.await.unwrap().unwrap(), expected(&[1.0]));
        assert_eq!(good.await.unwrap().unwrap(), expected(&[2.0]));
        let err = not_finite.await.unwrap().unwrap_err().to_string();
        assert!(err.contains("not finite"), "{err}");
        assert!(wrong_length.await.unwrap().is_err());
        assert_eq!(also_good.await.unwrap().unwrap(), expected(&[3.0]));
        assert_eq!(s.batches(), vec![1, 2]);
        assert_eq!(s.model.get_stats().error_count, 2);
    }

    #[tokio::test]
    async fn batches_respect_max_size_and_keep_order() {
        let s = setup(4, Duration::ZERO, Duration::from_millis(30));
        let first = s.submit(&[0.0]);
        settle().await;
        let sizes = [3, 2, 6, 1, 1];
        let receivers: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| s.submit(&vec![i as f32; n]))
            .collect();
        first.await.unwrap().unwrap();
        for (i, (&n, receiver)) in sizes.iter().zip(receivers).enumerate() {
            assert_eq!(
                receiver.await.unwrap().unwrap(),
                expected(&vec![i as f32; n])
            );
        }
        // 3 (2 does not fit), 2, then 6 alone (above the maximum), then 1 + 1
        assert_eq!(s.batches(), vec![1, 3, 2, 6, 2]);
    }

    #[tokio::test]
    async fn lone_request_does_not_wait_for_the_window() {
        let s = setup(8, Duration::from_secs(5), Duration::ZERO);
        let start = Instant::now();
        assert_eq!(s.submit(&[1.0]).await.unwrap().unwrap(), expected(&[1.0]));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn waits_for_a_request_on_its_way_in() {
        let s = setup(8, Duration::from_secs(5), Duration::ZERO);
        // A second request is still being received while the first is queued
        let arriving = s.batcher.arrival();
        let first = s.submit(&[1.0]);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let second = s.submit_raw(vec![2.0; WIDTH], 1, arriving);
        assert_eq!(first.await.unwrap().unwrap(), expected(&[1.0]));
        assert_eq!(second.await.unwrap().unwrap(), expected(&[2.0]));
        assert_eq!(s.batches(), vec![2]);
    }

    #[tokio::test]
    async fn abandoned_arrival_ends_the_wait() {
        let s = setup(8, Duration::from_secs(5), Duration::ZERO);
        let arriving = s.batcher.arrival();
        let start = Instant::now();
        let first = s.submit(&[1.0]);
        settle().await;
        // E.g. its body failed to decode
        drop(arriving);
        assert_eq!(first.await.unwrap().unwrap(), expected(&[1.0]));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn wait_is_bounded_by_the_timeout() {
        let s = setup(8, Duration::from_millis(20), Duration::ZERO);
        let _never_submitted = s.batcher.arrival();
        let start = Instant::now();
        assert_eq!(s.submit(&[1.0]).await.unwrap().unwrap(), expected(&[1.0]));
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(20), "{elapsed:?}");
        assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
    }

    #[tokio::test]
    async fn permits_are_held_until_answered() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        let available = s.slots.available_permits();
        let first = s.submit(&[1.0]);
        let second = s.submit(&[2.0]);
        assert_eq!(s.slots.available_permits(), available - 2);
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        // The worker frees each permit before sending its reply
        assert_eq!(s.slots.available_permits(), available);
    }

    #[tokio::test]
    async fn abandoned_request_is_skipped() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        let first = s.submit(&[1.0]);
        settle().await;
        // The client timed out while queued
        drop(s.submit(&[2.0]));
        let kept = s.submit(&[3.0]);
        first.await.unwrap().unwrap();
        assert_eq!(kept.await.unwrap().unwrap(), expected(&[3.0]));
        assert_eq!(s.batches(), vec![1, 1]);
    }

    #[tokio::test]
    async fn idle_workers_run_requests_side_by_side() {
        let s = setup_with_workers(2, 8, Duration::from_secs(5), Duration::ZERO);
        // The first two passes are held, so both run until released
        s.holds.hold(1.0);
        s.holds.hold(2.0);
        let first = s.submit(&[1.0]);
        let second = s.submit(&[2.0]);
        // Each went to an idle worker: both passes run side by side
        let deadline = Instant::now() + Duration::from_secs(10);
        while s.running.load(Ordering::SeqCst) < 2 {
            assert!(
                Instant::now() < deadline,
                "the first two never ran together"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        // Both workers are busy, so these two wait in the queue
        let third = s.submit(&[3.0]);
        let fourth = s.submit(&[4.0]);
        // The worker that frees up takes both as one batch while the other is still busy.
        // (Freeing both at once could leave two idle workers, which rightly run them apart.)
        s.holds.release(1.0);
        assert_eq!(first.await.unwrap().unwrap(), expected(&[1.0]));
        assert_eq!(third.await.unwrap().unwrap(), expected(&[3.0]));
        assert_eq!(fourth.await.unwrap().unwrap(), expected(&[4.0]));
        s.holds.release(2.0);
        assert_eq!(second.await.unwrap().unwrap(), expected(&[2.0]));
        assert_eq!(s.peak.load(Ordering::SeqCst), 2);
        assert_eq!(s.batches(), vec![1, 1, 2]);
    }

    #[tokio::test]
    async fn panicking_batch_fails_only_its_requests() {
        let s = setup(8, Duration::ZERO, Duration::from_millis(30));
        let first = s.submit(&[1.0]);
        settle().await;
        let poisoned = s.submit(&[-1.0]);
        let neighbour = s.submit(&[2.0]);
        assert!(first.await.unwrap().is_ok());
        // Both were in the batch that panicked, so their replies were dropped
        assert!(poisoned.await.is_err());
        assert!(neighbour.await.is_err());
        // The worker survives
        assert_eq!(s.submit(&[3.0]).await.unwrap().unwrap(), expected(&[3.0]));
    }
}
