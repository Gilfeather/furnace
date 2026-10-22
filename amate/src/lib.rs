//! Dynamic batching on worker threads, with no dependencies.
//!
//! Named after Amaterasu, the sun goddess of Japanese myth, who gathers scattered light and
//! sends it out at once, as amate gathers waiting requests into one batch.
//!
//! Items queue to a few worker threads, each running one batch at a time. While some worker
//! is idle, every item gets a batch of its own, started at once. Once all workers are busy,
//! the queue fills, and the next worker to finish takes everything queued, up to
//! [`Config::max_batch`] by weight, as one batch. If the queue runs dry before the batch is
//! full, the worker waits for more, but never longer than the [`Batching::window`] after
//! the first item was queued: a server waits only while an announced item is still on its
//! way in (its body is being received or decoded, see [`Engine::arrival`]), a batch that sums
//! over time waits out its window ([`Config::waits_unannounced`]). Items nobody wants any
//! more (their client gave up) are skipped, and a batch that panics loses only its own items,
//! where panics unwind: with `panic = "abort"` the process stops.
//!
//! What a batch is made of, how heavy each item is, how long the window is and what to do
//! with the batch are the [`Batching`]'s: the engine knows nothing about models or replies.
//! It is the algorithm behind [furnace](https://github.com/Gilfeather/furnace)'s dynamic
//! batching.
//!
//! ```
//! use std::sync::mpsc;
//! use std::time::Duration;
//! use amate::{Batching, Config, Engine, Queued};
//!
//! /// Sums the numbers of each batch and sends the totals on
//! struct Sum(mpsc::Sender<u64>);
//!
//! impl Batching for Sum {
//!     type Item = u64;
//!
//!     fn weight(&self, _: &u64) -> usize {
//!         1
//!     }
//!
//!     fn window(&self) -> Duration {
//!         Duration::from_millis(2)
//!     }
//!
//!     fn run(&self, batch: Vec<Queued<u64>>) {
//!         let _ = self.0.send(batch.iter().map(|queued| queued.item).sum());
//!     }
//! }
//!
//! let (totals, received) = mpsc::channel();
//! let config = Config { max_batch: 8, workers: 2, capacity: Some(1024), waits_unannounced: false };
//! let engine = Engine::start(Sum(totals), config, |i| format!("sum-{i}"));
//! for n in 1..=10 {
//!     engine.submit(None, n).unwrap();
//! }
//! // The workers run what is queued, then stop, and `Sum` with them
//! drop(engine);
//! assert_eq!(received.iter().sum::<u64>(), 55);
//! ```

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What a batch is made of and what to do with one
pub trait Batching: Send + Sync + 'static {
    type Item: Send + 'static;

    /// The item's share of a batch, counted against [`Config::max_batch`]. An item heavier
    /// than the maximum still runs, alone.
    fn weight(&self, item: &Self::Item) -> usize;

    /// Whether the item still wants to run; one that does not is skipped
    fn wanted(&self, _item: &Self::Item) -> bool {
        true
    }

    /// How long after the first item was queued a worker may wait for more; asked when a
    /// batch starts, so it may be learned
    fn window(&self) -> Duration;

    /// Run a batch; its results are the implementor's to deliver
    fn run(&self, batch: Vec<Queued<Self::Item>>);

    /// A batch of this total weight panicked, and whatever it would have delivered is lost.
    /// Called only where panics unwind; the panic hook has reported the panic itself. Does
    /// nothing by default.
    fn panicked(&self, _weight: usize) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Most weight in one batch
    pub max_batch: usize,
    /// Batches that may run at the same time; items are merged only while all of the
    /// workers are busy
    pub workers: usize,
    /// Items that may wait at once; beyond it [`Engine::submit`] fails. `None` queues
    /// without bound.
    pub capacity: Option<usize>,
    /// Whether a worker waits out the window with nothing queued and nothing announced
    pub waits_unannounced: bool,
}

/// An item and when it was queued
pub struct Queued<I> {
    pub item: I,
    pub queued_at: Instant,
}

enum Message<I> {
    Item {
        queued: Queued<I>,
        announced: bool,
    },
    /// An announced item gave up, so a worker may stop waiting for it
    Wake,
}

enum Sender<I> {
    Unbounded(mpsc::Sender<Message<I>>),
    Bounded(mpsc::SyncSender<Message<I>>),
}

impl<I> Sender<I> {
    /// The message comes back if the queue is full or the workers are gone
    fn send(&self, message: Message<I>) -> Result<(), Message<I>> {
        match self {
            Sender::Unbounded(sender) => sender.send(message).map_err(|e| e.0),
            Sender::Bounded(sender) => sender.try_send(message).map_err(|e| match e {
                TrySendError::Full(message) | TrySendError::Disconnected(message) => message,
            }),
        }
    }
}

/// Handle to the workers; they stop when it is dropped
pub struct Engine<I> {
    sender: Sender<I>,
    /// Items announced and not yet received by a worker
    arriving: Arc<AtomicUsize>,
    /// Items queued and not yet received by a worker
    waiting: Arc<AtomicUsize>,
    /// [`Config::capacity`], at least 1
    capacity: Option<usize>,
    /// A [`Message::Wake`] is queued and no worker has received it yet
    wake_queued: Arc<AtomicBool>,
}

impl<I: Send + 'static> Engine<I> {
    /// Start the workers, named by their index
    pub fn start<B: Batching<Item = I>>(
        batching: B,
        config: Config,
        thread_name: impl Fn(usize) -> String,
    ) -> Self {
        let (sender, receiver) = match config.capacity {
            None => {
                let (sender, receiver) = mpsc::channel();
                (Sender::Unbounded(sender), receiver)
            }
            Some(capacity) => {
                // Room for the items and the one wake-up that may be queued with them, so a
                // wake-up never takes an item's place
                let (sender, receiver) = mpsc::sync_channel(capacity.max(1) + 1);
                (Sender::Bounded(sender), receiver)
            }
        };
        let config = Config {
            workers: config.workers.max(1),
            max_batch: config.max_batch.max(1),
            ..config
        };
        let batching = Arc::new(batching);
        let queue = Arc::new(Mutex::new(Queue {
            receiver,
            carry: None,
        }));
        let arriving = Arc::new(AtomicUsize::new(0));
        let waiting = Arc::new(AtomicUsize::new(0));
        let busy = Arc::new(AtomicUsize::new(0));
        let wake_queued = Arc::new(AtomicBool::new(false));
        for i in 0..config.workers {
            let worker = Worker {
                batching: batching.clone(),
                config,
                queue: queue.clone(),
                arriving: arriving.clone(),
                waiting: waiting.clone(),
                busy: busy.clone(),
                wake_queued: wake_queued.clone(),
            };
            std::thread::Builder::new()
                .name(thread_name(i))
                .spawn(move || worker.run())
                .expect("failed to spawn a batching thread");
        }
        Self {
            sender,
            arriving,
            waiting,
            capacity: config.capacity.map(|c| c.max(1)),
            wake_queued,
        }
    }

    /// Announce an item that will probably be submitted soon: a worker collecting a batch
    /// waits for it. Hold the guard while preparing the item, then [`Engine::submit`] it.
    pub fn arrival(&self) -> Arrival<'_, I> {
        self.arriving.fetch_add(1, Ordering::SeqCst);
        Arrival {
            engine: self,
            submitted: false,
        }
    }

    /// Queue an item, announced by `arrival` or not. It comes back if the queue is full or
    /// the workers are gone. An announced item counts as arriving until a worker receives
    /// it, so a worker never waits for an item that is already in its hands.
    pub fn submit(&self, arrival: Option<Arrival<'_, I>>, item: I) -> Result<(), I> {
        let announced = arrival.is_some();
        let result = self.send(item, announced);
        if let Some(mut arrival) = arrival {
            arrival.submitted = result.is_ok();
        }
        result
    }

    fn send(&self, item: I, announced: bool) -> Result<(), I> {
        // Counted before it is sent, so a worker never takes an item that is not counted; the
        // count is also what bounds the queue
        let before = self.waiting.fetch_add(1, Ordering::Relaxed);
        if self.capacity.is_some_and(|capacity| before >= capacity) {
            self.waiting.fetch_sub(1, Ordering::Relaxed);
            return Err(item);
        }
        let message = Message::Item {
            queued: Queued {
                item,
                queued_at: Instant::now(),
            },
            announced,
        };
        match self.sender.send(message) {
            Ok(()) => Ok(()),
            Err(Message::Item { queued, .. }) => {
                self.waiting.fetch_sub(1, Ordering::Relaxed);
                Err(queued.item)
            }
            Err(Message::Wake) => unreachable!("an item was sent"),
        }
    }

    /// Items queued and not yet taken by a worker
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }
}

/// See [`Engine::arrival`]
pub struct Arrival<'a, I> {
    engine: &'a Engine<I>,
    submitted: bool,
}

impl<I> Drop for Arrival<'_, I> {
    fn drop(&mut self) {
        if !self.submitted {
            self.engine.arriving.fetch_sub(1, Ordering::SeqCst);
            // One wake-up queued is enough: the worker that receives it looks at what is still
            // arriving, this arrival's end included
            if !self.engine.wake_queued.swap(true, Ordering::SeqCst)
                && self.engine.sender.send(Message::Wake).is_err()
            {
                self.engine.wake_queued.store(false, Ordering::SeqCst);
            }
        }
    }
}

struct Worker<B: Batching> {
    batching: Arc<B>,
    config: Config,
    queue: Arc<Mutex<Queue<B::Item>>>,
    arriving: Arc<AtomicUsize>,
    waiting: Arc<AtomicUsize>,
    /// Workers running a batch
    busy: Arc<AtomicUsize>,
    /// See [`Engine::wake_queued`]
    wake_queued: Arc<AtomicBool>,
}

impl<B: Batching> Worker<B> {
    fn run(self) {
        loop {
            // One worker at a time collects a batch; the others are busy or wait their turn
            let batch = {
                let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                let batch = queue.next_batch(&self);
                // Counted before the next worker can look
                self.busy.fetch_add(1, Ordering::SeqCst);
                batch
            };
            match batch {
                Some(batch) => self.run_batch(batch),
                None => return,
            }
            self.busy.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn run_batch(&self, batch: Vec<Queued<B::Item>>) {
        let weight: usize = batch.iter().map(|q| self.batching.weight(&q.item)).sum();
        // Without unwinding (release builds abort on panic) this is a plain call. With it,
        // a panicking batch is dropped with whatever it would have delivered, and the
        // worker goes on.
        if catch_unwind(AssertUnwindSafe(|| self.batching.run(batch))).is_err() {
            self.batching.panicked(weight);
        }
    }

    /// A worker has the item now, so it no longer counts as waiting or arriving
    fn received(&self, queued: Queued<B::Item>, announced: bool) -> Queued<B::Item> {
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        if announced {
            self.arriving.fetch_sub(1, Ordering::SeqCst);
        }
        queued
    }
}

struct Queue<I> {
    receiver: mpsc::Receiver<Message<I>>,
    /// An item that did not fit into the previous batch; it starts the next one
    carry: Option<Queued<I>>,
}

impl<I> Queue<I> {
    /// Wait for the next batch; `None` once the [`Engine`] is gone
    fn next_batch<B: Batching<Item = I>>(&mut self, worker: &Worker<B>) -> Option<Vec<Queued<I>>> {
        let config = &worker.config;
        let first = loop {
            let queued = match self.carry.take() {
                Some(queued) => queued,
                None => match self.receiver.recv().ok()? {
                    Message::Item { queued, announced } => worker.received(queued, announced),
                    Message::Wake => {
                        worker.wake_queued.store(false, Ordering::SeqCst);
                        continue;
                    }
                },
            };
            if worker.batching.wanted(&queued.item) {
                break queued;
            }
        };

        // Another worker is idle: it takes the next item, and both run side by side
        if worker.busy.load(Ordering::SeqCst) + 1 < config.workers {
            return Some(vec![first]);
        }

        let deadline = first.queued_at + worker.batching.window();
        let mut weight = worker.batching.weight(&first.item);
        let mut batch = vec![first];
        while weight < config.max_batch {
            let message = match self.receiver.try_recv() {
                Ok(message) => message,
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {
                    // Nothing queued: wait only if something may still come
                    if !config.waits_unannounced && worker.arriving.load(Ordering::SeqCst) == 0 {
                        break;
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    match self.receiver.recv_timeout(deadline - now) {
                        Ok(message) => message,
                        Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                    }
                }
            };
            let Message::Item { queued, announced } = message else {
                // A wake-up: received before what is arriving is looked at again
                worker.wake_queued.store(false, Ordering::SeqCst);
                continue;
            };
            let queued = worker.received(queued, announced);
            if !worker.batching.wanted(&queued.item) {
                continue;
            }
            let item_weight = worker.batching.weight(&queued.item);
            if weight + item_weight > config.max_batch {
                self.carry = Some(queued);
                break;
            }
            weight += item_weight;
            batch.push(queued);
        }
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the batches it runs, as the items of each, and can be held so that items
    /// queue up behind the batch it is running
    struct Collect {
        window: Duration,
        hold: Arc<Mutex<()>>,
        batches: mpsc::Sender<Vec<u32>>,
    }

    impl Batching for Collect {
        type Item = u32;

        fn weight(&self, _: &u32) -> usize {
            1
        }

        fn window(&self) -> Duration {
            self.window
        }

        fn run(&self, batch: Vec<Queued<u32>>) {
            let items = batch.into_iter().map(|q| q.item).collect();
            let _ = self.batches.send(items);
            drop(self.hold.lock().unwrap_or_else(|e| e.into_inner()));
        }
    }

    struct Setup {
        engine: Engine<u32>,
        hold: Arc<Mutex<()>>,
        batches: mpsc::Receiver<Vec<u32>>,
    }

    fn setup(config: Config, window: Duration) -> Setup {
        let (batches, received) = mpsc::channel();
        let hold = Arc::new(Mutex::new(()));
        let engine = Engine::start(
            Collect {
                window,
                hold: hold.clone(),
                batches,
            },
            config,
            |i| format!("test-batching-{i}"),
        );
        Setup {
            engine,
            hold,
            batches: received,
        }
    }

    const ONE_WORKER: Config = Config {
        max_batch: 4,
        workers: 1,
        capacity: None,
        waits_unannounced: false,
    };

    #[test]
    fn waits_out_the_window_only_when_told_to() {
        // Told to, the worker that took the first item waits for the rest of the batch: with
        // a window of an hour, only a full batch ends the wait
        let s = setup(
            Config {
                waits_unannounced: true,
                ..ONE_WORKER
            },
            Duration::from_secs(3600),
        );
        s.engine.submit(None, 1).unwrap();
        wait_until_taken(&s.engine);
        for i in 2..=4 {
            s.engine.submit(None, i).unwrap();
        }
        assert_eq!(s.batches.recv().unwrap(), vec![1, 2, 3, 4]);

        // Otherwise it runs what it has at once, and the next item makes a batch of its own
        let s = setup(ONE_WORKER, Duration::from_secs(3600));
        let held = s.hold.lock().unwrap();
        s.engine.submit(None, 1).unwrap();
        assert_eq!(s.batches.recv().unwrap(), vec![1]);
        s.engine.submit(None, 2).unwrap();
        drop(held);
        assert_eq!(s.batches.recv().unwrap(), vec![2]);
    }

    /// Wait until a worker has taken what was submitted; fails the test after 10 seconds
    fn wait_until_taken(engine: &Engine<u32>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while engine.waiting() > 0 {
            assert!(Instant::now() < deadline, "not taken after 10 s");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_full_bounded_queue_returns_the_item() {
        let s = setup(
            Config {
                max_batch: 1,
                capacity: Some(2),
                ..ONE_WORKER
            },
            Duration::ZERO,
        );
        // The worker is held in its first batch, so the queue fills behind it
        let held = s.hold.lock().unwrap();
        s.engine.submit(None, 0).unwrap();
        assert_eq!(s.batches.recv().unwrap(), vec![0]);
        assert_eq!(s.engine.submit(None, 1), Ok(()));
        assert_eq!(s.engine.submit(None, 2), Ok(()));
        assert_eq!(s.engine.submit(None, 3), Err(3));
        assert_eq!(s.engine.waiting(), 2);
        drop(held);
        assert_eq!(s.batches.recv().unwrap(), vec![1]);
        assert_eq!(s.batches.recv().unwrap(), vec![2]);
        assert_eq!(s.engine.waiting(), 0);
    }

    #[test]
    fn abandoned_arrivals_do_not_take_the_queue_s_room() {
        let s = setup(
            Config {
                max_batch: 1,
                capacity: Some(2),
                ..ONE_WORKER
            },
            Duration::ZERO,
        );
        let held = s.hold.lock().unwrap();
        s.engine.submit(None, 0).unwrap();
        assert_eq!(s.batches.recv().unwrap(), vec![0]);
        // While the worker is busy, more arrivals give up than the queue has room for; each
        // would have queued a wake-up
        for _ in 0..5 {
            drop(s.engine.arrival());
        }
        // The queue still holds as many items as it was made for, and no more
        assert_eq!(s.engine.submit(None, 1), Ok(()));
        let arrival = s.engine.arrival();
        assert_eq!(s.engine.submit(Some(arrival), 2), Ok(()));
        assert_eq!(s.engine.submit(None, 3), Err(3));
        assert_eq!(s.engine.waiting(), 2);
        drop(held);
        assert_eq!(s.batches.recv().unwrap(), vec![1]);
        assert_eq!(s.batches.recv().unwrap(), vec![2]);
        // A worker that received the wake-up lets the next one be queued
        drop(s.engine.arrival());
        s.engine.submit(None, 4).unwrap();
        assert_eq!(s.batches.recv().unwrap(), vec![4]);
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        /// An item: its id, its weight, whether it is still wanted, and whether its batch
        /// panics
        #[derive(Clone, Copy, Debug)]
        struct Item {
            id: usize,
            weight: usize,
            wanted: bool,
            poison: bool,
        }

        /// What a batch ran with, and whether it panicked
        enum Ran {
            Batch(Vec<Item>),
            Panicked(usize),
        }

        /// Sends every batch it runs; a batch with a poisoned item panics after that
        struct Record(mpsc::Sender<Ran>);

        impl Batching for Record {
            type Item = Item;

            fn weight(&self, item: &Item) -> usize {
                item.weight
            }

            fn wanted(&self, item: &Item) -> bool {
                item.wanted
            }

            fn window(&self) -> Duration {
                Duration::ZERO
            }

            fn run(&self, batch: Vec<Queued<Item>>) {
                let items: Vec<Item> = batch.into_iter().map(|q| q.item).collect();
                let poisoned = items.iter().any(|item| item.poison);
                let _ = self.0.send(Ran::Batch(items));
                assert!(!poisoned, "a poisoned batch");
            }

            fn panicked(&self, weight: usize) {
                let _ = self.0.send(Ran::Panicked(weight));
            }
        }

        /// How each item is handed in
        #[derive(Clone, Copy, Debug)]
        enum Handed {
            Plain,
            /// Announced first, then submitted
            Announced,
            /// Announced, then given up on before it is submitted, then submitted plainly
            GivenUp,
        }

        fn items() -> impl Strategy<Value = Vec<(Item, Handed)>> {
            prop::collection::vec(
                (
                    1usize..6,
                    prop::bool::weighted(0.85),
                    prop::bool::weighted(0.05),
                    prop::sample::select(vec![Handed::Plain, Handed::Announced, Handed::GivenUp]),
                ),
                0..120,
            )
            .prop_map(|items| {
                items
                    .into_iter()
                    .enumerate()
                    .map(|(id, (weight, wanted, poison, handed))| {
                        let item = Item {
                            id,
                            weight,
                            wanted,
                            poison,
                        };
                        (item, handed)
                    })
                    .collect()
            })
        }

        fn config() -> impl Strategy<Value = Config> {
            (1usize..10, 1usize..4, prop::option::of(1usize..40)).prop_map(
                |(max_batch, workers, capacity)| Config {
                    max_batch,
                    workers,
                    capacity,
                    waits_unannounced: false,
                },
            )
        }

        proptest! {
            // At most 64 cases; PROPTEST_CASES lowers it
            #![proptest_config(ProptestConfig::with_cases(
                ProptestConfig::default().cases.min(64)
            ))]

            /// Every wanted item that was accepted runs in exactly one batch, unwanted ones
            /// never run, a batch stays within the maximum weight unless it is one heavy item,
            /// and a panicking batch costs only itself
            #[test]
            fn batches_lose_and_repeat_nothing(items in items(), config in config()) {
                let (sender, received) = mpsc::channel();
                let engine = Engine::start(Record(sender), config, |i| format!("prop-{i}"));
                let mut refused = Vec::new();
                for &(item, handed) in &items {
                    let result = match handed {
                        Handed::Plain => engine.submit(None, item),
                        Handed::Announced => engine.submit(Some(engine.arrival()), item),
                        Handed::GivenUp => {
                            drop(engine.arrival());
                            engine.submit(None, item)
                        }
                    };
                    if let Err(item) = result {
                        refused.push(item.id);
                    }
                }
                prop_assert!(engine.waiting() <= config.capacity.unwrap_or(usize::MAX));
                if config.capacity.is_none() {
                    prop_assert!(refused.is_empty());
                }
                // The workers run what was queued, then stop, and the recorder with them
                drop(engine);
                let mut ran = Vec::new();
                let mut batches = Vec::new();
                let mut panicked = Vec::new();
                for record in received {
                    match record {
                        Ran::Batch(batch) => {
                            ran.extend(batch.iter().map(|item| item.id));
                            batches.push(batch);
                        }
                        Ran::Panicked(weight) => panicked.push(weight),
                    }
                }
                for batch in &batches {
                    let weight: usize = batch.iter().map(|item| item.weight).sum();
                    prop_assert!(
                        weight <= config.max_batch || batch.len() == 1,
                        "{} > {}", weight, config.max_batch
                    );
                    prop_assert!(batch.iter().all(|item| item.wanted));
                }
                let mut poisoned: Vec<usize> = batches
                    .iter()
                    .filter(|batch| batch.iter().any(|item| item.poison))
                    .map(|batch| batch.iter().map(|item| item.weight).sum())
                    .collect();
                poisoned.sort_unstable();
                panicked.sort_unstable();
                prop_assert_eq!(poisoned, panicked);

                let mut expected: Vec<usize> = items
                    .iter()
                    .filter(|(item, _)| item.wanted && !refused.contains(&item.id))
                    .map(|(item, _)| item.id)
                    .collect();
                // One worker runs them in the order they came
                if config.workers == 1 {
                    prop_assert_eq!(&ran, &expected);
                }
                ran.sort_unstable();
                expected.sort_unstable();
                prop_assert_eq!(ran, expected);
            }
        }
    }
}
