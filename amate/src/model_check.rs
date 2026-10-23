//! Model checks: every interleaving of the workers, the submitters and the arrivals, with
//! loom (`RUSTFLAGS="--cfg loom" cargo test -p amate --release`). Time does not pass in a
//! model, so a worker waiting for an announced item waits until something wakes it: a
//! wake-up it would miss is a deadlock, which loom reports.

use std::time::Duration;

use loom::sync::{Condvar, Mutex};

use super::*;

/// Records the items of each batch it runs; an item weighs its own value
struct Record(Arc<Ran>);

#[derive(Default)]
struct Ran {
    batches: Mutex<Vec<Vec<u32>>>,
    more: Condvar,
}

impl Ran {
    /// Wait until `n` items have run, and return the batches
    fn wait_for(&self, n: usize) -> Vec<Vec<u32>> {
        let mut batches = self.batches.lock().unwrap();
        while batches.iter().map(Vec::len).sum::<usize>() < n {
            batches = self.more.wait(batches).unwrap();
        }
        batches.clone()
    }
}

impl Batching for Record {
    type Item = u32;

    fn weight(&self, item: &u32) -> usize {
        *item as usize
    }

    fn window(&self) -> Duration {
        // Longer than any model: only an item or a wake-up ends a worker's wait
        Duration::from_secs(3600)
    }

    fn run(&self, batch: Vec<Queued<u32>>) {
        let items = batch.into_iter().map(|q| q.item).collect();
        self.0.batches.lock().unwrap().push(items);
        self.0.more.notify_all();
    }
}

fn start(config: Config) -> (Arc<Engine<u32>>, Arc<Ran>) {
    let ran = Arc::new(Ran::default());
    let engine = Engine::start(Record(ran.clone()), config, |i| format!("model-{i}"));
    (Arc::new(engine), ran)
}

/// Every item once, whatever the batches
fn items(batches: &[Vec<u32>]) -> Vec<u32> {
    let mut items: Vec<u32> = batches.iter().flatten().copied().collect();
    items.sort_unstable();
    items
}

fn on_thread(
    engine: &Arc<Engine<u32>>,
    f: impl FnOnce(&Engine<u32>) + 'static,
) -> thread::JoinHandle<()> {
    let engine = engine.clone();
    thread::spawn(move || f(&engine))
}

const ONE_WORKER: Config = Config {
    max_batch: 3,
    workers: 1,
    capacity: None,
    waits_unannounced: false,
};

/// A worker collecting a batch waits for an announced item and wakes when it is given up:
/// no wake-up is lost, whenever the arrival ends, and no item is lost or run twice
#[test]
fn a_given_up_arrival_wakes_the_collecting_worker() {
    loom::model(|| {
        let (engine, ran) = start(ONE_WORKER);
        let submitted = on_thread(&engine, |engine| {
            engine.submit(Some(engine.arrival()), 1).unwrap();
        });
        let given_up = on_thread(&engine, |engine| drop(engine.arrival()));
        submitted.join().unwrap();
        given_up.join().unwrap();
        // The engine is still up: a worker stuck waiting for the arrival that gave up never
        // runs the item, and this wait deadlocks
        let batches = ran.wait_for(1);
        assert_eq!(items(&batches), [1]);
        assert_eq!(engine.waiting(), 0);
    });
}

/// Two arrivals, each submitted or given up, race the worker taking the first item and
/// waiting for the rest: everything accepted runs once, in batches within the maximum
#[test]
fn announced_items_join_the_batch_or_the_next() {
    loom::model(|| {
        let (engine, ran) = start(ONE_WORKER);
        let first = on_thread(&engine, |engine| {
            engine.submit(Some(engine.arrival()), 1).unwrap();
        });
        let second = on_thread(&engine, |engine| {
            let arrival = engine.arrival();
            engine.submit(Some(arrival), 2).unwrap();
            drop(engine.arrival());
        });
        first.join().unwrap();
        second.join().unwrap();
        let batches = ran.wait_for(2);
        assert_eq!(items(&batches), [1, 2]);
        assert!(batches.iter().all(|b| b.iter().sum::<u32>() <= 3));
    });
}

/// The bounded queue's room is the items': wake-ups of arrivals that gave up never take it,
/// so the one item submitted always gets in, and the queue never holds more than its capacity
#[test]
fn wake_ups_never_take_an_item_s_room() {
    loom::model(|| {
        let (engine, ran) = start(Config {
            capacity: Some(1),
            ..ONE_WORKER
        });
        let given_up = on_thread(&engine, |engine| {
            drop(engine.arrival());
            drop(engine.arrival());
        });
        assert_eq!(engine.submit(None, 1), Ok(()));
        assert!(engine.waiting() <= 1);
        given_up.join().unwrap();
        assert_eq!(items(&ran.wait_for(1)), [1]);
    });
}

/// Submitters racing for the last place: the queue holds at most its capacity, an item
/// refused comes back, and every item accepted runs exactly once
#[test]
fn a_full_queue_refuses_and_loses_nothing() {
    loom::model(|| {
        let (engine, ran) = start(Config {
            max_batch: 1,
            capacity: Some(1),
            ..ONE_WORKER
        });
        let accepted = Arc::new(AtomicUsize::new(0));
        let racer = {
            let accepted = accepted.clone();
            on_thread(&engine, move |engine| match engine.submit(None, 1) {
                Ok(()) => {
                    accepted.fetch_add(1, Ordering::SeqCst);
                }
                Err(item) => assert_eq!(item, 1),
            })
        };
        match engine.submit(None, 1) {
            Ok(()) => {
                accepted.fetch_add(1, Ordering::SeqCst);
            }
            Err(item) => assert_eq!(item, 1),
        }
        assert!(engine.waiting() <= 1);
        racer.join().unwrap();
        let accepted = accepted.load(Ordering::SeqCst);
        // The first in always fits
        assert!(accepted >= 1);
        assert_eq!(items(&ran.wait_for(accepted)), vec![1; accepted]);
    });
}

/// Two workers take turns at the queue while items are submitted and the engine is dropped:
/// the workers run everything queued (an item that did not fit a batch included), then stop
#[test]
fn shutdown_runs_everything_queued() {
    loom::model(|| {
        let ran = Arc::new(Ran::default());
        let engine = Engine::start(
            Record(ran.clone()),
            Config {
                max_batch: 2,
                workers: 2,
                ..ONE_WORKER
            },
            |i| format!("model-{i}"),
        );
        // 1 then 2 do not fit one batch of at most 2: the 2 is carried to the next
        engine.submit(None, 1).unwrap();
        engine.submit(None, 2).unwrap();
        drop(engine);
        let batches = ran.wait_for(2);
        assert_eq!(items(&batches), [1, 2]);
        assert!(batches.iter().all(|b| b.iter().sum::<u32>() <= 2));
    });
}
