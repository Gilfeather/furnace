//! Model checks of a nerve's lane: every interleaving of the threads entering it and freeing
//! its places, with loom (`RUSTFLAGS="--cfg furnace_loom" cargo test --lib model_check`).

use std::time::{Duration, Instant};

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

use super::*;

/// A lane and what became of what entered it, shared by the model's threads
struct Body {
    lane: Lane<u32>,
    limit: usize,
    /// Carried out at once, now
    running: AtomicUsize,
    carried: Mutex<Vec<u32>>,
    /// Dropped: the lane was full, the deadline passed or nobody wanted it any more
    gone: Mutex<Vec<u32>>,
    /// Items nobody wants any more
    unwanted: Mutex<Vec<u32>>,
}

impl Body {
    fn new(limit: usize, room: usize) -> Arc<Self> {
        Arc::new(Self {
            lane: Lane::with_room(Nerve::new("arm").in_flight(limit), room),
            limit,
            running: AtomicUsize::new(0),
            carried: Mutex::new(Vec::new()),
            gone: Mutex::new(Vec::new()),
            unwanted: Mutex::new(Vec::new()),
        })
    }

    fn enter(&self, item: u32, deadline: Option<Instant>) {
        match self.lane.enter(deadline, item) {
            Entered::Go(item) => self.carry(item),
            Entered::Waits { dropped } => self.gone.lock().unwrap().extend(dropped),
            Entered::Full(item) => self.gone.lock().unwrap().push(item),
        }
    }

    /// Carry `item` out in its place, then free the place: what takes it is carried out
    /// next, on this thread, as the fabric's runtime would
    fn carry(&self, item: u32) {
        let mut item = Some(item);
        while let Some(now) = item.take() {
            let at_once = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            assert!(at_once <= self.limit, "{at_once} carried out at once");
            self.carried.lock().unwrap().push(now);
            self.running.fetch_sub(1, Ordering::SeqCst);
            item = self.release();
        }
    }

    fn release(&self) -> Option<u32> {
        let unwanted = self.unwanted.lock().unwrap().clone();
        let released = self
            .lane
            .release(Instant::now(), |item| !unwanted.contains(item));
        let mut gone = self.gone.lock().unwrap();
        gone.extend(released.expired);
        gone.extend(released.unwanted);
        released.next
    }

    /// Everything that entered, carried out or dropped, each exactly once, and the lane
    /// empty and idle
    fn settled(&self, entered: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let carried = self.carried.lock().unwrap().clone();
        let gone = self.gone.lock().unwrap().clone();
        let mut all: Vec<u32> = carried.iter().chain(&gone).copied().collect();
        all.sort_unstable();
        assert_eq!(all, entered, "carried {carried:?}, dropped {gone:?}");
        let metrics = self.lane.metrics();
        assert_eq!((metrics.in_flight, metrics.waiting), (0, 0));
        (carried, gone)
    }
}

fn on_thread(body: &Arc<Body>, f: impl FnOnce(&Body) + 'static) -> thread::JoinHandle<()> {
    let body = body.clone();
    thread::spawn(move || f(&body))
}

/// Entering and freeing places at once: never more carried out than the lane's places, and
/// nothing left waiting once every place is free (no place freed without the next taking it)
#[test]
fn the_places_hold_and_nothing_is_stranded() {
    for limit in [1, 2] {
        loom::model(move || {
            let body = Body::new(limit, 8);
            let threads: Vec<_> = [1, 2]
                .into_iter()
                .map(|item| on_thread(&body, move |body| body.enter(item, None)))
                .collect();
            body.enter(3, None);
            for thread in threads {
                thread.join().unwrap();
            }
            let (carried, gone) = body.settled(&[1, 2, 3]);
            assert_eq!(carried.len(), 3);
            assert!(gone.is_empty());
        });
    }
}

/// A full lane drops the least urgent, whichever comes first: the newcomer that is less
/// urgent than all waiting, or the one waiting that a more urgent newcomer pushes out. The
/// more urgent one is carried out once the place is free.
#[test]
fn a_full_lane_drops_the_least_urgent() {
    loom::model(|| {
        let body = Body::new(1, 1);
        let Entered::Go(first) = body.lane.enter(None, 0) else {
            panic!("an idle lane lets it on");
        };
        let now = Instant::now();
        let hour = Duration::from_secs(3600);
        let later = on_thread(&body, move |body| body.enter(1, Some(now + 2 * hour)));
        let sooner = on_thread(&body, move |body| body.enter(2, Some(now + hour)));
        later.join().unwrap();
        sooner.join().unwrap();
        body.carry(first);
        let (carried, gone) = body.settled(&[0, 1, 2]);
        assert_eq!(carried, [0, 2]);
        assert_eq!(gone, [1]);
        assert_eq!(body.lane.metrics().dropped, 1);
    });
}

/// A waiting item given up on while its place frees: it is carried out or dropped, once, and
/// the place is free afterwards either way
#[test]
fn a_cancelled_item_is_carried_or_dropped_once() {
    loom::model(|| {
        let body = Body::new(1, 4);
        let Entered::Go(first) = body.lane.enter(None, 0) else {
            panic!("an idle lane lets it on");
        };
        let waiter = on_thread(&body, |body| body.enter(1, None));
        let canceller = on_thread(&body, |body| body.unwanted.lock().unwrap().push(1));
        body.carry(first);
        waiter.join().unwrap();
        canceller.join().unwrap();
        body.settled(&[0, 1]);
    });
}
