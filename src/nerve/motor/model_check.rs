//! Model checks of a motor neuron's program slot: every interleaving of commands preempting
//! each other, programs ending and stops, with loom
//! (`RUSTFLAGS="--cfg furnace_loom" cargo test --lib model_check`).

use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, Ordering as Seq};
use loom::thread;

use super::*;

/// A program's way to stop: raises its flag when dropped
struct Stop(Arc<AtomicBool>);

impl Drop for Stop {
    fn drop(&mut self) {
        assert!(!self.0.swap(true, Seq::SeqCst), "stopped twice");
    }
}

fn program() -> (Stop, Arc<AtomicBool>) {
    let stopped = Arc::new(AtomicBool::new(false));
    (Stop(stopped.clone()), stopped)
}

/// A program that starts and ends, racing a newer command: the one that took the slot last
/// runs, the other is stopped, and a program that ends never clears a newer one's slot
#[test]
fn the_newest_command_runs_and_an_older_end_never_clears_it() {
    loom::model(|| {
        let slot = Arc::new(Slot::new());
        let (a, a_stopped) = program();
        let (b, b_stopped) = program();
        let older = {
            let slot = slot.clone();
            thread::spawn(move || {
                let generation = slot.start(a);
                slot.end(generation);
            })
        };
        slot.start(b);
        older.join().unwrap();
        // `a` ended or was preempted: its stop is gone either way. `b` runs unless `a`
        // preempted it, and the slot says so.
        assert!(a_stopped.load(Seq::SeqCst));
        assert_eq!(slot.is_running(), !b_stopped.load(Seq::SeqCst));
    });
}

/// Two commands and a stop at once: at most one program is left running, and once it is
/// stopped too, every program's stop has been dropped exactly once
#[test]
fn a_stop_racing_two_commands_leaves_at_most_one_running() {
    loom::model(|| {
        let slot = Arc::new(Slot::new());
        let (a, a_stopped) = program();
        let (b, b_stopped) = program();
        let first = {
            let slot = slot.clone();
            thread::spawn(move || {
                slot.start(a);
            })
        };
        let stopper = {
            let slot = slot.clone();
            thread::spawn(move || {
                slot.stop();
            })
        };
        slot.start(b);
        first.join().unwrap();
        stopper.join().unwrap();
        let alive = [&a_stopped, &b_stopped]
            .iter()
            .filter(|stopped| !stopped.load(Seq::SeqCst))
            .count();
        assert_eq!(alive, usize::from(slot.is_running()));
        assert!(alive <= 1);
        slot.stop();
        assert!(a_stopped.load(Seq::SeqCst) && b_stopped.load(Seq::SeqCst));
    });
}
