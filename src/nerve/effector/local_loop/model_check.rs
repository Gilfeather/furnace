//! Model checks of an effector's local loop: every interleaving of commands admitted at once
//! and the ends of those already out, with loom
//! (`RUSTFLAGS="--cfg furnace_loom" cargo test --lib model_check`).

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering as Seq};
use loom::thread;

use super::super::{Act, Effector};
use super::*;

const HOUR: Duration = Duration::from_secs(3600);

struct Idle;

impl Act for Idle {
    fn name(&self) -> &str {
        "idle"
    }

    fn act(&self, _: &Command, _: &Signal) -> Action {
        Action::ok()
    }
}

fn admitted(admission: Admission) -> bool {
    matches!(admission, Admission::After(_))
}

/// Two commands at the same moment, in the refractory period of none: one is admitted, the
/// other ignored
#[test]
fn the_refractory_period_lets_one_of_two_through() {
    loom::model(|| {
        let effector = Effector::new(Idle).refractory(HOUR);
        let now = Instant::now();
        let other = {
            let effector = effector.clone();
            thread::spawn(move || admitted(effector.inner.admit(now)))
        };
        let here = admitted(effector.inner.admit(now));
        let there = other.join().unwrap();
        assert!(here != there, "admitted here {here}, there {there}");
        assert_eq!(effector.metrics().quiet_refractory, 1);
    });
}

/// After the cooldown, commands at once while trials end (given up on): never two trials out
/// at once, and the circuit never stays half open with no trial out
#[test]
fn half_open_has_one_trial_out_at_a_time() {
    loom::model(|| {
        let effector = Effector::new(Idle).breaker(Breaker {
            failures: 1,
            cooldown: HOUR,
        });
        let t0 = Instant::now();
        assert!(admitted(effector.inner.admit(t0)));
        effector.inner.failed(&ActError::fatal("down"));
        let later = t0 + 2 * HOUR;
        let out = Arc::new(AtomicUsize::new(0));
        let try_once = {
            let effector = effector.clone();
            let out = out.clone();
            move || {
                if admitted(effector.inner.admit(later)) {
                    assert_eq!(out.fetch_add(1, Seq::SeqCst), 0, "two trials out");
                    out.fetch_sub(1, Seq::SeqCst);
                    // Its commander stops waiting before it is tried
                    effector.inner.abandoned();
                }
            }
        };
        let other = thread::spawn(try_once.clone());
        try_once();
        other.join().unwrap();
        // No trial is out: the next command is one
        assert!(admitted(effector.inner.admit(later)));
    });
}
