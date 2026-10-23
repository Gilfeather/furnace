//! The threads, locks, atomics and channels the engine is made of: the standard library's, or,
//! when amate's own tests are built to model check the engine (`RUSTFLAGS="--cfg loom"`),
//! loom's, so that every interleaving of the workers and the submitters is tried. As a
//! dependency of a crate built that way, amate stays on the standard library's: loom is only
//! its dev-dependency.

#[cfg(not(all(loom, test)))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(not(all(loom, test)))]
pub(crate) use std::sync::{Arc, Mutex, mpsc};
#[cfg(not(all(loom, test)))]
pub(crate) use std::thread;

#[cfg(all(loom, test))]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(all(loom, test))]
pub(crate) use loom::sync::{Arc, Mutex};
#[cfg(all(loom, test))]
pub(crate) use loom::thread;

/// `std::sync::mpsc` as the engine uses it, on loom's mutex and condition variable: loom has
/// no bounded channel and no `recv_timeout`. Time does not pass in a model, so
/// `recv_timeout` waits as `recv` does: a worker that would miss a wake-up deadlocks, and
/// loom reports it.
#[cfg(all(loom, test))]
pub(crate) mod mpsc {
    use std::collections::VecDeque;
    use std::time::Duration;

    pub(crate) use std::sync::mpsc::{
        RecvError, RecvTimeoutError, SendError, TryRecvError, TrySendError,
    };

    use loom::sync::{Arc, Condvar, Mutex, MutexGuard};

    struct State<T> {
        queue: VecDeque<T>,
        /// The most messages queued at once; `None` without bound
        bound: Option<usize>,
        sender: bool,
        receiver: bool,
    }

    struct Shared<T> {
        state: Mutex<State<T>>,
        ready: Condvar,
    }

    impl<T> Shared<T> {
        fn state(&self) -> MutexGuard<'_, State<T>> {
            self.state.lock().unwrap_or_else(|e| e.into_inner())
        }
    }

    fn with_bound<T>(bound: Option<usize>) -> Arc<Shared<T>> {
        Arc::new(Shared {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                bound,
                sender: true,
                receiver: true,
            }),
            ready: Condvar::new(),
        })
    }

    pub(crate) fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let shared = with_bound(None);
        (Sender(shared.clone()), Receiver(shared))
    }

    pub(crate) fn sync_channel<T>(bound: usize) -> (SyncSender<T>, Receiver<T>) {
        let shared = with_bound(Some(bound));
        (SyncSender(shared.clone()), Receiver(shared))
    }

    /// The one sender: the engine's are never cloned
    pub(crate) struct Sender<T>(Arc<Shared<T>>);
    pub(crate) struct SyncSender<T>(Arc<Shared<T>>);
    pub(crate) struct Receiver<T>(Arc<Shared<T>>);

    fn push<T>(shared: &Shared<T>, message: T) -> Result<(), TrySendError<T>> {
        let mut state = shared.state();
        if !state.receiver {
            return Err(TrySendError::Disconnected(message));
        }
        if state.bound.is_some_and(|bound| state.queue.len() >= bound) {
            return Err(TrySendError::Full(message));
        }
        state.queue.push_back(message);
        drop(state);
        shared.ready.notify_one();
        Ok(())
    }

    impl<T> Sender<T> {
        pub(crate) fn send(&self, message: T) -> Result<(), SendError<T>> {
            push(&self.0, message).map_err(|e| match e {
                TrySendError::Full(message) | TrySendError::Disconnected(message) => {
                    SendError(message)
                }
            })
        }
    }

    impl<T> SyncSender<T> {
        pub(crate) fn try_send(&self, message: T) -> Result<(), TrySendError<T>> {
            push(&self.0, message)
        }
    }

    fn hang_up<T>(shared: &Shared<T>) {
        shared.state().sender = false;
        shared.ready.notify_all();
    }

    impl<T> Drop for Sender<T> {
        fn drop(&mut self) {
            hang_up(&self.0);
        }
    }

    impl<T> Drop for SyncSender<T> {
        fn drop(&mut self) {
            hang_up(&self.0);
        }
    }

    impl<T> Drop for Receiver<T> {
        fn drop(&mut self) {
            self.0.state().receiver = false;
        }
    }

    impl<T> Receiver<T> {
        pub(crate) fn recv(&self) -> Result<T, RecvError> {
            let mut state = self.0.state();
            loop {
                if let Some(message) = state.queue.pop_front() {
                    return Ok(message);
                }
                if !state.sender {
                    return Err(RecvError);
                }
                state = self.0.ready.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }

        /// [`recv`](Self::recv): time does not pass in a model
        pub(crate) fn recv_timeout(&self, _: Duration) -> Result<T, RecvTimeoutError> {
            self.recv()
                .map_err(|RecvError| RecvTimeoutError::Disconnected)
        }

        pub(crate) fn try_recv(&self) -> Result<T, TryRecvError> {
            let mut state = self.0.state();
            match state.queue.pop_front() {
                Some(message) => Ok(message),
                None if state.sender => Err(TryRecvError::Empty),
                None => Err(TryRecvError::Disconnected),
            }
        }
    }
}
