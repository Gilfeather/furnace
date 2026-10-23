//! The locks and atomics of the nerve's concurrent state machines (a nerve's lane, an
//! effector's local loop, a motor neuron's program): the standard library's, or, when the
//! crate's own tests are built to model check them (`RUSTFLAGS="--cfg furnace_loom"`),
//! loom's, so that every interleaving of the threads that share them is tried. Only the
//! `model_check` tests run then: elsewhere these would be loom's outside a model.

#[cfg(not(all(furnace_loom, test)))]
pub(crate) use std::sync::atomic::AtomicU64;
#[cfg(not(all(furnace_loom, test)))]
pub(crate) use std::sync::{Mutex, MutexGuard};

#[cfg(all(furnace_loom, test))]
pub(crate) use loom::sync::atomic::AtomicU64;
#[cfg(all(furnace_loom, test))]
pub(crate) use loom::sync::{Mutex, MutexGuard};
