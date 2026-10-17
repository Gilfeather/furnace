//! Weights kept in transformed layouts (such as Winograd filters) across calls.
//!
//! An entry is keyed by the identity of the tensor's storage (`Arc<Bytes>`), its layout and the
//! kind of transform, and holds only a [`Weak`] reference to that storage. This is enough to
//! never serve a stale transform:
//!
//! - Storage is only mutated through `Arc::make_mut` or `Arc::get_mut`. With a `Weak` present,
//!   `get_mut` refuses and `make_mut` moves the data to a new allocation and disassociates the
//!   `Weak`, so mutated storage gets a new identity and the old entry can no longer be upgraded.
//! - The `Weak` keeps the old allocation's control block alive, so its address cannot be
//!   reused by different storage while the entry exists.
//!
//! Entries whose storage is gone are dropped on the next lookup, and the cache drops its oldest
//! entries beyond [`MAX_BYTES`].

use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::any::Any;
use std::sync::Mutex;

use burn_std::Bytes;

use crate::FlexTensor;

/// Upper bound on the transformed data kept, in bytes
const MAX_BYTES: usize = 1 << 30;

#[derive(PartialEq, Eq)]
struct Key {
    storage: usize,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
    kind: &'static str,
}

struct Entry {
    key: Key,
    storage: Weak<Bytes>,
    value: Arc<dyn Any + Send + Sync>,
    bytes: usize,
}

static CACHE: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// The `kind` transform of `tensor`, from the cache or made by `make`, which returns the
/// transformed value and its size in bytes.
pub(crate) fn get_or_insert<T: Any + Send + Sync>(
    tensor: &FlexTensor,
    kind: &'static str,
    make: impl FnOnce() -> (T, usize),
) -> Arc<T> {
    let storage = tensor.data_arc();
    let layout = tensor.layout();
    let key = Key {
        storage: Arc::as_ptr(&storage) as usize,
        offset: layout.start_offset(),
        shape: layout.shape().to_vec(),
        strides: layout.strides().to_vec(),
        kind,
    };

    {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.retain(|entry| entry.storage.strong_count() > 0);
        let hit = cache.iter().find(|entry| {
            entry.key == key
                && entry
                    .storage
                    .upgrade()
                    .is_some_and(|live| Arc::ptr_eq(&live, &storage))
        });
        if let Some(entry) = hit
            && let Ok(value) = Arc::clone(&entry.value).downcast::<T>()
        {
            return value;
        }
    }

    // Transform outside the lock; two threads racing on a miss both compute it
    let (value, bytes) = make();
    let value = Arc::new(value);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.push(Entry {
        key,
        storage: Arc::downgrade(&storage),
        value: Arc::clone(&value) as Arc<dyn Any + Send + Sync>,
        bytes,
    });
    let mut total: usize = cache.iter().map(|entry| entry.bytes).sum();
    while total > MAX_BYTES && cache.len() > 1 {
        total -= cache.remove(0).bytes;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn reuses_until_the_storage_changes() {
        let mut tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let calls = core::cell::Cell::new(0);
        let sum = |t: &FlexTensor| {
            get_or_insert(t, "test-sum", || {
                calls.set(calls.get() + 1);
                (t.storage::<f32>().iter().sum::<f32>(), 4)
            })
        };
        assert_eq!(*sum(&tensor), 6.0);
        assert_eq!(*sum(&tensor.clone()), 6.0);
        assert_eq!(calls.get(), 1, "a clone shares the storage");

        // In-place mutation of the (unique) tensor must not serve the stale sum
        tensor.storage_mut::<f32>()[0] = 10.0;
        assert_eq!(*sum(&tensor), 15.0);
        assert_eq!(calls.get(), 2);
    }
}
