//! Reusable `f32` work buffers for the convolution kernels.
//!
//! Large allocations are expensive on some platforms (Windows returns big blocks to the OS on
//! free and page-faults them back in on the next allocation), so buffers that only live for
//! one operation are kept in a small pool instead of being freed.

use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use std::sync::Mutex;

/// Buffers kept for reuse; more than this are freed
const POOL_SIZE: usize = 4;

static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());

/// A work buffer of `f32`s, returned to the pool on drop. The vector keeps its full length so
/// reuse never zeroes memory again; `len` is the part handed out.
pub(crate) struct Scratch(Vec<f32>, usize);

impl Scratch {
    /// A buffer of `len` elements. Its contents are unspecified (left over from earlier use),
    /// so callers must write every element they read.
    pub(crate) fn take(len: usize) -> Self {
        let reused = {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            // The smallest buffer that fits, else the largest one to grow
            let pick = pool
                .iter()
                .enumerate()
                .filter(|(_, b)| b.capacity() >= len)
                .min_by_key(|(_, b)| b.capacity())
                .or_else(|| pool.iter().enumerate().max_by_key(|(_, b)| b.capacity()))
                .map(|(i, _)| i);
            pick.map(|i| pool.swap_remove(i))
        };
        let mut buffer = reused.unwrap_or_default();
        if buffer.len() < len {
            buffer.resize(len, 0.0);
        }
        Self(buffer, len)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let buffer = core::mem::take(&mut self.0);
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < POOL_SIZE {
            pool.push(buffer);
        }
    }
}

impl Deref for Scratch {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.0[..self.1]
    }
}

impl DerefMut for Scratch {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0[..self.1]
    }
}
