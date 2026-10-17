//! A global allocator that keeps freed large blocks for reuse.
//!
//! Every operator of a model allocates its output, and a ResNet-18 forward pass makes dozens of
//! allocations between 100 KB and a few MB. The system allocators hand blocks that large
//! straight back to the OS on free (Windows' heap past its threshold, glibc past
//! `M_MMAP_THRESHOLD`), so the next pass pays for fresh pages again: a page fault per 4 KiB on
//! first touch, plus zeroing. [`BlockCache`] keeps up to [`MAX_CACHED`] bytes of freed large
//! blocks and hands them out again to allocations of the same size and alignment, which a
//! repeated forward pass makes in the same order. On a 16-thread laptop this cut ResNet-18
//! batch-1 latency by about 15%.
//!
//! Small allocations go straight to [`System`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;

/// Smallest allocation whose block is kept on free
const MIN_SIZE: usize = 256 * 1024;

/// Most freed blocks kept at once
const SLOTS: usize = 64;

/// Most bytes kept in freed blocks at once
pub const MAX_CACHED: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Block {
    ptr: usize,
    size: usize,
    align: usize,
}

struct Cache {
    blocks: [Option<Block>; SLOTS],
    bytes: usize,
}

/// [`System`] plus a cache of freed large blocks; see the module docs.
///
/// ```no_run
/// #[global_allocator]
/// static ALLOCATOR: furnace::allocator::BlockCache = furnace::allocator::BlockCache::new();
/// ```
pub struct BlockCache {
    // `std::sync::Mutex` does not allocate (SRW lock on Windows, futex on Linux), so it can be
    // used inside the allocator
    cache: Mutex<Cache>,
}

impl BlockCache {
    pub const fn new() -> Self {
        Self {
            cache: Mutex::new(Cache {
                blocks: [None; SLOTS],
                bytes: 0,
            }),
        }
    }

    /// A cached block of exactly `layout`, if any
    fn take(&self, layout: Layout) -> Option<*mut u8> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let slot = cache.blocks.iter_mut().find(|slot| {
            slot.is_some_and(|b| b.size == layout.size() && b.align == layout.align())
        })?;
        let block = slot.take()?;
        cache.bytes -= block.size;
        Some(block.ptr as *mut u8)
    }

    /// Keep `ptr` for reuse; false when the cache is full
    fn keep(&self, ptr: *mut u8, layout: Layout) -> bool {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.bytes + layout.size() > MAX_CACHED {
            return false;
        }
        let Some(slot) = cache.blocks.iter_mut().find(|slot| slot.is_none()) else {
            return false;
        };
        *slot = Some(Block {
            ptr: ptr as usize,
            size: layout.size(),
            align: layout.align(),
        });
        cache.bytes += layout.size();
        true
    }
}

impl Default for BlockCache {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every block handed out is either fresh from `System` or a block `System` allocated
// with the same layout and that was freed (so no one else holds it); blocks not kept go back to
// `System` with their original layout.
unsafe impl GlobalAlloc for BlockCache {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= MIN_SIZE
            && let Some(ptr) = self.take(layout)
        {
            return ptr;
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= MIN_SIZE
            && let Some(ptr) = self.take(layout)
        {
            unsafe { ptr.write_bytes(0, layout.size()) };
            return ptr;
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() >= MIN_SIZE && self.keep(ptr, layout) {
            return;
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_freed_large_blocks() {
        let cache = BlockCache::new();
        let layout = Layout::from_size_align(MIN_SIZE, 64).unwrap();
        unsafe {
            let a = cache.alloc(layout);
            assert!(!a.is_null(), "allocation failed");
            a.write_bytes(7, layout.size());
            cache.dealloc(a, layout);
            let b = cache.alloc_zeroed(layout);
            assert!(!b.is_null(), "allocation failed");
            assert_eq!(a, b, "the freed block is handed out again");
            assert!(
                std::slice::from_raw_parts(b, layout.size())
                    .iter()
                    .all(|&x| x == 0)
            );
            // A different size does not get it
            let other = Layout::from_size_align(MIN_SIZE + 64, 64).unwrap();
            cache.dealloc(b, layout);
            let c = cache.alloc(other);
            assert!(!c.is_null(), "allocation failed");
            assert_ne!(c, b);
            cache.dealloc(c, other);
            assert_eq!(cache.alloc(layout), b);
            cache.dealloc(b, layout);
        }
    }

    #[test]
    fn small_allocations_bypass_the_cache() {
        let cache = BlockCache::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();
        unsafe {
            let a = cache.alloc(layout);
            assert!(!a.is_null(), "allocation failed");
            cache.dealloc(a, layout);
        }
        assert_eq!(cache.cache.lock().unwrap().bytes, 0);
    }
}
