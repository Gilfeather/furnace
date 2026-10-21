//! A global allocator that keeps freed large blocks for reuse, with no dependencies.
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
//!
//! The `furnace` command of [furnace](https://github.com/Gilfeather/furnace) runs on it.

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
/// static ALLOCATOR: furnace_alloc::BlockCache = furnace_alloc::BlockCache::new();
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
    use std::ptr::NonNull;

    /// `cache.alloc`, failing the test instead of handing out a null pointer
    fn alloc(cache: &BlockCache, layout: Layout) -> NonNull<u8> {
        NonNull::new(unsafe { cache.alloc(layout) }).expect("allocation failed")
    }

    fn alloc_zeroed(cache: &BlockCache, layout: Layout) -> NonNull<u8> {
        NonNull::new(unsafe { cache.alloc_zeroed(layout) }).expect("allocation failed")
    }

    #[test]
    fn reuses_freed_large_blocks() {
        let cache = BlockCache::new();
        let layout = Layout::from_size_align(MIN_SIZE, 64).unwrap();
        let a = alloc(&cache, layout);
        unsafe {
            a.as_ptr().write_bytes(7, layout.size());
            cache.dealloc(a.as_ptr(), layout);
        }
        let b = alloc_zeroed(&cache, layout);
        assert_eq!(a, b, "the freed block is handed out again");
        let zeroed = unsafe { std::slice::from_raw_parts(b.as_ptr(), layout.size()) };
        assert!(zeroed.iter().all(|&x| x == 0));

        // A different size does not get it
        let other = Layout::from_size_align(MIN_SIZE + 64, 64).unwrap();
        unsafe { cache.dealloc(b.as_ptr(), layout) };
        let c = alloc(&cache, other);
        assert_ne!(c, b);
        unsafe { cache.dealloc(c.as_ptr(), other) };
        let d = alloc(&cache, layout);
        assert_eq!(d, b);
        unsafe { cache.dealloc(d.as_ptr(), layout) };
    }

    fn cached(cache: &BlockCache) -> (usize, usize) {
        let cache = cache.cache.lock().unwrap();
        (cache.blocks.iter().flatten().count(), cache.bytes)
    }

    #[test]
    fn a_full_cache_hands_blocks_back_to_the_system() {
        let cache = BlockCache::new();
        let layout = Layout::from_size_align(MIN_SIZE, 64).unwrap();
        // One block more than there are slots: the last goes back to `System`
        let blocks: Vec<_> = (0..=SLOTS).map(|_| alloc(&cache, layout)).collect();
        for block in &blocks {
            unsafe { cache.dealloc(block.as_ptr(), layout) };
        }
        assert_eq!(cached(&cache), (SLOTS, SLOTS * MIN_SIZE));

        // Bytes: a block that would take the cache past MAX_CACHED goes back too, though a
        // slot is free
        let kept = alloc(&cache, layout);
        assert_eq!(cached(&cache), (SLOTS - 1, (SLOTS - 1) * MIN_SIZE));
        let big = Layout::from_size_align(MAX_CACHED - (SLOTS - 1) * MIN_SIZE + 1, 64).unwrap();
        let block = alloc(&cache, big);
        unsafe { cache.dealloc(block.as_ptr(), big) };
        assert_eq!(cached(&cache), (SLOTS - 1, (SLOTS - 1) * MIN_SIZE));
        // One that fits exactly is kept
        let exact = Layout::from_size_align(MAX_CACHED - (SLOTS - 1) * MIN_SIZE, 64).unwrap();
        let block = alloc(&cache, exact);
        unsafe { cache.dealloc(block.as_ptr(), exact) };
        assert_eq!(cached(&cache), (SLOTS, MAX_CACHED));

        // Give everything back to `System`
        unsafe { System.dealloc(kept.as_ptr(), layout) };
        let block = alloc(&cache, exact);
        unsafe { System.dealloc(block.as_ptr(), exact) };
        for _ in 0..SLOTS - 1 {
            let block = alloc(&cache, layout);
            unsafe { System.dealloc(block.as_ptr(), layout) };
        }
        assert_eq!(cached(&cache), (0, 0));
    }

    #[test]
    fn small_allocations_bypass_the_cache() {
        let cache = BlockCache::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let a = alloc(&cache, layout);
        unsafe { cache.dealloc(a.as_ptr(), layout) };
        assert_eq!(cache.cache.lock().unwrap().bytes, 0);
    }
}
