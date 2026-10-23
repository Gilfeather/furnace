//! A global allocator that keeps freed large blocks for reuse, with no dependencies.
//!
//! Named after Kamui (神威), the power to put things away in another space and bring them back,
//! as kamui keeps freed memory and hands it out again.
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
use std::ptr::NonNull;
use std::sync::Mutex;

/// Smallest allocation whose block is kept on free
const MIN_SIZE: usize = 256 * 1024;

/// Most freed blocks kept at once
const SLOTS: usize = 64;

/// Most bytes kept in freed blocks at once
pub const MAX_CACHED: usize = 128 * 1024 * 1024;

/// A freed block, kept as a pointer rather than an address so it keeps the provenance
/// `System` gave it: an address cast back to a pointer would have none
#[derive(Clone, Copy)]
struct Block {
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
}

// SAFETY: a kept block is owned by the cache alone (its user freed it), and `System` lets any
// thread use or free a block another thread allocated
unsafe impl Send for Block {}

struct Cache {
    blocks: [Option<Block>; SLOTS],
    bytes: usize,
}

/// [`System`] plus a cache of freed large blocks; see the module docs.
///
/// ```no_run
/// #[global_allocator]
/// static ALLOCATOR: kamui::BlockCache = kamui::BlockCache::new();
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
        Some(block.ptr.as_ptr())
    }

    /// Keep `ptr` for reuse; false when the cache is full
    fn keep(&self, ptr: *mut u8, layout: Layout) -> bool {
        let Some(ptr) = NonNull::new(ptr) else {
            return false;
        };
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.bytes + layout.size() > MAX_CACHED {
            return false;
        }
        let Some(slot) = cache.blocks.iter_mut().find(|slot| slot.is_none()) else {
            return false;
        };
        *slot = Some(Block {
            ptr,
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

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Small to small never touches the cache: `System` may grow or shrink the block in
        // place, where the default would allocate, copy and free
        if layout.size() < MIN_SIZE && new_size < MIN_SIZE {
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        // SAFETY: as the caller promised for `realloc`; `new_layout` is valid by its contract
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cache.alloc`, failing the test instead of handing out a null pointer
    fn alloc(cache: &BlockCache, layout: Layout) -> NonNull<u8> {
        NonNull::new(unsafe { cache.alloc(layout) }).expect("allocation failed")
    }

    fn alloc_zeroed(cache: &BlockCache, layout: Layout) -> NonNull<u8> {
        NonNull::new(unsafe { cache.alloc_zeroed(layout) }).expect("allocation failed")
    }

    /// The bytes of a block of `len` the tests write and check: every one natively, and under
    /// Miri, which interprets each access, both ends and every 997th (a prime, so a copy that
    /// shifts by a power of two still shows)
    fn positions(len: usize) -> impl Iterator<Item = usize> {
        let step = if cfg!(miri) { 997 } else { 1 };
        (0..len).step_by(step).chain(len.checked_sub(1))
    }

    fn zeroed(ptr: NonNull<u8>, len: usize) -> bool {
        positions(len).all(|i| unsafe { *ptr.as_ptr().add(i) } == 0)
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
        assert!(zeroed(b, layout.size()));

        // A different size does not get it
        let other = Layout::from_size_align(MIN_SIZE + 64, 64).unwrap();
        unsafe { cache.dealloc(b.as_ptr(), layout) };
        let c = alloc(&cache, other);
        assert_ne!(c, b);
        unsafe { cache.dealloc(c.as_ptr(), other) };
        let d = alloc(&cache, layout);
        assert_eq!(d, b);
        unsafe { cache.dealloc(d.as_ptr(), layout) };
        release(&cache);
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

    #[test]
    fn realloc_keeps_the_contents_across_the_size_limit() {
        let cache = BlockCache::new();
        let small = Layout::from_size_align(1024, 8).unwrap();
        let ptr = alloc(&cache, small);
        unsafe { ptr.as_ptr().write_bytes(5, small.size()) };
        // Small to small, small to large, large to large, large to small
        let mut layout = small;
        let mut ptr = ptr.as_ptr();
        for new_size in [4096, MIN_SIZE + 4096, 2 * MIN_SIZE, 2048] {
            ptr = unsafe { cache.realloc(ptr, layout, new_size) };
            assert!(!ptr.is_null());
            let kept = unsafe { std::slice::from_raw_parts(ptr, small.size()) };
            assert!(kept.iter().all(|&b| b == 5), "realloc to {new_size}");
            layout = Layout::from_size_align(new_size, 8).unwrap();
        }
        unsafe { cache.dealloc(ptr, layout) };
        // The large blocks freed on the way were kept
        assert_eq!(cached(&cache), (2, MIN_SIZE + 4096 + 2 * MIN_SIZE));
        release(&cache);
    }

    /// Give every kept block back to `System`, with the layout it was allocated with
    fn release(cache: &BlockCache) {
        let mut cache = cache.cache.lock().unwrap();
        for block in cache.blocks.iter_mut().filter_map(Option::take) {
            let layout = Layout::from_size_align(block.size, block.align).unwrap();
            unsafe { System.dealloc(block.ptr.as_ptr(), layout) };
        }
        cache.bytes = 0;
    }

    /// The byte at `i` of a filled block: bytes differ from one position to the next, so a
    /// copy that shifts or drops a byte shows; the pattern repeats every 64 KiB
    fn byte(i: usize) -> u8 {
        i as u8 ^ (i >> 8) as u8
    }

    /// Fill every byte of a block, copying the pattern in 64 KiB pieces (one bulk copy each,
    /// which Miri runs as one operation)
    fn fill(ptr: NonNull<u8>, len: usize) {
        const PERIOD: usize = 1 << 16;
        let pattern: Vec<u8> = (0..PERIOD).map(byte).collect();
        for start in (0..len).step_by(PERIOD) {
            let n = (len - start).min(PERIOD);
            unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), ptr.as_ptr().add(start), n) };
        }
    }

    /// Whether the first `len` bytes of a block are still as `fill` wrote them
    fn filled(ptr: NonNull<u8>, len: usize) -> bool {
        positions(len).all(|i| unsafe { *ptr.as_ptr().add(i) } == byte(i))
    }

    #[test]
    fn realloc_between_large_sizes_grows_shrinks_and_stays() {
        let cache = BlockCache::new();
        let mut layout = Layout::from_size_align(MIN_SIZE, 64).unwrap();
        let mut ptr = alloc(&cache, layout);
        fill(ptr, layout.size());
        // A block of the size grown to is cached already: realloc hands that one out
        let waiting = Layout::from_size_align(2 * MIN_SIZE, 64).unwrap();
        let block = alloc(&cache, waiting);
        unsafe { cache.dealloc(block.as_ptr(), waiting) };
        // Same size, grow onto the cached block, shrink, shrink below the limit. Only the
        // bytes filled at the start are known: growing adds bytes no one wrote.
        let mut written = layout.size();
        for new_size in [MIN_SIZE, 2 * MIN_SIZE, MIN_SIZE + 1, MIN_SIZE - 1] {
            let old = ptr;
            ptr = NonNull::new(unsafe { cache.realloc(ptr.as_ptr(), layout, new_size) })
                .expect("allocation failed");
            if new_size == 2 * MIN_SIZE {
                assert_eq!(ptr, block, "the cached block of that size is reused");
            }
            if new_size == layout.size() {
                assert_ne!(ptr, old, "a same-size realloc moves to a fresh block");
            }
            assert_eq!(ptr.as_ptr() as usize % 64, 0);
            written = written.min(new_size);
            assert!(filled(ptr, written), "realloc to {new_size}");
            layout = Layout::from_size_align(new_size, 64).unwrap();
        }
        // Below the limit now: this one goes back to `System`
        unsafe { cache.dealloc(ptr.as_ptr(), layout) };
        // Kept on the way: every block realloc moved away from (MIN_SIZE twice, 2 * MIN_SIZE
        // and MIN_SIZE + 1)
        assert_eq!(cached(&cache), (4, 5 * MIN_SIZE + 1));
        release(&cache);
    }

    #[test]
    fn a_kept_block_goes_only_to_its_own_alignment() {
        let cache = BlockCache::new();
        let loose = Layout::from_size_align(MIN_SIZE, 8).unwrap();
        // An alignment as large as the block itself
        let strict = Layout::from_size_align(MIN_SIZE, MIN_SIZE).unwrap();
        let a = alloc(&cache, loose);
        unsafe { cache.dealloc(a.as_ptr(), loose) };
        let b = alloc(&cache, strict);
        assert_eq!(b.as_ptr() as usize % MIN_SIZE, 0);
        assert_eq!(
            cached(&cache),
            (1, MIN_SIZE),
            "the loose block is not handed out"
        );
        unsafe { cache.dealloc(b.as_ptr(), strict) };
        let c = alloc_zeroed(&cache, strict);
        assert_eq!(c, b);
        assert_eq!(c.as_ptr() as usize % MIN_SIZE, 0);
        // realloc keeps the alignment, and the zeroed bytes
        let d = NonNull::new(unsafe { cache.realloc(c.as_ptr(), strict, 2 * MIN_SIZE) })
            .expect("allocation failed");
        assert_eq!(d.as_ptr() as usize % MIN_SIZE, 0);
        assert!(zeroed(d, MIN_SIZE));
        let grown = Layout::from_size_align(2 * MIN_SIZE, MIN_SIZE).unwrap();
        unsafe { cache.dealloc(d.as_ptr(), grown) };
        assert_eq!(cached(&cache), (3, 4 * MIN_SIZE));
        release(&cache);
    }

    /// A block handed from one thread to another, as a model's output is
    struct Sent(NonNull<u8>);

    // SAFETY: the block is `System` memory, which any thread may use and free; the tests hand
    // each one to a single thread at a time
    unsafe impl Send for Sent {}

    #[test]
    fn a_block_freed_on_another_thread_is_reused_on_a_third() {
        let cache = BlockCache::new();
        let layout = Layout::from_size_align(MIN_SIZE, 64).unwrap();
        let shared = &cache;
        std::thread::scope(|s| {
            let a = s
                .spawn(move || {
                    let a = alloc(shared, layout);
                    fill(a, layout.size());
                    Sent(a)
                })
                .join()
                .unwrap();
            let a = s
                .spawn(move || {
                    let a = a;
                    assert!(filled(a.0, layout.size()));
                    unsafe { shared.dealloc(a.0.as_ptr(), layout) };
                    a.0.as_ptr() as usize
                })
                .join()
                .unwrap();
            let b = s
                .spawn(move || {
                    let b = alloc(shared, layout);
                    // The bytes the first thread wrote, seen from the third
                    assert!(filled(b, layout.size()));
                    Sent(b)
                })
                .join()
                .unwrap();
            assert_eq!(b.0.as_ptr() as usize, a);
            unsafe { shared.dealloc(b.0.as_ptr(), layout) };
        });
        release(&cache);
    }

    /// Threads allocating and freeing blocks of the sizes the cache keeps, at once, each
    /// checking its block keeps what it wrote: no block is handed to two threads, and the
    /// cache's count stays the bytes it holds
    #[test]
    fn threads_share_the_cache() {
        const THREADS: usize = 4;
        let rounds = if cfg!(miri) { 4 } else { 200 };
        let cache = BlockCache::new();
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let cache = &cache;
                s.spawn(move || {
                    for round in 0..rounds {
                        let layout =
                            Layout::from_size_align(MIN_SIZE + (round + t) % 2 * 4096, 64).unwrap();
                        let block = alloc(cache, layout);
                        let byte = (t * rounds + round) as u8;
                        unsafe { block.as_ptr().write_bytes(byte, layout.size()) };
                        std::thread::yield_now();
                        for i in [0, layout.size() / 2, layout.size() - 1] {
                            assert_eq!(unsafe { *block.as_ptr().add(i) }, byte);
                        }
                        unsafe { cache.dealloc(block.as_ptr(), layout) };
                    }
                });
            }
        });
        // A fresh block of a size is made only while every block of that size is in use, one
        // per thread at most: so at most THREADS of each of the two sizes
        let (blocks, bytes) = cached(&cache);
        assert!(blocks <= 2 * THREADS);
        let kept: usize = cache
            .cache
            .lock()
            .unwrap()
            .blocks
            .iter()
            .flatten()
            .map(|b| b.size)
            .sum();
        assert_eq!(kept, bytes);
        release(&cache);
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        /// A block in use: where, its layout, and the byte it was filled with
        struct Live {
            ptr: NonNull<u8>,
            layout: Layout,
            fill: u8,
        }

        #[derive(Clone, Debug)]
        enum Op {
            /// Allocate the size of this index among a few large and small ones, at an
            /// alignment, zeroed or not
            Alloc(prop::sample::Index, usize, bool),
            /// Free the live block of this index
            Free(prop::sample::Index),
        }

        /// A few sizes, most of them large enough to be kept, so blocks are reused often
        const SIZES: [usize; 6] = [
            1024,
            MIN_SIZE - 1,
            MIN_SIZE,
            MIN_SIZE + 4096,
            3 * MIN_SIZE,
            MAX_CACHED / 16,
        ];

        fn ops() -> impl Strategy<Value = Vec<Op>> {
            prop::collection::vec(
                prop_oneof![
                    (
                        any::<prop::sample::Index>(),
                        prop::sample::select(vec![8usize, 64, 4096]),
                        any::<bool>()
                    )
                        .prop_map(|(size, align, zeroed)| Op::Alloc(size, align, zeroed)),
                    any::<prop::sample::Index>().prop_map(Op::Free),
                ],
                1..60,
            )
        }

        /// A block filled with one byte, checked at both ends and in the middle (a whole check
        /// of every byte would make each case slow)
        fn intact(live: &Live) -> bool {
            let size = live.layout.size();
            [0, size / 2, size - 1]
                .iter()
                .all(|&i| unsafe { *live.ptr.as_ptr().add(i) } == live.fill)
        }

        fn check_cache(cache: &BlockCache) -> Result<(), TestCaseError> {
            let cache = cache.cache.lock().unwrap();
            let kept: usize = cache.blocks.iter().flatten().map(|b| b.size).sum();
            prop_assert_eq!(kept, cache.bytes);
            prop_assert!(cache.bytes <= MAX_CACHED);
            Ok(())
        }

        proptest! {
            // At most 64 cases (8 under Miri); PROPTEST_CASES lowers it. Under Miri no file of
            // failed cases: Miri keeps the test away from the file system.
            #![proptest_config(ProptestConfig {
                cases: ProptestConfig::default().cases.min(if cfg!(miri) { 8 } else { 64 }),
                failure_persistence: if cfg!(miri) {
                    None
                } else {
                    ProptestConfig::default().failure_persistence
                },
                ..ProptestConfig::default()
            })]

            /// Blocks in use never overlap and keep what was written to them, a zeroed one is
            /// all zeros even when it is a reused block, and the cache's count of bytes is
            /// what it holds, within its limit
            #[test]
            fn blocks_in_use_never_overlap(ops in ops()) {
                let cache = BlockCache::new();
                let mut live: Vec<Live> = Vec::new();
                let mut next_fill = 1u8;
                for op in ops {
                    match op {
                        Op::Alloc(size, align, zeroed) => {
                            let layout = Layout::from_size_align(*size.get(&SIZES), align).unwrap();
                            let ptr = if zeroed {
                                alloc_zeroed(&cache, layout)
                            } else {
                                alloc(&cache, layout)
                            };
                            prop_assert_eq!(ptr.as_ptr() as usize % align, 0);
                            if zeroed {
                                prop_assert!(super::zeroed(ptr, layout.size()));
                            }
                            let start = ptr.as_ptr() as usize;
                            let end = start + layout.size();
                            for other in &live {
                                let other_start = other.ptr.as_ptr() as usize;
                                let other_end = other_start + other.layout.size();
                                prop_assert!(end <= other_start || other_end <= start, "overlap");
                            }
                            unsafe { ptr.as_ptr().write_bytes(next_fill, layout.size()) };
                            live.push(Live { ptr, layout, fill: next_fill });
                            next_fill = next_fill.wrapping_add(1).max(1);
                        }
                        Op::Free(index) => {
                            if live.is_empty() {
                                continue;
                            }
                            let block = live.swap_remove(index.index(live.len()));
                            prop_assert!(intact(&block));
                            unsafe { cache.dealloc(block.ptr.as_ptr(), block.layout) };
                        }
                    }
                    prop_assert!(live.iter().all(intact));
                    check_cache(&cache)?;
                }
                // Give everything back: the blocks in use, then the ones the cache kept
                for block in live {
                    unsafe { cache.dealloc(block.ptr.as_ptr(), block.layout) };
                }
                let mut cache = cache.cache.lock().unwrap();
                for block in cache.blocks.iter_mut().filter_map(Option::take) {
                    let layout = Layout::from_size_align(block.size, block.align).unwrap();
                    unsafe { System.dealloc(block.ptr.as_ptr(), layout) };
                }
                cache.bytes = 0;
            }
        }
    }
}
