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

/// Freed blocks kept, in `N` slots: the allocator's has [`SLOTS`], and the proofs check the
/// same code with fewer
struct Cache<const N: usize> {
    blocks: [Option<Block>; N],
    bytes: usize,
}

impl<const N: usize> Cache<N> {
    const fn new() -> Self {
        Self {
            blocks: [None; N],
            bytes: 0,
        }
    }

    /// A kept block of exactly `layout`, if any, no longer kept
    fn take(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let slot = self.blocks.iter_mut().find(|slot| {
            slot.is_some_and(|b| b.size == layout.size() && b.align == layout.align())
        })?;
        let block = slot.take()?;
        self.bytes -= block.size;
        Some(block.ptr)
    }

    /// Keep `ptr` of `layout`; false when the cache is full
    fn keep(&mut self, ptr: NonNull<u8>, layout: Layout) -> bool {
        if self.bytes + layout.size() > MAX_CACHED {
            return false;
        }
        let Some(slot) = self.blocks.iter_mut().find(|slot| slot.is_none()) else {
            return false;
        };
        *slot = Some(Block {
            ptr,
            size: layout.size(),
            align: layout.align(),
        });
        self.bytes += layout.size();
        true
    }
}

/// [`System`] plus a cache of freed large blocks; see the module docs.
///
/// ```no_run
/// #[global_allocator]
/// static ALLOCATOR: kamui::BlockCache = kamui::BlockCache::new();
/// ```
pub struct BlockCache {
    blocks: Blocks<SLOTS>,
}

impl BlockCache {
    pub const fn new() -> Self {
        Self {
            blocks: Blocks::new(),
        }
    }
}

impl Default for BlockCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The allocator, with a cache of `N` slots: [`BlockCache`]'s has [`SLOTS`], and the proofs
/// check the same code with fewer
struct Blocks<const N: usize> {
    // `std::sync::Mutex` does not allocate (SRW lock on Windows, futex on Linux), so it can be
    // used inside the allocator
    cache: Mutex<Cache<N>>,
}

impl<const N: usize> Blocks<N> {
    const fn new() -> Self {
        Self {
            cache: Mutex::new(Cache::new()),
        }
    }

    /// A cached block of exactly `layout`, if any
    fn take(&self, layout: Layout) -> Option<*mut u8> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.take(layout).map(NonNull::as_ptr)
    }

    /// Keep `ptr` for reuse; false when the cache is full
    fn keep(&self, ptr: *mut u8, layout: Layout) -> bool {
        let Some(ptr) = NonNull::new(ptr) else {
            return false;
        };
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.keep(ptr, layout)
    }

    // The four below have `GlobalAlloc`'s contracts. Every block handed out is either fresh
    // from `System` or a block `System` allocated with the same layout and that was freed (so
    // no one else holds it); blocks not kept go back to `System` with their original layout.

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

// SAFETY: `Blocks` keeps `GlobalAlloc`'s contracts (see there), and each call passes the
// caller's promises on unchanged
unsafe impl GlobalAlloc for BlockCache {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { self.blocks.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { self.blocks.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.blocks.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { self.blocks.realloc(ptr, layout, new_size) }
    }
}

/// Kani proofs (`cargo kani -p kamui`): the cache's bookkeeping from any state it can be in, and
/// the allocator's unsafe paths on real memory, checked for out-of-bounds access, use after free
/// and double free.
///
/// Each proof is written once for a cache of any number of slots and run for the allocator's 64
/// (`*_64_slots`), on every pull request but `keep`'s, which runs nightly, and for 4 as a quick
/// check.
#[cfg(kani)]
mod verification {
    use super::*;

    /// Any non-null pointer
    fn any_ptr() -> NonNull<u8> {
        let addr = kani::any_where(|&addr: &usize| addr != 0);
        NonNull::new(addr as *mut u8).unwrap()
    }

    impl kani::Arbitrary for Block {
        fn any() -> Self {
            Self {
                ptr: any_ptr(),
                size: kani::any(),
                align: kani::any(),
            }
        }
    }

    /// Any valid layout: any size and any power-of-two alignment `Layout` takes
    fn any_layout() -> Layout {
        let align = 1usize << kani::any_where(|&shift: &u32| shift < usize::BITS);
        let layout = Layout::from_size_align(kani::any(), align);
        kani::assume(layout.is_ok());
        layout.unwrap()
    }

    /// Any valid layout of `min..=max` bytes.
    ///
    /// The allocator proofs take sizes of 1 to `MIN_SIZE + 64` bytes, which is enough to reach
    /// every behaviour of `Blocks`. Its code looks at a size only to compare it with
    /// `MIN_SIZE` (below or not), to compare two layouts for equality, to take the smaller of
    /// two sizes for `realloc`'s copy, and to pass it on to `System` and `Cache`. This range
    /// has sizes on both sides of the limit, and pairs of sizes that are smaller, equal or
    /// larger than each other on each side. A larger size takes the same branches as one of
    /// these. `Cache`'s byte arithmetic, the only place a size is added up, is proved for
    /// sizes of any value (the `take` and `keep` proofs).
    fn any_layout_within(min: usize, max: usize) -> Layout {
        let layout = any_layout();
        kani::assume((min..=max).contains(&layout.size()));
        layout
    }

    /// The bytes held in a slot
    fn bytes_in(slot: &Option<Block>) -> u128 {
        slot.map_or(0, |block| block.size as u128)
    }

    /// The bytes held in every slot of `blocks` but `at`, in 128 bits, where no sum of `N`
    /// sizes overflows. With slot `at`'s added, it is the total in any order of slots, since
    /// the addition is exact.
    fn bytes_but<const N: usize>(blocks: &[Option<Block>; N], at: usize) -> u128 {
        let mut bytes = 0;
        for i in 0..N {
            if i != at {
                bytes += bytes_in(&blocks[i]);
            }
        }
        bytes
    }

    /// The first slot `pick` takes, or `N` for none
    fn first<const N: usize>(
        blocks: &[Option<Block>; N],
        pick: impl Fn(&Option<Block>) -> bool,
    ) -> usize {
        let mut first = N;
        for i in (0..N).rev() {
            if pick(&blocks[i]) {
                first = i;
            }
        }
        first
    }

    fn same(a: &Option<Block>, b: &Option<Block>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => a.ptr == b.ptr && a.size == b.size && a.align == b.align,
            _ => false,
        }
    }

    // The two proofs below start from any cache whose byte count is the sum of its blocks'
    // sizes, within MAX_CACHED. They sum the slots with the one the operation changes last,
    // which is the same total, and check every other slot unchanged at any index: the sum
    // after the operation is then that of the unchanged slots plus the changed one. So the
    // count's invariant is checked without comparing two sums of every slot, which takes Kani
    // hours at 64 slots.

    /// `take` from any cache: the first block of exactly the layout asked for, its slot emptied
    /// with nothing else touched (so a block kept once is handed out once) and its size taken
    /// off the count, which stays the sum of the blocks kept; or none when no block has the
    /// layout, the cache unchanged
    fn take_hands_out_a_block_once<const N: usize>() {
        let blocks: [Option<Block>; N] = kani::any();
        let layout = any_layout();
        let fits = |slot: &Option<Block>| {
            slot.is_some_and(|b| b.size == layout.size() && b.align == layout.align())
        };
        let hit = first(&blocks, fits);
        // The count is the sum of the sizes kept, slot `hit` (any slot, for none) added last
        let at = hit.min(N - 1);
        let rest = bytes_but(&blocks, at);
        let bytes: usize = kani::any();
        kani::assume(rest + bytes_in(&blocks[at]) == bytes as u128 && bytes <= MAX_CACHED);

        let mut cache = Cache { blocks, bytes };
        let taken = cache.take(layout);
        let j = kani::any_where(|&j: &usize| j < N);
        if hit < N {
            let block = blocks[hit].unwrap();
            assert_eq!(taken, Some(block.ptr));
            assert!(cache.blocks[hit].is_none());
            if j != hit {
                assert!(same(&blocks[j], &cache.blocks[j]));
            }
            // Every other slot unchanged and slot `hit` empty: the sum kept is `rest`
            assert_eq!(cache.bytes as u128, rest);
            assert!(cache.bytes <= MAX_CACHED);
        } else {
            assert!(taken.is_none());
            assert!(!fits(&blocks[j]));
            assert!(same(&blocks[j], &cache.blocks[j]));
            assert_eq!(cache.bytes, bytes);
        }
    }

    /// `keep` into any cache: the block with its layout in the first empty slot, nothing else
    /// touched, and its size added to the count, which stays the sum of the blocks kept within
    /// MAX_CACHED; or refused with the cache unchanged, so the caller hands it back to `System`
    fn keep_keeps_a_block_or_refuses_it<const N: usize>() {
        let blocks: [Option<Block>; N] = kani::any();
        let layout = any_layout();
        let ptr = any_ptr();
        let free = first(&blocks, |slot| slot.is_none());
        // The count is the sum of the sizes kept, slot `free` (any slot, for none) added last
        let at = free.min(N - 1);
        let rest = bytes_but(&blocks, at);
        let bytes: usize = kani::any();
        kani::assume(rest + bytes_in(&blocks[at]) == bytes as u128 && bytes <= MAX_CACHED);

        let mut cache = Cache { blocks, bytes };
        let kept = cache.keep(ptr, layout);
        let j = kani::any_where(|&j: &usize| j < N);
        if kept {
            assert!(free < N);
            let block = cache.blocks[free].expect("the slot holds the block");
            assert_eq!(block.ptr, ptr);
            assert_eq!((block.size, block.align), (layout.size(), layout.align()));
            if j != free {
                assert!(same(&blocks[j], &cache.blocks[j]));
            }
            // Every other slot unchanged and slot `free`, empty before, holding the block: the
            // sum kept is `rest` plus its size
            assert_eq!(cache.bytes as u128, rest + layout.size() as u128);
            assert!(cache.bytes <= MAX_CACHED);
        } else {
            assert!(same(&blocks[j], &cache.blocks[j]));
            assert_eq!(cache.bytes, bytes);
        }
    }

    /// How many blocks the cache keeps
    fn kept<const N: usize>(blocks: &Blocks<N>) -> usize {
        // Index loops: over 64 slots they unwind into far fewer paths for Kani than iterator
        // adapters
        let cache = blocks.cache.lock().unwrap();
        let mut count = 0;
        for i in 0..N {
            if cache.blocks[i].is_some() {
                count += 1;
            }
        }
        count
    }

    /// Give every kept block back to `System`, with the layout it was allocated with
    fn release<const N: usize>(blocks: &Blocks<N>) {
        let mut cache = blocks.cache.lock().unwrap();
        for i in 0..N {
            if let Some(block) = cache.blocks[i].take() {
                let layout = Layout::from_size_align(block.size, block.align).unwrap();
                unsafe { System.dealloc(block.ptr.as_ptr(), layout) };
            }
        }
        cache.bytes = 0;
    }

    /// A block freed and allocated again, zeroed or not, of any layout around the size limit
    /// (1 to `MIN_SIZE + 64` bytes, see `any_layout_within`; any alignment): every write and free stays
    /// within live memory `System` handed out, a reused block is the one freed, and a zeroed
    /// one is all zeros (at any byte) even when it is reused
    fn alloc_and_dealloc_stay_in_bounds<const N: usize>() {
        let blocks = Blocks::<N>::new();
        let first = any_layout_within(1, MIN_SIZE + 64);
        let a = unsafe { blocks.alloc(first) };
        kani::assume(!a.is_null());
        unsafe {
            a.write_bytes(0xa5, first.size());
            blocks.dealloc(a, first);
        }

        let second = any_layout_within(1, MIN_SIZE + 64);
        let zeroed: bool = kani::any();
        let kept_before = kept(&blocks);
        let b = unsafe {
            if zeroed {
                blocks.alloc_zeroed(second)
            } else {
                blocks.alloc(second)
            }
        };
        kani::assume(!b.is_null());
        if kept(&blocks) < kept_before {
            // From the cache: the block freed, and only for the layout `System` allocated it
            // with, so aligned as `System` aligned it
            assert_eq!(a, b);
            assert_eq!(second, first);
        } else {
            // Fresh from `System`, whose `GlobalAlloc` contract is to align it. (Not asserted
            // here: CBMC's `posix_memalign` hands out blocks for alignments as large as 2^62
            // unaligned, where Linux's refuses them, and `alloc` passes `System`'s pointer on
            // untouched.)
            assert_eq!(kept(&blocks), kept_before);
        }
        if first.size() >= MIN_SIZE && second == first {
            assert_eq!(a, b, "the freed block is handed out again");
        }
        let at = kani::any_where(|&i: &usize| i < second.size());
        if zeroed {
            assert_eq!(unsafe { *b.add(at) }, 0);
        }
        unsafe {
            b.add(at).write(1);
            blocks.dealloc(b, second);
        }
        release(&blocks);
    }

    /// `realloc` of a block of any layout around the size limit to any size around it, across
    /// it either way, keeps the bytes both sizes have (at any byte), and frees or keeps the
    /// old block exactly once
    fn realloc_keeps_the_contents<const N: usize>() {
        let blocks = Blocks::<N>::new();
        let old = any_layout_within(1, MIN_SIZE + 64);
        let ptr = unsafe { blocks.alloc(old) };
        kani::assume(!ptr.is_null());
        let at = kani::any_where(|&i: &usize| i < old.size());
        unsafe { ptr.add(at).write(0x5a) };

        let new_size = kani::any_where(|&n: &usize| (1..=MIN_SIZE + 64).contains(&n));
        let new = Layout::from_size_align(new_size, old.align()).unwrap();
        let moved = unsafe { blocks.realloc(ptr, old, new_size) };
        kani::assume(!moved.is_null());
        if at < new_size {
            assert_eq!(unsafe { *moved.add(at) }, 0x5a);
        }
        unsafe {
            moved.add(new_size - 1).write(1);
            blocks.dealloc(moved, new);
        }
        release(&blocks);
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn take_hands_out_a_block_once_4_slots() {
        take_hands_out_a_block_once::<4>();
    }

    #[kani::proof]
    #[kani::unwind(66)]
    fn take_hands_out_a_block_once_64_slots() {
        take_hands_out_a_block_once::<SLOTS>();
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn keep_keeps_a_block_or_refuses_it_4_slots() {
        keep_keeps_a_block_or_refuses_it::<4>();
    }

    #[kani::proof]
    #[kani::unwind(66)]
    fn keep_keeps_a_block_or_refuses_it_64_slots() {
        keep_keeps_a_block_or_refuses_it::<SLOTS>();
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn alloc_and_dealloc_stay_in_bounds_4_slots() {
        alloc_and_dealloc_stay_in_bounds::<4>();
    }

    #[kani::proof]
    #[kani::unwind(66)]
    fn alloc_and_dealloc_stay_in_bounds_64_slots() {
        alloc_and_dealloc_stay_in_bounds::<SLOTS>();
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn realloc_keeps_the_contents_4_slots() {
        realloc_keeps_the_contents::<4>();
    }

    #[kani::proof]
    #[kani::unwind(66)]
    fn realloc_keeps_the_contents_64_slots() {
        realloc_keeps_the_contents::<SLOTS>();
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
        let cache = cache.blocks.cache.lock().unwrap();
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
        assert_eq!(cache.blocks.cache.lock().unwrap().bytes, 0);
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
        let mut cache = cache.blocks.cache.lock().unwrap();
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
            .blocks
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
            let cache = cache.blocks.cache.lock().unwrap();
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
                let mut cache = cache.blocks.cache.lock().unwrap();
                for block in cache.blocks.iter_mut().filter_map(Option::take) {
                    let layout = Layout::from_size_align(block.size, block.align).unwrap();
                    unsafe { System.dealloc(block.ptr.as_ptr(), layout) };
                }
                cache.bytes = 0;
            }
        }
    }
}
