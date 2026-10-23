//! kamui as the process's allocator: every allocation of these tests, and of the test harness
//! around them, goes through it, as in the `furnace` command

use std::sync::mpsc;

#[global_allocator]
static ALLOCATOR: kamui::BlockCache = kamui::BlockCache::new();

/// Smallest block the cache keeps
const LARGE: usize = 256 * 1024;

/// The bytes of a buffer the tests check: every one natively, and under Miri, which
/// interprets each access, both ends and every 997th
fn positions(len: usize) -> impl Iterator<Item = usize> {
    let step = if cfg!(miri) { 997 } else { 1 };
    (0..len).step_by(step).chain(len.checked_sub(1))
}

fn all(bytes: &[u8], expected: impl Fn(usize) -> u8) -> bool {
    positions(bytes.len()).all(|i| bytes[i] == expected(i))
}

#[test]
fn a_vec_grows_and_shrinks_across_the_size_limit() {
    let mut v: Vec<u8> = (0..=255).collect();
    // Each doubling reallocates: small to small, small to large, large to large
    while v.len() < 2 * LARGE {
        v.extend_from_within(..);
        assert!(all(&v, |i| i as u8), "grown to {}", v.len());
    }
    // Large to small
    v.truncate(LARGE / 2);
    v.shrink_to_fit();
    assert!(all(&v, |i| i as u8));
    // And back
    v.extend_from_slice(&vec![7; LARGE]);
    assert!(all(&v, |i| if i < LARGE / 2 { i as u8 } else { 7 }));
}

#[test]
fn repeated_passes_reuse_their_buffers() {
    // The pattern the cache is for: the same large allocations, pass after pass
    let passes = if cfg!(miri) { 3 } else { 50 };
    for pass in 0..passes {
        let a = vec![pass as u8; LARGE];
        let b = vec![0u8; 2 * LARGE];
        let c = vec![1u8; LARGE + 4096].into_boxed_slice();
        assert!(all(&a, |_| pass as u8));
        assert!(all(&b, |_| 0), "a zeroed buffer is zeros when reused");
        assert!(all(&c, |_| 1));
    }
}

#[test]
fn buffers_freed_on_other_threads() {
    // Buffers made on one thread and dropped on another, as a model's outputs are
    let (send, recv) = mpsc::channel::<Vec<u8>>();
    let rounds = if cfg!(miri) { 3 } else { 50 };
    let consumer = std::thread::spawn(move || {
        let mut seen = 0;
        for (round, buffer) in recv.into_iter().enumerate() {
            assert!(all(&buffer, |_| round as u8));
            seen += 1;
        }
        seen
    });
    for round in 0..rounds {
        send.send(vec![round as u8; LARGE + round]).unwrap();
    }
    drop(send);
    assert_eq!(consumer.join().unwrap(), rounds);
}
