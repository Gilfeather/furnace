//! Size of the thread pool that runs inference.
//!
//! By default Furnace uses one thread per physical core, like ONNX Runtime. Inference is
//! bound by the FMA units, which two hyperthreads of a core share, so a thread per logical CPU
//! adds no throughput; it does add stragglers when the OS or a hypervisor deschedules one of the
//! threads that every parallel step waits for. In a Linux VM on a 16-thread laptop, ResNet-18
//! took 24 ms per image with 8 threads and 50 ms with 16.

/// One thread per physical core, at most the CPUs this process may use (cgroup and affinity
/// limits included)
pub fn default_threads() -> usize {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    num_cpus::get_physical().clamp(1, available)
}

/// Size the global pool: `threads` if given, else `RAYON_NUM_THREADS`, else
/// [`default_threads`]. Returns the size, or the existing pool's size if one was already built.
pub fn init(threads: Option<usize>) -> usize {
    let from_env = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let threads = threads.or(from_env).unwrap_or_else(default_threads);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        Ok(()) => threads,
        Err(_) => rayon::current_num_threads(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_within_the_available_cpus() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = default_threads();
        assert!(
            (1..=available).contains(&threads),
            "{threads} of {available}"
        );
    }
}
