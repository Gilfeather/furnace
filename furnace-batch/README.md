# furnace-batch

Dynamic batching on worker threads, with no dependencies.

Items queue to a few worker threads, each running one batch at a time. While some worker is
idle, every item runs alone, at once. Once every worker is busy the queue fills, and the next
worker to finish takes everything queued, up to a maximum weight, as one batch. A worker waits
for more items only while one is announced as on its way (or, if told to, for a fixed window),
so a lone item never waits. Items nobody wants any more are skipped, and a batch that panics
loses only its own items.

This order suits CPUs: a second small pass running alongside the first uses the cores the first
leaves idle, which gains more than merging the two. Past a few passes at once, batching wins.
It is the batching behind [Furnace](https://github.com/Gilfeather/furnace)'s model server and
nervous system.

```rust
use furnace_batch::{Batching, Config, Engine, Queued};
use std::time::Duration;

struct Forward; // your model

impl Batching for Forward {
    type Item = Vec<f32>;
    fn weight(&self, _: &Vec<f32>) -> usize { 1 }
    fn window(&self) -> Duration { Duration::from_millis(2) }
    fn run(&self, batch: Vec<Queued<Vec<f32>>>) {
        // one forward pass for the whole batch, then deliver each result
    }
}

let config = Config { max_batch: 8, workers: 4, capacity: Some(1024), waits_unannounced: false };
let engine = Engine::start(Forward, config, |i| format!("forward-{i}"));
let arrival = engine.arrival(); // a request is coming: workers may wait for it
let input = vec![0.0; 784];      // ... while its body is decoded
engine.submit(Some(arrival), input).unwrap();
```

The API documentation is on [docs.rs](https://docs.rs/furnace-batch). MIT licensed.
