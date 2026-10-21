# kuebiko

Named after Kuebiko, the scarecrow god of Japanese myth who cannot walk yet knows everything
under heaven, as a metric stands in one place and sees every request.
Built for and maintained in [Furnace](https://github.com/Gilfeather/furnace), the pure-Rust
inference server.

Lock-free latency histograms with percentile estimates, and the Prometheus text format, with no
dependencies.

- `Histogram`: fixed buckets (0.5 ms to 10 s by default, or 1 µs to 10 s), one atomic counter
  each, cheap enough to update on every request. It renders as Prometheus `_bucket`, `_sum` and
  `_count` series and estimates p50, p95 and p99.
- `render_value`, `render_family`, `render_labelled`, `label`: counters, gauges and labelled
  samples appended to a `String`, for a `GET /metrics` handler to return as is. No registry.
- Feature `serde`: `Serialize` and `Deserialize` for `Percentiles`.

```rust
use kuebiko::{Histogram, render_value};
use std::time::Duration;

let latency = Histogram::new();
latency.observe(Duration::from_millis(8));

let mut out = String::new();
latency.render(&mut out, "app_request_duration_seconds", "Time to answer a request");
render_value(&mut out, "app_up", "gauge", "Whether the app is up", 1);
println!("{out}");
println!("{:?}", latency.percentiles_ms()); // Percentiles { p50: Some(..), .. }
```

It serves the metrics of [Furnace](https://github.com/Gilfeather/furnace). The API
documentation is on [docs.rs](https://docs.rs/kuebiko). MIT licensed.
