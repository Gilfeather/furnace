//! N clients, each on its own connection, sending one request after another

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What [`load`] measured: every timed call's result, client after client, and the wall time
/// from the moment all clients were released to the last one finishing
#[derive(Debug)]
pub struct Run<T> {
    pub results: Vec<T>,
    pub wall: Duration,
}

/// Run `clients` clients at once, each with its own `connect()` (an HTTP client with one
/// keep-alive connection, say). Each sends `warmup` untimed calls, waits for every other
/// client to finish warming up, then sends `per_client` calls back to back, collecting what
/// they return.
///
/// `call(client, c, i)` is call `i` of client `c`; the warm-up calls are `0..warmup` and the
/// timed ones `warmup..warmup + per_client`, so inputs that depend on `i` never repeat.
/// `at_start` runs in the caller's task as the clients are released, before the wall clock
/// starts (to read a counter such as the server's CPU time).
///
/// # Panics
///
/// If a client's task panics.
pub async fn load<C, T, F, Fut>(
    clients: usize,
    warmup: usize,
    per_client: usize,
    connect: impl Fn() -> C,
    call: F,
    at_start: impl FnOnce(),
) -> Run<T>
where
    C: Clone + Send + 'static,
    T: Send + 'static,
    F: Fn(C, usize, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = T> + Send,
{
    let call = Arc::new(call);
    let barrier = Arc::new(tokio::sync::Barrier::new(clients + 1));
    let tasks: Vec<_> = (0..clients)
        .map(|c| {
            let (client, call, barrier) = (connect(), call.clone(), barrier.clone());
            tokio::spawn(async move {
                for i in 0..warmup {
                    call(client.clone(), c, i).await;
                }
                barrier.wait().await;
                let mut results = Vec::with_capacity(per_client);
                for i in warmup..warmup + per_client {
                    results.push(call(client.clone(), c, i).await);
                }
                results
            })
        })
        .collect();
    barrier.wait().await;
    at_start();
    let start = Instant::now();
    let mut results = Vec::with_capacity(clients * per_client);
    for task in tasks {
        results.extend(task.await.expect("client task"));
    }
    Run {
        results,
        wall: start.elapsed(),
    }
}
