//! Who may call the server and how often: API keys (`--api-key`) and a per-client rate limit
//! (`--rate-limit`), shared by the HTTP and gRPC transports.
//!
//! The rate limit is a GCRA (generic cell rate algorithm), i.e. a token bucket kept as a single
//! "theoretical arrival time" per client: `per_second` requests a second on average, and bursts
//! of up to `burst` at once. Clients are the API key they present or, without keys, their IP
//! address. Only a hash of either is kept.

use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// `--rate-limit` and `--rate-limit-burst`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClientRateLimit {
    /// Requests a second per client, on average
    pub per_second: f64,
    /// Requests a client may send at once
    pub burst: u32,
}

/// Why a request is turned away
#[derive(Debug, PartialEq)]
pub enum Rejection {
    /// API keys are required and it presented none or an unknown one
    Unauthorized,
    /// Over the rate limit; may retry after this long
    RateLimited(Duration),
}

/// API keys and rate limiting for one server
pub struct Access {
    api_keys: Vec<Vec<u8>>,
    limiter: Option<RateLimiter>,
}

impl Access {
    /// Admit requests presenting one of `api_keys` (any request when empty), each client
    /// within `rate_limit` if one is set
    pub fn new(api_keys: &[String], rate_limit: Option<ClientRateLimit>) -> Self {
        Self {
            api_keys: api_keys.iter().map(|k| k.as_bytes().to_vec()).collect(),
            limiter: rate_limit.map(RateLimiter::new),
        }
    }

    /// Whether requests need an API key
    pub fn requires_key(&self) -> bool {
        !self.api_keys.is_empty()
    }

    /// Whether every request is admitted: no API keys and no rate limit
    pub fn is_open(&self) -> bool {
        self.api_keys.is_empty() && self.limiter.is_none()
    }

    /// Admit a request presenting `key` (from `Authorization: Bearer` or `X-API-Key`) from
    /// `ip`, or say why not. `check_key` is false for routes with their own authentication.
    pub fn admit(
        &self,
        key: Option<&str>,
        ip: Option<IpAddr>,
        check_key: bool,
    ) -> Result<(), Rejection> {
        let known_key = key.filter(|key| self.is_known(key));
        if check_key && self.requires_key() && known_key.is_none() {
            return Err(Rejection::Unauthorized);
        }
        if let Some(limiter) = &self.limiter {
            let client = match (known_key, ip) {
                (Some(key), _) => Client::Key(key),
                (None, Some(ip)) => Client::Ip(ip),
                (None, None) => Client::Unknown,
            };
            limiter.check(client, Instant::now())?;
        }
        Ok(())
    }

    /// Compares with every key, each in constant time, so timing reveals neither the key nor
    /// which one matched
    fn is_known(&self, key: &str) -> bool {
        self.api_keys.iter().fold(false, |found, known| {
            found | constant_time_eq(known, key.as_bytes())
        })
    }
}

/// `a == b` in time depending only on the lengths
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The API key in `Authorization: Bearer <key>` or `X-API-Key: <key>`
pub fn presented_key<'a>(
    authorization: Option<&'a str>,
    x_api_key: Option<&'a str>,
) -> Option<&'a str> {
    authorization
        .and_then(|value| value.strip_prefix("Bearer "))
        .or(x_api_key)
        .map(str::trim)
}

#[derive(Hash)]
enum Client<'a> {
    Key(&'a str),
    Ip(IpAddr),
    Unknown,
}

const SHARDS: usize = 16;
/// A shard with more clients than this drops the ones whose bucket has refilled
const SHARD_PRUNE_AT: usize = 4096;

struct RateLimiter {
    /// Time one request uses up
    interval: u64,
    /// How far ahead of now a client's arrival time may run: (burst - 1) intervals
    tolerance: u64,
    hasher: RandomState,
    epoch: Instant,
    /// Per client hash, its theoretical arrival time in nanoseconds since `epoch`
    shards: [Mutex<HashMap<u64, u64>>; SHARDS],
}

impl RateLimiter {
    fn new(limit: ClientRateLimit) -> Self {
        let interval = (1e9 / limit.per_second).round().max(1.0) as u64;
        Self {
            interval,
            tolerance: interval.saturating_mul(u64::from(limit.burst.max(1) - 1)),
            hasher: RandomState::new(),
            epoch: Instant::now(),
            shards: std::array::from_fn(|_| Mutex::new(HashMap::new())),
        }
    }

    fn check(&self, client: Client<'_>, now: Instant) -> Result<(), Rejection> {
        let id = self.hasher.hash_one(&client);
        let now =
            u64::try_from(now.saturating_duration_since(self.epoch).as_nanos()).unwrap_or(u64::MAX);
        let mut shard = self.shards[id as usize % SHARDS]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let arrival = shard.get(&id).copied().unwrap_or(now).max(now);
        if arrival - now > self.tolerance {
            return Err(Rejection::RateLimited(Duration::from_nanos(
                arrival - now - self.tolerance,
            )));
        }
        if shard.len() >= SHARD_PRUNE_AT && !shard.contains_key(&id) {
            // A client whose arrival time has passed is the same as one never seen
            shard.retain(|_, &mut arrival| arrival > now);
        }
        shard.insert(id, arrival + self.interval);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(last: u8) -> Option<IpAddr> {
        Some(IpAddr::from([10, 0, 0, last]))
    }

    #[test]
    fn keys_are_required_when_configured() {
        let open = Access::new(&[], None);
        assert_eq!(open.admit(None, ip(1), true), Ok(()));
        let keys = ["first-key".to_string(), "second-key".to_string()];
        let access = Access::new(&keys, None);
        assert_eq!(access.admit(Some("second-key"), ip(1), true), Ok(()));
        assert_eq!(
            access.admit(None, ip(1), true),
            Err(Rejection::Unauthorized)
        );
        assert_eq!(
            access.admit(Some("second-ke"), ip(1), true),
            Err(Rejection::Unauthorized)
        );
        // Routes with their own authentication
        assert_eq!(access.admit(None, ip(1), false), Ok(()));
    }

    #[test]
    fn presented_key_from_either_header() {
        assert_eq!(presented_key(Some("Bearer abc"), None), Some("abc"));
        assert_eq!(presented_key(None, Some("abc")), Some("abc"));
        assert_eq!(presented_key(Some("Basic abc"), None), None);
        assert_eq!(presented_key(Some("Bearer a"), Some("b")), Some("a"));
    }

    #[test]
    fn bursts_then_the_average_rate() {
        let limiter = RateLimiter::new(ClientRateLimit {
            per_second: 10.0,
            burst: 3,
        });
        let t0 = Instant::now();
        let check = |at_ms: u64| {
            limiter.check(
                Client::Ip(IpAddr::from([10, 0, 0, 1])),
                t0 + Duration::from_millis(at_ms),
            )
        };
        // Three at once, then one every 100 ms
        for _ in 0..3 {
            assert_eq!(check(0), Ok(()));
        }
        let Err(Rejection::RateLimited(retry)) = check(0) else {
            panic!("the fourth request should be limited");
        };
        assert!(retry > Duration::from_millis(99) && retry <= Duration::from_millis(100));
        assert_eq!(check(100), Ok(()));
        assert!(check(150).is_err());
        assert_eq!(check(200), Ok(()));
        // An idle client gets its whole burst back
        for _ in 0..3 {
            assert_eq!(check(10_000), Ok(()));
        }
        assert!(check(10_000).is_err());
    }

    #[test]
    fn clients_are_limited_separately() {
        let keys = ["key-a".to_string(), "key-b".to_string()];
        let access = Access::new(
            &keys,
            Some(ClientRateLimit {
                per_second: 1.0,
                burst: 1,
            }),
        );
        assert_eq!(access.admit(Some("key-a"), ip(1), true), Ok(()));
        assert!(access.admit(Some("key-a"), ip(2), true).is_err());
        // Another key from the same address has its own budget
        assert_eq!(access.admit(Some("key-b"), ip(1), true), Ok(()));

        let open = Access::new(
            &[],
            Some(ClientRateLimit {
                per_second: 1.0,
                burst: 1,
            }),
        );
        assert_eq!(open.admit(None, ip(1), true), Ok(()));
        assert!(open.admit(None, ip(1), true).is_err());
        assert_eq!(open.admit(None, ip(2), true), Ok(()));
        // An unknown key does not open a new budget
        assert!(open.admit(Some("made-up"), ip(1), true).is_err());
    }

    #[test]
    fn many_clients_stay_bounded() {
        let limiter = RateLimiter::new(ClientRateLimit {
            per_second: 1000.0,
            burst: 1,
        });
        let t0 = Instant::now();
        for i in 0..20_000u32 {
            let ip = IpAddr::from(i.to_be_bytes());
            // 10 ms apart: each earlier client's bucket (1 ms) has refilled
            let _ = limiter.check(
                Client::Ip(ip),
                t0 + Duration::from_millis(u64::from(i) * 10),
            );
        }
        let clients: usize = limiter.shards.iter().map(|s| s.lock().unwrap().len()).sum();
        assert!(clients <= SHARDS * SHARD_PRUNE_AT, "{clients}");
    }
}
