//! Picking the next token from host logits: penalties, temperature, top-k, top-p and a seed

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::SamplingParams;

/// Below this temperature sampling is greedy
const GREEDY_BELOW: f32 = 1e-5;

/// Samples tokens for one sequence; it remembers the tokens it picked, for the penalties
#[derive(Debug, Clone)]
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    rng: SplitMix64,
    /// Times each generated token was picked
    counts: HashMap<u32, u32>,
    /// Candidates of the current step, reused between steps
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(params: &SamplingParams) -> Self {
        Self {
            temperature: params.temperature.unwrap_or(1.0),
            top_p: params.top_p.unwrap_or(1.0),
            top_k: params.top_k.unwrap_or(0),
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            rng: SplitMix64::new(params.seed.unwrap_or_else(random_seed)),
            counts: HashMap::new(),
            candidates: Vec::new(),
        }
    }

    /// The next token. `logits` are changed in place (penalties, temperature).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 {
            for (&token, &count) in &self.counts {
                if let Some(logit) = logits.get_mut(token as usize) {
                    *logit -= self.presence_penalty + self.frequency_penalty * count as f32;
                }
            }
        }
        let token = if self.temperature < GREEDY_BELOW || self.top_k == 1 {
            argmax(logits)
        } else {
            self.sample_random(logits)
        };
        *self.counts.entry(token).or_insert(0) += 1;
        token
    }

    fn sample_random(&mut self, logits: &[f32]) -> u32 {
        let candidates = &mut self.candidates;
        candidates.clear();
        candidates.extend(
            logits
                .iter()
                .enumerate()
                .map(|(i, &l)| (i as u32, if l.is_nan() { f32::NEG_INFINITY } else { l })),
        );
        let descending = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        if self.top_k > 0 && self.top_k < candidates.len() {
            candidates.select_nth_unstable_by(self.top_k - 1, descending);
            candidates.truncate(self.top_k);
        }
        // Softmax at the temperature
        let max = candidates
            .iter()
            .map(|c| c.1)
            .fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            return argmax(logits);
        }
        let mut sum = 0.0;
        for c in candidates.iter_mut() {
            c.1 = ((c.1 - max) / self.temperature).exp();
            sum += c.1;
        }
        for c in candidates.iter_mut() {
            c.1 /= sum;
        }
        if self.top_p < 1.0 {
            // The smallest prefix of the most likely tokens reaching top_p, without sorting the
            // whole vocabulary: look at the best k, doubling k until they are enough
            let mut k = candidates.len().min(64);
            loop {
                if k < candidates.len() {
                    candidates.select_nth_unstable_by(k - 1, descending);
                }
                candidates[..k].sort_unstable_by(descending);
                let mut cumulative = 0.0;
                let reached = candidates[..k].iter().position(|c| {
                    cumulative += c.1;
                    cumulative >= self.top_p
                });
                if let Some(last) = reached {
                    candidates.truncate(last + 1);
                    break;
                }
                if k == candidates.len() {
                    break;
                }
                k = (k * 2).min(candidates.len());
            }
        }
        let total: f32 = candidates.iter().map(|c| c.1).sum();
        let mut threshold = self.rng.next_f32() * total;
        for &(token, p) in candidates.iter() {
            threshold -= p;
            if threshold <= 0.0 {
                return token;
            }
        }
        candidates.last().map_or(0, |c| c.0)
    }
}

/// The index of the largest logit (NaN counts as the smallest)
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_value {
            best = i;
            best_value = l;
        }
    }
    best as u32
}

/// A seed for requests that bring none: the clock mixed with a counter
fn random_seed() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos
        ^ NEXT
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// A small, fast generator (SplitMix64); sampling needs no cryptographic randomness
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f32) -> SamplingParams {
        SamplingParams {
            temperature: Some(temperature),
            seed: Some(7),
            ..SamplingParams::default()
        }
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut sampler = Sampler::new(&params(0.0));
        assert_eq!(sampler.sample(&mut [0.1, 3.0, f32::NAN, 2.0]), 1);
    }

    #[test]
    fn the_same_seed_gives_the_same_tokens() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 / 10.0).collect();
        let run = || {
            let mut sampler = Sampler::new(&params(1.0));
            (0..50)
                .map(|_| sampler.sample(&mut logits.clone()))
                .collect::<Vec<_>>()
        };
        let tokens = run();
        assert_eq!(tokens, run());
        assert!(tokens.iter().any(|&t| t != tokens[0]), "not all one token");
    }

    #[test]
    fn top_k_keeps_only_the_best_tokens() {
        let mut sampler = Sampler::new(&SamplingParams {
            top_k: Some(2),
            ..params(5.0)
        });
        for _ in 0..200 {
            let token = sampler.sample(&mut [1.0, 9.0, 0.0, 8.0, 2.0]);
            assert!(token == 1 || token == 3, "{token}");
        }
    }

    #[test]
    fn top_p_keeps_the_smallest_set_reaching_it() {
        // One token holds almost all the probability, across a large vocabulary
        let mut logits = vec![0.0f32; 150_000];
        logits[123_456] = 20.0;
        let mut sampler = Sampler::new(&SamplingParams {
            top_p: Some(0.9),
            ..params(1.0)
        });
        for _ in 0..20 {
            assert_eq!(sampler.sample(&mut logits.clone()), 123_456);
        }
    }

    #[test]
    fn top_p_over_a_flat_distribution_widens_its_search() {
        let mut sampler = Sampler::new(&SamplingParams {
            top_p: Some(0.5),
            ..params(1.0)
        });
        let logits = vec![1.0f32; 1000];
        let picked: std::collections::HashSet<u32> = (0..300)
            .map(|_| sampler.sample(&mut logits.clone()))
            .collect();
        // About half of the tokens are candidates
        assert!(picked.len() > 100, "{}", picked.len());
    }

    #[test]
    fn penalties_steer_away_from_repeated_tokens() {
        let mut sampler = Sampler::new(&SamplingParams {
            frequency_penalty: 2.0,
            ..params(0.0)
        });
        let logits = [5.0, 4.0, 0.0];
        assert_eq!(sampler.sample(&mut logits.clone()), 0);
        // 5 - 2 < 4
        assert_eq!(sampler.sample(&mut logits.clone()), 1);
    }
}
