//! Timing tests in the style of dudect (Reparaz, Balasch and Verbauwhede, "Dude, is my code
//! constant time?", 2017): every operation that handles a secret is timed on two classes of
//! input, the class chosen at random for every sample, and Welch's t-test asks whether the two
//! timing distributions differ. |t| below 4.5 means no difference was detected, not that there
//! is none (CONSTANT_TIME.md says what this does and does not show).
//!
//! A control runs first: the same operation on the same kind of input in both classes, where
//! there is nothing to find. If it fails, the machine or the way it is timed would make any
//! other result meaningless, and the run says so instead of reporting leaks or passes.
//!
//! The test is `#[ignore]`d, because timing on a shared machine is too noisy for every pull
//! request and only a release build means anything:
//! `cargo test --release -p rustls-pure --lib constant_time -- --ignored --nocapture`,
//! run weekly by .github/workflows/rustls-pure-ct.yml.

use std::hint::black_box;
use std::time::{Duration, Instant};

use super::*;

/// Welch's |t| at or above this counts as a difference, as in dudect
const THRESHOLD: f64 = 4.5;
/// Roughly how long each target is timed, including preparing its inputs
const BUDGET: Duration = Duration::from_secs(60);
const MIN_SAMPLES: usize = 20_000;
const MAX_SAMPLES: usize = 2_000_000;
/// A sample lasts at least this many ticks of the clock, as many operations as that takes, so
/// a coarse clock does not round the difference away
const TICKS_PER_SAMPLE: u32 = 200;
/// How long a signed message is, as in a TLS 1.3 CertificateVerify: 64 spaces, a context
/// string, a zero byte and a transcript hash
const MESSAGE_LEN: usize = 130;
/// How many peer keys each class of the X25519 all-zero check has
const PEERS: usize = 512;

/// One class of input or the other, chosen at random for every sample: 0 for class A, 1 for
/// class B. A number rather than an enum, so inputs are made from it by arithmetic, the same
/// work and the same memory accesses for both classes, never by a branch on the class or by
/// reading from a place that depends on it. On Neoverse N2, reading a class's input from its own
/// address, even outside the timing, changed the time of the operation after it: |t| up to 58.
type Class = usize;

/// `a` for class A, `b` for class B, with both read and the mask deciding which survives
fn select<const N: usize>(class: Class, a: &[u8; N], b: &[u8; N]) -> [u8; N] {
    let mask = black_box(0u8.wrapping_sub(class as u8));
    std::array::from_fn(|i| a[i] ^ ((a[i] ^ b[i]) & mask))
}

/// A small generator for classes and inputs. The secrets it makes are only inputs to time,
/// so they need not be unpredictable.
struct Rng(u64);

impl Rng {
    fn new() -> Self {
        Self(u64::from_le_bytes(*random::<8>().unwrap()) | 1)
    }

    /// xorshift64*
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn class(&mut self) -> Class {
        (self.next() >> 63) as Class
    }

    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut bytes = [0; N];
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes()[..chunk.len()]);
        }
        bytes
    }

    /// "Fixed vs random": `fixed` for class A, fresh random bytes for class B. Both classes
    /// draw the random bytes
    fn fixed_or_random<const N: usize>(&mut self, class: Class, fixed: &[u8; N]) -> [u8; N] {
        select(class, fixed, &self.bytes::<N>())
    }
}

/// A running mean and variance (Welford's method)
#[derive(Clone, Copy, Default)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn push(&mut self, x: f64) {
        self.n += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.n;
        self.m2 += delta * (x - self.mean);
    }

    fn variance(&self) -> f64 {
        self.m2 / (self.n - 1.0)
    }
}

/// Welch's t of class A against class B
fn welch(a: &Moments, b: &Moments) -> f64 {
    if a.n < 2.0 || b.n < 2.0 {
        return 0.0;
    }
    let spread = (a.variance() / a.n + b.variance() / b.n).sqrt();
    if spread == 0.0 {
        0.0
    } else {
        (a.mean - b.mean) / spread
    }
}

/// The smallest step the clock takes
fn clock_resolution() -> Duration {
    (0..1000)
        .map(|_| {
            let start = Instant::now();
            loop {
                let now = Instant::now();
                if now > start {
                    break now - start;
                }
            }
        })
        .min()
        .unwrap()
}

/// What one target gave
struct Outcome {
    name: &'static str,
    /// Welch's t at the crop where |t| is largest, and that crop
    t: f64,
    crop: String,
    samples: usize,
    operations_per_sample: usize,
    /// Mean time of one operation, in nanoseconds, in each class
    means: [f64; 2],
    classes: [&'static str; 2],
}

impl Outcome {
    fn passed(&self) -> bool {
        self.t.abs() < THRESHOLD
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: max |t| = {:.2} ({}), {} samples of {} operation(s); mean {:.0} ns ({}) vs \
             {:.0} ns ({})",
            self.name,
            self.t.abs(),
            self.crop,
            self.samples,
            self.operations_per_sample,
            self.means[0],
            self.classes[0],
            self.means[1],
            self.classes[1],
        )
    }
}

/// Times `op` on inputs that `prepare` makes for each sample's class, outside the timing.
/// Every operation of a sample has an input of the sample's class; `op` takes it by value, so
/// it finds it in the same place in memory whatever the class. As in dudect, the t-test is
/// repeated on the samples below a range of percentiles, since a difference can hide under the
/// tail that the scheduler and interrupts add; the largest |t| is the result.
fn measure<I, O>(
    name: &'static str,
    classes: [&'static str; 2],
    resolution: Duration,
    mut prepare: impl FnMut(Class, &mut Rng) -> I,
    mut op: impl FnMut(I) -> O,
) -> Outcome {
    let mut rng = Rng::new();

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut sample = |rng: &mut Rng, batch: usize| {
        let class = rng.class();
        inputs.extend((0..batch).map(|_| prepare(class, rng)));
        outputs.reserve(batch);
        let start = Instant::now();
        for input in inputs.drain(..) {
            outputs.push(black_box(op(black_box(input))));
        }
        let elapsed = start.elapsed();
        outputs.clear();
        (class, elapsed.as_nanos() as f64 / batch as f64)
    };

    // How many operations make a sample long enough for the clock, and how many samples fit
    // the budget
    let calibration = Instant::now();
    let one: f64 = (0..200).map(|_| sample(&mut rng, 1).1).sum::<f64>() / 200.0;
    let per_sample = calibration.elapsed() / 200;
    let ticks = resolution.as_nanos() as f64 * f64::from(TICKS_PER_SAMPLE);
    let batch = (ticks / one.max(1.0)).ceil().max(1.0) as usize;
    let samples = (BUDGET.as_nanos() / (per_sample.as_nanos().max(1) * batch as u128)) as usize;
    let samples = samples.clamp(MIN_SAMPLES, MAX_SAMPLES);

    let times: Vec<(Class, f64)> = (0..samples).map(|_| sample(&mut rng, batch)).collect();
    // The first samples, while caches and frequency settle, are left out
    let times = &times[samples / 100..];

    let mut sorted: Vec<f64> = times.iter().map(|&(_, t)| t).collect();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
    // All samples, then those below the percentiles 1 - 0.5^(k + 1), as in dudect
    let crops = std::iter::once(("all".to_string(), f64::INFINITY)).chain((0..10).map(|k| {
        let p = 1.0 - 0.5_f64.powf(f64::from(k + 1));
        (format!("below p{:.1}", 100.0 * p), percentile(p))
    }));

    let mut worst = (0.0_f64, String::new());
    let mut means = [0.0; 2];
    for (i, (label, limit)) in crops.enumerate() {
        let mut moments = [Moments::default(); 2];
        for &(class, t) in times.iter().filter(|&&(_, t)| t <= limit) {
            moments[class].push(t);
        }
        if i == 0 {
            means = moments.map(|m| m.mean);
        }
        let t = welch(&moments[0], &moments[1]);
        if t.abs() > worst.0.abs() {
            worst = (t, label);
        }
    }
    Outcome {
        name,
        t: worst.0,
        crop: worst.1,
        samples: times.len(),
        operations_per_sample: batch,
        means,
        classes,
    }
}

#[test]
#[ignore = "timing: release build, .github/workflows/rustls-pure-ct.yml"]
fn constant_time() {
    let resolution = clock_resolution();
    println!("ct: clock resolution {resolution:?}");
    let control = control(resolution);
    println!("ct: {control}");

    let targets: [fn(Duration) -> Outcome; 13] = [
        x25519,
        x25519_all_zero_check,
        p256_key_exchange,
        p384_key_exchange,
        mlkem_decapsulation,
        x25519mlkem768_client,
        mlkem_encapsulation,
        ecdsa_p256_signing,
        ecdsa_p384_signing,
        ed25519_signing,
        aes_128_gcm_open,
        aes_256_gcm_open,
        chacha20_poly1305_open,
    ];
    let outcomes: Vec<Outcome> = targets
        .iter()
        .map(|target| {
            let outcome = target(resolution);
            println!("ct: {outcome}");
            outcome
        })
        .collect();

    assert!(
        control.passed(),
        "the control differs between two classes of the same inputs (|t| = {:.2}): timing on \
         this machine is not reliable, so no result here says anything about rustls-pure",
        control.t.abs()
    );
    let differ: Vec<&str> = outcomes
        .iter()
        .filter(|outcome| !outcome.passed())
        .map(|outcome| outcome.name)
        .collect();
    assert!(
        differ.is_empty(),
        "the time depends on the secret: {differ:?}"
    );
}

/// The harness on this machine: ML-KEM decapsulation, as below, but of valid ciphertexts in
/// both classes, made the same way. There is nothing to find.
fn control(resolution: Duration) -> Outcome {
    mlkem_decapsulation_of(
        "Control: ML-KEM-768 decapsulation",
        ["valid ciphertexts", "valid ciphertexts"],
        resolution,
        |_| 0,
    )
}

// ---------------------------------------------------------------------------------------------
// Key exchange

fn x25519(resolution: Duration) -> Outcome {
    let fixed = *random::<32>().unwrap();
    let peer = x25519_key(2).public;
    measure(
        "X25519 key exchange",
        ["fixed key", "random keys"],
        resolution,
        |class, rng| {
            let secret = x25519_dalek::StaticSecret::from(rng.fixed_or_random(class, &fixed));
            let public = x25519_dalek::PublicKey::from(&secret);
            Box::new(X25519Exchange { secret, public })
        },
        |exchange| exchange.complete(peer.as_bytes()),
    )
}

/// rustls-pure's one comparison of a secret: X25519's result against zero. If it stopped at
/// the first byte that differs, results that start with a zero byte would take longer.
fn x25519_all_zero_check(resolution: Duration) -> Outcome {
    let pairs = peers_by_first_byte(&x25519_key(1));
    measure(
        "X25519 all-zero check",
        [
            "secrets starting with a zero byte",
            "secrets starting with another byte",
        ],
        resolution,
        |class, rng| {
            let [a, b] = &pairs[rng.below(pairs.len())];
            let peer = select(class, a.as_bytes(), b.as_bytes());
            (Box::new(x25519_key(1)), x25519_dalek::PublicKey::from(peer))
        },
        |(exchange, peer)| exchange.complete(peer.as_bytes()),
    )
}

/// Pairs of peer keys: one whose result with `ours` starts with a zero byte, one whose result
/// does not, next to each other in memory
fn peers_by_first_byte(ours: &X25519Exchange) -> Vec<[x25519_dalek::PublicKey; 2]> {
    let (mut leading_zero, mut other) = (Vec::new(), Vec::new());
    while leading_zero.len() < PEERS || other.len() < PEERS {
        let peer = x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from(
            *random::<32>().unwrap(),
        ));
        if ours.secret.diffie_hellman(&peer).as_bytes()[0] == 0 {
            leading_zero.push(peer);
        } else {
            other.push(peer);
        }
    }
    leading_zero
        .into_iter()
        .zip(other)
        .map(|(a, b)| [a, b])
        .take(PEERS)
        .collect()
}

fn p256_key_exchange(resolution: Duration) -> Outcome {
    let fixed = *random::<32>().unwrap();
    let peer = SECP256R1.start().unwrap().pub_key().to_vec();
    measure(
        "P-256 key exchange",
        ["fixed key", "random keys"],
        resolution,
        |class, rng| {
            // A random 32 bytes is a valid scalar but for a chance of 2^-32
            let secret = p256::SecretKey::from_slice(&rng.fixed_or_random(class, &fixed));
            Box::new(Secp256r1Exchange {
                secret: secret.unwrap(),
                public: Vec::new(),
            })
        },
        |exchange| exchange.complete(&peer),
    )
}

fn p384_key_exchange(resolution: Duration) -> Outcome {
    let fixed = *random::<48>().unwrap();
    let peer = SECP384R1.start().unwrap().pub_key().to_vec();
    measure(
        "P-384 key exchange",
        ["fixed key", "random keys"],
        resolution,
        |class, rng| {
            let secret = p384::SecretKey::from_slice(&rng.fixed_or_random(class, &fixed));
            Box::new(Secp384r1Exchange {
                secret: secret.unwrap(),
                public: Vec::new(),
            })
        },
        |exchange| exchange.complete(&peer),
    )
}

/// A fresh ML-KEM ciphertext to `key`, with one random bit flipped if `tamper` is 1: such a
/// ciphertext decapsulates to the implicit-rejection secret
fn mlkem_ciphertext(
    key: &ml_kem::EncapsulationKey<ml_kem::MlKem768>,
    tamper: u8,
    rng: &mut Rng,
) -> ml_kem::Ciphertext<ml_kem::MlKem768> {
    let (mut ciphertext, _) = key.encapsulate_deterministic(&ml_kem::B32::from(rng.bytes()));
    ciphertext[rng.below(MLKEM768_CIPHERTEXT_LEN)] ^= tamper << rng.below(8);
    ciphertext
}

/// ML-KEM decapsulation of ciphertexts that `tamper` decides for each class
fn mlkem_decapsulation_of(
    name: &'static str,
    classes: [&'static str; 2],
    resolution: Duration,
    tamper: impl Fn(Class) -> u8,
) -> Outcome {
    let key = ml_kem::DecapsulationKey::<ml_kem::MlKem768>::from_seed(ml_kem::Seed::from(
        *random::<64>().unwrap(),
    ));
    let encapsulation_key = key.encapsulation_key().clone();
    measure(
        name,
        classes,
        resolution,
        |class, rng| mlkem_ciphertext(&encapsulation_key, tamper(class), rng),
        |ciphertext| key.decapsulate(&ciphertext),
    )
}

/// Implicit rejection: a tampered ciphertext must take as long as a valid one, or a timing
/// oracle tells the attacker which of its ciphertexts decapsulated
fn mlkem_decapsulation(resolution: Duration) -> Outcome {
    mlkem_decapsulation_of(
        "ML-KEM-768 decapsulation",
        ["valid ciphertexts", "tampered ciphertexts"],
        resolution,
        |class| class as u8,
    )
}

/// The same through rustls-pure's X25519MLKEM768 exchange, as a client completes it
fn x25519mlkem768_client(resolution: Duration) -> Outcome {
    let seed = ml_kem::Seed::from(*random::<64>().unwrap());
    let client = X25519MlKem768Group::start_with(seed, x25519_key(1));
    let encapsulation_key = client.decapsulation.encapsulation_key().clone();
    let server_x25519 = x25519_key(2).public;
    measure(
        "X25519MLKEM768 (client)",
        ["valid ciphertexts", "tampered ciphertexts"],
        resolution,
        |class, rng| {
            let ciphertext = mlkem_ciphertext(&encapsulation_key, class as u8, rng);
            (
                Box::new(X25519MlKem768Group::start_with(seed, x25519_key(1))),
                concat(&ciphertext, server_x25519.as_bytes()),
            )
        },
        |(exchange, share)| exchange.complete(&share),
    )
}

/// The server's side: `m` decides ML-KEM's secret
fn mlkem_encapsulation(resolution: Duration) -> Outcome {
    let key = ml_kem::DecapsulationKey::<ml_kem::MlKem768>::from_seed(ml_kem::Seed::from(
        *random::<64>().unwrap(),
    ))
    .encapsulation_key()
    .clone();
    let fixed = *random::<32>().unwrap();
    measure(
        "ML-KEM-768 encapsulation",
        ["fixed m", "random m"],
        resolution,
        |class, rng| ml_kem::B32::from(rng.fixed_or_random(class, &fixed)),
        |m| key.encapsulate_deterministic(&m),
    )
}

// ---------------------------------------------------------------------------------------------
// Signing, through the signers rustls asks for (each dropped after the timing, with its
// signature). The message is random in both classes: signing is deterministic, so with one
// message the fixed key would give the same signature every time, and the time to encode that
// signature, which is public (ECDSA's DER encoding is 70 to 72 bytes long for P-256, depending
// on the signature), would differ between the classes for reasons that are not secret.

fn ecdsa_p256_signing(resolution: Duration) -> Outcome {
    let fixed = *random::<32>().unwrap();
    measure(
        "ECDSA P-256 signing",
        ["fixed key", "random keys"],
        resolution,
        |class, rng| {
            let key = p256::ecdsa::SigningKey::from_slice(&rng.fixed_or_random(class, &fixed));
            let signer = EcdsaSigner(EcdsaKey::P256(Arc::new(key.unwrap())));
            (signer, rng.bytes::<MESSAGE_LEN>())
        },
        |(signer, message)| (signer.sign(&message), signer),
    )
}

fn ecdsa_p384_signing(resolution: Duration) -> Outcome {
    let fixed = *random::<48>().unwrap();
    measure(
        "ECDSA P-384 signing",
        ["fixed key", "random keys"],
        resolution,
        |class, rng| {
            let key = p384::ecdsa::SigningKey::from_slice(&rng.fixed_or_random(class, &fixed));
            let signer = EcdsaSigner(EcdsaKey::P384(Arc::new(key.unwrap())));
            (signer, rng.bytes::<MESSAGE_LEN>())
        },
        |(signer, message)| (signer.sign(&message), signer),
    )
}

fn ed25519_signing(resolution: Duration) -> Outcome {
    let fixed = *random::<32>().unwrap();
    measure(
        "Ed25519 signing",
        ["fixed key", "random keys"],
        resolution,
        |class, rng| {
            let key = ed25519_dalek::SigningKey::from_bytes(&rng.fixed_or_random(class, &fixed));
            (Ed25519Key(Arc::new(key)), rng.bytes::<MESSAGE_LEN>())
        },
        |(signer, message)| (signer.sign(&message), signer),
    )
}

// ---------------------------------------------------------------------------------------------
// Records

/// A full record under a random key: 2^14 bytes of content and its type, sealed, with its
/// additional data and tag
fn sealed_record<C: Cipher>() -> (Crypter<C>, Vec<u8>, [u8; 5], [u8; TAG_LEN]) {
    let key = random::<32>().unwrap();
    let crypter = Crypter::<C>::new(&key[..C::KEY_LEN], Iv::from(*random::<12>().unwrap()));
    let mut text = vec![0x17; (1 << 14) + 1];
    let aad = make_tls13_aad(sealed_payload_len(text.len() - 1));
    let tag = crypter.seal(0, &aad, &mut text).unwrap().into();
    (crypter, text, aad, tag)
}

/// A wrong tag must be refused as fast whichever of its bytes is wrong, or the attacker learns
/// how much of a tag was right. (A right tag takes longer by design, since only then is the
/// record decrypted; whether a record was accepted is not secret, as a refused one ends the
/// connection.)
fn aead_open<C: Cipher>(name: &'static str, resolution: Duration) -> Outcome {
    let (crypter, mut text, aad, tag) = sealed_record::<C>();
    measure(
        name,
        ["first tag byte wrong", "last tag byte wrong"],
        resolution,
        |class, rng| {
            // The same bytes changed for both classes: the first by `flip` for class A, the last
            // for class B, the other by nothing
            let flip = 1 + rng.below(255) as u8;
            let mut wrong = tag;
            wrong[0] ^= select(class, &[flip], &[0])[0];
            wrong[TAG_LEN - 1] ^= select(class, &[0], &[flip])[0];
            wrong
        },
        |wrong| crypter.open(0, &aad, &mut text, &wrong).is_err(),
    )
}

fn aes_128_gcm_open(resolution: Duration) -> Outcome {
    aead_open::<aes_gcm::Aes128Gcm>("AES-128-GCM open, wrong tag", resolution)
}

fn aes_256_gcm_open(resolution: Duration) -> Outcome {
    aead_open::<aes_gcm::Aes256Gcm>("AES-256-GCM open, wrong tag", resolution)
}

fn chacha20_poly1305_open(resolution: Duration) -> Outcome {
    aead_open::<chacha20poly1305::ChaCha20Poly1305>("ChaCha20-Poly1305 open, wrong tag", resolution)
}
