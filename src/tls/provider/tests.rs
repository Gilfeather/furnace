//! Checks of the provider against known answers and known attacks.
//!
//! Most come from Wycheproof (`tests/fixtures/wycheproof`): for every algorithm the provider
//! offers, inputs that must be accepted and inputs that must be refused, such as forged
//! signatures, points off the curve, altered tags and low-order keys. The rest check the glue
//! between rustls and the primitives, which Wycheproof cannot see: nonces, sequence numbers,
//! randomness and the wiping of keys.

use std::cell::Cell;
use std::collections::HashSet;
use std::path::Path;

use p256::elliptic_curve::zeroize::ZeroizeOnDrop;
use rustls::crypto::cipher::OutboundChunks;
use serde_json::Value;

use super::*;

const VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/wycheproof");

/// What a Wycheproof test expects of an implementation
#[derive(Debug, Clone, Copy, PartialEq)]
enum Expect {
    Valid,
    Invalid,
    /// Either answer is within the standard, e.g. X25519 with a public key on the twist
    Acceptable,
}

/// One test of a Wycheproof file, with its group's shared fields
struct Case<'a> {
    group: &'a Value,
    test: &'a Value,
    /// Set when furnace refuses an input that Wycheproof lets through, and why
    stricter: Cell<Option<&'static str>>,
}

impl Case<'_> {
    fn id(&self) -> u64 {
        self.test["tcId"].as_u64().expect("a test id")
    }

    fn hex(&self, field: &str) -> Vec<u8> {
        hex(&self.test[field])
    }

    fn expect(&self) -> Expect {
        match self.test["result"].as_str() {
            Some("valid") => Expect::Valid,
            Some("invalid") => Expect::Invalid,
            Some("acceptable") => Expect::Acceptable,
            other => panic!("test {}: unknown result {other:?}", self.id()),
        }
    }

    /// The provider must refuse this input, whatever Wycheproof says
    fn must_refuse(&self, why: &'static str) {
        self.stricter.set(Some(why));
    }
}

fn hex(value: &Value) -> Vec<u8> {
    let text = value.as_str().expect("a hex string");
    assert!(text.len().is_multiple_of(2), "odd-length hex {text:?}");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex digits"))
        .collect()
}

/// How the tests of one file went; every one that ran went as expected
#[derive(Debug, Default)]
struct Tally {
    valid: usize,
    invalid: usize,
    acceptable: usize,
    /// Refused although Wycheproof allows them (see [`Case::must_refuse`])
    stricter: usize,
    /// Inputs the provider is never given, such as AES-GCM with 64-bit nonces
    skipped: usize,
}

/// Runs `check` on every test of a Wycheproof file. `check` says whether the provider accepted
/// the input, or `None` to skip a test outside what rustls ever hands the provider. Valid tests
/// must be accepted and invalid ones refused; the failures, if any, are the panic message.
fn run(file: &str, mut check: impl FnMut(&Case<'_>) -> Option<bool>) -> Tally {
    let path = Path::new(VECTORS).join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let vectors: Value = serde_json::from_str(&text).unwrap();
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    let mut total = 0;
    for group in vectors["testGroups"].as_array().expect("test groups") {
        for test in group["tests"].as_array().expect("tests") {
            total += 1;
            let case = Case {
                group,
                test,
                stricter: Cell::new(None),
            };
            let Some(accepted) = check(&case) else {
                tally.skipped += 1;
                continue;
            };
            let expect = case.expect();
            let ok = match (case.stricter.get(), expect) {
                (Some(_), _) => {
                    tally.stricter += 1;
                    !accepted
                }
                (None, Expect::Valid) => {
                    tally.valid += 1;
                    accepted
                }
                (None, Expect::Invalid) => {
                    tally.invalid += 1;
                    !accepted
                }
                (None, Expect::Acceptable) => {
                    tally.acceptable += 1;
                    true
                }
            };
            if !ok {
                let expected = match case.stricter.get() {
                    Some(why) => format!("refused, as {why}"),
                    None => format!("{expect:?}"),
                };
                failures.push(format!(
                    "test {} ({}, flags {}): expected {expected}, was {}",
                    case.id(),
                    test["comment"],
                    test["flags"],
                    if accepted { "accepted" } else { "refused" },
                ));
            }
        }
    }
    assert_eq!(
        Some(total),
        vectors["numberOfTests"].as_u64().map(|n| n as usize),
        "{file}: every test was read"
    );
    assert!(
        failures.is_empty(),
        "{file}: {} of {total} tests failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(tally.valid > 0, "{file}: no valid test ran: {tally:?}");
    tally
}

// ---------------------------------------------------------------------------------------------
// Record protection

/// The AEAD tests the provider can be given: TLS 1.3 always uses 96-bit nonces and full tags
fn aead<C: Cipher>(file: &str) -> Tally {
    run(file, |case| {
        let (key, iv, tag) = (case.hex("key"), case.hex("iv"), case.hex("tag"));
        if key.len() != C::KEY_LEN || iv.len() != 12 || tag.len() != TAG_LEN {
            return None;
        }
        let (aad, msg, ct) = (case.hex("aad"), case.hex("msg"), case.hex("ct"));
        // The test's nonce as some record's: the IV that gives it for that sequence number, so
        // the nonce construction is checked along with the cipher
        let seq = case.id().wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut iv: [u8; 12] = iv.try_into().unwrap();
        for (byte, seq) in iv[4..].iter_mut().zip(seq.to_be_bytes()) {
            *byte ^= seq;
        }
        let crypter = Crypter::<C>::new(&key, Iv::from(iv));
        let mut text = ct.clone();
        let mut accepted = crypter.open(seq, &aad, &mut text, &tag).is_ok() && text == msg;
        if case.expect() == Expect::Valid {
            let mut sealed = msg;
            let sealed_tag = crypter.seal(seq, &aad, &mut sealed).unwrap();
            accepted &= sealed == ct && sealed_tag[..] == tag[..];
        }
        Some(accepted)
    })
}

#[test]
fn wycheproof_aes_128_gcm() {
    let tally = aead::<aes_gcm::Aes128Gcm>("aes_gcm_test.json");
    assert!(tally.invalid > 0, "{tally:?}");
}

#[test]
fn wycheproof_aes_256_gcm() {
    let tally = aead::<aes_gcm::Aes256Gcm>("aes_gcm_test.json");
    assert!(tally.invalid > 0, "{tally:?}");
}

#[test]
fn wycheproof_chacha20_poly1305() {
    let tally = aead::<chacha20poly1305::ChaCha20Poly1305>("chacha20_poly1305_test.json");
    assert!(tally.invalid > 0, "{tally:?}");
}

/// The IV with the sequence number XORed into its last 8 bytes (RFC 8446 section 5.3)
fn expected_nonce(iv: [u8; 12], seq: u64) -> [u8; 12] {
    let mut nonce = iv;
    for (byte, seq) in nonce[4..].iter_mut().zip(seq.to_be_bytes()) {
        *byte ^= seq;
    }
    nonce
}

#[test]
fn every_record_gets_its_own_nonce() {
    let iv = *random::<12>().unwrap();
    let crypter = Crypter::<aes_gcm::Aes128Gcm>::new(&[0; 16], Iv::from(iv));
    let mut seen = HashSet::new();
    for seq in (0..100_000).chain([u64::from(u32::MAX), u64::MAX - 1, u64::MAX]) {
        let nonce: [u8; 12] = crypter.nonce(seq).into();
        assert_eq!(nonce, expected_nonce(iv, seq), "record {seq}");
        assert!(seen.insert(nonce), "record {seq} reuses a nonce");
    }
}

/// Records through the provider's encrypter and decrypter, as rustls uses them
fn records<C: Cipher>() {
    let key = random::<32>().unwrap();
    let iv = *random::<12>().unwrap();
    let mut sender = Crypter::<C>::new(&key[..C::KEY_LEN], Iv::from(iv));
    let mut receiver = Crypter::<C>::new(&key[..C::KEY_LEN], Iv::from(iv));
    let mut seal = |seq| {
        let plain = OutboundPlainMessage {
            typ: ContentType::ApplicationData,
            version: ProtocolVersion::TLSv1_2,
            payload: OutboundChunks::Single(b"GET /healthz"),
        };
        sender
            .encrypt(plain, seq)
            .unwrap()
            .payload
            .as_ref()
            .to_vec()
    };
    let mut open = |seq, record: &[u8]| {
        let mut record = record.to_vec();
        let opaque = InboundOpaqueMessage::new(
            ContentType::ApplicationData,
            ProtocolVersion::TLSv1_2,
            &mut record,
        );
        receiver
            .decrypt(opaque, seq)
            .map(|plain| (plain.typ, plain.payload.to_vec()))
    };
    let first = seal(1);
    // The same message in the next record encrypts differently, under the next nonce
    assert_ne!(first, seal(2));
    assert_eq!(
        open(1, &first).unwrap(),
        (ContentType::ApplicationData, b"GET /healthz".to_vec())
    );
    // A record replayed or moved to another place in the stream does not decrypt
    for seq in [0, 2, u64::MAX] {
        assert!(
            matches!(open(seq, &first), Err(Error::DecryptError)),
            "seq {seq}"
        );
    }
    // Nor does one with any bit flipped, in the ciphertext or the tag
    for bit in 0..first.len() * 8 {
        let mut altered = first.clone();
        altered[bit / 8] ^= 1 << (bit % 8);
        assert!(
            matches!(open(1, &altered), Err(Error::DecryptError)),
            "bit {bit}"
        );
    }
    // Nor one cut short, down to less than a tag
    for len in 0..first.len() {
        assert!(
            matches!(open(1, &first[..len]), Err(Error::DecryptError)),
            "{len} bytes"
        );
    }
}

#[test]
fn records_are_bound_to_their_place_and_content() {
    records::<aes_gcm::Aes128Gcm>();
    records::<aes_gcm::Aes256Gcm>();
    records::<chacha20poly1305::ChaCha20Poly1305>();
}

// ---------------------------------------------------------------------------------------------
// Key exchange

#[test]
fn wycheproof_x25519() {
    let tally = run("x25519_test.json", |case| {
        let private: [u8; 32] = case.hex("private").try_into().ok()?;
        let shared = case.hex("shared");
        if shared.iter().all(|&byte| byte == 0) {
            case.must_refuse("a low-order key gives an all-zero secret (RFC 8446 section 7.4.2)");
        }
        let secret = x25519_dalek::StaticSecret::from(private);
        let exchange = Box::new(X25519Exchange {
            public: x25519_dalek::PublicKey::from(&secret),
            secret,
        });
        Some(matches!(
            exchange.complete(&case.hex("public")),
            Ok(secret) if secret.secret_bytes() == shared
        ))
    });
    assert!(tally.stricter > 0, "{tally:?}");
    assert_eq!(tally.skipped, 0, "{tally:?}");
}

/// A private scalar of `len` bytes, from Wycheproof's big-endian integer (which may have a
/// leading zero byte, or lack leading zeros)
fn scalar(bytes: &[u8], len: usize) -> Option<Vec<u8>> {
    let digits = &bytes[bytes.iter().take_while(|&&byte| byte == 0).count()..];
    let mut scalar = vec![0; len.checked_sub(digits.len())?];
    scalar.extend_from_slice(digits);
    Some(scalar)
}

fn ecdh(file: &str, len: usize, exchange: impl Fn(&[u8]) -> Box<dyn ActiveKeyExchange>) {
    let tally = run(file, |case| {
        let private = scalar(&case.hex("private"), len)?;
        let public = case.hex("public");
        if public.first() != Some(&4) {
            case.must_refuse(
                "TLS 1.3 key shares are uncompressed points (RFC 8446 section 4.2.8.2)",
            );
        }
        let shared = case.hex("shared");
        Some(matches!(
            exchange(&private).complete(&public),
            Ok(secret) if secret.secret_bytes() == shared
        ))
    });
    // Points off the curve, at infinity, with bad coordinates, ...
    assert!(tally.invalid > 0, "{tally:?}");
    assert_eq!(tally.skipped, 0, "{tally:?}");
}

#[test]
fn wycheproof_p256_key_exchange() {
    ecdh("ecdh_secp256r1_ecpoint_test.json", 32, |private| {
        Box::new(Secp256r1Exchange {
            secret: p256::SecretKey::from_slice(private).unwrap(),
            public: Vec::new(),
        })
    });
}

#[test]
fn wycheproof_p384_key_exchange() {
    ecdh("ecdh_secp384r1_ecpoint_test.json", 48, |private| {
        Box::new(Secp384r1Exchange {
            secret: p384::SecretKey::from_slice(private).unwrap(),
            public: Vec::new(),
        })
    });
}

#[test]
fn key_exchanges_agree_and_draw_fresh_keys() {
    for group in provider().kx_groups {
        // Each exchange starts from a fresh random key...
        let (a, b) = (group.start().unwrap(), group.start().unwrap());
        let (a_key, b_key) = (a.pub_key().to_vec(), b.pub_key().to_vec());
        assert_ne!(a_key, b_key, "{:?}", group.name());
        // ...and both sides derive the same secret
        assert_eq!(
            a.complete(&b_key).unwrap().secret_bytes(),
            b.complete(&a_key).unwrap().secret_bytes(),
            "{:?}",
            group.name()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Signatures

fn ecdsa(file: &str, hash: &str, algorithm: &EcdsaVerify) {
    let tally = run(file, |case| {
        assert_eq!(case.group["sha"], hash, "{file}");
        let key = hex(&case.group["publicKey"]["uncompressed"]);
        Some(
            algorithm
                .verify_signature(&key, &case.hex("msg"), &case.hex("sig"))
                .is_ok(),
        )
    });
    // Forged signatures and malformed DER among them
    assert!(tally.invalid > 0, "{tally:?}");
    assert_eq!(tally.skipped, 0, "{tally:?}");
}

#[test]
fn wycheproof_ecdsa_p256_sha256() {
    ecdsa(
        "ecdsa_secp256r1_sha256_test.json",
        "SHA-256",
        &ECDSA_P256_SHA256,
    );
}

#[test]
fn wycheproof_ecdsa_p384_sha256() {
    ecdsa(
        "ecdsa_secp384r1_sha256_test.json",
        "SHA-256",
        &ECDSA_P384_SHA256,
    );
}

#[test]
fn wycheproof_ecdsa_p384_sha384() {
    ecdsa(
        "ecdsa_secp384r1_sha384_test.json",
        "SHA-384",
        &ECDSA_P384_SHA384,
    );
}

#[test]
fn wycheproof_ed25519() {
    let tally = run("ed25519_test.json", |case| {
        let key = hex(&case.group["publicKey"]["pk"]);
        Some(
            ED25519
                .verify_signature(&key, &case.hex("msg"), &case.hex("sig"))
                .is_ok(),
        )
    });
    assert!(tally.invalid > 0, "{tally:?}");
    assert_eq!(tally.skipped, 0, "{tally:?}");
}

// ---------------------------------------------------------------------------------------------
// HMAC and HKDF

fn hmac(file: &str, hmac: &dyn rustls_hmac::Hmac) {
    let tally = run(file, |case| {
        let size = case.group["tagSize"].as_u64().expect("a tag size") as usize / 8;
        let tag = hmac.with_key(&case.hex("key")).sign(&[&case.hex("msg")]);
        // Wycheproof's truncated tags are compared as such
        Some(tag.as_ref()[..size] == case.hex("tag")[..])
    });
    assert!(tally.invalid > 0, "{tally:?}");
}

#[test]
fn wycheproof_hmac_sha256() {
    hmac("hmac_sha256_test.json", &HMAC_SHA256);
}

#[test]
fn wycheproof_hmac_sha384() {
    hmac("hmac_sha384_test.json", &HMAC_SHA384);
}

/// HKDF as the suite derives its keys with it
fn hkdf(file: &str, suite: SupportedCipherSuite) {
    let hkdf = suite.tls13().expect("a TLS 1.3 suite").hkdf_provider;
    let tally = run(file, |case| {
        let size = case.test["size"].as_u64().expect("an output size") as usize;
        let mut okm = vec![0; size];
        let expander = hkdf.extract_from_secret(Some(&case.hex("salt")), &case.hex("ikm"));
        Some(
            expander
                .expand_slice(&[&case.hex("info")], &mut okm)
                .is_ok()
                && okm == case.hex("okm"),
        )
    });
    // Outputs longer than HKDF can give
    assert!(tally.invalid > 0, "{tally:?}");
}

#[test]
fn wycheproof_hkdf_sha256() {
    hkdf("hkdf_sha256_test.json", TLS13_AES_128_GCM_SHA256);
    hkdf("hkdf_sha256_test.json", TLS13_CHACHA20_POLY1305_SHA256);
}

#[test]
fn wycheproof_hkdf_sha384() {
    hkdf("hkdf_sha384_test.json", TLS13_AES_256_GCM_SHA384);
}

// ---------------------------------------------------------------------------------------------
// Randomness and secrets

#[test]
fn randomness_comes_from_the_operating_system() {
    // rustls draws its handshake randoms from the provider's `secure_random`, which is
    // `Random`; it and `random` (for key shares) both call getrandom, the OS's generator
    assert_eq!(format!("{:?}", provider().secure_random), "Random");
    let (mut a, mut b) = ([0; 32], [0; 32]);
    Random.fill(&mut a).unwrap();
    Random.fill(&mut b).unwrap();
    assert_ne!(a, b);
    assert_ne!(a, [0; 32]);
    assert_ne!(*random::<32>().unwrap(), *random::<32>().unwrap());
}

/// Every type holding a key or a secret wipes it when dropped. rustls wipes its own
/// (`AeadKey`, `SharedSecret`, the traffic secrets); these are the provider's.
#[test]
fn secrets_are_wiped_when_dropped() {
    fn wiped<T: ZeroizeOnDrop>() {}
    // Record protection: each connection's keys, and AES's key schedule
    wiped::<aes_gcm::Aes128Gcm>();
    wiped::<aes_gcm::Aes256Gcm>();
    wiped::<chacha20poly1305::ChaCha20Poly1305>();
    // Key exchange
    wiped::<x25519_dalek::StaticSecret>();
    wiped::<x25519_dalek::SharedSecret>();
    wiped::<p256::SecretKey>();
    wiped::<p384::SecretKey>();
    wiped::<p256::ecdh::SharedSecret>();
    wiped::<p384::ecdh::SharedSecret>();
    // The server's key
    wiped::<p256::ecdsa::SigningKey>();
    wiped::<p384::ecdsa::SigningKey>();
    wiped::<ed25519_dalek::SigningKey>();
    // Fresh random bytes before they become one of the above
    wiped::<Zeroizing<[u8; 32]>>();
}
