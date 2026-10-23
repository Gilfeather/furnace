//! A pure-Rust crypto provider for [rustls], built on [RustCrypto] primitives: no C, no
//! assembly, so it builds wherever Rust does.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use rustls::pki_types::pem::PemObject;
//! use rustls::pki_types::{CertificateDer, PrivateKeyDer};
//!
//! let provider = Arc::new(rustls_pure::provider());
//! // A server
//! let certs = CertificateDer::pem_file_iter("cert.pem")?.collect::<Result<Vec<_>, _>>()?;
//! let key = PrivateKeyDer::from_pem_file("key.pem")?;
//! let server = rustls::ServerConfig::builder_with_provider(provider.clone())
//!     .with_protocol_versions(&[&rustls::version::TLS13])?
//!     .with_no_client_auth()
//!     .with_single_cert(certs, key)?;
//! // A client
//! let mut roots = rustls::RootCertStore::empty();
//! roots.add(CertificateDer::from_pem_file("ca.pem")?)?;
//! let client = rustls::ClientConfig::builder_with_provider(provider)
//!     .with_protocol_versions(&[&rustls::version::TLS13])?
//!     .with_root_certificates(roots)
//!     .with_no_client_auth();
//! # let _ = (server, client);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! rustls does the TLS protocol itself; a provider only supplies the primitives. The ones here
//! are deliberately few, which keeps the code small enough to review:
//!
//! - TLS 1.3 only, with AES-128-GCM, AES-256-GCM and ChaCha20-Poly1305
//! - key exchange with X25519MLKEM768 (post-quantum, preferred), X25519, P-256 and P-384
//! - server keys and client certificates (for mutual TLS) using ECDSA P-256 / P-384 or Ed25519
//!
//! RSA is not offered, for keys or for verifying certificates: the pure-Rust `rsa` crate has an
//! unfixed timing side channel (RUSTSEC-2023-0071). Certificate chains signed with RSA are
//! refused, so a client for arbitrary public websites, many of which still use RSA, is not what
//! this is for; connections where you choose the certificates are. Certificate authorities,
//! cert-manager and Let's Encrypt all issue ECDSA certificates.
//!
//! The provider has not had an outside security audit. It is checked mechanically instead:
//! its tests run Google's [Wycheproof](https://github.com/C2SP/wycheproof) test vectors for
//! every algorithm it offers, and pin nonce construction, record integrity, the source of
//! randomness and the wiping of keys. Where the standards allow a choice it refuses: X25519
//! exchanges that end in an all-zero secret, compressed or malformed key shares, and malleable
//! Ed25519 signatures.
//!
//! [rustls]: https://docs.rs/rustls
//! [RustCrypto]: https://github.com/RustCrypto

#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;

use aes_gcm::aead::consts::{U12, U16};
use aes_gcm::aead::{AeadCore, AeadInOut, KeyInit, KeySizeUser};
use ed25519_dalek::pkcs8::DecodePrivateKey as _;
use ed25519_dalek::pkcs8::{ObjectIdentifier, PrivateKeyInfoRef};
// zeroize through the curve crates' re-export, which is the version they use themselves
use ml_kem::{Decapsulate as _, KeyExport as _};
use p256::ecdsa::signature::hazmat::PrehashVerifier;
use p256::elliptic_curve::zeroize::Zeroizing;
use rustls::crypto::cipher::{
    AeadKey, InboundOpaqueMessage, InboundPlainMessage, Iv, MessageDecrypter, MessageEncrypter,
    Nonce, OutboundOpaqueMessage, OutboundPlainMessage, PrefixedPayload, Tls13AeadAlgorithm,
    UnsupportedOperationError, make_tls13_aad,
};
use rustls::crypto::{
    ActiveKeyExchange, CompletedKeyExchange, CryptoProvider, GetRandomFailed, KeyProvider,
    SecureRandom, SharedSecret, SupportedKxGroup, WebPkiSupportedAlgorithms, hash,
    hmac as rustls_hmac, tls13,
};
use rustls::pki_types::{
    AlgorithmIdentifier, InvalidSignature, PrivateKeyDer, SignatureVerificationAlgorithm, alg_id,
};
use rustls::sign::{Signer, SigningKey};
use rustls::{
    CipherSuite, CipherSuiteCommon, ConnectionTrafficSecrets, ContentType, Error, NamedGroup,
    ProtocolVersion, SignatureAlgorithm, SignatureScheme, SupportedCipherSuite, Tls13CipherSuite,
};
use sha2::Digest as _;

/// The provider: pass it to `rustls::ServerConfig::builder_with_provider` or
/// `rustls::ClientConfig::builder_with_provider`, with TLS 1.3 as the protocol version
pub fn provider() -> CryptoProvider {
    CryptoProvider {
        cipher_suites: vec![
            TLS13_AES_128_GCM_SHA256,
            TLS13_AES_256_GCM_SHA384,
            TLS13_CHACHA20_POLY1305_SHA256,
        ],
        // The post-quantum hybrid first, as rustls's own default provider orders them
        kx_groups: vec![&X25519MLKEM768, &X25519, &SECP256R1, &SECP384R1],
        signature_verification_algorithms: SUPPORTED_SIG_ALGS,
        secure_random: &Random,
        key_provider: &Keys,
    }
}

// ---------------------------------------------------------------------------------------------
// Randomness

#[derive(Debug)]
struct Random;

impl SecureRandom for Random {
    fn fill(&self, buf: &mut [u8]) -> Result<(), GetRandomFailed> {
        getrandom::fill(buf).map_err(|_| GetRandomFailed)
    }
}

/// Fresh secret bytes, wiped when dropped
fn random<const N: usize>() -> Result<Zeroizing<[u8; N]>, Error> {
    let mut bytes = Zeroizing::new([0; N]);
    getrandom::fill(&mut *bytes).map_err(|_| Error::FailedToGetRandomBytes)?;
    Ok(bytes)
}

// ---------------------------------------------------------------------------------------------
// Hashes and HMAC

struct Sha<D>(std::marker::PhantomData<fn() -> D>);

static SHA256: Sha<sha2::Sha256> = Sha(std::marker::PhantomData);
static SHA384: Sha<sha2::Sha384> = Sha(std::marker::PhantomData);

trait ShaAlgorithm: sha2::Digest + Clone + Send + Sync + 'static {
    const ALGORITHM: rustls::crypto::hash::HashAlgorithm;
}

impl ShaAlgorithm for sha2::Sha256 {
    const ALGORITHM: hash::HashAlgorithm = hash::HashAlgorithm::SHA256;
}

impl ShaAlgorithm for sha2::Sha384 {
    const ALGORITHM: hash::HashAlgorithm = hash::HashAlgorithm::SHA384;
}

impl<D: ShaAlgorithm> hash::Hash for Sha<D> {
    fn start(&self) -> Box<dyn hash::Context> {
        Box::new(ShaContext(D::new()))
    }

    fn hash(&self, data: &[u8]) -> hash::Output {
        hash::Output::new(&D::digest(data))
    }

    fn output_len(&self) -> usize {
        <D as sha2::Digest>::output_size()
    }

    fn algorithm(&self) -> hash::HashAlgorithm {
        D::ALGORITHM
    }
}

struct ShaContext<D>(D);

impl<D: ShaAlgorithm> hash::Context for ShaContext<D> {
    fn fork_finish(&self) -> hash::Output {
        hash::Output::new(&self.0.clone().finalize())
    }

    fn fork(&self) -> Box<dyn hash::Context> {
        Box::new(ShaContext(self.0.clone()))
    }

    fn finish(self: Box<Self>) -> hash::Output {
        hash::Output::new(&self.0.finalize())
    }

    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
}

struct Hmac<D>(std::marker::PhantomData<fn() -> D>);

static HMAC_SHA256: Hmac<sha2::Sha256> = Hmac(std::marker::PhantomData);
static HMAC_SHA384: Hmac<sha2::Sha384> = Hmac(std::marker::PhantomData);

impl<D> rustls_hmac::Hmac for Hmac<D>
where
    D: ShaAlgorithm + hmac::EagerHash,
    hmac::Hmac<D>: KeyInit + hmac::Mac + Clone + Send + Sync,
{
    fn with_key(&self, key: &[u8]) -> Box<dyn rustls_hmac::Key> {
        // HMAC takes keys of any length
        Box::new(HmacKey(
            <hmac::Hmac<D> as KeyInit>::new_from_slice(key).expect("HMAC accepts any key length"),
        ))
    }

    fn hash_output_len(&self) -> usize {
        <D as sha2::Digest>::output_size()
    }
}

struct HmacKey<M>(M);

impl<M: hmac::Mac + Clone + Send + Sync> rustls_hmac::Key for HmacKey<M> {
    fn sign_concat(&self, first: &[u8], middle: &[&[u8]], last: &[u8]) -> rustls_hmac::Tag {
        let mut mac = self.0.clone();
        mac.update(first);
        for part in middle {
            mac.update(part);
        }
        mac.update(last);
        rustls_hmac::Tag::new(&mac.finalize().into_bytes())
    }

    fn tag_len(&self) -> usize {
        <M as hmac::digest::OutputSizeUser>::output_size()
    }
}

// ---------------------------------------------------------------------------------------------
// Cipher suites (TLS 1.3)

/// AES-128-GCM with SHA-256
static TLS13_AES_128_GCM_SHA256: SupportedCipherSuite =
    SupportedCipherSuite::Tls13(&Tls13CipherSuite {
        common: CipherSuiteCommon {
            suite: CipherSuite::TLS13_AES_128_GCM_SHA256,
            hash_provider: &SHA256,
            // RFC 8446 section 5.5: 2^24.5 full-size records per key for AES-GCM, rounded down
            confidentiality_limit: 1 << 24,
        },
        hkdf_provider: &tls13::HkdfUsingHmac(&HMAC_SHA256),
        aead_alg: &Aead::<aes_gcm::Aes128Gcm>(std::marker::PhantomData),
        quic: None,
    });

/// AES-256-GCM with SHA-384
static TLS13_AES_256_GCM_SHA384: SupportedCipherSuite =
    SupportedCipherSuite::Tls13(&Tls13CipherSuite {
        common: CipherSuiteCommon {
            suite: CipherSuite::TLS13_AES_256_GCM_SHA384,
            hash_provider: &SHA384,
            confidentiality_limit: 1 << 24,
        },
        hkdf_provider: &tls13::HkdfUsingHmac(&HMAC_SHA384),
        aead_alg: &Aead::<aes_gcm::Aes256Gcm>(std::marker::PhantomData),
        quic: None,
    });

/// ChaCha20-Poly1305 with SHA-256
static TLS13_CHACHA20_POLY1305_SHA256: SupportedCipherSuite =
    SupportedCipherSuite::Tls13(&Tls13CipherSuite {
        common: CipherSuiteCommon {
            suite: CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
            hash_provider: &SHA256,
            confidentiality_limit: u64::MAX,
        },
        hkdf_provider: &tls13::HkdfUsingHmac(&HMAC_SHA256),
        aead_alg: &Aead::<chacha20poly1305::ChaCha20Poly1305>(std::marker::PhantomData),
        quic: None,
    });

/// A TLS 1.3 AEAD: 12-byte nonces and 16-byte tags, as all three suites have
struct Aead<C>(std::marker::PhantomData<fn() -> C>);

/// What the AEAD must provide: 12-byte nonces and 16-byte tags, as types, so building a nonce
/// needs no check at run time; the key sizes are checked below, at compile time
trait Cipher:
    AeadInOut + AeadCore<NonceSize = U12, TagSize = U16> + KeyInit + Send + Sync + 'static
{
    /// The key length rustls is told, so the keys it derives always fit `new_from_slice`
    const KEY_LEN: usize =
        <<Self as KeySizeUser>::KeySize as aes_gcm::aead::array::typenum::Unsigned>::USIZE;
    fn traffic_secrets(key: AeadKey, iv: Iv) -> ConnectionTrafficSecrets;
}

impl Cipher for aes_gcm::Aes128Gcm {
    fn traffic_secrets(key: AeadKey, iv: Iv) -> ConnectionTrafficSecrets {
        ConnectionTrafficSecrets::Aes128Gcm { key, iv }
    }
}

impl Cipher for aes_gcm::Aes256Gcm {
    fn traffic_secrets(key: AeadKey, iv: Iv) -> ConnectionTrafficSecrets {
        ConnectionTrafficSecrets::Aes256Gcm { key, iv }
    }
}

impl Cipher for chacha20poly1305::ChaCha20Poly1305 {
    fn traffic_secrets(key: AeadKey, iv: Iv) -> ConnectionTrafficSecrets {
        ConnectionTrafficSecrets::Chacha20Poly1305 { key, iv }
    }
}

const TAG_LEN: usize = 16;

impl<C: Cipher> Tls13AeadAlgorithm for Aead<C> {
    fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageEncrypter> {
        Box::new(Crypter::<C>::new(key.as_ref(), iv))
    }

    fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageDecrypter> {
        Box::new(Crypter::<C>::new(key.as_ref(), iv))
    }

    fn key_len(&self) -> usize {
        C::KEY_LEN
    }

    fn extract_keys(
        &self,
        key: AeadKey,
        iv: Iv,
    ) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
        Ok(C::traffic_secrets(key, iv))
    }
}

struct Crypter<C> {
    cipher: C,
    iv: Iv,
}

impl<C: Cipher> Crypter<C> {
    fn new(key: &[u8], iv: Iv) -> Self {
        Self {
            cipher: C::new_from_slice(key).expect("key of the suite's length"),
            iv,
        }
    }

    /// The record's nonce: the IV XORed with its sequence number (RFC 8446 section 5.3), so no
    /// two records under one key share one
    fn nonce(&self, seq: u64) -> aes_gcm::aead::Nonce<C> {
        Nonce::new(&self.iv, seq).0.into()
    }

    /// Encrypts record `seq` in place and returns its tag
    fn seal(&self, seq: u64, aad: &[u8], text: &mut [u8]) -> Result<aes_gcm::aead::Tag<C>, Error> {
        self.cipher
            .encrypt_inout_detached(&self.nonce(seq), aad, text.into())
            .map_err(|_| Error::EncryptError)
    }

    /// Decrypts record `seq` in place if `tag` is right; the AEAD compares tags in constant time
    fn open(&self, seq: u64, aad: &[u8], text: &mut [u8], tag: &[u8]) -> Result<(), Error> {
        let tag = aes_gcm::aead::Tag::<C>::try_from(tag).map_err(|_| Error::DecryptError)?;
        self.cipher
            .decrypt_inout_detached(&self.nonce(seq), aad, text.into(), &tag)
            .map_err(|_| Error::DecryptError)
    }
}

impl<C: Cipher> MessageEncrypter for Crypter<C> {
    fn encrypt(
        &mut self,
        msg: OutboundPlainMessage<'_>,
        seq: u64,
    ) -> Result<OutboundOpaqueMessage, Error> {
        let total_len = self.encrypted_payload_len(msg.payload.len());
        let mut payload = PrefixedPayload::with_capacity(total_len);
        payload.extend_from_chunks(&msg.payload);
        // TLSInnerPlaintext: the content, then its real type (RFC 8446 section 5.2)
        payload.extend_from_slice(&msg.typ.to_array());
        let tag = self.seal(seq, &make_tls13_aad(total_len), payload.as_mut())?;
        payload.extend_from_slice(&tag);
        Ok(OutboundOpaqueMessage::new(
            ContentType::ApplicationData,
            // TLS 1.3 records carry the legacy version 1.2 (RFC 8446 section 5.1)
            ProtocolVersion::TLSv1_2,
            payload,
        ))
    }

    fn encrypted_payload_len(&self, payload_len: usize) -> usize {
        sealed_payload_len(payload_len)
    }
}

/// A record's payload for `text_len` bytes of content: the content, its type, then the tag
fn sealed_payload_len(text_len: usize) -> usize {
    text_len + 1 + TAG_LEN
}

/// How much of a record's payload of `payload_len` bytes is the encrypted text, before its tag;
/// a payload too short for a tag is refused
fn sealed_text_len(payload_len: usize) -> Option<usize> {
    payload_len.checked_sub(TAG_LEN)
}

impl<C: Cipher> MessageDecrypter for Crypter<C> {
    fn decrypt<'a>(
        &mut self,
        mut msg: InboundOpaqueMessage<'a>,
        seq: u64,
    ) -> Result<InboundPlainMessage<'a>, Error> {
        let payload = &mut msg.payload;
        let plain_len = sealed_text_len(payload.len()).ok_or(Error::DecryptError)?;
        let aad = make_tls13_aad(payload.len());
        let (text, tag) = payload.split_at_mut(plain_len);
        self.open(seq, &aad, text, tag)?;
        payload.truncate(plain_len);
        msg.into_tls13_unpadded_message()
    }
}

// The sizes the code above relies on, for every suite: the tag `TAG_LEN` stands for, and the
// key that RFC 8446 gives the suite (rustls derives keys of `KEY_LEN`, so the `expect`s on
// `new_from_slice` cannot fail)
const _: () = {
    use aes_gcm::aead::array::typenum::Unsigned;
    macro_rules! check {
        ($($cipher:ty => $key:expr),*) => {$(
            assert!(<$cipher as Cipher>::KEY_LEN == $key);
            assert!(<<$cipher as AeadCore>::TagSize as Unsigned>::USIZE == TAG_LEN);
        )*};
    }
    check!(
        aes_gcm::Aes128Gcm => 16,
        aes_gcm::Aes256Gcm => 32,
        chacha20poly1305::ChaCha20Poly1305 => 32
    );
};

// ---------------------------------------------------------------------------------------------
// Key exchange

// The shape of the peer's key share, checked here before any of it reaches a curve or ML-KEM:
// apart from the cryptography, and without rustls's `Error` (whose drop Kani cannot bound), so
// Kani proves these on every input (`mod verification`)

fn invalid_key_share() -> Error {
    rustls::PeerMisbehaved::InvalidKeyShare.into()
}

/// An X25519 share: 32 bytes
fn x25519_share(share: &[u8]) -> Option<[u8; X25519_SHARE_LEN]> {
    share.try_into().ok()
}

/// A P-256 or P-384 share: an uncompressed point, 4 then both coordinates, as TLS 1.3 requires
/// (RFC 8446 section 4.2.8.2); compressed points and the point at infinity are refused
fn ec_share(share: &[u8], coordinate_len: usize) -> Option<&[u8]> {
    match share {
        [4, coordinates @ ..] if coordinates.len() == 2 * coordinate_len => Some(share),
        _ => None,
    }
}

/// An X25519MLKEM768 share: ML-KEM's part of `mlkem_len` bytes (the client's encapsulation key or
/// the server's ciphertext), then X25519's, and nothing else
fn hybrid_share(share: &[u8], mlkem_len: usize) -> Option<(&[u8], &[u8])> {
    (share.len() == mlkem_len + X25519_SHARE_LEN).then(|| share.split_at(mlkem_len))
}

/// `first`, then `second`: the hybrid's shares and its secret
fn concat(first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(first.len() + second.len());
    bytes.extend_from_slice(first);
    bytes.extend_from_slice(second);
    bytes
}

#[derive(Debug)]
struct X25519Group;
static X25519: X25519Group = X25519Group;

impl SupportedKxGroup for X25519Group {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        Ok(Box::new(X25519Exchange::new()?))
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::X25519
    }
}

struct X25519Exchange {
    secret: x25519_dalek::StaticSecret,
    public: x25519_dalek::PublicKey,
}

impl X25519Exchange {
    /// A fresh random key
    fn new() -> Result<Self, Error> {
        let secret = x25519_dalek::StaticSecret::from(*random::<32>()?);
        let public = x25519_dalek::PublicKey::from(&secret);
        Ok(Self { secret, public })
    }
}

impl ActiveKeyExchange for X25519Exchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
        let peer = x25519_share(peer_pub_key).ok_or_else(invalid_key_share)?;
        let shared = self
            .secret
            .diffie_hellman(&x25519_dalek::PublicKey::from(peer));
        // An all-zero result means a low-order peer key (RFC 8446 section 7.4.2)
        if !shared.was_contributory() {
            return Err(invalid_key_share());
        }
        Ok(SharedSecret::from(&shared.as_bytes()[..]))
    }

    fn pub_key(&self) -> &[u8] {
        self.public.as_bytes()
    }

    fn group(&self) -> NamedGroup {
        NamedGroup::X25519
    }
}

#[derive(Debug)]
struct Secp256r1Group;
static SECP256R1: Secp256r1Group = Secp256r1Group;

impl SupportedKxGroup for Secp256r1Group {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        // A random scalar; out-of-range values (chance below 2^-32) are drawn again
        let secret = loop {
            if let Ok(secret) = p256::SecretKey::from_slice(random::<32>()?.as_slice()) {
                break secret;
            }
        };
        let public = secret.public_key().to_sec1_bytes().into_vec();
        Ok(Box::new(Secp256r1Exchange { secret, public }))
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::secp256r1
    }
}

struct Secp256r1Exchange {
    secret: p256::SecretKey,
    /// Uncompressed, as TLS requires
    public: Vec<u8>,
}

impl ActiveKeyExchange for Secp256r1Exchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
        let peer = p256::PublicKey::from_sec1_bytes(
            ec_share(peer_pub_key, 32).ok_or_else(invalid_key_share)?,
        )
        .map_err(|_| invalid_key_share())?;
        let shared = self.secret.diffie_hellman(&peer);
        Ok(SharedSecret::from(&shared.raw_secret_bytes()[..]))
    }

    fn pub_key(&self) -> &[u8] {
        &self.public
    }

    fn group(&self) -> NamedGroup {
        NamedGroup::secp256r1
    }
}

#[derive(Debug)]
struct Secp384r1Group;
static SECP384R1: Secp384r1Group = Secp384r1Group;

impl SupportedKxGroup for Secp384r1Group {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        let secret = loop {
            if let Ok(secret) = p384::SecretKey::from_slice(random::<48>()?.as_slice()) {
                break secret;
            }
        };
        let public = secret.public_key().to_sec1_bytes().into_vec();
        Ok(Box::new(Secp384r1Exchange { secret, public }))
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::secp384r1
    }
}

struct Secp384r1Exchange {
    secret: p384::SecretKey,
    public: Vec<u8>,
}

impl ActiveKeyExchange for Secp384r1Exchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
        let peer = p384::PublicKey::from_sec1_bytes(
            ec_share(peer_pub_key, 48).ok_or_else(invalid_key_share)?,
        )
        .map_err(|_| invalid_key_share())?;
        let shared = self.secret.diffie_hellman(&peer);
        Ok(SharedSecret::from(&shared.raw_secret_bytes()[..]))
    }

    fn pub_key(&self) -> &[u8] {
        &self.public
    }

    fn group(&self) -> NamedGroup {
        NamedGroup::secp384r1
    }
}

/// X25519MLKEM768 (draft-ietf-tls-ecdhe-mlkem): ML-KEM-768 (FIPS 203) and X25519 at once, so
/// the secret stays safe as long as either holds, including against a future quantum computer.
/// Shares and the secret put ML-KEM's part first and X25519's second, as rustls's own providers
/// do. The X25519 part is checked as the plain X25519 group checks it.
#[derive(Debug)]
struct X25519MlKem768Group;
static X25519MLKEM768: X25519MlKem768Group = X25519MlKem768Group;

/// The client's ML-KEM-768 share: its encapsulation key
const MLKEM768_ENCAPSULATION_KEY_LEN: usize = 1184;
/// The server's ML-KEM-768 share: the ciphertext
const MLKEM768_CIPHERTEXT_LEN: usize = 1088;
const X25519_SHARE_LEN: usize = 32;

impl X25519MlKem768Group {
    /// The client's side, from an ML-KEM seed (64 random bytes) and an X25519 key
    fn start_with(seed: ml_kem::Seed, x25519: X25519Exchange) -> X25519MlKem768Exchange {
        let decapsulation = ml_kem::DecapsulationKey::<ml_kem::MlKem768>::from_seed(seed);
        let share = concat(
            &decapsulation.encapsulation_key().to_bytes(),
            x25519.public.as_bytes(),
        );
        X25519MlKem768Exchange {
            decapsulation,
            x25519,
            share,
        }
    }

    /// The server's side: encapsulates to the client's ML-KEM key with `m` (32 random bytes),
    /// and does X25519 with `x25519`
    fn respond(
        client_share: &[u8],
        m: &ml_kem::B32,
        x25519: X25519Exchange,
    ) -> Result<CompletedKeyExchange, Error> {
        let (key, x25519_share) = hybrid_share(client_share, MLKEM768_ENCAPSULATION_KEY_LEN)
            .ok_or_else(invalid_key_share)?;
        let key = ml_kem::Key::<ml_kem::EncapsulationKey<ml_kem::MlKem768>>::try_from(key)
            .map_err(|_| invalid_key_share())?;
        // Refuses keys whose coefficients are not reduced (FIPS 203 section 7.2)
        let key = ml_kem::EncapsulationKey::<ml_kem::MlKem768>::new(&key)
            .map_err(|_| invalid_key_share())?;
        let (ciphertext, mlkem_secret) = key.encapsulate_deterministic(m);
        let mlkem_secret = Zeroizing::new(mlkem_secret);
        let pub_key = concat(&ciphertext, x25519.public.as_bytes());
        let x25519_secret = Box::new(x25519).complete(x25519_share)?;
        Ok(CompletedKeyExchange {
            group: NamedGroup::X25519MLKEM768,
            pub_key,
            secret: hybrid_secret(&mlkem_secret, x25519_secret.secret_bytes()),
        })
    }
}

/// ML-KEM's secret, then X25519's
fn hybrid_secret(mlkem: &[u8], x25519: &[u8]) -> SharedSecret {
    // rustls's SharedSecret wipes it when dropped
    SharedSecret::from(concat(mlkem, x25519))
}

impl SupportedKxGroup for X25519MlKem768Group {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        let seed = random::<64>()?;
        Ok(Box::new(Self::start_with(
            ml_kem::Seed::from(*seed),
            X25519Exchange::new()?,
        )))
    }

    fn start_and_complete(&self, client_share: &[u8]) -> Result<CompletedKeyExchange, Error> {
        let m = random::<32>()?;
        Self::respond(client_share, &ml_kem::B32::from(*m), X25519Exchange::new()?)
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::X25519MLKEM768
    }

    fn usable_for_version(&self, version: ProtocolVersion) -> bool {
        version == ProtocolVersion::TLSv1_3
    }
}

struct X25519MlKem768Exchange {
    decapsulation: ml_kem::DecapsulationKey<ml_kem::MlKem768>,
    x25519: X25519Exchange,
    /// The encapsulation key, then the X25519 public key
    share: Vec<u8>,
}

impl ActiveKeyExchange for X25519MlKem768Exchange {
    fn complete(self: Box<Self>, server_share: &[u8]) -> Result<SharedSecret, Error> {
        let (ciphertext, x25519_share) =
            hybrid_share(server_share, MLKEM768_CIPHERTEXT_LEN).ok_or_else(invalid_key_share)?;
        let ciphertext = ml_kem::Ciphertext::<ml_kem::MlKem768>::try_from(ciphertext)
            .map_err(|_| invalid_key_share())?;
        // A ciphertext that was tampered with gives an unrelated secret (FIPS 203's implicit
        // rejection), so the handshake fails at the Finished messages
        let mlkem_secret = Zeroizing::new(self.decapsulation.decapsulate(&ciphertext));
        let x25519_secret = Box::new(self.x25519).complete(x25519_share)?;
        Ok(hybrid_secret(&mlkem_secret, x25519_secret.secret_bytes()))
    }

    /// The X25519 part is offered on its own too, so a server without ML-KEM can pick it with
    /// no HelloRetryRequest
    fn hybrid_component(&self) -> Option<(NamedGroup, &[u8])> {
        Some((NamedGroup::X25519, self.x25519.public.as_bytes()))
    }

    fn complete_hybrid_component(
        self: Box<Self>,
        peer_pub_key: &[u8],
    ) -> Result<SharedSecret, Error> {
        Box::new(self.x25519).complete(peer_pub_key)
    }

    fn pub_key(&self) -> &[u8] {
        &self.share
    }

    fn group(&self) -> NamedGroup {
        NamedGroup::X25519MLKEM768
    }
}

// ---------------------------------------------------------------------------------------------
// The server's private key

#[derive(Debug)]
struct Keys;

impl KeyProvider for Keys {
    fn load_private_key(
        &self,
        key_der: PrivateKeyDer<'static>,
    ) -> Result<Arc<dyn SigningKey>, Error> {
        let unsupported = |what: &str| {
            Error::General(format!(
                "{what}; rustls-pure takes ECDSA P-256 / P-384 or Ed25519 keys"
            ))
        };
        let key: Arc<dyn SigningKey> = match &key_der {
            PrivateKeyDer::Pkcs8(der) => {
                let der = der.secret_pkcs8_der();
                if let Ok(key) = p256::ecdsa::SigningKey::from_pkcs8_der(der) {
                    Arc::new(EcdsaKey::P256(Arc::new(key)))
                } else if let Ok(key) = p384::ecdsa::SigningKey::from_pkcs8_der(der) {
                    Arc::new(EcdsaKey::P384(Arc::new(key)))
                } else if let Ok(key) = ed25519_dalek::SigningKey::from_pkcs8_der(der) {
                    Arc::new(Ed25519Key(Arc::new(key)))
                } else {
                    return Err(unsupported(&unsupported_pkcs8(der)));
                }
            }
            PrivateKeyDer::Sec1(der) => {
                let der = der.secret_sec1_der();
                if let Ok(key) = p256::SecretKey::from_sec1_der(der) {
                    Arc::new(EcdsaKey::P256(Arc::new(key.into())))
                } else if let Ok(key) = p384::SecretKey::from_sec1_der(der) {
                    Arc::new(EcdsaKey::P384(Arc::new(key.into())))
                } else {
                    return Err(unsupported("unsupported EC key curve"));
                }
            }
            PrivateKeyDer::Pkcs1(_) => return Err(unsupported("RSA keys are not supported")),
            _ => return Err(unsupported("unsupported key format")),
        };
        Ok(key)
    }
}

/// What is wrong with a PKCS#8 key none of the parsers took: its algorithm, from the key itself
fn unsupported_pkcs8(der: &[u8]) -> String {
    const RSA: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.113549.1.1.1");
    const EC: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
    let Ok(info) = PrivateKeyInfoRef::try_from(der) else {
        return "not a valid PKCS#8 key".to_string();
    };
    let algorithm = info.algorithm;
    if algorithm.oid == RSA {
        "RSA keys are not supported".to_string()
    } else if algorithm.oid == EC {
        match algorithm.parameters_oid() {
            Ok(curve) => format!("unsupported EC key curve {curve}"),
            Err(_) => "an EC key without a named curve".to_string(),
        }
    } else {
        format!("unsupported PKCS#8 key algorithm {}", algorithm.oid)
    }
}

/// The key, shared with the signer rustls asks for in each handshake rather than copied into it,
/// so the secret is in memory once
#[derive(Clone)]
enum EcdsaKey {
    P256(Arc<p256::ecdsa::SigningKey>),
    P384(Arc<p384::ecdsa::SigningKey>),
}

impl fmt::Debug for EcdsaKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print key material
        f.write_str(match self {
            Self::P256(_) => "EcdsaKey(P-256)",
            Self::P384(_) => "EcdsaKey(P-384)",
        })
    }
}

impl EcdsaKey {
    fn scheme(&self) -> SignatureScheme {
        match self {
            Self::P256(_) => SignatureScheme::ECDSA_NISTP256_SHA256,
            Self::P384(_) => SignatureScheme::ECDSA_NISTP384_SHA384,
        }
    }
}

impl SigningKey for EcdsaKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        offered
            .contains(&self.scheme())
            .then(|| Box::new(EcdsaSigner(self.clone())) as Box<dyn Signer>)
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::ECDSA
    }
}

#[derive(Debug)]
struct EcdsaSigner(EcdsaKey);

impl Signer for EcdsaSigner {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        use p256::ecdsa::signature::Signer as _;
        // Deterministic nonces (RFC 6979); DER-encoded, as TLS sends them
        let der = match &self.0 {
            EcdsaKey::P256(key) => {
                let signature: p256::ecdsa::Signature = key
                    .try_sign(message)
                    .map_err(|e| Error::General(e.to_string()))?;
                signature.to_der().to_bytes().into_vec()
            }
            EcdsaKey::P384(key) => {
                let signature: p384::ecdsa::Signature = key
                    .try_sign(message)
                    .map_err(|e| Error::General(e.to_string()))?;
                signature.to_der().to_bytes().into_vec()
            }
        };
        Ok(der)
    }

    fn scheme(&self) -> SignatureScheme {
        self.0.scheme()
    }
}

/// Shared with each handshake's signer, as [`EcdsaKey`] is
struct Ed25519Key(Arc<ed25519_dalek::SigningKey>);

impl fmt::Debug for Ed25519Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Ed25519Key")
    }
}

impl SigningKey for Ed25519Key {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        offered
            .contains(&SignatureScheme::ED25519)
            .then(|| Box::new(Ed25519Key(self.0.clone())) as Box<dyn Signer>)
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::ED25519
    }
}

impl Signer for Ed25519Key {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        use ed25519_dalek::Signer as _;
        Ok(self.0.sign(message).to_bytes().to_vec())
    }

    fn scheme(&self) -> SignatureScheme {
        SignatureScheme::ED25519
    }
}

// ---------------------------------------------------------------------------------------------
// Signature verification, for client certificates and their chains

/// ECDSA on `curve` with the message hashed by `hash`
#[derive(Debug)]
struct EcdsaVerify {
    curve: Curve,
    hash: Prehash,
}

#[derive(Debug, Clone, Copy)]
enum Curve {
    P256,
    P384,
}

/// The hashes ECDSA signatures are verified with; two, so every match on them is exhaustive
#[derive(Debug, Clone, Copy)]
enum Prehash {
    Sha256,
    Sha384,
}

impl Prehash {
    /// The message's digest, kept on the stack
    fn digest(self, message: &[u8]) -> hash::Output {
        match self {
            Self::Sha256 => hash::Output::new(&sha2::Sha256::digest(message)),
            Self::Sha384 => hash::Output::new(&sha2::Sha384::digest(message)),
        }
    }
}

static ECDSA_P256_SHA256: EcdsaVerify = EcdsaVerify {
    curve: Curve::P256,
    hash: Prehash::Sha256,
};
static ECDSA_P256_SHA384: EcdsaVerify = EcdsaVerify {
    curve: Curve::P256,
    hash: Prehash::Sha384,
};
static ECDSA_P384_SHA256: EcdsaVerify = EcdsaVerify {
    curve: Curve::P384,
    hash: Prehash::Sha256,
};
static ECDSA_P384_SHA384: EcdsaVerify = EcdsaVerify {
    curve: Curve::P384,
    hash: Prehash::Sha384,
};

impl SignatureVerificationAlgorithm for EcdsaVerify {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        // The key and the signature are parsed first: a malformed one is refused before the
        // message is hashed
        match self.curve {
            Curve::P256 => {
                let key = p256::ecdsa::VerifyingKey::from_sec1_bytes(public_key)
                    .map_err(|_| InvalidSignature)?;
                let signature =
                    p256::ecdsa::Signature::from_der(signature).map_err(|_| InvalidSignature)?;
                key.verify_prehash(self.hash.digest(message).as_ref(), &signature)
                    .map_err(|_| InvalidSignature)
            }
            Curve::P384 => {
                let key = p384::ecdsa::VerifyingKey::from_sec1_bytes(public_key)
                    .map_err(|_| InvalidSignature)?;
                let signature =
                    p384::ecdsa::Signature::from_der(signature).map_err(|_| InvalidSignature)?;
                key.verify_prehash(self.hash.digest(message).as_ref(), &signature)
                    .map_err(|_| InvalidSignature)
            }
        }
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        match self.curve {
            Curve::P256 => alg_id::ECDSA_P256,
            Curve::P384 => alg_id::ECDSA_P384,
        }
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        match self.hash {
            Prehash::Sha256 => alg_id::ECDSA_SHA256,
            Prehash::Sha384 => alg_id::ECDSA_SHA384,
        }
    }
}

#[derive(Debug)]
struct Ed25519Verify;
static ED25519: Ed25519Verify = Ed25519Verify;

impl SignatureVerificationAlgorithm for Ed25519Verify {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let (key, signature) = ed25519_parts(public_key, signature)?;
        let key = ed25519_dalek::VerifyingKey::from_bytes(&key).map_err(|_| InvalidSignature)?;
        let signature = ed25519_dalek::Signature::from_bytes(&signature);
        // Strict: rejects malleable signatures and small-order keys
        key.verify_strict(message, &signature)
            .map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::ED25519
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::ED25519
    }
}

/// An Ed25519 public key and signature: 32 and 64 bytes, as RFC 8032 has them; the key is the
/// raw one inside the certificate's SubjectPublicKeyInfo, which webpki takes out
fn ed25519_parts(
    public_key: &[u8],
    signature: &[u8],
) -> Result<([u8; 32], [u8; 64]), InvalidSignature> {
    match (public_key.try_into(), signature.try_into()) {
        (Ok(key), Ok(signature)) => Ok((key, signature)),
        _ => Err(InvalidSignature),
    }
}

static SUPPORTED_SIG_ALGS: WebPkiSupportedAlgorithms = WebPkiSupportedAlgorithms {
    all: &[
        &ECDSA_P256_SHA256,
        &ECDSA_P256_SHA384,
        &ECDSA_P384_SHA256,
        &ECDSA_P384_SHA384,
        &ED25519,
    ],
    mapping: &[
        (
            SignatureScheme::ECDSA_NISTP384_SHA384,
            &[&ECDSA_P384_SHA384, &ECDSA_P256_SHA384],
        ),
        (
            SignatureScheme::ECDSA_NISTP256_SHA256,
            &[&ECDSA_P256_SHA256, &ECDSA_P384_SHA256],
        ),
        (SignatureScheme::ED25519, &[&ED25519]),
    ],
};

#[cfg(test)]
mod tests;

// Kani's proofs of the code above that takes the peer's bytes: every input up to the bounds
// given, not samples. The cryptography is the RustCrypto crates' and the protocol rustls's, so
// these cover what rustls-pure itself does with a share, a record or a signature before handing
// it on. `cargo kani -p rustls-pure` (Linux; .github/workflows/kani.yml)
#[cfg(kani)]
mod verification {
    use super::*;

    /// Any length up to `N`
    fn any_len<const N: usize>() -> usize {
        kani::any_where(|&len: &usize| len <= N)
    }

    /// Exactly the 32-byte shares are taken for X25519, as they are
    #[kani::proof]
    #[kani::unwind(41)]
    fn x25519_shares_are_exactly_32_bytes() {
        let bytes: [u8; 40] = kani::any();
        let share = &bytes[..any_len::<40>()];
        match x25519_share(share) {
            Some(key) => {
                assert!(share.len() == 32);
                let i = kani::any_where(|&i: &usize| i < 32);
                assert!(key[i] == share[i]);
            }
            None => {
                assert!(share.len() != 32);
            }
        }
    }

    /// P-256 and P-384 take exactly the uncompressed points of their size, and pass them on
    /// untouched: the curve crates never see a compressed point, the point at infinity or bytes
    /// past the point
    #[kani::proof]
    fn ec_shares_are_uncompressed_points_of_the_curve() {
        let bytes: [u8; 100] = kani::any();
        let share = &bytes[..any_len::<100>()];
        let coordinate_len = if kani::any() { 32 } else { 48 };
        let uncompressed = share.len() == 1 + 2 * coordinate_len && share[0] == 4;
        match ec_share(share, coordinate_len) {
            Some(point) => {
                assert!(uncompressed);
                assert!(point.as_ptr() == share.as_ptr() && point.len() == share.len());
            }
            None => {
                assert!(!uncompressed);
            }
        }
    }

    /// A hybrid share of any length: refused unless it is ML-KEM's part and X25519's exactly,
    /// and split where ML-KEM's ends, for the client's share and the server's
    #[kani::proof]
    fn hybrid_shares_split_where_ml_kem_ends() {
        const MAX: usize = MLKEM768_ENCAPSULATION_KEY_LEN + X25519_SHARE_LEN + 8;
        // Only the length matters to the split, so the bytes need not be symbolic
        let bytes = [0u8; MAX];
        let share = &bytes[..any_len::<MAX>()];
        let mlkem_len = if kani::any() {
            MLKEM768_ENCAPSULATION_KEY_LEN
        } else {
            MLKEM768_CIPHERTEXT_LEN
        };
        match hybrid_share(share, mlkem_len) {
            Some((mlkem, x25519)) => {
                assert!(share.len() == mlkem_len + X25519_SHARE_LEN);
                assert!(mlkem.as_ptr() == share.as_ptr() && mlkem.len() == mlkem_len);
                assert!(x25519.as_ptr() == share[mlkem_len..].as_ptr());
                assert!(x25519.len() == X25519_SHARE_LEN);
            }
            None => {
                assert!(share.len() != mlkem_len + X25519_SHARE_LEN);
            }
        }
    }

    /// What one side puts together, the other takes apart into the same two parts
    fn hybrid_round_trip<const MLKEM: usize>() {
        let mlkem: [u8; MLKEM] = kani::any();
        let x25519: [u8; X25519_SHARE_LEN] = kani::any();
        let share = concat(&mlkem, &x25519);
        assert!(share.len() == MLKEM + X25519_SHARE_LEN);
        let (mlkem_part, x25519_part) = hybrid_share(&share, MLKEM).unwrap();
        let i = kani::any_where(|&i: &usize| i < MLKEM);
        assert!(mlkem_part.len() == MLKEM && mlkem_part[i] == mlkem[i]);
        let j = kani::any_where(|&j: &usize| j < X25519_SHARE_LEN);
        assert!(x25519_part.len() == X25519_SHARE_LEN && x25519_part[j] == x25519[j]);
        assert!(x25519_share(x25519_part).unwrap() == x25519);
    }

    /// The client's share (encapsulation key, then X25519's key), as the server reads it
    #[kani::proof]
    #[kani::unwind(33)]
    fn client_hybrid_shares_round_trip() {
        hybrid_round_trip::<MLKEM768_ENCAPSULATION_KEY_LEN>();
    }

    /// The server's share (ciphertext, then X25519's key), as the client reads it
    #[kani::proof]
    #[kani::unwind(33)]
    fn server_hybrid_shares_round_trip() {
        hybrid_round_trip::<MLKEM768_CIPHERTEXT_LEN>();
    }

    /// The hybrid's secret: ML-KEM's 32 bytes, then X25519's 32, nothing else (the bound covers
    /// its wiping when dropped)
    #[kani::proof]
    #[kani::unwind(65)]
    fn the_hybrid_secret_is_ml_kem_then_x25519() {
        let mlkem: [u8; 32] = kani::any();
        let x25519: [u8; 32] = kani::any();
        let secret = hybrid_secret(&mlkem, &x25519);
        let bytes = secret.secret_bytes();
        let i = kani::any_where(|&i: &usize| i < 32);
        assert!(bytes.len() == 64 && bytes[i] == mlkem[i] && bytes[32 + i] == x25519[i]);
    }

    /// A record's payload of any length: refused if shorter than a tag, never an underflow; and
    /// a record this side seals, of any content up to TLS's largest (2^14 bytes and up to 256 of
    /// padding, RFC 8446 section 5.2), opens to its content and type
    #[kani::proof]
    fn record_lengths_never_underflow_or_overflow() {
        let payload_len: usize = kani::any();
        match sealed_text_len(payload_len) {
            Some(text_len) => {
                assert!(text_len + TAG_LEN == payload_len);
            }
            None => {
                assert!(payload_len < TAG_LEN);
            }
        }
        let text_len = kani::any_where(|&len: &usize| len <= (1 << 14) + 256);
        assert!(sealed_text_len(sealed_payload_len(text_len)) == Some(text_len + 1));
    }

    /// Exactly the 32-byte keys and 64-byte signatures go on to Ed25519, as they are
    #[kani::proof]
    #[kani::unwind(73)]
    fn ed25519_keys_and_signatures_have_their_sizes() {
        let key_bytes: [u8; 40] = kani::any();
        let signature_bytes: [u8; 72] = kani::any();
        let key = &key_bytes[..any_len::<40>()];
        let signature = &signature_bytes[..any_len::<72>()];
        match ed25519_parts(key, signature) {
            Ok((key_part, signature_part)) => {
                assert!(key.len() == 32 && signature.len() == 64);
                let i = kani::any_where(|&i: &usize| i < 32);
                let j = kani::any_where(|&j: &usize| j < 64);
                assert!(key_part[i] == key[i] && signature_part[j] == signature[j]);
            }
            Err(_) => {
                assert!(key.len() != 32 || signature.len() != 64);
            }
        }
    }
}
