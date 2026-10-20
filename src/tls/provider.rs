//! A pure-Rust crypto provider for rustls, built on RustCrypto primitives.
//!
//! rustls does the TLS protocol itself; a provider only supplies the primitives. The ones here
//! are deliberately few, which keeps the code small enough to review:
//!
//! - TLS 1.3 only, with AES-128-GCM, AES-256-GCM and ChaCha20-Poly1305
//! - key exchange with X25519, P-256 and P-384
//! - server keys and client certificates (for mutual TLS) using ECDSA P-256 / P-384 or Ed25519
//!
//! RSA is not offered: the pure-Rust `rsa` crate has an unfixed timing side channel
//! (RUSTSEC-2023-0071). Certificate authorities, cert-manager and Let's Encrypt all issue ECDSA
//! certificates.

use std::fmt;
use std::sync::Arc;

use aes_gcm::aead::{AeadCore, AeadInOut, KeyInit, KeySizeUser};
use ed25519_dalek::pkcs8::DecodePrivateKey as _;
use ed25519_dalek::pkcs8::{ObjectIdentifier, PrivateKeyInfoRef};
// zeroize through the curve crates' re-export, which is the version they use themselves
use p256::ecdsa::signature::hazmat::PrehashVerifier;
use p256::elliptic_curve::zeroize::Zeroizing;
use rustls::crypto::cipher::{
    AeadKey, InboundOpaqueMessage, InboundPlainMessage, Iv, MessageDecrypter, MessageEncrypter,
    Nonce, OutboundOpaqueMessage, OutboundPlainMessage, PrefixedPayload, Tls13AeadAlgorithm,
    UnsupportedOperationError, make_tls13_aad,
};
use rustls::crypto::{
    ActiveKeyExchange, CryptoProvider, GetRandomFailed, KeyProvider, SecureRandom, SharedSecret,
    SupportedKxGroup, WebPkiSupportedAlgorithms, hash, hmac as rustls_hmac, tls13,
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

/// The provider: pass it to `rustls::ServerConfig::builder_with_provider`
pub(super) fn provider() -> CryptoProvider {
    CryptoProvider {
        cipher_suites: vec![
            TLS13_AES_128_GCM_SHA256,
            TLS13_AES_256_GCM_SHA384,
            TLS13_CHACHA20_POLY1305_SHA256,
        ],
        kx_groups: vec![&X25519, &SECP256R1, &SECP384R1],
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
pub(super) static TLS13_AES_128_GCM_SHA256: SupportedCipherSuite =
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
pub(super) static TLS13_AES_256_GCM_SHA384: SupportedCipherSuite =
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
pub(super) static TLS13_CHACHA20_POLY1305_SHA256: SupportedCipherSuite =
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

/// What the AEAD must provide; the key, nonce and tag sizes are checked below, at compile time
trait Cipher: AeadInOut + KeyInit + Send + Sync + 'static {
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
        Box::new(Crypter {
            cipher: C::new_from_slice(key.as_ref()).expect("key of the suite's length"),
            iv,
        })
    }

    fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageDecrypter> {
        Box::new(Crypter {
            cipher: C::new_from_slice(key.as_ref()).expect("key of the suite's length"),
            iv,
        })
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
    fn nonce(&self, seq: u64) -> aes_gcm::aead::Nonce<C> {
        let nonce = Nonce::new(&self.iv, seq).0;
        aes_gcm::aead::Nonce::<C>::try_from(&nonce[..]).expect("12-byte nonces")
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
        let tag = self
            .cipher
            .encrypt_inout_detached(
                &self.nonce(seq),
                &make_tls13_aad(total_len),
                payload.as_mut().into(),
            )
            .map_err(|_| Error::EncryptError)?;
        debug_assert_eq!(tag.len(), TAG_LEN);
        payload.extend_from_slice(&tag);
        Ok(OutboundOpaqueMessage::new(
            ContentType::ApplicationData,
            // TLS 1.3 records carry the legacy version 1.2 (RFC 8446 section 5.1)
            ProtocolVersion::TLSv1_2,
            payload,
        ))
    }

    fn encrypted_payload_len(&self, payload_len: usize) -> usize {
        payload_len + 1 + TAG_LEN
    }
}

impl<C: Cipher> MessageDecrypter for Crypter<C> {
    fn decrypt<'a>(
        &mut self,
        mut msg: InboundOpaqueMessage<'a>,
        seq: u64,
    ) -> Result<InboundPlainMessage<'a>, Error> {
        let payload = &mut msg.payload;
        let Some(plain_len) = payload.len().checked_sub(TAG_LEN) else {
            return Err(Error::DecryptError);
        };
        let nonce = self.nonce(seq);
        let aad = make_tls13_aad(payload.len());
        let (text, tag) = payload.split_at_mut(plain_len);
        let tag = aes_gcm::aead::Tag::<C>::try_from(&*tag).map_err(|_| Error::DecryptError)?;
        self.cipher
            .decrypt_inout_detached(&nonce, &aad, text.into(), &tag)
            .map_err(|_| Error::DecryptError)?;
        payload.truncate(plain_len);
        msg.into_tls13_unpadded_message()
    }
}

// The sizes the code above relies on, for every suite: the nonce and tag, and the key that
// RFC 8446 gives the suite (rustls derives keys of `KEY_LEN`, so the `expect`s on
// `new_from_slice` cannot fail)
const _: () = {
    use aes_gcm::aead::array::typenum::Unsigned;
    macro_rules! check {
        ($($cipher:ty => $key:expr),*) => {$(
            assert!(<$cipher as Cipher>::KEY_LEN == $key);
            assert!(<<$cipher as AeadCore>::NonceSize as Unsigned>::USIZE == 12);
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

#[derive(Debug)]
struct X25519Group;
static X25519: X25519Group = X25519Group;

impl SupportedKxGroup for X25519Group {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        let secret = x25519_dalek::StaticSecret::from(*random::<32>()?);
        let public = x25519_dalek::PublicKey::from(&secret);
        Ok(Box::new(X25519Exchange { secret, public }))
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::X25519
    }
}

struct X25519Exchange {
    secret: x25519_dalek::StaticSecret,
    public: x25519_dalek::PublicKey,
}

impl ActiveKeyExchange for X25519Exchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
        let peer: [u8; 32] = peer_pub_key
            .try_into()
            .map_err(|_| Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
        let shared = self
            .secret
            .diffie_hellman(&x25519_dalek::PublicKey::from(peer));
        // An all-zero result means a low-order peer key (RFC 8446 section 7.4.2)
        if !shared.was_contributory() {
            return Err(rustls::PeerMisbehaved::InvalidKeyShare.into());
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
        // TLS 1.3 key shares are uncompressed points (RFC 8446 section 4.2.8.2)
        if peer_pub_key.first() != Some(&4) {
            return Err(rustls::PeerMisbehaved::InvalidKeyShare.into());
        }
        let peer = p256::PublicKey::from_sec1_bytes(peer_pub_key)
            .map_err(|_| Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
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
        if peer_pub_key.first() != Some(&4) {
            return Err(rustls::PeerMisbehaved::InvalidKeyShare.into());
        }
        let peer = p384::PublicKey::from_sec1_bytes(peer_pub_key)
            .map_err(|_| Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
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
                "{what}; furnace's pure-Rust TLS takes ECDSA P-256 / P-384 or Ed25519 keys"
            ))
        };
        let key: Arc<dyn SigningKey> = match &key_der {
            PrivateKeyDer::Pkcs8(der) => {
                let der = der.secret_pkcs8_der();
                if let Ok(key) = p256::ecdsa::SigningKey::from_pkcs8_der(der) {
                    Arc::new(EcdsaKey::P256(key))
                } else if let Ok(key) = p384::ecdsa::SigningKey::from_pkcs8_der(der) {
                    Arc::new(EcdsaKey::P384(key))
                } else if let Ok(key) = ed25519_dalek::SigningKey::from_pkcs8_der(der) {
                    Arc::new(Ed25519Key(key))
                } else {
                    return Err(unsupported(&unsupported_pkcs8(der)));
                }
            }
            PrivateKeyDer::Sec1(der) => {
                let der = der.secret_sec1_der();
                if let Ok(key) = p256::SecretKey::from_sec1_der(der) {
                    Arc::new(EcdsaKey::P256(key.into()))
                } else if let Ok(key) = p384::SecretKey::from_sec1_der(der) {
                    Arc::new(EcdsaKey::P384(key.into()))
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

enum EcdsaKey {
    P256(p256::ecdsa::SigningKey),
    P384(p384::ecdsa::SigningKey),
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
        offered.contains(&self.scheme()).then(|| {
            Box::new(EcdsaSigner(match self {
                Self::P256(key) => Self::P256(key.clone()),
                Self::P384(key) => Self::P384(key.clone()),
            })) as Box<dyn Signer>
        })
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

struct Ed25519Key(ed25519_dalek::SigningKey);

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
        let digest = match self.hash {
            Prehash::Sha256 => sha2::Sha256::digest(message).to_vec(),
            Prehash::Sha384 => sha2::Sha384::digest(message).to_vec(),
        };
        match self.curve {
            Curve::P256 => {
                let key = p256::ecdsa::VerifyingKey::from_sec1_bytes(public_key)
                    .map_err(|_| InvalidSignature)?;
                let signature =
                    p256::ecdsa::Signature::from_der(signature).map_err(|_| InvalidSignature)?;
                key.verify_prehash(&digest, &signature)
                    .map_err(|_| InvalidSignature)
            }
            Curve::P384 => {
                let key = p384::ecdsa::VerifyingKey::from_sec1_bytes(public_key)
                    .map_err(|_| InvalidSignature)?;
                let signature =
                    p384::ecdsa::Signature::from_der(signature).map_err(|_| InvalidSignature)?;
                key.verify_prehash(&digest, &signature)
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
        let key: [u8; 32] = public_key.try_into().map_err(|_| InvalidSignature)?;
        let key = ed25519_dalek::VerifyingKey::from_bytes(&key).map_err(|_| InvalidSignature)?;
        let signature =
            ed25519_dalek::Signature::from_slice(signature).map_err(|_| InvalidSignature)?;
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
