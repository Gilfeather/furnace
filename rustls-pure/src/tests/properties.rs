//! Random and broken input, in place of fuzzing: key shares of every group, ML-KEM ciphertexts,
//! signatures and records. Each must be refused (or, for ML-KEM ciphertexts, give an unrelated
//! secret), never accepted and never a panic. proptest is pure Rust; cargo-fuzz and libFuzzer
//! would bring C++ in.

use proptest::collection::vec;
use proptest::prelude::*;
use rustls::crypto::cipher::{InboundOpaqueMessage, MessageDecrypter};

use super::*;

/// The share sizes a group takes: the client's (what the server gets) and the server's
fn share_sizes(group: NamedGroup) -> (usize, usize) {
    match group {
        NamedGroup::X25519 => (32, 32),
        NamedGroup::secp256r1 => (65, 65),
        NamedGroup::secp384r1 => (97, 97),
        NamedGroup::X25519MLKEM768 => (
            MLKEM768_ENCAPSULATION_KEY_LEN + X25519_SHARE_LEN,
            MLKEM768_CIPHERTEXT_LEN + X25519_SHARE_LEN,
        ),
        other => panic!("no group {other:?}"),
    }
}

fn ecdsa_p256_signature(message: &[u8]) -> (Vec<u8>, Vec<u8>) {
    use p256::ecdsa::signature::Signer as _;
    let key = p256::ecdsa::SigningKey::from_slice(&[7; 32]).unwrap();
    let signature: p256::ecdsa::Signature = key.sign(message);
    let public = key.verifying_key().to_sec1_bytes().to_vec();
    (public, signature.to_der().to_bytes().to_vec())
}

fn ed25519_signature(message: &[u8]) -> (Vec<u8>, Vec<u8>) {
    use ed25519_dalek::Signer as _;
    let key = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
    let public = key.verifying_key().to_bytes().to_vec();
    (public, key.sign(message).to_bytes().to_vec())
}

fn decrypt<C: Cipher>(record: &[u8]) -> Result<(), Error> {
    let mut decrypter = Crypter::<C>::new(&[3; 32][..C::KEY_LEN], Iv::from([5; 12]));
    let mut record = record.to_vec();
    let message = InboundOpaqueMessage::new(
        ContentType::ApplicationData,
        ProtocolVersion::TLSv1_2,
        &mut record,
    );
    decrypter.decrypt(message, 1).map(|_| ())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn key_shares_of_the_wrong_size_are_refused(share in vec(any::<u8>(), 0..1300)) {
        for group in provider().kx_groups {
            let (client_size, server_size) = share_sizes(group.name());
            // The server's side, given a client's share
            let answered = group.start_and_complete(&share);
            if share.len() != client_size {
                prop_assert!(answered.is_err(), "{:?}: {} bytes", group.name(), share.len());
            }
            // The client's side, given a server's share
            let completed = group.start().unwrap().complete(&share);
            if share.len() != server_size {
                prop_assert!(completed.is_err(), "{:?}: {} bytes", group.name(), share.len());
            }
        }
    }

    #[test]
    fn random_points_are_refused(coordinates in vec(any::<u8>(), 96)) {
        // An uncompressed point with random coordinates is off the curve (all but surely)
        let p256 = [&[4], &coordinates[..64]].concat();
        prop_assert!(SECP256R1.start().unwrap().complete(&p256).is_err());
        prop_assert!(SECP256R1.start_and_complete(&p256).is_err());
        let p384 = [&[4], &coordinates[..]].concat();
        prop_assert!(SECP384R1.start().unwrap().complete(&p384).is_err());
        prop_assert!(SECP384R1.start_and_complete(&p384).is_err());
    }

    #[test]
    fn random_mlkem_keys_are_refused(key in vec(any::<u8>(), MLKEM768_ENCAPSULATION_KEY_LEN)) {
        // Random coefficients are almost never all below the modulus (FIPS 203's check)
        let share = [&key[..], x25519_key(1).public.as_bytes()].concat();
        prop_assert!(X25519MLKEM768.start_and_complete(&share).is_err());
    }

    #[test]
    fn altered_mlkem_ciphertexts_give_an_unrelated_secret(
        seed in vec(any::<u8>(), 64),
        m in any::<[u8; 32]>(),
        bit in 0..MLKEM768_CIPHERTEXT_LEN * 8,
    ) {
        let seed = ml_kem::Seed::try_from(&seed[..]).unwrap();
        let client = || X25519MlKem768Group::start_with(seed, x25519_key(1));
        let server = X25519MlKem768Group::respond(
            &client().share,
            &ml_kem::B32::from(m),
            x25519_key(2),
        )
        .unwrap();
        let mut altered = server.pub_key.clone();
        altered[bit / 8] ^= 1 << (bit % 8);
        // Implicit rejection: no error, but ML-KEM's half of the secret is another one, so the
        // handshake fails at the Finished messages; X25519's half is unchanged
        let secret = Box::new(client()).complete(&altered).unwrap();
        let expected = server.secret.secret_bytes();
        prop_assert_ne!(&secret.secret_bytes()[..32], &expected[..32]);
        prop_assert_eq!(&secret.secret_bytes()[32..], &expected[32..]);
    }

    #[test]
    fn random_signatures_are_refused(
        message in vec(any::<u8>(), 0..256),
        signature in vec(any::<u8>(), 0..120),
    ) {
        let (p256, _) = ecdsa_p256_signature(&message);
        let (ed25519, _) = ed25519_signature(&message);
        for algorithm in [&ECDSA_P256_SHA256, &ECDSA_P256_SHA384] {
            prop_assert!(algorithm.verify_signature(&p256, &message, &signature).is_err());
        }
        prop_assert!(ED25519.verify_signature(&ed25519, &message, &signature).is_err());
    }

    #[test]
    fn altered_signatures_and_messages_are_refused(
        message in vec(any::<u8>(), 1..256),
        bit in any::<usize>(),
    ) {
        for (algorithm, (key, signature)) in [
            (&ECDSA_P256_SHA256 as &dyn SignatureVerificationAlgorithm, ecdsa_p256_signature(&message)),
            (&ED25519, ed25519_signature(&message)),
        ] {
            prop_assert!(algorithm.verify_signature(&key, &message, &signature).is_ok());
            let mut altered = signature.clone();
            let bit = bit % (altered.len() * 8);
            altered[bit / 8] ^= 1 << (bit % 8);
            prop_assert!(algorithm.verify_signature(&key, &message, &altered).is_err());
            let mut other = message.clone();
            let bit = bit % (other.len() * 8);
            other[bit / 8] ^= 1 << (bit % 8);
            prop_assert!(algorithm.verify_signature(&key, &other, &signature).is_err());
        }
    }

    #[test]
    fn random_records_are_refused(record in vec(any::<u8>(), 0..2000)) {
        prop_assert!(matches!(decrypt::<aes_gcm::Aes128Gcm>(&record), Err(Error::DecryptError)));
        prop_assert!(matches!(decrypt::<aes_gcm::Aes256Gcm>(&record), Err(Error::DecryptError)));
        prop_assert!(matches!(
            decrypt::<chacha20poly1305::ChaCha20Poly1305>(&record),
            Err(Error::DecryptError)
        ));
    }
}
