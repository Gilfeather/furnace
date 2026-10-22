//! Full TLS 1.3 handshakes between rustls-pure clients and servers, in memory: every cipher
//! suite, key exchange group and key type, with client certificates, HelloRetryRequest,
//! resumption and key updates.
//!
//! The peer is rustls-pure itself. rustls's own providers (ring, aws-lc-rs) would be the
//! obvious other side, but they bring C and assembly in, even as dev-dependencies; other
//! implementations are checked by `interop/run.sh` instead (OpenSSL, GnuTLS, curl).

use std::io::{Read, Write};
use std::sync::Arc;

use rustls::client::Resumption;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{
    CipherSuite, ClientConfig, ClientConnection, Error, HandshakeKind, NamedGroup, RootCertStore,
    ServerConfig, ServerConnection,
};

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata");
const KEY_TYPES: [&str; 3] = ["p256", "p384", "ed25519"];
const SUITES: [CipherSuite; 3] = [
    CipherSuite::TLS13_AES_128_GCM_SHA256,
    CipherSuite::TLS13_AES_256_GCM_SHA384,
    CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
];
const GROUPS: [NamedGroup; 4] = [
    NamedGroup::X25519MLKEM768,
    NamedGroup::X25519,
    NamedGroup::secp256r1,
    NamedGroup::secp384r1,
];

fn certificate(name: &str) -> Vec<CertificateDer<'static>> {
    CertificateDer::pem_file_iter(format!("{TESTDATA}/{name}.crt"))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn private_key(name: &str) -> PrivateKeyDer<'static> {
    PrivateKeyDer::from_pem_file(format!("{TESTDATA}/{name}.key")).unwrap()
}

fn roots(key_type: &str) -> Arc<RootCertStore> {
    let mut roots = RootCertStore::empty();
    roots
        .add(certificate(&format!("{key_type}-ca")).remove(0))
        .unwrap();
    Arc::new(roots)
}

/// The provider with only `suites` and `groups`, in that order (all when empty)
fn provider(suites: &[CipherSuite], groups: &[NamedGroup]) -> Arc<CryptoProvider> {
    let mut provider = rustls_pure::provider();
    if !suites.is_empty() {
        provider
            .cipher_suites
            .retain(|suite| suites.contains(&suite.suite()));
    }
    if !groups.is_empty() {
        provider.kx_groups = groups
            .iter()
            .map(|name| {
                *provider
                    .kx_groups
                    .iter()
                    .find(|group| group.name() == *name)
                    .unwrap()
            })
            .collect();
    }
    Arc::new(provider)
}

/// A server with the `key_type` certificate, requiring client certificates from that CA
fn server(key_type: &str, provider: Arc<CryptoProvider>) -> Arc<ServerConfig> {
    let verifier = WebPkiClientVerifier::builder_with_provider(roots(key_type), provider.clone())
        .build()
        .unwrap();
    let config = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .unwrap()
        .with_client_cert_verifier(verifier)
        .with_single_cert(
            certificate(&format!("{key_type}-server")),
            private_key(&format!("{key_type}-server")),
        )
        .unwrap();
    Arc::new(config)
}

/// A client trusting the `key_type` CA, with its client certificate
fn client(key_type: &str, provider: Arc<CryptoProvider>) -> Arc<ClientConfig> {
    let config = ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .unwrap()
        .with_root_certificates(roots(key_type))
        .with_client_auth_cert(
            certificate(&format!("{key_type}-client")),
            private_key(&format!("{key_type}-client")),
        )
        .unwrap();
    Arc::new(config)
}

/// A connected pair, after their handshake and whatever followed it (session tickets)
struct Pair {
    client: ClientConnection,
    server: ServerConnection,
}

impl Pair {
    fn connect(client: &Arc<ClientConfig>, server: &Arc<ServerConfig>) -> Result<Self, Error> {
        let name = ServerName::try_from("localhost").unwrap();
        let mut pair = Self {
            client: ClientConnection::new(client.clone(), name).unwrap(),
            server: ServerConnection::new(server.clone()).unwrap(),
        };
        pair.transfer()?;
        assert!(!pair.client.is_handshaking() && !pair.server.is_handshaking());
        Ok(pair)
    }

    /// Moves records both ways until neither side has anything left to send
    fn transfer(&mut self) -> Result<(), Error> {
        loop {
            let mut records = Vec::new();
            while self.client.wants_write() {
                self.client.write_tls(&mut records).unwrap();
            }
            let to_server = !records.is_empty();
            let mut incoming = &records[..];
            while !incoming.is_empty() {
                self.server.read_tls(&mut incoming).unwrap();
                self.server.process_new_packets()?;
            }

            let mut records = Vec::new();
            while self.server.wants_write() {
                self.server.write_tls(&mut records).unwrap();
            }
            let to_client = !records.is_empty();
            let mut incoming = &records[..];
            while !incoming.is_empty() {
                self.client.read_tls(&mut incoming).unwrap();
                self.client.process_new_packets()?;
            }
            if !to_server && !to_client {
                return Ok(());
            }
        }
    }

    /// Application data reaches the other side, both ways
    fn exchange(&mut self, message: &[u8]) {
        self.client.writer().write_all(message).unwrap();
        self.transfer().unwrap();
        let mut received = vec![0; message.len()];
        self.server.reader().read_exact(&mut received).unwrap();
        assert_eq!(received, message);

        self.server.writer().write_all(message).unwrap();
        self.transfer().unwrap();
        let mut received = vec![0; message.len()];
        self.client.reader().read_exact(&mut received).unwrap();
        assert_eq!(received, message);
    }

    fn agreed(&self) -> (CipherSuite, NamedGroup, HandshakeKind) {
        let suite = self.client.negotiated_cipher_suite().unwrap().suite();
        let group = self.client.negotiated_key_exchange_group().unwrap().name();
        assert_eq!(
            self.server.negotiated_cipher_suite().unwrap().suite(),
            suite
        );
        assert_eq!(
            self.server.negotiated_key_exchange_group().unwrap().name(),
            group
        );
        (suite, group, self.client.handshake_kind().unwrap())
    }
}

#[test]
fn every_suite_group_and_key_type() {
    for key_type in KEY_TYPES {
        let server = server(key_type, provider(&[], &[]));
        for suite in SUITES {
            for group in GROUPS {
                let client = client(key_type, provider(&[suite], &[group]));
                let what = format!("{key_type}, {suite:?}, {group:?}");
                let mut pair =
                    Pair::connect(&client, &server).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(pair.agreed(), (suite, group, HandshakeKind::Full), "{what}");
                // Each side saw the other's certificate
                let server_chain = pair.client.peer_certificates().unwrap();
                assert!(
                    server_chain[0] == certificate(&format!("{key_type}-server"))[0],
                    "{what}"
                );
                let client_chain = pair.server.peer_certificates().unwrap();
                assert!(
                    client_chain[0] == certificate(&format!("{key_type}-client"))[0],
                    "{what}"
                );
                pair.exchange(b"GET / HTTP/1.1\r\n\r\n");
            }
        }
    }
}

#[test]
fn the_post_quantum_hybrid_is_agreed_by_default() {
    let mut pair = Pair::connect(
        &client("p256", provider(&[], &[])),
        &server("p256", provider(&[], &[])),
    )
    .unwrap();
    assert_eq!(pair.agreed().1, NamedGroup::X25519MLKEM768);
    assert_eq!(pair.agreed().2, HandshakeKind::Full);
    pair.exchange(b"hello");
}

#[test]
fn a_server_without_mlkem_takes_the_hybrids_x25519_part_without_a_retry() {
    // The client offers X25519MLKEM768 first and its X25519 part alongside
    let mut pair = Pair::connect(
        &client("p256", provider(&[], &[])),
        &server("p256", provider(&[], &[NamedGroup::X25519])),
    )
    .unwrap();
    assert_eq!(pair.agreed().1, NamedGroup::X25519);
    assert_eq!(pair.agreed().2, HandshakeKind::Full);
    pair.exchange(b"hello");
}

#[test]
fn hello_retry_request() {
    // The client's first key share is for a group the server does not take, so the server
    // asks again for one it does
    for (client_groups, server_group) in [
        (
            [NamedGroup::X25519, NamedGroup::secp384r1],
            NamedGroup::secp384r1,
        ),
        (
            [NamedGroup::X25519MLKEM768, NamedGroup::secp256r1],
            NamedGroup::secp256r1,
        ),
        (
            [NamedGroup::secp256r1, NamedGroup::X25519MLKEM768],
            NamedGroup::X25519MLKEM768,
        ),
    ] {
        let mut pair = Pair::connect(
            &client("p384", provider(&[], &client_groups)),
            &server("p384", provider(&[], &[server_group])),
        )
        .unwrap();
        let (_, group, kind) = pair.agreed();
        assert_eq!(group, server_group);
        assert_eq!(
            kind,
            HandshakeKind::FullWithHelloRetryRequest,
            "{client_groups:?}"
        );
        pair.exchange(b"hello");
    }
}

#[test]
fn sessions_are_resumed() {
    for suite in SUITES {
        for group in [NamedGroup::X25519MLKEM768, NamedGroup::secp256r1] {
            let client = client("ed25519", provider(&[suite], &[group]));
            let server = server("ed25519", provider(&[], &[]));
            let mut first = Pair::connect(&client, &server).unwrap();
            assert_eq!(first.agreed().2, HandshakeKind::Full);
            first.exchange(b"first");
            // The tickets the server sent after the handshake let the client come back
            let mut second = Pair::connect(&client, &server).unwrap();
            assert_eq!(
                second.agreed(),
                (suite, group, HandshakeKind::Resumed),
                "{suite:?}"
            );
            second.exchange(b"second");
        }
    }
}

#[test]
fn a_client_without_resumption_gets_a_full_handshake() {
    let mut config = (*client("p256", provider(&[], &[]))).clone();
    config.resumption = Resumption::disabled();
    let (client, server) = (Arc::new(config), server("p256", provider(&[], &[])));
    Pair::connect(&client, &server).unwrap();
    let pair = Pair::connect(&client, &server).unwrap();
    assert_eq!(pair.agreed().2, HandshakeKind::Full);
}

#[test]
fn keys_are_updated() {
    for suite in SUITES {
        let mut pair = Pair::connect(
            &client("p256", provider(&[suite], &[])),
            &server("p256", provider(&[], &[])),
        )
        .unwrap();
        for round in 0..3 {
            pair.client.refresh_traffic_keys().unwrap();
            pair.exchange(format!("after the client's update {round}").as_bytes());
            pair.server.refresh_traffic_keys().unwrap();
            pair.exchange(format!("after the server's update {round}").as_bytes());
        }
    }
}

#[test]
fn certificates_from_another_ca_are_refused() {
    // A server whose certificate the client's CA did not sign
    let result = Pair::connect(
        &client("p384", provider(&[], &[])),
        &server("p256", provider(&[], &[])),
    );
    assert!(result.is_err());
    // A client certificate the server's CA did not sign
    let stranger = ClientConfig::builder_with_provider(provider(&[], &[]))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .unwrap()
        .with_root_certificates(roots("p384"))
        .with_client_auth_cert(certificate("ed25519-client"), private_key("ed25519-client"))
        .unwrap();
    let result = Pair::connect(&Arc::new(stranger), &server("p384", provider(&[], &[])));
    assert!(result.is_err());
}
