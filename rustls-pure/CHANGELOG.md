# Changelog

rustls-pure has a version of its own, apart from Furnace's, and is released only when it
changes. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html): before 1.0,
a breaking change, a new minimum Rust version or a new rustls minor version is a new minor
version.

## [0.15.0] - Unreleased

### Added
- Post-quantum key exchange: X25519MLKEM768 (ML-KEM-768 from FIPS 203, with X25519), as
  draft-ietf-tls-ecdhe-mlkem and rustls's own providers have it, on RustCrypto's `ml-kem`. It
  is offered first, as in rustls's default provider; a client offers its X25519 part as a share
  of its own too, so a server without ML-KEM takes plain X25519 without a HelloRetryRequest.
  Checked against Wycheproof's ML-KEM-768 vectors (key generation, encapsulation, decapsulation),
  with tests of how the hybrid puts its shares and secret together, and against OpenSSL 3.5.
- Full handshakes in memory between rustls-pure clients and servers (`tests/handshake.rs`):
  every cipher suite, key exchange group and key type, with client certificates,
  HelloRetryRequest, resumption, key updates and refused certificates.
- Random and broken input with proptest, in place of fuzzing (`src/tests/properties.rs`):
  key shares of every group, ML-KEM keys and ciphertexts, signatures and records.
- The interoperability tests also run on pull requests that change rustls-pure; on Linux they
  run on Debian 13 for its OpenSSL 3.5, and X25519MLKEM768 must be checked, not skipped.

### Changed
- X25519MLKEM768 is the first key exchange group, before X25519, P-256 and P-384. Peers without
  it keep working (see above); its key shares are larger (1216 bytes from the client, 1120 from
  the server).

## [0.14.1] - 2026-10-22

### Changed
- The minimum supported Rust version is 1.89 (it was 1.95, Furnace's), the lowest its
  dependencies allow; a weekly CI job tests it, and the newest rustls 0.23 release.
- The README says plainly that RSA signatures are not verified, so rustls-pure does not suit a
  client for arbitrary public websites, and which rustls and Rust versions are supported.

### Added
- Interoperability tests against OpenSSL, GnuTLS and curl (`interop/run.sh`), weekly on Linux
  and macOS.
- A security policy (SECURITY.md), including how advisories reach RustSec.

## [0.14.0] - 2026-10-22

Released with Furnace 0.14.0; no change to the provider.

## [0.13.0] - 2026-10-22

The provider behind Furnace's `tls` feature, as a crate of its own: TLS 1.3 with AES-GCM and
ChaCha20-Poly1305, X25519, P-256 and P-384 key exchange, and ECDSA P-256 / P-384 and Ed25519
keys and certificates, on RustCrypto primitives. Its tests run Google's Wycheproof vectors for
every algorithm offered and check nonce construction, record integrity, the source of
randomness and the wiping of keys.
