# Changelog

rustls-pure has a version of its own, apart from Furnace's, and is released only when it
changes. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html): before 1.0,
a breaking change, a new minimum Rust version or a new rustls minor version is a new minor
version.

## [0.14.1] - Unreleased

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
