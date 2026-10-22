# rustls-pure

A crypto provider for [rustls](https://github.com/rustls/rustls) built on
[RustCrypto](https://github.com/RustCrypto) primitives.

- **Pure Rust.** No C, no assembly, no `ring` or `aws-lc-rs`: it builds wherever Rust does,
  with no C toolchain, and `#![forbid(unsafe_code)]` holds in the provider itself.
- **Checked against Wycheproof.** The tests run Google's
  [Wycheproof](https://github.com/C2SP/wycheproof) test vectors for every algorithm offered,
  about 4,300 cases: valid inputs must be accepted, crafted bad ones (forged signatures, points
  off the curve, altered tags, low-order keys, malformed DER) refused. Further tests pin what the
  vectors cannot see: each record's nonce is unique, a record that is altered, replayed,
  reordered or cut short does not decrypt, randomness comes from the operating system, and
  every key is wiped when dropped.
- **Strict about bad input.** Where the standards leave a choice, it refuses: an X25519
  exchange that ends in an all-zero secret (RFC 8446 section 7.4.2), compressed or malformed
  key shares, malleable Ed25519 signatures and small-order Ed25519 keys (`verify_strict`).
- **TLS 1.3 only.** No TLS 1.2, no RSA, no legacy suites: less code to get wrong and to review.

It has **not had an outside security audit**, and neither have the RustCrypto crates under it.
The checks above are mechanical; they are not a substitute for one. If you need audited TLS,
use rustls with its default provider, or terminate TLS in a proxy.

| | Supported |
|---|---|
| Protocol | TLS 1.3 |
| Cipher suites | `TLS13_AES_128_GCM_SHA256`, `TLS13_AES_256_GCM_SHA384`, `TLS13_CHACHA20_POLY1305_SHA256` |
| Key exchange | X25519, P-256, P-384 |
| Keys and certificates | ECDSA P-256, ECDSA P-384, Ed25519 (PKCS#8, or SEC1 for ECDSA) |

RSA is not offered: the pure-Rust `rsa` crate has an unfixed timing side channel
([RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)). Every CA,
cert-manager and Let's Encrypt can issue ECDSA certificates.

## Usage

```toml
[dependencies]
rustls = { version = "0.23", default-features = false, features = ["std"] }
rustls-pure = "0.13"
```

Pass the provider to rustls's config builders, with TLS 1.3 as the protocol version. A server:

```rust
use std::sync::Arc;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

let certs = CertificateDer::pem_file_iter("cert.pem")?.collect::<Result<Vec<_>, _>>()?;
let key = PrivateKeyDer::from_pem_file("key.pem")?;
let config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls_pure::provider()))
    .with_protocol_versions(&[&rustls::version::TLS13])?
    .with_no_client_auth()
    .with_single_cert(certs, key)?;
```

A client:

```rust
use std::sync::Arc;

use rustls::pki_types::CertificateDer;
use rustls::pki_types::pem::PemObject;

let mut roots = rustls::RootCertStore::empty();
roots.add(CertificateDer::from_pem_file("ca.pem")?)?;
let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls_pure::provider()))
    .with_protocol_versions(&[&rustls::version::TLS13])?
    .with_root_certificates(roots)
    .with_no_client_auth();
```

Both work with tokio-rustls and anything else that takes a rustls config.

## Compared with rustls-rustcrypto

[rustls-rustcrypto](https://github.com/RustCrypto/rustls-rustcrypto), from the RustCrypto
organization, has the same aim and a wider reach. As of October 2026:

| | rustls-pure | rustls-rustcrypto |
|---|---|---|
| Release | 0.13 | 0.0.2-alpha (April 2024); its README says not to use it in production |
| Outside audit | none | none |
| Wycheproof tests | yes, every algorithm offered | none in the repository |
| TLS versions | 1.3 | 1.2 and 1.3 |
| Cipher suites | the three TLS 1.3 suites | the same, plus six TLS 1.2 ECDHE suites |
| Key exchange | X25519, P-256, P-384 | the same |
| Signatures | ECDSA P-256 / P-384, Ed25519 | the same, plus RSA PKCS#1 and PSS |
| X25519 all-zero secret | refused | not checked |
| Ed25519 verification | `verify_strict` | `verify` |
| Other | | QUIC, groundwork for `no_std` |

Pick rustls-rustcrypto if you need TLS 1.2, RSA or QUIC; rustls-pure if TLS 1.3 with ECDSA
or Ed25519 keys is enough and you want the narrower, stricter, test-vector-checked provider.

## Tests

`cargo test -p rustls-pure` from a checkout of the repository. The Wycheproof vectors are in
[`wycheproof/`](wycheproof/README.md) (Apache-2.0, copied unchanged from a pinned commit); they
are left out of the published crate.

MIT licensed. Forged in kamiforge: built as part of [Furnace](https://github.com/Gilfeather/furnace),
the pure-Rust inference server, which serves HTTPS and gRPC with it.
