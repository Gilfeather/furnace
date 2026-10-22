# Wycheproof test vectors

Test vectors from [Project Wycheproof](https://github.com/C2SP/wycheproof), copied unchanged
from `testvectors_v1/` at commit
[`7a325932231bd7a724c791c8b0d65ea4d6d2f435`](https://github.com/C2SP/wycheproof/tree/7a325932231bd7a724c791c8b0d65ea4d6d2f435/testvectors_v1).
They are licensed under the Apache License 2.0 (`LICENSE` here, from the same commit).

`src/tls/provider/tests.rs` runs them against furnace's TLS crypto provider: the vectors mark
which inputs must be accepted and which (forged signatures, points off the curve, altered tags,
low-order keys and so on) must be refused. Only the algorithms the provider offers are here:

| File | Provider part |
|---|---|
| `aes_gcm_test.json` | AES-128-GCM and AES-256-GCM record protection (96-bit nonces, 128-bit tags) |
| `chacha20_poly1305_test.json` | ChaCha20-Poly1305 record protection |
| `x25519_test.json` | X25519 key exchange |
| `ecdh_secp256r1_ecpoint_test.json`, `ecdh_secp384r1_ecpoint_test.json` | P-256 and P-384 key exchange (raw points, as TLS sends them) |
| `ecdsa_secp256r1_sha256_test.json`, `ecdsa_secp384r1_sha256_test.json`, `ecdsa_secp384r1_sha384_test.json` | ECDSA signature verification (DER signatures) |
| `ed25519_test.json` | Ed25519 signature verification |
| `hmac_sha256_test.json`, `hmac_sha384_test.json` | HMAC |
| `hkdf_sha256_test.json`, `hkdf_sha384_test.json` | HKDF, as the cipher suites use it |

To update them, download the same files from a newer commit and change the commit above.
