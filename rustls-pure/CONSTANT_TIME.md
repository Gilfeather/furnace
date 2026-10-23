# Constant time in rustls-pure

What has been checked about timing side channels, how, and what that does not prove. Last
reviewed on 2026-10-23, against rustls-pure 0.15.2 and the versions in the repository's
`Cargo.lock`.

In short: rustls-pure's own code compares no secret with `==` and branches on no secret, and
its machine code for the two places it touches secrets has no branch on x86_64 or aarch64; the
primitives under it are written to run in constant time and say so; statistical timing tests
of every secret-dependent operation find no difference on x86_64, aarch64 Linux and Apple
silicon. None of this is a proof, and there has been no outside audit.

## 1. rustls-pure's own code

Read line by line for branches, indexing and early returns that depend on a secret:

- **Key shares.** Their lengths and the first byte of a P-256 / P-384 point are checked before
  any cryptography (the Kani proofs in `src/lib.rs` cover these). They are the peer's public
  bytes, so branching on them leaks nothing.
- **X25519MLKEM768.** The shares are split at fixed offsets and the secret is ML-KEM's 32 bytes,
  then X25519's, copied without looking at them (`put_together`, whose machine code section 3
  checks). ML-KEM's implicit rejection is the `ml-kem` crate's (section 2): rustls-pure takes
  whatever secret decapsulation gives and never asks whether the ciphertext was valid. A
  tampered ciphertext fails the handshake later, at the Finished messages.
- **Comparisons of secrets.** rustls-pure makes one: X25519's all-zero check (`all_zero`). It
  ORs every byte together with no early exit and compares only the result with zero, so only
  its yes-or-no answer, which is public (the handshake fails), is branched on; section 3 checks
  the machine code. Before this review it called x25519-dalek's `was_contributory`, which is
  also constant time (`subtle`'s `ct_eq` against the identity) but compiled in another crate,
  where rustls-pure's machine-code check cannot see it; x25519-dalek's results are always fully
  reduced, so the two agree.
  Tags and Finished messages are compared by the AEAD crates and by rustls itself, in constant
  time (`ctutils` and `subtle`); rustls-pure hands them over and does not compare them.
- **Records.** Only lengths, which are public, are checked before the AEAD opens a record.
  Removing the padding after it opens is rustls's code, which scans for the content type and
  so takes time that depends on the length of the padding (RFC 8446 section 5.4 allows this).
- **Keys.** Random ECDH scalars that fall outside the curve's order are drawn again; the
  branch reveals only that a discarded value was too large. Private keys are parsed from DER at
  load time, once, not in a handshake.
- **Zeroization.** Secrets live in types that wipe themselves (`Zeroizing`, the curve crates'
  keys, ML-KEM's decapsulation key, rustls's `SharedSecret`); `src/tests.rs` pins this. Found
  and fixed in this review: ML-KEM's encapsulation randomness `m`, from which the server's
  ML-KEM secret follows, was copied out of its wiped buffer into a temporary that was not
  wiped. It is now made in place and only lent out. What is left: a secret passed by value
  (the X25519 scalar and the ML-KEM seed, which their crates take by value) can leave a copy on
  the stack, which Rust gives no way to wipe.

## 2. The primitives

From each crate's sources at the locked version, its documentation and the RustSec database.
This records what they claim and how; it does not re-verify them.

| Crate | Locked | How secrets are handled | Timing advisories |
|---|---|---|---|
| curve25519-dalek, x25519-dalek, ed25519-dalek | 5.0.0, 3.0.0, 3.0.0 | Constant time by design, documented, through `subtle`, except functions named `vartime` (used only to verify signatures, on public data). X25519 is a Montgomery ladder with a conditional swap on every bit. Backend: serial 64-bit, or AVX2 picked at run time. | RUSTSEC-2024-0344 (a branch the compiler put into `Scalar52::sub`), fixed in 4.1.3, so not in 5.0 |
| p256, p384, ecdsa, primefield, primeorder | 0.14.0, 0.14.0, 0.17.0, 0.14.0, 0.14.0 | `subtle` / `ctutils` types throughout. P-256's field is hand-written Montgomery arithmetic; P-384's is `crypto-bigint` (its default backend is no longer fiat-crypto, whatever its comments say). Inversion is the constant-time Bernstein–Yang algorithm. Signing (RFC 6979 nonces) multiplies with a table that is scanned whole and complete formulas; `vartime` code appears only in verification. | none |
| crypto-bigint | 0.7.5 | Constant time by design, checked in its CI with `cargo-checkct` (BINSEC). It builds its masks from a `Choice` the way `ctutils` did before the fix below; whether that compiles to a branch has not been checked here. | none |
| ml-kem, module-lattice | 0.3.3, 0.2.3 | Decapsulation compares the re-encrypted ciphertext with `ct_eq` and picks the real or the rejection secret with `ct_select`, without a branch. Compression multiplies and shifts instead of dividing. | none filed; KyberSlash (division by a secret) fixed in 0.1.1, secret-indexed table lookups in sampling removed in 0.3.3 |
| aes, aes-gcm, ghash / polyval | 0.9.3, 0.11.1, 0.6.0 / 0.7.3 | AES-NI and CLMUL (x86) or ARMv8 AES and PMULL, picked at run time; otherwise bitsliced AES with no tables, and a GHASH multiply that is constant time where the CPU's multiplier is. The tag is checked with `ct_eq` before anything is decrypted. | none for timing |
| chacha20poly1305, chacha20, poly1305 | 0.11.0, 0.10.2, 0.9.1 | ChaCha20 is add-rotate-xor, constant time by construction; Poly1305 is constant time on a constant-time multiplier; the tag is checked with `ct_eq` first. Poly1305's AVX2 backend has not been audited, its README says. | none for timing |
| sha2, hmac | 0.11.1, 0.13.0 | No table lookups or branches on the data. | none for timing |
| ctutils, cmov | 0.4.4, 0.5.4 | The constant-time comparisons and selections under most of the above: `cmov` in Rust inline assembly on x86 and ARM. | **RUSTSEC-2026-0385** (ctutils below 0.4.4: masks that could compile to a branch, and branching `u128` comparisons on 32-bit targets). The lockfile had 0.4.3 until the day it was published; it now has 0.4.4. RUSTSEC-2026-0003 (cmov), fixed in the locked version |

Of these crates only `ctutils`, `cmov` and `crypto-bigint` check constant time in their own CI
(with `cargo-checkct`); none runs timing tests. The NCC Group audit of 2020 that the AEAD and
AES crates cite covered older versions.

rustls-pure's version requirements allow every version above, so an application with an older
lockfile can still resolve `ctutils` 0.4.3: `cargo update -p ctutils` fixes that, and
`cargo audit` reports it.

## 3. rustls-pure's machine code

`ct-asm.sh` builds rustls-pure in release, as assembly, for the machine it runs on, and checks
the two functions of its own that touch secrets: `all_zero` (X25519's all-zero check) and
`put_together` (the X25519MLKEM768 secret). Neither may contain a branch or jump, a call
(whose code the check would not see) or a load whose address is computed from a second
register, which a table lookup by a secret would need. Both are straight-line code: 32 loads,
ORs and one compare with `sete` / `cset` for `all_zero`, and four 16-byte loads and stores for
`put_together`, on x86_64 and on aarch64 (Linux and macOS). The check was seen to fail on a
version of `all_zero` that stops at the first non-zero byte. It runs in `rustls-pure.yml` on
every pull request that changes rustls-pure, on x86_64 Linux, aarch64 Linux and macOS, and in
the timing workflow on the same runners.

This needs no measurement, but it covers only rustls-pure's own code. The RustCrypto crates'
code is covered by their design (section 2) and by the timing tests (section 4).

## 4. Timing tests

`src/tests/constant_time.rs`, in the style of dudect (Reparaz, Balasch and Verbauwhede, 2017).
Each target is timed on two classes of input, the class chosen at random for every sample;
Welch's t-test compares the two classes on all samples and on those below ten percentiles (a
difference can hide under the tail that interrupts add), and the largest |t| is the result.
|t| below 4.5 is "no difference detected", as dudect has it. About a minute per target, up to
two million samples; a sample times as many operations as make it last 200 ticks of the clock.

| Target | Class A | Class B |
|---|---|---|
| **Control**: ML-KEM-768 decapsulation | valid ciphertexts | valid ciphertexts |
| X25519 key exchange (rustls-pure's) | fixed key | random keys |
| X25519 all-zero check (rustls-pure's) | 512 peers whose result starts with a zero byte | 512 whose result does not |
| P-256, P-384 key exchange (rustls-pure's) | fixed key | random keys |
| ML-KEM-768 decapsulation | valid ciphertexts | ciphertexts with one bit flipped |
| X25519MLKEM768, the client's side (rustls-pure's) | valid ciphertexts | ciphertexts with one bit flipped |
| ML-KEM-768 encapsulation | fixed `m` | random `m` |
| ECDSA P-256, P-384 and Ed25519 signing (rustls-pure's signers), random messages | fixed key | random keys |
| AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305 opening a 16 KiB record | tag wrong in its first byte | tag wrong in its last byte |

**The control comes first.** It times the same operation on the same kind of input in both
classes, where there is nothing to find. If its |t| reaches 4.5, the machine or the way it is
timed would make every other result meaningless, and the run fails as unreliable rather than
reporting leaks or passes.

How the harness avoids differences that are not the secret's, each found on a real runner:

- Inputs are made afresh for every sample, by arithmetic on the class rather than a branch on
  it: a "fixed" secret is the random bytes masked out, so both classes do the same work, and
  `op` takes its input by value, so it finds it in the same place in memory whatever the
  class. A pool of inputs per class put each class in its own memory, and on aarch64 Linux
  that alone gave the control |t| of 24.
- Where inputs come from a table, both classes' entries are read and a mask keeps one, so the
  class decides no memory access, not even before the timer starts. On Neoverse N2, reading
  only the sample's class's entry was enough to make the X25519 all-zero check differ (below).
- Signing signs a random message in both classes. With one message a fixed key gives the same
  signature every time, and encoding it (ECDSA's DER is 70 to 72 bytes for P-256, depending on
  the signature) took a different time from the random keys' signatures: |t| up to 19 for
  P-256, from the length of a public signature. Without the DER encoding, or with random
  messages, it was gone on every runner.
- An AEAD is tested with two wrong tags, not a right one and a wrong one: a record is decrypted
  only once its tag is right, so the two take different times by design, and whether a record
  was accepted is not secret, since a refused one ends the connection.

The test is `#[ignore]`d, since timing on shared machines is too noisy for every pull request
and only a release build counts. `.github/workflows/rustls-pure-ct.yml` runs it weekly and on
demand, on x86_64 Linux, aarch64 Linux (Neoverse N2) and macOS (Apple silicon), pinned to one
core on Linux, and puts every |t| in the run's summary:

```sh
cargo test --release -p rustls-pure --lib constant_time -- --ignored --nocapture
```

### Results

The run of 2026-10-23 on the code as it is here, |t| per target (the largest over the crops;
4.5 or more counts as a difference):

| Target | x86_64 Linux (AMD EPYC 7763) | aarch64 Linux (Neoverse N2) | Apple silicon (macOS) |
|---|---|---|---|
| **Control** | 2.15 | 2.65 | 1.39 |
| X25519 key exchange | 0.77 | 1.46 | 2.30 |
| X25519 all-zero check | 1.52 | 1.17 | 1.45 |
| P-256 key exchange | 2.50 | 1.24 | 2.42 |
| P-384 key exchange | 1.79 | 1.40 | 1.45 |
| ML-KEM-768 decapsulation, valid vs tampered | 1.38 | 1.47 | 1.59 |
| X25519MLKEM768, client, valid vs tampered | 0.88 | 2.49 | 1.06 |
| ML-KEM-768 encapsulation | 1.93 | 1.36 | 1.49 |
| ECDSA P-256 signing | 2.01 | 1.68 | 1.05 |
| ECDSA P-384 signing | 1.29 | 0.61 | 1.92 |
| Ed25519 signing | 1.90 | 0.90 | 4.22 |
| AES-128-GCM open, wrong tag | 0.85 | 1.81 | 1.90 |
| AES-256-GCM open, wrong tag | 2.36 | 1.83 | 1.39 |
| ChaCha20-Poly1305 open, wrong tag | 1.07 | 1.87 | 0.98 |

Each target took 56,000 to 2,000,000 samples; the clocks' steps were 20 ns (x86_64), 24 ns
(Neoverse N2, a 1 GHz counter) and 41 ns (Apple silicon). Earlier runs with the same
harness but for the change below, on AMD EPYC 9V45 and 9V74, found no difference in any target
on x86_64 or Apple silicon either.

**Explained: X25519's all-zero check on Neoverse N2.** Before the harness read both classes'
entries, this target's |t| on GitHub's aarch64 Linux runners was between 0.8 and 58 over ten
runs, at 4.5 or more in seven, with the means within a few nanoseconds of 44 µs and the
difference in the tail; never on x86_64 or Apple silicon. It came from the harness, not from
the secret. On Neoverse N2 (runs of 2026-10-23, |t|):

| Variant | \|t\| |
|---|---|
| Peer keys of the two classes side by side, only the class's one read (as before) | 9.36, 9.57 |
| The two classes' peer keys in separate vectors, only the class's one read | 19.74, 1.18, 1.91 |
| Side by side, both read, a mask keeps one | 1.08, 1.29 |
| Separate vectors, both read, a mask keeps one | 0.86, 1.13 |
| The labels swapped / the same peers in both classes (A/A) | 1.61 / 2.03, 2.12 |
| Classes by private key instead of by result, with one peer | 2.48 |
| Without the all-zero step (Diffie-Hellman and the copy) | 1.86 |
| x25519-dalek's Diffie-Hellman alone | 2.52, 1.95 |
| `all_zero` alone / rustls's copy alone | 0.75 / 1.83 |

Whether the result starts with a zero byte makes no difference however it is timed; where the
harness read the public peer key from, outside the timing, did, by an amount and in a direction
that changed with where the vectors happened to lie in memory. Reading both entries removed it
in every layout. The classes' timings were also checked against a run order effect (each half
of a run alone) and against the null (the labels shuffled: |t| at most 3.6).

## 5. What this does not show

- **No leak detected is not constant time.** A t-test finds differences larger than the noise
  of the machine it runs on, for the inputs it is given; a smaller difference, or one that only
  other inputs trigger, passes. A cloud runner is noisier than an attacker's quiet machine
  next to the victim, and an attacker can take far more samples.
- **Only time is measured.** Cache-timing attacks from another process on the same core,
  power, electromagnetic emissions and frequency scaling (Hertzbleed) are not.
- **Only the machines CI runs on.** Other CPUs, other compilers and other target features can
  compile the same code differently. Without AES-NI / ARMv8 AES, AES and GHASH use their
  software code, which the runners do not exercise.
- **The machine code of the primitives is not checked.** Section 3 checks rustls-pure's own
  two functions; a tool like `cargo-checkct` (BINSEC) would follow every branch and memory
  access on secrets through the RustCrypto crates too, which has not been done here.
- **Arm's DIT and Intel's DOIT are off.** Without them the CPU makers do not promise that even
  plain arithmetic takes the same time for all data; the tests measure the machines as they
  run by default. The aarch64 Linux runners do not offer DIT at all.
