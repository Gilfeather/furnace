# 🛠️ Development Guide

## Branch Strategy

We use a feature branch workflow to maintain code quality and enable collaborative development.

### Branch Naming Convention

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes  
- `docs/documentation-update` - Documentation changes
- `perf/performance-improvement` - Performance optimizations
- `refactor/code-cleanup` - Code refactoring
- `test/test-improvements` - Test additions/improvements

### Workflow

1. **Create a branch from main**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   ```bash
   # Make changes
   cargo test  # Ensure tests pass
   cargo clippy  # Check for lints
   cargo fmt  # Format code
   ```

3. **Commit with descriptive messages**
   ```bash
   git add .
   git commit -m "feat: add new inference optimization
   
   - Implement tensor caching for repeated inputs
   - Reduce memory allocations by 30%
   - Add benchmarks for performance validation"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   gh pr create --title "Add inference optimization" --body "Description of changes"
   ```

5. **After PR approval, merge and cleanup**
   ```bash
   git checkout main
   git pull origin main
   git branch -d feature/your-feature-name
   ```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

## Code Quality Checks

Before submitting a PR, ensure:

```bash
# All tests pass
cargo test

# No clippy warnings
cargo clippy --all-targets -- -D warnings

# Code is formatted
cargo fmt --check

# No security advisories, and licenses, bans and sources as deny.toml allows (what CI runs;
# `cargo install cargo-deny`)
cargo deny check
```

## When CI Runs

- **Pull requests to main** run every verification workflow: CI, Kani, TLA+, Loom, Miri,
  Security Audit and KServe end-to-end. Each starts with a `Changed files` job
  (`bash .github/changed-files.sh <workflow>`) that decides which of its jobs the pull request
  needs, from the diff since the pull request's merge base and the rules in
  `.github/relevance`. Each ends in one gate job, `<Workflow> required`, which passes when
  every job passed or was skipped as not needed, and fails when one failed or was cancelled. A
  newer push to the same pull request cancels the older run.
- **Pushes to main** run nothing: what is merged has passed as a pull request. Only the
  release workflows run on tags.
- **Nightly on main** CI runs everything pull requests leave out (Windows and macOS, beta and
  the MSRV, Metal, the vendored kernels, the bench harnesses, furnace-candle, the Kubernetes
  manifests, coverage). TLA+, Loom and `kani-nightly.yml` (the Kani harnesses too long for a
  pull request) run nightly, Miri weekly, each on what changed since its last successful
  scheduled run. None of these, nor the weekly soak test, runs twice on one commit of main:
  when the workflow already ran to the end on it, passed or failed, a scheduled run checks
  nothing, since the lockfile and pinned tools would give the same result and the earlier run
  shows it. Kani's pull request harnesses have no schedule: they were proved on the pull
  request that merged them. Weekly and in full, since what they check changes outside the
  repository: Security Audit (new advisories; these alone also on the other days, for the
  workspace and the bench harnesses) and KServe end-to-end (downloads the release and KServe).
  Scheduled runs are never cancelled. The README's CI badge shows the last nightly CI run on
  main.
- **By hand only:** the benchmarks (`bench.yml`); nothing measures on a schedule or a release.

### Kani proofs not run in CI

One Kani proof is not run in CI until it is decomposed: it never completed on GitHub runners
(cut at the six-hour limit or by a runner shutdown, with no counterexample). Its code stays, and
`cargo kani -p <package> --harness <name> --exact -Z function-contracts -Z stubbing` runs it
locally. Kani's `Every Kani harness runs` check lists it as parked, so no other proof can drop out
of CI unnoticed.

- furnace-server: `tensor::verification::concat_batch_never_panics` (`joined_batch`'s contract,
  which it builds on, runs on every pull request).

### Which changes run which jobs

`.github/relevance` holds the rules, one line per workflow and output: path patterns in the
syntax of GitHub's `paths` filters, and `crate:<package>`. The rules follow what each job
reads:

- **Docs, licences and changelogs run nothing**, except in CI: the Markdown files whose Rust
  examples compile as doc-tests (README.md, CONTRIBUTING.md, examples/README.md, docs/*.md)
  and docs/openapi.json, which the server's tests compare with its routes.
- **Each check counts only the files it reads.** Kani a package's proofs: `kamui/src`, and furnace-server's files with proofs (`src/tensor.rs`, `src/model.rs`,
  `src/api/infer.rs`, `src/json_fast.rs`, `src/nerve/wire.rs`) and `build.rs`. Miri
  `kamui/src` and `kamui/tests`. Loom `amate/src` and `src/nerve`, each for its own models. TLA+
  `specs/`. CI everything else in the workspace, but other workflows' files, `specs/`, `deploy/`,
  `benches/` (workspaces of their own, which the nightly run checks), `dev/` and packaging.
  cargo-deny the lockfile, the workspace's manifests and `deny.toml` (the advisories the bench
  harnesses' lockfiles too); CodeQL the workspace's Rust code.
- **Cargo.toml and Cargo.lock count by what a package is built from.** For `crate:<package>`,
  `.github/dependency-tree.py` lists the package's resolved dependencies from the lockfile and
  the manifests that decide its build (its own and those of the workspace packages it reaches,
  and the workspace's profiles, lints and patches) at both commits, without the workspace's
  version numbers and the manifests' descriptive fields. A release's version bump, or an
  update of a dependency the package does not reach, leaves it unchanged.
- **A dependency update counts for the checks that run its code.** A check that runs only some
  of a package's dependencies names them, `crate:<package>=<crate>,...` (an empty list: only
  the package's own code), read off the code it runs:

  | Check | Dependencies it runs |
  | --- | --- |
  | Kani kamui, Kani furnace-server | none: their proofs call std only |
  | Loom amate | `loom` |
  | Loom furnace-server | `loom`, `tokio` (the nerve's channels), `bytes`, `tracing` |
  | Miri | all of kamui's: its tests run proptest and its random number generator |
  | CI's tests | all of the workspace's |

  So a crypto-bigint update runs CI and no proof; a tokio update runs the nerve's Loom models and CI. A Kani
  version change is in the workflow file, which counts for all of its jobs. A crate the list
  names that the package no longer reaches fails the run, so the lists cannot go stale
  quietly; a proof that starts calling a new dependency needs it added.
- **The deploy manifests count by more than their versions.** KServe end-to-end runs when
  `deploy/kserve` changes other than in its version numbers (`~` in the rules): a release's
  new image tag and download URL name what is published only after the merge.
- A workflow's own file counts for all of its jobs; another workflow's file counts for none.
  The relevance files (`.github/relevance`, `changed-files.sh`, `dependency-tree.py` and the
  self-test) count for the self-test only: a pull request that changes them runs every
  workflow's `Changed files` job with the new rules, and the self-test checks them.
- **CodeQL on a pull request** scans the workspace's Rust code with the `security-extended`
  queries (clippy covers what the quality queries add), when Rust code or any workspace
  package's dependencies change; not the benchmark harnesses or the vendored Burn backend. The
  weekly run on main scans everything with `security-and-quality`. The Rust extractor builds
  the dependencies' build scripts and procedural macros for every feature; it keeps them in a
  target directory the weekly run saves to the Actions cache and pull requests restore.

`bash .github/changed-files-test.sh` checks the rules against example pull requests (which
outputs each must turn on); CI runs it when they change. Change the rules and the examples
together.

## GitHub Branch Protection

`main` requires the gate jobs to pass before a pull request is merged (Settings → Branches →
`main` → Require status checks to pass): `CI required`, `Kani required`, `TLA+ required`,
`Loom required`, `Miri required`, `Security Audit required` and
`KServe end-to-end required`. The branch need not be up to date with main first.

## Local Development

### Prerequisites
- Rust 1.70+
- Git
- GitHub CLI (optional but recommended)

### Setup
```bash
git clone https://github.com/Gilfeather/furnace.git
cd furnace
cargo build
cargo test
```

### Running the Server
```bash
# Create sample model
cargo run --bin create_sample_model

# Start server
cargo run --bin furnace -- --model-path ./sample_model --port 3000
```

### Testing
```bash
# Unit tests
cargo test --release

# Integration tests
cargo test --release --test integration_tests

# Soak run: the server and the nervous system under a steady load (here for 10 minutes);
# .github/workflows/soak.yml runs it for hours, by hand and weekly (once per commit)
FURNACE_SOAK_SECS=600 cargo test --profile ci --test soak -- --ignored --nocapture

# Benchmarks (furnace-burn's criterion bench; `onnx` builds in furnace-burn/models/*.onnx)
cargo bench -p furnace-burn --features onnx
```

## Release Process

1. Create release branch: `git checkout -b release/v1.0.0`
2. Update the version in `Cargo.toml`, `amate/Cargo.toml`, `kuebiko/Cargo.toml`, `kamui/Cargo.toml`, `furnace-burn/Cargo.toml` and `furnace-cli/Cargo.toml` (and their dependencies on each other)
3. Update `CHANGELOG.md`
4. Create PR to main
5. After merge, create GitHub release
6. GitHub Actions will build and publish binaries

The release workflow (on a `v*` tag) publishes amate, kuebiko, kamui,
furnace-server, furnace-burn and furnace-cli to crates.io in that order, through crates.io's
Trusted Publishing (no token stored in the repository): amate, kuebiko and kamui only when the repository variable `PUBLISH_PART_CRATES` is `true`, furnace-burn and furnace-cli
only when `PUBLISH_BURN_CRATES` is. A crate whose version is on crates.io already is skipped
(`.github/publish-crate.sh`), so a version published by hand before the tag does not fail the
release. furnace-server depends on amate and kuebiko, furnace-cli on kamui: their versions must
be on crates.io before the workflow reaches those crates, published by the workflow itself or by
hand. With `tls`, furnace-server also depends on rustls-pure, which has a repository and
releases of its own ([Gilfeather/rustls-pure](https://github.com/Gilfeather/rustls-pure)); to
use a new version of it, raise furnace-server's `rustls-pure` dependency once it is on
crates.io.

### Publishing a new crate for the first time

Trusted Publishing works only for a crate that is already on crates.io, so a new crate's first
version is published by hand, by an owner, with an API token (crates.io → Account Settings → API
Tokens, scope `publish-new`). Whether that comes before or after the tag depends on where the
new crate sits:

- **A crate that furnace-server or furnace-cli depends on** (amate, kuebiko and kamui in
  0.13.0): by hand *before* the tag, or the workflow fails at furnace-server. A part that
  depends on nothing else of the workspace can instead be published at its current version as
  soon as it is on `main` (steps 2 to 4).
  1. Merge the release PR (versions bumped).
  2. `cargo login` with the token, then, from an up-to-date `main`:
     `cargo publish --locked -p amate`, `cargo publish --locked -p kuebiko`,
     `cargo publish --locked -p kamui`.
  3. Trusted Publishing for each (below), then
     `gh variable set PUBLISH_PART_CRATES --body true` (already set since 0.13.0).
  4. Revoke the API token (`cargo logout`, and delete it on crates.io).
  5. Push the `v<version>` tag on that `main` commit. The workflow skips the versions already
     on crates.io and publishes the rest.
- **A crate that depends on furnace-server** (furnace-burn and furnace-cli in 0.12.0): by hand
  *after* the tag.
  1. Wait until the release workflow has published furnace-server of the new version.
  2. `cargo login` with the token, then, from a checkout of the release tag:
     `cargo publish -p furnace-burn`, and once it is on crates.io, `cargo publish -p furnace-cli`.
  3. Trusted Publishing for each (below), then
     `gh variable set PUBLISH_BURN_CRATES --body true`.
  4. Revoke the API token.

Trusted Publishing on crates.io, for each new crate: Settings → Trusted Publishing → Add →
GitHub, with repository owner `Gilfeather`, repository name `furnace`, workflow filename
`release.yml` and no environment.
