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

## GitHub Branch Protection

To protect the main branch:

1. Go to **Settings** → **Branches** in GitHub
2. Add rule for `main` branch
3. Enable:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

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

The release workflow (on a `v*` tag) publishes amate, kuebiko, kamui, furnace-server,
furnace-burn and furnace-cli to crates.io in that order, through crates.io's Trusted Publishing
(no token stored in the repository): amate, kuebiko and kamui only when the repository variable
`PUBLISH_PART_CRATES` is `true`, furnace-burn and furnace-cli only when `PUBLISH_BURN_CRATES` is.
A crate whose version is on crates.io already is skipped (`.github/publish-crate.sh`), so a
version published by hand before the tag does not fail the release. furnace-server depends on
amate and kuebiko, furnace-cli on kamui: their versions must be on crates.io before the
workflow reaches those crates, published by the workflow itself or by hand.

### Publishing a new crate for the first time

Trusted Publishing works only for a crate that is already on crates.io, so a new crate's first
version is published by hand, by an owner, with an API token (crates.io → Account Settings → API
Tokens, scope `publish-new`). Whether that comes before or after the tag depends on where the
new crate sits:

- **A crate that furnace-server or furnace-cli depends on** (amate, kuebiko and kamui in
  0.13.0): by hand *before* the tag, or the workflow fails at furnace-server.
  1. Merge the release PR (versions bumped).
  2. `cargo login` with the token, then, from an up-to-date `main`:
     `cargo publish --locked -p amate`, `cargo publish --locked -p kuebiko`,
     `cargo publish --locked -p kamui`.
  3. Trusted Publishing for each (below), then
     `gh variable set PUBLISH_PART_CRATES --body true`.
  4. Revoke the API token (`cargo logout`, and delete it on crates.io).
  5. Push the `v<version>` tag on that `main` commit. The workflow skips the three versions
     already on crates.io and publishes furnace-server, furnace-burn and furnace-cli.
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
