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

The release workflow publishes amate, kuebiko, kamui, furnace-server,
furnace-burn and furnace-cli to crates.io in that order, through crates.io's Trusted Publishing
(no token stored in the repository); the first three only when the repository variable
`PUBLISH_PART_CRATES` is `true`, furnace-burn and furnace-cli only when `PUBLISH_BURN_CRATES` is.
furnace-server depends on amate and kuebiko, so it can be published only once
they are on crates.io.

### Publishing a new crate for the first time

Trusted Publishing works only for a crate that is already on crates.io, so a new crate's first
version is published by hand, by an owner, with an API token (crates.io → Account Settings → API
Tokens, scope `publish-new`):

1. Wait until the release workflow has published the crates the new one depends on (for
   furnace-burn, furnace-server of the new version). amate, kuebiko and
   kamui depend on nothing of furnace's, so they come first, before the release.
2. `cargo login` with the token, then, from a checkout of the release tag:
   `cargo publish -p <crate>`, one at a time in the order above (furnace-burn before
   furnace-cli).
3. On crates.io, for each new crate: Settings → Trusted Publishing → Add → GitHub, with
   repository owner `Gilfeather`, repository name `furnace`, workflow filename `release.yml` and
   no environment.
4. `gh variable set PUBLISH_PART_CRATES --body true` (amate, kuebiko,
   kamui) or `gh variable set PUBLISH_BURN_CRATES --body true` (furnace-burn,
   furnace-cli), so the next releases publish them too.
5. Revoke the API token (`cargo logout`, and delete it on crates.io).