#!/usr/bin/env bash
# The rules in .github/relevance against examples: a pull request that changes the files on the
# left must make exactly the outputs on the right true (every other one false), and so must a
# scheduled or manual run, given the workflow's earlier runs. The pull requests are made in a
# scratch repository holding a small workspace with the same package names (furnace-server,
# amate, kamui, rustls-pure), so the dependency comparison runs too; a stand-in for gh answers
# what the earlier runs were.
#
#   bash .github/changed-files-test.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(mktemp -d)
fake=$(mktemp -d)
trap 'rm -rf "$repo" "$fake"' EXIT

# The stand-in for gh: the workflow's run that ended on the commit (FAKE_RAN, a run id, or
# empty for none) and the commit of its last successful scheduled run (FAKE_LAST_SUCCESS)
cat > "$fake/gh" <<'EOF'
#!/usr/bin/env bash
case " $* " in
  *" --commit "*) echo "${FAKE_RAN:-}" ;;
  *) echo "${FAKE_LAST_SUCCESS:-}" ;;
esac
EOF
chmod +x "$fake/gh"

# Every workflow and output the rules name
mapfile -t all < <(grep -v -E '^\s*(#|$)' "$here/relevance" | awk '$2 != "schedule" {print $1 "." $2}' | sort -u)

# The scratch workspace, its dependencies shaped like the real ones: furnace-server depends on
# amate, rustls-pure, tokio, bytes, tracing and loom; amate on loom; rustls-pure on
# crypto-bigint and rustls, which depends on zeroize; kamui on nothing
cd "$repo"
git init -q
git config user.email test@example.com
git config user.name test
mkdir -p .github/workflows amate kamui rustls-pure
cp "$here/relevance" "$here/changed-files.sh" "$here/dependency-tree.py" .github/
manifest() {
  printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2024"\nlicense = "MIT"\n\n[dependencies]\n%s\n' "$1" "$2"
}
{
  manifest furnace-server 'amate = { path = "amate", version = "0.1.0" }
rustls-pure = { path = "rustls-pure", version = "0.1.0" }
tokio = "1"
bytes = "1"
tracing = "0.1"
loom = "0.7"'
  printf '\n[workspace]\nmembers = [".", "amate", "kamui", "rustls-pure"]\n\n[profile.ci]\ninherits = "release"\n'
} > Cargo.toml
manifest amate 'loom = "0.7"' > amate/Cargo.toml
manifest kamui '' > kamui/Cargo.toml
manifest rustls-pure 'crypto-bigint = "0.7"
rustls = "0.23"' > rustls-pure/Cargo.toml
lock_package() {
  printf '\n[[package]]\nname = "%s"\nversion = "%s"\n' "$1" "$2"
  [ -z "$3" ] || printf 'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "%s"\n' "$3"
  [ -z "$4" ] || printf 'dependencies = [\n%b]\n' "$4"
}
{
  printf 'version = 4\n'
  lock_package amate 0.1.0 '' ' "loom",\n'
  lock_package bytes 1.0.0 bbb0 ''
  lock_package crypto-bigint 0.7.0 aaa0 ''
  lock_package furnace-server 0.1.0 '' ' "amate",\n "bytes",\n "loom",\n "rustls-pure",\n "tokio",\n "tracing",\n'
  lock_package kamui 0.1.0 '' ''
  lock_package loom 0.7.2 ddd0 ''
  lock_package rustls 0.23.0 eee0 ' "zeroize",\n'
  lock_package rustls-pure 0.1.0 '' ' "crypto-bigint",\n "rustls",\n'
  lock_package tokio 1.40.0 ccc0 ''
  lock_package tracing 0.1.0 fff0 ''
  lock_package zeroize 1.0.0 999a ''
} > Cargo.lock
# Deploy manifests naming the release, as deploy/ does
mkdir -p deploy/kserve deploy/helm/furnace
printf 'spec:\n  containers:\n  - image: ghcr.io/gilfeather/furnace:0.1.0\n' > deploy/kserve/servingruntime.yaml
printf 'storageUri: https://github.com/Gilfeather/furnace/releases/download/v0.1.0/resnet18.tar.gz\n' \
  > deploy/kserve/inferenceservice.yaml
printf 'version: 0.1.0\nappVersion: "0.1.0"\n' > deploy/helm/furnace/Chart.yaml
git add -A
git commit -q -m base
base=$(git rev-parse HEAD)

failed=0
# commit_changes '<changed files or @edits>': a commit on top of the base with those changes, in
# $head
commit_changes() {
  local changes=$1
  git checkout -q --detach "$base"
  for change in $changes; do
    case $change in
      @bump-versions)  # a release: every workspace package's version and the requirements on it
        sed -i 's/0\.1\.0/0.2.0/' Cargo.toml amate/Cargo.toml kamui/Cargo.toml rustls-pure/Cargo.toml
        sed -i '/^name = "\(amate\|furnace-server\|kamui\|rustls-pure\)"/{n;s/0\.1\.0/0.2.0/}' Cargo.lock ;;
      @release)  # the versions of a release, wherever they are written
        sed -i 's/0\.1\.0/0.2.0/' Cargo.toml amate/Cargo.toml kamui/Cargo.toml rustls-pure/Cargo.toml
        sed -i '/^name = "\(amate\|furnace-server\|kamui\|rustls-pure\)"/{n;s/0\.1\.0/0.2.0/}' Cargo.lock
        sed -i 's/0\.1\.0/0.2.0/g' deploy/kserve/*.yaml deploy/helm/furnace/Chart.yaml
        echo 'furnace 0.2.0' >> README.md ;;
      @kserve-resources)
        printf '    resources:\n      limits:\n        memory: 2Gi\n' >> deploy/kserve/servingruntime.yaml ;;
      @kamui-dependency)  # a new dependency of kamui, in its manifest and the lockfile
        sed -i 's/^\[dependencies\]$/[dependencies]\nlibm = "0.2"/' kamui/Cargo.toml
        lock_package libm 0.2.0 bbb0 '' >> Cargo.lock
        sed -i '/^name = "kamui"/{n;s/$/\ndependencies = [\n "libm",\n]/}' Cargo.lock ;;
      @update-tokio)
        sed -i 's/^version = "1\.40\.0"/version = "1.40.1"/; s/ccc0/ccc1/' Cargo.lock ;;
      @update-crypto-bigint)  # a crypto crate rustls-pure's proofs and machine code never call
        sed -i 's/^version = "0\.7\.0"/version = "0.7.1"/; s/aaa0/aaa1/' Cargo.lock ;;
      @update-zeroize)  # a crate rustls-pure's proofs call
        sed -i 's/999a/999b/' Cargo.lock ;;
      @kamui-feature)
        printf '\n[features]\nstd = []\n' >> kamui/Cargo.toml ;;
      @profile)
        sed -i 's/inherits = "release"/inherits = "release"\nlto = false/' Cargo.toml ;;
      *)
        mkdir -p "$(dirname "$change")"
        echo "$RANDOM" >> "$change" ;;
    esac
  done
  git add -A
  git commit -q -m change
  head=$(git rev-parse HEAD)
}

# compare '<what>' '<outputs that must be true>' '<outputs that are true>' [<workflow>]: every
# output (of that workflow) true exactly when it must be
compare() {
  local what=$1 expected=" $2 " got=" $3 " only=${4:-} name want have
  for name in "${all[@]}"; do
    [ -z "$only" ] || [[ $name == "$only".* ]] || continue
    want=false
    have=false
    [[ $expected == *" $name "* ]] && want=true
    [[ $got == *" $name "* ]] && have=true
    if [ "$want" != "$have" ]; then
      echo "FAIL: $what: $name is $have, expected $want"
      failed=1
    fi
  done
  for name in $2; do
    printf '%s\n' "${all[@]}" | grep -qx -- "$name" || { echo "FAIL: no output $name"; failed=1; }
  done
}

# check '<changed files or @edits>' '<workflow.output that must be true>...': a pull request
check() {
  local changes=$1 got="" result
  commit_changes "$changes"
  for workflow in $(printf '%s\n' "${all[@]}" | cut -d. -f1 | sort -u); do
    result=$(mktemp)
    GITHUB_EVENT_NAME=pull_request BASE_SHA=$base HEAD_SHA=$head GITHUB_OUTPUT=$result \
      bash .github/changed-files.sh "$workflow" > /dev/null
    while IFS='=' read -r out value; do
      [ "$value" = true ] && got+="$workflow.$out "
    done < "$result"
    rm -f "$result"
  done
  compare "$changes" "$2" "$got"
}

# check_run <event> <workflow> <the workflow's run that ended on the commit, or -> <its last
# successful scheduled run: base, or -> '<changes since the base>' '<outputs that must be true>'
check_run() {
  local event=$1 workflow=$2 ran=$3 last=$4 got="" result
  commit_changes "$5"
  [ "$ran" != - ] || ran=""
  if [ "$last" = base ]; then last=$base; else last=""; fi
  result=$(mktemp)
  GITHUB_EVENT_NAME=$event HEAD_SHA=$head GITHUB_OUTPUT=$result FAKE_RAN=$ran \
    FAKE_LAST_SUCCESS=$last PATH="$fake:$PATH" bash .github/changed-files.sh "$workflow" > /dev/null
  while IFS='=' read -r out value; do
    [ "$value" = true ] && got+="$workflow.$out "
  done < "$result"
  rm -f "$result"
  compare "$event $workflow ($5; ran $3; last success $4)" "$6" "$got" "$workflow"
}

# Docs, licences and changelogs alone run nothing, but CI's doc-tests
check 'LICENSE kamui/LICENSE rustls-pure/LICENSE CHANGELOG.md rustls-pure/CHANGELOG.md kamui/README.md rustls-pure/README.md specs/README.md deploy/README.md docs/images/a.png dev/AUDIT.md' ''
check 'README.md' 'ci.relevant'
check 'docs/API.md' 'ci.relevant'
check 'docs/openapi.json' 'ci.relevant'
# A pull request like one that ships licence files, docs and a change to rustls-pure's code
check 'kamui/LICENSE amate/LICENSE CHANGELOG.md rustls-pure/README.md rustls-pure/src/lib.rs .github/workflows/rustls-pure.yml' \
  'ci.relevant kani.rustls-pure rustls-pure.machine-code rustls-pure.interop rustls-pure.targets security.codeql'
# Manifests and the lockfile count by what the package is built from
check '@bump-versions' 'ci.relevant security.advisories security.deny security.dependency-review'
# A release: versions in the manifests, the lockfile, the deploy manifests and the README run
# only the cheap checks; no proof, Miri, Loom, interoperability or KServe
check '@release' 'ci.relevant security.advisories security.deny security.dependency-review'
# A dependency counts for the checks that run its code: a new one of kamui for Miri, which runs
# kamui's tests, not for the proofs of kamui's own code
check '@kamui-dependency' 'ci.relevant miri.relevant security.advisories security.deny security.dependency-review'
# tokio for the nerve's Loom models, which build its channels; no proof calls it
check '@update-tokio' 'ci.relevant loom.server security.advisories security.deny security.dependency-review'
# crypto-bigint for the tests that run rustls-pure's crypto (interoperability, the target builds,
# the timing tests); not for the proofs or the machine-code check, which never call it
check '@update-crypto-bigint' 'ci.relevant rustls-pure.interop rustls-pure.targets rustls-pure-ct.timing security.advisories security.deny security.dependency-review'
check '@update-zeroize' 'ci.relevant kani.rustls-pure rustls-pure.interop rustls-pure.targets rustls-pure-ct.timing security.advisories security.deny security.dependency-review'
check '@kamui-feature' 'ci.relevant kani.kamui kani-nightly.kamui miri.relevant security.deny security.dependency-review'
check '@profile' 'ci.relevant kani.rustls-pure kani.kamui kani.server kani-nightly.kamui kani-nightly.server miri.relevant loom.amate loom.server rustls-pure.machine-code rustls-pure.interop rustls-pure.targets rustls-pure-ct.timing security.deny security.dependency-review'
# Each check's own files
check 'kamui/src/lib.rs' 'ci.relevant kani.kamui kani-nightly.kamui miri.relevant security.codeql'
check 'kamui/tests/global.rs' 'ci.relevant miri.relevant security.codeql'
check 'amate/src/lib.rs' 'ci.relevant loom.amate security.codeql'
check 'src/nerve/motor/mod.rs' 'ci.relevant loom.server security.codeql'
check 'src/nerve/wire.rs' 'ci.relevant kani.server kani-nightly.server loom.server security.codeql'
check 'src/api.rs' 'ci.relevant security.codeql'
check 'build.rs' 'ci.relevant kani.server kani-nightly.server loom.server security.codeql'
check 'specs/Motor.tla' 'tla.relevant'
check 'rustls-pure/ct-asm.sh' 'rustls-pure.machine-code'
check 'rustls-pure/interop/run.sh' 'rustls-pure.interop'
check 'rustls-pure/examples/interop-client.rs' 'ci.relevant rustls-pure.interop security.codeql'
check 'rustls-pure/wycheproof/x.json rustls-pure/tests/handshake.rs' 'ci.relevant security.codeql'
check 'deploy/kserve/servingruntime.yaml' 'kserve-e2e.relevant'
check '@kserve-resources' 'kserve-e2e.relevant'
check 'deploy/helm/furnace/values.yaml' ''
check 'deny.toml' 'security.advisories security.deny'
check 'benches/nerve/src/main.rs benches/nerve/Cargo.toml' 'security.codeql security.dependency-review'
check 'furnace-burn/vendor/burn-flex/src/lib.rs' 'ci.relevant'
# A workflow's own file runs all of it; the relevance files run everything
check '.github/workflows/kani.yml' 'kani.rustls-pure kani.kamui kani.server'
check '.github/workflows/kserve-e2e.yml' 'kserve-e2e.relevant'
check '.github/workflows/rustls-pure-ct.yml' 'rustls-pure-ct.timing'
check '.github/workflows/release.yml .github/workflows/bench.yml' ''
check '.github/workflows/soak.yml' 'soak.run'
check '.github/workflows/ci.yml' 'ci.relevant ci.selftest'
check '.github/relevance' "${all[*]}"

# Scheduled runs: the deterministic checks never run twice on one commit, whether the earlier run
# passed or failed; the others always run. A manual run checks everything.
#         event             workflow     ran  last success  changes             must be true
check_run schedule          kani-nightly 1234 base          'kamui/src/lib.rs'  ''
check_run schedule          kani-nightly -    base          'kamui/src/lib.rs'  'kani-nightly.kamui'
check_run schedule          kani-nightly -    base          'README.md'         ''
check_run schedule          kani-nightly -    -             'README.md'         'kani-nightly.kamui kani-nightly.server'
check_run schedule          miri         1234 -             'kamui/src/lib.rs'  ''
check_run schedule          loom         -    base          'amate/src/lib.rs'  'loom.amate'
check_run schedule          tla          1234 base          'specs/Motor.tla'   ''
check_run schedule          ci           1234 -             'src/api.rs'        ''
check_run schedule          ci           -    -             'README.md'         'ci.relevant ci.selftest'
check_run schedule          soak         1234 -             'src/api.rs'        ''
check_run schedule          soak         -    -             'src/api.rs'        'soak.run'
check_run schedule          security     1234 -             'README.md'         'security.advisories security.deny security.dependency-review security.codeql'
check_run schedule          rustls-pure  1234 -             'README.md'         'rustls-pure.machine-code rustls-pure.interop rustls-pure.targets'
check_run workflow_dispatch kani-nightly 1234 base          'README.md'         'kani-nightly.kamui kani-nightly.server'

if [ "$failed" = 0 ]; then echo "every example as expected"; fi
exit $failed
