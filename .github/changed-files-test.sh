#!/usr/bin/env bash
# The rules in .github/relevance against examples: a pull request that changes the files on the
# left must make exactly the outputs on the right true (every other one false). The pull
# requests are made in a scratch repository holding a small workspace with the same package
# names (furnace-server, amate, kamui, rustls-pure), so the dependency comparison runs too.
#
#   bash .github/changed-files-test.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(mktemp -d)
trap 'rm -rf "$repo"' EXIT

# Every workflow and output the rules name
mapfile -t all < <(grep -v -E '^\s*(#|$)' "$here/relevance" | awk '$2 != "schedule" {print $1 "." $2}' | sort -u)

# The scratch workspace: furnace-server depends on amate and tokio, rustls-pure on aes, kamui
# on nothing
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
tokio = "1"'
  printf '\n[workspace]\nmembers = [".", "amate", "kamui", "rustls-pure"]\n\n[profile.ci]\ninherits = "release"\n'
} > Cargo.toml
manifest amate '' > amate/Cargo.toml
manifest kamui '' > kamui/Cargo.toml
manifest rustls-pure 'aes = "0.8"' > rustls-pure/Cargo.toml
lock_package() {
  printf '\n[[package]]\nname = "%s"\nversion = "%s"\n' "$1" "$2"
  [ -z "$3" ] || printf 'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "%s"\n' "$3"
  [ -z "$4" ] || printf 'dependencies = [\n%b]\n' "$4"
}
{
  printf 'version = 4\n'
  lock_package aes 0.8.0 aaa0 ''
  lock_package amate 0.1.0 '' ''
  lock_package furnace-server 0.1.0 '' ' "amate",\n "tokio",\n'
  lock_package kamui 0.1.0 '' ''
  lock_package rustls-pure 0.1.0 '' ' "aes",\n'
  lock_package tokio 1.0.0 ccc0 ''
} > Cargo.lock
# Deploy manifests naming the release, as deploy/ does
mkdir -p deploy/kserve deploy/helm/furnace
printf 'spec:
  containers:
  - image: ghcr.io/gilfeather/furnace:0.1.0
' > deploy/kserve/servingruntime.yaml
printf 'storageUri: https://github.com/Gilfeather/furnace/releases/download/v0.1.0/resnet18.tar.gz
' > deploy/kserve/inferenceservice.yaml
printf 'version: 0.1.0
appVersion: "0.1.0"
' > deploy/helm/furnace/Chart.yaml
git add -A
git commit -q -m base
base=$(git rev-parse HEAD)

failed=0
# check '<changed files or @edits>' '<workflow.output that must be true>...'
check() {
  local changes=$1 expected=" $2 " got=" "
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
        sed -i 's/^version = "1\.0\.0"/version = "1.0.1"/; s/ccc0/ccc1/' Cargo.lock ;;
      @update-aes)
        sed -i 's/^version = "0\.8\.0"/version = "0.8.1"/; s/aaa0/aaa1/' Cargo.lock ;;
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
  local head result
  head=$(git rev-parse HEAD)
  for workflow in $(printf '%s\n' "${all[@]}" | cut -d. -f1 | sort -u); do
    result=$(mktemp)
    GITHUB_EVENT_NAME=pull_request BASE_SHA=$base HEAD_SHA=$head GITHUB_OUTPUT=$result \
      bash .github/changed-files.sh "$workflow" > /dev/null
    while IFS='=' read -r out value; do
      [ "$value" = true ] && got+="$workflow.$out "
    done < "$result"
    rm -f "$result"
  done
  for name in "${all[@]}"; do
    local want=false have=false
    [[ $expected == *" $name "* ]] && want=true
    [[ $got == *" $name "* ]] && have=true
    if [ "$want" != "$have" ]; then
      echo "FAIL: $changes: $name is $have, expected $want"
      failed=1
    fi
  done
  for name in $expected; do
    printf '%s\n' "${all[@]}" | grep -qx -- "$name" || { echo "FAIL: no output $name"; failed=1; }
  done
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
check '@kamui-dependency' 'ci.relevant kani.kamui kani-nightly.kamui miri.relevant security.advisories security.deny security.dependency-review'
check '@update-tokio' 'ci.relevant kani.server kani-nightly.server loom.server security.advisories security.deny security.dependency-review'
check '@update-aes' 'ci.relevant kani.rustls-pure rustls-pure.machine-code rustls-pure.interop rustls-pure.targets security.advisories security.deny security.dependency-review'
check '@kamui-feature' 'ci.relevant kani.kamui kani-nightly.kamui miri.relevant security.deny security.dependency-review'
check '@profile' 'ci.relevant kani.rustls-pure kani.kamui kani.server kani-nightly.kamui kani-nightly.server miri.relevant loom.amate loom.server rustls-pure.machine-code rustls-pure.interop rustls-pure.targets security.deny security.dependency-review'
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
check '.github/workflows/release.yml .github/workflows/bench.yml' ''
check '.github/workflows/soak.yml' 'soak.run'
check '.github/workflows/ci.yml' 'ci.relevant ci.selftest'
check '.github/relevance' "${all[*]}"

if [ "$failed" = 0 ]; then echo "every example as expected"; fi
exit $failed
