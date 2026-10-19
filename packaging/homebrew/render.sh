#!/usr/bin/env bash
# Render the Homebrew formula for a release from GitHub's asset digests.
#
#     packaging/homebrew/render.sh 0.9.0 > Formula/furnace.rb
#
# Reads the release's assets with `gh api` (GitHub records each asset's SHA-256), fills
# furnace.rb.in, and fails if an asset the formula needs is missing.
set -euo pipefail

version=$1
repo=Gilfeather/furnace
template="$(dirname "$0")/furnace.rb.in"

# "name sha256" per asset that has a digest
digests=$(gh api "repos/$repo/releases/tags/v$version" --jq \
  '.assets[] | select((.digest // "") | startswith("sha256:"))
   | "\(.name) \(.digest | ltrimstr("sha256:"))"')

formula=$(<"$template")
formula=${formula//@VERSION@/$version}
for name in $(grep -o '@SHA_[^@]*@' "$template" | sed 's/^@SHA_//; s/@$//' | sort -u); do
  sha=$(awk -v name="$name" '$1 == name { print $2 }' <<<"$digests")
  if [ -z "$sha" ]; then
    echo "release v$version has no SHA-256 for $name" >&2
    exit 1
  fi
  formula=${formula//"@SHA_${name}@"/$sha}
done
printf '%s\n' "$formula"
