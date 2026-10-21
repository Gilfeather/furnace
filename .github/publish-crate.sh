#!/usr/bin/env bash
# Publish one crate of the workspace to crates.io, unless its version is there already: a new
# crate's first version is published by hand (see DEVELOPMENT.md), and the release workflow then
# finds it and moves on instead of failing on the duplicate.
#
# Usage: bash .github/publish-crate.sh <crate>   (with CARGO_REGISTRY_TOKEN set)
set -euo pipefail

crate="$1"
version=$(cargo metadata --no-deps --format-version 1 \
  | jq -r --arg crate "$crate" '.packages[] | select(.name == $crate) | .version')
if [ -z "$version" ]; then
  echo "::error::no crate named $crate in this workspace"
  exit 1
fi

# crates.io asks API clients for a User-Agent that says who they are
status=$(curl -s -o /dev/null -w '%{http_code}' \
  -A "furnace release workflow (https://github.com/Gilfeather/furnace)" \
  "https://crates.io/api/v1/crates/$crate/$version")
case "$status" in
  200)
    echo "$crate $version is on crates.io already; not publishing it again"
    ;;
  404)
    cargo publish --locked -p "$crate"
    ;;
  *)
    echo "::error::crates.io answered $status for $crate $version"
    exit 1
    ;;
esac
