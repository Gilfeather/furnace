#!/usr/bin/env bash
# Whether a pull request changes a file a workflow checks: writes `relevant=true` or
# `relevant=false` to $GITHUB_OUTPUT, for the workflow's heavy jobs to run on or skip. The
# arguments are patterns as GitHub's `paths` filters take them, applied in order with a later one
# overriding an earlier one: `**` matches any part of a path, `*` and `?` stay within one
# directory, and a leading `!` leaves the files it matches out again. The pull request's changes
# are those since its merge base with the base branch (BASE_SHA and HEAD_SHA, from the event;
# the checkout needs their history). Every run that is not for a pull request (the scheduled and
# manual ones) checks everything.
set -euo pipefail

output=${GITHUB_OUTPUT:-/dev/stdout}
if [ "${GITHUB_EVENT_NAME:-}" != pull_request ]; then
  echo "A $GITHUB_EVENT_NAME run: everything is checked"
  echo relevant=true >> "$output"
  exit 0
fi

# A pattern as an extended regular expression; `**/` may match nothing, so `**/Cargo.toml`
# matches the root's Cargo.toml too
regex() {
  printf '^%s$' "$(printf '%s' "$1" | sed -e 's/[.+^$()|{}\\[]/\\&/g' \
    -e 's|\*\*/|\x01|g' -e 's/\*\*/\x02/g' -e 's|\*|[^/]*|g' -e 's|?|[^/]|g' \
    -e 's|\x01|(.*/)?|g' -e 's/\x02/.*/g')"
}

includes=()
regexes=()
for pattern in "$@"; do
  if [ "${pattern:0:1}" = '!' ]; then
    includes+=(false)
    regexes+=("$(regex "${pattern:1}")")
  else
    includes+=(true)
    regexes+=("$(regex "$pattern")")
  fi
done

changed=$(git diff --name-only "$BASE_SHA...$HEAD_SHA")
relevant=false
while IFS= read -r file; do
  [ -n "$file" ] || continue
  hit=false
  for i in "${!regexes[@]}"; do
    if [[ $file =~ ${regexes[$i]} ]]; then
      hit=${includes[$i]}
    fi
  done
  if [ "$hit" = true ]; then
    echo "checked: $file"
    relevant=true
  fi
done <<< "$changed"

echo "relevant=$relevant"
echo "relevant=$relevant" >> "$output"
