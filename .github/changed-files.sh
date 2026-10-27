#!/usr/bin/env bash
# Which of a workflow's jobs a run needs: for each output the workflow has in .github/relevance,
# writes `<output>=true` or `<output>=false` to $GITHUB_OUTPUT (stdout without it), true when
# the run's changes include a file that output's rules count.
#
#   bash .github/changed-files.sh <workflow>
#
# A pull request's changes are those since its merge base with the base branch (BASE_SHA and
# HEAD_SHA, from the event). A scheduled run of a workflow with `schedule once-per-commit` or
# `schedule since-last-success` checks nothing when the workflow already ran to the end on this
# commit of main, passed or failed: its checks are deterministic, and the earlier run shows the
# result. Otherwise the first checks everything and the second the changes since the commit of
# the workflow's last successful scheduled run (both with gh, GH_TOKEN and `actions: read`).
# Every other run checks everything. The checkout needs the history of both commits
# (`fetch-depth: 0`).
set -euo pipefail

workflow=$1
here=$(dirname "$0")
output=${GITHUB_OUTPUT:-/dev/stdout}

# The workflow's outputs and their rules, in order; every output counts the workflow's own file
# and the files that decide relevance
declare -A rules=()
outputs=()
schedule=everything
while read -r name out rest; do
  [ "$name" = "$workflow" ] || continue
  if [ "$out" = schedule ]; then
    schedule=$rest
    continue
  fi
  if [ -z "${rules[$out]+set}" ]; then
    outputs+=("$out")
    rules[$out]=""
  fi
  rules[$out]+=" $rest"
done < <(grep -v -E '^\s*(#|$)' "$here/relevance")
if [ ${#outputs[@]} -eq 0 ]; then
  echo "::error::$workflow has no rules in .github/relevance" >&2
  exit 1
fi
common=".github/workflows/$workflow.yml .github/relevance .github/changed-files.sh .github/dependency-tree.py"

everything() {
  echo "$1"
  for out in "${outputs[@]}"; do echo "$out=true" >> "$output"; done
  exit 0
}
nothing() {
  echo "$1"
  for out in "${outputs[@]}"; do echo "$out=false" >> "$output"; done
  exit 0
}

event=${GITHUB_EVENT_NAME:-pull_request}
case $event in
  pull_request)
    base=$(git merge-base "$BASE_SHA" "$HEAD_SHA")
    head=$HEAD_SHA
    ;;
  schedule)
    head=${HEAD_SHA:-$(git rev-parse HEAD)}
    if [ "$schedule" = everything ]; then
      everything "A scheduled run of $workflow: everything is checked"
    fi
    # A run of this workflow that ended on this commit, whatever its conclusion, but a cancelled
    # one, which did not finish its checks
    ran=$(gh run list --workflow "$workflow.yml" --branch main --commit "$head" \
      --status completed --limit 20 --json databaseId,conclusion \
      --jq '[.[] | select(.conclusion != "cancelled")][0].databaseId // empty')
    if [ -n "$ran" ]; then
      nothing "$workflow already ran on $head (run $ran): nothing to check"
    fi
    if [ "$schedule" = once-per-commit ]; then
      everything "$workflow has not run on $head yet: everything is checked"
    fi
    base=$(gh run list --workflow "$workflow.yml" --branch main --event schedule \
      --status success --limit 1 --json headSha --jq '.[0].headSha // empty')
    [ -n "$base" ] || everything "No successful scheduled run yet: everything is checked"
    git cat-file -e "$base^{commit}" 2>/dev/null ||
      everything "$base, the last successful scheduled run, is not on main: everything is checked"
    echo "Changes since $base, the last successful scheduled run"
    ;;
  *)
    everything "A $event run: everything is checked"
    ;;
esac

# A pattern as an extended regular expression; `**/` may match nothing, so `**/Cargo.toml`
# matches the root's Cargo.toml too
regex() {
  printf '^%s$' "$(printf '%s' "$1" | sed -e 's/[.+^$()|{}\\[]/\\&/g' \
    -e 's|\*\*/|\x01|g' -e 's/\*\*/\x02/g' -e 's|\*|[^/]*|g' -e 's|?|[^/]|g' \
    -e 's|\x01|(.*/)?|g' -e 's/\x02/.*/g')"
}

# Whether a file differs other than in its version numbers (1.2.3, v1.2.3-rc.1): a release
# moves the image tags and download URLs in the deploy manifests, and nothing else
versionless() {
  if git cat-file -e "$1:$2" 2>/dev/null; then
    git show "$1:$2" | sed -E 's/v?[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.]+)?/<version>/g'
  else
    echo "<no file>"
  fi
}
beyond_versions() {
  [ "$(versionless "$base" "$1")" != "$(versionless "$head" "$1")" ]
}

# Whether a package's dependencies and build manifests differ between the two commits, once
# per package
declare -A crate_changed=()
crate_differs() {
  if [ -z "${crate_changed[$1]+set}" ]; then
    local before after
    before=$(python3 "$here/dependency-tree.py" "$base" "$1") || exit 1
    after=$(python3 "$here/dependency-tree.py" "$head" "$1") || exit 1
    if [ "$before" = "$after" ]; then
      crate_changed[$1]=false
    else
      crate_changed[$1]=true
    fi
    echo "  $1's dependencies or build manifests changed: ${crate_changed[$1]}"
  fi
  [ "${crate_changed[$1]}" = true ]
}

changed=$(git diff --name-only "$base...$head")
for out in "${outputs[@]}"; do
  read -r -a list <<< "${rules[$out]}"
  crates=()
  patterns=()
  regexes=()
  for rule in "${list[@]}"; do
    case $rule in
      crate:*) crates+=("${rule#crate:}") ;;
      *) patterns+=("$rule") ;;
    esac
  done
  for pattern in "${patterns[@]}" $common; do
    pattern=${pattern#!}
    regexes+=("$(regex "${pattern#\~}")")
  done
  patterns+=($common)

  relevant=false
  echo "$out:"
  while IFS= read -r file; do
    [ -n "$file" ] || continue
    hit=false
    case $file in
      Cargo.lock | Cargo.toml | */Cargo.toml)
        if [ ${#crates[@]} -gt 0 ]; then
          for crate in "${crates[@]}"; do
            if crate_differs "$crate"; then hit=true; fi
          done
          if $hit; then echo "  checked: $file"; relevant=true; fi
          continue
        fi
        ;;
    esac
    for i in "${!patterns[@]}"; do
      if [[ $file =~ ${regexes[$i]} ]]; then
        case ${patterns[$i]} in
          '!'*) hit=false ;;
          '~'*) if beyond_versions "$file"; then hit=true; else hit=false; fi ;;
          *) hit=true ;;
        esac
      fi
    done
    if $hit; then
      echo "  checked: $file"
      relevant=true
    fi
  done <<< "$changed"
  echo "  $out=$relevant"
  echo "$out=$relevant" >> "$output"
done
