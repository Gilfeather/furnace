#!/usr/bin/env bash
# Rewrites the measured numbers in README.md from one runner's benchmark results.
#
#   benches/readme-numbers.sh DIR [README]
#
# DIR holds what the benchmarks workflow (.github/workflows/bench.yml) saved: cpu.json,
# burn.json and overhead.txt (furnace-burn-bench), cold-start.txt and nerve.json
# (furnace-nerve-bench). README defaults to README.md. Only the text between a pair of markers
# changes, each marker on a line of its own:
#
#   <!-- bench:server-table:start -->   where, when and on what commit; the server table
#   <!-- bench:server-notes:start -->   the request overhead, the ratios, the cold start
#   <!-- bench:nerve:start -->          the nervous system's figures (needs nerve.json)
#
# and the matching `:end` markers. The workflow's manual run with `update_readme` set runs this
# and opens a pull request (docs/PERFORMANCE.md). Optional environment:
#
#   BENCH_DATE      the date of the measurement (default: today, UTC)
#   BENCH_COMMIT    the commit measured (default: git rev-parse HEAD)
#   BENCH_REPO_URL  links the commit (default: https://github.com/Gilfeather/furnace)
#   BENCH_RUN_URL   links the workflow run, when set
set -euo pipefail

dir=${1:?usage: $0 DIR [README]}
readme=${2:-README.md}
date=${BENCH_DATE:-$(date -u +%Y-%m-%d)}
commit=${BENCH_COMMIT:-$(git rev-parse HEAD)}
repo_url=${BENCH_REPO_URL:-https://github.com/Gilfeather/furnace}
run_url=${BENCH_RUN_URL:-}

for file in cpu.json burn.json overhead.txt cold-start.txt; do
  [ -f "$dir/$file" ] || { echo "$0: $dir/$file is missing" >&2; exit 1; }
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Number formatting shared by every block
formats='
  def two: . * 100 | round / 100;
  def one: . * 10 | round / 10;
  def group: if length <= 3 then . else (.[:-3] | group) + "," + .[-3:] end;
  def commas: round | tostring | group;
  def ms: if . >= 1000 then "\(. / 1000 | one) s" elif . >= 10 then "\(round) ms"
          elif . >= 1 then "\(one) ms" else "\(two) ms" end;
  def mb: if . >= 10 then "\(round) MB" else "\(one) MB" end;
  def percent: . * 100 | round;
'

# `overhead echo` prints "furnace   41.9 us": each server's own time per request
overhead() { awk -v kind="$1" '$1 == kind && $3 == "us" { print $2 }' "$dir/overhead.txt"; }
# `cold-start` prints "binary ready  median  52.3 ms ...": the median of one moment
cold() { awk -v name="$1" '{ for (i = 2; i < NF; i++) if ($i == "median") { n = $2; for (j = 3; j < i; j++) n = n " " $j; if (n == name) print $(i + 1) } }' "$dir/cold-start.txt"; }

# Where, when, on what: the line above the table
cpu=$(jq -r '"\(.model), \(.cores) cores"' "$dir/cpu.json")
runner=$(jq -r .label "$dir/burn.json")
clients=$(jq -r '.models[0].servers[0].load[-1].clients' "$dir/burn.json")
runs=$(jq -r .runs "$dir/burn.json")
run=""
[ -n "$run_url" ] && run=" ([workflow run]($run_url))"

{
  echo "On GitHub's \`$runner\` runner ($cpu), $date, commit [\`${commit:0:7}\`]($repo_url/commit/$commit),"
  echo "$clients concurrent clients, medians of $runs runs$run:"
  echo
  jq -r "$formats"'
    def server($m; $s): .models[] | select(.model == $m) | .servers[] | select(.server == $s);
    def at_load: .load[-1];
    def name: {echo: "Model that returns its input", digits: "Small CNN digit classifier in Burn",
                encoder: "Transformer text classifier in Burn", resnet18: "ResNet-18 in Burn"}[.] // .;
    def cell($latency): "\(.requests_per_second | commas), \(.[$latency] | ms)";
    . as $r
    | "| | **Furnace** | Hand-written axum | FastAPI (uvicorn, 4 workers) |",
      "|---|---|---|---|",
      (.models[] | .model as $m
        | (if $m == "echo" then ["median latency", "p50_ms"] else ["p95", "p95_ms"] end) as [$label, $key]
        | ($r | [server($m; "fastapi x4")][0] | if . then at_load | cell($key) else "" end) as $fastapi
        | "| \($m | name): requests/s, \($label) | **\($r | server($m; "furnace") | at_load | cell($key))** | \($r | server($m; "hand-written") | at_load | cell($key)) | \($fastapi) |"),
      "| Ready after start, returning its input | **\(server("echo"; "furnace").ready_ms | ms)** | \(server("echo"; "hand-written").ready_ms | ms) | \(server("echo"; "fastapi x4").ready_ms | ms) (\(server("echo"; "fastapi").ready_ms | ms) with 1 worker) |",
      "| Memory (peak private), returning its input | **\(server("echo"; "furnace").peak_private_mb | mb)** | \(server("echo"; "hand-written").peak_private_mb | mb) | \(server("echo"; "fastapi").peak_private_mb | mb) with 1 worker |"
  ' "$dir/burn.json"
} >"$work/server-table"

# Sentences one per line: Markdown joins them into one paragraph
jq -r "$formats"'
  def rps($m; $s): .models[] | select(.model == $m) | .servers[] | select(.server == $s) | .load[-1].requests_per_second;
  def share($m): "\(rps($m; "furnace") / rps($m; "hand-written") | percent)%";
  "Furnace adds request IDs, metrics, input checks, backpressure and the Triton protocol to the hand-written server: \($furnace | tonumber - ($hand | tonumber) | one) µs per request in process, one request at a time (\($furnace) µs against \($hand) µs).",
  "Under load it serves \(share("echo")) of the hand-written server'"'"'s requests per second with the model that returns its input and \(share("resnet18")) with ResNet-18.",
  "The release binary with ResNet-18 built in is ready \($ready | tonumber | ms) after it starts and has answered its first inference at \($first | tonumber | ms)."
' --arg furnace "$(overhead furnace)" --arg hand "$(overhead hand-written)" \
  --arg ready "$(cold ready)" --arg first "$(cold "first inference")" \
  "$dir/burn.json" >"$work/server-notes"

blocks=(server-table server-notes)
if [ -f "$dir/nerve.json" ]; then
  jq -r "$formats"'
    def server($s): .servers[] | select(.server == $s) | .load[-1];
    (server("furnace").requests_per_second / server("hand-written").requests_per_second - 1) as $d
    | "On the same run, against a hand-written server doing the same work: \(.hop.fabric_ns_per_hop | round) ns per hop, \(.arc.fabric_p50_us | one) µs from stimulus to effector call on a reflex arc, a brain fraction of \(.brain.overall | two) on a stream with half repeats, \(.webhook.added_p50_us | round) µs added to a webhook.",
      "At \(server("furnace").clients) clients the fabric server serves \(if $d < 0 then -$d else $d end | percent)% \(if $d < 0 then "fewer" else "more" end) requests per second than the bare handler\(if $d < 0 then " (the server layer around it, not the fabric)" else "" end)."
  ' "$dir/nerve.json" >"$work/nerve"
  blocks+=(nerve)
else
  echo "$0: no $dir/nerve.json, the nervous system's figures stay as they are" >&2
fi

# Replace what is between each pair of markers; a README checked out with CRLF line ends
# still matches
for name in "${blocks[@]}"; do
  start="<!-- bench:$name:start -->"
  end="<!-- bench:$name:end -->"
  for marker in "$start" "$end"; do
    count=$(tr -d '\r' <"$readme" | grep -cxF -- "$marker" || true)
    [ "$count" = 1 ] || { echo "$0: $readme needs exactly one line '$marker', has $count" >&2; exit 1; }
  done
  awk -v start="$start" -v end="$end" -v file="$work/$name" '
    { line = $0; sub(/\r$/, "", line) }
    line == start { print; print ""; while ((getline l < file) > 0) print l; close(file); print ""; skip = 1; next }
    line == end { skip = 0 }
    !skip { print }
  ' "$readme" >"$work/readme"
  cat "$work/readme" >"$readme"
done
