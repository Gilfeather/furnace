#!/usr/bin/env bash
# Markdown for the benchmarks workflow (.github/workflows/bench.yml), on standard output.
#
#   benches/ci-summary.sh runner DIR        one runner's results: its CPU, the ratios, then each
#                                           harness's own tables; also writes DIR/ratios.json
#   benches/ci-summary.sh combine DIR...    the ratios of several runners side by side
#
# DIR holds what the workflow saved: cpu.json, burn.json and burn.md (furnace-burn-bench),
# overhead.txt, llm.json and llm.md (furnace-llm-bench), llm-overhead.txt and cold-start.txt.
set -euo pipefail

runner() {
  local dir=$1
  # `cold-start` prints "binary ready  median  12.3 ms ...": the three medians
  local cold
  cold=$(awk '{ for (i = 1; i < NF; i++) if ($i == "median") print $(i + 1) }' "$dir/cold-start.txt" | paste -sd ' ')
  jq -n --slurpfile burn "$dir/burn.json" --slurpfile llm "$dir/llm.json" --arg cold "$cold" '
    def two: . * 100 | round / 100;
    def one: . * 10 | round / 10;
    def times($a; $b): "\($a / $b | two)×";
    def burn($m; $s): $burn[0].models[] | select(.model == $m) | .servers[] | select(.server == $s);
    def llm($s): $llm[0].servers[] | select(.server == $s);
    def fastapi($s): if $s == "fastapi" then "FastAPI, 1 worker" else "FastAPI, 4 workers" end;
    ($cold | split(" ")) as $c
    | [
        ($burn[0].models[].model as $m
          | burn($m; "furnace").load[-1] as $f | burn($m; "hand-written").load[-1] as $h
          | {measure: "\($m), \($f.clients) clients: requests/s, Furnace / hand-written axum",
             value: times($f.requests_per_second; $h.requests_per_second)}),
        (burn("echo"; "furnace").load[-1] as $f | ("fastapi", "fastapi x4") as $s
          | burn("echo"; $s).load[-1] as $o
          | {measure: "echo, \($f.clients) clients: requests/s, Furnace / \(fastapi($s))",
             value: times($f.requests_per_second; $o.requests_per_second)}),
        ([llm("furnace").load, llm("fastapi").load] | transpose[]
          | {measure: "LLM stream, \(.[0].clients) client\(if .[0].clients == 1 then "" else "s" end): time to first token p50, FastAPI / Furnace",
             value: times(.[1].ttft_p50_ms; .[0].ttft_p50_ms)}),
        {measure: "echo server ready after start: Furnace / hand-written axum / FastAPI, 1 worker",
         value: "\(burn("echo"; "furnace").ready_ms | round) / \(burn("echo"; "hand-written").ready_ms | round) / \(burn("echo"; "fastapi").ready_ms | round) ms"},
        {measure: "furnace binary, ResNet-18 cold start: listening / ready / first inference",
         value: "\($c[0]) / \($c[1]) / \($c[2]) ms"},
        {measure: "echo server memory after the load (RssAnon): Furnace / FastAPI, 1 worker",
         value: "\(burn("echo"; "furnace").peak_private_mb | one) / \(burn("echo"; "fastapi").peak_private_mb | one) MB"},
        {measure: "LLM server memory after the load (RssAnon): Furnace / FastAPI",
         value: "\(llm("furnace").peak_private_mb | one) / \(llm("fastapi").peak_private_mb | one) MB"}
      ]' >"$dir/ratios.json"

  local cpu runs
  cpu=$(jq -r '"\(.model), \(.cores) cores (\(.arch))"' "$dir/cpu.json")
  runs=$(jq -r .runs "$dir/burn.json")
  echo "## $cpu"
  echo
  echo "Medians of $runs runs. Shared runners are noisy: the ratios matter, not the absolute numbers."
  echo
  echo "| Measure | Value |"
  echo "|---|---|"
  jq -r '.[] | "| \(.measure) | \(.value) |"' "$dir/ratios.json"
  echo
  echo "### Burn models: benches/burn"
  echo
  cat "$dir/burn.md"
  echo
  echo "Each server's own work, in process, one request at a time (\`overhead echo\`):"
  echo
  echo '```'
  cat "$dir/overhead.txt"
  echo '```'
  echo
  echo "### LLM server layer: benches/llm"
  cat "$dir/llm.md"
  echo
  echo "In process, no network (\`overhead\`):"
  echo
  cat "$dir/llm-overhead.txt"
  echo
  echo "### Cold start: furnace binary, ResNet-18"
  echo
  echo '```'
  cat "$dir/cold-start.txt"
  echo '```'
}

combine() {
  local header="| Measure |" rule="|---|" dir
  for dir in "$@"; do
    header+=" $(jq -r '"\(.arch): \(.model), \(.cores) cores"' "$dir/cpu.json") |"
    rule+="---|"
  done
  echo "$header"
  echo "$rule"
  local files=()
  for dir in "$@"; do files+=("$dir/ratios.json"); done
  jq -rs 'transpose[] | "| \(.[0].measure) | \(map(.value) | join(" | ")) |"' "${files[@]}"
}

case ${1:-} in
  runner) runner "$2" ;;
  combine) shift; combine "$@" ;;
  *) echo "usage: $0 runner DIR | combine DIR..." >&2; exit 2 ;;
esac
