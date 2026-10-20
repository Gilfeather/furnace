#!/usr/bin/env bash
# The public datasets of the expectations experiments (docs/NERVE.md), into DIR (not the
# repository): Loghub's BGL log (labelled) and the parts of the Numenta Anomaly Benchmark the
# experiment uses.
#
#   bash benches/nerve/fetch-data.sh DIR
#
# DIR/BGL.log                       4,747,963 lines, 709 MiB (57.5 MB zipped), from Zenodo
# DIR/NAB/data/<group>/<series>.csv 58 series
# DIR/NAB/labels/combined_windows.json
# DIR/NAB/config/profiles.json
# DIR/NAB/results/{numenta,null}/   published detector outputs, to check the scorer against
set -euo pipefail

dir="${1:?usage: fetch-data.sh DIR}"
mkdir -p "$dir"
cd "$dir"

if [ ! -f BGL.log ]; then
  curl -fsSL --retry 3 -o BGL.zip "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
  unzip -o -q BGL.zip BGL.log
  rm BGL.zip
fi

nab="https://raw.githubusercontent.com/numenta/NAB/master"
mkdir -p NAB
tree=$(curl -fsSL "https://api.github.com/repos/numenta/NAB/git/trees/master?recursive=1")
paths=$(printf '%s' "$tree" | grep -o '"path": *"[^"]*"' | sed 's/"path": *"\(.*\)"/\1/' |
  grep -E '^(data/.*\.csv|labels/combined_windows\.json|config/profiles\.json|results/(numenta|null)/.*\.csv)$')
for path in $paths; do
  if [ ! -f "NAB/$path" ]; then
    mkdir -p "NAB/$(dirname "$path")"
    curl -fsSL --retry 3 -o "NAB/$path" "$nab/$path"
  fi
done
echo "BGL: $(wc -l < BGL.log) lines; NAB: $(find NAB/data -name '*.csv' | wc -l) series"
