#!/usr/bin/env bash
# Decompress the derived datasets next to their .gz files (the .gz files are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
for f in data/*.gz results_v6/*.gz; do
  [ -e "$f" ] || continue
  out="${f%.gz}"
  if [ ! -e "$out" ] || [ "$f" -nt "$out" ]; then
    echo "unpacking $f"
    gunzip -c "$f" > "$out"
  fi
done
