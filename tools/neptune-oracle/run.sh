#!/usr/bin/env bash
# Run the built NEPTUNE over an input deck and leave its output files behind.
#
# NEPTUNE resolves its data, input and output paths relative to the working directory, and the
# vendor's start_neptune.sh both downloads fresh space-weather files and runs from the tree it was
# unpacked into. Neither is wanted here: an oracle whose inputs move between runs is not an oracle,
# and writing into the pinned snapshot would make the next fetch's digest check meaningless. So a
# run directory is assembled instead -- data symlinked from the snapshot, the deck copied in, output
# fresh -- and the snapshot is only ever read.
#
# Usage: tools/neptune-oracle/run.sh -i INPUT_DIR [-o RUN_DIR] [-d OVERLAY_DIR]
#
# INPUT_DIR holds neptune.inp and whatever it names (neptune.mnv, neptune.srf).
# RUN_DIR defaults to data/oracles/neptune-run and is recreated on every call.
# OVERLAY_DIR replaces data files of the same name -- the way an alternative gravity field is
# supplied, since NEPTUNE fixes each model's filename in its source and the source is not edited.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dir="${NEPTUNE_DIR:-$root/data/oracles/neptune}"

input=""
overlay=""
run="$root/data/oracles/neptune-run"
while getopts "i:o:d:" opt; do
  case "$opt" in
  i) input="$OPTARG" ;;
  o) run="$OPTARG" ;;
  d) overlay="$OPTARG" ;;
  *)
    echo "usage: $0 -i INPUT_DIR [-o RUN_DIR] [-d OVERLAY_DIR]" >&2
    exit 2
    ;;
  esac
done

if [[ -z "$input" ]]; then
  echo "usage: $0 -i INPUT_DIR [-o RUN_DIR] [-d OVERLAY_DIR]" >&2
  exit 2
fi
if [[ ! -f "$input/neptune.inp" ]]; then
  echo "no neptune.inp in $input" >&2
  exit 1
fi

exe="$dir/build/src/neptune-sa"
if [[ ! -x "$exe" ]]; then
  echo "no neptune-sa -- run tools/neptune-oracle/build.sh first" >&2
  exit 1
fi

rm -rf "$run"
mkdir -p "$run/data" "$run/input" "$run/output"
ln -s "$dir"/work/data/* "$run/data/"
if [[ -n "$overlay" ]]; then
  ln -sf "$(cd "$overlay" && pwd)"/* "$run/data/"
fi
cp "$input"/* "$run/input/"
ln -s "$exe" "$run/neptune-sa"

# LD_LIBRARY_PATH rather than an rpath: libslam is built in place inside the snapshot, and the
# snapshot is a fetched artefact that no install step owns. The console goes to console.log and not
# to neptune.log, which NEPTUNE's own logger opens in the working directory and refuses to share.
(cd "$run" && LD_LIBRARY_PATH="$dir/libslam/build/lib" timeout 1800 ./neptune-sa) >"$run/console.log" 2>&1 ||
  {
    tail -20 "$run/console.log" >&2
    exit 1
  }

echo "neptune -> $run/output"
