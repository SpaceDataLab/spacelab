#!/usr/bin/env bash
# Re-measure the two crates whose survivor lists this session's tests and fixes made stale.
#
# `mutants-pkg` snapshots the working tree at the moment it is invoked, so this waits for the
# running chain to drain *before* invoking it -- launching it early would snapshot a tree that
# predates the work it is meant to score. The batch lock inside `_capped-batch` would serialise it
# either way; what the lock cannot do is un-snapshot.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
JUST=${JUST:-$HOME/.cargo/bin/just}
LOG=$HOME/.cache/spacelab-mutants/meta/queue.log

while pgrep -x cargo-mutants >/dev/null; do sleep 300; done

run() {
    local name=$1 pkg=$2
    local out=$HOME/.cache/spacelab-mutants/$name/mutants.out
    "$JUST" wait-clean || { echo "$(date -Is) $name: tree not clean, stopping" >> "$LOG"; exit 1; }
    echo "$(date -Is) $name: start" >> "$LOG"
    "$JUST" mutants-pkg "$name" "$pkg"
    echo "$(date -Is) $name: caught=$(grep -c . "$out/caught.txt" 2>/dev/null || echo ?) \
missed=$(grep -c . "$out/missed.txt" 2>/dev/null || echo ?) \
timeout=$(grep -c . "$out/timeout.txt" 2>/dev/null || echo 0)" >> "$LOG"
}

run estim-rescan2 spacelab-estimation
run fa-rescan2 spacelab-fa
echo "$(date -Is) queue10 finished" >> "$LOG"
