#!/usr/bin/env bash
# Work the #213 blocks that have never been run, one after another, unattended.
#
# Serial on purpose: `just mutants` already saturates the machine at --jobs 4, and two concurrent
# cargo-mutants runs would contend for the same target directory. A block that fails does not stop
# the queue -- the verdict lives in each block's own missed.txt, and the point of the queue is to
# spend the machine time, not to gate on it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
JUST=${JUST:-$HOME/.cargo/bin/just}
SRC=crates/spacelab-propagation/src

run() {
    local name=$1; shift
    local out=$HOME/.cache/spacelab-mutants/$name/mutants.out
    if [ -s "$out/caught.txt" ] && [ -f "$out/missed.txt" ]; then
        echo "$(date -Is) $name: already has a result, skipping"
        return
    fi
    # `just mutants` snapshots the tree. A replayer holding a live mutation would be baked into the
    # snapshot's own baseline, and every verdict in the block would be meaningless.
    "$JUST" wait-clean || { echo "$(date -Is) $name: tree not clean, stopping the queue"; exit 1; }
    echo "$(date -Is) $name: start ($#  file(s))"
    "$JUST" mutants "$name" "$@"
    echo "$(date -Is) $name: caught=$(grep -c . "$out/caught.txt" 2>/dev/null || echo ?) \
missed=$(grep -c . "$out/missed.txt" 2>/dev/null || echo ?) \
timeout=$(grep -c . "$out/timeout.txt" 2>/dev/null || echo 0)"
}

# Phase 2 -- the never-run numerical blocks, cheapest first so a result arrives early.
run atmospheres $SRC/harris_priester/*.rs $SRC/jb2008/*.rs
run nrlmsise $SRC/nrlmsise00/*.rs
run density $SRC/density_calibration.rs $SRC/density_data.rs $SRC/density_rom.rs $SRC/space_weather.rs
run models $SRC/spacecraft.rs $SRC/drag_coefficient.rs $SRC/maneuver.rs $SRC/attitude.rs \
    $SRC/attitude_law.rs $SRC/stt.rs $SRC/variational.rs $SRC/linalg.rs $SRC/ephemeris.rs

# Class A -- machine-generated tables. One run each for the record; the justification is the
# generator's own oracle, not this count.
run tables_small $SRC/rkn1210_consts.rs $SRC/dop853_consts.rs $SRC/tide_step2_tables.rs \
    $SRC/jb2008/tables.rs
run tables_tides $SRC/ocean_tide_fes2004.rs
run tables_nrlmsise $SRC/nrlmsise00/tables.rs
run tables_geopotential $SRC/geopotential_coeffs.rs

# The two confirmation legs: those files were triaged from runs that predate the tests which
# closed them, so only a re-run says whether the closure held.
run confirm_forces $SRC/shadow.rs $SRC/relativity.rs $SRC/time_variable_gravity.rs \
    $SRC/solid_tide.rs $SRC/ocean_tide.rs $SRC/earth_radiation.rs $SRC/geopotential.rs
run confirm_integrators2 $SRC/ks.rs $SRC/gbs.rs $SRC/attitude.rs
run confirm_dsst $SRC/dsst/*.rs

echo "$(date -Is) queue done"
