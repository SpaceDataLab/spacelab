#!/usr/bin/env bash
# Verify the NEPTUNE build against NEPTUNE's own reference before trusting it as an oracle.
#
# NEPTUNE ships a selenocentric validation case in validation/lro: an input deck holding an LRO
# state at 2025-10-01, a seven-day span, and the NAIF reconstructed SPK covering it. That is the
# vendor's own statement of what its build should reproduce, so it is what the build is measured
# against -- not against spacelab, which is the thing this oracle exists to judge.
#
# The comparison is done with spiceypy rather than with spacelab-ephemeris for the same reason: an
# arbiter that shares code with one of the parties is not an arbiter. A compiler, an -Ofast, or a
# libslam at the wrong commit shows up here rather than as a quiet few hundred metres inside a
# comparison that is supposed to be judging *us*.
#
# Usage: tools/neptune-oracle/check.sh
#
# Prints the RMS and worst position residual against the reconstruction, and fails if the worst
# exceeds NEPTUNE_CHECK_CAP_KM (default 5 km over seven days -- the level the vendor's own
# validation script plots, not a level anyone should read as agreement).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dir="${NEPTUNE_DIR:-$root/data/oracles/neptune}"
cap="${NEPTUNE_CHECK_CAP_KM:-5.0}"

run="$dir/../neptune-check"
spk="$dir/validation/lro/work/data/lrorg_2025258_2025349_v01.bsp"

if [[ ! -f "$spk" ]]; then
  echo "no LRO reconstruction at $spk -- re-run tools/fetch-neptune.sh" >&2
  exit 1
fi

"$here/run.sh" -i "$dir/validation/lro/work/input" -o "$run"

timeout 600 uv run --with spiceypy --with numpy python "$here/check_lro.py" \
  --mcs "$run/output/neptune.mcs" \
  --spk "$spk" \
  --lsk "$dir/work/data/naif0012.tls" \
  --cap-km "$cap"
