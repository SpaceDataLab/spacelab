#!/usr/bin/env bash
# Sweep a whole GP catalogue through both SGP4 implementations and rewrite the committed fixture.
#
# Offline and out of CI, like every other oracle here: the sweep needs Node and a fetched
# satellite.js, and it takes minutes. What lands in the repository is the fixture it writes, which
# crates/spacelab-sgp4/tests/catalogue_oracle.rs then checks with nothing but the workspace.
#
# The catalogue is *not* fetched by this script. Celestrak's usage policy asks that GP data be
# retrieved no more often than every two hours, and a script that downloads 2.7 MB every time
# someone runs it is how a client ends up blocked. Fetch it once, by hand:
#
#   curl -o /tmp/gp-active.tle 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
#
# Usage: tools/sgp4-catalogue-oracle/run.sh CATALOGUE_TLE [FIRST_MIN] [LAST_MIN] [STEP_MIN]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CATALOGUE_TLE [FIRST_MIN] [LAST_MIN] [STEP_MIN]" >&2
  exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
catalogue="$(readlink -f "$1")"
first="${2:--1440}"
last="${3:-1440}"
step="${4:-5}"

oracle="${SATELLITE_JS_DIR:-$root/data/oracles/satellite-js}/satellite.js"
fixture="$root/crates/spacelab-sgp4/tests/fixtures/sgp4_catalogue_oracle.txt"

if [[ ! -f "$oracle" ]]; then
  echo "no satellite.js in $(dirname "$oracle") -- run tools/fetch-satellite-js.sh first" >&2
  exit 1
fi
if [[ ! -f "$catalogue" ]]; then
  echo "no catalogue at $catalogue" >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1; then
  echo "node is not on PATH" >&2
  exit 1
fi

# Build first, so that cargo's own progress output cannot land in the middle of the pipe and so a
# compile error is not diagnosed as a malformed oracle stream.
cargo build --release -p spacelab-sgp4 --example catalogue_sweep

node "$here/oracle.cjs" "$oracle" "$catalogue" "$first" "$last" "$step" |
  "$root/target/release/examples/catalogue_sweep" "$catalogue" "$fixture"
