#!/usr/bin/env bash
# Build the HWM14 oracle and run it over a request file.
#
# Two things are built: NRL's unmodified hwm14.f90, fetched by tools/fetch-hwm14.sh into gitignored
# space, and our own hwm14_oracle.f90 driver beside it. Nothing here enters the Cargo workspace.
#
# HWM14 opens its three coefficient files by bare name, so the run happens with the data directory
# as the working directory; the request and response paths are resolved before that cd, so relative
# paths on the command line mean what the caller meant.
#
# Usage: tools/hwm14-oracle/run.sh REQUESTS RESPONSES
#
# REQUESTS holds one line per point: iyd sec alt_km glat_deg glon_deg stl_h f107a f107 ap3h
# RESPONSES gets one line per point: meridional_m_s zonal_m_s

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REQUESTS RESPONSES" >&2
  exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${HWM14_DIR:-$(cd "$here/../.." && pwd)/data/oracles/hwm14}"
requests="$(readlink -f "$1")"
responses="$(readlink -f "$(dirname "$2")")/$(basename "$2")"

if [[ ! -f "$data/hwm14.f90" ]]; then
  echo "no HWM14 source in $data -- run tools/fetch-hwm14.sh first" >&2
  exit 1
fi

build="$data/build"
mkdir -p "$build"
# -std=legacy: hwm14.f90 is NRL's, unmodified, and warns under a strict front end. Silencing the
# warnings by editing it would be the derivative work the licence forbids.
if [[ ! -x "$build/hwm14_oracle" ]] ||
  [[ "$data/hwm14.f90" -nt "$build/hwm14_oracle" ]] ||
  [[ "$here/hwm14_oracle.f90" -nt "$build/hwm14_oracle" ]]; then
  (cd "$build" && gfortran -O2 -std=legacy -w -o hwm14_oracle "$data/hwm14.f90" "$here/hwm14_oracle.f90")
fi

(cd "$data" && "$build/hwm14_oracle" <"$requests" >"$responses")
echo "$(wc -l <"$responses") responses -> $responses"
