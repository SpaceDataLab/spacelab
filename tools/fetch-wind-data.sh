#!/usr/bin/env bash
# Fetch measured thermospheric crosswinds — the observation an empirical wind model is fitted to.
#
# TU Delft thermosphere data (Siemes et al., J. Space Weather Space Clim. 2023, CC BY 4.0), the
# `*_WND_ACC_*` product: the accelerometer-derived neutral wind component perpendicular to the
# satellite velocity, one ASCII file per satellite-month, 10 s cadence. Same header, same first
# eight columns as the `*_DNS_*` density product this repo already reads; the wind product adds a
# speed and a north/east/down unit vector.
#
#   CHAMP     2000-07 .. 2010-05, i = 87.3 deg, 456 -> 300 km  (solar cycle 23)
#   GRACE-A   2002-04 .. 2017-05, i = 89.0 deg, 500 -> 330 km  (cycles 23 and 24)
#   GOCE      2009-11 .. 2013-10, i = 96.7 deg, 270 -> 225 km  (the lowest sampling there is)
#   GRACE-FO  2023-01 .. 2026-04, i = 89.0 deg, ~500 km        (entirely after HWM14's fit ended)
#
# The sampling is zonal-dominated but not zonal-only, and that is why four missions are here rather
# than one. Measured over the downloaded arcs: the crosswind direction is within 8 deg of east-west
# in 79-93% of CHAMP / GRACE / GRACE-FO rows, but the polar passes swing it right through north-south
# (the minimum |d.e| in every file is ~0), and GOCE's 96.7 deg sun-synchronous inclination puts a
# meridional projection |d.n| > 0.1 in *every* one of its rows and > 0.3 in 27% of them. So the
# meridional wind is weakly but genuinely observed, and it is observed best by GOCE.
#
# The bulk data are an external download (open, keyless) and are not vendored; a one-day excerpt
# lives under crates/spacelab-propagation/tests/fixtures for the always-run parser test.
#
# The default months span three solar cycles, both solstices and an equinox, 225 km to 500 km, and
# two of the largest storms of the record (2003-10 Halloween, 2024-05 Gannon) — a wind model fitted
# to one altitude or one activity level cannot be told from a wrong one.
#
# What is recorded is the snapshot: where each file came from, when, what it spans, and its digest.
#
# Usage: tools/fetch-wind-data.sh [MISSION:YYYY-MM ...]      (MISSION in champ|grace|goce|gracefo)

set -euo pipefail

DEFAULT_ARCS=(
  champ:2002-04 champ:2003-10 champ:2004-10 champ:2008-07
  grace:2003-01 grace:2014-07
  goce:2011-03 goce:2013-05
  gracefo:2023-04 gracefo:2024-05 gracefo:2025-06 gracefo:2025-10
)
arcs=("$@")
if [[ ${#arcs[@]} -eq 0 ]]; then
  arcs=("${DEFAULT_ARCS[@]}")
fi

DEST="${WIND_DIR:-data/validation/wind}"
BASE="${TUDELFT_URL:-https://thermosphere.tudelft.nl/data/data/version_02}"
mkdir -p "$DEST"

# The mission directory, the file stem, and how the month is spelled in the file name — CHAMP is the
# only one that separates year and month with a dash, and a script that assumes otherwise silently
# fetches nothing for the whole of solar cycle 23.
mission_spec() {
  case "$1" in
  champ) echo "CHAMP_data:CH_WND_ACC:dash" ;;
  grace) echo "GRACE_data:GA_WND_ACC:underscore" ;;
  goce) echo "GOCE_data:GO_WND_ACC:underscore" ;;
  gracefo) echo "GRACE-FO_data:GC_WND_ACC:underscore" ;;
  *) return 1 ;;
  esac
}

# Each arc reaches a URL, so it is validated here rather than at the consumer: a known mission, four
# digits, a dash, and a month in 01..12. Anything else never becomes a request.
for arc in "${arcs[@]}"; do
  mission="${arc%%:*}"
  month="${arc#*:}"
  if ! mission_spec "$mission" >/dev/null; then
    echo "unknown mission '$mission' in '$arc' (champ|grace|goce|gracefo)" >&2
    exit 2
  fi
  if [[ ! "$month" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]; then
    echo "month must be YYYY-MM with MM in 01..12, got '$month'" >&2
    exit 2
  fi
done

fetch() {
  local url="$1" out="$2"
  local tmp
  tmp="$(mktemp)"
  if ! curl -sS -L --fail --max-time 900 -o "$tmp" "$url"; then
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$out"
}

manifest() {
  local path="$1" url="$2" extra="$3"
  local sha bytes
  sha="$(sha256sum "$path" | cut -d' ' -f1)"
  bytes="$(stat -c%s "$path")"
  cat >"${path}.json" <<EOF
{
  "url": "$url",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$sha",
  "bytes": $bytes$extra
}
EOF
}

# The reprocessing suffix is not uniform across missions or months, so it is discovered rather than
# assumed: newest first, first hit wins.
suffixes=(v02c v02b v02)

for arc in "${arcs[@]}"; do
  mission="${arc%%:*}"
  month="${arc#*:}"
  IFS=: read -r dir stem sep <<<"$(mission_spec "$mission")"
  if [[ "$sep" == "underscore" ]]; then
    ym="${month/-/_}"
  else
    ym="$month"
  fi

  got=""
  for sfx in "${suffixes[@]}"; do
    name="${stem}_${ym}_${sfx}"
    url="${BASE}/${dir}/${name}.zip"
    if fetch "$url" "$DEST/${name}.zip"; then
      got="$name"
      break
    fi
  done
  if [[ -z "$got" ]]; then
    echo "no $stem product for $month at any known reprocessing suffix" >&2
    continue
  fi
  unzip -o -q -d "$DEST" "$DEST/${got}.zip"
  rm -f "$DEST/${got}.zip"
  txt="$DEST/${got}.txt"
  rows="$(grep -vc '^#' "$txt" || true)"
  manifest "$txt" "${BASE}/${dir}/${got}.zip" ",
  \"month\": \"$month\",
  \"mission\": \"$mission\",
  \"observable\": \"crosswind speed and north-east-down direction\",
  \"rows\": $rows,
  \"license\": \"CC BY 4.0, Delft University of Technology\""
  echo "$got  $rows rows"
done
