#!/usr/bin/env bash
# Fetch measured thermospheric densities and the space-weather indices that drive the models.
#
# Two products, both open and keyless:
#
#   TU Delft thermosphere data (Siemes et al., J. Space Weather Space Clim. 2023, CC BY 4.0) —
#   neutral mass density along the orbit of GRACE-FO C (from the accelerometer) and Swarm A (from
#   precise orbit determination), one ASCII file per satellite-month, 10 s and 30 s cadence.
#   This is the measurement NRLMSISE-00 / JB2008 / DTM2020 are wrong about; it is ~30 MB a month,
#   so it is fetched, never vendored. A one-day CHAMP excerpt lives under
#   crates/spacelab-propagation/tests/fixtures for the always-run parser test.
#
#   The drivers: CelesTrak's SW-All.csv (F10.7 and the eight 3-hourly ap of every day since 1957)
#   and Space Environment Technologies' SOLFSMY.TXT / DTCFILE.TXT, which carry the S10/M10/Y10
#   EUV indices and the Dst-derived exospheric temperature change JB2008 needs and nothing else
#   publishes. A model evaluated on the wrong day's flux is measuring the driver, not the model.
#
#   And GFZ's Hpo family (CC BY 4.0), the same construction as Kp at 30- and 60-minute cadence and
#   uncapped above Kp 9 — 66 MB for the two complete series, which is why they are fetched rather
#   than vendored; a Gannon-storm week of each lives under crates/spacelab-propagation/tests/
#   fixtures for the always-run comparison. No model here is specified on them yet, and
#   `hourly_geomagnetic_index.rs` is what measures whether one should be.
#
# The months default to a spread across solar cycles 24 and 25 rather than one window, because the
# thing under test is whether the error is flat in solar activity — it is not, and an aggregate
# over one month cannot say so. 2024-05 carries the Gannon superstorm (Kp 9, Dst -412 nT).
#
# What is recorded is the snapshot: where each file came from, when, what it spans, and its digest.
#
# Usage: tools/fetch-density-data.sh [YYYY-MM ...]

set -euo pipefail

DEFAULT_MONTHS=(2019-01 2021-11 2022-04 2023-04 2024-05 2025-06)
months=("$@")
if [[ ${#months[@]} -eq 0 ]]; then
  months=("${DEFAULT_MONTHS[@]}")
fi

# Each month reaches a URL, so it is validated here rather than at the consumer: four digits, a
# dash, and a month in 01..12. Anything else never becomes a request.
for m in "${months[@]}"; do
  if [[ ! "$m" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]; then
    echo "month must be YYYY-MM with MM in 01..12, got '$m'" >&2
    exit 2
  fi
done

DEST="${DENSITY_DIR:-data/validation/density}"
BASE="${TUDELFT_URL:-https://thermosphere.tudelft.nl/data/data/version_02}"
mkdir -p "$DEST"

# The mission directory, the file stem, and the density-derivation method, one record per
# satellite. GRACE-FO C flies an accelerometer; Swarm A has none, so its density comes from the
# drag signature in the precise orbit — a factor the two products differ by in resolution, and the
# reason both are here.
satellites=(
  "GRACE-FO_data:GC_DNS_ACC:accelerometer"
  "Swarm_data:SA_DNS_POD:precise-orbit-determination"
)

# The reprocessing suffix is not uniform across missions or months, so it is discovered rather
# than assumed: newest first, first hit wins.
suffixes=(v02c v02b v02)

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

for m in "${months[@]}"; do
  ym="${m/-/_}"
  for sat in "${satellites[@]}"; do
    IFS=: read -r dir stem method <<<"$sat"
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
      echo "no $stem product for $m at any known reprocessing suffix" >&2
      continue
    fi
    unzip -o -q -d "$DEST" "$DEST/${got}.zip"
    rm -f "$DEST/${got}.zip"
    txt="$DEST/${got}.txt"
    rows="$(grep -vc '^#' "$txt" || true)"
    manifest "$txt" "${BASE}/${dir}/${got}.zip" ",
  \"month\": \"$m\",
  \"satellite\": \"${stem%%_*}\",
  \"density_from\": \"$method\",
  \"rows\": $rows,
  \"license\": \"CC BY 4.0, Delft University of Technology\""
    echo "$got  $rows rows"
  done
done

# The drivers are a single current snapshot, not per-month: both series are appended to in place
# and cover the whole span, so re-fetching them is how they stay in step with the data above.
drivers=(
  "SW-All.csv|https://celestrak.org/SpaceData/SW-All.csv|F10.7 and 3-hourly ap, CelesTrak"
  "SOLFSMY.TXT|https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT|F10/S10/M10/Y10 solar indices, Space Environment Technologies"
  "DTCFILE.TXT|https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT|hourly Dst-derived exospheric temperature change, Space Environment Technologies"
  "Hp30_ap30_complete_series.txt|https://kp.gfz.de/app/files/Hp30_ap30_complete_series.txt|Hp30/ap30 since 1985, GFZ Helmholtz Centre for Geosciences"
  "Hp60_ap60_complete_series.txt|https://kp.gfz.de/app/files/Hp60_ap60_complete_series.txt|Hp60/ap60 since 1985, GFZ Helmholtz Centre for Geosciences"
)
for d in "${drivers[@]}"; do
  IFS='|' read -r name url what <<<"$d"
  if fetch "$url" "$DEST/$name"; then
    manifest "$DEST/$name" "$url" ",
  \"contents\": \"$what\""
    echo "$name  $(stat -c%s "$DEST/$name") bytes"
  else
    echo "could not fetch $name from $url" >&2
  fi
done
