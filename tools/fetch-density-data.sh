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
#   CelesTrak's file is asked for under its usage policy (fetch_sw_all below, tools/celestrak.sh):
#   at most once per three-hour update, conditionally, and a 301 or a 403 stops the script with
#   CelesTrak's own message.
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
#        tools/fetch-density-data.sh --sw-all      (CelesTrak's SW-All.csv alone)

set -euo pipefail

# shellcheck source-path=SCRIPTDIR source=celestrak.sh
source "$(dirname "${BASH_SOURCE[0]}")/celestrak.sh"

SW_ALL_ONLY=0
if [[ "${1:-}" == --sw-all ]]; then
  if [[ $# -ne 1 ]]; then
    echo "--sw-all takes no months" >&2
    exit 2
  fi
  SW_ALL_ONLY=1
  shift
fi

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
SW_ALL_URL="${SW_ALL_URL:-https://celestrak.org/SpaceData/SW-All.csv}"
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

# CelesTrak's usage policy (https://celestrak.org/usage-policy.php, read 2026-11-24): "Space
# weather data updates every 3 hours", and "only download data once per update". So SW-All.csv is
# not requested within three hours of CelesTrak's last answer, and after that the request carries
# the copy's ETag and Last-Modified (kept in SW-All.csv.http) as If-None-Match and
# If-Modified-Since: an unchanged file answers 304 and the copy is kept, not downloaded again.
SW_UPDATE_SECONDS=10800

fetch_sw_all() {
  local url="$1" out="$2" what="$3"
  local validators=()
  if [[ -f "$out" && -f "$out.json" && -f "$out.http" ]] && grep -qF "\"url\": \"$url\"" "$out.json"; then
    local checked
    checked="$(sed -n 's/^  "checked_utc": "\(.*\)",$/\1/p' "$out.json")"
    if [[ -n "$checked" ]] && (($(date -u +%s) - $(date -u -d "$checked" +%s) < SW_UPDATE_SECONDS)); then
      echo "SW-All.csv  CelesTrak answered at $checked, inside one three-hour update -- not requested"
      return 0
    fi
    local etag modified
    etag="$(celestrak_header ETag "$out.http")"
    modified="$(celestrak_header Last-Modified "$out.http")"
    if [[ -n "$etag" ]]; then
      validators+=("If-None-Match: $etag")
    fi
    if [[ -n "$modified" ]]; then
      validators+=("If-Modified-Since: $modified")
    fi
  fi
  celestrak_get "$url" "$out" SW-All.csv "${validators[@]}"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$CELESTRAK_STATUS" == 304 ]]; then
    rm -f "$out.part" "$out.http.part"
    sed -i "s/^  \"checked_utc\": \".*\",\$/  \"checked_utc\": \"$now\",/" "$out.json"
    echo "SW-All.csv  not modified (304) -- the copy is kept"
    return 0
  fi
  if ! head -n 1 "$out.part" | grep -q '^DATE,'; then
    echo "SW-All.csv: CelesTrak answered 200, but the body is not its space-weather CSV; stopping" >&2
    rm -f "$out.part" "$out.http.part"
    exit 1
  fi
  mv "$out.http.part" "$out.http"
  mv "$out.part" "$out"
  manifest "$out" "$url" ",
  \"checked_utc\": \"$now\",
  \"contents\": \"$what\""
  echo "SW-All.csv  $(stat -c%s "$out") bytes"
}

if ((SW_ALL_ONLY)); then
  fetch_sw_all "$SW_ALL_URL" "$DEST/SW-All.csv" "F10.7 and 3-hourly ap, CelesTrak"
  exit 0
fi

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
# CelesTrak's is re-fetched by its policy, above.
fetch_sw_all "$SW_ALL_URL" "$DEST/SW-All.csv" "F10.7 and 3-hourly ap, CelesTrak"
drivers=(
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
