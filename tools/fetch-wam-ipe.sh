#!/usr/bin/env bash
# Fetch NOAA's operational WAM-IPE neutral density -- the public physics nowcast the density
# calibration layer is scored with.
#
# WHAT IS FETCHED
#
# WAM-IPE's forecast system (`wfs`) runs at 00, 06, 12 and 18 UT. Each run starts 3 h before its
# cycle time and writes a `wam_fixed_height` file every 10 minutes from cycle - 2 h 50 min to
# cycle + 48 h: neutral mass density `den(time, hlevs, lat, lon)` on 100-1000 km x 2 deg x 4 deg,
# NetCDF classic, 2 982 952 bytes. The part before the cycle time is driven by observed inputs and
# the part after it is a forecast, so the same instant appears in up to nine runs at nine leads.
#
# One file per hour is selected, from the newest run that covers the hour: the cycle
# `floor6(t + 2 h 50 min)`. That gives leads from -2 h to +3 h -- what an operator holds at the
# hour, and the reason this is a nowcast rather than a forecast campaign. The archive has holes: a
# run can lack hours its window covers. Such an hour is left out rather than taken from an older
# run at a longer lead, so every selected file is a nowcast and a gap is a gap. Listed 2026-10-24,
# 2024-05 has 740 of its 744 hours and 2025-06 has 651 of 720, before the stubs `--pin` drops.
#
# TERMS
#
# The AWS Open Data registry page (read 2026-10-24): "NOAA data disseminated through NODD are open
# to the public and can be used as desired"; attribution is requested for unaltered data and no
# endorsement may be implied. Nothing here is committed but the list of digests. The bucket's
# NON-OPERATIONAL_RESEARCH/ reruns (hourly 2008-2023, observed solar wind for 2024-04/05) ask users
# to contact the WAM-IPE team before access, so they are not fetched.
#
# DIGESTS
#
# `tools/wam-ipe.sha256` is the selection and its pin in one file, in `sha256sum` format relative to
# DEST: the default mode fetches exactly those keys and checks them, so a file NOAA reprocesses in
# place stops the script instead of reaching a test. Making that file is two steps, so that a slow
# link can resume the second: `--select` applies the rule above to a fresh bucket listing and prints
# the keys, and `--pin` fetches a list of keys and prints their digests.
#
# Usage: tools/fetch-wam-ipe.sh [DEST]
#        tools/fetch-wam-ipe.sh --select YYYY-MM ... > keys
#        tools/fetch-wam-ipe.sh --pin keys > tools/wam-ipe.sha256

set -euo pipefail

BUCKET="https://noaa-nws-wam-ipe-pds.s3.amazonaws.com"
HERE="$(cd "$(dirname "$0")" && pwd)"
PINNED="$HERE/wam-ipe.sha256"
# Four transfers at once. Measured 2026-10-24 the link to the bucket gave one transfer anything
# from 0.05 to 0.75 MB/s, and eight together no more than four.
PARALLEL=4
# A key from a listing becomes a path, so it is held to the one shape the product has.
KEY_RE='^v1\.2/wfs\.[0-9]{8}/(00|06|12|18)/wam_fixed_height\.wfs\.t(00|06|12|18)z\.wam10\.[0-9]{8}_[0-9]{6}\.nc$'
export BUCKET KEY_RE

fetch() {
  local key="$1" dest="$2"
  if [[ ! "$key" =~ $KEY_RE ]]; then
    echo "refusing key '$key'" >&2
    return 1
  fi
  [[ -s "$dest/$key" ]] && return 0
  mkdir -p "$dest/$(dirname "$key")"
  local tmp
  tmp="$(mktemp "$dest/.part.XXXXXX")"
  if ! curl -sS --fail --max-time 300 -o "$tmp" "$BUCKET/$key"; then
    rm -f "$tmp"
    echo "could not fetch $key" >&2
    return 1
  fi
  mv "$tmp" "$dest/$key"
}
export -f fetch

fetch_all() {
  local dest="$1"
  # shellcheck disable=SC2016 # the positional parameters belong to the inner shell
  xargs -r -P "$PARALLEL" -I{} bash -c 'fetch "$1" "$2"' _ {} "$dest"
}

# The hours of a month, each as the key of the newest run that covers it, where that run has it.
select_month() {
  local month="$1" listing="$2"
  local start end t c day cycle valid key
  start="$(date -u -d "${month}-01" +%s)"
  end="$(date -u -d "${month}-01 +1 month" +%s)"
  for ((t = start; t < end; t += 3600)); do
    c=$(((t + 10200) / 21600 * 21600))
    day="$(date -u -d "@$c" +%Y%m%d)"
    cycle="$(date -u -d "@$c" +%H)"
    valid="$(date -u -d "@$t" +%Y%m%d_%H%M%S)"
    if [[ ! -f "$listing/$day.$cycle" ]]; then
      curl -sS --fail --max-time 120 \
        "$BUCKET/?list-type=2&max-keys=1000&prefix=v1.2/wfs.${day}/${cycle}/wam_fixed_height" |
        grep -o '<Key>[^<]*</Key>' | sed 's/<Key>//; s/<\/Key>//' >"$listing/$day.$cycle" || true
    fi
    key="v1.2/wfs.${day}/${cycle}/wam_fixed_height.wfs.t${cycle}z.wam10.${valid}.nc"
    if grep -qxF "$key" "$listing/$day.$cycle"; then
      echo "$key"
    else
      echo "no nowcast for $valid in run ${day} ${cycle}z" >&2
    fi
  done
}

if [[ "${1:-}" == "--select" ]]; then
  shift
  listing="$(mktemp -d)"
  trap 'rm -rf "$listing"' EXIT
  for m in "$@"; do
    if [[ ! "$m" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]; then
      echo "month must be YYYY-MM with MM in 01..12, got '$m'" >&2
      exit 2
    fi
    select_month "$m" "$listing"
  done
  exit 0
fi

if [[ "${1:-}" == "--pin" ]]; then
  keys="$(realpath "$2")"
  dest="${WAM_IPE_DIR:-$HOME/.cache/spacelab/wam-ipe}"
  mkdir -p "$dest"
  fetch_all "$dest" <"$keys"
  # The bucket holds the odd stub in a product's name -- a 239-byte HDF5 header where 2024-05-01
  # 04:00 should be -- so a file is pinned only if it is NetCDF classic, and a stub is a gap.
  while read -r key; do
    if [[ "$(head -c 4 "$dest/$key" | od -An -tx1 | tr -d ' ')" == "43444601" ]]; then
      (cd "$dest" && sha256sum "$key")
    else
      echo "not NetCDF classic, left out: $key ($(stat -c%s "$dest/$key") bytes)" >&2
    fi
  done <"$keys"
  exit 0
fi

dest="${1:-${WAM_IPE_DIR:-$HOME/.cache/spacelab/wam-ipe}}"
mkdir -p "$dest"
cut -d' ' -f3 "$PINNED" | fetch_all "$dest"
(cd "$dest" && sha256sum --quiet -c "$PINNED")
echo "$(wc -l <"$PINNED") WAM-IPE files -> $dest, digests match $PINNED"
