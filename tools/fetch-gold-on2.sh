#!/usr/bin/env bash
# Fetch NASA GOLD's level-2 ΣO/N2 day files (V04) from SPDF, for `tools/column-on2.py` (plans/17 Ф23.P).
#
# GOLD's data are NASA's, open with attribution (`docs/provenance.md`, `gold-on2`). Each day is one netCDF-4
# file of 12 scans; where SPDF holds several revisions of a day, the last in name order is the newest. A file
# is kept only if it starts with the HDF5 signature, and its sha256 goes to the manifest beside it.
#
# Usage: tools/column-on2.py days | tools/fetch-gold-on2.sh     (lines `year doy`; present files are skipped)

set -euo pipefail

DEST="${GOLD_DIR:-data/validation/gold-on2}"
BASE="https://spdf.gsfc.nasa.gov/pub/data/gold/level2/on2"
mkdir -p "$DEST/raw"
listing="$(mktemp -d)"
trap 'rm -rf "$listing"' EXIT

while read -r year doy; do
  [ -f "$listing/$year" ] || {
    curl -sS --fail --max-time 120 "$BASE/$year/" | grep -oE "gold_l2_on2_${year}_[0-9]{3}_v04_r[0-9]+_c[0-9]+\.nc" | sort -u >"$listing/$year"
    sleep 1
  }
  name="$(grep "^gold_l2_on2_${year}_${doy}_" "$listing/$year" | tail -1 || true)"
  if [ -z "$name" ]; then
    echo "no file for $year $doy" >&2
    continue
  fi
  [ -f "$DEST/raw/$name" ] && continue
  curl -sS --fail --max-time 300 -o "$DEST/raw/$name.part" "$BASE/$year/$name"
  if [ "$(head -c 4 "$DEST/raw/$name.part" | od -An -tx1 | tr -d ' ')" != "89484446" ]; then
    echo "not HDF5: $name" >&2
    rm -f "$DEST/raw/$name.part"
    continue
  fi
  mv "$DEST/raw/$name.part" "$DEST/raw/$name"
  printf '%s\t%s\n' "$name" "$(sha256sum "$DEST/raw/$name" | cut -d' ' -f1)" >>"$DEST/manifest.tsv"
  sleep 1
done
echo "$(find "$DEST/raw" -name '*.nc' | wc -l) day files in $DEST/raw"
