#!/usr/bin/env bash
# Fetch a snapshot of the IERS finals2000A series.
#
# The file is 3.7 MB and is rewritten daily as the IERS observes and re-predicts.
# It is therefore fetched, never vendored: git would accumulate a megabyte a day of
# noise, and a committed copy would silently rot.
#
# What is recorded instead is the snapshot: where it came from, when, how big, what
# it spans, and its digest. Backward-mode reconstruction must be able to reproduce
# what was known *then*, so an EOP file is never overwritten in place.

set -euo pipefail

URL="${EOP_URL:-https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt}"
DEST="${EOP_DIR:-data/eop}"

mkdir -p "$DEST"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

echo "fetching $URL"
curl -fsSL --retry 3 --retry-delay 2 -o "$tmp" "$URL"

digest="$(sha256sum "$tmp" | cut -d' ' -f1)"
short="${digest:0:8}"
snapshot="$DEST/finals2000A.$short.txt"

if [[ -e "$snapshot" ]]; then
  echo "snapshot already held: $snapshot"
  exit 0
fi

cp "$tmp" "$snapshot"

# The tail of the file carries reserved-but-empty days; the last *usable* MJD is
# the one that matters for the tabulated span.
first_mjd="$(awk 'NR==1 {print substr($0,8,8)}' "$snapshot" | tr -d ' ')"
last_mjd="$(awk 'substr($0,17,1) ~ /[IP]/ {m=substr($0,8,8)} END {print m}' "$snapshot" | tr -d ' ')"
lines="$(wc -l < "$snapshot")"

cat > "$DEST/finals2000A.$short.json" <<JSON
{
  "source_url": "$URL",
  "fetched_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$digest",
  "bytes": $(stat -c%s "$snapshot"),
  "lines": $lines,
  "first_mjd": $first_mjd,
  "last_usable_mjd": $last_mjd,
  "format": "finals2000A.all, fixed-width, 187 columns"
}
JSON

echo "snapshot   $snapshot"
echo "manifest   $DEST/finals2000A.$short.json"
echo "sha256     $digest"
echo "span       MJD $first_mjd .. $last_mjd  ($lines lines)"
