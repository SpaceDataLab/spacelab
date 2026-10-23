#!/usr/bin/env bash
# Fetch an IGS final precise-orbit product (SP3) for one day, as validation truth.
#
# The final combination is the IGS product an orbit fit is measured against: a weighted
# average of ten analysis centres, in the IGS20 realization of ITRF2020, quoted at the
# satellite centre of mass on a 15-minute grid. Published ~13 days in arrears and never
# revised, so unlike the EOP series it does not rot — but it is 250 kB a day per file and
# there are 30 years of them, so it is fetched, never vendored. A single satellite's
# 24-hour excerpt is committed under crates/spacelab/tests/fixtures for the always-run test.
#
# What is recorded is the snapshot: where it came from, when, what it spans, and its digest.
#
# The rapid combination (RAP) is the same product ~17 hours after the fact instead of ~13 days,
# accurate to ~2.5 cm against the final's ~2.5 cm — close enough that the choice is about latency,
# not accuracy. It is selectable because a comparison against a current element set cannot wait two
# weeks; the snapshot's *name* carries which one it is, so the two can never be confused after the
# fact.
#
# Usage: tools/fetch-validation-data.sh [YYYY-MM-DD] [FIN|RAP|ULT]   (default 2025-01-01 FIN)

set -euo pipefail

date_arg="${1:-2025-01-01}"
kind="${2:-FIN}"
case "$kind" in
FIN | RAP | ULT) ;;
*)
  echo "product must be FIN, RAP or ULT, got '$kind'" >&2
  exit 2
  ;;
esac
# The date reaches a URL, so it is validated here rather than at the consumer: an
# allow-list of exactly ten digits and two dashes, then a calendar check by `date`.
if [[ ! "$date_arg" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "date must be YYYY-MM-DD, got '$date_arg'" >&2
  exit 2
fi
if ! epoch="$(date -u -d "$date_arg" +%s 2>/dev/null)"; then
  echo "not a calendar date: $date_arg" >&2
  exit 2
fi

year="$(date -u -d "$date_arg" +%Y)"
doy="$(date -u -d "$date_arg" +%j)"
# GPS week zero began 1980-01-06 00:00 UTC = 315964800 POSIX. Leap seconds are ignored:
# they shift the boundary by 18 s in 604800, and no IGS product day starts inside that.
gps_week=$(( (epoch - 315964800) / 604800 ))

DEST="${VALIDATION_DIR:-data/validation}"
name="IGS0OPS${kind}_${year}${doy}0000_01D_15M_ORB.SP3"
URL="${SP3_URL:-https://igs.bkg.bund.de/root_ftp/IGS/products/${gps_week}/${name}.gz}"

mkdir -p "$DEST"
tmp="$(mktemp)"
trap 'rm -f "$tmp" "$tmp.gz"' EXIT

echo "fetching $URL"
curl -fsSL --retry 3 --retry-delay 2 -o "$tmp.gz" "$URL"
gunzip -c "$tmp.gz" > "$tmp"

digest="$(sha256sum "$tmp" | cut -d' ' -f1)"
snapshot="$DEST/$name"

if [[ -e "$snapshot" ]]; then
  echo "snapshot already held: $snapshot"
  exit 0
fi

cp "$tmp" "$snapshot"

epochs="$(grep -c '^\*' "$snapshot")"
sats="$(awk '/^\+/ {print substr($0,4,3); exit}' "$snapshot" | tr -d ' ')"
frame="$(awk 'NR==1 {print substr($0,47,5)}' "$snapshot")"
interval="$(awk 'NR==2 {print $4}' "$snapshot")"

cat > "$DEST/${name}.json" <<JSON
{
  "url": "$URL",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$digest",
  "bytes": $(stat -c%s "$snapshot"),
  "date": "$date_arg",
  "gps_week": $gps_week,
  "epochs": $epochs,
  "satellites": $sats,
  "frame": "$frame",
  "interval_seconds": $interval,
  "format": "SP3-c, positions at the satellite centre of mass, epochs in GPS time"
}
JSON

echo "snapshot   $snapshot"
echo "manifest   $DEST/${name}.json"
echo "sha256     $digest"
echo "span       $epochs epochs of $sats satellites at ${interval}s, frame $frame"
