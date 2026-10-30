#!/usr/bin/env bash
# Fetch NASA/GSFC's hourly OMNI2 solar wind, the IMF and plasma columns only.
#
# OMNI is the solar wind and the interplanetary magnetic field at the bow-shock nose, merged from
# ACE, Wind, IMP 8 and others and time-shifted from L1 (King & Papitashvili 2005,
# doi:10.1029/2004JA010649). Its SPASE record, spase://NASA/NumericalData/OMNI/PT1H
# (doi:10.48322/1shr-ht18), declares CC0-1.0 for these files. The same files carry WDC Kyoto's Dst,
# AE, AL and AU, which are closed to commercial applications (`docs/provenance.md`, `wdc-kyoto`),
# and other providers' indices: only the IMF and plasma columns are written, so none of them
# reaches the cache.
#
# Each hourly value is the average from its hour to the next (OMNIWeb's `ow_data.html`, word 3).
# The yearly files are fetched into a temporary directory, checked for 55 words a record and an
# unbroken hourly grid, and deleted once the columns are out.
#
# Usage: tools/fetch-omni.sh [FIRST_YEAR [LAST_YEAR]]    (default 1999 to the current year)

set -euo pipefail

DEST="${DENSITY_DIR:-data/validation/density}"
BASE="https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
first="${1:-1999}"
last="${2:-$(date -u +%Y)}"
out="$DEST/omni2_sw_hourly.txt"
mkdir -p "$DEST"
raw="$(mktemp -d)"
trap 'rm -rf "$raw" "$out.tmp"' EXIT

for ((y = first; y <= last; y++)); do
  curl -sS -L --fail --max-time 300 -o "$raw/omni2_$y.dat" "$BASE/omni2_$y.dat"
  sleep 1
done

python3 -I - "$raw" "$first" "$last" "$BASE" >"$out.tmp" <<'PY'
import datetime as dt, hashlib, pathlib, sys

raw, first, last, base = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
# (word, fill as the format writes it); words are 1-based, as omni2.text numbers them.
COLUMNS = (
    (5, "99"),          # IMF spacecraft ID
    (6, "99"),          # plasma spacecraft ID
    (9, "999.9"),       # |B|, the field magnitude average, nT
    (13, "999.9"),      # Bx, GSE = GSM, nT
    (16, "999.9"),      # By GSM, nT
    (17, "999.9"),      # Bz GSM, nT
    (23, "9999999."),   # proton temperature, K
    (24, "999.9"),      # proton density, cm^-3
    (25, "9999."),      # flow speed, km/s
    (29, "99.99"),      # flow pressure, nPa
)
print("# NASA/GSFC OMNI2 hourly solar wind at the bow-shock nose: the IMF and plasma columns only")
print("# LICENSE: CC0-1.0, which the SPASE record spase://NASA/NumericalData/OMNI/PT1H")
print("#   (doi:10.48322/1shr-ht18) declares for these files. WDC Kyoto's Dst, AE, AL, AU and every")
print("#   other index column of the source are dropped.")
print("# ACKNOWLEDGEMENT, as OMNIWeb asks: The OMNI data were obtained from the GSFC/SPDF OMNIWeb")
print("#   interface at omniweb.gsfc.nasa.gov.")
print("# CITATION: King, J. H., & Papitashvili, N. E. (2005). Solar wind spatial scales in and")
print("#   comparisons of hourly Wind and ACE plasma and magnetic field data. JGR 110, A02209,")
print("#   doi:10.1029/2004JA010649")
print(f"# SOURCE: {base}/omni2_YYYY.dat for {first}-{last}")
print(f"# RETRIEVED: {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
for y in range(first, last + 1):
    digest = hashlib.sha256((raw / f"omni2_{y}.dat").read_bytes()).hexdigest()
    print(f"# RAW omni2_{y}.dat sha256 {digest}")
print("# Each value is the mean over the hour that starts at its stamp. Missing values are null.")
print("#hour (UTC)    imf  sw    B     Bx     By     Bz         T      n      V      P")
expected = dt.datetime(first, 1, 1)
for y in range(first, last + 1):
    for number, line in enumerate((raw / f"omni2_{y}.dat").read_text().splitlines(), 1):
        words = line.split()
        if len(words) != 55:
            sys.exit(f"omni2_{y}.dat:{number}: {len(words)} words, not 55")
        hour = dt.datetime(int(words[0]), 1, 1) + dt.timedelta(days=int(words[1]) - 1, hours=int(words[2]))
        if hour != expected:
            sys.exit(f"omni2_{y}.dat:{number}: hour {hour:%Y-%m-%dT%H}, expected {expected:%Y-%m-%dT%H}")
        expected += dt.timedelta(hours=1)
        cells = ["null" if words[w - 1] == fill else words[w - 1] for w, fill in COLUMNS]
        print(f"{hour:%Y-%m-%dT%H}  " + " ".join(f"{c:>6}" for c in cells))
if expected != dt.datetime(last + 1, 1, 1):
    sys.exit(f"the files end at {expected:%Y-%m-%dT%H}, not at the end of {last}")
PY
mv "$out.tmp" "$out"
cat >"$out.json" <<EOF
{
  "url": "$BASE/omni2_YYYY.dat, $first-$last",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$(sha256sum "$out" | cut -d' ' -f1)",
  "bytes": $(stat -c%s "$out"),
  "contents": "OMNI2 hourly IMF and plasma columns at the bow-shock nose, NASA/GSFC SPDF",
  "license": "CC0-1.0 (SPASE spase://NASA/NumericalData/OMNI/PT1H)"
}
EOF
echo "$out  $(grep -vc '^#' "$out") hours, $first-$last"
