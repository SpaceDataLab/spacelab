#!/usr/bin/env bash
# Fetch Celestrak Supplemental GP (SupGP) element sets, and the standard GP catalogue beside them.
#
# WHAT SUPGP IS, AND WHY IT IS AN ORACLE RATHER THAN MORE DATA
#
# SupGP element sets are contributed by the *operators* -- SpaceX, OneWeb, Iridium, Planet, the GPS
# and GLONASS control segments, the ILRS analysis centres for the laser targets -- and are fitted
# *forward* from an ephemeris the operator already knows. The standard GP catalogue is fitted
# *backward* from tracking observations. So for one object at one time there are two element sets
# built from disjoint information, and for the laser targets there are often several, from different
# analysis centres. That is what makes the comparison a measurement rather than a duplicate.
#
# It is not truth. An operator's element set is a fit to the operator's own ephemeris, and the CPF
# entries are fits to a *prediction*. Where a genuinely precise product exists for the same object
# -- IGS for GPS -- that product is the truth and both element sets are the things being measured.
#
# USAGE POLICY, which this script exists partly to encode
#
# Celestrak asks that GP and SupGP be retrieved no more often than every two hours, directory
# queries no more than hourly, and that a machine client stop on any non-200 and report to a human.
# `curl --fail` plus `set -e` is that stop. Nothing here retries a non-200 or loops.
#
# Usage: tools/fetch-supgp.sh [FILE ...]      (default: gps cpf starlink oneweb iridium)
#
# FILE is one of Celestrak's SupGP file names: ast cpf css eumetsat glonass gps intelsat iridium
# iss kuiper oneweb orbcomm planet ses starlink telesat.

set -euo pipefail

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  FILES=(gps cpf starlink oneweb iridium)
fi

DEST="${SUPGP_DIR:-data/supgp}"
BASE="${SUPGP_URL:-https://celestrak.org/NORAD/elements/supplemental/sup-gp.php}"
GP_BASE="${GP_URL:-https://celestrak.org/NORAD/elements/gp.php}"

mkdir -p "$DEST"

fetch() {
  local url="$1" out="$2" label="$3"
  # The URL is assembled from an allow-listed name, never from free text: a SupGP file name is
  # lowercase letters only, and anything else is refused before it reaches curl.
  if ! curl -sS -L --fail --max-time 300 -o "$out.part" "$url"; then
    echo "$label: fetch failed -- Celestrak returned a non-200; stopping, as its usage policy asks" >&2
    rm -f "$out.part"
    exit 1
  fi
  local records
  records="$(grep -c '^1 ' "$out.part" || true)"
  if [[ "$records" -eq 0 ]]; then
    echo "$label: 200 but no element sets in the body" >&2
    rm -f "$out.part"
    exit 1
  fi
  mv "$out.part" "$out"
  cat >"$out.json" <<EOF
{
  "url": "$url",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$(sha256sum "$out" | cut -d' ' -f1)",
  "bytes": $(stat -c%s "$out"),
  "records": $records,
  "role": "oracle input; the fixtures under crates/*/tests/fixtures are cut from this by tools/gen_supgp_fixtures.py"
}
EOF
  echo "$label: $records records -> $out"
}

for file in "${FILES[@]}"; do
  if [[ ! "$file" =~ ^[a-z]+$ ]]; then
    echo "not a SupGP file name: '$file'" >&2
    exit 2
  fi
  fetch "${BASE}?FILE=${file}&FORMAT=tle" "$DEST/sup-$file.tle" "sup-$file"
done

# The standard catalogue, for the same objects. Without it the SupGP records have nothing to be
# compared against, and fetching it here keeps the two snapshots within minutes of each other --
# which matters, because the whole comparison is between two fits of the same object at one time.
fetch "${GP_BASE}?GROUP=active&FORMAT=tle" "$DEST/gp-active.tle" "gp-active"
