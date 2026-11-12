#!/usr/bin/env bash
# Fetch the primary-source files the generated coefficient tables are transcribed from.
#
# `derivations/table_seals.py` seals each table against a second parse of the Rust it emitted, which
# catches a hand edit and nothing else. `derivations/spec_seals.py` closes the loop by re-deriving
# the table from the file the issuing agency publishes — and needs that file. These are the files.
#
# They are fetched, never vendored: XGM2019e is 1.1 GB in full, FES2014b is 10.8 MB, and neither is
# ours to redistribute. What is committed instead is the *provenance* — URL, digest, size, date —
# in a sidecar next to each snapshot, so a later run can say whether it is looking at the same file.
#
# Snapshots are digest-named and never overwritten in place: a model that is revised gets a second
# snapshot alongside the first, and `spec_seals.py` then refuses to guess which one to read.

set -euo pipefail

DEST="${SPEC_DIR:-data/specs}"
mkdir -p "$DEST"

# name|url|range|note
#
# XGM2019e is fetched as a byte-range prefix. The full model runs to degree 5540 and 1.1 GB; the
# embedded table stops at degree 120, and 4 MiB of the file reaches degree 267 — so the prefix
# covers everything the table can contain, with room for the truncation degree to be raised.
SPECS=(
  "xgm2019e|https://icgem.gfz-potsdam.de/getmodel/gfc/8e01a2aba4f291ac0638b79a390155713b70fdc9b266981023c756839c11c9b8/XGM2019e.gfc|0-4194303|gfc|ICGEM XGM2019e, first 4 MiB (through degree 267)"
  "fes2014b|https://api.sedoo.fr/formater-catalogue-prod/grace/direct/fes2014b.v1.Cnm-Snm_Om1+Om2C20_without_S1_wave.POT_format.txt||txt|FES2014b.v1 Cnm-Snm, POT format, Om1+Om2 equilibrium, C20 without S1"
  "igrf14coeffs|https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt||txt|IAGA IGRF-14 Gauss coefficients, 1900.0-2025.0 plus 2025-30 secular variation"
  "iers-tab5.2a|https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2a.txt||txt|IERS Conventions (2010) Table 5.2a, CIP X from IAU 2006 precession and IAU 2000A_R06 nutation"
  "iers-tab5.2b|https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2b.txt||txt|IERS Conventions (2010) Table 5.2b, CIP Y from IAU 2006 precession and IAU 2000A_R06 nutation"
  "iers-tab5.2d|https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2d.txt||txt|IERS Conventions (2010) Table 5.2d, the CIO locator s + XY/2"
  "iers-tab5.3a|https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.3a.txt||txt|IERS Conventions (2010) Table 5.3a, IAU 2000A_R06 nutation in longitude"
  "iers-tab5.3b|https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.3b.txt||txt|IERS Conventions (2010) Table 5.3b, IAU 2000A_R06 nutation in obliquity"
  "iers-fes2004|https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat||dat|IERS Conventions (2010) chapter 6, FES2004 Cnm-Snm to (100,100), the Conventions Centre copy"
  "python-sgp4-propagation|https://raw.githubusercontent.com/brandon-rhodes/python-sgp4/2.27/sgp4/propagation.py||py|python-sgp4 2.27 sgp4/propagation.py (MIT), Vallado's SGP4/SDP4 routines line for line"
  "gpt3-5-grd|https://vmf.geo.tuwien.ac.at/codes/gpt3_5.grd||grd|TU Wien GPT3 5x5 degree grid (CC BY 4.0), CRLF line ends"
  "h3-faceijk|https://raw.githubusercontent.com/uber/h3/v4.5.0/src/h3lib/lib/faceijk.c||c|Uber H3 v4.5.0 faceijk.c (Apache-2.0): icosahedron face centres and axis azimuths"
  "h3-constants|https://raw.githubusercontent.com/uber/h3/v4.5.0/src/h3lib/include/constants.h||h|Uber H3 v4.5.0 constants.h (Apache-2.0): resolution-0 gnomonic unit, aperture-7 rotation"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name url range ext note <<< "$spec"

  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' EXIT

  echo "fetching $name"
  args=(-fsSL --retry 3 --retry-delay 5 -o "$tmp")
  [[ -n "$range" ]] && args+=(-r "$range")
  if ! curl "${args[@]}" "$url"; then
    echo "  FAILED — $url unreachable; $name not refreshed" >&2
    continue
  fi

  digest="$(sha256sum "$tmp" | cut -d' ' -f1)"
  short="${digest:0:8}"
  snapshot="$DEST/$name.$short.$ext"

  if [[ -e "$snapshot" ]]; then
    echo "  snapshot already held: $snapshot"
    continue
  fi

  cp "$tmp" "$snapshot"
  cat > "$DEST/$name.$short.json" <<JSON
{
  "source_url": "$url",
  "byte_range": "${range:-full}",
  "fetched_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$digest",
  "bytes": $(stat -c%s "$snapshot"),
  "lines": $(wc -l < "$snapshot"),
  "format": "$note"
}
JSON
  echo "  snapshot   $snapshot"
  echo "  sha256     $digest"
done

echo
echo "now re-derive the tables from them:"
echo "  uv run --no-project python derivations/spec_seals.py $DEST"
