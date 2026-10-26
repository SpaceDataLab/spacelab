#!/usr/bin/env bash
# Fetch the GOT and EOT20 ocean-tide atlases -- the cotidal grids `derivations/tide_to_stokes.py`
# analyses.
#
# WHY AN ATLAS AND NOT A COEFFICIENT FILE
#
# `crates/spacelab-propagation/src/ocean_tide_fes2004.rs` and `..._fes2014.rs` transcribe geopotential
# coefficient files that somebody else computed from a tide atlas. No such file is published for any
# modern *openly licensed* atlas: GOT ships elevation grids and nothing else, and the FES grids that
# do have a published coefficient file are behind AVISO registration. That is why this repository
# computes the coefficients itself; the grids are the input.
#
# LICENSING, READ BEFORE USING ANOTHER ATLAS
#
# GOT is produced by R. D. Ray at NASA Goddard and served from `earth.gsfc.nasa.gov` with no stated
# terms, no registration and no click-through, and that finding is recorded on its row of
# `docs/provenance.md`. The FES family is *not* in that position: `swami-h2020-eu/mcm`
# taught this workspace what an academic-non-commercial clause costs, and the FES grids require an
# AVISO account whose terms are accepted at registration. So this script fetches GOT and only prints
# instructions for FES -- a registration is a decision for the user, not a curl invocation.
#
# GOT5.6 is a *supplement* to GOT5.5, adding four third-degree constituents; its own README says to
# combine the two. Both are fetched and unpacked into one tree for that reason.
#
# EOT20 (DGFI-TUM) is CC BY 4.0, read on its SEANOE page, and is fetched by default. SEANOE serves
# one 2.3 GB archive holding both the ocean-tide and the load-tide grids; only the ocean-tide half is
# unpacked, because the load tide is a different quantity on an identical grid. The digest is pinned:
# SEANOE withdrew an earlier release of EOT20 whose netCDF conversion was corrupt, and a changed
# archive must stop the script rather than reach the generator.
#
# The atlases are ~3.5 GB and are never committed. It lands in the cache, outside the repository, and what
# this repository commits is the generator and the table it produces.
#
# Usage: tools/fetch-tide-atlas.sh [DEST]

set -euo pipefail

DEST="${1:-${TIDE_ATLAS_DIR:-$HOME/.cache/spacelab/tides}}"
BASE="https://earth.gsfc.nasa.gov/sites/default/files/2024-07"

mkdir -p "$DEST"

# name|url|what it is
atlases=(
  "GOT5.5.tar.gz|${BASE}/GOT5.5.tar.gz|GOT5.5 cotidal grids, R. D. Ray, NASA GSFC"
  "GOT5.6.tar.gz|${BASE}/GOT5.6.tar%201.gz|GOT5.6 third-degree supplement to GOT5.5"
)

for entry in "${atlases[@]}"; do
  IFS='|' read -r name url what <<<"$entry"
  if [[ -s "$DEST/$name" ]]; then
    echo "have $name ($what)"
  else
    echo "fetching $name -- $what"
    if ! curl -sS -L --fail --max-time 1800 -o "$DEST/$name" "$url"; then
      echo "could not fetch $url" >&2
      exit 1
    fi
  fi
  tar xzf "$DEST/$name" -C "$DEST" 2>/dev/null || tar xzf "$DEST/$name" -C "$DEST"
done

# GOT5.6's four supplements belong in the same directory as GOT5.5's constituents; its README says
# so explicitly ("You can combine the GOT5.5 and GOT5.6 files and call it GOT5.6").
if [[ -d "$DEST/GOT5.6/ocean_tides" && -d "$DEST/GOT5.5/ocean_tides" ]]; then
  cp -n "$DEST"/GOT5.6/ocean_tides/*.nc "$DEST/GOT5.5/ocean_tides/" 2>/dev/null || true
fi

{
  echo "{"
  echo "  \"retrieved_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"license\": \"no stated terms; NASA GSFC public data, R. D. Ray\","
  echo "  \"role\": \"input to derivations/tide_to_stokes.py\","
  echo "  \"files\": {"
  first=1
  for entry in "${atlases[@]}"; do
    IFS='|' read -r name url _ <<<"$entry"
    [[ $first -eq 1 ]] || echo ","
    first=0
    printf '    "%s": {"url": "%s", "sha256": "%s", "bytes": %s}' \
      "$name" "$url" "$(sha256sum "$DEST/$name" | cut -d' ' -f1)" "$(stat -c%s "$DEST/$name")"
  done
  echo
  echo "  }"
  echo "}"
} >"$DEST/manifest.json"

EOT20_DIR="$DEST/EOT20"
EOT20_URL="https://www.seanoe.org/data/00683/79489/data/85762.zip"
EOT20_SHA256="bced7af7eb7c34896d4fd04680a751cc9f5a4c7a3c03852997d9263d61e07018"
mkdir -p "$EOT20_DIR"
if [[ -s "$EOT20_DIR/85762.zip" ]]; then
  echo "have 85762.zip (EOT20, final version)"
else
  echo "fetching 85762.zip -- EOT20 cotidal grids, DGFI-TUM, final version"
  if ! curl -sS -L --fail --max-time 3600 -o "$EOT20_DIR/85762.zip" "$EOT20_URL"; then
    echo "could not fetch $EOT20_URL" >&2
    exit 1
  fi
fi
if [[ "$(sha256sum "$EOT20_DIR/85762.zip" | cut -d' ' -f1)" != "$EOT20_SHA256" ]]; then
  echo "85762.zip does not match the pinned digest; not unpacking it" >&2
  exit 1
fi
if [[ ! -d "$EOT20_DIR/ocean_tides_x/ocean_tides" ]]; then
  unzip -q -o "$EOT20_DIR/85762.zip" ocean_tides.zip -d "$EOT20_DIR"
  unzip -q -o "$EOT20_DIR/ocean_tides.zip" -d "$EOT20_DIR/ocean_tides_x"
  rm -f "$EOT20_DIR/ocean_tides.zip"
fi
{
  echo "{"
  echo "  \"retrieved_utc\": \"$(date -u -r "$EOT20_DIR/85762.zip" +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"license\": \"CC BY 4.0 (SEANOE dataset page); cite doi:10.17882/79489 and doi:10.5194/essd-13-3869-2021\","
  echo "  \"role\": \"input to derivations/tide_to_stokes.py --atlas eot20\","
  echo "  \"files\": {"
  printf '    "85762.zip": {"url": "%s", "sha256": "%s", "bytes": %s}\n' \
    "$EOT20_URL" "$EOT20_SHA256" "$(stat -c%s "$EOT20_DIR/85762.zip")"
  echo "  }"
  echo "}"
} >"$EOT20_DIR/manifest.json"

echo
echo "atlas -> $DEST/GOT5.5/ocean_tides ($(find "$DEST/GOT5.5/ocean_tides" -name '*.nc' -not -name '._*' | wc -l) constituents)"
echo "atlas -> $EOT20_DIR/ocean_tides_x/ocean_tides ($(find "$EOT20_DIR/ocean_tides_x/ocean_tides" -name '*.nc' -not -name '._*' | wc -l) constituents)"
echo
echo "FES2014b / FES2022b are not fetched here. They require an AVISO account"
echo "(https://www.aviso.altimetry.fr/en/data/data-access.html) whose terms are accepted at"
echo "registration, so obtaining them is the user's decision, not this script's. The generator"
echo "reads the GOT and EOT20 grid layouts only; a FES reader is deliberately unwritten until"
echo "there are FES grids to test it against."
