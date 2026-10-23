#!/usr/bin/env bash
# Fetch the GOT ocean-tide atlas -- the cotidal grids `derivations/tide_to_stokes.py` analyses.
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
# terms, no registration and no click-through -- the same position as JB2008, and recorded on the
# same row of `docs/provenance.md`. The FES family is *not* in that position: `swami-h2020-eu/mcm`
# taught this workspace what an academic-non-commercial clause costs, and the FES grids require an
# AVISO account whose terms are accepted at registration. So this script fetches GOT and only prints
# instructions for FES -- a registration is a decision for the user, not a curl invocation.
#
# GOT5.6 is a *supplement* to GOT5.5, adding four third-degree constituents; its own README says to
# combine the two. Both are fetched and unpacked into one tree for that reason.
#
# The atlas is ~1 GB and is never committed. It lands in the cache, outside the repository, and what
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

echo
echo "atlas -> $DEST/GOT5.5/ocean_tides ($(find "$DEST/GOT5.5/ocean_tides" -name '*.nc' -not -name '._*' | wc -l) constituents)"
echo
echo "FES2014b / FES2022b are not fetched here. They require an AVISO account"
echo "(https://www.aviso.altimetry.fr/en/data/data-access.html) whose terms are accepted at"
echo "registration, so obtaining them is the user's decision, not this script's. The generator"
echo "reads the GOT grid layout only; a FES reader is deliberately unwritten until there are"
echo "FES grids to test it against."
