#!/usr/bin/env bash
# Fetch the NRL Horizontal Wind Model HWM14 as an *oracle*. Nothing in the workspace depends on it.
#
# WHY IT IS FETCHED AND NOT VENDORED
#
# HWM14's software and coefficients are the Supporting Information of Drob et al. (2015, Earth and
# Space Science 2, 301-319, doi:10.1002/2014EA000089) and carry a CC BY-NC-ND licence:
# NonCommercial, NoDerivatives. That licence permits running the model and redistributing it
# *unmodified*; it forbids derivative works, which is what a port would be, and it forbids
# commercial use, which is what this repository's product eventually is. So HWM14 is treated the
# way this repo treats every other third-party reference implementation whose licence does not let
# it ship: downloaded on demand into gitignored space, run as an independent check, never linked
# into a crate and never translated into one. crates/spacelab-propagation/src/wind_model.rs is our
# own model, fitted to open CC BY 4.0 crosswind measurements, and shares no line with it.
#
# The download is the gemini3d/hwm14 mirror, which carries the original src/hwm14.f90, the three
# NRL coefficient files, and NRL's own checkhwm14.f90 acceptance program. That repository applies
# Apache-2.0 to its build system; the model source and coefficients inside it remain NRL's under
# the terms above, and a third party cannot relicense them. Treat the fetched tree as read-only.
#
# The pin is a commit, not a branch: an oracle whose version moves under the comparison is not an
# oracle. What is recorded is the snapshot -- where it came from, when, and its digest.
#
# Usage: tools/fetch-hwm14.sh [COMMIT]

set -euo pipefail

COMMIT="${1:-${HWM14_COMMIT:-19e222dbc0e219869b8cdf4e22fc1d4ab969b51a}}"
DEST="${HWM14_DIR:-data/oracles/hwm14}"
BASE="${HWM14_URL:-https://codeload.github.com/gemini3d/hwm14/tar.gz}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "COMMIT must be a full 40-character git object id, got '$COMMIT'" >&2
  exit 2
fi

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

url="${BASE}/${COMMIT}"
if ! curl -sS -L --fail --max-time 300 -o "$tmp/hwm14.tar.gz" "$url"; then
  echo "could not fetch $url" >&2
  exit 1
fi

sha="$(sha256sum "$tmp/hwm14.tar.gz" | cut -d' ' -f1)"
bytes="$(stat -c%s "$tmp/hwm14.tar.gz")"
tar -xzf "$tmp/hwm14.tar.gz" -C "$tmp"
root="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'hwm14-*' -print -quit)"
if [[ -z "$root" ]]; then
  echo "unexpected archive layout" >&2
  exit 1
fi

# Only the model and its coefficients. The mirror's own CMake and interface wrapper are not needed
# and are not what is being verified against.
for f in src/hwm14.f90 test/checkhwm14.f90 data/hwm123114.bin data/dwm07b104i.dat data/gd2qd.dat LICENSE.txt; do
  if [[ ! -f "$root/$f" ]]; then
    echo "missing $f in the fetched archive" >&2
    exit 1
  fi
  cp "$root/$f" "$DEST/$(basename "$f")"
done

cat >"$DEST/manifest.json" <<EOF
{
  "url": "$url",
  "commit": "$COMMIT",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$sha",
  "bytes": $bytes,
  "model": "NRL HWM14 (Drob et al. 2015, doi:10.1002/2014EA000089)",
  "license": "CC BY-NC-ND -- run and redistribute unmodified; no derivatives, no commercial use",
  "role": "oracle only; no crate in this workspace depends on it"
}
EOF

echo "hwm14 $COMMIT -> $DEST"
ls -1 "$DEST"
