#!/usr/bin/env bash
# Fetch NEPTUNE as a *fourth oracle*. Nothing in the workspace depends on it.
#
# WHY THIS ONE
#
# NEPTUNE (Space-Systems/neptune, TU Braunschweig ILR / IRAS, Apache-2.0, Fortran-2008) is a
# numerical orbit propagator of a lineage that shares nothing with the three references already in
# use: it is not derived from Orekit, does not read Orekit's conventions, and was written by a
# different group against the same standards documents rather than against the same code. That
# independence is the whole value -- a fourth implementation that agrees is evidence; a fourth
# implementation copied from the first is a tautology.
#
# It is Apache-2.0, so unlike HWM14 it *could* be linked. It still is not: the point of an oracle is
# that it is a separate implementation run separately, and a Fortran propagator inside the Cargo
# workspace would be a build dependency for everybody and a licence question for the product.
#
# The build pulls libslam (IRAS/libslam) and pFUnit at configure time, so the first build needs the
# network. That is the "build-expensive" the ticket warns about.
#
# The pin is a commit, not a branch: an oracle whose version moves under the comparison is not an
# oracle. What is recorded is the snapshot -- where it came from, when, and its digest.
#
# Usage: tools/fetch-neptune.sh [COMMIT]

set -euo pipefail

COMMIT="${1:-${NEPTUNE_COMMIT:-9c4b4cae0bc82c038c2a8a6c0259f3c2b8a760cc}}"
DEST="${NEPTUNE_DIR:-data/oracles/neptune}"
BASE="${NEPTUNE_URL:-https://codeload.github.com/Space-Systems/neptune}"

# A codeload tarball carries no submodules, so libslam is fetched separately at the commit NEPTUNE's
# own tree pins. OPI and pFUnit are deliberately absent: the build is configured with
# -DENABLE_OPI_SUPPORT=OFF and -DENABLE_PFUNIT=OFF, since the oracle is driven through NEPTUNE's own
# standalone executable and its own test framework is not what is being verified here.
LIBSLAM_COMMIT="${LIBSLAM_COMMIT:-f3d0484bede85ec12457b457bc1c6dce4e3cef36}"
LIBSLAM_URL="${LIBSLAM_URL:-https://codeload.github.com/Space-Systems/libslam}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "COMMIT must be a full 40-character git object id, got '$COMMIT'" >&2
  exit 2
fi

if [[ -f "$DEST/manifest.json" ]] && grep -q "\"commit\": \"$COMMIT\"" "$DEST/manifest.json"; then
  echo "already present at $COMMIT: $DEST"
  exit 0
fi

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

url="${BASE}/tar.gz/${COMMIT}"
if ! curl -sS -L --fail --max-time 900 -o "$tmp/neptune.tar.gz" "$url"; then
  echo "could not fetch $url" >&2
  exit 1
fi

sha="$(sha256sum "$tmp/neptune.tar.gz" | cut -d' ' -f1)"
bytes="$(stat -c%s "$tmp/neptune.tar.gz")"
tar -xzf "$tmp/neptune.tar.gz" -C "$tmp"
root="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'neptune-*' -print -quit)"
if [[ -z "$root" ]]; then
  echo "unexpected archive layout" >&2
  exit 1
fi
for marker in CMakeLists.txt src LICENSE validation/lro validation/sentinel-1b; do
  if [[ ! -e "$root/$marker" ]]; then
    echo "missing $marker in the fetched archive" >&2
    exit 1
  fi
done

libslam_url="${LIBSLAM_URL}/tar.gz/${LIBSLAM_COMMIT}"
if ! curl -sS -L --fail --max-time 900 -o "$tmp/libslam.tar.gz" "$libslam_url"; then
  echo "could not fetch $libslam_url" >&2
  exit 1
fi
libslam_sha="$(sha256sum "$tmp/libslam.tar.gz" | cut -d' ' -f1)"
tar -xzf "$tmp/libslam.tar.gz" -C "$tmp"
libslam_root="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'libslam-*' -print -quit)"
if [[ -z "$libslam_root" ]] || [[ ! -f "$libslam_root/CMakeLists.txt" ]]; then
  echo "unexpected libslam archive layout" >&2
  exit 1
fi

# The whole tree, unlike HWM14: NEPTUNE is built rather than compiled into a driver of ours, and its
# CMake, data files and validation cases are the thing being run.
rm -rf "${DEST:?}/src" "${DEST:?}/libslam"
cp -a "$root/." "$DEST/"
rmdir "$DEST/libslam" 2>/dev/null || true
cp -a "$libslam_root" "$DEST/libslam"

cat >"$DEST/manifest.json" <<EOF
{
  "url": "$url",
  "commit": "$COMMIT",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$sha",
  "bytes": $bytes,
  "libslam_commit": "$LIBSLAM_COMMIT",
  "libslam_sha256": "$libslam_sha",
  "model": "NEPTUNE -- NPI Ephemeris Propagation Tool with Uncertainty Extrapolation (TU Braunschweig ILR / IRAS)",
  "license": "Apache-2.0",
  "role": "oracle only; no crate in this workspace depends on it, and it is not linked"
}
EOF

echo "neptune $COMMIT -> $DEST"
du -sh "$DEST"
