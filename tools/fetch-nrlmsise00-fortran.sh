#!/usr/bin/env bash
# Fetch the NRL FORTRAN for NRLMSISE-00 -- the source `derivations/nrlmsise00_tables.py` reads.
#
# WHY THIS FILE AND NOT THE C
#
# The C translation by Dominik Brodowski (release 20041227) is the copy most of the world uses, and
# it carries **no licence grant at all**: no COPYING, no permission, only a warranty disclaimer, and
# GitHub reports `license: null` for the maintained mirror. No licence is a worse position than a
# copyleft licence, not a better one -- copyleft grants permission on conditions, silence grants
# nothing. `docs/provenance.md` records the reading.
#
# `nrlmsise00_sub.for` is the model's own distribution by M. Picone, A. E. Hedin and D. Drob at the
# U.S. Naval Research Laboratory, published through NASA GSFC's modelweb, and a work of the United
# States Government is not subject to domestic copyright (17 U.S.C. section 105). The C is a
# transcription of it. So the FORTRAN is both the more authoritative source and the one this
# workspace may descend from, and the Rust module descends from it.
#
# The claim is checked rather than asserted. Regenerating the coefficient tables from this file
# reproduces the 3 290 shipped values byte for byte -- same digits, same order -- which is what makes
# the provenance a measurement:
#
#   tools/fetch-nrlmsise00-fortran.sh
#   uv run --no-project python derivations/nrlmsise00_tables.py \
#       data/oracles/nrlmsise00/nrlmsise00_sub.for /tmp/tables.rs
#   diff <(tail -n +7 /tmp/tables.rs) \
#        <(tail -n +7 crates/spacelab-propagation/src/nrlmsise00/tables.rs)
#
# NASA's own modelweb FTP directory has not survived the agency's web moves, so the pin is a mirror
# holding the unmodified `src_f/` tree, fixed to a commit rather than a branch: a source whose
# version moves under the comparison is not a source. What is recorded is the snapshot.
#
# Usage: tools/fetch-nrlmsise00-fortran.sh [COMMIT]

set -euo pipefail

COMMIT="${1:-${NRLMSISE00_COMMIT:-}}"
DEST="${NRLMSISE00_DIR:-data/oracles/nrlmsise00}"
REPO="${NRLMSISE00_REPO:-raabrp/nrlmsise00}"

if [[ -z "$COMMIT" ]]; then
  COMMIT="$(curl -sS -L --fail --max-time 60 \
    "https://api.github.com/repos/${REPO}/commits?path=src_f&per_page=1" |
    sed -n 's/.*"sha": "\([0-9a-f]\{40\}\)".*/\1/p' | head -1)"
  if [[ -z "$COMMIT" ]]; then
    echo "could not resolve the latest commit touching src_f; pass one explicitly" >&2
    exit 1
  fi
  echo "no commit given; using the newest that touched src_f: $COMMIT" >&2
fi

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "COMMIT must be a full 40-character git object id, got '$COMMIT'" >&2
  exit 2
fi

mkdir -p "$DEST"
base="https://raw.githubusercontent.com/${REPO}/${COMMIT}/src_f"

# The subroutines carry the model and BLOCK DATA GTD7BK; the driver and its recorded output are the
# authors' own acceptance case, which is where the 17 canonical test vectors in the Rust come from.
for f in nrlmsise00_sub.for nrlmsise00_driver.for nrlmsise00_output.txt readme.txt; do
  if ! curl -sS -L --fail --max-time 120 -o "$DEST/$f" "$base/$f"; then
    echo "could not fetch $base/$f" >&2
    exit 1
  fi
done

sha="$(sha256sum "$DEST/nrlmsise00_sub.for" | cut -d' ' -f1)"
bytes="$(stat -c%s "$DEST/nrlmsise00_sub.for")"

cat >"$DEST/manifest.json" <<EOF
{
  "url": "$base/nrlmsise00_sub.for",
  "commit": "$COMMIT",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$sha",
  "bytes": $bytes,
  "model": "NRLMSISE-00 (Picone, Hedin & Drob 2002), NRL FORTRAN nrlmsise00_sub.for",
  "license": "US Government work, not subject to domestic copyright (17 U.S.C. 105)",
  "role": "source of record for crates/spacelab-propagation/src/nrlmsise00 and for its coefficient tables"
}
EOF

echo "nrlmsise00 $COMMIT -> $DEST"
ls -1 "$DEST"
