#!/usr/bin/env bash
# Fetch the NASA CARA "Alfano" sample Conjunction Data Messages.
#
# WHY THESE FILES AND NOT THE NUMBERS
#
# crates/spacelab-estimation/src/conjunction.rs verifies four collision-probability methods against
# these eleven conjunctions, and it does it against numbers *transcribed by hand* from these files.
# A transcription is a place a digit goes missing, and the transcription is what the whole Pc oracle
# rests on. With the files present, tests/cara_cdm_ingest.rs recomputes each Pc from the CDM itself
# and asserts it equals the Pc from the transcribed table -- which checks the reader and the
# transcription at once, and is why this fetch exists.
#
# LICENCE, AND WHY THEY ARE FETCHED RATHER THAN VENDORED
#
# nasa/CARA_Analysis_Tools carries no LICENSE file; what it ships is four NASA Open Source Agreement
# documents (NOSA_GSC-*.pdf). NOSA is OSI-approved but is not a permissive licence, and this
# workspace is heading for a permissive release -- so nothing from that repository is committed here.
# The files are fetched into an ignored directory, the tests that use them skip when it is absent,
# and what is committed is the provenance: URL, digest, size, date.
#
# The underlying test cases are Alfano's published conjunction geometries (Alfano 2009, "Satellite
# Conjunction Monte Carlo Analysis"), which is why the reference probabilities themselves may be
# quoted: a number in a paper is a fact, not an expressive work.
#
# Usage: tools/fetch-cara-cdms.sh

set -euo pipefail

DEST="${CARA_DIR:-data/validation/cara}"
BASE="${CARA_URL:-https://raw.githubusercontent.com/nasa/CARA_Analysis_Tools/HEAD/DataFiles/SampleCDMs}"

mkdir -p "$DEST"

fetched=0
for n in 01 02 03 04 05 06 07 08 09 10 11; do
  name="AlfanoTestCase${n}.cdm"
  target="$DEST/$name"

  if [[ -e "$target" ]]; then
    echo "already held: $target"
    continue
  fi

  tmp="$(mktemp)"
  if ! curl -fsSL --retry 3 --retry-delay 5 -o "$tmp" "$BASE/$name"; then
    echo "  FAILED -- $BASE/$name unreachable; $name not fetched" >&2
    rm -f "$tmp"
    continue
  fi

  # A CDM that does not open with its own version keyword is an error page, not a message.
  if ! head -1 "$tmp" | grep -q '^CCSDS_CDM_VERS'; then
    echo "  REFUSED -- $name does not open with CCSDS_CDM_VERS" >&2
    rm -f "$tmp"
    continue
  fi

  mv "$tmp" "$target"
  fetched=$((fetched + 1))
  echo "fetched $name"
done

cat > "$DEST/manifest.json" <<JSON
{
  "source_repository": "https://github.com/nasa/CARA_Analysis_Tools",
  "source_path": "DataFiles/SampleCDMs/AlfanoTestCase{01..11}.cdm",
  "licence": "NASA Open Source Agreement (NOSA_GSC-18593-1 and others in the repository root)",
  "redistribution": "not committed to this repository; fetched on demand",
  "fetched_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "files": [
$(for f in "$DEST"/AlfanoTestCase*.cdm; do
    [[ -e "$f" ]] || continue
    printf '    {"name": "%s", "sha256": "%s", "bytes": %s},\n' \
      "$(basename "$f")" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c%s "$f")"
  done | sed '$ s/,$//')
  ]
}
JSON

echo "$fetched new file(s); manifest at $DEST/manifest.json"
