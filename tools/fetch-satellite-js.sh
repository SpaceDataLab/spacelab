#!/usr/bin/env bash
# Fetch satellite.js as an *oracle* for SGP4. Nothing in the workspace depends on it.
#
# WHY THIS PARTICULAR FILE
#
# Celestrak publishes satellite.js on its own site, beside the SGP4 reference distribution, as a
# single self-contained UMD bundle. That is what is fetched: one file, one digest, no package
# manager, no lockfile, no transitive tree. `npm install satellite.js` would pull the same code
# through a dependency resolver whose output is not pinned by anything in this repository, and the
# whole point of an oracle is that it does not move under the comparison.
#
# WHAT AGREEMENT WITH IT DOES AND DOES NOT ESTABLISH
#
# satellite.js is a JavaScript port of the same Vallado listing the `sgp4` crate ports. Agreement
# between the two therefore establishes that both transcribed the source correctly -- across the
# whole catalogue rather than the ~30 hand-picked cases of SGP4-VER.TLE -- and nothing about
# whether the model is right. crates/spacelab-sgp4/src/lib.rs states the same limit for the crate
# itself; this script does not lift it. For SGP4 there is no independent implementation to be had:
# the specification *is* a particular listing.
#
# The licence is MIT (Shashwat Kandadai and UCSC), which would permit vendoring. It is fetched
# rather than vendored so that every third-party oracle in this repository lives in one gitignored
# place with one manifest format, and so `git log` never carries someone else's 100 KB bundle.
#
# Usage: tools/fetch-satellite-js.sh [URL]

set -euo pipefail

URL="${1:-${SATELLITE_JS_URL:-https://celestrak.org/software/SGP4/satellite.js}}"
DEST="${SATELLITE_JS_DIR:-data/oracles/satellite-js}"
# The pin is a content digest, not a version string: Celestrak serves this path without a version in
# it, so a silent replacement would otherwise be invisible. Override only after reading the diff.
EXPECT="${SATELLITE_JS_SHA256:-48c444c7e07893429e559ec2c93ab555c5a36799b7b2a268d100bd4fb34b2366}"

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! curl -sS -L --fail --max-time 300 -o "$tmp/satellite.js" "$URL"; then
  echo "could not fetch $URL" >&2
  exit 1
fi

sha="$(sha256sum "$tmp/satellite.js" | cut -d' ' -f1)"
bytes="$(stat -c%s "$tmp/satellite.js")"
if [[ "$sha" != "$EXPECT" ]]; then
  echo "digest mismatch: expected $EXPECT, got $sha" >&2
  echo "the served file changed -- read the diff before moving the pin" >&2
  exit 1
fi

version="$(grep -m1 -o 'satellite-js v[0-9.]*' "$tmp/satellite.js" | cut -d' ' -f2)"
mv "$tmp/satellite.js" "$DEST/satellite.js"
chmod 444 "$DEST/satellite.js"

cat >"$DEST/manifest.json" <<EOF
{
  "url": "$URL",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$sha",
  "bytes": $bytes,
  "version": "$version",
  "model": "satellite.js -- JavaScript port of the Vallado SGP4/SDP4 listing",
  "license": "MIT (Shashwat Kandadai and UCSC)",
  "role": "oracle only; no crate in this workspace depends on it",
  "known_quirk": "twoline2satrec/csv2satrec/json2satrec in this build print three debug lines to stdout per element set; tools/sgp4-catalogue-oracle/oracle.cjs silences console.log for that reason"
}
EOF

echo "satellite.js $version ($sha) -> $DEST"
