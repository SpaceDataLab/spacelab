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
# WHAT CELESTRAK SERVES, read 2026-11-12 (`docs/provenance.md`, row `celestrak`; `plans/17` F51)
#
# * SupGP is selected by SOURCE, the provider of a group of element sets: GPS-A, GPS-E, CPF,
#   SpaceX-E, SpaceX-SV, OneWeb-E, Iridium-E, Planet-E, ... (sup-gp-queries.php). FILE, the sets on
#   the SupGP page, still works; this script uses it only for the fixture snapshot below.
# * FORMAT defaults to CSV since 2026-05-09, OMM keywords with nine-digit catalog numbers. Celestrak
#   ran out of five-digit catalog numbers on 2026-07-11: an object numbered 100000 or above is absent
#   from TLE, 3LE and 2LE, without a word. Only CSV, XML, KVN and JSON carry it.
# * GP data update once every two hours, and since 2026-03-26 a second download inside one update
#   answers HTTP 403 with a message saying so. Fifty 301/403/404 answers in two hours put the address
#   in Celestrak's firewall, and repeating a 403 or a 404 never changes it.
#
# USAGE POLICY, which this script exists partly to encode
#
# Nothing here retries or loops. A non-200 stops the script, and the provider's message is printed,
# not swallowed: it says why. A URL fetched successfully less than two hours ago is not requested
# again, because the answer could be that 403; the copy on disk is kept, as current as a repeat.
#
# Usage: tools/fetch-supgp.sh [SOURCE ...]     (default: GPS-A CPF SpaceX-E OneWeb-E Iridium-E), CSV
#        tools/fetch-supgp.sh --fixture-snapshot
#
# The second form takes the TLE snapshots `tools/gen_supgp_fixtures.py` cuts the committed fixtures
# from (FILE=gps cpf starlink oneweb iridium, and GROUP=active): those fixtures are TLE pairs, so they
# hold no object numbered 100000 or above. FORMAT=xml|kvn|json overrides CSV in the first form.

set -euo pipefail

DEST="${SUPGP_DIR:-data/supgp}"
BASE="${SUPGP_URL:-https://celestrak.org/NORAD/elements/supplemental/sup-gp.php}"
GP_BASE="${GP_URL:-https://celestrak.org/NORAD/elements/gp.php}"
UPDATE_SECONDS=7200

mkdir -p "$DEST"

fetch() {
  local url="$1" out="$2" label="$3" format="$4"
  if [[ -f "$out.json" ]]; then
    local last now
    last="$(sed -n 's/^  "retrieved_utc": "\(.*\)",$/\1/p' "$out.json")"
    now="$(date -u +%s)"
    if [[ -n "$last" ]] && grep -qF "\"url\": \"$url\"" "$out.json" &&
      ((now - $(date -u -d "$last" +%s) < UPDATE_SECONDS)); then
      echo "$label: kept the copy fetched at $last, less than one two-hour update ago -- not requested" >&2
      return 0
    fi
  fi
  local status
  status="$(curl -sS -L --max-time 300 -o "$out.part" -w '%{http_code}' "$url")" || {
    echo "$label: no answer from Celestrak (curl failed); stopping" >&2
    rm -f "$out.part"
    exit 1
  }
  if [[ "$status" != 200 ]]; then
    echo "$label: Celestrak answered HTTP $status; stopping, as its usage policy asks. Its message:" >&2
    head -c 2000 "$out.part" | tr -cd '[:print:]\n' | sed 's/^/  | /' >&2
    echo >&2
    rm -f "$out.part"
    exit 1
  fi
  local records
  if [[ "$format" == tle ]]; then
    records="$(grep -c '^1 ' "$out.part" || true)"
  elif [[ "$format" == csv ]]; then
    if ! head -1 "$out.part" | grep -q '^OBJECT_NAME,'; then
      echo "$label: 200 but the body is not OMM CSV" >&2
      rm -f "$out.part"
      exit 1
    fi
    records=$(($(grep -c . "$out.part") - 1))
  else
    records="$(grep -c 'NORAD_CAT_ID' "$out.part" || true)"
  fi
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
  "format": "$format",
  "role": "oracle input; the fixtures under crates/*/tests/fixtures are cut from the TLE snapshot by tools/gen_supgp_fixtures.py"
}
EOF
  echo "$label: $records records -> $out"
}

if [[ "${1:-}" == --fixture-snapshot ]]; then
  for file in gps cpf starlink oneweb iridium; do
    fetch "${BASE}?FILE=${file}&FORMAT=tle" "$DEST/sup-$file.tle" "sup-$file" tle
  done
  # The standard catalogue, for the same objects. Without it the SupGP records have nothing to be
  # compared against, and fetching it here keeps the two snapshots within minutes of each other --
  # which matters, because the whole comparison is between two fits of the same object at one time.
  fetch "${GP_BASE}?GROUP=active&FORMAT=tle" "$DEST/gp-active.tle" "gp-active" tle
  exit 0
fi

FORMAT="${FORMAT:-csv}"
if [[ ! "$FORMAT" =~ ^(csv|xml|kvn|json)$ ]]; then
  echo "FORMAT is csv, xml, kvn or json here; TLE drops every object numbered 100000 or above (use --fixture-snapshot for the TLE fixtures)" >&2
  exit 2
fi
SOURCES=("$@")
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  SOURCES=(GPS-A CPF SpaceX-E OneWeb-E Iridium-E)
fi
for source in "${SOURCES[@]}"; do
  # The URL is assembled from an allow-listed shape, never from free text: a SOURCE is letters and
  # digits with at most one hyphen, and anything else is refused before it reaches curl.
  if [[ ! "$source" =~ ^[A-Za-z0-9]+(-[A-Za-z0-9]+)?$ ]]; then
    echo "not a SupGP SOURCE: '$source'" >&2
    exit 2
  fi
  fetch "${BASE}?SOURCE=${source}&FORMAT=${FORMAT}" "$DEST/sup-$source.$FORMAT" "sup-$source" "$FORMAT"
done
fetch "${GP_BASE}?GROUP=active&FORMAT=${FORMAT}" "$DEST/gp-active.$FORMAT" "gp-active" "$FORMAT"
