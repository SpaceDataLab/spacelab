#!/usr/bin/env bash
# Fetch full RINEX broadcast-navigation products -- the oracle for `spacelab-gnss`'s RINEX readers.
#
# WHY A WHOLE PRODUCT AND NOT A FIXTURE
#
# `crates/spacelab-gnss/tests/fixtures/` holds excerpts of these files, and the excerpts are chosen:
# one record of every shape the format produces. That is the right fixture for a unit test and it is
# not an answer to the question a parser actually has to survive, which is a quarter of a million
# lines of records interleaved in whatever order the satellites transmitted them. A reader one line
# out at any single transition reads every record after it from the wrong columns, and there is no
# way to see that on 200 lines someone picked.
#
# So the committed tests use the fixtures and `#[ignore]`d tests use these, gated on the cache being
# present. The products are ~12 MB each and are never committed.
#
# LICENSING
#
# IGS data are covered by the *IGS Data and Product Disclaimer and Terms of Use* (5 August 2020):
# "provided openly for the benefit of all scientific, educational, and commercial users", "made
# openly available for use without restriction", with attribution required and use constituting
# acceptance. `docs/provenance.md` §6 quotes it and records the position. BRD400DLR additionally
# carries its own DOI (10.57677/BRD400DLR) and names its authors in the file header; both are kept
# verbatim in the committed excerpt.
#
# Usage: tools/fetch-gnss-nav.sh [DEST]

set -euo pipefail

DEST="${1:-$HOME/.cache/spacelab/gnss}"
BASE="https://igs.bkg.bund.de/root_ftp/IGS/BRDC"

mkdir -p "$DEST"

# BRD400DLR is the RINEX 4 merged broadcast product; BRDC00WRD is BKG's RINEX 3 one. Four days
# spread over two and a half years, so that a change in what the constellations broadcast -- a new
# message type, a new constellation -- shows up as a difference between days rather than as a single
# file that happens to work.
#
# 2024/001 is deliberately a RINEX **4.00** file: the version predates the 4.02 the reader targets,
# and a reader that only works on the newest revision of a format is a reader that will break.
fetch() {
    local name="$1" url="$2" out="$DEST/$3"
    if [ -s "$out" ]; then
        printf '  %-24s already present\n' "$name"
        return
    fi
    printf '  %-24s ' "$name"
    if curl -sSfL --retry 3 --max-time 600 -o "$out.gz" "$url"; then
        gunzip -f "$out.gz"
        printf '%s lines\n' "$(wc -l < "$out")"
    else
        printf 'FAILED (%s)\n' "$url"
        rm -f "$out.gz"
        return 1
    fi
}

echo "RINEX 4 broadcast navigation (DLR/GSOC BCEmerge, doi:10.57677/BRD400DLR):"
# 2026/224 is the day `data/validation/IGS0OPSRAP_20262240000_01D_15M_ORB.SP3` covers, and it is in
# the list for that reason alone: it is the only day for which this repository holds both a
# broadcast navigation file and an IGS precise orbit, which is what makes
# `the_cnav_propagator_lands_where_the_precise_orbit_says` an independent oracle rather than a
# comparison of the algorithm with itself.
for spec in 2024/001 2025/300 2026/150 2026/200 2026/224; do
    y="${spec%/*}"; d="${spec#*/}"
    fetch "BRD400DLR $y/$d" \
        "$BASE/$y/$d/BRD400DLR_S_${y}${d}0000_01D_MN.rnx.gz" \
        "brd400dlr_${y}${d}.rnx"
done

echo "RINEX 3 broadcast navigation (BKG Frankfurt), for the cross-version comparison:"
for spec in 2024/001; do
    y="${spec%/*}"; d="${spec#*/}"
    fetch "BRDC00WRD $y/$d" \
        "$BASE/$y/$d/BRDC00WRD_R_${y}${d}0000_01D_MN.rnx.gz" \
        "brdc00wrd_${y}${d}.rnx"
done

cat <<'NOTE'

Fetched. The `#[ignore]`d tests in crates/spacelab-gnss/tests/ pick these up automatically:

    cargo test -p spacelab-gnss -- --ignored

Attribution, per the IGS Terms of Use: data from the International GNSS Service, via the BKG
Frankfurt data centre; the RINEX 4 product is DLR/GSOC's BCEmerge (O. Montenbruck,
P. Steigenberger), doi:10.57677/BRD400DLR.
NOTE
