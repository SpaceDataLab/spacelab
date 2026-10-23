#!/usr/bin/env bash
# Recompute the JRC column of crates/spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv.
#
# The JRC NeQuickG library is EUPL-1.2 and is fetched, not vendored: what is committed here is the
# driver that calls it. The CCIR and MODIP files it reads are its own copies, which are the ICD's.
#
# Usage: tools/nequick-jrc-oracle/run.sh   (prints the largest difference to the committed column)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fixture="$here/../../crates/spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv"
repo="https://github.com/odrisci/NeQuickJRC.git"
commit="02701f280fcfd1c7909a1d1bbc348bc55c625832"
work="${NEQUICK_JRC_DIR:-$(mktemp -d)}"

if [[ ! -d "$work/NeQuickJRC" ]]; then
  git clone -q "$repo" "$work/NeQuickJRC"
fi
git -C "$work/NeQuickJRC" checkout -q "$commit"

src="$work/NeQuickJRC/src/lib"
mapfile -t lib < <(find "$src" -name '*.c' ! -path '*/CCIR/*' ! -path '*/UT/*')
gcc -std=c11 -O2 -I"$src/public" -I"$src/private" "$here/stec.c" "${lib[@]}" -lm -o "$work/stec"

grep -v '^#' "$fixture" | tail -n +2 | cut -f2-12 | tr '\t' ' ' |
  "$work/stec" "$work/NeQuickJRC/modip/modip2001_wrapped.asc" "$work/NeQuickJRC/ccir/" \
  > "$work/stec.out"

grep -v '^#' "$fixture" | tail -n +2 | cut -f13 | paste - "$work/stec.out" |
  awk -F'\t' '{d = $1 - $2; if (d < 0) d = -d; if (d > m) m = d} END {printf "%d rows, max |committed - recomputed| = %.3g TECU\n", NR, m}'
