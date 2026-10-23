#!/usr/bin/env bash
# Compile and run the IERS Conventions reference routines the spacelab-slr models are ported from.
#
# The routines are fetched, not vendored: they are the IERS Conventions Centre's own distribution
# and belong to it. What is committed here is the driver that calls them and the README recording
# what came out.
#
# Two builds, because they disagree. FCULZD_HPA declares its working variables with no KIND, so a
# default gfortran build evaluates the Mendes-Pavlis zenith delay in single precision; the
# `-fdefault-real-8` build evaluates it in double. The difference is 2 parts in 10^6 -- 3.8 um on a
# 1.93 m delay -- and the values printed in the routine's own header comment match neither exactly.
# See README.md.
#
# Usage: tools/iers-oracle/run.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${IERS_URL:-https://iers-conventions.obspm.fr/content}"
work="${IERS_ORACLE_DIR:-$(mktemp -d)}"
mkdir -p "$work"

declare -A sources=(
  [FCUL_ZD_HPA.F]="chapter9/software"
  [FCUL_A.F]="chapter9/software"
  [DEHANTTIDEINEL.F]="chapter7/software/dehanttideinel"
  [CAL2JD.F]="chapter7/software/dehanttideinel"
  [DAT.F]="chapter7/software/dehanttideinel"
  [NORM8.F]="chapter7/software/dehanttideinel"
  [SPROD.F]="chapter7/software/dehanttideinel"
  [ST1IDIU.F]="chapter7/software/dehanttideinel"
  [ST1ISEM.F]="chapter7/software/dehanttideinel"
  [ST1L1.F]="chapter7/software/dehanttideinel"
  [STEP2DIU.F]="chapter7/software/dehanttideinel"
  [STEP2LON.F]="chapter7/software/dehanttideinel"
  [ZERO_VEC8.F]="chapter7/software/dehanttideinel"
  [TDFRPH.F]="chapter7/software/hardisp"
  [TOYMD.F]="chapter7/software/hardisp"
  [JULDAT.F]="chapter7/software/hardisp"
  [LEAP.F]="chapter7/software/hardisp"
  [ETUTC.F]="chapter7/software/hardisp"
  [SPLINE.F]="chapter7/software/hardisp"
  [EVAL.F]="chapter7/software/hardisp"
)

# HARDISP.F itself is fetched only for its published test values, which live in its header
# comment; it is a PROGRAM and would collide with the driver, so it is not compiled in.

for name in "${!sources[@]}"; do
  if [[ ! -s "$work/$name" ]]; then
    curl -fsSL --retry 3 --retry-delay 2 -o "$work/$name" "$BASE/${sources[$name]}/$name"
  fi
done
(cd "$work" && sha256sum ./*.F | sed 's| \./| |' | sort > SHA256SUMS)

# CAL2JD and DAT are SOFA routines redistributed under their SOFA names, `iau_CAL2JD` and
# `iau_DAT`, while DEHANTTIDEINEL calls them unprefixed. The IERS distribution note says to strip
# the prefix; doing it here rather than by hand keeps the fetched sources untouched on disk.
sed -i 's/iau_//g' "$work/CAL2JD.F" "$work/DAT.F"

cp "$here/oracle.f" "$work/oracle.f"
cd "$work"

echo "=== gfortran, default reals ==="
gfortran -O2 -w -o oracle-single oracle.f ./*.F
./oracle-single

echo
echo "=== gfortran -fdefault-real-8 ==="
gfortran -O2 -w -fdefault-real-8 -o oracle-double oracle.f ./*.F
./oracle-double

echo
echo "sources in $work; digests in $work/SHA256SUMS"
