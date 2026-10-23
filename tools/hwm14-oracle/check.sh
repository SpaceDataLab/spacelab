#!/usr/bin/env bash
# Verify the HWM14 build against its own author's reference before trusting it as an oracle.
#
# NRL ships checkhwm14.f90 with the expected output embedded as a trailing comment block. This
# compiles it against the unmodified hwm14.f90, runs it, and compares every printed number. A
# compiler, a libm or a fetch that changed the model would show up here rather than as a quiet few
# metres per second inside a comparison that is supposed to be judging *our* model.
#
# The tolerance is the print format's own last digit: the reference is written to three decimals.
#
# Usage: tools/hwm14-oracle/check.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${HWM14_DIR:-$(cd "$here/../.." && pwd)/data/oracles/hwm14}"

if [[ ! -f "$data/checkhwm14.f90" ]]; then
  echo "no HWM14 source in $data -- run tools/fetch-hwm14.sh first" >&2
  exit 1
fi

build="$data/build"
mkdir -p "$build"
(cd "$build" && gfortran -O2 -std=legacy -w -o checkhwm14 "$data/hwm14.f90" "$data/checkhwm14.f90")
(cd "$data" && "$build/checkhwm14" >"$build/checkhwm14.out")

python3 - "$data/checkhwm14.f90" "$build/checkhwm14.out" <<'PY'
import re
import sys

source, produced = sys.argv[1], sys.argv[2]
tail = open(source).read().split("end program")[-1]
expected = [ln[1:].rstrip() if ln.startswith("!") else ln.rstrip() for ln in tail.splitlines()]
expected = [ln for ln in expected if ln.strip()]
got = [ln.rstrip() for ln in open(produced).read().splitlines() if ln.strip()]


def numbers(line):
    return [float(v) for v in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)]


worst, where, compared, cursor = 0.0, "", 0, 0
for line in got:
    while cursor < len(expected) and len(numbers(expected[cursor])) != len(numbers(line)):
        cursor += 1
    if cursor >= len(expected):
        break
    reference, ours = numbers(expected[cursor]), numbers(line)
    if reference and len(reference) == len(ours):
        for a, b in zip(reference, ours):
            compared += 1
            if abs(a - b) > worst:
                worst, where = abs(a - b), f"{a} vs {b} in '{line.strip()[:60]}'"
    cursor += 1

if compared < 500:
    sys.exit(f"only {compared} numbers matched up to the reference -- the layout changed")
print(f"compared {compared} numbers, worst |difference| {worst:g} m/s")
if worst > 0.0011:
    sys.exit(f"HWM14 does not reproduce its own reference: {where}")
print("HWM14 reproduces NRL's published check output to the printed precision")
PY
