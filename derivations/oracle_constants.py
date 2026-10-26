#!/usr/bin/env python3
"""Every reference value a Maxima generator prints must still be somewhere in the tree.

The `.mac` files under `derivations/` and `tools/` that exist only to print high-precision reference
values are not proofs -- nothing in them can reduce to zero -- so `verify.sh` cannot check them the
way it checks an identity. What it can check is the thing they are for: the values were pasted into a
Rust test, and if the generator now prints something else, the paste is stale.

The match is numeric, not textual. A generator prints at 20 or 60 digits and the test stores an f64,
so their decimal strings differ in the last place even when the two are the same number; a digit
prefix would call that a mismatch and a rounded digit string would too. Comparing values with a
relative tolerance is the only comparison that answers the question actually being asked.

    uv run python derivations/oracle_constants.py

Exit status is 0 when every value is accounted for. A value the tree does not hold is either a stale
paste or something the generator prints to show its working; the second kind is listed per generator
below, with the reason, so a new one has to be looked at rather than absorbed.
"""

from __future__ import annotations

import bisect
import re
import subprocess
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parent.parent

# Values a generator prints that nothing consumes. Each is an input or an intermediate shown for
# traceability, never a result -- a result that stopped matching would be the finding this exists for.
ALLOWED: dict[str, list[tuple[float, str]]] = {
    "tools/gen_glonass_oracle.mac": [
        (
            2.5495097567963924e7,
            "|r| of the test state, so the J2 factor can be checked by hand",
        ),
    ],
    "tools/gen_se3_oracle.mac": [
        (
            4.2678561945965885e-9,
            "the small-angle rotation angle: an input, below the 1e-8 branch",
        ),
        (1.3379088160259651, "the large-angle rotation angle: the other input"),
    ],
}

GENERATORS = [
    "derivations/csa_fm_rate.mac",
    "derivations/csa_phases.mac",
    "derivations/geopotential.mac",
    "derivations/src_fm_rate.mac",
    "tools/gen_fundarg_oracle.mac",
    "tools/gen_glonass_oracle.mac",
    "tools/gen_se3_oracle.mac",
    "tools/gen_shbasis_oracle.mac",
    "tools/gen_slr_vapour_oracle.mac",
]

SOURCE_SUFFIXES = {".rs", ".py", ".c", ".cpp", ".h", ".hpp", ".md", ".txt"}
# A Maxima bigfloat prints as 1.234...b5, a float as 1.234...e5, and a Rust literal as 0.001234 or
# 1.234e-3. Twelve fractional digits is far more than a hand-typed tolerance and far less than f64.
NUMBER = re.compile(r"[-+]?\d+\.\d{12,}(?:[eEbB][-+]?\d+)?")
# Anything in the tree that could be a pasted constant. Rust literals carry rustfmt's digit
# separators -- the reference value above is stored as `-9.222_615_065_805_873e-1` -- so a regex
# without `_` finds nothing and reports every generator stale.
LITERAL = re.compile(r"[-+]?\d[\d_]*\.\d[\d_]*(?:[eE][-+]?\d[\d_]*)?")
RELATIVE = 1e-13


def value(token: str) -> float | None:
    try:
        return float(re.sub(r"[bB]", "e", token.replace("_", "")))
    except ValueError:
        return None


def tree_literals() -> list[float]:
    found: set[float] = set()
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        # This file quotes a reference value in a comment; a scanner that reads itself would
        # then hold every value it names, and the check would pass on its own prose.
        if "target" in path.parts or ".git" in path.parts or path == SELF:
            continue
        for token in LITERAL.findall(path.read_text(errors="ignore")):
            v = value(token)
            if v is not None:
                found.add(abs(v))
    return sorted(found)


def held(needle: float, sorted_literals: list[float]) -> bool:
    """Is some literal in the tree equal to `needle` to within `RELATIVE`?"""
    target = abs(needle)
    window = target * RELATIVE
    lo = bisect.bisect_left(sorted_literals, target - window)
    hi = bisect.bisect_right(sorted_literals, target + window)
    return hi > lo


def main() -> int:
    literals = tree_literals()
    missing: list[tuple[str, str]] = []
    checked = 0

    for name in GENERATORS:
        run = subprocess.run(
            [
                "timeout",
                "300",
                "maxima",
                "--very-quiet",
                "--batch-string",
                f'batchload("{name}")$',
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
        if run.returncode != 0:
            missing.append((name, f"the generator exited {run.returncode}"))
            continue
        allowed = ALLOWED.get(name, [])
        # Maxima wraps a long value across lines with a trailing backslash.
        output = run.stdout.replace("\\\n", "")
        for token in dict.fromkeys(NUMBER.findall(output)):
            v = value(token)
            if v is None or v == 0.0:
                continue
            if any(abs(v - a) <= abs(a) * RELATIVE for a, _ in allowed):
                continue
            checked += 1
            if not held(v, literals):
                missing.append((name, token))

    print(f"{checked} reference values checked across {len(GENERATORS)} generators")
    for name, token in missing:
        print(f"  MISSING  {name}: {token}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
