#!/usr/bin/env -S uv run --quiet --with sgp4 python
"""Fixtures for the official SGP4 verification cases (AIAA 2006-6753).

Run:
    ./derivations/sgp4_verification.py

Provenance
----------
The element sets and the reference states come from `python-sgp4` (MIT), which ships
`SGP4-VER.TLE` and `tcppver.out`. Both are numbers; numbers are facts.

`python-sgp4` ships a compiled extension built from Vallado's own ``SGP4.cpp`` — the
companion code to AIAA 2006-6753 — and ``sgp4.api.Satrec`` dispatches to it whenever
it imports. So on an accelerated install these reference states are produced by
Vallado's C++, not by a Python transcription of it. The header of the states file
records which, because the two are not the same claim.

The Rust ``sgp4`` crate this workspace wraps is itself a *port* of that same Celestrak
source. Agreement therefore establishes that the transcription is faithful; it does
not establish independent confirmation, and nothing can. Every SGP4 implementation in
circulation descends from the same listing: SGP4 is a model whose every quirk *is* its
specification, and a "better" implementation is a wrong one.
"""

from __future__ import annotations

import pathlib
import sys

import sgp4
from sgp4.api import WGS72, Satrec, accelerated

OUT = pathlib.Path(__file__).parent.parent / "crates/spacelab-sgp4/tests/fixtures"

# Which implementation actually produced the numbers below. `accelerated` is True when
# python-sgp4's compiled Vallado C++ extension imported and `Satrec` dispatches to it;
# False when the pure-Python transcription in `sgp4/propagation.py` ran instead. The
# fixtures record this because "the reference implementation" names two different
# programs depending on the answer.
PROVENANCE = (
    f"python-sgp4 {sgp4.__version__} (MIT), WGS72, "
    f"{'Vallado C++ extension' if accelerated else 'pure-Python transcription'}"
)


def cases() -> list[tuple[str, str, float, float, float]]:
    """Element sets, with the propagation span the published file attaches to each."""
    lines = (
        (pathlib.Path(sgp4.__file__).parent / "SGP4-VER.TLE").read_text().splitlines()
    )
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("1 "):
            l1, l2 = lines[i], lines[i + 1]
            tail = l2[69:].split()
            start, stop, step = (
                map(float, tail[:3]) if len(tail) >= 3 else (0.0, 0.0, 1.0)
            )
            out.append((l1[:69], l2[:69], start, stop, step))
            i += 2
        else:
            i += 1
    return out


def main() -> int:
    all_cases = cases()
    print(f"{len(all_cases)} published verification element sets")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sgp4_ver_tle.txt").open("w") as fh:
        fh.write(
            "# Official SGP4 verification element sets (AIAA 2006-6753), via python-sgp4 (MIT).\n"
        )
        fh.write("# line1|line2|start_min|stop_min|step_min\n")
        fh.write(f"# reference: {PROVENANCE}\n")
        for l1, l2, a, b, c in all_cases:
            fh.write(f"{l1}|{l2}|{a}|{b}|{c}\n")

    rows, codes = [], {}
    for l1, l2, start, stop, step in all_cases:
        sat = Satrec.twoline2rv(l1, l2, WGS72)
        t, guard = start, 0
        while (step > 0 and t <= stop + 1e-9) or (step < 0 and t >= stop - 1e-9):
            error, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF + t / 1440.0)
            rows.append((sat.satnum, t, error, *r, *v))
            codes[error] = codes.get(error, 0) + 1
            t += step
            guard += 1
            if step == 0.0 or guard > 5000:
                break

    with (OUT / "sgp4_ver_states.csv").open("w") as fh:
        fh.write(f"# norad,minutes,error,rx_km,ry_km,rz_km,vx,vy,vz -- {PROVENANCE}\n")
        for row in rows:
            fh.write(
                ",".join(f"{x:.17g}" if isinstance(x, float) else str(x) for x in row)
                + "\n"
            )

    print(f"{len(rows)} reference states; error codes {codes}")
    print("  1 = eccentricity out of range   3 = perturbed elements out of range")
    print("  4 = semi-latus rectum negative  6 = decayed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
