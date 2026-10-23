#!/usr/bin/env python
"""Reference GCRF→ITRF matrices for the IAU 2006/2000A CIO chain, from pyerfa.

Builds the GCRS→ITRS matrix through exactly the pipeline `spacelab-frames::gcrf_to_itrf`
implements — CIP X,Y from the `xy06` *series*, CIO locator `s06`, `c2ixys`, ERA, TIO
locator `sp00`, polar motion `pom00`, assembled by `c2tcio` — using pyerfa (ERFA,
BSD-3-Clause). UT1 and TT are deliberately offset so the fixture would catch either
being used for the other.

Run:
    timeout 60 uv run --no-project --with pyerfa --with numpy python derivations/cio_frame.py
"""

from __future__ import annotations

import math
from pathlib import Path

import erfa

J2000 = 2451545.0
DAS2R = math.pi / 648_000.0
DMAS2R = DAS2R / 1000.0
OUT = (
    Path(__file__).resolve().parent.parent
    / "crates/spacelab-frames/tests/fixtures/cio_gcrf_itrf_pyerfa.csv"
)

# (tt_days_since_j2000, ut1_days_since_j2000, xp_arcsec, yp_arcsec, dx_mas, dy_mas).
# UT1 offset from TT by a visible amount on purpose.
#
# The last four carry celestial-pole offsets. Without them dX and dY enter the transform
# only as `+0`, where a sign error, a swapped pair and a dropped term are all the same
# matrix; the values are the size IERS finals2000A actually reports (|dX|, |dY| < 0.5 mas)
# and one deliberately larger pair, and they are asymmetric in magnitude and opposite in
# sign so that swapping dX for dY is a different matrix too.
CASES = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (2191.5, 2191.5, 0.052_600, 0.383_800, 0.0, 0.0),  # SOFA c2t06a epoch
    (8766.0, 8765.6, 0.161_000, 0.301_000, 0.0, 0.0),  # ~2024
    (18262.0, 18261.55, -0.052_000, 0.244_000, 0.0, 0.0),  # ~2050
    (-3652.0, -3652.4, 0.210_000, 0.120_000, 0.0, 0.0),  # ~1990
    (5478.0, 5477.7, 0.098_000, 0.412_000, 0.176_0, -0.104_0),  # ~2015, IERS-sized
    (8766.0, 8765.6, 0.161_000, 0.301_000, -0.263_0, 0.081_0),  # ~2024, signs flipped
    (
        2191.5,
        2191.5,
        0.052_600,
        0.383_800,
        12.0,
        -7.0,
    ),  # exaggerated, resolves the sign
    (0.0, 0.0, 0.0, 0.0, 0.0, 5.0),  # dY alone: dX = 0 cannot mask a swap
]


def main() -> None:
    rows = []
    for tt_d, ut1_d, xp_as, yp_as, dx_mas, dy_mas in CASES:
        x, y = erfa.xy06(J2000, tt_d)
        s = erfa.s06(J2000, tt_d, x, y)
        # The celestial-pole offsets correct the CIP coordinates and leave the CIO
        # locator at its model value — IERS Conventions 2010 §5.5.4, and what `c2ixys`
        # is given by `c2t06a`'s caller. Adding them here rather than to `s` is the
        # distinction the fixture pins.
        rc2i = erfa.c2ixys(x + dx_mas * DMAS2R, y + dy_mas * DMAS2R, s)
        era = erfa.era00(J2000, ut1_d)
        sp = erfa.sp00(J2000, tt_d)
        rpom = erfa.pom00(xp_as * DAS2R, yp_as * DAS2R, sp)
        rc2t = erfa.c2tcio(rc2i, era, rpom)  # GCRS -> ITRS, 3x3
        m = [float(v) for v in rc2t.reshape(9)]
        rows.append([tt_d, ut1_d, xp_as, yp_as, dx_mas, dy_mas, *m])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write(
            "# tt_days,ut1_days,xp_arcsec,yp_arcsec,dx_mas,dy_mas,"
            "m00,m01,m02,m10,m11,m12,m20,m21,m22\n"
        )
        f.write(
            "# GCRF->ITRF, IAU 2006/2000A CIO (xy06 series) pipeline, pyerfa (ERFA, BSD)\n"
        )
        for r in rows:
            f.write(",".join(f"{v:.17e}" for v in r) + "\n")
    print(f"wrote {OUT} with {len(rows)} epochs")


if __name__ == "__main__":
    main()
