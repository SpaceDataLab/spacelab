#!/usr/bin/env python
"""Brouwer-Lyddane analytical propagation — fast mean-element J2-J5 propagation, no integration.

Propagate a LEO for a day with the Brouwer-Lyddane theory (validated against Orekit to ~micron
position / nm/s velocity) and show the J2 signatures it captures in closed form: the short-period
altitude oscillation and the secular regression of the right ascension of the ascending node.

    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/brouwer_lyddane_demo.py
"""

from __future__ import annotations

import numpy as np
import spacelab as sl


def raan_deg(r: np.ndarray, v: np.ndarray) -> float:
    h = np.cross(r, v)  # specific angular momentum; the node lies along ẑ × h
    return float(np.degrees(np.arctan2(h[0], -h[1])))


def main() -> None:
    # A ~98° inclined LEO (a ≈ 7200 km, e ≈ 0.01) — the same osculating state used to validate the port.
    r0 = [4_485_195.611_065_954, 3_180_300.412_195_459, 4_537_882.996_315_523]
    v0 = [-3_045.222_561_050_032, -3_821.479_786_492_346, 5_708.385_836_216_403]
    day = 86_400.0
    ts = np.linspace(0.0, day, 289)  # every 5 minutes

    states = sl.brouwer_lyddane_propagate(r0, v0, [float(t) for t in ts])
    rs = np.array([s[0] for s in states])
    vs = np.array([s[1] for s in states])

    alt = np.linalg.norm(rs, axis=1) / 1e3 - 6378.137  # spherical altitude, km
    node0 = raan_deg(rs[0], vs[0])
    node1 = raan_deg(rs[-1], vs[-1])
    dnode = ((node1 - node0 + 180.0) % 360.0) - 180.0

    print("Brouwer-Lyddane analytical propagation (zonal J2-J5, LEO, 1 day, no integration):")
    print(f"  spherical altitude oscillates {alt.min():.1f}-{alt.max():.1f} km (J2 short-period)")
    print(f"  RAAN regresses {dnode:+.3f} deg over the day (J2 secular nodal drift)")
    assert abs(dnode) > 0.1, "J2 should drift the node by more than 0.1 deg/day for this orbit"


if __name__ == "__main__":
    main()
