#!/usr/bin/env python
"""Does the rigorous CIO frame close the J2/ITRF gap against Orekit?

Earlier (orekit_benchmark.py) spacelab's fixed-axis J2 disagreed with Orekit's true-ITRF J2 by 1476 m
over a day — the accumulated pole precession, a frame convention, not integration error. With the
IAU 2006/2000A CIO frame now built and wired into the geopotential (BodyOriented), spacelab can fly the
*same* frame Orekit does. Feeding it Orekit's own EOP (xp, yp, dUT1) and degree-2 constants, the gap
should collapse to the integrator floor.

Compile the oracle once (it now emits the EOP it used):
    javac -cp "/tmp/orekit-bench/jars/*" -d /tmp/orekit-bench/classes tools/orekit-oracle/OrekitPropagationBench.java
    java -Dorekit.data=... -Dout.dir=... -cp "...:/tmp/orekit-bench/classes" OrekitPropagationBench
Then, with the spacelab wheel at /tmp/w:
    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/orekit_itrf_geopotential.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import spacedatalab as sl

OUT = Path(os.environ.get("OREKIT_BENCH_DIR", "/tmp/orekit-bench/out"))

# TAI - UTC at the 2024-01-01 epoch. The oracle's default epoch is far from any leap second, so the
# naive (leap-second-free) UTC->J2000 mapping is exact to the nanosecond here.
DAT_SECONDS = 37.0
TT_MINUS_TAI = 32.184
# 2024-01-01T00:00:00 UTC is 8765.5 uniform days after J2000 (2000-01-01T12:00:00).
NAIVE_NS = round(8765.5 * 86_400.0 * 1e9)


def read_meta() -> dict:
    meta: dict[str, object] = {}
    for line in (OUT / "orekit_meta.csv").read_text().splitlines():
        key, _, rest = line.partition(",")
        values = [float(x) for x in rest.split(",")]
        meta[key] = values[0] if len(values) == 1 else values
    return meta


def positions(name: str) -> np.ndarray:
    rows = (OUT / name).read_text().splitlines()[1:]
    return np.array([[float(x) for x in r.split(",")] for r in rows])


def main() -> None:
    m = read_meta()
    mu, ae, j2 = m["mu"], m["ae"], m["j2"]
    xp, yp, dut1, lod = m["xp_rad"], m["yp_rad"], m["dut1_s"], m["lod_s"]
    dx, dy = m["dx_rad"], m["dy_rad"]  # IERS celestial-pole offsets
    r0, v0 = list(m["r0"]), list(m["v0"])
    span = float(m["span_s"])

    epoch_tt_ns = NAIVE_NS + round((DAT_SECONDS + TT_MINUS_TAI) * 1e9)
    epoch_ut1_ns = NAIVE_NS + round(dut1 * 1e9)

    okj2 = positions("orekit_j2.csv")  # GCRF samples of Orekit's J2 in the true ITRF
    t, truth = okj2[:, 0], okj2[:, 1:]

    # spacelab J2 in the rigorous ITRF, Orekit's own mu/J2/Re and xp/yp/dUT1.
    eph = sl.propagate_j2_itrf_dense(
        r0, v0, span, mu, j2, ae, epoch_tt_ns, epoch_ut1_ns, xp, yp, dx, dy, lod, tol=1e-12
    )
    err_itrf = np.linalg.norm(
        np.array([eph.eval(float(x))[0] for x in t]) - truth, axis=1
    ).max()

    # The pre-CIO path: J2 about a fixed inertial z (propagate_dense) — the convention that left the gap.
    eph_fixed = sl.propagate_dense(r0, v0, span, mu, j2=j2, re=ae, tol=1e-12)
    err_fixed = np.linalg.norm(
        np.array([eph_fixed.eval(float(x))[0] for x in t]) - truth, axis=1
    ).max()

    # Micron parity: feed Orekit's own EOP series so the pole tracks it exactly over the day.
    eop = np.array(
        [[float(x) for x in r.split(",")] for r in (OUT / "orekit_eop.csv").read_text().splitlines()[1:]]
    )
    eph_eop = sl.propagate_j2_itrf_eop_dense(
        r0, v0, span, mu, j2, ae, NAIVE_NS, DAT_SECONDS + TT_MINUS_TAI,
        list(eop[:, 0]), list(eop[:, 1]), list(eop[:, 2]), list(eop[:, 3]), list(eop[:, 4]), list(eop[:, 5]),
        tol=1e-12,
    )
    err_eop = np.linalg.norm(np.array([eph_eop.eval(float(x))[0] for x in t]) - truth, axis=1).max()

    print("spacelab J2 vs Orekit J2, both in the true ITRF over a day (matched constants + EOP):")
    print(f"  fixed inertial z (pre-CIO):         {err_fixed:.3e} m")
    print(f"  CIO frame, constant epoch EOP:      {err_itrf:.3e} m")
    print(f"  CIO frame, time-varying EOP series: {err_eop:.3e} m")
    print(f"  interpolating the EOP tightens the residual {err_itrf / err_eop:,.0f}x, to Orekit parity")
    assert err_eop < 1e-4, f"time-varying-EOP J2 should match Orekit to microns, got {err_eop:.2e} m"


if __name__ == "__main__":
    main()
