#!/usr/bin/env python
"""Two SpaceDataLab differentiators, exercised through the Python bridge.

1. The rigorous IAU 2006/2000A CIO frame (`gcrf_to_itrf`) versus pyerfa's identical pipeline.
2. Switchable f64 / double-double precision (`propagate_two_body`): the same symplectic method, one in
   f64 and one in ~32-digit double-double, scored against the exact circular-orbit return.

    uv run --no-project --with /tmp/w/*.whl --with numpy --with pyerfa python experiments/precision_and_frames.py
"""

from __future__ import annotations

import math

import erfa
import numpy as np
import spacedatalab as sl

J2000_JD = 2451545.0


def ns_since_j2000(days: float) -> int:
    return round(days * 86_400.0 * 1e9)


def frame_check() -> None:
    tt_days = 8766.0  # ~2024
    ut1_days = 8766.0  # dUT1 = 0 for this check
    r_gcrf = [7.0e6, 1.0e6, 2.0e6]
    v_gcrf = [0.0, 7000.0, 1000.0]

    r_itrf, _ = sl.gcrf_to_itrf(
        r_gcrf, v_gcrf, ns_since_j2000(ut1_days), ns_since_j2000(tt_days), 0.0, 0.0, 0.0, 0.0, 0.0
    )

    x, y = erfa.xy06(J2000_JD, tt_days)
    s = erfa.s06(J2000_JD, tt_days, x, y)
    rc2i = erfa.c2ixys(x, y, s)
    era = erfa.era00(J2000_JD, ut1_days)
    sp = erfa.sp00(J2000_JD, tt_days)
    rpom = erfa.pom00(0.0, 0.0, sp)
    rc2t = erfa.c2tcio(rc2i, era, rpom)
    r_ref = rc2t @ np.array(r_gcrf)

    err = float(np.linalg.norm(np.array(r_itrf) - r_ref))
    print(f"[1] gcrf_to_itrf vs pyerfa CIO pipeline: {err:.2e} m  (position at 7300 km radius)")
    assert err < 1e-6, f"frame disagreement {err:.2e} m too large"


def precision_check() -> None:
    mu = 3.986_004_418e14
    r0 = 7.0e6
    vc = math.sqrt(mu / r0)
    period = 2.0 * math.pi * math.sqrt(r0**3 / mu)
    n, revs = 1024, 15
    dt = period / n
    steps = n * revs

    start = [r0, 0.0, 0.0]
    r_f, _ = sl.propagate_two_body(start, [0.0, vc, 0.0], mu, dt, steps, precision="f64")
    r_d, _ = sl.propagate_two_body(start, [0.0, vc, 0.0], mu, dt, steps, precision="dd")

    err_f = float(np.linalg.norm(np.array(r_f) - start))
    err_d = float(np.linalg.norm(np.array(r_d) - start))
    print(
        f"[2] two-body return, {revs} revs: f64 = {err_f:.3e} m, dd = {err_d:.3e} m"
        f"  ({err_f / err_d:.0f}x tighter)"
    )
    assert err_d * 5.0 < err_f, "dd should clearly beat f64"


def geopotential_frame_check() -> None:
    tt_days = 8766.0  # ~2024
    ut1_days = 8766.0
    era = float(erfa.era00(J2000_JD, ut1_days))  # Earth-rotation angle at the epoch
    r0 = [6.9e6, 1.2e6, 2.0e6]
    v0 = [-1500.0, 6000.0, 3500.0]
    day = 86_400.0

    # EarthFixed with theta0 = ERA(epoch) matches the rotation *angle* at t=0, so the only remaining
    # difference from the CIO chain is the pole *axis*: fixed GCRF-z versus the precessing/nutating CIP.
    eph_fixed = sl.propagate_geopotential_dense(r0, v0, day, 8, theta0=era, tol=1e-11)
    eph_itrf = sl.propagate_geopotential_itrf_dense(
        r0, v0, day, 8, ns_since_j2000(tt_days), ns_since_j2000(ut1_days), tol=1e-11
    )
    rf = np.array(eph_fixed.eval(day)[0])
    ri = np.array(eph_itrf.eval(day)[0])
    gap = float(np.linalg.norm(rf - ri))
    print(f"[3] geopotential (deg 8) over 1 day, fixed-axis vs rigorous ITRF: {gap:.1f} m apart")
    print("    (the accumulated precession + nutation the CIO frame tracks and a fixed axis drops)")


if __name__ == "__main__":
    frame_check()
    precision_check()
    geopotential_frame_check()
    print("OK — rigorous CIO frame, switchable f64/dd precision, and true-ITRF geopotential all live.")
