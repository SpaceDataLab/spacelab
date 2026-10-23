#!/usr/bin/env python
"""Brouwer-Lyddane M2 (empirical along-track term) fitted against a first-principles numerical orbit.

The Brouwer-Lyddane theory with M2 = 0 has a known in-track drift (the orbit *plane* is exact but the
phase leads/lags) — an independent numerical J2-J5 propagation reveals ~tens of km/day. Fitting the
single empirical M2 coefficient (rad/s², like a TLE B*) against that truth collapses the error to the
theory's residual, confirming the M2 path works and is useful. This is an Orekit-independent check.

    uv run --no-project --with /tmp/w/*.whl --with numpy --with scipy \
        python experiments/brouwer_lyddane_m2_fit.py
"""

from __future__ import annotations

import numpy as np
import spacelab as sl
from scipy.integrate import solve_ivp

MU = 3.986_004_415e14
RE = 6_378_136.3
JN = [1.08262668e-3, -2.53265649e-6, -1.61962159e-6, -2.27296083e-7]  # J2..J5
C = [-j for j in JN]  # c_n0 = -J_n
R0 = [4_485_195.611_065_954, 3_180_300.412_195_459, 4_537_882.996_315_523]
V0 = [-3_045.222_561_050_032, -3_821.479_786_492_346, 5_708.385_836_216_403]
DAY = 86_400.0


def _legendre(u: float, n: int) -> np.ndarray:
    p = np.zeros(n + 1)
    p[0], p[1] = 1.0, u
    for k in range(2, n + 1):
        p[k] = ((2 * k - 1) * u * p[k - 1] - (k - 1) * p[k - 2]) / k
    return p


def _accel(r: np.ndarray) -> np.ndarray:
    def v(rr: np.ndarray) -> float:
        rn = np.linalg.norm(rr)
        pl = _legendre(rr[2] / rn, 5)
        return MU / rn * (1 - sum(JN[n - 2] * (RE / rn) ** n * pl[n] for n in range(2, 6)))

    h, a = 10.0, np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        a[i] = (-v(r + 2 * e) + 8 * v(r + e) - 8 * v(r - e) + v(r - 2 * e)) / (12 * h)
    return a


def main() -> None:
    truth = solve_ivp(
        lambda t, y: np.concatenate([y[3:], _accel(y[:3])]),
        [0, DAY], np.array(R0 + V0), method="DOP853", rtol=1e-12, atol=1e-6,
    ).y[:3, -1]

    def err(m2: float) -> float:
        r = sl.brouwer_lyddane_propagate(R0, V0, [DAY], MU, RE, *C, m2)[0][0]
        return float(np.linalg.norm(np.array(r) - truth))

    # M2 shifts the along-track linearly, so a two-point fit gives the zero-crossing.
    e0, e1 = err(0.0), err(1e-13)
    m2_fit = -1e-13 * e0 / (e1 - e0)

    print("Brouwer-Lyddane M2 fit against numerical J2-J5 truth (1 day, Orekit-independent):")
    print(f"  M2 = 0            -> |Δr| = {e0 / 1e3:8.2f} km  (known in-track drift)")
    print(f"  M2 = {m2_fit:.3e} -> |Δr| = {err(m2_fit):8.2f} m   (fitted; along-track absorbed)")
    assert err(m2_fit) < 0.5 * e0, "fitting M2 must substantially reduce the along-track error"


if __name__ == "__main__":
    main()
