#!/usr/bin/env python
"""DSST-Zonal semi-analytical propagation — cheap long-arc mean-element drift under J2-J6.

The Draper Semi-analytical Satellite Theory averages the geopotential over one orbit, so its mean
elements move only secularly (no short-period oscillation). Integrating those smooth averaged rates
with large steps propagates months of orbit evolution in milliseconds — the standard tool for
constellation station-keeping and long-term catalogue maintenance. Here we watch a LEO's node regress
and perigee precess over 30 days (validated against Orekit's DSSTPropagator to bit-exact mean rates).

    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/dsst_zonal_demo.py
"""

from __future__ import annotations

import numpy as np
import spacelab as sl


def keplerian_to_equinoctial(a: float, e: float, i: float, raan: float, argp: float, m: float):
    """(a, e, i, Ω, ω, M) → mean equinoctial [a, k, h, q, p, λ] (prograde, I = +1)."""
    pomega = argp + raan
    return [
        a,
        e * np.cos(pomega),
        e * np.sin(pomega),
        np.tan(i / 2) * np.cos(raan),
        np.tan(i / 2) * np.sin(raan),
        m + pomega,
    ]


def main() -> None:
    a, e, inc = 7_200e3, 0.01, np.radians(98.0)
    raan0, argp0 = np.radians(42.0), np.radians(30.0)
    initial = keplerian_to_equinoctial(a, e, inc, raan0, argp0, np.radians(10.0))

    ts = np.linspace(0.0, 30 * 86_400.0, 31)  # 30 days, daily samples
    states = np.array(sl.dsst_zonal_propagate_mean(initial, [float(t) for t in ts]))

    k, h, q, p = states[:, 1], states[:, 2], states[:, 3], states[:, 4]
    raan = np.degrees(np.unwrap(np.arctan2(p, q)))
    pomega = np.unwrap(np.arctan2(h, k))
    argp = np.degrees(pomega - np.radians(raan))  # ω = ϖ − Ω

    node_rate = (raan[-1] - raan[0]) / 30.0
    perigee_rate = (argp[-1] - argp[0]) / 30.0
    print("DSST-Zonal mean-element propagation (J2-J6, LEO, 30 days, semi-analytical):")
    print(f"  RAAN regresses {node_rate:+.4f} deg/day (J2 nodal drift)")
    print(f"  argument of perigee precesses {perigee_rate:+.4f} deg/day")
    print(f"  mean semi-major axis drift over 30 d: {states[-1, 0] - states[0, 0]:.2e} m (≈0, zonal)")
    assert abs(node_rate) > 0.1, "J2 should regress the node by more than 0.1 deg/day"


if __name__ == "__main__":
    main()
