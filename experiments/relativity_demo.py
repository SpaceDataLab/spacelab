#!/usr/bin/env python
"""IERS 2010 relativistic corrections as an orbit perturbation — spacelab now models all three terms.

Propagate a LEO for a day under point-mass gravity plus the three post-Newtonian accelerations
(Schwarzschild + Lense-Thirring + de Sitter, IERS Conventions 2010 Eq. 10.12) and compare against the
exact two-body (Kepler) motion. The difference is the relativistic drift that precise orbit
determination must model; the Schwarzschild term dominates (the relativistic perigee precession).

    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/relativity_demo.py
"""

from __future__ import annotations

import numpy as np
import spacedatalab as sl


def main() -> None:
    mu = 3.986_004_418e14
    r0 = [6.9e6, 1.2e6, 2.0e6]
    v0 = [-1_500.0, 6_500.0, 3_600.0]  # mildly eccentric, so the perigee precession is visible
    day = 86_400.0
    ej2000 = 8765.5 * 86_400.0 + 69.184  # 2024-01-01, J2000 seconds for the analytic Sun

    rel = sl.propagate_relativistic_dense(r0, v0, day, ej2000, mu=mu, tol=1e-11)
    ts = np.linspace(0.0, day, 49)
    drift = np.array(
        [
            np.linalg.norm(
                np.array(rel.eval(float(t))[0]) - np.array(sl.kepler_propagate(r0, v0, mu, float(t))[0])
            )
            for t in ts
        ]
    )
    print("relativistic (Schwarzschild + Lense-Thirring + de Sitter) vs two-body (LEO, 1 day):")
    print(f"  general relativity displaces the orbit by up to {drift.max():.2f} m over the day")
    assert drift.max() > 0.1, "GR should move a LEO orbit by well over a decimetre per day"


if __name__ == "__main__":
    main()
