#!/usr/bin/env python
"""IERS 2010 solid Earth tide as an orbit perturbation — spacelab now models it (an Orekit-parity force).

Propagate a LEO for a day under the tesseral geopotential in the rigorous ITRF, once with and once
without the Sun/Moon-raised solid Earth tide, and report how far the tide moves the orbit. The tide is a
real centimetre-to-decimetre perturbation that precise orbit determination and geodesy must model.

    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/solid_tide_demo.py
"""

from __future__ import annotations

import numpy as np
import spacelab as sl


def main() -> None:
    r0 = [6.9e6, 1.2e6, 2.0e6]
    v0 = [-1500.0, 6000.0, 3500.0]
    day = 86_400.0
    naive_ns = round(8765.5 * 86_400.0 * 1e9)  # 2024-01-01T00:00:00 UTC
    tt_ns = naive_ns + round(69.184 * 1e9)
    ut1_ns = naive_ns + round(0.008757 * 1e9)
    ej2000 = 8765.5 * 86_400.0 + 69.184  # epoch in J2000 seconds for the analytic Sun/Moon

    with_tide = sl.propagate_geopotential_tide_dense(r0, v0, day, 8, tt_ns, ut1_ns, ej2000, tol=1e-11)
    without = sl.propagate_geopotential_itrf_dense(r0, v0, day, 8, tt_ns, ut1_ns, tol=1e-11)

    ts = np.linspace(0.0, day, 49)
    disp = np.array(
        [
            np.linalg.norm(np.array(with_tide.eval(float(t))[0]) - np.array(without.eval(float(t))[0]))
            for t in ts
        ]
    )
    print("solid Earth tide vs no tide (geopotential deg 8, LEO, 1 day):")
    print(f"  the tide displaces the orbit by up to {disp.max() * 100:.1f} cm over the day")
    assert disp.max() > 0.01, "the solid Earth tide should move the orbit by centimetres"


if __name__ == "__main__":
    main()
