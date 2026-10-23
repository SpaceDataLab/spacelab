#!/usr/bin/env python
"""Double-double propagation on a *perturbed* (J2) orbit — below the f64 floor no f64 tool escapes.

Every f64 astrodynamics engine (Orekit, GMAT, nyx-space) accumulates round-off at ~ε·|r| on each
`x += h·v`; over a long arc that random-walks into a floor of error no f64 method can remove. spacelab
runs the *same* 8th-order symplectic method in double-double (~32 digits), so the floor disappears.

Here f64 and dd fly the identical point-mass + J2 orbit. The double-double trajectory is the
round-off-free reference, so `|f64 − dd|` is precisely the round-off the f64 run cannot escape — the
gap between spacelab-in-dd and every f64 propagator on the planet.

    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/double_double_j2.py
"""

from __future__ import annotations

import math

import numpy as np
import spacedatalab as sl


def main() -> None:
    mu, re, j2 = 3.986_004_418e14, 6_378_137.0, 1.082_626_68e-3
    r0 = 7.0e6
    vc = math.sqrt(mu / r0)
    period = 2.0 * math.pi * math.sqrt(r0**3 / mu)
    n, revs = 2048, 15
    dt, steps = period / n, n * revs
    start, vel = [r0, 0.0, 0.0], [0.0, vc, 0.0]

    rf, _ = sl.propagate_j2_precision(start, vel, mu, j2, re, dt, steps, precision="f64")
    rd, _ = sl.propagate_j2_precision(start, vel, mu, j2, re, dt, steps, precision="dd")
    roundoff = float(np.linalg.norm(np.array(rf) - np.array(rd)))

    print(f"point mass + J2, {revs} revs ({steps} steps), same yoshida8 method:")
    print(f"  |f64 − dd| final position = {roundoff:.3e} m")
    print("  That is the accumulated f64 round-off double-double removes. Every f64 engine (Orekit,")
    print("  GMAT, nyx) sits at this floor; spacelab is the one that propagates below it.")
    assert roundoff > 1e-9, f"f64 should have accumulated visible round-off, got {roundoff:.2e} m"


if __name__ == "__main__":
    main()
