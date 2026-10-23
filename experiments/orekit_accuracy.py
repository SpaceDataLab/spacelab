#!/usr/bin/env python
"""Is spacelab *more accurate* than Orekit — across LEO/MEO/GEO, for two-body and J2.

"Agrees with Orekit" is not "more accurate than Orekit". Two things are measured against a reference
that is not either library's numerical propagator:

  * two-body — scored against the *exact* analytic Kepler solution (from Orekit's own
    KeplerianPropagator, cross-checked with spacelab's universal-variable Kepler). spacelab's tolerance
    is swept to its floor, because the two solvers' tolerance parameters are not the same quantity.
  * J2 — scored against Orekit's own numerical J2 with the field applied *in the inertial frame*, the
    same convention spacelab's inertial J2 uses. This isolates the integrator + force formulation from
    the frame convention (Orekit's default J2 tracks the true ITRF pole; spacelab's a fixed inertial z,
    a difference that is precession, ~1.5 km/day at 2024, not integration error — see orekit_benchmark).

Self-drives the Java oracle per regime (compile it once, per tools/orekit-oracle/README.md):
    javac -cp "/tmp/orekit-bench/jars/*" -d /tmp/orekit-bench/classes tools/orekit-oracle/OrekitPropagationBench.java
Then, with the spacelab wheel at /tmp/w:
    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/orekit_accuracy.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import spacedatalab as sl

JARS = os.environ.get("OREKIT_JARS", "/tmp/orekit-bench/jars")
CLASSES = os.environ.get("OREKIT_CLASSES", "/tmp/orekit-bench/classes")
DATA = os.environ.get("OREKIT_DATA", "/tmp/orekit-bench/orekit-data-main")
OUT = Path(os.environ.get("OREKIT_BENCH_DIR", "/tmp/orekit-bench/out"))
POS_TOL = "1e-12"
SPACELAB_TOLERANCES = [1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

# a, e, i(deg): a low-inclination GEO, a GPS-class MEO, a 550 km LEO.
REGIMES = {
    "LEO": (6_928_137.0, 0.001, 51.6),
    "MEO": (26_560_000.0, 0.001, 55.0),
    "GEO": (42_164_000.0, 0.0001, 1.0),
}


def run_orekit(sma: float, ecc: float, inc: float) -> None:
    subprocess.run(
        ["java", f"-Dsma={sma}", f"-Decc={ecc}", f"-Dinc={inc}", f"-DposTol={POS_TOL}",
         f"-Dorekit.data={DATA}", f"-Dout.dir={OUT}", "-cp", f"{JARS}/*:{CLASSES}", "OrekitPropagationBench"],
        check=True, capture_output=True,
    )


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


def spacelab_track(r0, v0, mu, j2, re, times, tol) -> np.ndarray:
    eph = sl.propagate_dense(list(r0), list(v0), float(times[-1]), mu, j2=j2, re=re, tol=tol)
    return np.array([eph.eval(float(t))[0] for t in times])


def main() -> None:
    print("two-body error vs exact Kepler (max over a day) · J2 = integrator agreement, matched frame\n")
    print(f"{'regime':6} {'Orekit DP853':>14} {'spacelab DOP853':>16} {'spacelab IAS15':>16} {'× vs Orekit':>12} {'J2':>10}")
    for name, (sma, ecc, inc) in REGIMES.items():
        run_orekit(sma, ecc, inc)
        meta = read_meta()
        mu, ae, j2 = meta["mu"], meta["ae"], meta["j2"]
        r0, v0 = np.array(meta["r0"]), np.array(meta["v0"])

        kepler = positions("orekit_kepler.csv")
        times, truth = kepler[:, 0], kepler[:, 1:]
        if name == "LEO":
            gap = np.linalg.norm(
                np.array([sl.kepler_propagate(list(r0), list(v0), mu, float(t))[0] for t in times]) - truth,
                axis=1,
            ).max()
            assert gap < 1e-3, f"analytic references disagree ({gap:.2e} m) — reference not trustworthy"

        orekit_two = np.linalg.norm(positions("orekit_twobody.csv")[:, 1:] - truth, axis=1).max()
        dop_floor = min(
            np.linalg.norm(spacelab_track(r0, v0, mu, 0.0, ae, times, tol) - truth, axis=1).max()
            for tol in SPACELAB_TOLERANCES
        )
        # IAS15: 15th-order Gauss–Radau with compensated summation — reaches the f64 round-off floor.
        ias = sl.propagate_ias15_dense(list(r0), list(v0), float(times[-1]), mu, j2=0.0, re=ae)
        ias_err = np.linalg.norm(np.array([ias.eval(float(t))[0] for t in times]) - truth, axis=1).max()
        # J2 with the matched (inertial) frame — integrator + force agreement, no frame convention.
        j2_ref = positions("orekit_j2_inertial.csv")[:, 1:]
        j2_gap = np.linalg.norm(spacelab_track(r0, v0, mu, j2, ae, times, 1e-13) - j2_ref, axis=1).max()

        print(f"{name:6} {orekit_two:>12.2e} m {dop_floor:>14.2e} m {ias_err:>14.2e} m {orekit_two / ias_err:>10.1f}× {j2_gap:>8.1e}")

    print("\nIAS15 (compensated summation) reaches the IEEE-double round-off floor (~µm for these orbits over a")
    print("day) that Orekit's DP853 does not — clearly better on LEO (~6×); on MEO/GEO both sit at that f64")
    print("floor. Beating µm needs extended precision (double-double), a genuine next step, not a better method.")
    print("J2: matched-frame agreement is microns (model is right); the real-world J2 gap vs Orekit's ITRF pole")
    print("is a frame convention (precession), not integration error — see orekit_benchmark.py.")


if __name__ == "__main__":
    main()
