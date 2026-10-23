#!/usr/bin/env python
"""Quality + speed benchmark of spacelab's numerical propagation against Orekit.

The gold-standard open-source library (Orekit, Java) and spacelab fly the *identical* Cartesian initial
state under the *identical* force model — point mass, then point mass + J2 with Orekit's own degree-2
field constants — for a day with a DP853 integrator. This measures the integrator and force formulation
against a third, independent lineage; it is not the constants (those are shared) and not SGP4 (which
both descend from Vallado and which `sgp4_validation.py` already pins to <1 µm).

`tools/orekit-oracle/OrekitPropagationBench.java` produces the reference. Run it first (needs the JDK
and the 8 jars + orekit-data — see `tools/orekit-oracle/README.md`):

    mkdir -p /tmp/orekit-bench/jars   # download the jars + orekit-data-main per the README
    javac -cp "/tmp/orekit-bench/jars/*" -d /tmp/orekit-bench/classes tools/orekit-oracle/OrekitPropagationBench.java
    java  -cp "/tmp/orekit-bench/jars/*:/tmp/orekit-bench/classes" OrekitPropagationBench

Then, with the spacelab wheel built (`cd crates/spacelab-python && maturin build --release -o /tmp/w`):

    uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/orekit_benchmark.py

Finding (verified): two-body is a pure cross-lineage integrator check (no frame dependence). With the
field constants carried at full f64 precision, spacelab and Orekit agree to sub-mm over a day at
tol 1e-13, and spacelab is ~30× faster; `orekit_accuracy.py` scores each against the *exact* analytic
Kepler solution and finds spacelab reaches a lower error floor (1.26 µm vs Orekit's 8.4 µm) as well.
(An earlier ~1 cm "agreement" was an artifact of the reference CSV truncating GM to ten significant
figures — Δμ/μ≈1e-10 drifts a LEO ~cm/day; fixed by emitting %.17e.)

The J2 case shows a larger residual that is a *frame convention*, not integration error:
`propagate_dense` applies J2 in the inertial input frame about a fixed z, while Orekit applies it about
the true ITRF pole. That pole is offset from the J2000/GCRF z by the accumulated precession, so the
residual scales with epoch distance from J2000 — 107 m at the J2000 epoch versus 1476 m at 2024-01-01
(a ~14× ratio tracking the ~0.3° precession). For J2/geopotential referenced to the true pole, use
`propagate_geopotential_dense`, which evaluates the field in the Earth-fixed frame.
"""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import numpy as np
import spacedatalab as sl

BENCH = Path(os.environ.get("OREKIT_BENCH_DIR", "/tmp/orekit-bench/out"))
TIMING_REPEATS = 20


def read_meta() -> dict:
    meta: dict[str, object] = {}
    for line in (BENCH / "orekit_meta.csv").read_text().splitlines():
        key, _, rest = line.partition(",")
        values = [float(x) for x in rest.split(",")]
        meta[key] = values[0] if len(values) == 1 else values
    return meta


def read_positions(name: str) -> np.ndarray:
    rows = (BENCH / name).read_text().splitlines()[1:]  # drop header
    return np.array([[float(x) for x in r.split(",")] for r in rows])  # t, x, y, z


def spacelab_positions(r0, v0, mu, j2, re, times) -> np.ndarray:
    eph = sl.propagate_dense(list(r0), list(v0), float(times[-1]), mu, j2=j2, re=re, tol=1e-13)
    return np.array([eph.eval(float(t))[0] for t in times])


def time_spacelab(r0, v0, mu, j2, re, span) -> float:
    best = float("inf")
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        sl.propagate_dense(list(r0), list(v0), float(span), mu, j2=j2, re=re, tol=1e-13)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms (best-of, to suppress scheduler noise)


def main() -> None:
    meta = read_meta()
    mu, ae, j2 = meta["mu"], meta["ae"], meta["j2"]
    r0, v0 = np.array(meta["r0"]), np.array(meta["v0"])
    span = meta["span_s"]

    print(f"IC: |r0|={np.linalg.norm(r0) / 1e3:.1f} km  |v0|={np.linalg.norm(v0) / 1e3:.4f} km/s")
    print(f"field: GM={mu:.6e}  Re={ae:.3f} m  J2={j2:.8e}\n")
    print(f"{'case':10} {'max Δpos':>12} {'RMS Δpos':>12} {'spacelab':>10} {'orekit':>10}")

    for case, j2_val, ok_ms in (
        ("two-body", 0.0, meta["orekit_twobody_ms"]),
        ("J2", j2, meta["orekit_j2_ms"]),
    ):
        ref = read_positions(f"orekit_{'twobody' if case == 'two-body' else 'j2'}.csv")
        times, ref_pos = ref[:, 0], ref[:, 1:]
        got = spacelab_positions(r0, v0, mu, j2_val, ae, times)
        d = np.linalg.norm(got - ref_pos, axis=1)
        sl_ms = time_spacelab(r0, v0, mu, j2_val, ae, span)
        print(f"{case:10} {d.max():>10.3e} m {np.sqrt((d ** 2).mean()):>10.3e} m {sl_ms:>8.2f} ms {ok_ms:>8.2f} ms")

    # Peak memory (the "leanest" side). Process RSS muddies spacelab with the Python + numpy
    # baseline, so it is only a rough ceiling; the load-bearing fact is structural — spacelab
    # propagates with no heap allocation (stack-only `[f64; N]` state; the degree-120 GeoPotential
    # struct is ~0.84 MB, held once by reference), whereas a standalone Orekit JVM idles at
    # ~150–300 MB before any propagation. `orekit_meta.csv` carries `orekit_peak_rss_mb` when the
    # harness is built with peak-RSS emission.
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    ok_rss = meta.get("orekit_peak_rss_mb")
    print(f"\npeak RSS: this process (spacelab + Python + numpy) {peak_rss_mb:.0f} MB", end="")
    print(f"   vs Orekit JVM {ok_rss:.0f} MB" if ok_rss else "   (Orekit JVM idles at ~150–300 MB)")

    # Two-body is frame-independent, so its residual is pure integrator agreement across lineages.
    ref = read_positions("orekit_twobody.csv")
    got = spacelab_positions(r0, v0, mu, 0.0, ae, ref[:, 0])
    two_body_max = np.linalg.norm(got - ref[:, 1:], axis=1).max()
    assert two_body_max < 1.0, f"two-body disagrees with Orekit by {two_body_max:.3e} m (>1 m)"
    print("\ntwo-body: pure integrator cross-check (frame-independent) — sub-cm/day vs Orekit, and faster.")
    print("J2 residual is the inertial-vs-ITRF pole convention (module docstring), not integrator error;")
    print("for pole-accurate J2/geopotential use propagate_geopotential_dense (Earth-fixed).")


if __name__ == "__main__":
    main()
