#!/usr/bin/env python
"""Phase-5 differentiator: the Rust access-window engine vs a naive fixed-cadence Python sampler.

The strategy doc's central claim (plans/00-strategy, edit #3) is that event detection — bracketing
sign changes of the coverage constraint and refining to milliseconds — beats "a Python script that
samples once a minute", which loses short passes and mis-times window edges by up to ±Δt. This
quantifies it: the Rust engine (via the spacelab bridge) is the reference; a naive NumPy sampler
that marks a cell covered at each fixed step is compared to it at several cadences.

The point is *accuracy*, not only speed: coarse sampling systematically under-counts coverage seconds
and misses whole passes; only as Δt → 0 does it converge to the event-detected truth — and it is not
faster there.

Build the wheel, then run with it on the path:
  cd crates/spacelab-python && maturin build --release -o /tmp/w
  uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/python_vs_rust.py
"""

from __future__ import annotations

import math
import time

import numpy as np
import spacedatalab as sl

RE = 6_378_137.0
MU = 3.986_004_418e14
WE = 7.292_115_146_706_979e-5  # Earth rotation rate, rad/s (matches the engine)
HORIZON = 86_400.0
MASK = math.radians(5.0)


def naive_coverage(centres, zeniths, a, inc, dt):
    """Fixed-cadence Python/NumPy coverage: per cell, coverage fraction and pass count at step `dt`."""
    t = np.arange(0.0, HORIZON, dt)
    n = math.sqrt(MU / a**3)
    u = n * t
    ex, ey, ez = a * np.cos(u), a * np.sin(u) * math.cos(inc), a * np.sin(u) * math.sin(inc)
    c, s = np.cos(WE * t), np.sin(WE * t)
    sat = np.stack([c * ex + s * ey, -s * ex + c * ey, ez], axis=1)  # ECEF, matches Mat3::rot3(ω t)·eci
    los = sat[None, :, :] - centres[:, None, :]  # (cell, time, 3)
    rng = np.linalg.norm(los, axis=2)
    sin_el = np.einsum("ctd,cd->ct", los, zeniths) / rng
    covered = sin_el >= math.sin(MASK)
    cov_frac = covered.mean(axis=1)
    passes = (covered[:, 1:] & ~covered[:, :-1]).sum(axis=1) + covered[:, 0].astype(int)
    return cov_frac, passes


def main():
    lats = list(range(-80, 81, 20))
    lons = [30.0 * j for j in range(12)]
    cells = sl.grid_cells([float(x) for x in lats], lons)
    centres = np.array([c for c, _z in cells])
    zeniths = np.array([z for _c, z in cells])
    a, inc = RE + 550e3, math.radians(53.0)

    # reference: the Rust event-detection engine (ms-accurate window edges)
    t0 = time.perf_counter()
    m = np.array(sl.coverage_metrics(a, inc, 0.0, 0.0, sl.Sensor.elevation(5.0), cells, HORIZON, tol=1.0, threads=8))
    rust_ms = (time.perf_counter() - t0) * 1e3
    rust_cov_s = m[:, 1].sum() * HORIZON
    rust_passes = int(m[:, 0].sum())

    print("=" * 84)
    print(f"Access windows — Rust event engine vs naive fixed-cadence Python ({len(cells)} cells, 24 h)")
    print("=" * 84)
    print(f"\n  Rust engine (reference): {rust_cov_s / 3600:8.2f} cell·h covered, {rust_passes} passes, {rust_ms:6.1f} ms")
    print(f"\n  {'naive Δt':>9}{'cell·h':>10}{'cov err':>9}{'passes':>8}{'missed':>8}{'time':>9}")
    for dt in (300.0, 60.0, 10.0):
        t0 = time.perf_counter()
        cov_frac, passes = naive_coverage(centres, zeniths, a, inc, dt)
        naive_ms = (time.perf_counter() - t0) * 1e3
        naive_cov_s = cov_frac.sum() * HORIZON
        naive_passes = int(passes.sum())
        cov_err = 100.0 * (naive_cov_s - rust_cov_s) / rust_cov_s
        missed = rust_passes - naive_passes
        print(f"  {dt:>7.0f}s{naive_cov_s / 3600:>10.2f}{cov_err:>+8.1f}%{naive_passes:>8}{missed:>+8}{naive_ms:>7.1f} ms")

    print("\n" + "=" * 84)
    print("  Coarse sampling under-counts coverage and misses short passes; it only approaches the")
    print("  event-detected truth as Δt→0, and is not faster there. Event detection is the differentiator.")


if __name__ == "__main__":
    main()
