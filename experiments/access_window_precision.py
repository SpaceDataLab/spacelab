#!/usr/bin/env python
"""Quantify the access-window engine's edge precision against a naive fixed-cadence sampler.

This is the plan's headline differentiator (edit #3) made measurable: "a Python script that samples
once a minute" places every window edge only on a grid point, so it misplaces each edge by up to the
sample step and drops any pass shorter than the step that falls between samples. The engine instead
steps by |g|/max_rate — an interval the switching function provably cannot cross — so it never steps
over a pass, and refines every edge to the millisecond with Brent.

The comparison is exact: the naive detector below samples the *same* elevation switching function the
engine scans (same analytic orbit, same ground cell, same mask), so any difference is the sampling
method, not the geometry. It needs no external data.

Build the wheel first, then run with it on the path:
  cd crates/spacelab-python && maturin build --release -o /tmp/w
  uv run --no-project --with /tmp/w/*.whl python experiments/access_window_precision.py
"""

from __future__ import annotations

import math

import spacelab as sl

MU = 3.986004418e14  # m^3/s^2
OMEGA = 7.292_115_9e-5  # rad/s, Earth rotation rate (matches the engine's constant)

A = 6_378_137.0 + 550e3  # 550 km circular
INC = math.radians(53.0)
RAAN = 0.0
PHASE = 0.0
MASK = math.radians(5.0)
HORIZON = 86_400.0  # one day
CELL = sl.grid_cells([40.0], [30.0])[0]  # (centre_ecef, zenith), spherical Earth


def norm(v):
    return sum(c * c for c in v) ** 0.5


def sat_ecef(t: float):
    """The engine's analytic circular orbit: ECI element set, then rotate into the Earth-fixed frame."""
    u = PHASE + math.sqrt(MU / A**3) * t
    cu, su = math.cos(u), math.sin(u)
    cr, sr = math.cos(RAAN), math.sin(RAAN)
    ci, si = math.cos(INC), math.sin(INC)
    ex = A * (cu * cr - su * sr * ci)
    ey = A * (cu * sr + su * cr * ci)
    ez = A * (su * si)
    th = OMEGA * t
    c, s = math.cos(th), math.sin(th)  # Mat3::rot3(th) = [[c,s,0],[-s,c,0],[0,0,1]]
    return (c * ex + s * ey, -s * ex + c * ey, ez)


def switching(t: float) -> float:
    """g(t) >= 0 iff the satellite is above the cell's elevation mask — same sign as the engine's g."""
    centre, zenith = CELL
    sx = sat_ecef(t)
    d = [sx[i] - centre[i] for i in range(3)]
    sin_elev = sum(d[i] * zenith[i] for i in range(3)) / norm(d)
    return sin_elev - math.sin(MASK)


def naive_windows(step: float):
    """Sample coverage on a fixed `step` grid; a window is a run of covered samples, edges quantized."""
    n = int(HORIZON / step)
    covered = [switching(k * step) >= 0.0 for k in range(n + 1)]
    out = []
    k = 0
    while k <= n:
        if covered[k]:
            j = k
            while j + 1 <= n and covered[j + 1]:
                j += 1
            out.append((k * step, j * step))
            k = j + 1
        else:
            k += 1
    return out


def match_edge_errors(engine, naive):
    """For each engine window, the worst edge error of the naive window that overlaps it; plus misses."""
    errors = []
    missed = []
    for rise, sett in engine:
        overlap = [(nr, ns) for nr, ns in naive if not (ns < rise or nr > sett)]
        if not overlap:
            missed.append((rise, sett))
            continue
        err = max(max(abs(nr - rise), abs(ns - sett)) for nr, ns in overlap)
        errors.append(err)
    return errors, missed


def main() -> None:
    sensor = sl.Sensor.elevation(5.0)
    engine = sl.access_windows(A, INC, RAAN, PHASE, sensor, CELL, HORIZON, tol=1e-3)
    engine_durations = [s - r for r, s in engine]
    print(f"engine (Brent, tol=1 ms): {len(engine)} passes, "
          f"durations {min(engine_durations):.0f}–{max(engine_durations):.0f} s")

    for step in (30.0, 60.0):
        naive = naive_windows(step)
        errors, missed = match_edge_errors(engine, naive)
        max_err = max(errors) if errors else 0.0
        mean_err = sum(errors) / len(errors) if errors else 0.0
        short = [d for d in engine_durations if d < step]
        print(f"  naive {step:>4.0f}-s grid: {len(naive)} passes, "
              f"edge error mean {mean_err:5.1f} s / max {max_err:5.1f} s, "
              f"missed {len(missed)}, engine passes shorter than the step: {len(short)}")

        # The differentiator: a fixed grid cannot beat its own step on edge placement, while the engine
        # is at the millisecond. Across a day of passes the worst edge error is a large fraction of the
        # step, and the mean is well above anything Brent leaves.
        assert max_err > step / 4.0, f"naive {step}s edges suspiciously precise: {max_err:.1f} s"
        assert mean_err > step / 8.0, f"naive {step}s mean edge error too small: {mean_err:.1f} s"

    assert len(engine) > 0, "no passes found"
    print("engine refines every edge to the millisecond; the naive grid is bounded by its step")


if __name__ == "__main__":
    main()
