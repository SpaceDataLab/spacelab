#!/usr/bin/env -S uv run --quiet --with numpy --with scipy python
"""Independent oracle for the high-degree geopotential: validate the Rust Gottlieb recursion to degree 70.

The Rust `GeoPotential` uses a Cartesian solid-harmonic (Cunningham/Gottlieb) recursion. This script is
a *different* implementation of the same field: it evaluates the potential
`U = (μ/r) Σₙ (Rₑ/r)ⁿ Σₘ P̄_nm(sinφ)[C̄_nm cos mλ + S̄_nm sin mλ]` via SciPy's associated Legendre
functions (`scipy.special.lpmv`, a recursion unrelated to ours) and takes the acceleration as the
*numerical* gradient of U — so nothing in the acceleration path is shared with Rust. Agreement to ~1e-8
at degree 70 validates the normalized recursion independently.

Two steps:
  1. self-check at degree 4 against the Maxima gradient oracle (calibrates normalization / CS phase);
  2. fetch XGM2019e to degree 70 and print a(x,y,z) at two points — the numbers the Rust test asserts.

Run:  timeout 300 uv run --no-project --with numpy --with scipy python derivations/geopotential_validate.py
"""

from __future__ import annotations

import urllib.request

import numpy as np
from scipy.special import lpmv

URL = (
    "https://icgem.gfz-potsdam.de/getmodel/gfc/"
    "8e01a2aba4f291ac0638b79a390155713b70fdc9b266981023c756839c11c9b8/XGM2019e.gfc"
)


def normalization(n: int, m: int) -> float:
    delta = 1.0 if m == 0 else 2.0
    ratio = 1.0
    for k in range(n - m + 1, n + m + 1):
        ratio /= k
    return np.sqrt(delta * (2 * n + 1) * ratio)


def potential(x, y, z, gm, re, cbar, sbar, degree):
    """U at (x,y,z) from normalized coefficients, via SciPy associated Legendre (geodesy sign)."""
    r = np.sqrt(x * x + y * y + z * z)
    sphi = z / r
    lam = np.arctan2(y, x)
    total = 0.0
    for n in range(degree + 1):
        rr = (re / r) ** n
        acc = 0.0
        for m in range(n + 1):
            c = cbar.get((n, m), 0.0)
            s = sbar.get((n, m), 0.0)
            if c == 0.0 and s == 0.0:
                continue
            # geodesy P̄_nm = N_nm · (−1)^m · lpmv(m, n, sinφ)  (lpmv carries the Condon–Shortley phase)
            pbar = normalization(n, m) * ((-1) ** m) * lpmv(m, n, sphi)
            acc += pbar * (c * np.cos(m * lam) + s * np.sin(m * lam))
        total += rr * acc
    return gm / r * total


def accel(p, gm, re, cbar, sbar, degree, h=1.0):
    x, y, z = p
    args = (gm, re, cbar, sbar, degree)
    ax = (potential(x + h, y, z, *args) - potential(x - h, y, z, *args)) / (2 * h)
    ay = (potential(x, y + h, z, *args) - potential(x, y - h, z, *args)) / (2 * h)
    az = (potential(x, y, z + h, *args) - potential(x, y, z - h, *args)) / (2 * h)
    return np.array([ax, ay, az])


def self_check():
    """Degree-4 unnormalized set from derivations/geopotential.mac; must reproduce the Maxima gradient."""
    unnorm = {
        (2, 0): (-1.0826e-3, 0.0), (2, 1): (1.0e-6, 2.0e-6), (2, 2): (1.5e-6, -0.9e-6),
        (3, 0): (2.5e-6, 0.0), (3, 1): (1.1e-6, -0.7e-6), (3, 2): (0.8e-6, 0.5e-6), (3, 3): (-0.6e-6, 0.4e-6),
        (4, 0): (-1.6e-6, 0.0), (4, 2): (0.3e-6, -0.2e-6),
    }
    cbar, sbar = {(0, 0): 1.0}, {}
    for (n, m), (c, s) in unnorm.items():
        inv = 1.0 / normalization(n, m)
        cbar[(n, m)], sbar[(n, m)] = c * inv, s * inv
    gm, re = 3.986004418e14, 6378137.0
    a = accel([7_000_123.0, 1_200_456.0, 900_789.0], gm, re, cbar, sbar, 4)
    expect = np.array([-7.613097285278765, -1.3054327859934878, -0.9820712253543767])
    rel = np.abs(a - expect) / np.abs(expect)
    print(f"  self-check vs Maxima (degree 4): a = {a}, max rel err = {rel.max():.2e}")
    assert rel.max() < 1e-6, "self-check failed — normalization/CS phase wrong"


def fetch(degree):
    gm = re = None
    cbar, sbar = {(0, 0): 1.0}, {}
    with urllib.request.urlopen(URL, timeout=90) as resp:
        body = False
        for raw in resp:
            line = raw.decode("latin-1").strip()
            if not body:
                if line.startswith("earth_gravity_constant"):
                    gm = float(line.split()[1].replace("D", "e"))
                elif line.startswith("radius"):
                    re = float(line.split()[1])
                elif line.startswith("end_of_head"):
                    body = True
                continue
            if not line.startswith("gfc"):
                continue
            _, ns, ms, cs, ss = line.split()[:5]
            n, m = int(ns), int(ms)
            if n > degree:
                break
            if n >= 2:
                cbar[(n, m)] = float(cs.replace("D", "e"))
                sbar[(n, m)] = float(ss.replace("D", "e"))
    return gm, re, cbar, sbar


def main():
    print("STEP 1  self-check against the Maxima oracle")
    self_check()
    print("\nSTEP 2  XGM2019e degree 70 — independent (SciPy) acceleration at two points")
    gm, re, cbar, sbar = fetch(70)
    print(f"  GM={gm:.6e} Re={re:.4f}  ({len(cbar)} coeffs)")
    for p in [[5_000_000.0, 3_500_000.0, 2_600_000.0], [6_600_000.0, 1_200_000.0, 900_000.0]]:
        a = accel(p, gm, re, cbar, sbar, 70)
        print(f"  r={p}:  a = [{a[0]:.10e}, {a[1]:.10e}, {a[2]:.10e}]")
    print("DONE")


if __name__ == "__main__":
    main()
