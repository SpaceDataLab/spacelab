#!/usr/bin/env python
"""IERS 2010 relativistic corrections to satellite acceleration (Eq. 10.12) — reference oracle.

Three terms, all in the GCRS (geocentric celestial) frame with the general-relativity PPN values
beta = gamma = 1:
  * Schwarzschild  — the dominant static-field 1/c^2 correction (~1e-8 m/s^2 at LEO),
  * Lense-Thirring — frame dragging by the Earth's rotation (~1e-11 m/s^2),
  * de Sitter      — geodesic precession from the Earth's orbit about the Sun (~1e-11 m/s^2).

This mirrors the closed-form the Rust port evaluates, so it catches a transcription bug in either
port; the genuine independent third-party check is Orekit (tools/orekit-oracle ->
orekit_relativity.csv), whose Relativity / LenseThirringRelativity / DeSitterRelativity implement the
same equation. Prints machine-readable golden vectors for the Rust unit test.

    uv run --no-project --with numpy python derivations/relativity.py
"""

from __future__ import annotations

import numpy as np

C = 299_792_458.0  # speed of light, m/s (exact, SI)
GM_E = 3.986_004_415e14  # m^3/s^2
GM_S = 1.327_124_400_18e20  # m^3/s^2
J_EARTH = np.array(
    [0.0, 0.0, 9.8e8]
)  # Earth angular momentum per unit mass, m^2/s (IERS 2010)


def schwarzschild(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Eq. 10.12 line 1, beta = gamma = 1: GM/(c^2 r^3) {[4 GM/r - v.v] r + 4 (r.v) v}."""
    rn = float(np.linalg.norm(r))
    vv = float(v @ v)
    rv = float(r @ v)
    return GM_E / (C**2 * rn**3) * ((4.0 * GM_E / rn - vv) * r + 4.0 * rv * v)


def lense_thirring(r: np.ndarray, v: np.ndarray, j: np.ndarray = J_EARTH) -> np.ndarray:
    """Eq. 10.12 line 2, gamma = 1: 2 GM/(c^2 r^3) {(3/r^2)(r x v)(r.J) + (v x J)}."""
    rn = float(np.linalg.norm(r))
    return (
        2.0
        * GM_E
        / (C**2 * rn**3)
        * (3.0 / rn**2 * np.cross(r, v) * float(r @ j) + np.cross(v, j))
    )


def de_sitter(v: np.ndarray, r_sun: np.ndarray, v_sun: np.ndarray) -> np.ndarray:
    """Eq. 10.12 line 3, gamma = 1: 3 {[R_dot x (-GM_S R/(c^2 R^3))] x v}.

    R = Earth position relative to Sun = -r_sun; R_dot = -v_sun (r_sun, v_sun are geocentric Sun).
    Hence -GM_S R/(c^2 R^3) = +GM_S r_sun/(c^2 |r_sun|^3) and R_dot = -v_sun.
    """
    rsn = float(np.linalg.norm(r_sun))
    u = GM_S * r_sun / (C**2 * rsn**3)
    v_earth = -v_sun
    return 3.0 * np.cross(np.cross(v_earth, u), v)


def main() -> None:
    # Shared GCRF test geometry (LEO satellite, plausible geocentric Sun position and velocity).
    r = np.array([7_000e3, 1_000e3, 2_000e3])
    v = np.array([-1_000.0, 6_500.0, 3_500.0])
    r_sun = np.array([1.35e11, -5.0e10, -2.2e10])
    v_sun = np.array([11_000.0, 25_000.0, 10_800.0])

    # A spin axis that is not the GCRF z axis. With the Earth's J = [0, 0, 9.8e8] both the x and y
    # products of r.J and the z component of v x J are exactly zero, so an axial-only case leaves
    # three of the eight arithmetic operations in the Lense-Thirring sum unexercised.
    j_tilted = np.array([2.7e8, -4.1e8, 8.9e8])

    a_s = schwarzschild(r, v)
    a_lt = lense_thirring(r, v)
    a_lt_tilt = lense_thirring(r, v, j_tilted)
    a_ds = de_sitter(v, r_sun, v_sun)
    for name, a in [
        ("Schwarzschild", a_s),
        ("Lense-Thirring", a_lt),
        ("L-T tilted J", a_lt_tilt),
        ("de Sitter", a_ds),
    ]:
        print(f"{name:14s} a = {a}  |a| = {np.linalg.norm(a):.4e} m/s^2")
    print("REF_SCH," + ",".join(f"{x:.17e}" for x in a_s))
    print("REF_LT," + ",".join(f"{x:.17e}" for x in a_lt))
    print("REF_LT_TILT," + ",".join(f"{x:.17e}" for x in a_lt_tilt))
    print("REF_DS," + ",".join(f"{x:.17e}" for x in a_ds))


if __name__ == "__main__":
    main()
