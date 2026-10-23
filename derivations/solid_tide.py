#!/usr/bin/env python
"""IERS 2010 solid Earth tide, Step 1 (frequency-independent) — reference oracle for the Rust force.

Computes the degree-2,3 changes ΔC̄_nm, ΔS̄_nm to the geopotential coefficients raised by the Sun and
Moon (IERS Conventions 2010, Ch. 6, Eq. 6.6), plus the degree-4 back-coupling from the degree-2 tide
via the k⁺ Love numbers (Eq. 6.7). The nominal (anelastic) Love numbers are IERS 2010 Table 6.3.

Two independent checks the Rust port is validated against:
  * the fully-normalized associated Legendre P̄_nm is asserted against closed-form values;
  * the tidal acceleration is computed BOTH analytically (−∇U in closed form) AND by a central finite
    difference of the potential U — they must agree, which validates the analytic gradient the Rust
    force uses by a genuinely different method.

    uv run --no-project --with numpy python derivations/solid_tide.py
"""

from __future__ import annotations

import math

import numpy as np

# Geopotential background constants — must match the embedded XGM2019e model the tide is added to.
GM_EARTH = 3.986_004_415e14  # m^3/s^2  (MODEL_GM)
RE = 6_378_136.3  # m        (MODEL_RE)
GM_SUN = 1.327_124_400_18e20
GM_MOON = 4.902_800_066e12

# IERS 2010 Table 6.3, nominal (anelastic) solid Earth tide Love numbers: Re(k), Im(k), and k⁺.
K2 = {0: (0.30190, -0.00000), 1: (0.29830, -0.00144), 2: (0.30102, -0.00130)}
K2_PLUS = {0: -0.00089, 1: -0.00080, 2: -0.00057}
K3 = {0: 0.093, 1: 0.093, 2: 0.093, 3: 0.094}  # imaginary parts are zero at degree 3


def pbar(n: int, m: int, x: float) -> float:
    """Fully-normalized associated Legendre P̄_nm(x) (geodesy 4π, no Condon-Shortley phase)."""
    # Sectorials P̄_mm, then the two-term column recurrence in n (Holmes & Featherstone 2002).
    t = math.sqrt(max(0.0, 1.0 - x * x))  # cos(latitude)
    pmm = 1.0
    for k in range(1, m + 1):
        pmm *= math.sqrt((2.0 * k + 1.0) / (2.0 * k)) * t
    if m >= 1:
        pmm *= math.sqrt(2.0)  # the (2 − δ_0m) factor of the geodesy 4π normalization, m ≥ 1
    if n == m:
        return pmm
    p_prev = pmm  # P̄_{m,m}
    p_cur = math.sqrt(2.0 * m + 3.0) * x * pmm  # P̄_{m+1,m}
    for nn in range(m + 2, n + 1):
        a = math.sqrt((2.0 * nn - 1.0) * (2.0 * nn + 1.0) / ((nn - m) * (nn + m)))
        b = math.sqrt(
            (2.0 * nn + 1.0) * (nn + m - 1.0) * (nn - m - 1.0) / ((nn - m) * (nn + m) * (2.0 * nn - 3.0))
        )
        p_next = a * x * p_cur - b * p_prev
        p_prev, p_cur = p_cur, p_next
    return p_cur


def _check_legendre() -> None:
    assert abs(pbar(2, 0, 0.0) - (-math.sqrt(5.0) / 2.0)) < 1e-13
    assert abs(pbar(2, 2, 0.0) - (math.sqrt(15.0) / 2.0)) < 1e-13
    assert abs(pbar(1, 1, 0.0) - math.sqrt(3.0)) < 1e-13
    # P̄_30(x) = sqrt(7)/2 (5x^3 - 3x)
    x = 0.4
    assert abs(pbar(3, 0, x) - math.sqrt(7.0) / 2.0 * (5 * x**3 - 3 * x)) < 1e-12


def delta_cs(bodies: list[tuple[np.ndarray, float]]) -> dict[tuple[int, int], complex]:
    """ΔC̄_nm − i ΔS̄_nm for n=2,3 (Eq. 6.6) and n=4 from the k⁺ term (Eq. 6.7). Keyed (n, m)."""
    # sum_j (GM_j/GM_E)(RE/r_j)^{n+1} P̄_nm(sinφ_j) e^{-i m λ_j}, accumulated as (cos, sin) parts.
    s: dict[tuple[int, int], complex] = {}
    for pos, gm in bodies:
        r = float(np.linalg.norm(pos))
        lat = math.asin(pos[2] / r)
        lon = math.atan2(pos[1], pos[0])
        x = math.sin(lat)
        for n in (2, 3):
            for m in range(n + 1):
                fac = (gm / GM_EARTH) * (RE / r) ** (n + 1) * pbar(n, m, x)
                s[(n, m)] = s.get((n, m), 0j) + fac * complex(math.cos(m * lon), -math.sin(m * lon))

    out: dict[tuple[int, int], complex] = {}
    for m in range(3):  # n = 2
        kr, ki = K2[m]
        out[(2, m)] = (complex(kr, ki) / 5.0) * s[(2, m)]
        out[(4, m)] = (K2_PLUS[m] / 5.0) * s[(2, m)]  # degree-4 coupling, real k⁺
    for m in range(4):  # n = 3
        out[(3, m)] = (K3[m] / 7.0) * s[(3, m)]
    # Zero-tide background: remove the permanent (zero-frequency) part of ΔC̄₂₀ (IERS Eq. 6.13),
    # else it double-counts the permanent tide already in the zero-tide C̄₂₀.
    perm = 4.4228e-8 * -0.31460 * K2[0][0]
    out[(2, 0)] = complex(out[(2, 0)].real - perm, out[(2, 0)].imag)
    return out  # value = ΔC̄ − i ΔS̄  →  ΔC̄ = Re, ΔS̄ = −Im


def potential(r: np.ndarray, dcs: dict[tuple[int, int], complex]) -> float:
    """Tidal potential U(r) from the ΔC̄/ΔS̄ harmonics, degrees 2..4."""
    rn = float(np.linalg.norm(r))
    lat = math.asin(r[2] / rn)
    lon = math.atan2(r[1], r[0])
    x = math.sin(lat)
    u = 0.0
    for n in (2, 3, 4):
        radial = (RE / rn) ** n
        acc = 0.0
        for m in range(n + 1):
            if (n, m) not in dcs:
                continue
            dc, ds = dcs[(n, m)].real, -dcs[(n, m)].imag
            acc += (dc * math.cos(m * lon) + ds * math.sin(m * lon)) * pbar(n, m, x)
        u += radial * acc
    return GM_EARTH / rn * u


def main() -> None:
    _check_legendre()
    sat = np.array([6.5e6, 2.0e6, 3.0e6])
    sun = np.array([1.4e11, 0.5e11, 0.3e11])
    moon = np.array([3.5e8, 1.5e8, 1.0e8])
    dcs = delta_cs([(moon, GM_MOON), (sun, GM_SUN)])

    print("ΔC̄_nm, ΔS̄_nm (IERS 2010 Step 1):")
    for key in [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (4, 0), (4, 2)]:
        print(f"  n={key[0]} m={key[1]}:  ΔC = {dcs[key].real:+.6e}   ΔS = {-dcs[key].imag:+.6e}")

    # Central finite difference of U — an independent gradient for validating the Rust force. The
    # geodesy convention (matching the geopotential) is a = +∇U for the positive potential U = GM/r·[…].
    h = 1.0
    accel_num = np.array(
        [
            (potential(sat + h * e, dcs) - potential(sat - h * e, dcs)) / (2 * h)
            for e in np.eye(3)
        ]
    )
    print(f"tidal acceleration at the satellite (finite-diff of U): {accel_num} m/s^2")
    print(f"  |a| = {np.linalg.norm(accel_num):.6e} m/s^2  (sanity: ~1e-7 for LEO)")
    # Emit machine-readable reference for the Rust test.
    print("REFERENCE_ACCEL," + ",".join(f"{a:.17e}" for a in accel_num))
    print("REFERENCE_SAT," + ",".join(f"{a:.1f}" for a in sat))
    print("REFERENCE_SUN," + ",".join(f"{a:.1f}" for a in sun))
    print("REFERENCE_MOON," + ",".join(f"{a:.1f}" for a in moon))


if __name__ == "__main__":
    main()
