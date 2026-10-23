#!/usr/bin/env -S uv run --quiet --with sympy --with mpmath --with pyerfa --with numpy python
"""Derivation, constant generation and fixture generation for GMST (IAU 1982).

Run:
    ./derivations/gmst82.py            # derive, check, regenerate fixtures
    ./derivations/gmst82.py --check    # derive and check only; touch nothing

The derivation is the source. The Rust is a build artifact that happens to live
in git. See plans/07-symbolic-derivation.md.

Provenance
----------
The equation is Aoki et al. (1982), as printed in Vallado, *Fundamentals of
Astrodynamics and Applications*, Eq. 3-47. It is read from the published
literature. No implementation's source code is consulted.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import mpmath as mp
import numpy as np
import sympy as sp

FIXTURE = pathlib.Path(__file__).parent.parent / "crates/spacelab-fa/tests/fixtures/gmst82_pyerfa.csv"

# Published coefficients, seconds of time. Vallado Eq. 3-47.
C0 = sp.Rational("67310.54841")
C1_HOURS = sp.Integer(876600)  # hours per Julian century, the whole-day part
C1_SECONDS = sp.Rational("8640184.812866")
C2 = sp.Rational("0.093104")
C3 = sp.Rational("-6.2e-6")


def derive() -> None:
    """Establish, symbolically, the identity that the Rust implementation relies on."""
    T, day, frac = sp.symbols("T day frac", real=True)

    # Step 1: the whole-day coefficient is exactly 86400 seconds per day.
    #
    #   876600 h x 3600 s/h  ==  86400 s/day x 36525 day/century
    #
    # If this were only approximate, the reduction below would be invalid.
    lhs = C1_HOURS * 3600
    rhs = sp.Integer(86400) * sp.Integer(36525)
    assert sp.simplify(lhs - rhs) == 0, "the whole-day term is not exact"
    print(f"  876600 h = {lhs} s/century = 86400 x 36525   [exact]")

    # Step 2: therefore that term contributes 86400 * D seconds, D = days since J2000.
    D = 36525 * T
    theta = C0 + lhs * T + C1_SECONDS * T + C2 * T**2 + C3 * T**3
    theta_split = C0 + 86400 * D + C1_SECONDS * T + C2 * T**2 + C3 * T**3
    assert sp.simplify(theta - theta_split) == 0
    print("  theta(T) = C0 + 86400*D + C1*T + C2*T^2 + C3*T^3   [equivalent]")

    # Step 3: modulo one day, 86400*D collapses onto the fractional day, because
    # D = day + frac with `day` an integer.
    #
    #   86400*(day + frac)  ==  86400*frac   (mod 86400)
    #
    # This is what lets the implementation avoid ever forming 3.15e9 * T.
    residual = sp.simplify((86400 * (day + frac)) - 86400 * frac)
    assert sp.simplify(residual / 86400 - day) == 0
    print("  86400*D == 86400*frac  (mod 86400)                 [the reduction]")

    # Step 4: what the reduction buys, stated as a ratio of the largest
    # intermediate formed by each evaluation order, at T = 1 century.
    naive = float(lhs)  # 3.15576e9
    ours = float(C1_SECONDS)  # 8.64e6
    print(f"  largest intermediate: naive {naive:.3e} s vs ours {ours:.3e} s  ({naive / ours:.0f}x)")


def emit_constants() -> None:
    """Correctly-rounded f64 for each published decimal constant.

    An f64 literal in source is the decimal the author typed; nothing guarantees
    it is the nearest double to the exact value. Evaluate at high precision, round
    once, and print the bit pattern so the compiler cannot re-interpret it.
    """
    mp.mp.dps = 50
    print("\n  correctly-rounded constants (50-digit evaluation):")
    for name, value in [
        ("C0", "67310.54841"),
        ("C1", "8640184.812866"),
        ("C2", "0.093104"),
        ("C3", "-6.2e-6"),
    ]:
        exact = mp.mpf(value)
        nearest = float(exact)
        bits = struct.unpack("<Q", struct.pack("<d", nearest))[0]
        err_ulp = abs(mp.mpf(nearest) - exact) / mp.mpf(np.spacing(abs(nearest) or 1.0))
        print(f"    {name:>3} = {nearest!r:<24} 0x{bits:016X}   error {float(err_ulp):.3f} ulp")


def regenerate_fixtures(seed: int = 20260710, n_random: int = 40) -> None:
    """Fixtures from pyerfa, the SOFA-lineage oracle.

    NOTE: SOFA and ERFA agree bit-for-bit on every epoch tested — verified by
    compiling SOFA directly (tools/sofa-oracle). They share an ancestor, so they
    are ONE oracle, not two. A genuinely independent check requires a different
    lineage. See plans/04-validation-and-risks.md 1.2.
    """
    import erfa

    rng = np.random.default_rng(seed)
    anchors = [0.0, 0.5, -0.5, 1.0, 9000.0]
    random = rng.uniform(-1_826_250, 1_826_250, n_random)  # +/- 5000 years
    days = np.concatenate([anchors, random])

    gmst = erfa.gmst82(np.full_like(days, 2451545.0), days)

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(FIXTURE, np.column_stack([days, gmst]), delimiter=",", fmt="%.17g")
    print(f"\n  wrote {len(days)} fixtures to {FIXTURE.relative_to(FIXTURE.parents[4])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="derive and check only")
    args = parser.parse_args()

    print("gmst82: symbolic derivation")
    derive()
    emit_constants()

    if not args.check:
        regenerate_fixtures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
