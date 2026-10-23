#!/usr/bin/env -S uv run --quiet --with mpmath python
"""Reference values for the `spacelab-autodiff` double-double transcendentals.

The double-double `exp`, `ln`, `sin`, `cos`, `atan2` are easy to get subtly wrong
(argument reduction, Taylor truncation, quadrant sign). This emits an independent
oracle: for a spread of arguments — small, medium, large (heavy range reduction),
negative, near-zero — it evaluates each function to 50 digits in mpmath and splits
the result into the double-double pair the Rust test asserts against:

    hi = float(V)              # nearest double to the true value V
    lo = float(V - mpf(hi))    # nearest double to the residual

Every input is an *exact* f64 (so `mpf(input)` is exact and the reference is the
value of the function at precisely the number the Rust code sees — no input-rounding
slack). The printed `(input, hi, lo)` tuples paste straight into the Rust test
tables. The reference pair itself is the true value rounded to double-double, so it
carries its own ~1e-32 relative error — comparisons are meaningful down to ~1e-32,
which is the correctly-rounded-double-double target.

Run:
    ./derivations/dd_transcendentals.py
    timeout 60 uv run --quiet --with mpmath python derivations/dd_transcendentals.py

The derivation is the source; the Rust tables are a build artifact in git.
"""

from __future__ import annotations

import mpmath as mp

mp.mp.dps = 50


def split(v: mp.mpf) -> tuple[float, float]:
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return hi, lo


def unary(name: str, fn, inputs: list[float]) -> None:
    print(f"    // {name}: (input, hi, lo)   [true value shown to 36 digits]")
    for x in inputs:
        v = fn(mp.mpf(x))
        hi, lo = split(v)
        print(f"    ({x!r}, {hi!r}, {lo!r}),  // {mp.nstr(v, 36)}")
    print()


def binary_atan2(cases: list[tuple[float, float]]) -> None:
    print("    // atan2(y, x): (y, x, hi, lo)   [true value shown to 36 digits]")
    for y, x in cases:
        v = mp.atan2(mp.mpf(y), mp.mpf(x))
        hi, lo = split(v)
        print(f"    ({y!r}, {x!r}, {hi!r}, {lo!r}),  // {mp.nstr(v, 36)}")
    print()


def powf(cases: list[tuple[float, float]]) -> None:
    print("    // powf: (base, p, hi, lo)   base^p, exp(p*ln base)   [true value to 36 digits]")
    for base, p in cases:
        v = mp.mpf(base) ** mp.mpf(p)
        hi, lo = split(v)
        print(f"    ({base!r}, {p!r}, {hi!r}, {lo!r}),  // {mp.nstr(v, 36)}")
    print()


def main() -> None:
    print("double-double transcendental reference values (mpmath, 50-digit)\n")

    # exp: 0, +-small, +-medium, large-magnitude (range reduction), near overflow.
    unary(
        "exp",
        mp.exp,
        [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1e-10, 30.0, -30.0, 700.0, -700.0],
    )

    # ln: 1 (=0), around 1, small (heavy: exp(-x) ~ 1e8), large.
    unary(
        "ln",
        mp.log,
        [1.0, 2.0, 0.5, 1.5, 10.0, 100.0, 1e-8, 1e8, 2.718281828459045],
    )

    # sin: near-zero, small, medium spanning octants, large (heavy reduction), negative.
    unary(
        "sin",
        mp.sin,
        [0.001, 0.5, -0.5, 1.0, 2.0, 3.0, -3.0, 6.0, 100.0, 1000.0, -1000.0],
    )

    # cos: 0 (=1), the task's -0.5, small, medium across octants, large, negative.
    unary(
        "cos",
        mp.cos,
        [0.0, -0.5, 0.5, 1.0, 2.0, 3.0, 6.0, 100.0, 1000.0, -1000.0],
    )

    # atan2: all four quadrants incl. the exact corners, axes, and off-axis points.
    binary_atan2(
        [
            (1.0, 1.0),    # Q1  ->  pi/4
            (1.0, -1.0),   # Q2  ->  3pi/4
            (-1.0, -1.0),  # Q3  -> -3pi/4
            (-1.0, 1.0),   # Q4  -> -pi/4
            (1.0, 0.0),    # +y axis ->  pi/2
            (-1.0, 0.0),   # -y axis -> -pi/2
            (0.0, 1.0),    # +x axis ->  0
            (0.0, -1.0),   # -x axis ->  pi
            (3.0, 4.0),    # Q1 off-corner
            (-3.0, 4.0),   # Q4
            (3.0, -4.0),   # Q2
            (-3.0, -4.0),  # Q3
            (0.5, 2.0),    # shallow angle (|x| > |y|, eqn-1 branch)
            (2.0, 0.5),    # steep angle   (|y| > |x|, eqn-2 branch)
        ]
    )

    # powf = exp(p * ln base): integer, fractional, and root exponents on a positive base.
    powf(
        [
            (2.0, 0.5),    # sqrt(2), cross-checks the independent true-dd sqrt
            (2.0, 10.0),   # 1024 exactly
            (10.0, 3.0),   # 1000 exactly
            (3.0, 1.0 / 3.0),  # cube-ish root of 3
            (1.5, 2.5),    # general fractional power
            (2.718281828459045, 2.0),  # e^2 (base = e rounded to f64)
        ]
    )


if __name__ == "__main__":
    main()
