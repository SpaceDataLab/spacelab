#!/usr/bin/env -S uv run --quiet --with mpmath python
"""Double-double constants for the transcendental functions in `spacelab-autodiff`.

A double-double stores a real as `hi + lo`, two non-overlapping f64s. The
transcendental routines (`exp`, `ln`, `sin`, `cos`, `atan2`) reduce their
argument against `ln 2`, `pi`, `2*pi`, `pi/2`, `pi/4`, so those constants must
themselves carry ~106 bits, not the ~53 of a single f64 literal.

For each constant x this evaluates x to 50 digits in mpmath, then splits it into
the canonical double-double pair

    hi = float(x)              # nearest double to x
    lo = float(x - mpf(hi))    # nearest double to the exact residual

and prints (a) the residual `x - hi - lo`, i.e. the part of x that even the pair
cannot represent, and (b) a ready-to-paste `DoubleDouble::new(hi, lo)` literal.

The residual matters: reducing a large argument `a` as `a - N*(2*pi)` inherits an
absolute error of about `N * |2*pi - dd(2*pi)|`, which is the accuracy floor for
e.g. `sin(1000)` (N ~ 159). It is printed so that floor is documented, not
guessed. See derivations/dd_transcendentals.py for the end-to-end check.

Run:
    ./derivations/dd_constants.py          # print residuals + Rust literals
    timeout 60 uv run --quiet --with mpmath python derivations/dd_constants.py

The derivation is the source; the Rust literals it prints are a build artifact
that happens to live in git. Do NOT hand-edit the constants in double_double.rs;
regenerate them here. (Constants-from-a-CAS rule, plans/07-symbolic-derivation.md.)
"""

from __future__ import annotations

import struct

import mpmath as mp

mp.mp.dps = 50


def bits(x: float) -> int:
    """The IEEE-754 bit pattern of a double, so the compiler cannot re-round it."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def split(x: mp.mpf) -> tuple[float, float, mp.mpf]:
    """Canonical double-double split: (hi, lo, residual) with residual = x - hi - lo."""
    hi = float(x)
    lo = float(x - mp.mpf(hi))
    residual = x - mp.mpf(hi) - mp.mpf(lo)
    return hi, lo, residual


def emit(name: str, x: mp.mpf, note: str, rust_hi: str) -> None:
    """Print a constant's split.

    `rust_hi` is the `core::f64::consts` path whose value is the correctly-rounded
    f64 of x — bit-identical to `hi` here. The Rust literal uses it for the high
    word (so `clippy::approx_constant` sees a named constant, not a bare pi/ln2
    literal) and pastes only the generated `lo` residual. The equality of the two
    is asserted below via the printed hex.
    """
    hi, lo, residual = split(x)
    ulp_lo = mp.mpf(2) ** (mp.floor(mp.log(abs(mp.mpf(lo)) or mp.mpf(1), 2)) - 52)
    print(f"  {name}  ({note})")
    print(f"    value    = {mp.nstr(x, 40)}")
    print(f"    hi       = {hi!r:<26} 0x{bits(hi):016X}  == {rust_hi}")
    print(f"    lo       = {lo!r:<26} 0x{bits(lo):016X}")
    print(f"    residual = {mp.nstr(residual, 6)}  (~{mp.nstr(abs(residual) / ulp_lo, 3)} ulp of lo)")
    print(f"    rust: const {name}: DoubleDouble = DoubleDouble::new({rust_hi}, {lo!r});")
    print()


def main() -> None:
    print("double-double constants (mpmath, 50-digit)\n")

    emit("LN2", mp.log(2), "ln 2 — exp/ln argument reduction", "core::f64::consts::LN_2")
    emit("PI", mp.pi, "pi — atan2 quadrant", "core::f64::consts::PI")
    emit("TWO_PI", 2 * mp.pi, "2*pi — sin/cos reduction modulo a full turn", "core::f64::consts::TAU")
    emit("FRAC_PI_2", mp.pi / 2, "pi/2 — sin/cos octant reduction", "core::f64::consts::FRAC_PI_2")
    emit("FRAC_PI_4", mp.pi / 4, "pi/4 — reduced-argument bound, atan2 corners", "core::f64::consts::FRAC_PI_4")

    # The Taylor loops stop when a term falls below the double-double resolution
    # 2^-104 (~106-bit significand). It is an exact power of two, so its f64 is
    # exact; printed here so the Rust literal is provenanced too.
    eps = mp.mpf(2) ** -104
    print("  DD_EPS  (2^-104, double-double epsilon — Taylor cutoff)")
    print(f"    value = {mp.nstr(eps, 30)}")
    print(f"    rust: const DD_EPS: f64 = {float(eps)!r};")
    print()

    # Reduction-error floor for the sin(1000) stress case, printed explicitly.
    two_pi = 2 * mp.pi
    _, _, res_2pi = split(two_pi)
    n = round(1000.0 / float(two_pi))
    floor_abs = abs(mp.mpf(n) * res_2pi)
    print("  sin(1000) range-reduction floor")
    print(f"    N = round(1000 / 2pi) = {n}")
    print(f"    |2pi - dd(2pi)|            = {mp.nstr(abs(res_2pi), 4)}")
    print(f"    N * |2pi - dd(2pi)| (abs)  = {mp.nstr(floor_abs, 4)}")
    print(f"    / |sin(1000)|      (rel)   = {mp.nstr(floor_abs / abs(mp.sin(1000)), 4)}")


if __name__ == "__main__":
    main()
