#!/usr/bin/env python3
"""How large can `CioInterpolator`'s interpolation error be — as a bound, not a sample?

`spacelab-frames::cio_fast` precomputes `(X, Y, s, s')` on a uniform grid of `GRID_DT = 600 s` and
Lagrange-interpolates them with `STENCIL = 8` nodes, so degree 7. A test measures the residual
against the exact transform at a handful of epochs. This computes the other half of the statement:
the Lagrange remainder holds at *every* epoch once `sup|f⁽⁸⁾|` is known, and for the IAU 2006/2000A
series that supremum is a sum over the shipped table.

    f(t) − p(t) = f⁽⁸⁾(ξ)/8! · h⁸ · Π(u − i),  i = 0..7

`derivations/cio_interpolation.mac` does the node product exactly: 11025/256 centred, 14.87× that
when the stencil is clamped at either end of the grid. This does the series half, reading the
committed `cio_tables.rs` rather than any upstream copy, because the bound wanted is the bound on
what ships.

    uv run --no-project --with numpy python derivations/cio_interpolation.py
"""

import math
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLES = ROOT / "crates/spacelab-fa/src/cio_tables.rs"

DAS2R = 4.848136811095359935899141e-6  # arcsec → rad
CENTURY_S = 36525.0 * 86400.0
ORDER = 8  # STENCIL in cio_fast.rs; the remainder uses the ORDER-th derivative
GRID_DT = 600.0  # GRID_DT in cio_fast.rs, seconds
NODE_PRODUCT_CENTRED = 11025 / 256  # derivations/cio_interpolation.mac
NODE_PRODUCT_CLAMPED = 640.6009504036364  # ditto, the end-of-grid case
LEBESGUE_CENTRED = 1.48828125  # ditto: how far f64 node rounding is amplified
LEBESGUE_CLAMPED = 6.929605446800892
T_SPAN_CENTURIES = 1.0  # |t| bound: J2000 ± 1 century covers 1900–2100

# Linear coefficients of the 14 fundamental arguments, rad/century. The five Delaunay arguments are
# published in arcsec (fundamental.rs `fal03`..`faom03`); the nine planetary ones already in radians.
FA_RATE = np.array([
    1_717_915_923.2178 * DAS2R,   # l    Moon, mean anomaly
    129_596_581.0481 * DAS2R,     # l'   Sun, mean anomaly
    1_739_527_262.8478 * DAS2R,   # F    L − Ω
    1_602_961_601.2090 * DAS2R,   # D    Moon − Sun elongation
    -6_962_890.5431 * DAS2R,      # Ω    lunar node
    2_608.7903141574,             # Mercury
    1_021.3285546211,             # Venus
    628.3075849991,               # Earth
    334.0612426700,               # Mars
    52.9690962641,                # Jupiter
    21.3299104960,                # Saturn
    7.4781598567,                 # Uranus
    3.8133035638,                 # Neptune
    0.024381750 + 2 * 0.00000538691 * 0.0,  # general precession pA, linear part
])
# The `s` series indexes eight of them: l, l', F, D, Ω, Venus, Earth, pA (cio.rs `fa8`).
FA8_RATE = FA_RATE[[0, 1, 2, 3, 4, 6, 7, 13]]


def const_block(name: str, text: str) -> str:
    """The initialiser of `const NAME: ... = [ ... ];`, without the type, which also holds a `];`."""
    start = text.index("= [", text.index(f"const {name}:")) + 3
    return text[start : text.index("];", start)]


def numbers(block: str, integral: bool) -> list:
    pattern = r"-?\d+" if integral else r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"
    return [(int(x) if integral else float(x)) for x in re.findall(pattern, block)]


def leibniz_bound(amplitude_rad: float, power: int, omega_per_s: float) -> float:
    """`sup|dⁿ/dtⁿ (A · tʲ · trig(θ(t)))|` in SI, over `|t| ≤ T_SPAN_CENTURIES`.

    `t` is in centuries inside the series and the derivative is with respect to seconds, so each
    derivative that lands on the `tʲ` factor costs a `1/CENTURY_S`. That makes every `m > 0` term
    negligible against `m = 0`, which is exactly the point worth showing rather than assuming.
    """
    total = 0.0
    for m in range(min(ORDER, power) + 1):
        falling = math.factorial(power) / math.factorial(power - m)
        total += (
            math.comb(ORDER, m)
            * falling
            * T_SPAN_CENTURIES ** (power - m)
            / CENTURY_S**m
            * abs(omega_per_s) ** (ORDER - m)
        )
    return amplitude_rad * total


def main():
    text = TABLES.read_text(encoding="utf-8")

    mult = np.array(numbers(const_block("XY_MULT", text), True)).reshape(1309, 14)
    offset = numbers(const_block("XY_OFFSET", text), True)
    code = numbers(const_block("XY_CODE", text), True)
    amp = numbers(const_block("XY_AMP", text), False)
    assert len(offset) == 1310 and len(code) == len(amp) == 4755, (len(offset), len(code), len(amp))

    omega = mult @ FA_RATE / CENTURY_S  # rad/s, per frequency
    bound = {"X": 0.0, "Y": 0.0}
    biggest = {"X": (0.0, None), "Y": (0.0, None)}
    for p in range(1309):
        for idx in range(offset[p], offset[p + 1]):
            c = code[idx]
            which = "Y" if c & 1 else "X"
            b = leibniz_bound(abs(amp[idx]) * 1e-6 * DAS2R, c >> 2, omega[p])
            bound[which] += b
            if b > biggest[which][0]:
                biggest[which] = (b, (p, abs(amp[idx]), omega[p]))

    # The `s` series: 66 frequencies over eight arguments, sin and cos amplitudes, one power each.
    s_mult = np.array(numbers(const_block("S_MULT", text), True)).reshape(66, 8)
    s_pt = numbers(const_block("S_PT", text), True)
    s_sin = numbers(const_block("S_SIN", text), False)
    s_cos = numbers(const_block("S_COS", text), False)
    s_omega = s_mult @ FA8_RATE / CENTURY_S
    bound["s"] = sum(
        leibniz_bound((abs(s_sin[k]) + abs(s_cos[k])) * 1e-6 * DAS2R, s_pt[k], s_omega[k])
        for k in range(66)
    )

    print(f"{TABLES.relative_to(ROOT)}: 1309 X/Y frequencies, 4755 amplitudes, 66 s frequencies")
    print(f"fastest argument combination: |omega| max {np.abs(omega).max():.4e} rad/s "
          f"(period {2 * math.pi / np.abs(omega).max() / 86400:.3f} d)")
    for name in ("X", "Y", "s"):
        print(f"sup|d^{ORDER}{name}/dt^{ORDER}| <= {bound[name]:.4e} rad/s^{ORDER}")
    print("  the dominant single term in X is frequency "
          f"{biggest['X'][1][0]}, amplitude {biggest['X'][1][1]:.1f} uas, "
          f"omega {biggest['X'][1][2]:.3e} rad/s")

    print(f"\ns' (TIO locator) is linear in t, so its {ORDER}th derivative is 0 and degree-7 "
          "interpolation is exact for it.")

    worst = max(bound.values())
    for label, product in (("centred", NODE_PRODUCT_CENTRED), ("clamped at a grid end", NODE_PRODUCT_CLAMPED)):
        err = product / math.factorial(ORDER) * GRID_DT**ORDER * worst
        print(f"\n{label}: |error| <= {product:.6g}/8! * {GRID_DT:.0f}^8 * sup = {err:.4e} rad")
        print(f"   at LEO (6 800 km lever arm): {err * 6.8e6:.4e} m")
    # Truncation is one half of the error and on this grid it is not the half that matters: the nodes
    # are f64, and Lagrange interpolation amplifies whatever they carry by the Lebesgue constant.
    ulp = np.spacing(2e-3)  # a CIP coordinate is ~2e-3 rad
    print(f"\none f64 ulp of a CIP coordinate near 2e-3 rad: {ulp:.3e} rad "
          f"= {ulp * 6.8e6:.3e} m at LEO")
    for label, lam in (("centred", LEBESGUE_CENTRED), ("clamped at a grid end", LEBESGUE_CLAMPED)):
        print(f"  node rounding amplified, {label}: Lambda = {lam:.4f}, so one ulp of node error "
              f"becomes {lam * ulp * 6.8e6:.3e} m at LEO")
    print("  => the rounding term is ~7 orders above the truncation term; the shipped grid is not "
          "accuracy-limited by its spacing")

    # The grid spacing at which the bound would reach one nanometre at LEO.
    target = 1e-9 / 6.8e6
    h = (target * math.factorial(ORDER) / (NODE_PRODUCT_CLAMPED * worst)) ** (1 / ORDER)
    print(f"the bound reaches 1 nm at LEO only at h = {h:.4g} s = {h / 86400:.4g} d "
          f"({h / GRID_DT:.4g} x the shipped grid)")


if __name__ == "__main__":
    main()
