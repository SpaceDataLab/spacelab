#!/usr/bin/env python
"""IERS mean-pole models, at 50 digits, as golden tables for `pole_wobble`.

The Rust `solid_tide::pole_wobble` evaluates these by Horner in *seconds* since J2000, with the
mas-to-radian factor pre-divided by the Julian year once per power (`F1 = F0/yr`, `F2 = F1/yr`,
`F3 = F2/yr`). This script evaluates the same models the way the Conventions print them -- in
milliarcseconds, as polynomials in *years* with direct powers -- and converts at the end. Same
number, different arrangement and different precision, so a rearranged Horner nest is a
disagreement rather than a shared assumption.

Two models, because the Conventions replaced one with the other.

`Iers2010`, Table 7.7 of the 2010 printing, `t = year − 2000.0`:

    before 2010.0   x̄p = 55.974 + 1.8243 t + 0.18413 t² + 0.007024 t³   (mas)
                    ȳp = 346.346 + 1.7896 t − 0.10729 t² − 0.000908 t³  (mas)
    from 2010.0     x̄p = 23.513 + 7.6141 t                              (mas)
                    ȳp = 358.891 − 0.6287 t                             (mas)

The two branches meet in `x̄p` exactly (99.654 mas at t = 10) but not in `ȳp`, which steps by
0.001 mas -- 3e-6 relative. That discontinuity is what makes the changeover epoch worth a golden:
it is the only place the `t <= T_CHANGE` comparison is observable.

`Secular2018`, Eq. (21) of the 2018/02/01 update to Chapter 7 -- a single straight line, fitted to
1900-2017 polar motion, valid at every epoch, and what §6.4/§6.5 now mean by `(m1, m2)`:

    xs = 55.0 + 1.677 (t − 2000),   ys = 320.5 + 3.460 (t − 2000)        (mas)

"where t is the date in years of 365.25 days" -- the same Julian year the 2010 table uses, so the
`JULIAN_YEAR` divisor is shared and only the coefficients differ.

    uv run --no-project --with mpmath python derivations/pole_wobble.py
"""

from __future__ import annotations

from mpmath import mp, mpf

mp.dps = 50

ARC_SEC_TO_RAD = mpf("4.84813681109536e-6")  # IERS/Hipparchus ARC_SECONDS_TO_RADIANS
JULIAN_YEAR = mpf("31557600")
MAS_TO_RAD = ARC_SEC_TO_RAD / 1000
T_CHANGE = mpf("3652.5") * 86400  # 2010-01-01 00:00 TT, seconds since J2000

# The polar motion of the existing Orekit fixture epoch, in radians -- so the returned
# `(m1, m2) = (xp − x̄p, ȳp − yp)` pins the two subtractions and their opposite signs.
XP = mpf("6.636808406180881e-7")
YP = mpf("9.802205411513153e-7")

EPOCHS = [
    ("-1.5e8", mpf("-1.5e8")),  # 1995.25: negative t, so the odd powers change sign
    ("0.0", mpf(0)),  # 2000.0: the constant term alone
    ("1.0e8", mpf("1.0e8")),  # 2003.17
    ("3.0e8", mpf("3.0e8")),  # 2009.51, still the cubic
    ("315576000.0", T_CHANGE),  # 2010.0 exactly -- the last epoch on the cubic branch
    ("315576001.0", T_CHANGE + 1),  # one second later -- the linear extension
    ("7.57339269184e8", mpf("7.57339269184e8")),  # 2024.0, the Orekit fixture epoch
]


def mean_pole_2010(t_seconds: mpf) -> tuple[mpf, mpf]:
    """`(x̄p, ȳp)` in radians at `t_seconds` since J2000, IERS 2010 Table 7.7."""
    t = t_seconds / JULIAN_YEAR
    if t_seconds <= T_CHANGE:
        x = (
            mpf("55.974")
            + mpf("1.8243") * t
            + mpf("0.18413") * t**2
            + mpf("0.007024") * t**3
        )
        y = (
            mpf("346.346")
            + mpf("1.7896") * t
            - mpf("0.10729") * t**2
            - mpf("0.000908") * t**3
        )
    else:
        x = mpf("23.513") + mpf("7.6141") * t
        y = mpf("358.891") - mpf("0.6287") * t
    return x * MAS_TO_RAD, y * MAS_TO_RAD


def secular_pole_2018(t_seconds: mpf) -> tuple[mpf, mpf]:
    """`(xs, ys)` in radians at `t_seconds` since J2000, Chapter 7 update Eq. (21)."""
    t = t_seconds / JULIAN_YEAR
    x = mpf("55.0") + mpf("1.677") * t
    y = mpf("320.5") + mpf("3.460") * t
    return x * MAS_TO_RAD, y * MAS_TO_RAD


def report(name: str, model) -> None:
    print(f"// {name}: (t seconds since J2000, m1 at xp = yp = 0, m2 at xp = yp = 0)")
    for label, t in EPOCHS:
        x_bar, y_bar = model(t)
        print(f"    ({label}, {mp.nstr(-x_bar, 25)}, {mp.nstr(y_bar, 25)}),")
    print()
    print(f"// {name}: at xp = {mp.nstr(XP, 20)}, yp = {mp.nstr(YP, 20)}")
    for label, t in EPOCHS:
        x_bar, y_bar = model(t)
        print(f"    ({label}, {mp.nstr(XP - x_bar, 25)}, {mp.nstr(y_bar - YP, 25)}),")
    print()


def main() -> None:
    report("Iers2010", mean_pole_2010)
    report("Secular2018", secular_pole_2018)
    print("// how far apart the two models put the pole, in mas")
    for label, t in EPOCHS:
        x10, y10 = mean_pole_2010(t)
        x18, y18 = secular_pole_2018(t)
        dx = (x10 - x18) / MAS_TO_RAD
        dy = (y10 - y18) / MAS_TO_RAD
        print(f"    // {label}: dx {mp.nstr(dx, 8)}, dy {mp.nstr(dy, 8)}")


if __name__ == "__main__":
    main()
