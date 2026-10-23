#!/usr/bin/env python
"""Celestrak's consolidated Vallado errata, used as a second opinion on two things we implement.

Nothing in this workspace cites *Fundamentals of Astrodynamics and Applications* -- the models come
from the IERS Conventions, the Astronomical Almanac, Montenbruck & Gill and the original papers. That
is what makes the errata useful here: each item names an equation whose most common published form is
wrong, which is precisely where a transcription with a single source fails silently. Where our form
already differs the errata confirm it; where it agrees with the *uncorrected* form, that is a finding.

Two items need an external high-precision oracle rather than a Rust test, and they are the two here.
The geopotential item (4th ed. p. 550, Eq. 8-27) is checked in
``crates/spacelab-propagation/tests/vallado_errata.rs`` instead, because both sides of that
comparison need the embedded coefficients.

**4th ed. p. 277-281, the Sun algorithm.** "The equations should be made identical to the Almanac
version, which is TOD, not MOD." ``ephemeris.rs``'s ``AnalyticSun`` runs the Almanac's low-precision
series and treats its result as referred to the **mean** equator and equinox of date, precessing it
back to J2000 with the IAU 2006 angles and nothing else. If the erratum's frame claim carries over to
the way we consume the series, adding the missing rotation should reduce the residual against an
independent solar ephemeris. ``sun_frame`` measures each candidate against ERFA ``epv00``.

**4th ed. p. 225, Eq. 3-82, the Delaunay arguments.** The erratum replaces the printed polynomials
with the 1996 values in full. ``spacelab-fa``'s ``fal03``..``faom03`` are the IERS Conventions
(2003) polynomials from Simon et al. (1994) -- a *later* realization, so this is a currency
measurement rather than a correctness one: ``delaunay_realizations`` reports how far apart the two
sets run and what that is worth in the nutation series they drive.

    uv run --no-project --with pyerfa --with mpmath python derivations/vallado_errata.py
"""

from __future__ import annotations

import erfa
from mpmath import cos, mp, mpf, pi, sin, sqrt

mp.dps = 40

AU = mpf("1.495978707e11")
SECONDS_PER_JULIAN_CENTURY = mpf("86400") * 36525
JD_J2000 = mpf("2451545.0")
ARCSEC_PER_TURN = mpf(1296000)
ARCSEC_PER_DEGREE = mpf(3600)
# IAU 1976 constant of annual aberration, arcseconds.
CONSTANT_OF_ABERRATION = mpf("20.49552")


def deg(x):
    return x * pi / 180


def arcsec_of(radians):
    return radians * 180 / pi * 3600


# ---------------------------------------------------------------- the Sun's frame


def almanac_of_date(t_seconds):
    """The Astronomical Almanac low-precision Sun, in its own frame: equatorial, of date, AU.

    Written the way the Almanac prints it -- degrees, the equation of centre spelled out, the
    obliquity applied last -- so that this is a second arrangement of the series and not a copy of
    the Rust.
    """
    t = t_seconds / SECONDS_PER_JULIAN_CENTURY
    g = deg(mpf("357.528") + mpf("35999.050") * t)
    lam = deg(mpf("280.460") + mpf("36000.771") * t) + deg(
        mpf("1.915") * sin(g) + mpf("0.020") * sin(2 * g)
    )
    eps = deg(mpf("23.439291") - mpf("0.0130042") * t)
    r = mpf("1.00014") - mpf("0.01671") * cos(g) - mpf("0.00014") * cos(2 * g)
    return [r * cos(lam), r * cos(eps) * sin(lam), r * sin(eps) * sin(lam)]


def erfa_geometric_sun(t_seconds):
    """Geocentric geometric Sun from ERFA ``epv00``, AU, ICRS -- the oracle."""
    helio, _bary = erfa.epv00(float(JD_J2000 + t_seconds / 86400), 0.0)
    return [-mpf(float(c)) for c in helio["p"]]


def erfa_earth_velocity(t_seconds):
    """Barycentric velocity of the Earth, AU/day -- what annual aberration is built from."""
    _helio, bary = erfa.epv00(float(JD_J2000 + t_seconds / 86400), 0.0)
    return [mpf(float(c)) for c in bary["v"]]


def apply(matrix, v, transpose=False):
    if transpose:
        return [sum(matrix[k][i] * v[k] for k in range(3)) for i in range(3)]
    return [sum(matrix[i][k] * v[k] for k in range(3)) for i in range(3)]


def separation_arcsec(a, b):
    na = sqrt(sum(c * c for c in a))
    nb = sqrt(sum(c * c for c in b))
    dot = sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
    return arcsec_of(mp.acos(min(mpf(1), max(-mpf(1), dot))))


def sun_frame(span_years: int, samples: int) -> None:
    """Four readings of the same series, against one oracle."""
    span = mpf(span_years) * mpf("365.25") * 86400
    worst: dict[str, mpf] = {}
    total: dict[str, mpf] = {}
    for i in range(samples):
        t = -span + 2 * span * i / (samples - 1)
        jd = float(JD_J2000 + t / 86400)
        truth = erfa_geometric_sun(t)
        v = almanac_of_date(t)
        precession = [[mpf(float(c)) for c in row] for row in erfa.pmat06(jd, 0.0)]
        nutated = [[mpf(float(c)) for c in row] for row in erfa.pnm06a(jd, 0.0)]
        dpsi, _deps = erfa.nut06a(jd, 0.0)

        # Aberration is removed in the frame the series' vector already lives in: rotate the Earth's
        # barycentric velocity into that frame rather than the other way round.
        speed_of_light = mpf(float(erfa.DC))  # AU/day
        v_earth = apply(precession, erfa_earth_velocity(t))
        norm = sqrt(sum(c * c for c in v))
        deaberrated = [
            (c / norm - w / speed_of_light) * norm
            for c, w in zip(v, v_earth, strict=True)
        ]

        # The same series with nutation in longitude taken out before the obliquity is applied --
        # the reading the erratum's "TOD, not MOD" implies.
        t_cy = t / SECONDS_PER_JULIAN_CENTURY
        g = deg(mpf("357.528") + mpf("35999.050") * t_cy)
        lam = (
            deg(mpf("280.460") + mpf("36000.771") * t_cy)
            + deg(mpf("1.915") * sin(g) + mpf("0.020") * sin(2 * g))
            - mpf(float(dpsi))
        )
        eps = deg(mpf("23.439291") - mpf("0.0130042") * t_cy)
        r = mpf("1.00014") - mpf("0.01671") * cos(g) - mpf("0.00014") * cos(2 * g)
        de_nutated = [r * cos(lam), r * cos(eps) * sin(lam), r * sin(eps) * sin(lam)]

        # The classical constant-of-aberration form, which is what the Almanac documents and what a
        # `no_std` implementation can afford: the apparent longitude runs kappa/r behind the
        # geometric one, with no Earth velocity anywhere. Whether that is enough is the measurement.
        lam_classical = (
            deg(mpf("280.460") + mpf("36000.771") * t_cy)
            + deg(mpf("1.915") * sin(g) + mpf("0.020") * sin(2 * g))
            + deg(CONSTANT_OF_ABERRATION / (ARCSEC_PER_DEGREE * r))
        )
        classical = [
            r * cos(lam_classical),
            r * cos(eps) * sin(lam_classical),
            r * sin(eps) * sin(lam_classical),
        ]

        candidates = {
            "no rotation at all": v,
            "mean of date -> J2000 (today)": apply(precession, v, transpose=True),
            "true of date -> J2000": apply(nutated, v, transpose=True),
            "- nutation, then precession": apply(
                precession, de_nutated, transpose=True
            ),
            "- aberration (v_earth), then precession": apply(
                precession, deaberrated, transpose=True
            ),
            "- aberration (kappa/r), then precession": apply(
                precession, classical, transpose=True
            ),
        }
        for name, candidate in candidates.items():
            s = separation_arcsec(candidate, truth)
            worst[name] = max(worst.get(name, mpf(0)), s)
            total[name] = total.get(name, mpf(0)) + s * s

    print(
        f"// Sun: the Almanac series against ERFA epv00, +/-{span_years} y, {samples} epochs"
    )
    for name in worst:
        rms = sqrt(total[name] / samples)
        print(
            f"//   {name:30s} worst {mp.nstr(worst[name], 4):>10s} arcsec"
            f"   rms {mp.nstr(rms, 4):>10s} arcsec"
        )


# ------------------------------------------------------- the Delaunay realizations

# Ours: IERS Conventions (2003) from Simon et al. (1994), arcseconds, quartic in t.
IERS2003 = {
    "l  (Moon anomaly)": [
        "485868.249036",
        "1717915923.2178",
        "31.8792",
        "0.051635",
        "-0.00024470",
    ],
    "l' (Sun anomaly)": [
        "1287104.793048",
        "129596581.0481",
        "-0.5532",
        "0.000136",
        "-0.00001149",
    ],
    "F  (arg latitude)": [
        "335779.526232",
        "1739527262.8478",
        "-12.7512",
        "-0.001037",
        "0.00000417",
    ],
    "D  (elongation)": [
        "1072260.703692",
        "1602961601.2090",
        "-6.3706",
        "0.006593",
        "-0.00003169",
    ],
    "Om (node)": [
        "450160.398036",
        "-6962890.5431",
        "7.4722",
        "0.007702",
        "-0.00005939",
    ],
}

# The errata's replacement for 4th ed. Eq. 3-82: degrees and revolutions, cubic in t. Transcribed as
# printed; the conversion to arcseconds happens below rather than by hand, so the published digits
# stay visible.
ERRATA_1996 = {
    "l  (Moon anomaly)": ("134.96298139", 1325, "198.8673981", "0.0086972", "1.78e-5"),
    "l' (Sun anomaly)": ("357.52772333", 99, "359.0503400", "-0.0001603", "-3.3e-6"),
    "F  (arg latitude)": ("93.27191028", 1342, "82.0175381", "-0.0036825", "3.1e-6"),
    "D  (elongation)": ("297.85036306", 1236, "307.1114800", "-0.0019142", "5.3e-6"),
    # Printed with the revolutions and the linear degrees both *subtracted*.
    "Om (node)": ("125.04452222", -5, "-134.1362608", "0.0020708", "2.2e-6"),
}


def horner(coefficients, t):
    total = mpf(0)
    for c in reversed(coefficients):
        total = total * t + mpf(c)
    return total


def errata_arcseconds(entry, t):
    constant, revolutions, linear_deg, quadratic_deg, cubic_deg = entry
    return horner(
        [
            mpf(constant) * ARCSEC_PER_DEGREE,
            mpf(revolutions) * ARCSEC_PER_TURN + mpf(linear_deg) * ARCSEC_PER_DEGREE,
            mpf(quadratic_deg) * ARCSEC_PER_DEGREE,
            mpf(cubic_deg) * ARCSEC_PER_DEGREE,
        ],
        t,
    )


def delaunay_realizations(span_years: int, samples: int) -> None:
    """How far the errata's 1996 arguments run from the IERS 2003 ones we implement."""
    print(
        f"// Delaunay arguments: IERS 2003 (ours) vs the errata's 1996 values,"
        f" +/-{span_years} y, {samples} epochs"
    )
    worst_node = mpf(0)
    for name in IERS2003:
        worst = mpf(0)
        at_epoch = mpf(0)
        for i in range(samples):
            t = mpf(-span_years + 2 * span_years * i / (samples - 1)) / 100
            gap = abs(
                horner(IERS2003[name], t) - errata_arcseconds(ERRATA_1996[name], t)
            )
            worst = max(worst, gap)
            if i == (samples - 1) // 2:
                at_epoch = gap
        if name.startswith("Om"):
            worst_node = worst
        print(
            f"//   {name:20s} at J2000 {mp.nstr(at_epoch, 4):>9s} arcsec,"
            f" worst {mp.nstr(worst, 4):>9s} arcsec"
        )
    # The nutation term the node drives is -17.2 sin(Om) in longitude, so a node error costs
    # 17.2 cos(Om) of it -- the sharpest way to say what the realization difference is worth.
    print(
        f"//   the node gap is worth <= {mp.nstr(mpf('17.2') * worst_node / arcsec_of(1), 4)}"
        " arcsec of nutation in longitude through the -17.2 sin(Om) term"
    )


if __name__ == "__main__":
    sun_frame(25, 2001)
    sun_frame(100, 2001)
    delaunay_realizations(100, 2001)
