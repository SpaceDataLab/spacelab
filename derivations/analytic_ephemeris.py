#!/usr/bin/env python
"""Goldens and an accuracy budget for the analytic Sun and Moon of `src/ephemeris.rs`.

Three things are checked here, because one alone would not be enough.

**Is the series the published one?** Both models are transcribed tables -- the Astronomical
Almanac's low-precision solar formulae and Montenbruck & Gill's lunar series (`Satellite Orbits`
sec. 3.3.2, the Brown series also printed as `Moon` in Montenbruck & Pfleger). Re-typing the same
amplitudes in Python and comparing would only prove that two copies of one transcription agree, so
the transcription here is first held against **ERFA** -- `epv00` for the Earth-Sun vector and
`moon98`, an abridged ELP2000, for the Moon. Those are independent of the series in origin and in
method, and they resolve the arcsecond, so agreement at the accuracy the module's docstring claims
(0.01 deg and 0.0003 AU for the Sun, a few arcminutes and a few hundred kilometres for the Moon) is
evidence that both transcriptions are right.

**Is the code that series?** The arrangement here is the published one -- degrees and arcseconds,
sums written out, the rotation applied at the end -- against the module's radians-as-you-go Horner.
At 50 digits the goldens below then pin every amplitude and every argument in the Rust: a single
sign or coefficient changed anywhere moves a printed digit.

**Is it in the frame it claims?** It was not. Held against ERFA in the ICRS the solar series is
out by up to 5054 arcsec (1.4 deg) at the ends of the span -- 120x its own accuracy -- and the
residual is precession exactly: rotate ERFA to the mean equator and equinox of date and the
disagreement drops to 42 arcsec. The Almanac's series is of-date (the obliquity's `-0.0130042 T`
is precession in the obliquity; the mean longitude's rate is the tropical one), so `sun` here
precesses it back to J2000, and `precession_check` holds those angles against `erfa.pmat06`. The
lunar series is already J2000-referred and is left alone.

**Is it geometry?** The Almanac's solar formulae print the *apparent* longitude, which carries
annual aberration; every consumer of this module wants a geometric vector, so `sun` adds `kappa/r`
back. `light_time_check` asks the same question of the lunar series, where the answer is different
and has to be measured rather than assumed.

    uv run --no-project --with pyerfa --with mpmath python derivations/analytic_ephemeris.py
"""

from __future__ import annotations

import erfa
from mpmath import cos, floor, mp, mpf, pi, sin, sqrt

mp.dps = 50

AU = mpf("1.495978707e11")  # IAU 2012, metres
SECONDS_PER_JULIAN_CENTURY = mpf("86400") * 36525
JD_J2000 = mpf("2451545.0")
# IAU 1976 constant of annual aberration, arcseconds.
CONSTANT_OF_ABERRATION = mpf("20.49552")


def deg(x):
    return x * pi / 180


def arcsec(x):
    return deg(x) / 3600


def frac(x):
    return x - floor(x)


def precession_matrix(t):
    """Mean equator and equinox of J2000 -> of date, by the IAU 2006 equatorial angles.

    Capitaine, Wallace & Chapront (2003), arcseconds; also IERS TN36 eq. (5.?). Excludes the
    frame bias between J2000 and the GCRS, which is ~17 mas -- three orders below the series.
    """
    zeta = arcsec(
        mpf("2.650545")
        + t
        * (
            mpf("2306.083227")
            + t
            * (
                mpf("0.2988499")
                + t
                * (
                    mpf("0.01801828")
                    + t * (mpf("-0.000005971") + t * mpf("-0.0000003173"))
                )
            )
        )
    )
    z = arcsec(
        mpf("-2.650545")
        + t
        * (
            mpf("2306.077181")
            + t
            * (
                mpf("1.0927348")
                + t
                * (
                    mpf("0.01826837")
                    + t * (mpf("-0.000028596") + t * mpf("-0.0000002904"))
                )
            )
        )
    )
    theta = arcsec(
        t
        * (
            mpf("2004.191903")
            + t
            * (
                mpf("-0.4294934")
                + t
                * (
                    mpf("-0.04182264")
                    + t * (mpf("-0.000007089") + t * mpf("-0.0000001274"))
                )
            )
        )
    )
    cze, sze = cos(zeta), sin(zeta)
    cz, sz = cos(z), sin(z)
    cth, sth = cos(theta), sin(theta)
    return [
        [cze * cth * cz - sze * sz, -sze * cth * cz - cze * sz, -sth * cz],
        [cze * cth * sz + sze * cz, -sze * cth * sz + cze * cz, -sth * sz],
        [cze * sth, -sze * sth, cth],
    ]


def j2000_from_of_date(t, v):
    """Apply the transpose of `precession_matrix`: mean of date -> J2000."""
    p = precession_matrix(t)
    return [sum(p[k][i] * v[k] for k in range(3)) for i in range(3)]


def sun(seconds_past_j2000):
    """Geocentric equatorial Sun, metres. Astronomical Almanac low-precision formulae.

    The series is referred to the mean equinox and equator *of date* -- the obliquity's
    `-0.0130042 T` is precession in the obliquity, and the mean longitude's rate is the tropical
    one -- so the result is precessed back to J2000, which is the frame the module promises.

    The Almanac prints the *apparent* longitude, which trails the geometric one by the constant of
    aberration over the Earth-Sun distance. Every consumer here wants geometry, so `kappa/r` is
    added back; `derivations/vallado_errata.py` measures that against the full `v_earth/c` form and
    against ERFA `epv00`.
    """
    t = seconds_past_j2000 / SECONDS_PER_JULIAN_CENTURY
    mean_longitude = mpf("280.460") + mpf("36000.771") * t  # degrees
    mean_anomaly = deg(mpf("357.528") + mpf("35999.050") * t)
    r_au = (
        mpf("1.00014")
        - mpf("0.01671") * cos(mean_anomaly)
        - mpf("0.00014") * cos(2 * mean_anomaly)
    )
    ecliptic_longitude = deg(
        mean_longitude
        + mpf("1.915") * sin(mean_anomaly)
        + mpf("0.020") * sin(2 * mean_anomaly)
        + CONSTANT_OF_ABERRATION / (3600 * r_au)
    )
    obliquity = deg(mpf("23.439291") - mpf("0.0130042") * t)
    r = r_au * AU
    return j2000_from_of_date(
        t,
        [
            r * cos(ecliptic_longitude),
            r * cos(obliquity) * sin(ecliptic_longitude),
            r * sin(obliquity) * sin(ecliptic_longitude),
        ],
    )


def moon(seconds_past_j2000):
    """Geocentric equatorial Moon, metres. Montenbruck & Gill sec. 3.3.2 (Brown series)."""
    t = seconds_past_j2000 / SECONDS_PER_JULIAN_CENTURY
    turn = 2 * pi
    l0 = frac(
        mpf("0.606433") + mpf("1336.851344") * t
    )  # revolutions, kept as a fraction
    ll = turn * frac(mpf("0.374897") + mpf("1325.552410") * t)  # Moon's mean anomaly
    lp = turn * frac(mpf("0.993133") + mpf("99.997361") * t)  # Sun's mean anomaly
    d = turn * frac(mpf("0.827361") + mpf("1236.853086") * t)  # mean elongation
    f = turn * frac(mpf("0.259086") + mpf("1342.227825") * t)  # argument of latitude

    # Longitude correction, arcseconds.
    d_l = (
        mpf("22640") * sin(ll)
        - mpf("4586") * sin(ll - 2 * d)
        + mpf("2370") * sin(2 * d)
        + mpf("769") * sin(2 * ll)
        - mpf("668") * sin(lp)
        - mpf("412") * sin(2 * f)
        - mpf("212") * sin(2 * ll - 2 * d)
        - mpf("206") * sin(ll + lp - 2 * d)
        + mpf("192") * sin(ll + 2 * d)
        - mpf("165") * sin(lp - 2 * d)
        - mpf("125") * sin(d)
        - mpf("110") * sin(ll + lp)
        + mpf("148") * sin(ll - lp)
        - mpf("55") * sin(2 * f - 2 * d)
    )
    longitude = turn * frac(l0 + d_l / 1296000)

    s = f + arcsec(d_l + mpf("412") * sin(2 * f) + mpf("541") * sin(lp))
    h = f - 2 * d
    n = (
        -mpf("526") * sin(h)
        + mpf("44") * sin(ll + h)
        - mpf("31") * sin(-ll + h)
        - mpf("23") * sin(lp + h)
        + mpf("11") * sin(-lp + h)
        - mpf("25") * sin(-2 * ll + f)
        + mpf("21") * sin(-ll + f)
    )
    latitude = arcsec(mpf("18520") * sin(s) + n)

    r_km = (
        mpf("385000")
        - mpf("20905") * cos(ll)
        - mpf("3699") * cos(2 * d - ll)
        - mpf("2956") * cos(2 * d)
        - mpf("570") * cos(2 * ll)
        + mpf("246") * cos(2 * ll - 2 * d)
        - mpf("205") * cos(lp - 2 * d)
        - mpf("171") * cos(ll + 2 * d)
        - mpf("152") * cos(ll + lp - 2 * d)
    )
    r = r_km * 1000

    x_ec = r * cos(longitude) * cos(latitude)
    y_ec = r * sin(longitude) * cos(latitude)
    z_ec = r * sin(latitude)
    eps = deg(mpf("23.43929111"))  # J2000 mean obliquity
    return [x_ec, cos(eps) * y_ec - sin(eps) * z_ec, sin(eps) * y_ec + cos(eps) * z_ec]


def erfa_sun(seconds_past_j2000):
    """Geocentric equatorial Sun from ERFA `epv00` (the Earth's heliocentric vector, negated)."""
    jd = JD_J2000 + seconds_past_j2000 / 86400
    heliocentric, _barycentric = erfa.epv00(float(jd), 0.0)
    return [-mpf(float(c)) * AU for c in heliocentric["p"]]


def erfa_moon(seconds_past_j2000):
    """Geocentric equatorial Moon from ERFA `moon98` (abridged ELP2000)."""
    jd = JD_J2000 + seconds_past_j2000 / 86400
    position, _velocity = erfa.moon98(float(jd), 0.0)
    return [mpf(float(c)) * AU for c in position]


def norm(v):
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def separation_arcsec(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True)) / (norm(a) * norm(b))
    return 3600 * 180 / pi * mp.acos(min(mpf(1), max(-mpf(1), dot)))


# One century either side of J2000, the span the module claims. Seconds, exactly representable --
# integers below 2^53 and short binary fractions round-trip through f64, so the Rust evaluates the
# same argument these goldens were computed at. The four epochs `sweep` reports as worst are
# included, rounded to the second, so the accuracy assertions witness their own bound.
GOLDEN_EPOCHS = [
    ("-3.15576e9", mpf("-3.15576e9")),  # 1900.0
    ("-1.5e9", mpf("-1.5e9")),
    ("-7.84308258e8", mpf("-784308258")),  # worst lunar angle
    ("0.0", mpf(0)),  # 2000-01-01 12:00 TT
    ("8.64e7", mpf("8.64e7")),  # +1000 days
    ("7.57339269184e8", mpf("7.57339269184e8")),  # 2024.0, the tide oracles' epoch
    ("1.5e9", mpf("1.5e9")),
    ("2.698761487e9", mpf("2698761487")),  # worst solar range
    ("3.019895577e9", mpf("3019895577")),  # worst lunar range
    ("3.069300822e9", mpf("3069300822")),  # worst solar angle
    ("3.15576e9", mpf("3.15576e9")),  # 2100.0
    # Three centuries out, far outside the span either series is *accurate* over. Transcription
    # pins only: the quintic precession terms are ~1e-12 of the answer at one century, which is
    # the f64 evaluation noise, and only here do they clear it by two orders.
    ("-9.46728e9", mpf("-9.46728e9")),  # 1700.0
    ("9.46728e9", mpf("9.46728e9")),  # 2300.0
]


def precession_check() -> None:
    """The precession angles against ERFA `pmat06`, whose frame bias we deliberately omit."""
    span = mpf("3.15576e9")
    worst = mpf(0)
    for i in range(201):
        seconds = -span + 2 * span * i / 200
        mine = precession_matrix(seconds / SECONDS_PER_JULIAN_CENTURY)
        theirs = erfa.pmat06(float(JD_J2000 + seconds / 86400), 0.0)
        worst = max(
            worst,
            max(
                abs(mine[r][c] - mpf(float(theirs[r][c])))
                for r in range(3)
                for c in range(3)
            ),
        )
    print(
        f"// precession vs ERFA pmat06 over +/-1 century: element <= {mp.nstr(worst, 4)} "
        f"(= {mp.nstr(worst * 3600 * 180 / pi, 4)} arcsec, the omitted frame bias)"
    )


def light_time_check() -> None:
    """Whether the lunar series is apparent or geometric, decided by measurement.

    An apparent position at `t` is the geometric one at `t - tau`, so if the Brown series carried
    light time then evaluating it at `t + tau` would move it *toward* `moon98`. The Sun's answer to
    this question was worth 39 % of its RMS; the Moon's is worth reporting either way.
    """
    span = mpf("3.15576e9")
    speed_of_light = mpf("299792458")
    plain = mpf(0)
    shifted = mpf(0)
    worst_shift = mpf(0)
    for i in range(512):
        t = -span + 2 * span * i / 511
        truth = erfa_moon(t)
        tau = norm(moon(t)) / speed_of_light
        plain = max(plain, separation_arcsec(moon(t), truth))
        shifted = max(shifted, separation_arcsec(moon(t + tau), truth))
        worst_shift = max(worst_shift, separation_arcsec(moon(t + tau), moon(t)))
    print(
        f"// Moon light time over +/-1 century: as printed {mp.nstr(plain, 4)} arcsec, "
        f"with the {mp.nstr(worst_shift, 3)} arcsec light-time shift applied "
        f"{mp.nstr(shifted, 4)} arcsec"
    )


def sweep() -> None:
    """Worst disagreement with ERFA over the claimed span, at 512 epochs."""
    span = mpf("3.15576e9")
    for label, model, oracle, kind in [
        ("Sun ", sun, erfa_sun, "AU"),
        ("Moon", moon, erfa_moon, "km"),
    ]:
        worst_angle = (mpf(0), mpf(0))
        worst_range = (mpf(0), mpf(0))
        for i in range(512):
            t = -span + 2 * span * i / 511
            a, b = model(t), oracle(t)
            scale = AU if kind == "AU" else mpf(1000)
            worst_angle = max(worst_angle, (separation_arcsec(a, b), t))
            worst_range = max(worst_range, (abs(norm(a) - norm(b)) / scale, t))
        print(
            f"// {label} vs ERFA over +/-1 century, 512 epochs: "
            f"angle <= {mp.nstr(worst_angle[0], 4)} arcsec "
            f"({mp.nstr(worst_angle[0] / 3600, 3)} deg) at t={mp.nstr(worst_angle[1], 17)}, "
            f"range <= {mp.nstr(worst_range[0], 4)} {kind} "
            f"at t={mp.nstr(worst_range[1], 17)}"
        )


def goldens() -> None:
    for label, model, oracle in [("SUN", sun, erfa_sun), ("MOON", moon, erfa_moon)]:
        print(f"\n// {label}: (t, series x, y, z, ERFA x, y, z) in metres")
        for name, t in GOLDEN_EPOCHS:
            a, b = model(t), oracle(t)
            cells = ", ".join(mp.nstr(c, 22) for c in [*a, *b])
            print(f"    ({name}, {cells}),")


if __name__ == "__main__":
    precession_check()
    light_time_check()
    sweep()
    goldens()
