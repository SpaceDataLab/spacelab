#!/usr/bin/env python3
"""astropy as an oracle for the rise/set and twilight hour angle.

`spacelab_coverage::observability::diurnal_arc` solves one spherical triangle:
`cos H = (sin h₀ − sin φ sin δ)/(cos φ cos δ)`, with a polar-day and a polar-night branch where
`|cos H| > 1`. #279 verified the branches — Tromsø at both solstices, an exactly-π/2 equatorial
equinox arc — but nothing pinned an hour angle against an independent implementation, because
`spacelab-coverage` has no time system and no solar ephemeris and should not grow either for this.

An integration test can own both. astropy's frame machinery (ERFA underneath) is an independent
implementation of the whole chain, and this script asks it two questions at one instant:

  1. **When does the Sun's altitude cross the threshold?** Found by bisection on `AltAz` altitude
     with `pressure = 0`, so the altitude is geometric — the standard −0°50′ already folds in the
     34′ of refraction and the 16′ semidiameter that a sunset convention assumes, and applying
     astropy's refraction on top would count it twice.
  2. **What are the hour angle and declination there?** Read off the `HADec` frame at the same
     instant, not computed from the altitude — computing them from the altitude would be the
     formula under test, run backwards.

**Why both come from the same instant.** Neither frame is reachable without UT1, and this runs with
IERS auto-download off, so UT1 = UTC and the sidereal angle is up to 0.9 s of rotation wrong. That
error moves *which UTC* is called sunset; it does not move the `(H, δ, h)` triple at a given
instant, because `AltAz` and `HADec` are the same ERFA rotation stopped one step apart. Taking both
at one instant cancels it exactly, which is why the fixture carries no times.

**What is left in the residual**, deliberately rather than by omission: astropy is topocentric and
`diurnal_arc` is not. The Sun's diurnal parallax is 8.79″, and near the horizon an altitude error
divides by `cos φ cos δ sin H` to become an hour angle — under a second of time at mid latitudes,
unbounded as `sin H → 0`. The Rust side measures it rather than tolerating it, and the sites are
chosen to include one where it is large.

    uv run --with astropy --with numpy --with scipy python tools/gen_twilight_oracle.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, HADec, get_sun
from astropy.time import Time
from astropy.utils import iers
from scipy.optimize import brentq

# No network, and no silent dependence on a downloaded table: the 0.9 s of UT1 this costs cancels
# between the two frames, and a generator that reaches the network is a generator that stops
# reproducing.
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "ignore"

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crates/spacelab-coverage/tests/fixtures/twilight_hour_angle.txt"
)

# (name, geodetic latitude deg, longitude deg, height m)
SITES = [
    ("quito", 0.0, -78.5, 2850.0),
    ("mauna_kea", 19.8207, -155.4681, 4205.0),
    ("paranal", -24.6272, -70.4039, 2635.0),
    ("greenwich", 51.4779, 0.0, 47.0),
    ("tromso", 69.6489, 18.9551, 10.0),
]

# Solstices and equinoxes, so the solar declination spans its range.
DATES = ["2026-03-20", "2026-06-21", "2026-09-23", "2026-12-21"]

# The four thresholds the module exposes: rise/set with its refraction-and-semidiameter allowance,
# then civil, nautical and astronomical twilight.
THRESHOLDS = {
    "rise_set": -0.833_333_333_333_333_3,
    "civil": -6.0,
    "nautical": -12.0,
    "astronomical": -18.0,
}


def altitudes(location: EarthLocation, when: Time) -> np.ndarray:
    """Geometric altitude of the Sun's centre, degrees, over an array of instants.

    Vectorised because it has to be: one `transform_to` over 289 epochs is two orders of magnitude
    cheaper than 289 of them, and the scan below needs every one of those samples.

    No refraction — `AltAz`'s `pressure` defaults to zero — because the -0d50' a sunset convention
    uses already contains 34' of it, and applying astropy's on top would count it twice.
    """
    frame = AltAz(obstime=when, location=location)
    return np.atleast_1d(get_sun(when).transform_to(frame).alt.to_value(u.deg))


def altitude(location: EarthLocation, when: Time) -> float:
    return float(altitudes(location, when)[0])


def hour_angle_and_declination(
    location: EarthLocation, when: Time
) -> tuple[float, float]:
    """`(H, δ)` in radians from astropy's own frame, not from the altitude."""
    frame = HADec(obstime=when, location=location)
    sun = get_sun(when).transform_to(frame)
    return (
        float(sun.ha.wrap_at(180.0 * u.deg).to_value(u.rad)),
        float(sun.dec.to_value(u.rad)),
    )


def setting_crossing(
    location: EarthLocation, day: str, threshold_deg: float
) -> Time | None:
    """The instant the Sun's altitude falls through `threshold_deg`, or `None` if it never does."""
    start = Time(f"{day} 00:00:00", scale="utc")
    # 5-minute scan for a downward crossing, then bisect inside the bracket it found.
    steps = 24 * 12 + 1
    offsets = np.linspace(0.0, 1.0, steps)
    samples = altitudes(location, start + offsets * u.day)
    for i in range(steps - 1):
        if samples[i] > threshold_deg >= samples[i + 1]:
            root = brentq(
                lambda x: altitude(location, start + x * u.day) - threshold_deg,
                offsets[i],
                offsets[i + 1],
                xtol=1e-12,
            )
            return start + float(root) * u.day
    return None


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Sun altitude crossings from astropy (ERFA), as an oracle for diurnal_arc.",
        "# Generated by tools/gen_twilight_oracle.py -- do not edit.",
        "#",
        "# Hour angle and declination are read from astropy's HADec frame at the instant its AltAz",
        "# altitude crosses the threshold, so the UT1 the generator does not have cancels.",
        "#",
        "# site date event latitude_deg declination_rad threshold_deg hour_angle_rad outcome",
    ]
    counts = {"rises": 0, "always_above": 0, "always_below": 0}

    for name, latitude, longitude, height in SITES:
        location = EarthLocation(
            lat=latitude * u.deg, lon=longitude * u.deg, height=height * u.m
        )
        for day in DATES:
            for event, threshold in THRESHOLDS.items():
                when = setting_crossing(location, day, threshold)
                if when is None:
                    # No crossing: sample noon and midnight to say which side it stayed on, and
                    # record the declination at local noon so the Rust side has a δ to ask about.
                    start = Time(f"{day} 00:00:00", scale="utc")
                    offsets = np.linspace(0.0, 1.0, 24 * 4 + 1)
                    samples = altitudes(location, start + offsets * u.day)
                    peak = start + float(offsets[int(np.argmax(samples))]) * u.day
                    _, declination = hour_angle_and_declination(location, peak)
                    outcome = (
                        "always_above" if min(samples) > threshold else "always_below"
                    )
                    counts[outcome] += 1
                    rows.append(
                        f"{name:12s} {day} {event:12s} {latitude:22.15e} "
                        f"{declination:22.15e} {threshold:22.15e} "
                        f"{float('nan'):22.15e} {outcome}"
                    )
                    continue
                hour_angle, declination = hour_angle_and_declination(location, when)
                counts["rises"] += 1
                rows.append(
                    f"{name:12s} {day} {event:12s} {latitude:22.15e} "
                    f"{declination:22.15e} {threshold:22.15e} "
                    f"{hour_angle:22.15e} rises"
                )

    FIXTURE.write_text("\n".join(rows) + "\n")
    print(f"twilight_hour_angle.txt: {sum(counts.values())} rows {counts}")


if __name__ == "__main__":
    main()
