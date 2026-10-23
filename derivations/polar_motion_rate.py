#!/usr/bin/env python3
"""How fast does the pole actually move, and how much does `teme_to_itrf` lose by dropping it?

`spacelab-frames::teme::teme_to_itrf` attaches a single angular velocity `omega*z` to a two-rotation
chain. `derivations/frame_transform.mac` proves the sidereal half of that is exact -- `rot3` fixes
`z`, so the axis survives the first rotation unchanged -- which leaves the rate of polar motion as
the whole of the neglected term. Its size is a measurement, not an identity, so it is measured here
against the IERS series itself rather than estimated from a quoted wobble amplitude.

    tools/fetch-eop.sh                                    # populates data/eop/
    uv run --no-project --with numpy python derivations/polar_motion_rate.py
"""

import glob

import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)
DAY = 86400.0
EARTH_RADIUS_M = 6378137.0
EARTH_ROTATION_RATE = 7.292115146706979e-5  # teme.rs's EARTH_ROTATION_RATE


def read_finals(path):
    """`(mjd, xp, yp)` in days and arcseconds, from the fixed columns of a `finals2000A` file."""
    mjd, xp, yp = [], [], []
    with open(path) as fh:
        for line in fh:
            try:
                m, x, y = float(line[7:15]), float(line[18:27]), float(line[37:46])
            except ValueError:
                continue  # a row whose Bulletin A columns are blank: no prediction that far out
            mjd.append(m)
            xp.append(x)
            yp.append(y)
    return np.array(mjd), np.array(xp), np.array(yp)


def main():
    paths = sorted(glob.glob("data/eop/finals2000A.*.txt"))
    if not paths:
        raise SystemExit("no data/eop/finals2000A.*.txt -- run tools/fetch-eop.sh first")
    mjd, xp, yp = read_finals(paths[-1])
    years = (mjd[-1] - mjd[0]) / 365.25
    print(f"{paths[-1]}: {mjd.size} daily rows, MJD {mjd[0]:.0f}..{mjd[-1]:.0f} ({years:.1f} yr)")

    rate = np.hypot(
        np.gradient(xp * ARCSEC, mjd * DAY), np.gradient(yp * ARCSEC, mjd * DAY)
    )
    print(f"|d(pole)/dt|   median {np.median(rate):.3e}   p99 {np.percentile(rate, 99):.3e}   "
          f"max {rate.max():.3e}  rad/s")
    print(f"  at the equator, m/s:  median {np.median(rate) * EARTH_RADIUS_M:.3e}   "
          f"max {rate.max() * EARTH_RADIUS_M:.3e}")
    print(f"  against the sidereal term: omega is {EARTH_ROTATION_RATE / rate.max():.3e} x the "
          f"worst polar-motion rate")
    print(f"  omega x r at the equator = {EARTH_ROTATION_RATE * EARTH_RADIUS_M:.3f} m/s")

    # The amplitude that is easy to mistake for a rate: the wobble is +-0.3 arcsec, but it takes a
    # 433-day Chandler beat to cross it, so 0.3 arcsec "a year" overstates the rate it implies.
    print(f"peak-to-peak wander: xp {xp.max() - xp.min():.3f}, yp {yp.max() - yp.min():.3f} arcsec")
    print(f"secular drift: xp {(xp[-1] - xp[0]) / years:+.4f}, "
          f"yp {(yp[-1] - yp[0]) / years:+.4f} arcsec/yr")


if __name__ == "__main__":
    main()
