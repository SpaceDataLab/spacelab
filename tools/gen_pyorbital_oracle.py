#!/usr/bin/env python3
"""pyorbital as an oracle for `spacelab-sensors`' instrument catalogue and direct sensor model.

`spacelab_sensors::scan` and `spacelab_sensors::geoloc` are ports of `pyorbital`'s
`geoloc_instrument_definitions` and `geoloc` (pytroll/pyorbital, **Apache-2.0**, verified against the
`LICENSE.txt` in the distribution — no GPL headers remain anywhere in the package after the
relicensing, so this is portable into a commercial product in a way `satpy` (GPL-3.0) and
`pyresample` (LGPL-3.0) are not). A port needs an oracle more than an original does: the failure mode
is not "wrong physics" but "one sign, one index, one factor transcribed wrong", and that is invisible
to inspection and obvious to a comparison.

**Two fixtures, because there are two claims and they should be able to fail separately.**

  * `pyorbital_scan.txt` — the catalogue alone: `(instrument, scan, line, pixel) -> roll, pitch,
    time_offset`. No orbit, no ellipsoid. If a scan angle is mistyped this is where it shows, without
    a geolocation residual to hide in.
  * `pyorbital_geoloc.txt` — the geometry alone: `(position, velocity, roll, pitch) -> ground point`.
    The state vectors are **exported, not recomputed**, so the Rust side is handed pyorbital's own
    `pos`/`vel` and the residual cannot absorb a difference between two SGP4s. That is the same
    discipline the NEPTUNE comparison uses: feed identical inputs, or the number means nothing.

**One place the two implementations deliberately differ.** pyorbital uses two ellipsoids in one
computation — its `subpoint`/`geodetic_lat` default to GRS80's semi-minor axis (6356.752 314 14 km)
while its ray-ellipsoid intersection uses WGS-84's (6356.752 314 245 km). The port uses WGS-84
throughout. The two axes differ by 105 µm and this script exports pyorbital's answer as it is, so the
Rust test measures what that costs rather than assuming it away.

The TLE is NOAA-18's, taken from pyorbital's own `geoloc.__main__` example, so it comes with the code
under the same licence and needs no separate provenance. Which platform it is barely matters — the
model under test is geometry, not an orbit — but a real one keeps the altitudes and rates honest.

    uv run --with pyorbital python tools/gen_pyorbital_oracle.py
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pyorbital.geoloc_instrument_definitions as gid
from pyorbital.geoloc import compute_pixels
from pyorbital.orbital import Orbital

FIXTURES = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crates/spacelab-sensors/tests/fixtures"
)

# NOAA 18, epoch 2011-10-12 (pyorbital's own example TLE).
TLE1 = "1 28654U 05018A   11284.35271227  .00000478  00000-0  28778-3 0  9246"
TLE2 = "2 28654  99.0096 235.8581 0014859 135.4286 224.8087 14.11526826329313"
START = dt.datetime(2011, 10, 12, 13, 45)

SCANS = 3


def edge_and_middle(count: int) -> list[int]:
    """Both edges, both next-to-edges and the two middle samples.

    The edges are where a sign error is largest and the middle is where it is smallest, so a set
    containing only one of them can be passed by the wrong formula.
    """
    picks = {0, 1, count // 2 - 1, count // 2, count - 2, count - 1}
    return sorted(p for p in picks if 0 <= p < count)


# name -> (builder, whether `scan_points` selects samples or redefines the ramp)
#
# `olci` and `slstr_nadir` are the second kind: they read `len(scan_points)` as the *number* of
# samples and re-interpolate the swath across it, so passing a subset silently changes the angles
# rather than selecting them. They are built whole and sliced afterwards. Getting this backwards
# would produce a fixture that agrees with nothing.
BUILDERS = {
    "avhrr": (lambda pts: gid.avhrr(SCANS, np.asarray(pts)), True),
    "avhrr_gac": (
        lambda pts: gid.avhrr_gac_from_times(
            [START + dt.timedelta(seconds=0.5 * i) for i in range(SCANS)],
            np.asarray(pts),
        ),
        True,
    ),
    "viirs": (lambda pts: gid.viirs(SCANS, list(pts)), True),
    "amsua": (lambda pts: gid.amsua(SCANS, np.asarray(pts)), True),
    "mhs": (lambda pts: gid.mhs(SCANS, np.asarray(pts)), True),
    "hirs4": (lambda pts: gid.hirs4(SCANS, np.asarray(pts)), True),
    "atms": (lambda pts: gid.atms(SCANS, np.asarray(pts)), True),
    "mwhs2": (lambda pts: gid.mwhs2(SCANS, np.asarray(pts)), True),
    "olci": (lambda _pts: gid.olci(SCANS), False),
    "ascat": (lambda pts: gid.ascat(SCANS, np.asarray(pts)), True),
    "slstr_nadir": (lambda _pts: gid.slstr_nadir(SCANS), False),
}

SAMPLE_COUNT = {
    "avhrr": 2048,
    "avhrr_gac": 409,
    "viirs": 6400,
    "amsua": 30,
    "mhs": 90,
    "hirs4": 56,
    "atms": 96,
    "mwhs2": 98,
    "olci": 4000,
    "ascat": 42,
    "slstr_nadir": 3000,
}

DETECTOR_LINES = {"viirs": 32}


def scan_arrays(geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(roll, pitch, seconds)`, each shaped `(scans * lines, pixels)`."""
    fovs = np.asarray(geometry.fovs)
    seconds = np.asarray(geometry._times) / np.timedelta64(1000000000, "ns")
    return fovs[0], fovs[1], seconds


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    orbit = Orbital("NOAA 18", line1=TLE1, line2=TLE2)

    scan_rows = [
        "# pyorbital instrument scan geometry. pytroll/pyorbital, Apache-2.0.",
        "# Generated by tools/gen_pyorbital_oracle.py -- do not edit.",
        "#",
        "# instrument  scan  line  pixel  roll_rad  pitch_rad  time_offset_s",
    ]
    geoloc_rows = [
        "# pyorbital direct geolocation. pytroll/pyorbital, Apache-2.0.",
        "# State vectors are pyorbital's own, exported so the residual is the sensor model alone.",
        "# Generated by tools/gen_pyorbital_oracle.py -- do not edit.",
        "#",
        "# instrument  roll_rad  pitch_rad  pos_x_km pos_y_km pos_z_km"
        "  vel_x_kms vel_y_kms vel_z_kms  pixel_x_km pixel_y_km pixel_z_km",
    ]

    for name, (build, selects) in BUILDERS.items():
        count = SAMPLE_COUNT[name]
        lines = DETECTOR_LINES.get(name, 1)
        picks = edge_and_middle(count)
        geometry = build(picks if selects else None)
        roll, pitch, seconds = scan_arrays(geometry)
        # Where the sample sits in the array depends on which kind of instrument it is: a "selects"
        # geometry was built from `picks` alone, so pick `i` is column `i`; a "redefines the ramp"
        # one holds every sample, so pick `p` is column `p`.
        for scan in range(SCANS):
            for line in range(lines):
                row = scan * lines + line
                for i, pixel in enumerate(picks):
                    c = i if selects else pixel
                    scan_rows.append(
                        f"{name:12s} {scan:5d} {line:5d} {pixel:6d} "
                        f"{roll[row, c]:22.15e} {pitch[row, c]:22.15e} "
                        f"{seconds[row, c]:22.15e}"
                    )

        # Geolocation: pyorbital's own pipeline, on the times its own ScanGeometry produced.
        times = geometry.times(START)
        position, velocity = orbit.get_position(times, normalize=False)
        pixels = compute_pixels((TLE1, TLE2), geometry, times)
        pos = np.asarray(position).reshape(3, -1)
        vel = np.asarray(velocity).reshape(3, -1)
        pix = np.asarray(pixels).reshape(3, -1)
        flat_roll, flat_pitch = roll.ravel(), pitch.ravel()
        # A handful per instrument: the geometry is the same function at every pixel, and what a
        # denser set would buy is a slower test rather than a stronger claim.
        stride = max(1, flat_roll.size // 12)
        for k in range(0, flat_roll.size, stride):
            geoloc_rows.append(
                f"{name:12s} {flat_roll[k]:22.15e} {flat_pitch[k]:22.15e} "
                f"{pos[0, k]:22.15e} {pos[1, k]:22.15e} {pos[2, k]:22.15e} "
                f"{vel[0, k]:22.15e} {vel[1, k]:22.15e} {vel[2, k]:22.15e} "
                f"{pix[0, k]:22.15e} {pix[1, k]:22.15e} {pix[2, k]:22.15e}"
            )

    (FIXTURES / "pyorbital_scan.txt").write_text("\n".join(scan_rows) + "\n")
    (FIXTURES / "pyorbital_geoloc.txt").write_text("\n".join(geoloc_rows) + "\n")
    print(f"pyorbital_scan.txt:   {len(scan_rows) - 4} rows")
    print(f"pyorbital_geoloc.txt: {len(geoloc_rows) - 5} rows")


if __name__ == "__main__":
    main()
