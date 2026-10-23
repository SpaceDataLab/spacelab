#!/usr/bin/env -S uv run --quiet --with skyfield --with numpy python
"""ISS ground-track fixtures with measured Earth orientation.

Run (after tools/fetch-eop.sh):
    ./derivations/ground_track_eop.py

Skyfield is fed the polar-motion table from the same `finals2000A` snapshot this
workspace parses. Its frame code and its geodetic solver are its own; only the
input measurements are shared. Skyfield's UT1 comes from its own delta-T tables
rather than from the file, so column 7 is exported for the test to compare against.
"""

from __future__ import annotations

import glob
import pathlib
import sys

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.data import iers as iers_data

LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
OUT = pathlib.Path(__file__).parent.parent / "crates/spacelab/tests/fixtures"


def main() -> int:
    snapshots = sorted(glob.glob("data/eop/*.txt"))
    if not snapshots:
        print("run tools/fetch-eop.sh first", file=sys.stderr)
        return 1

    ts = load.timescale()
    with open(snapshots[0], "rb") as fh:
        iers_data.install_polar_motion_table(ts, iers_data.parse_x_y_dut1_from_finals_all(fh))
    print(f"polar motion from {snapshots[0]}")

    satellite = EarthSatellite(LINE1, LINE2, "ISS", ts)
    t = ts.utc(2008, 9, 20, 12, np.arange(0, 100, 2.0))
    subpoint = wgs84.subpoint(satellite.at(t))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "iss_groundtrack_eop_skyfield.csv"
    with path.open("w") as fh:
        fh.write("# ISS 2008-264; skyfield subpoint WITH polar motion from finals2000A\n")
        fh.write("# year,month,day,hour,minute,second,ut1_ns_since_j2000,lat_deg,lon_deg,height_m\n")
        for i in range(len(t)):
            ti = t[i]
            year, month, day, hour, minute, second = ti.utc
            ut1_ns = round(((float(ti.whole) - 2451545.0) + float(ti.ut1_fraction)) * 86400e9)
            fh.write(
                f"{int(year)},{int(month)},{int(day)},{int(hour)},{int(minute)},{float(second):.17g},"
                f"{ut1_ns},{subpoint.latitude.degrees[i]:.17g},"
                f"{subpoint.longitude.degrees[i]:.17g},{subpoint.elevation.m[i]:.17g}\n"
            )
    print(f"wrote {len(t)} points to {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
