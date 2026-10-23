"""Diurnal-aberration fixture from ERFA `ab` -- the SOFA lineage -- for `spacelab-astrometry`.

Rewrites the `CASES` table in `crates/spacelab-astrometry/tests/aberration_erfa.rs` in place,
between the two sentinel comments, so the fixture is generated rather than transcribed.

    uv run --no-project --with pyerfa --with numpy python tools/gen_aberration_oracle.py

`erfa.ab(pnat, v, s, bm1)` is `pnat*bm1 + w1*v + w2*(v - pdv*pnat)`, normalised, with
`w1 = 1 + (pnat.v)/(1+bm1)` and `w2 = SRS/s`. That `w2` piece is *solar light bending*, not
aberration at all; generating at `s = 1e12` au drives it to about 2e-20 and leaves exactly the
special-relativistic boost `Aberration::apply` implements. So the Rust test compares two
implementations of one formula, and anything above round-off there is a defect rather than a
difference of model.

The angle between two nearly-parallel unit vectors is `atan2(|u x v|, u.v)`, never `arccos(u.v)`:
arccos returns `sqrt(2*delta)` near 1, so at `delta ~ 1e-16` it loses half the mantissa and reports
0.003 arcsec of shift for a direction that lies exactly along the velocity and cannot shift at all.
The first run of this generator did precisely that, and the plausible-looking small number is why it
took a second reading to catch.
"""

import pathlib
import re

import erfa
import numpy as np

C = 299_792_458.0
OMEGA = 7.292_115_146_706_979e-5

FIXTURE = pathlib.Path("crates/spacelab-astrometry/tests/aberration_erfa.rs")
BEGIN = "// BEGIN GENERATED -- tools/gen_aberration_oracle.py\n"
END = "// END GENERATED\n"

# Five stations from the equator to the Arctic: the diurnal speed falls as cos(latitude), so the
# span is what shows the fixture is not one number repeated. Southern and eastern longitudes are in
# there because `omega x r` is where a sign convention goes wrong silently.
STATIONS = [
    (0.0, 0.0, 0.0),
    (34.9, -117.0, 900.0),
    (45.0, 10.0, 200.0),
    (-33.8688, 151.2093, 58.0),
    (78.0, 15.0, 10.0),
]


def station_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    """WGS-84 geodetic to ECEF, so the fixture's positions are a station's own coordinates."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + height_m) * np.cos(lat) * np.cos(lon),
            (n + height_m) * np.cos(lat) * np.sin(lon),
            (n * (1 - e2) + height_m) * np.sin(lat),
        ]
    )


def separation(u: np.ndarray, v: np.ndarray) -> float:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def cases() -> list[str]:
    out = []
    for lat, lon, height in STATIONS:
        r = station_ecef(lat, lon, height)
        v = np.cross([0.0, 0.0, OMEGA], r)
        beta = v / C
        bm1 = np.sqrt(1.0 - beta @ beta)
        oblique = np.array([0.3, 0.5, 0.8])
        directions = [
            ("+x", np.array([1.0, 0.0, 0.0])),
            ("+z", np.array([0.0, 0.0, 1.0])),
            ("oblique", oblique / np.linalg.norm(oblique)),
            # The one direction that provably cannot shift: no transverse component to aberrate.
            ("along -v", -v / np.linalg.norm(v)),
        ]
        for name, pnat in directions:
            got = erfa.ab(pnat, beta, 1e12, bm1)
            shift = np.degrees(separation(pnat, got)) * 3600.0
            out.append(
                "    Case {\n"
                f'        what: "{lat}\N{DEGREE SIGN}, {name}",\n'
                f"        station: {rust(r)},\n"
                f"        natural: {rust(pnat)},\n"
                f"        erfa: {rust(got)},\n"
                f"        speed: {float(np.linalg.norm(v))!r},\n"
                f"        shift_arcsec: {float(shift)!r},\n"
                "    },"
            )
    return out


def rust(v: np.ndarray) -> str:
    """`repr` of a float round-trips exactly in Python and parses the same way in Rust."""
    return "[" + ", ".join(repr(float(x)) for x in v) + "]"


def main() -> None:
    source = FIXTURE.read_text()
    block = BEGIN + "const CASES: &[Case] = &[\n" + "\n".join(cases()) + "\n];\n" + END
    patched, count = re.subn(
        re.escape(BEGIN) + ".*?" + re.escape(END), block, source, flags=re.S
    )
    if count != 1:
        raise SystemExit(
            f"{FIXTURE}: found {count} generated blocks, expected exactly 1"
        )
    FIXTURE.write_text(patched)
    print(f"{FIXTURE}: {len(cases())} cases regenerated from ERFA {erfa.__version__}")


if __name__ == "__main__":
    main()
