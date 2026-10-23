"""Build the force-matched Sentinel-1B deck from NEPTUNE's own LRO deck and one POE state.

NEPTUNE ships a Sentinel-1B validation case, but it drives a separate executable with its own input
format and its own comparison against the ESA precise orbit. What is wanted here is the ordinary
`neptune-sa` deck so that the same harness runs both the selenocentric and the geocentric case, so
the LRO deck is used as the template and only the input type, the epoch, the state and the force
switches are rewritten.

The state is one Orbit State Vector lifted verbatim from
`S1B_OPER_AUX_POEORB_OPOD_20210330T193037_V20190624T225942_20190626T005942.EOF`, whose header
declares `EARTH_FIXED` and `UTC` -- so it goes in as input type 4 and NEPTUNE's own frame chain
converts it. That conversion is then a *measurement*, not a shared step: it is what the geocentric
comparison checks first.

Run: python3 tools/neptune-oracle/decks/make_sentinel1b_deck.py
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "data/oracles/neptune/validation/lro/work/input/neptune.inp"
DEST = ROOT / "tools/neptune-oracle/decks/sentinel1b-matched"

# UTC=2019-06-25T00:00:02, EARTH_FIXED, metres and metres per second, straight from the EOF.
STATE_KM = (
    -999.068229447,
    -776.132523648,
    6951.328124086,
    -0.183002918,
    7.536746464,
    0.813463694,
)

REPLACEMENTS = [
    # (exact line in the template, its replacement)
    ("  6", "  4"),  # input type: cartesian state vector, ITRF-2000
    ("  2025 10 01 00 00 00.000", "  2019 06 25 00 00 02.000"),
    ("  2025 10 08 00 00 00.000", "  2019 06 26 00 00 02.000"),
    ("  6   Geopotential", "  70  Geopotential"),
    (
        "  1   Atmospheric drag    ! (0 = off, 1 = on, 2 = on+wind (HMW07))",
        "  0   Atmospheric drag    ! (0 = off, 1 = on, 2 = on+wind (HMW07))",
    ),
    (
        "  72  Moon gravity       ! (0 = off, 1 = point-mass, >1 = degree of lunar harmonics AIUB-GRL350A)",
        "  1   Moon gravity       ! point mass, to match a third body rather than a central body",
    ),
    ("  1   Solar radiation pressure", "  0   Solar radiation pressure"),
    ("  1   Earth albedo", "  0   Earth albedo"),
    ("  3", "  2"),  # geopotential model: EGM2008 rather than EIGEN-GL04C
]

STATE_LABELS = [
    "radius   x-component (km)",
    "radius   y-component (km)",
    "radius   z-component (km)",
    "velocity x-component (km/s)",
    "velocity y-component (km/s)",
    "velocity z-component (km/s)",
]


def main() -> int:
    lines = TEMPLATE.read_text().split("\n")

    # Each replacement is applied to the *first* line that matches exactly and only once: several of
    # the switches are bare numbers that appear many times in a deck this commented, and a global
    # substitution would silently rewrite the wrong one.
    for old, new in REPLACEMENTS:
        index = next(i for i, line in enumerate(lines) if line == old)
        lines[index] = new

    for value, label in zip(STATE_KM, STATE_LABELS, strict=True):
        index = next(i for i, line in enumerate(lines) if line.rstrip().endswith(label))
        lines[index] = f"  {value!r}               {label}"

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "neptune.inp").write_text("\n".join(lines))
    for auxiliary in ("neptune.mnv", "neptune.srf"):
        (DEST / auxiliary).write_text((TEMPLATE.parent / auxiliary).read_text())
    print(f"wrote {DEST / 'neptune.inp'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
