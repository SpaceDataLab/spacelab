#!/usr/bin/env -S uv run --quiet --with pyerfa --with numpy python
"""Comparator fixture for the leap-second table.

Run:
    ./derivations/leap_seconds.py            # regenerate fixtures
    ./derivations/leap_seconds.py --check    # verify the committed table only

Provenance
----------
The table in `leap_seconds.rs` is the IERS's own `leap-seconds.list`, vendored in
`crates/spacelab-fa/tests/fixtures/` and held to the table by
`tests/leap_second_list.rs`. This script writes what `pyerfa` reports for the
same dates, for the comparator `tests/delta_at_oracle.rs`: benchmark material,
not a gate.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import erfa
import numpy as np

FIXTURE = pathlib.Path(__file__).parent.parent / "crates/spacelab-fa/tests/fixtures/delta_at_pyerfa.csv"

FIRST_YEAR, LAST_YEAR = 1972, 2026


def transitions() -> list[tuple[int, int, int]]:
    """Scan month starts and report every step in TAI-UTC."""
    found, previous = [], None
    for year in range(FIRST_YEAR, LAST_YEAR + 2):
        for month in range(1, 13):
            dat = int(erfa.dat(year, month, 1, 0.0))
            if previous is None or dat != previous:
                found.append((year, month, dat))
                previous = dat
    return found


def sample_dates(rng: np.random.Generator, n: int = 300) -> list[tuple[int, int, int]]:
    """All transition boundaries, plus the day either side, plus random dates."""
    dates: set[tuple[int, int, int]] = set()

    for year, month, _ in transitions():
        for offset in (-1, 0, 1):
            jd = erfa.cal2jd(year, month, 1)[1] + offset
            y, m, d, _ = erfa.jd2cal(2400000.5, jd)
            dates.add((int(y), int(m), int(d)))

    while len(dates) < n:
        jd = rng.uniform(*[erfa.cal2jd(y, 1, 1)[1] for y in (FIRST_YEAR, LAST_YEAR)])
        y, m, d, _ = erfa.jd2cal(2400000.5, jd)
        dates.add((int(y), int(m), int(d)))

    # Before 1972, TAI-UTC was fractional and drifting; the oracle returns a
    # non-integer and int() would silently truncate it into a plausible lie.
    # That era is not modelled, and is covered by a unit test asserting an error.
    return sorted(d for d in dates if d[0] >= FIRST_YEAR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    found = transitions()
    print(f"leap seconds: pyerfa reports {len(found)} transitions in [{FIRST_YEAR}, {LAST_YEAR + 1}]")
    print(f"  first {found[0][0]}-{found[0][1]:02d} -> {found[0][2]} s")
    print(f"  last  {found[-1][0]}-{found[-1][1]:02d} -> {found[-1][2]} s")
    print("  NOTE: the table is closed; the IERS list is valid past 2027-05-20.")

    if args.check:
        return 0

    rng = np.random.default_rng(20260710)
    rows = [(y, m, d, int(erfa.dat(y, m, d, 0.0))) for (y, m, d) in sample_dates(rng)]

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as fh:
        for y, m, d, dat in rows:
            fh.write(f"{y},{m},{d},{dat}\n")
    print(f"\n  wrote {len(rows)} fixtures to {FIXTURE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
