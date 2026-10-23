#!/usr/bin/env python3
"""Cut the committed SupGP fixtures from the snapshots ``tools/fetch-supgp.sh`` leaves in ``data/``.

Three fixtures come out, and each answers a different question:

``crates/spacelab-sgp4/tests/fixtures/supgp_paired.tle``
    Operator element sets beside the standard-catalogue element set for the same object, tagged by
    source and origin.  Feeds the divergence measurement -- how far apart two fits of one object
    drift -- and the laser-target case, where several analysis centres fit the same object and the
    spread between *them* is the number.

``crates/spacelab/tests/fixtures/supgp_gps_paired.tle``
    The GPS subset of the same thing, in the crate that can reach the frame chain.

``crates/spacelab/tests/fixtures/igs20_gps_2026224.sp3``
    Every GPS satellite from the IGS rapid product for 2026-08-12, thinned to the hour.  This is the
    only *precise* leg: the two element sets are both measured against it.

The selection rules are here rather than in the tests because a fixture whose contents depend on
what someone happened to paste is not reproducible.  They are deliberately blunt -- lowest catalog
numbers first, never a random sample -- so re-running on a newer snapshot changes the numbers and
not the shape.

Usage: uv run tools/gen_supgp_fixtures.py [--data DIR] [--sp3 FILE]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

# How many objects of each source reach the paired fixture. `None` means every object that also
# appears in the standard catalogue.
QUOTA: dict[str, int | None] = {
    "gps": None,
    "cpf": None,
    "iridium": None,
    "starlink": 60,
    "oneweb": 40,
}

# Keep every Nth SP3 epoch. The product is a 15-minute grid; the hour is plenty to characterise a
# residual that is kilometres, and it keeps the committed file under 100 kB.
SP3_EPOCH_STRIDE = 4

REPO = Path(__file__).resolve().parent.parent


def read_tle(path: Path) -> list[tuple[str, str, str]]:
    """Return ``(name, line1, line2)`` for every element set in a 3LE file."""
    lines = [line.rstrip("\n").rstrip("\r") for line in path.read_text().splitlines()]
    lines = [line for line in lines if line.strip()]
    out: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if (
            lines[i].startswith("1 ")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("2 ")
        ):
            name = (
                lines[i - 1].strip()
                if i > 0 and not lines[i - 1].startswith(("1 ", "2 "))
                else ""
            )
            out.append((name, lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return out


def catalog_key(line1: str) -> str:
    """Columns 3-7 verbatim -- the key both encodings agree on, Alpha-5 included."""
    return line1[2:7].strip()


def write_paired(path: Path, blocks: list[tuple[str, str, str, str, str]]) -> None:
    """Write ``R <source> <origin> <name>`` followed by the two element-set lines, per record."""
    body = [
        "# Operator-supplied (SupGP) element sets beside the standard GP element set for the same",
        "# object. Written by tools/gen_supgp_fixtures.py; do not hand-edit.",
        "#",
        "# Each record is an `R <source> <origin> <name>` line and the two TLE lines under it. One",
        "# object may carry several `supgp` records -- the laser targets are fitted independently by",
        "# several analysis centres -- and dropping the extras would throw away the only case in the",
        "# catalogue where two providers disagree about one object at one time.",
    ]
    for source, origin, name, line1, line2 in blocks:
        body.append(f"R {source} {origin} {name}")
        body.append(line1)
        body.append(line2)
    path.write_text("\n".join(body) + "\n")


def thin_sp3(text: str, stride: int) -> str:
    """Keep the header and every ``stride``-th epoch block.

    The header's own epoch count goes stale, which is deliberate: ``Sp3::parse`` reads the records
    and not the count, so a thinned file stays valid, and leaving the count alone makes it obvious
    the file is an excerpt rather than a product.
    """
    out: list[str] = []
    epoch = -1
    for line in text.splitlines():
        if line.startswith("*"):
            epoch += 1
            if epoch % stride == 0:
                out.append(line)
        elif line.startswith(("P", "V")):
            if epoch < 0 or epoch % stride == 0:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "supgp")
    parser.add_argument("--sp3", type=Path, default=None)
    args = parser.parse_args()

    gp = {
        catalog_key(l1): (name, l1, l2)
        for name, l1, l2 in read_tle(args.data / "gp-active.tle")
    }
    print(f"standard catalogue: {len(gp)} objects")

    blocks: list[tuple[str, str, str, str, str]] = []
    gps_blocks: list[tuple[str, str, str, str, str]] = []
    for source, quota in QUOTA.items():
        records = read_tle(args.data / f"sup-{source}.tle")
        by_object: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for name, line1, line2 in records:
            by_object[catalog_key(line1)].append((name, line1, line2))

        shared = sorted(key for key in by_object if key in gp)
        chosen = shared if quota is None else shared[:quota]
        multi = sum(1 for key in chosen if len(by_object[key]) > 1)
        emitted = 0
        for key in chosen:
            for name, line1, line2 in by_object[key]:
                blocks.append((source, "supgp", name or "-", line1, line2))
                emitted += 1
            name, line1, line2 = gp[key]
            blocks.append((source, "gp", name or "-", line1, line2))
        print(
            f"{source}: {len(by_object)} objects, {len(shared)} also in GP, "
            f"{len(chosen)} kept, {emitted} SupGP records, {multi} objects with more than one"
        )
        if source == "gps":
            gps_blocks = [b for b in blocks if b[0] == "gps"]

    write_paired(REPO / "crates/spacelab-sgp4/tests/fixtures/supgp_paired.tle", blocks)
    write_paired(
        REPO / "crates/spacelab/tests/fixtures/supgp_gps_paired.tle", gps_blocks
    )
    print(f"paired fixture: {len(blocks)} records")

    if args.sp3 is not None:
        text = args.sp3.read_text()
        thinned = thin_sp3(text, SP3_EPOCH_STRIDE)
        out = REPO / "crates/spacelab/tests/fixtures/igs20_gps_2026224.sp3"
        out.write_text(thinned)
        kept = sum(1 for line in thinned.splitlines() if line.startswith("*"))
        print(f"sp3: {kept} epochs kept -> {out} ({len(thinned)} bytes)")


if __name__ == "__main__":
    main()
