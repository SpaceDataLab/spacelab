#!/usr/bin/env python3
"""Fetch the re-entries the lifetime campaign scores: GCAT's catalogue and McDowell's element sets.

GCAT (J. McDowell, planet4589.org/space/gcat) is CC BY 4.0 and names each object's decay date. The
historical TLE archive (planet4589.org/space/elements) states no licence, so its files stay under
`data/`, which is never committed, and only numbers derived from them leave the machine.

One request a second, and it stops on anything other than 200 or 404. A file already on disk is not
fetched again. Standard library only.

    tools/fetch-reentry-elements.py [--dry-run] [--objects N ...]

`--objects` fetches these catalogue numbers instead of the campaign's candidates: the spheres of known
area-to-mass that `plans/17` Ф23.T scores are payloads, which the candidates exclude.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "validation" / "reentry"
GCAT = "https://planet4589.org/space/gcat/tsv/cat/satcat.tsv"
ELEMENTS = "https://planet4589.org/space/elements"
AGENT = "spacelab-lifetime-campaign (spacedatalab.org)"
# The years in which a sample of the archive holds element sets up to the decay itself.
WINDOWS = ((1990, 2006), (2012, 2014))
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
# A debris fragment or a rocket body does not manoeuvre, so its decay is drag alone.
UNPOWERED = re.compile(r"^(D|R\d|C)\b")
MAX_APOGEE_KM = 900
MIN_PERIGEE_KM = 120


def fetch(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def decay_date(text: str) -> dt.date | None:
    match = re.match(r"(\d{4}) (\w{3})\s+(\d+)", text)
    if not match:
        return None
    return dt.date(int(match[1]), MONTHS.index(match[2]) + 1, int(match[3]))


def candidates(catalogue: str) -> list[int]:
    header: list[str] = []
    found = []
    for line in catalogue.splitlines():
        if line.startswith("#JCAT"):
            header = [h.strip() for h in line[1:].split("\t")]
        if line.startswith("#") or not line.strip():
            continue
        row = dict(
            zip(header, (cell.strip() for cell in line.split("\t")), strict=False)
        )
        decay = decay_date(row["DDate"])
        if decay is None or not any(lo <= decay.year <= hi for lo, hi in WINDOWS):
            continue
        if not UNPOWERED.match(row["Type"]) or not row["Satcat"].isdigit():
            continue
        try:
            perigee, apogee = float(row["Perigee"]), float(row["Apogee"])
        except ValueError:
            continue
        if perigee >= MIN_PERIGEE_KM and 0 < apogee <= MAX_APOGEE_KM:
            found.append(int(row["Satcat"]))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--objects", type=int, nargs="+", metavar="N")
    args = parser.parse_args()
    (OUT / "elements").mkdir(parents=True, exist_ok=True)
    catalogue = OUT / "satcat.tsv"
    if not catalogue.exists():
        body = fetch(GCAT)
        if body is None:
            sys.exit("GCAT's catalogue is gone")
        catalogue.write_bytes(body)
    objects = args.objects or candidates(
        catalogue.read_text(encoding="utf-8", errors="replace")
    )
    print(f"{len(objects)} candidates")
    if args.dry_run:
        return 0
    fetched = 0
    for number in objects:
        target = OUT / "elements" / f"S{number:05d}"
        if target.exists():
            continue
        body = fetch(f"{ELEMENTS}/{number // 100 * 100:05d}/S{number:05d}")
        target.write_bytes(body or b"")
        fetched += 1
        if fetched % 200 == 0:
            print(f"{fetched} fetched", flush=True)
        time.sleep(1.0)
    print(f"{fetched} fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
