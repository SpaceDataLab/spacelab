#!/usr/bin/env python3
"""Build the thermospheric-density corpus from TU Delft's version 02 archive (`plans/16` Ф15.A).

Siemes et al. (2023), J. Space Weather Space Clim. 13, 1; CC BY 4.0 (`docs/provenance.md`,
`tudelft-thermosphere`). The archive (`Archive.zip`, 23 GB, one nested zip per satellite-month) is
read in place, member by member: unpacked it is 29.8 GB, more than the disk it sits on can spare.

What is kept, and why:

* **The density products only** (`*_DNS_ACC_*`, `*_DNS_POD_*`) at the top of each mission's
  directory. `GRACE_data_v02/nrtdm_output/` holds the same GRACE months before
  `change_header_density.sh` gave them the final header, and GOCE's `all.zip` / `GO.zip` bundle
  months already present one by one; neither is read.
* **The newest reprocessing of each satellite-month** (`v02c` over `v02b` over `v02`), the rule
  `tools/fetch-density-data.sh` applies. A month present twice at the same version (a loose `.txt`
  beside the `.zip`) is read once, after checking the two are byte-identical.
* **Nominal rows only.** Accelerometer products flag anomalous densities in column 11 (`1`); those
  rows are dropped. Orbit-determination products (Swarm) carry no flag. GOCE's column 13 marks rows
  degraded by aliased thruster noise; they are kept and the flag travels with them.
* **One row per 120 s of the product's own clock**, the cadence the fit's loader already uses — 45
  samples to an orbit, enough to average one. The full cadence is 10 s (CHAMP, GRACE, GOCE,
  GRACE-FO) or 30 s (Swarm), about 2e8 rows in all.
* **Time in UTC.** CHAMP, GRACE, GOCE and GRACE-FO stamp GPS time, Swarm UTC; GPS - UTC is
  TAI - UTC - 19 s, read from `spacelab-fa/src/leap_seconds.rs` so the corpus and the library share
  one table.

Output, under `--out` (git-ignored): `corpus_<SAT>.npz` per satellite — `t_ms` (UTC milliseconds
since 2000-01-01T00:00:00, leap seconds not counted), `altitude_m`, `longitude`, `latitude` (deg,
geodetic), `lst` (h), `arglat` (deg), `rho` (kg/m3), `degraded` (GOCE only, else all 0) — and
`manifest.json`: every member read, its sha256, version, time system, rows read, kept and flagged.

Run:

    uv run --no-project --with numpy --with pandas python -I tools/build-density-corpus.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import pathlib
import re
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "validation" / "tudelft-v02" / "Archive.zip"
OUT = ROOT / "data" / "validation" / "tudelft-v02" / "corpus"
LEAP_SECONDS = ROOT / "crates" / "spacelab-fa" / "src" / "leap_seconds.rs"
PILOT = ROOT / "data" / "validation" / "density"

MEMBER = re.compile(
    r"^(?P<mission>CHAMP|GOCE|GRACE|GRACE_FO|Swarm)_data_v02/"
    r"(?P<sat>CH|GO|GA|GB|GC|SA|SB|SC)_DNS_(?P<method>ACC|POD)_"
    r"(?P<year>\d{4})[-_](?P<month>\d{2})_(?P<version>v02[a-z]?)\.(?P<kind>zip|txt)$"
)
EPOCH = dt.datetime(2000, 1, 1)
CADENCE_S = 120
GPS_MINUS_TAI = -19


def gps_offsets() -> tuple[np.ndarray, np.ndarray]:
    """`(gps_start_ms, gps_minus_utc_ms)`: from each GPS instant on, GPS - UTC is that many ms."""
    rows = re.findall(
        r"e\((\d{4}), (\d{1,2}), (\d+)\),", LEAP_SECONDS.read_text(encoding="utf-8")
    )
    if not rows:
        sys.exit(f"no leap-second rows in {LEAP_SECONDS}")
    starts, offsets = [], []
    for year, month, tai_utc in rows:
        utc_ms = (dt.datetime(int(year), int(month), 1) - EPOCH) // dt.timedelta(
            milliseconds=1
        )
        offset = (int(tai_utc) + GPS_MINUS_TAI) * 1000
        # The table's first rows predate GPS (TAI - UTC < 19 s); only rows that GPS time can carry
        # matter here, and every mission flies after 1999.
        starts.append(utc_ms + offset)
        offsets.append(offset)
    return np.array(starts), np.array(offsets)


def payload(archive: zipfile.ZipFile, name: str) -> bytes:
    data = archive.read(name)
    if name.endswith(".txt"):
        return data
    inner = zipfile.ZipFile(io.BytesIO(data))
    texts = [
        m
        for m in inner.namelist()
        if m.endswith(".txt") and not m.startswith("__MACOSX")
    ]
    if len(texts) != 1:
        sys.exit(f"{name}: expected one .txt inside, found {texts}")
    return inner.read(texts[0])


def parse(
    text: bytes, sat: str, gps: tuple[np.ndarray, np.ndarray]
) -> tuple[dict, dict]:
    frame = pd.read_csv(
        io.BytesIO(text),
        sep=r"\s+",
        comment="#",
        header=None,
        engine="c",
        dtype={0: str, 1: str, 2: str},
    )
    tsys = set(frame[2].unique())
    if len(tsys) != 1 or not tsys <= {"GPS", "UTC"}:
        sys.exit(f"{sat}: time system {tsys}")
    clock = pd.to_datetime(frame[0] + " " + frame[1], format="%Y-%m-%d %H:%M:%S.%f")
    t_ms = ((clock - pd.Timestamp(EPOCH)) // pd.Timedelta(milliseconds=1)).to_numpy()
    read = len(frame)
    flagged = np.zeros(read, dtype=bool)
    if frame.shape[1] >= 11:
        flagged = frame[10].to_numpy() != 0.0
    degraded = (
        frame[12].to_numpy() != 0.0 if sat == "GO" else np.zeros(read, dtype=bool)
    )
    rho = frame[8].to_numpy(dtype=float)
    on_grid = t_ms % (CADENCE_S * 1000) == 0
    keep = on_grid & ~flagged & (rho > 0.0)
    if tsys == {"GPS"}:
        starts, offsets = gps
        t_ms = t_ms - offsets[np.searchsorted(starts, t_ms, side="right") - 1]
    out = {
        "t_ms": t_ms[keep],
        "altitude_m": frame[3].to_numpy(dtype=float)[keep],
        "longitude": frame[4].to_numpy(dtype=float)[keep],
        "latitude": frame[5].to_numpy(dtype=float)[keep],
        "lst": frame[6].to_numpy(dtype=float)[keep],
        "arglat": frame[7].to_numpy(dtype=float)[keep],
        "rho": rho[keep],
        "degraded": degraded[keep].astype(np.int8),
    }
    stats = {
        "time_system": tsys.pop(),
        "rows": read,
        "on_grid": int(on_grid.sum()),
        "flagged_on_grid": int((on_grid & flagged).sum()),
        "kept": int(keep.sum()),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", type=pathlib.Path, default=ARCHIVE)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument(
        "--sat", nargs="*", help="only these satellites (CH GA GB GO GC SA SB SC)"
    )
    argument = parser.parse_args()

    archive = zipfile.ZipFile(argument.archive)
    months: dict[tuple[str, str, str], list[dict]] = {}
    for name in archive.namelist():
        hit = MEMBER.match(name)
        if hit and (not argument.sat or hit["sat"] in argument.sat):
            key = (hit["sat"], hit["year"], hit["month"])
            months.setdefault(key, []).append({"name": name, **hit.groupdict()})

    gps = gps_offsets()
    argument.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "archive": str(argument.archive.name),
        "cadence_s": CADENCE_S,
        "months": [],
    }
    columns: dict[str, dict[str, list]] = {}
    for key in sorted(months):
        newest = max(c["version"] for c in months[key])
        chosen = [c for c in months[key] if c["version"] == newest]
        texts = {c["name"]: payload(archive, c["name"]) for c in chosen}
        digests = {name: hashlib.sha256(t).hexdigest() for name, t in texts.items()}
        if len(set(digests.values())) != 1:
            sys.exit(f"{key}: copies at {newest} differ: {digests}")
        name = sorted(texts, key=lambda n: not n.endswith(".zip"))[0]
        data, stats = parse(texts[name], key[0], gps)
        manifest["months"].append(
            {
                "sat": key[0],
                "month": f"{key[1]}-{key[2]}",
                "member": name,
                "version": newest,
                "sha256": digests[name],
                "superseded": sorted(
                    c["name"] for c in months[key] if c["version"] != newest
                ),
                "duplicates": sorted(n for n in texts if n != name),
                **stats,
            }
        )
        bucket = columns.setdefault(key[0], {})
        for column, values in data.items():
            bucket.setdefault(column, []).append(values)
        print(
            f"{key[0]} {key[1]}-{key[2]} {newest} {stats['kept']:7d} kept "
            f"{stats['flagged_on_grid']:6d} flagged",
            flush=True,
        )

    for sat, bucket in columns.items():
        arrays = {c: np.concatenate(v) for c, v in bucket.items()}
        order = np.argsort(arrays["t_ms"], kind="stable")
        if np.any(np.diff(arrays["t_ms"][order]) == 0):
            sys.exit(f"{sat}: duplicate instants across months")
        np.savez_compressed(
            argument.out / f"corpus_{sat}.npz",
            **{c: a[order] for c, a in arrays.items()},
        )

    # The twelve pilot satellite-months, fetched from the server in 2026-08, are the same bytes as
    # the archive's members, or the earlier campaigns measured different data.
    pilots = []
    for cached in sorted(PILOT.glob("*_DNS_*_v02*.txt")):
        record = next(
            (
                m
                for m in manifest["months"]
                if m["member"].endswith(cached.name.replace(".txt", ".zip"))
                or m["member"].endswith(cached.name)
            ),
            None,
        )
        same = (
            record is not None
            and record["sha256"] == hashlib.sha256(cached.read_bytes()).hexdigest()
        )
        pilots.append({"file": cached.name, "same_bytes": same})
    manifest["pilot_months"] = pilots
    (argument.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    identical = sum(1 for p in pilots if p["same_bytes"])
    print(f"pilot months byte-identical: {identical} of {len(pilots)}")


if __name__ == "__main__":
    main()
