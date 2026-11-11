#!/usr/bin/env python3
"""GOLD's column O/N2 against SDTM-1's and NRLMSISE-00's, by storm phase (`plans/17` Ф23.P), validation only.

GOLD retrieves ΣO/N2, the ratio of O's vertical column to N2's above the height where N2's is 1e17 cm^-2
(Strickland et al. 1995), over the day-side disk it sees from geostationary orbit. The models' ratios are
`crates/spacelab-propagation/examples/column_on2.rs`'s, checked against `derivations/sdtm_column.mac` and
`.gp`. The storms are the protocol of Bruinsma & Laurens (2024) on the 3-hourly `ap`, from GOLD's first ON2
day on; the phases are `tools/residual-maps.py`'s, pre-storm through post-storm.

    days    the GOLD days the storms' phases cover, `year doy` a line, for `tools/fetch-gold-on2.sh`
    points  GOLD's good pixels, per scan, as the median of each 5° cell; writes the cells and the example's input
    score   the models' ratios beside GOLD's: the bias by phase and magnetic latitude, and each cell's change
            from its own pre-storm value, with intervals resampling whole storms

A pixel is good where its own and its scan's quality flag are 0, the solar zenith and emission angles are
below 75°, and the magnetic latitude is below 60° (GOLD's V04 release notes: contaminated poleward of 60°).
Days within 10 days of the 2024-10-07 grating shift are left out, which removes the October 2024 storm.

    uv run --no-project --python 3.13 --with numpy --with pandas --with scipy --with h5py \
        python -I tools/column-on2.py days|points|score
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = ROOT / "data" / "validation" / "gold-on2"
FIRST_DAY = pd.Timestamp("2018-10-05")
GRATING_SHIFT = pd.Timestamp("2024-10-07")
SHIFT_GUARD = pd.Timedelta(days=10)
CELL = 5.0
MIN_PIXELS = 4
MAX_ANGLE = 75.0
MAX_MLAT = 60.0
MLAT_BANDS = ((0.0, 30.0), (30.0, 60.0))


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def storms(g) -> tuple[list[dict], pd.Timestamp]:
    """The protocol's storms from GOLD's first day on, and the `ap` epoch."""
    flux = pd.read_csv(
        g.DENSITY / "SW-All.csv", usecols=["DATE"] + [f"AP{i}" for i in range(1, 9)]
    )
    epoch = pd.Timestamp(flux.DATE.iloc[0])
    ap = flux[[f"AP{i}" for i in range(1, 9)]].to_numpy().reshape(-1).astype(float)
    ap = np.where(np.isfinite(ap), ap, 0.0)
    start = (FIRST_DAY - epoch).days * 8
    return g.storm_onsets(ap[start:], epoch + pd.Timedelta(days=start // 8)), epoch


def edges(g, storm: dict) -> tuple[float, ...]:
    """Hours from `t0` bounding pre-storm, onset, main and recovery, post-storm."""
    return (
        g.PHASES[0][0],
        g.PHASES[1][0],
        0.0,
        storm["phase3_end"],
        storm["phase3_end"] + 12.0,
    )


def excluded(t: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(abs(t - GRATING_SHIFT) <= SHIFT_GUARD)


def phases(g, found: list[dict], t: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Each time's phase (0-3) and storm, -1 outside every storm; a time in two storms' phases goes to the first."""
    phase = np.full(t.size, -1)
    storm = np.full(t.size, -1)
    for n, s in enumerate(found):
        hours = ((t - s["t0"]) / pd.Timedelta(hours=1)).to_numpy()
        e = edges(g, s)
        for k in range(4):
            here = (phase < 0) & (hours >= e[k]) & (hours < e[k + 1])
            phase[here], storm[here] = k, n
    return phase, storm


def days(g) -> int:
    found, _ = storms(g)
    wanted: set[pd.Timestamp] = set()
    for s in found:
        e = edges(g, s)
        first = (s["t0"] + pd.Timedelta(hours=e[0])).normalize()
        last = (s["t0"] + pd.Timedelta(hours=e[-1])).normalize()
        wanted.update(pd.date_range(first, last, freq="D"))
    kept = sorted(d for d in wanted if not excluded(pd.DatetimeIndex([d]))[0])
    print(
        f"{len(found)} storms from {FIRST_DAY:%Y-%m-%d}; {len(kept)} GOLD days",
        file=sys.stderr,
    )
    for d in kept:
        print(f"{d.year} {d.day_of_year:03d}")
    return 0


def cells_of(path: pathlib.Path, maps) -> pd.DataFrame:
    """One day file's good pixels, per scan and 5° cell: median ratio and random uncertainty, pixel count."""
    import h5py

    with h5py.File(path, "r") as f:
        on2, dqi = f["on2"][:], f["on2_dqi"][:]
        sza, ea, unc = (
            f["solar_zenith_angle"][:],
            f["emission_angle"][:],
            f["on2_unc_ran"][:],
        )
        lat, lon = f["latitude"][:].astype(float), f["longitude"][:].astype(float)
        scan_ok = f["dqi"][:] == 0
        start = [b"".join(r).decode() for r in f["scan_start_time"][:]]
        stop = [b"".join(r).decode() for r in f["scan_stop_time"][:]]
    t_start = pd.to_datetime(start, utc=True).tz_localize(None)
    t_stop = pd.to_datetime(stop, utc=True).tz_localize(None)
    middle = t_start + (t_stop - t_start) / 2
    rows = []
    for s in np.flatnonzero(scan_ok):
        good = (
            (dqi[s] == 0)
            & np.isfinite(on2[s])
            & (on2[s] > 0)
            & (sza[s] < MAX_ANGLE)
            & (ea[s] < MAX_ANGLE)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )
        if not good.any():
            continue
        la, lo = lat[good], lon[good]
        year = middle[s].year + (middle[s].day_of_year - 0.5) / 365.25
        axis = maps.dipole_axis(np.full(la.size, year))
        mlat = np.degrees(maps.magnetic(axis, np.radians(la), np.radians(lo))[0])
        keep = np.abs(mlat) < MAX_MLAT
        frame = pd.DataFrame(
            {
                "lat": la[keep],
                "lon": lo[keep],
                "mlat": mlat[keep],
                "on2": on2[s][good][keep].astype(float),
                "unc": unc[s][good][keep].astype(float),
            }
        )
        frame["i"] = np.floor(frame.lat / CELL).astype(int)
        frame["j"] = np.floor(frame.lon / CELL).astype(int)
        grouped = frame.groupby(["i", "j"])
        cell = grouped.median()
        cell["n"] = grouped.size()
        cell = cell[cell.n >= MIN_PIXELS].reset_index()
        cell["t"] = middle[s]
        rows.append(cell)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def points(g, maps) -> int:
    files = sorted((GOLD / "raw").glob("gold_l2_on2_*.nc"))
    table = pd.concat([cells_of(p, maps) for p in files], ignore_index=True)
    table = table[~excluded(table.t)].reset_index(drop=True)
    table["lon"] = (table.lon + 180.0) % 360.0 - 180.0
    ut = (table.t - table.t.dt.normalize()) / pd.Timedelta(seconds=1)
    table["sec"] = ut
    table["lst"] = (ut / 3600.0 + table.lon / 15.0) % 24.0
    table.to_csv(GOLD / "cells.csv", index=False)
    with (GOLD / "cells.in").open("w") as out:
        for r in table.itertuples():
            out.write(
                f"{r.t.year} {r.t.month} {r.t.day} {r.sec:.1f} {r.lat:.4f} {r.lon:.4f} {r.lst:.5f}\n"
            )
    print(
        f"{len(files)} days, {len(table)} cells ({table.n.sum()} pixels) -> {GOLD / 'cells.in'}"
    )
    return 0


def score(g, maps, draws: int) -> int:
    table = pd.read_csv(GOLD / "cells.csv", parse_dates=["t"])
    models = pd.read_csv(
        GOLD / "cells.out",
        sep=" ",
        header=None,
        names=["t_inf", "sdtm", "sdtm_z", "msis", "msis_z"],
    )
    if len(models) != len(table):
        raise ValueError(f"{len(models)} model rows for {len(table)} cells")
    table = pd.concat([table, models], axis=1)
    found, _ = storms(g)
    table["phase"], table["storm"] = phases(g, found, table.t)
    table = table[
        (table.phase >= 0) & np.isfinite(table.sdtm) & np.isfinite(table.msis)
    ].reset_index(drop=True)
    names = maps.PHASE_NAMES
    rng = np.random.default_rng(23)
    clusters = table.storm.to_numpy()
    print(
        f"{table.storm.nunique()} storms, {len(table)} cells; GOLD median {table.on2.median():.3f}, "
        f"SDTM-1 {table.sdtm.min():.3f} to {table.sdtm.max():.3f}, NRLMSISE-00 median {table.msis.median():.3f}"
    )
    print(
        "\nbias, 100 (exp(mean ln(model/GOLD)) - 1) %, [95 % storm-resampled]: SDTM-1 | NRLMSISE-00"
    )
    print("phase | magnetic latitude | cells | storms | SDTM-1 | NRLMSISE-00")
    for k, name in enumerate(names):
        for low, high in ((0.0, MAX_MLAT), *MLAT_BANDS):
            a = (
                (table.phase == k)
                & (table.mlat.abs() >= low)
                & (table.mlat.abs() < high)
            ).to_numpy()
            if not a.any():
                continue
            cells = []
            for model in ("sdtm", "msis"):
                v = np.log(table[model] / table.on2).to_numpy()
                m, lo, hi = maps.mean_interval(v, a, clusters, draws, rng)
                cells.append(
                    f"{100 * np.expm1(m):+.1f} [{100 * np.expm1(lo):+.1f}, {100 * np.expm1(hi):+.1f}]"
                )
            print(
                f"{name} | {low:g}-{high:g}° | {a.sum()} | {table.storm[a].nunique()} | "
                + " | ".join(cells)
            )
    # Each cell's change from the same cell's pre-storm mean in the same storm: what the storm did there.
    base = (
        table[table.phase == 0]
        .groupby(["storm", "i", "j"])[["on2", "sdtm", "msis"]]
        .mean()
    )
    joined = table.join(base, on=["storm", "i", "j"], rsuffix="_pre").dropna(
        subset=["on2_pre"]
    )
    joined = joined[joined.phase > 0].reset_index(drop=True)
    clusters = joined.storm.to_numpy()
    print(
        "\nchange from the cell's own pre-storm mean, %, [95 % storm-resampled]: GOLD | SDTM-1 | NRLMSISE-00"
    )
    print("phase | magnetic latitude | cells | storms | GOLD | SDTM-1 | NRLMSISE-00")
    for k, name in enumerate(names):
        if k == 0:
            continue
        for low, high in ((0.0, MAX_MLAT), *MLAT_BANDS):
            a = (
                (joined.phase == k)
                & (joined.mlat.abs() >= low)
                & (joined.mlat.abs() < high)
            ).to_numpy()
            if not a.any():
                continue
            cells = []
            for col in ("on2", "sdtm", "msis"):
                v = np.log(joined[col] / joined[f"{col}_pre"]).to_numpy()
                m, lo, hi = maps.mean_interval(v, a, clusters, draws, rng)
                cells.append(
                    f"{100 * np.expm1(m):+.1f} [{100 * np.expm1(lo):+.1f}, {100 * np.expm1(hi):+.1f}]"
                )
            print(
                f"{name} | {low:g}-{high:g}° | {a.sum()} | {joined.storm[a].nunique()} | "
                + " | ".join(cells)
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("days", "points", "score"))
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    g = load("gen_sdtm_coefficients", "gen_sdtm_coefficients.py")
    if args.step == "days":
        return days(g)
    maps = load("residual_maps", "residual-maps.py")
    if args.step == "points":
        return points(g, maps)
    return score(g, maps, args.draws)


if __name__ == "__main__":
    sys.exit(main())
