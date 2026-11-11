#!/usr/bin/env python3
"""TianMu-1's in-situ O/N2 through the storm of 10-12 October 2024 against SDTM-1's and NRLMSISE-00's
(`plans/17` Ф23.P), validation only.

The record (Ai et al. 2026, Zenodo doi:10.5281/zenodo.20195710, CC BY 4.0) is the paper's plot data. Its
`figure9.mat` holds TM-19's neutral gas mass spectrometer O and N2 every 2 s, ascending (about 01 h local time)
and descending (about 13 h), with time, latitude and local time but no altitude; `figure5.mat`'s axis ticks
put the orbit at 518-545 km. So the models are run at 520 and 545 km, and what is compared is the change of
ln(O/N2) from the quiet window to the main and recovery windows, per node and 10° latitude band, which an
orbit that does not change over three days leaves free of the unknown height. The windows are the paper's
own (its figures 7-8): quiet before 15:40 UT on 10 October, main 16:00 UT to 03:00 UT on 11 October,
recovery 15:00 UT on 11 October to 02:30 UT on 12 October. `figure10-11.mat`'s NRLMSIS 2.0 columns are not
read.

    points  every 15th sample (30 s) as `column_on2 --local`'s input
    score   the changes, measured and modelled, by node and latitude band

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/tianmu-on2.py points|score
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
TIANMU = ROOT / "data" / "validation" / "tianmu-ngms"
HEIGHTS_KM = (520, 545)
THIN = 15
WINDOWS = {
    "quiet": ("2024-10-10T00:00", "2024-10-10T15:40"),
    "main": ("2024-10-10T16:00", "2024-10-11T03:00"),
    "recovery": ("2024-10-11T15:00", "2024-10-12T02:30"),
}
BANDS = ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0))


def samples() -> pd.DataFrame:
    from scipy.io import loadmat

    m = loadmat(TIANMU / "raw" / "figure9.mat")
    frames = []
    for node, key in (
        ("ascending", "TM19_Asc_O_N2_ON2ratio"),
        ("descending", "TM19_Des_O_N2_ON2ratio"),
    ):
        a = m[key][::THIN]
        frames.append(
            pd.DataFrame(
                {
                    "node": node,
                    # MATLAB's datenum counts days from year 0; 719529 is 1970-01-01.
                    "t": pd.to_datetime(a[:, 0] - 719529.0, unit="D").round("s"),
                    "lat": a[:, 1],
                    "lst": a[:, 2],
                    "o": a[:, 3],
                    "n2": a[:, 4],
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    ut = (table.t - table.t.dt.normalize()) / pd.Timedelta(hours=1)
    table["lon"] = ((table.lst - ut) * 15.0 + 180.0) % 360.0 - 180.0
    table["sec"] = ut * 3600.0
    return table


def points() -> int:
    table = samples()
    table.to_csv(TIANMU / "samples.csv", index=False)
    with (TIANMU / "points.in").open("w") as out:
        for r in table.itertuples():
            out.write(
                f"{r.t.year} {r.t.month} {r.t.day} {r.sec:.1f} {r.lat:.4f} {r.lon:.4f} {r.lst:.5f}\n"
            )
    print(f"{len(table)} samples -> {TIANMU / 'points.in'}")
    return 0


def window_of(t: pd.Series) -> np.ndarray:
    out = np.full(t.size, "", dtype=object)
    for name, (lo, hi) in WINDOWS.items():
        out[((t >= pd.Timestamp(lo)) & (t < pd.Timestamp(hi))).to_numpy()] = name
    return out


def score() -> int:
    table = pd.read_csv(TIANMU / "samples.csv", parse_dates=["t"])
    table["measured"] = np.log(table.o / table.n2)
    for h in HEIGHTS_KM:
        models = pd.read_csv(
            TIANMU / f"points-{h}.out",
            sep=" ",
            header=None,
            names=["t_inf", "sdtm", "z_sdtm", "msis", "z_msis"],
        )
        if len(models) != len(table):
            raise ValueError(f"{len(models)} model rows for {len(table)} samples")
        table[f"sdtm {h}"] = np.log(models.sdtm)
        table[f"msis {h}"] = np.log(models.msis)
    table["window"] = window_of(table.t)
    table = table[table.window != ""]
    table["band"] = pd.cut(
        table.lat.abs(), [b[0] for b in BANDS] + [BANDS[-1][1]], right=False
    )
    columns = ["measured"] + [f"{m} {h}" for h in HEIGHTS_KM for m in ("sdtm", "msis")]
    mean = table.groupby(["node", "band", "window"], observed=True)[columns].mean()
    counts = table.groupby(["node", "band", "window"], observed=True).size()
    print(
        "quiet O/N2 at TM-19, measured / SDTM-1 / NRLMSISE-00 at "
        + " and ".join(f"{h} km" for h in HEIGHTS_KM)
    )
    for node in ("ascending", "descending"):
        for band in mean.loc[node].index.get_level_values(0).unique():
            q = np.exp(mean.loc[(node, band, "quiet")])
            print(
                f"{node} {band}: {q['measured']:.1f} / "
                + " / ".join(
                    f"{q[f'sdtm {h}']:.1f}, {q[f'msis {h}']:.1f}" for h in HEIGHTS_KM
                )
            )
    print(
        "\nchange of O/N2 from quiet, %: measured | SDTM-1 "
        + " | ".join(f"{h}" for h in HEIGHTS_KM)
        + " km | NRLMSISE-00 "
        + " | ".join(f"{h}" for h in HEIGHTS_KM)
        + " km   (samples quiet/window)"
    )
    for node in ("ascending", "descending"):
        for band in mean.loc[node].index.get_level_values(0).unique():
            for w in ("main", "recovery"):
                if (node, band, w) not in mean.index:
                    continue
                d = 100.0 * np.expm1(
                    mean.loc[(node, band, w)] - mean.loc[(node, band, "quiet")]
                )
                print(
                    f"{node} {band} {w}: {d['measured']:+.1f} | "
                    + " | ".join(f"{d[f'sdtm {h}']:+.1f}" for h in HEIGHTS_KM)
                    + " | "
                    + " | ".join(f"{d[f'msis {h}']:+.1f}" for h in HEIGHTS_KM)
                    + f"   ({counts[(node, band, 'quiet')]}/{counts[(node, band, w)]})"
                )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("points", "score"))
    args = parser.parse_args()
    return points() if args.step == "points" else score()


if __name__ == "__main__":
    sys.exit(main())
