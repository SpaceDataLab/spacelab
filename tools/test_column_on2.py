#!/usr/bin/env python3
"""`tools/column-on2.py`'s phases, its grating-shift guard and its per-scan cells, on synthetic input.

    uv run --no-project --python 3.13 --with numpy --with pandas --with scipy --with h5py \
        python -I tools/test_column_on2.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("column_on2", HERE / "column-on2.py")
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["column_on2"] = tool
spec.loader.exec_module(tool)
g = tool.load("gen_sdtm_coefficients", "gen_sdtm_coefficients.py")
maps = tool.load("residual_maps", "residual-maps.py")

T0 = pd.Timestamp("2021-11-04T12:00")


def test_phases_split_at_the_protocol_s_edges_and_the_first_storm_wins() -> None:
    found = [
        {"t0": T0, "phase3_end": 36.0},
        {"t0": T0 + pd.Timedelta(hours=40), "phase3_end": 36.0},
    ]
    hours = [-30.1, -30.0, -18.0, -0.1, 0.0, 35.9, 36.0, 47.9, 48.0, 76.0, 88.0]
    t = pd.Series([T0 + pd.Timedelta(hours=h) for h in hours])
    phase, storm = tool.phases(g, found, t)
    assert phase.tolist() == [-1, 0, 1, 1, 2, 2, 3, 3, 2, 3, -1], phase
    assert storm.tolist() == [-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1], storm


def test_the_guard_covers_ten_days_either_side_of_the_grating_shift() -> None:
    t = pd.Series(
        pd.to_datetime(
            [
                "2024-09-26T23:59",
                "2024-09-27T00:00",
                "2024-10-17T00:00",
                "2024-10-17T00:01",
            ]
        )
    )
    assert tool.excluded(t).tolist() == [False, True, True, False]


def write_day(path: pathlib.Path) -> np.ndarray:
    """Twelve scans on a 4 x 6 grid: scan 0 good, the rest flagged. In scan 0, rows 0-1 sit in one 5° cell near
    the equator with one pixel flagged and one past the zenith-angle cut; row 2 is two pixels of a second
    cell (below the pixel floor) and four of a third past 60° magnetic latitude; row 3 is all fill. Returns the ratios
    the first cell should take its median of."""
    scans, rows, cols = 12, 4, 6
    lat = np.array(
        [[1.0] * 6, [2.0] * 6, [12.0] * 2 + [75.0] * 4, [np.nan] * 6], dtype=np.float32
    )
    lon = np.array(
        [[1.0, 2.0, 3.0, 4.0, 1.5, 2.5]] * 2
        + [[11.0, 12.0, 110.0, 110.5, 111.0, 111.5]]
        + [[np.nan] * 6],
        dtype=np.float32,
    )
    on2 = np.full((scans, rows, cols), np.nan, dtype=np.float32)
    on2[0, :3] = (np.arange(18, dtype=np.float32).reshape(3, 6) + 1.0) / 10.0
    dqi = np.zeros((scans, rows, cols), dtype=np.int32)
    dqi[0, 0, 0] = 1
    sza = np.full((scans, rows, cols), 30.0, dtype=np.float32)
    sza[0, 1, 5] = 80.0
    with h5py.File(path, "w") as f:
        f["on2"], f["on2_dqi"] = on2, dqi
        f["solar_zenith_angle"], f["emission_angle"] = sza, np.full_like(sza, 20.0)
        f["on2_unc_ran"] = np.full_like(sza, 0.02)
        f["latitude"], f["longitude"] = lat, lon
        f["dqi"] = np.array([0] + [131072] * 11, dtype=np.int32)
        for name, time in (
            ("scan_start_time", "2021-11-04T12:00:00Z"),
            ("scan_stop_time", "2021-11-04T12:20:00Z"),
        ):
            chars = np.array([list(time.ljust(50).encode())], dtype=np.uint8)
            f[name] = np.repeat(chars, scans, axis=0).view("S1")
    return np.concatenate([on2[0, 0, 1:], on2[0, 1, :5]]).astype(float)


def test_a_scan_s_cells_keep_good_pixels_only_and_drop_thin_cells() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "day.nc"
        kept = write_day(path)
        cells = tool.cells_of(path, maps)
    assert len(cells) == 1, cells
    cell = cells.iloc[0]
    assert (cell.i, cell.j, cell.n) == (0, 0, kept.size)
    assert np.isclose(cell.on2, np.median(kept), rtol=1e-6)
    assert cell.t == pd.Timestamp("2021-11-04T12:10")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
