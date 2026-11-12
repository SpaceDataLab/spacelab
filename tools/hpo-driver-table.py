#!/usr/bin/env python3
"""SDTM-1 driven by GFZ's hourly `ap60` against the 3-hourly `ap`, by the day's `Ap` (`plans/17` Ф20.F).

Zhu et al. (2026, Earth Space Sci., doi:10.1029/2025EA004454) drive three empirical models by the Hpo indices
instead of Kp/ap and find JB2008 3-9 % better in the mean computed/observed ratio and 4-11 % in its RMSE in
moderate and intense storms, and worse in super storms. This is the same comparison for SDTM-1, on the harness's
family folds (`tools/evaluate-sdtm.py`, fields `default` and `ap60`): every row is held out from the fit that
scores it, and both fields are scored on the same rows. Their storm classes are not used here; the day's `Ap`
stands in for them. The ratio is model over measured, `C/O = exp(residual)`. Each difference has an interval that
resamples storm windows and 27-day blocks (`tools/cluster-intervals.py`'s `storm` level), where there are at
least ten of them.

    uv run --project tools --locked python -I tools/hpo-driver-table.py
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FIELDS = ("default", "ap60")
AP_EDGES = (0.0, 15.0, 30.0, 50.0, 100.0, 200.0, np.inf)
# Below this many storms and blocks a resampled interval says more about the draw than about the data.
MIN_CLUSTERS = 10


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def family_folds(harness, field: str) -> dict[str, np.ndarray]:
    """The field's family folds, concatenated in the declared order."""
    paths = [
        harness.OUT / field / f"{s['name']}.npz"
        for s in harness.read_splits()
        if s["kind"] == "family"
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f"not yet run: {', '.join(missing)}")
    parts = [np.load(p) for p in paths]
    return {
        k: np.concatenate([z[k] for z in parts])
        for k in ("test", "residual", "msis", "ap_daily")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    maps = load("residual_maps", "residual-maps.py")
    fit, g = harness.fit, harness.g
    hourly, three = (family_folds(harness, f) for f in reversed(FIELDS))
    for key in ("test", "ap_daily"):
        if not np.array_equal(hourly[key], three[key]):
            raise ValueError(f"the two fields were scored on different rows ({key})")
    rows = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    t = rows.t.to_numpy()[three["test"]]
    family = rows.family.to_numpy()[three["test"]]
    _, _, found = maps.storm_phases(rows, g, t)
    windows = [
        (
            np.datetime64(s["t0"]) + np.timedelta64(int(g.PHASES[0][0]), "h"),
            np.datetime64(s["t0"]) + np.timedelta64(int(s["phase3_end"]), "h"),
        )
        for s in found
    ]
    clusters = intervals.levels(t, family, intervals.storm_numbers(t, windows))["storm"]
    done = np.isfinite(three["msis"])
    ratio = {
        "3-hourly ap": np.exp(three["residual"]),
        "ap60": np.exp(hourly["residual"]),
    }
    print(
        f"{done.sum()} held-out rows on {len(FIELDS)} fields; C/O = model/measured; "
        "change = ap60 minus 3-hourly ap, 95 % interval over storms and 27-day blocks"
    )
    print(
        "day's Ap | rows | clusters | C/O mean (3-h, ap60) | C/O RMSE (3-h, ap60) | RMSE change | "
        "MAPE (3-h, ap60) | MAPE change, points"
    )
    for low, high in itertools.pairwise(AP_EDGES):
        use = done & (three["ap_daily"] >= low) & (three["ap_daily"] < high)
        a, b = ratio["3-hourly ap"][use], ratio["ap60"][use]
        values = np.column_stack(
            [
                (b - 1.0) ** 2,
                (a - 1.0) ** 2,
                100.0 * np.abs(b - 1.0),
                100.0 * np.abs(a - 1.0),
            ]
        )
        sums, counts = intervals.cluster_sums(
            values, intervals.dense_ids(clusters[use])
        )
        draws = intervals.bootstrap_means(sums, counts, args.draws)
        rmse_change = 100.0 * (np.sqrt(draws[:, 0] / draws[:, 1]) - 1.0)
        mape_change = draws[:, 2] - draws[:, 3]
        rmse = np.sqrt(values[:, :2].mean(axis=0))
        mape = values[:, 2:].mean(axis=0)
        cells = []
        for point, draw in (
            (100.0 * (rmse[0] / rmse[1] - 1.0), rmse_change),
            (mape[0] - mape[1], mape_change),
        ):
            if counts.size < MIN_CLUSTERS:
                cells.append(f"{point:+.1f} [—]")
                continue
            lo, hi = intervals.interval(draw)
            cells.append(f"{point:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        label = f"{low:g}-{high:g}" if np.isfinite(high) else f">= {low:g}"
        print(
            f"{label} | {use.sum()} | {counts.size} | {a.mean():.3f}, {b.mean():.3f} | {rmse[1]:.3f}, {rmse[0]:.3f} | "
            f"{cells[0]} % | {mape[1]:.1f}, {mape[0]:.1f} | {cells[1]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
