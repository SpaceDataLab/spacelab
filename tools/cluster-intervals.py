#!/usr/bin/env python3
"""Clustered intervals and paired differences for SDTM-1's scores: `plans/17` Ф30.C, rule 11.

A row of the corpus is not an independent sample: its neighbours along the orbit, the other satellites of
the same hour and the days of the same storm share the driver's error. So an interval on rows is far too
narrow, by the design effect `1 + (m - 1) rho` (`derivations/cluster_design_effect.mac`). This tool resamples
whole clusters, with the two models always read on the same draw, so that a difference keeps what they share.

The levels: `row`, the naive one, printed to show the size of the error; `day`; `rotation`, 27-day blocks;
`year`; `storm`, each storm window one cluster and the rest in 27-day blocks; and `family`, enumerated exactly
because there are only five. Five clusters give a coarse interval and its edges are not a calibrated 95 %.

    tools/cluster-intervals.py [--field default] [--kind family] [--draws 4000] [--against no_gain]

`--against` puts another fit of the same form in NRLMSISE-00's column, on the same rows and the same draws.

Run under `uv run --no-project --with numpy --with pandas --with scipy python -I`; `summary` of
`tools/evaluate-sdtm.py` holds the per-split numbers this one puts intervals on.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys
from collections.abc import Callable

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ENUMERATE_BELOW = 7
ROTATION_DAYS = 27
CONFIDENCE = 0.95
SEED = 20261101


def cluster_sums(values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sums of `values` (rows x series) and per-cluster row counts, with `ids` 0..K-1."""
    k = int(ids.max()) + 1
    sums = np.column_stack(
        [
            np.bincount(ids, weights=values[:, j], minlength=k)
            for j in range(values.shape[1])
        ]
    )
    return sums, np.bincount(ids, minlength=k).astype(float)


def bootstrap_means(
    sums: np.ndarray, counts: np.ndarray, draws: int, seed: int = SEED
) -> np.ndarray:
    """Whole clusters drawn with replacement; the mean of each series on each draw: draws x series.

    Under `ENUMERATE_BELOW` clusters there are few enough resamples to take all of them, once each, and
    the result carries no Monte-Carlo error.
    """
    k = counts.size
    if k < ENUMERATE_BELOW:
        picks = np.array(list(itertools.product(range(k), repeat=k)))
        return sums[picks].sum(axis=1) / counts[picks].sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, sums.shape[1]))
    for i in range(draws):
        pick = rng.integers(0, k, k)
        out[i] = sums[pick].sum(axis=0) / counts[pick].sum()
    return out


def interval(draws: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return float(low), float(high)


def anova_rho(values: np.ndarray, ids: np.ndarray) -> float:
    """The ANOVA estimator of the correlation inside a cluster, with Searle's size for unequal clusters."""
    k = int(ids.max()) + 1
    counts = np.bincount(ids, minlength=k).astype(float)
    means = np.bincount(ids, weights=values, minlength=k) / counts
    grand = values.mean()
    n = values.size
    between = float((counts * (means - grand) ** 2).sum()) / (k - 1)
    within = float(((values - means[ids]) ** 2).sum()) / (n - k)
    size = (n - float((counts**2).sum()) / n) / (k - 1)
    return (between - within) / (between + (size - 1.0) * within)


def fisher_rho_variance(rho: float, m: float, k: float) -> float:
    """Fisher's large-sample variance of `anova_rho` at `k` clusters of `m`: the derivation's item 5."""
    return (
        2.0
        * (k * m - 1.0)
        * (1.0 - rho) ** 2
        * (1.0 + (m - 1.0) * rho) ** 2
        / (m**2 * k * (m - 1.0) * (k - 1.0))
    )


def dense_ids(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1]


def absolute_percent(residual: np.ndarray) -> np.ndarray:
    return 100.0 * np.abs(np.exp(residual) - 1.0)


def levels(
    t: np.ndarray, family: np.ndarray, storm: np.ndarray
) -> dict[str, np.ndarray]:
    """Cluster labels of each level. `storm` holds a storm's number, or -1 outside every storm."""
    days = (t - np.datetime64("1970-01-01")) // np.timedelta64(1, "D")
    rotation = days // ROTATION_DAYS
    years = t.astype("datetime64[Y]").astype(int)
    in_storm = storm >= 0
    labels = np.where(in_storm, storm, storm.max() + 1 + rotation - rotation.min())
    return {
        "row": np.arange(t.size),
        "day": dense_ids(days),
        "rotation": dense_ids(rotation),
        "year": dense_ids(years),
        "storm": dense_ids(labels),
        "family": dense_ids(family),
    }


def storm_numbers(
    t: np.ndarray, windows: list[tuple[np.datetime64, np.datetime64]]
) -> np.ndarray:
    """Each row's storm, -1 outside them; a row in two windows goes to the first."""
    out = np.full(t.size, -1)
    for i, (start, end) in enumerate(windows):
        out[(out < 0) & (t >= start) & (t < end)] = i
    return out


def report_table(
    name: str,
    selected: np.ndarray,
    series: dict[str, np.ndarray],
    clusters: dict[str, np.ndarray],
    draws: int,
    transform: Callable[[np.ndarray], np.ndarray] = absolute_percent,
) -> None:
    """One group of rows: each series' mean by level, and each series minus the last, paired."""
    names = list(series)
    values = np.column_stack([transform(series[n][selected]) for n in names])
    means = values.mean(axis=0)
    print(f"\n{name}: {int(selected.sum())} rows")
    print(
        "level | clusters | "
        + " | ".join(names)
        + f" | {names[0]} - {names[-1]} | its design effect"
    )
    row_variance = None
    for level, labels in clusters.items():
        ids = dense_ids(labels[selected])
        if ids.max() < 1:
            continue
        sums, counts = cluster_sums(values, ids)
        result = bootstrap_means(sums, counts, draws)
        difference = result[:, 0] - result[:, -1]
        if row_variance is None:
            row_variance = float(difference.var())
        cells = [
            f"{means[j]:.2f} [{interval(result[:, j])[0]:.2f}, {interval(result[:, j])[1]:.2f}]"
            for j in range(len(names))
        ]
        low, high = interval(difference)
        verdict = "" if low > 0.0 or high < 0.0 else " spans 0"
        print(
            f"{level} | {counts.size} | "
            + " | ".join(cells)
            + f" | {means[0] - means[-1]:+.2f} [{low:+.2f}, {high:+.2f}]{verdict} | {difference.var() / row_variance:.0f}"
        )


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "evaluate_sdtm", HERE / "evaluate-sdtm.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_sdtm"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", default="default")
    parser.add_argument("--kind", default="family")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument(
        "--against", help="another field's folds, in NRLMSISE-00's place"
    )
    args = parser.parse_args()
    harness = load_harness()
    fit, g = harness.fit, harness.g
    rows = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    parts = []
    for split in harness.read_splits():
        path = harness.OUT / args.field / f"{split['name']}.npz"
        if split["kind"] != args.kind or not path.exists():
            continue
        z = np.load(path)
        part = {k: z[k] for k in ("test", "residual", "msis", "sat", "ap_daily")}
        if args.against:
            other = np.load(harness.OUT / args.against / f"{split['name']}.npz")
            if not np.array_equal(other["test"], z["test"]):
                raise ValueError(
                    f"{split['name']}: the two fits were scored on different rows"
                )
            part["msis"] = other["residual"]
        parts.append(part)
    if not parts:
        print(f"no {args.kind} folds under {args.field}", file=sys.stderr)
        return 1
    test = np.concatenate([p["test"] for p in parts])
    residual = np.concatenate([p["residual"] for p in parts])
    msis = np.concatenate([p["msis"] for p in parts])
    # Each satellite's mean is removed inside its fold, as `evaluate-sdtm.py summary` prints it.
    debiased = np.concatenate([fit.debias(p["residual"], p["sat"]) for p in parts])
    ap_daily = np.concatenate([p["ap_daily"] for p in parts])
    done = np.isfinite(msis)
    t = rows.t.to_numpy()[test]
    family = rows.family.to_numpy()[test]
    excess = (rows.f107.to_numpy()[test] - rows.f107a.to_numpy()[test]) / 100.0
    ap, epoch = rows.attrs["ap"], rows.attrs["ap_epoch"]
    first = rows.t.min().normalize()
    start_index = (first - epoch).days * 8
    found = g.storm_onsets(
        ap[start_index:], epoch + np.timedelta64(start_index // 8, "D")
    )
    windows = [
        (
            np.datetime64(s["t0"] + np.timedelta64(int(g.PHASES[0][0]), "h")),
            np.datetime64(s["t0"] + np.timedelta64(int(s["phase3_end"]), "h")),
        )
        for s in found
    ]
    other_name = args.against or "NRLMSISE-00"
    print(
        f"{test.size} rows scored, {int(done.sum())} with {other_name}; {len(windows)} storm windows"
    )
    storms = storm_numbers(t, windows)
    clusters = levels(t, family, storms)
    series = {args.field: residual, "debiased": debiased, other_name: msis}
    groups = {
        "all rows": done,
        "Ap < 15": done & (ap_daily < 15.0),
        "15 <= Ap < 50": done & (ap_daily >= 15.0) & (ap_daily < 50.0),
        "Ap >= 50": done & (ap_daily >= 50.0),
        "Ap >= 100": done & (ap_daily >= 100.0),
        "flux excess d >= 0.5": done & (excess >= 0.5),
        "inside a storm window": done & (storms >= 0),
    }
    print("MAPE in %, 95 % percentile intervals, the difference paired on each draw:")
    for name, selected in groups.items():
        report_table(name, selected, series, clusters, args.draws)
    rho = anova_rho(
        absolute_percent(residual[done]), dense_ids(clusters["rotation"][done])
    )
    print(f"\nrho of the absolute error inside a 27-day block: {rho:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
