#!/usr/bin/env python3
"""SDTM-1's ablation ladder (`plans/17` Ф30.F): each rung's parameters and held-out error, and what it adds.

Each rung is a field of `tools/evaluate-sdtm.py` and the rung above it in a limit
(`derivations/sdtm_ablation.mac`). On the family folds the error is printed over all rows, outside every
storm, and by storm phase (`tools/residual-maps.py`'s phases); on the storm-out folds over all their rows.
A rung's gain is its MAPE minus that of the rung it extends, in percentage points, paired on the same rows,
with an interval that resamples storm windows and 27-day blocks (`tools/cluster-intervals.py`'s `storm`
level). Wall time per call is not measured here: it needs the `bench` procedure on a quiet machine.

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/ablation-table.py
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# (label, field, the field it extends); the kernel and the cooling are the two ways up from heating alone.
RUNGS = (
    ("Bates on the flux", "bates", None),
    ("+ geography and season", "harmonics", "bates"),
    ("+ ap, held over its interval", "instant", "harmonics"),
    ("+ first-order heating", "first_order", "instant"),
    ("+ a second constant: the kernel", "two_constant", "first_order"),
    ("+ nitric-oxide cooling", "nitric", "first_order"),
    ("+ flux saturation", "no_gain", "nitric"),
    ("+ flux gain: shipped", "default", "no_gain"),
    ("+ solar wind", "solar_wind", "default"),
)
# Variants of the shipped form beside the ladder, paired with it the same way: Ф30.G's one level for every
# satellite, and Ф23.J's rivals for the post-storm drop.
RIVALS = (
    ("one level, no satellite scales", "no_scales", "default"),
    ("+ composition: boundary oxygen", "composition", "default"),
    ("+ delta: the cooling's quiet offset", "delta", "default"),
    ("linear heating and cooling", "linear", "default"),
    ("+ nitric-oxide time scaled by the flux", "nitric_flux", "default"),
    ("GRACE-FO C on v02, GRACE-FO rows left out", "gc_v02", "default"),
)
# A field on a corpus variant measures one family differently (`tools/corpus-variant.py`): its MAPE and its
# pairing leave that family's rows out, so both sides are scored on the same truth.
CHANGED_TRUTH = {"gc_v02": "GRACE-FO"}
KINDS = ("family", "storm")


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def folds(harness, field: str, kind: str) -> dict[str, np.ndarray] | None:
    """The field's folds of one kind, concatenated in the declared order; `None` if any is missing."""
    paths = [
        harness.OUT / field / f"{s['name']}.npz"
        for s in harness.read_splits()
        if s["kind"] == kind
    ]
    if not all(p.exists() for p in paths):
        return None
    parts = [np.load(p) for p in paths]
    return {
        "test": np.concatenate([z["test"] for z in parts]),
        "residual": np.concatenate([z["residual"] for z in parts]),
        "parameters": parts[0]["parameters"],
    }


def paired(child, parent, selected, ids, draws, intervals) -> str:
    """MAPE of `child` minus `parent` on `selected`, with the interval resampling the clusters `ids`."""
    values = np.column_stack(
        [intervals.absolute_percent(r[selected]) for r in (child, parent)]
    )
    sums, counts = intervals.cluster_sums(values, intervals.dense_ids(ids[selected]))
    draws_ = intervals.bootstrap_means(sums, counts, draws)
    low, high = intervals.interval(draws_[:, 0] - draws_[:, 1])
    mark = " spans 0" if low <= 0.0 <= high else ""
    return f"{values[:, 0].mean() - values[:, 1].mean():+.2f} [{low:+.2f}, {high:+.2f}]{mark}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    maps = load("residual_maps", "residual-maps.py")
    fit, g = harness.fit, harness.g
    rows = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    t_all = rows.t.to_numpy()
    family_all = rows.family.to_numpy()
    read = {(f, k): folds(harness, f, k) for _, f, _ in RUNGS + RIVALS for k in KINDS}
    missing = [f"{f}/{k}" for (f, k), z in read.items() if z is None]
    if missing:
        print("not yet run: " + ", ".join(missing), flush=True)
    selections, clusters, families = {}, {}, {}
    for kind in KINDS:
        tests = [z["test"] for (f, k), z in read.items() if k == kind and z is not None]
        if not tests:
            continue
        if any(not np.array_equal(x, tests[0]) for x in tests):
            raise ValueError(f"the {kind} folds were scored on different rows")
        t, family = t_all[tests[0]], family_all[tests[0]]
        families[kind] = family
        phase, _, found = maps.storm_phases(rows, g, t)
        windows = [
            (
                np.datetime64(s["t0"]) + np.timedelta64(int(g.PHASES[0][0]), "h"),
                np.datetime64(s["t0"]) + np.timedelta64(int(s["phase3_end"]), "h"),
            )
            for s in found
        ]
        clusters[kind] = intervals.levels(
            t, family, intervals.storm_numbers(t, windows)
        )["storm"]
        every = np.ones(t.size, dtype=bool)
        selections[kind] = (
            {"all": every}
            if kind == "storm"
            else {
                "all": every,
                "outside storms": phase < 0,
                **{maps.PHASE_NAMES[k]: phase == k for k in range(4)},
            }
        )
    columns = [(k, name) for k in KINDS if k in selections for name in selections[k]]
    print(
        "\nMAPE in %, held out; parameters as fitted in the first fold of the family kind"
    )
    print(
        "rung | field | parameters | "
        + " | ".join(f"{k}: {name}" for k, name in columns)
    )

    def scored(field: str, kind: str, name: str) -> np.ndarray:
        """The selection's rows, less the family whose truth the field's corpus changes."""
        out = selections[kind][name]
        if field in CHANGED_TRUTH:
            out = out & (families[kind] != CHANGED_TRUTH[field])
        return out

    for label, field, _ in RUNGS + RIVALS:
        cells = []
        for kind, name in columns:
            z = read[field, kind]
            cells.append(
                "—"
                if z is None
                else f"{g.mape(z['residual'][scored(field, kind, name)]):.2f}"
            )
        z = read[field, "family"]
        size = "—" if z is None else str(z["parameters"].size)
        print(f"{label} | {field} | {size} | " + " | ".join(cells))
    print(
        "\nwhat a rung adds: its MAPE minus that of the rung it extends, percentage points, paired; "
        "interval resampling storm windows and 27-day blocks"
    )
    shown = [
        ("family", "all"),
        ("family", "outside storms"),
        ("family", "main and recovery"),
        ("storm", "all"),
    ]
    shown = [c for c in shown if c[0] in selections]
    print("rung | against | " + " | ".join(f"{k}: {name}" for k, name in shown))
    for label, field, parent in RUNGS + RIVALS:
        if parent is None:
            continue
        cells = []
        for kind, name in shown:
            child, base = read[field, kind], read[parent, kind]
            cells.append(
                "—"
                if child is None or base is None
                else paired(
                    child["residual"],
                    base["residual"],
                    scored(field, kind, name),
                    clusters[kind],
                    args.draws,
                    intervals,
                )
            )
        print(f"{label} | {parent} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
