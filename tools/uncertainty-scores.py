#!/usr/bin/env python3
"""SDTM-1's uncertainty beyond six bins, held out by family (`plans/17` Ф30.I).

The shipped `DensityUncertainty` is a zero-mean normal on `ln(model/measured)` whose sigma is one number per
regime cell, the 81-day flux by the storm state (`gen_sdtm_coefficients.uncertainty`). This tool scores it and
three rivals on the five family folds of `tools/evaluate-sdtm.py`: each family's rows are forecast by a method
built from the other four families' held-out residuals only, as `tools/regime-coverage.py` builds the table.

* `bins6`, the shipped table;
* `continuous`, `ln sigma(x)` linear in the model's own inputs (the 81-day flux, the storm state, the altitude,
  `Y20` of the latitude and the local time's diurnal harmonic), fitted by the zero-mean normal likelihood, a
  convex fit (`derivations/gaussian_scores.mac` 7);
* `conformal_bins`, per cell the `ceil((n + 1) 0.9)`-th smallest `|r|` of the cell's calibration rows
  (`derivations/split_conformal_coverage.mac`);
* `conformal_continuous`, the same rank on `|r|/sigma(x)`, so the interval is `q sigma(x)`.

The calibration rows are not exchangeable with the forecast rows, being other satellites and other years, so the
conformal rank's guarantee does not carry over; what is printed is what was measured. Each method is scored at
90 % by its coverage and half-width, and as a forecast by the log score and the CRPS of its normal
(`derivations/gaussian_scores.mac`, `.gp`); a conformal half-width `w` is read as the normal of
`sigma = w/1.645`. The rows are split by regime cell, family, height, storm phase and §10.2's campaign rows.

Sources: the parameter part is the spread of the five fold fits' predictions on every row, each fit's satellite
scales taken out; the structural part is what remains of the held-out sigma. The driver part is not measured
here.

The verdict, fixed before the first run: a method reaches the ticket's target if in every regime cell the worst
family covers at least 90 %, and on the campaign rows every cell with at least 50 rows covers at least 90 %.
Otherwise the limitation stays printed. Between methods, the mean log score and CRPS decide, each paired against
`bins6` with an interval resampling 27-day blocks.

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/uncertainty-scores.py [--field default]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys

import numpy as np
from scipy import special

HERE = pathlib.Path(__file__).resolve().parent
FAMILIES = ("CHAMP", "GOCE", "GRACE", "GRACE-FO", "Swarm")
HEIGHTS = ((0.0, 300.0), (300.0, 400.0), (400.0, 475.0), (475.0, 1000.0))
CAMPAIGN_SATS = ("GC", "SA")
CAMPAIGN_MONTHS = ("2019-01", "2021-11", "2022-04", "2023-04", "2024-05", "2025-06")
LEVEL = 0.9
MIN_CAMPAIGN_ROWS = 50
METHODS = ("bins6", "continuous", "conformal_bins", "conformal_continuous")


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def conformal_rank(n: int) -> int:
    """The split-conformal rank `ceil((n + 1) LEVEL)`, 1-based; above `n` the interval is unbounded."""
    return math.ceil((n + 1) * LEVEL)


def conformal_quantile(scores: np.ndarray) -> float:
    k = conformal_rank(scores.size)
    if k > scores.size:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def fit_log_sigma(x: np.ndarray, r: np.ndarray, iterations: int = 60) -> np.ndarray:
    """`beta` minimising `sum(eta + r^2 exp(-2 eta)/2)`, `eta = x beta`: Newton with a halving line search."""
    beta = np.zeros(x.shape[1])
    beta[0] = 0.5 * math.log(float(np.mean(r * r)))
    r2 = r * r

    def objective(b: np.ndarray) -> float:
        eta = x @ b
        return float(np.sum(eta + 0.5 * r2 * np.exp(-2.0 * eta)))

    value = objective(beta)
    for _ in range(iterations):
        eta = x @ beta
        w = r2 * np.exp(-2.0 * eta)
        gradient = x.T @ (1.0 - w)
        hessian = (x * (2.0 * w)[:, None]).T @ x
        step = np.linalg.solve(hessian, gradient)
        t = 1.0
        while True:
            trial = beta - t * step
            new = objective(trial)
            if new <= value or t < 1e-8:
                break
            t *= 0.5
        if value - new < 1e-10 * abs(value):
            beta, value = trial, new
            break
        beta, value = trial, new
    return beta


def features(rows, f107a: np.ndarray, storm: np.ndarray, g) -> np.ndarray:
    f = (f107a - g.REFERENCE_F107) / g.F107_SCALE
    a = np.log1p(np.maximum(storm, 0.0) / g.REFERENCE_AP)
    h = (rows.altitude.to_numpy() / 1000.0 - 400.0) / 100.0
    s = np.sin(np.radians(rows.latitude.to_numpy()))
    lst = 2.0 * np.pi * rows.lst.to_numpy() / 24.0
    return np.column_stack(
        [
            np.ones_like(f),
            f,
            f * f,
            a,
            a * a,
            a * f,
            h,
            h * h,
            0.5 * (3.0 * s * s - 1.0),
            np.cos(lst),
            np.sin(lst),
        ]
    )


def normal_scores(r: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the log score `-ln N(r; 0, sigma^2)` and the CRPS `sigma crps(r/sigma)`."""
    z = r / sigma
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    big_phi = 0.5 * special.erfc(-z / math.sqrt(2.0))
    log_score = np.log(sigma) + 0.5 * math.log(2.0 * math.pi) + 0.5 * z * z
    crps = sigma * (z * (2.0 * big_phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return log_score, crps


def fold_spread(harness, rows, arg, sat, family) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the spread of the five family fits' field predictions, raw and with each fit's mean level
    over all rows taken out; and checks each fit reproduces its own held-out residual."""
    fit, g = harness.fit, harness.g
    settings = harness.FIELDS["default"]
    every = np.arange(len(rows))
    field = np.empty((len(FAMILIES), len(rows)))
    for k, fam in enumerate(FAMILIES):
        z = np.load(harness.OUT / "default" / f"family-{fam}.npz")
        train = family != fam
        seen = set(sat[train])
        train_sats = tuple(s for s in fit.SATELLITES if s in seen)
        model = fit.CorpusModel(
            arg,
            sat,
            train_sats,
            g.CAPS[settings["caps"]],
            settings["ridge"],
            settings["sigma"],
            settings["cooling"],
            settings["excess"],
            settings["gain"],
            None,
            None,
            settings["wind"],
            penalty_rows=np.flatnonzero(train),
            limit=settings["limit"],
        )
        p = z["parameters"]
        residual, _ = model.score(p, every)
        pk = model.pack
        residual = (
            residual - model.indicator @ p[pk.i_scale : pk.i_scale + len(pk.scaled)]
        )
        worst = np.max(np.abs(residual[z["test"]] - z["residual"]))
        if worst > 1e-9:
            raise ValueError(
                f"{fam}: the refitted model misses its saved residual by {worst:.2e}"
            )
        field[k] = residual
        print(
            f"fold {fam}: reproduces its held-out residual to {worst:.1e}", flush=True
        )
    raw = field.std(axis=0, ddof=1)
    level = field - field.mean(axis=1, keepdims=True)
    return raw, level.std(axis=0, ddof=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", default="default")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--no-spread", action="store_true")
    args = parser.parse_args()
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    maps = load("residual_maps", "residual-maps.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    fit, g = harness.fit, harness.g
    rows = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    arg = g.fit_arguments(rows, 2)
    heating, nitric, saturation = fit.shipped_kernel()
    per_day = round(86_400.0 / arg["cadence"])
    first = max(int(arg["interval"].min()) - 120 * per_day, 0)
    storm = fit.cooling_grid(
        arg["ap"], heating, nitric, saturation, arg["cadence"], first
    )[arg["interval"]]
    f107a = arg["f107a"]
    cells = g.regime(f107a, storm)
    n_cells = int(cells.max()) + 1
    family = rows.family.to_numpy()
    sat = rows.sat.to_numpy()
    held = np.full(len(rows), np.nan)
    for fam in FAMILIES:
        z = np.load(harness.OUT / args.field / f"family-{fam}.npz")
        held[z["test"]] = z["residual"]
    if not np.isfinite(held).all():
        sys.exit(f"{args.field}: the family folds do not cover every row")
    x = features(rows, f107a, storm, g)
    z90 = g.NORMAL_90
    sigma = {m: np.full(len(rows), np.nan) for m in METHODS}
    half = {m: np.full(len(rows), np.nan) for m in METHODS}
    for fam in FAMILIES:
        here, cal = family == fam, family != fam
        table, _ = g.uncertainty(held[cal], f107a[cal], storm[cal])
        sigma["bins6"][here] = table[cells[here]]
        beta = fit_log_sigma(x[cal], held[cal])
        smooth = np.exp(x @ beta)
        sigma["continuous"][here] = smooth[here]
        q_cells = np.array(
            [
                conformal_quantile(np.abs(held[cal & (cells == c)]))
                for c in range(n_cells)
            ]
        )
        half["conformal_bins"][here] = q_cells[cells[here]]
        q = conformal_quantile(np.abs(held[cal]) / smooth[cal])
        half["conformal_continuous"][here] = q * smooth[here]
        print(
            f"{fam}: continuous beta {np.array2string(beta, precision=3, separator=', ')}; "
            f"normalised conformal q {q:.3f} (1.645 if the normal held)",
            flush=True,
        )
    for m in ("bins6", "continuous"):
        half[m] = z90 * sigma[m]
    for m in ("conformal_bins", "conformal_continuous"):
        sigma[m] = half[m] / z90
    covered = {m: np.abs(held) <= half[m] for m in METHODS}
    scores = {m: normal_scores(held, sigma[m]) for m in METHODS}

    t = rows.t.to_numpy()
    block = ((rows.t - rows.t.min()).dt.days // 27).to_numpy()
    phase, _, _ = maps.storm_phases(rows, g, t)
    altitude = rows.altitude.to_numpy() / 1000.0
    campaign = (
        rows.sat.isin(CAMPAIGN_SATS).to_numpy()
        & rows.month.isin(CAMPAIGN_MONTHS).to_numpy()
    )
    rng = np.random.default_rng(intervals.SEED)
    every = np.ones(len(rows), dtype=bool)

    print(
        f"\n{args.field}: {len(rows)} rows, every one held out by its family; nominal {100 * LEVEL:.0f} %"
    )
    print(
        "method | covered % | half-width, ln x 100 | log score | CRPS x 100 | "
        "log score minus bins6 | CRPS x 100 minus bins6"
    )
    for m in METHODS:
        log_score, crps = scores[m]
        row = (
            f"{m} | {100 * covered[m].mean():.1f} | {100 * np.mean(half[m]):.1f} | "
            f"{log_score.mean():+.4f} | {100 * crps.mean():.3f}"
        )
        if m != "bins6":
            d_log = maps.mean_interval(
                log_score - scores["bins6"][0], every, block, args.draws, rng
            )
            d_crps = maps.mean_interval(
                100.0 * (crps - scores["bins6"][1]), every, block, args.draws, rng
            )
            row += (
                f" | {d_log[0]:+.4f} [{d_log[1]:+.4f}, {d_log[2]:+.4f}]"
                f" | {d_crps[0]:+.3f} [{d_crps[1]:+.3f}, {d_crps[2]:+.3f}]"
            )
        else:
            row += " | — | —"
        print(row)

    print(
        "\ncoverage % by regime cell 0..5: pooled; the worst family (its name); campaign rows"
    )
    target_met = {}
    for m in METHODS:
        pooled, worst, camp = [], [], []
        ok = True
        for c in range(n_cells):
            in_cell = cells == c
            pooled.append(f"{100 * covered[m][in_cell].mean():.1f}")
            by_family = {
                fam: covered[m][in_cell & (family == fam)].mean()
                for fam in FAMILIES
                if (in_cell & (family == fam)).any()
            }
            name = min(by_family, key=by_family.__getitem__)
            worst.append(f"{100 * by_family[name]:.1f} ({name})")
            ok &= by_family[name] >= LEVEL
            n_camp = int((in_cell & campaign).sum())
            if n_camp:
                value = covered[m][in_cell & campaign].mean()
                camp.append(f"{100 * value:.1f} (n {n_camp})")
                if n_camp >= MIN_CAMPAIGN_ROWS:
                    ok &= value >= LEVEL
            else:
                camp.append("—")
        target_met[m] = ok
        print(f"{m} | pooled {' / '.join(pooled)}")
        print(f"{m} | worst family {' / '.join(worst)}")
        print(f"{m} | campaign {' / '.join(camp)}")

    groups = (
        [(fam, family == fam) for fam in FAMILIES]
        + [
            (
                f"{lo:g}-{'' if hi >= 1000.0 else f'{hi:g}'} km",
                (altitude >= lo) & (altitude < hi),
            )
            for lo, hi in HEIGHTS
        ]
        + [("outside storms", phase < 0)]
        + [(name, phase == k) for k, name in enumerate(maps.PHASE_NAMES)]
        + [("campaign rows", campaign)]
    )
    print("\nby group: covered % / CRPS x 100, per method | rows")
    print("group | " + " | ".join(METHODS) + " | rows")
    for name, keep in groups:
        cells_text = [
            f"{100 * covered[m][keep].mean():.1f} / {100 * scores[m][1][keep].mean():.2f}"
            for m in METHODS
        ]
        print(f"{name} | " + " | ".join(cells_text) + f" | {int(keep.sum())}")

    print("\nverdict, the rule fixed before the first run:")
    for m in METHODS:
        print(
            f"{m}: {'target met' if target_met[m] else 'target not met, the limitation stays printed'}"
        )

    if args.no_spread:
        return 0
    if args.field != "default":
        print("\nthe parameter spread is measured on the default field only")
        return 0
    print(
        "\nsources: the five fold fits' spread on every row (parameter) against the held-out sigma"
    )
    raw, shape = fold_spread(harness, rows, arg, sat, family)
    print(
        "cell | rows | bins6 sigma (all families) | fold spread, rms | level removed, rms | parameter share of sigma^2"
    )
    table_all, count = g.uncertainty(held, f107a, storm)
    for c in range(n_cells):
        in_cell = cells == c
        spread = math.sqrt(float(np.mean(raw[in_cell] ** 2)))
        spread_shape = math.sqrt(float(np.mean(shape[in_cell] ** 2)))
        print(
            f"{c} | {int(count[c])} | {table_all[c]:.3f} | {spread:.3f} | {spread_shape:.3f} | "
            f"{100 * spread_shape**2 / table_all[c] ** 2:.1f} %"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
