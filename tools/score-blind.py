#!/usr/bin/env python3
"""The scorer of SDTM-1's prospective blind test: `docs/sdtm1-blind-protocol.md`, `plans/17` Ф30.J.

Two reads, one code path:

    tools/score-blind.py cut          # the 2020 cut of the frozen corpus, under the fields frozen at 2020
    tools/score-blind.py rival        # B5's rival, fitted on every row of the frozen corpus
    tools/score-blind.py blind DIR    # the blind rows in DIR, under the shipped fields
    tools/score-blind.py gap DIR      # Swarm A, B and C of 2026-03, its own row (§9 A4)
    tools/score-blind.py level        # the level of §9 A2's rule under the null, by simulation

`cut` is the protocol's §7 check. It scores the rows `tools/evaluate-sdtm.py run --kind frozen` scored,
under the fields `tools/evaluate-sdtm.py export --split frozen-2020 [--field solar_wind]` wrote, and
must print `docs/verification.md` §10.38's public table to its printed digits. It also compares each
row with the harness's own residual, model by model, where the harness kept one.

`blind` reads `corpus_<SAT>.npz` for each satellite DIR holds, built by `tools/build-density-corpus.py
--out DIR`, every 60th row from the first (`STEP, PHASE = 60, 0`). It refuses a row inside the frozen
corpus's span and a field that `tools/check-sdtm-freeze.py --data` does not pass. The shipped fields are
scored as their coefficient tables hold them, through the library (`examples/blind_rows.rs`).

Models: SDTM-1 and its solar-wind field (`blind_rows`), NRLMSISE-00 (`nrlmsise00_rows`).
The clean-room JB2008's column is computed in the private tree only: its drivers are SET's index files,
and the path that reads them is not in the public cut. Where the column is absent, its row and B2 print
as a named skip with that reason (§9 A3).
Each satellite's log scale from the field's table enters SDTM-1, as the harness scored §10.38. The
metrics are §4's, the paired intervals `tools/cluster-intervals.py`'s, and the verdicts §5's as amended
on 2026-11-24 (§9):

- **B1:** SDTM-1 minus NRLMSISE-00, on MAPE over all rows; read once there are 180 days of blind rows.
- **B2:** SDTM-1 minus the clean-room JB2008, on MAPE over all rows; by the group-sequential rule.
- **B3:** SDTM-1's pooled median `M/O` in [0.95, 1.05], a point check; read with B1.
- **B4:** onset phase, magnetic local time 9-15 h, `|MLat|` >= 60: SDTM-1's mean `100 ln M/O`; and the
  onset contrast `|MLat|` >= 60 minus `|MLat|` < 40 on each mission family. The phases, the dipole and
  the sectors are `tools/residual-maps.py`'s, the intervals resample whole storms. Read once, at the
  first read with 10 storms with onset rows (§9 A1).
- **B5:** the field `flux_boundary_latitude` minus SDTM-1, on MAPE over all rows; by the
  group-sequential rule. `rival` fits it on every row of the frozen corpus at the harness's cadence,
  once, at the runner's commit; the fit is deterministic. Its storm-out folds, where
  `tools/evaluate-sdtm.py run --kind storm --field flux_boundary_latitude` has written them, are
  printed beside it.

The group-sequential rule (§9 A2) reads B2 and B5 at three looks: the blind rows before 2026-11-01,
2027-05-01 and 2027-11-01. At each level, `day` and `storm`, the statistic is the row mean of the paired
difference over its CR2 standard error, read against Student's law with Bell and McCaffrey's degrees of
freedom. Look k may spend in all what the registered O'Brien-Fleming bounds spend by look k at equal
information; its bound follows the clusters realised at the looks read, through the information
`sum n_c^2` of each. A word needs both levels.

Run under `uv run --project tools --locked python -I`, after
`cargo build -p spacelab-propagation --release --example blind_rows --example nrlmsise00_rows`.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import itertools
import math
import pathlib
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
EXAMPLE = ROOT / "target" / "release" / "examples" / "blind_rows"
SPLIT = "frozen-2020"
FIELDS = ("default", "solar_wind")
STORM_AP = 50.0
DRAWS = 4000
FROZEN_END = np.datetime64("2026-05-01")
VERDICT_DAYS = 180
VERDICT_LEVELS = ("day", "storm")
MIN_CLUSTERS = 10
LEVEL_BAND = (0.95, 1.05)
RIVAL = "flux_boundary_latitude"
ONSET = 1
HIGH_MLAT, LOW_MLAT = 60.0, 40.0
DAYSIDE = (9.0, 15.0)
# §9 A1: B4 is read once, at the first read with this many storms with onset rows.
B4_STORMS = 10
# §9 A2: look k reads the blind rows before LOOK_ENDS[k - 1].
LOOK_ENDS = (
    np.datetime64("2026-11-01"),
    np.datetime64("2027-05-01"),
    np.datetime64("2027-11-01"),
)
# A satellite with no row in a look's last month is taken to have ended there once any satellite has
# rows this far past the look's end, so a satellite that stops cannot hold a look open for ever.
ENDED_AFTER = np.timedelta64(183, "D")
# O'Brien-Fleming for three equal looks at a two-sided 0.05, C sqrt(3/k) with C = 2.004036, rounded up:
# `derivations/group_sequential_obf.mac` and `.gp`.
REGISTERED_BOUNDS = (3.4711, 2.4545, 2.0041)
SIMPSON_POINTS = 1001
# §9 A4: the month of Swarm A, B and C the frozen corpus lacks, scored apart by `gap`.
GAP_MONTH = np.datetime64("2026-03")
GAP_SATELLITES = ("SA", "SB", "SC")
# §9 A3: B2's rival, and why its column is absent from the public tree.
JB2008 = "JB2008, clean-room"
JB2008_SKIP = (
    "skipped: computed in the private tree, because its drivers are SET's index files SOLFSMY.TXT and "
    "DTCFILE.TXT and the path that reads them is not in the public cut (protocol §8, §9 A3)"
)

# Each model's drivers and what its licence lets a number of it do: the protocol's §3.
MODELS = {
    "SDTM-1": (
        "F10.7 of the day before, its trailing 81-day mean, 3-hourly ap",
        "ours: public",
    ),
    "SDTM-1 with the solar wind": (
        "as above, with hourly ap60 and OMNI's hourly solar wind",
        "ours, OMNI CC0: public",
    ),
    "NRLMSISE-00": (
        "F10.7 of the day before, its centred 81-day mean, the storm-time ap[]",
        "NRL's public-domain code, our port: public",
    ),
}


# The models `blind_rows` prints, one column each.
BLIND_ROWS = ("SDTM-1", "SDTM-1 with the solar wind")


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feeds(g) -> list[pathlib.Path]:
    """`blind_rows`'s driver files, in its argument order."""
    return [
        g.DENSITY / name
        for name in (
            "SW-All.csv",
            "Hp60_ap60_complete_series.txt",
            "omni2_sw_hourly.txt",
        )
    ]


def evaluate(rows: pd.DataFrame, g, fields: list[pathlib.Path]) -> np.ndarray:
    """`ln(model / measured)` of each `BLIND_ROWS` model: one column each, in that order."""
    if not EXAMPLE.exists():
        sys.exit(
            f"no {EXAMPLE}; run cargo build -p spacelab-propagation --release --example blind_rows"
        )
    t = rows.t
    dates = np.column_stack([t.dt.year, t.dt.month, t.dt.day])
    values = np.column_stack(
        [
            (t - t.dt.normalize()).dt.total_seconds(),
            rows.altitude,
            rows.latitude,
            rows.longitude,
            rows.lst,
        ]
    )
    # `repr` is the shortest string that round-trips, so the evaluator sees the corpus's own doubles.
    text = "".join(
        f"{s} {d[0]} {d[1]} {d[2]} " + " ".join(repr(float(x)) for x in v) + "\n"
        for s, d, v in zip(rows.sat, dates, values, strict=True)
    )
    done = subprocess.run(
        [str(EXAMPLE), *map(str, feeds(g)), *map(str, fields)],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        sys.exit(f"blind_rows failed:\n{done.stderr}")
    out = np.array(done.stdout.split(), dtype=float).reshape(-1, len(BLIND_ROWS))
    if out.shape[0] != len(rows):
        sys.exit(f"blind_rows answered {out.shape[0]} rows of {len(rows)}")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(out) - np.log(rows.rho.to_numpy())[:, None]


def storm_windows(rows: pd.DataFrame, g) -> list[tuple[np.datetime64, np.datetime64]]:
    """The storm windows of `tools/cluster-intervals.py`, from the 3-hourly `ap` of the rows' span."""
    ap, epoch = rows.attrs["ap"], rows.attrs["ap_epoch"]
    start_index = (rows.t.min().normalize() - epoch).days * 8
    found = g.storm_onsets(
        ap[start_index:], epoch + np.timedelta64(start_index // 8, "D")
    )
    return [
        (
            np.datetime64(s["t0"] + np.timedelta64(int(g.PHASES[0][0]), "h")),
            np.datetime64(s["t0"] + np.timedelta64(int(s["phase3_end"]), "h")),
        )
        for s in found
    ]


def daily_ap(rows: pd.DataFrame) -> np.ndarray:
    day_ap = rows.attrs["ap"].reshape(-1, 8).mean(axis=1)
    return day_ap[(rows.t.dt.normalize() - rows.attrs["ap_epoch"]).dt.days.to_numpy()]


def paired(
    intervals,
    labels: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float, float, int]:
    """`mean(transform(a)) - mean(transform(b))`, its interval over whole clusters, and their number.

    The draws are `report_table`'s: the same seed and the same clusters give the same picks."""
    values = np.column_stack([transform(a), transform(b)])
    ids = intervals.dense_ids(labels)
    sums, counts = intervals.cluster_sums(values, ids)
    result = intervals.bootstrap_means(sums, counts, DRAWS)
    low, high = intervals.interval(result[:, 0] - result[:, 1])
    means = values.mean(axis=0)
    return float(means[0] - means[1]), low, high, counts.size


def verdict(cells: list[tuple[float, float, float, int]]) -> str:
    """§5: `holds` when every level's interval lies below zero, `fails` when every one lies above."""
    if any(n < MIN_CLUSTERS for *_, n in cells):
        return "not resolved"
    if all(high < 0.0 for _, _, high, _ in cells):
        return "holds"
    if all(low > 0.0 for _, low, _, _ in cells):
        return "fails"
    return "not shown"


def simpson_weights(points: int, step: float) -> np.ndarray:
    weights = np.full(points, 2.0)
    weights[1::2] = 4.0
    weights[0] = weights[-1] = 1.0
    return weights * step / 3.0


def crossing(bounds: list[float], info: list[float]) -> float:
    """P0 that some `|Z_k|` reaches `bounds[k]`, with `Z_k = S_k / sqrt(info[k])` and `S` a sum of
    independent Gaussian increments of variance `info[k] - info[k - 1]`: the canonical joint law of a
    cumulative mean over independent clusters, `corr(Z_i, Z_j) = sqrt(info[i] / info[j])` (§9 A2 (a)).

    The density of `S` on the region not yet crossed is carried from look to look on a Simpson grid
    (Armitage, McPherson and Rowe 1969)."""
    total, previous = 0.0, 0.0
    grid = mass = None
    for k, (bound, n) in enumerate(zip(bounds, info, strict=True)):
        edge, sd = bound * math.sqrt(n), math.sqrt(n - previous)
        if mass is None:
            total += 2.0 * stats.norm.sf(edge / sd)
        else:
            total += mass @ (
                stats.norm.sf((edge - grid) / sd) + stats.norm.cdf((-edge - grid) / sd)
            )
        if k + 1 < len(bounds):
            x = np.linspace(-edge, edge, SIMPSON_POINTS)
            density = (
                stats.norm.pdf(x / sd) / sd
                if mass is None
                else stats.norm.pdf((x[:, None] - grid[None, :]) / sd) @ mass / sd
            )
            mass = simpson_weights(SIMPSON_POINTS, x[1] - x[0]) * density
            grid = x
        previous = n
    return float(total)


@functools.cache
def spend() -> tuple[float, ...]:
    """What looks 1, 2 and 3 may have spent in all: the crossing probability of the registered bounds
    by each look, at equal information (§9 A2 (a))."""
    return tuple(
        crossing(list(REGISTERED_BOUNDS[: k + 1]), [float(j + 1) for j in range(k + 1)])
        for k in range(len(REGISTERED_BOUNDS))
    )


def z_bounds(info: list[float | None]) -> list[float | None]:
    """§9 A2 (a): each look's bound on the normal scale, from the information realised at the looks read
    so far. A look given `None` takes no decision and leaves its spend to the next. Each bound solves
    the spend to 1e-12 and is then rounded up to four decimals, so a look never spends more than the
    schedule allows; at information in the ratio 1 : 2 : 3 the bounds are the registered ones."""
    bounds: list[float] = []
    seen: list[float] = []
    out: list[float | None] = []
    for k, n in enumerate(info):
        if n is None:
            out.append(None)
            continue
        if seen and n <= seen[-1]:
            raise ValueError(
                f"look {k + 1} holds information {n} and an earlier look {seen[-1]}"
            )
        seen.append(float(n))
        found = optimize.brentq(
            lambda z, target: crossing([*bounds, z], seen) - target,
            0.5,
            8.0,
            args=(spend()[k],),
            xtol=1e-12,
        )
        # Rounded to 1e-6 first, so a bound the solver returns a hair above 3.4711 stays 3.4711.
        bounds.append(math.ceil(round(found * 1e4, 6)) / 1e4)
        out.append(bounds[-1])
    return out


def t_bound(z: float, nu: float) -> float:
    """§9 A2 (b): the bound on a Student statistic with `nu` degrees of freedom at the nominal two-sided
    level the normal bound `z` has."""
    return float(stats.t.isf(stats.norm.sf(z), nu))


def cr2_design(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """§9 A2 (b): the clusters' weights in the row mean, their CR2 factors, and Bell and McCaffrey's
    degrees of freedom, under the working model of equal cluster-mean variance
    (`derivations/group_sequential_obf.mac`). Equal counts give `s^2 / K` and `K - 1`."""
    w = counts / counts.sum()
    s2 = float(w @ w)
    c = w**2 / (1.0 - 2.0 * w + s2)
    b = np.eye(w.size) - w[:, None] - w[None, :] + s2
    return w, c, s2**2 / float(c @ (b**2) @ c)


def cr2_mean(
    sums: np.ndarray, counts: np.ndarray, w: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The row mean and its CR2 standard error, from per-cluster sums (`... x K`)."""
    means = sums / counts
    mean = means @ w
    return mean, np.sqrt(((means - mean[..., None]) ** 2) @ c)


def studentized(
    values: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, int, float]:
    """`values`' row mean, its CR2 standard error over the clusters `labels`, the degrees of freedom,
    the number of clusters, and the information `sum n_c^2`: the variance of the clusters' summed
    differences under the working model, which is proportional to the count when clusters are equal."""
    ids = np.unique(labels, return_inverse=True)[1]
    counts = np.bincount(ids).astype(float)
    info = float(counts @ counts)
    if counts.size < 2:
        return float(values.mean()), math.nan, math.nan, counts.size, info
    w, c, nu = cr2_design(counts)
    mean, se = cr2_mean(np.bincount(ids, weights=values), counts, w, c)
    return float(mean), float(se), nu, counts.size, info


def sequential(
    looks: list[dict[str, tuple[float, float, float, int, float]]],
) -> list[tuple[str, dict[str, float | None]]]:
    """§9 A2: the word at each look read, from each verdict level's `(mean, se, nu, clusters,
    information)`, with each level's bound on its Student statistic (`None` where the look takes no
    decision).

    A look decides only when every level holds `MIN_CLUSTERS` clusters. It reads `holds` when every
    level's statistic lies at or below minus its bound, `fails` when every one lies at or above its
    bound, `continue` otherwise; after a word the rule stops. At the last registered look `continue`
    is `not shown`, and a last look that cannot decide is `not resolved`."""
    decides = [
        all(look[level][3] >= MIN_CLUSTERS for level in VERDICT_LEVELS)
        for look in looks
    ]
    z = {
        level: z_bounds(
            [
                look[level][4] if ok else None
                for look, ok in zip(looks, decides, strict=True)
            ]
        )
        for level in VERDICT_LEVELS
    }
    out: list[tuple[str, dict[str, float | None]]] = []
    stopped = 0
    for k, look in enumerate(looks):
        last = k + 1 == len(LOOK_ENDS)
        limit = {
            level: None if z[level][k] is None else t_bound(z[level][k], look[level][2])
            for level in VERDICT_LEVELS
        }
        if stopped:
            out.append((f"stopped at look {stopped}", limit))
            continue
        if not decides[k]:
            word = (
                "not resolved"
                if last
                else f"no decision: fewer than {MIN_CLUSTERS} clusters at a level"
            )
            out.append((word, limit))
            continue
        statistic = {level: look[level][0] / look[level][1] for level in VERDICT_LEVELS}
        if all(statistic[v] <= -limit[v] for v in VERDICT_LEVELS):
            word = "holds"
        elif all(statistic[v] >= limit[v] for v in VERDICT_LEVELS):
            word = "fails"
        else:
            out.append(("not shown" if last else "continue", limit))
            continue
        out.append((word, limit))
        stopped = k + 1
    return out


def complete_looks(t: np.ndarray, sat: np.ndarray) -> int:
    """§9 A2: how many looks the blind rows complete. Look k is complete when every satellite with a
    row in its six months has a row in their last month, or the rows of some satellite reach
    `ENDED_AFTER` past the look's end, which takes a satellite without that month to have ended."""
    done, start, latest = 0, FROZEN_END, t.max() if t.size else FROZEN_END
    for end in LOOK_ENDS:
        last = (end.astype("datetime64[M]") - 1).astype("datetime64[D]")
        here = (t >= start) & (t < end)
        if not here.any():
            break
        ended = latest >= end + ENDED_AFTER
        if not all(
            ended or ((sat == s) & (t >= last) & (t < end)).any()
            for s in set(sat[here])
        ):
            break
        done, start = done + 1, end
    return done


def look_table(
    name: str,
    t: np.ndarray,
    values: np.ndarray,
    labels: dict[str, np.ndarray],
    looks: int,
) -> None:
    """Print §9 A2's looks of one hypothesis: per look and level the clusters, the mean, its CR2
    standard error, the degrees of freedom, the statistic and its bound; then the word."""
    if looks == 0:
        print(
            f"{name}: no look complete; the first reads the rows before {LOOK_ENDS[0]}"
        )
        return
    cells = [
        {
            level: studentized(values[t < end], labels[level][t < end])
            for level in VERDICT_LEVELS
        }
        for end in LOOK_ENDS[:looks]
    ]
    read = sequential(cells)
    print(
        f"{name}, by look: look | rows before | level | clusters | mean | se | nu | t | bound"
    )
    for k, (cell, (word, limit)) in enumerate(zip(cells, read, strict=True)):
        for level in VERDICT_LEVELS:
            mean, se, nu, n, _ = cell[level]
            bound = "-" if limit[level] is None else f"{limit[level]:.4f}"
            print(
                f"  {k + 1} | {LOOK_ENDS[k]} | {level} | {n} | {mean:+.3f} | {se:.3f} | {nu:.1f} | "
                f"{mean / se:+.3f} | {bound}"
            )
        print(f"  look {k + 1}: {word}")
    print(f"{name}: {read[-1][0]}")


def score(
    rows: pd.DataFrame,
    series: dict[str, np.ndarray],
    g,
    fit,
    intervals,
    every_interval: bool = False,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """§4's table and the paired differences at every level; the rows every model answered, each
    difference's day and storm cells, and the clusters, for the verdicts.

    A level of fewer than `MIN_CLUSTERS` prints its count and no interval (§4), unless `every_interval`:
    §10.38 printed its seven years, and the cut reproduces them."""
    every = np.logical_and.reduce([np.isfinite(v) for v in series.values()])
    storm = daily_ap(rows) >= STORM_AP
    sat = rows.sat.to_numpy()
    t = rows.t.to_numpy()
    print(
        f"\n{every.sum()} of {len(rows)} rows answered by every model, {(every & storm).sum()} of them at "
        f"daily Ap >= {STORM_AP:.0f}; {t[every].min()} to {t[every].max()}"
    )
    print(
        "model | input mode | licence | MAPE | each satellite's mean removed | "
        f"Ap >= {STORM_AP:.0f} | 100 mean |ln M/O| | median M/O"
    )
    for name, residual in series.items():
        r = residual[every]
        mode, licence = MODELS[name]
        print(
            f"{name} | {mode} | {licence} | {g.mape(r):.1f} % | "
            f"{g.mape(fit.debias(r, sat[every])):.1f} % | {g.mape(residual[every & storm]):.1f} % | "
            f"{100.0 * np.mean(np.abs(r)):.1f} | {np.exp(np.median(r)):.3f}"
        )
    if JB2008 not in series:
        print(f"{JB2008} | {JB2008_SKIP}")
    print("\nby satellite, MAPE: satellite | rows | " + " | ".join(series))
    for s in sorted(set(sat[every])):
        here = every & (sat == s)
        print(
            f"{s} | {here.sum()} | "
            + " | ".join(f"{g.mape(v[here]):.1f} %" for v in series.values())
        )

    clusters = intervals.levels(
        t[every],
        rows.family.to_numpy()[every],
        intervals.storm_numbers(t[every], storm_windows(rows, g)),
    )
    names = list(series)
    sdtm = series["SDTM-1"][every]
    metrics = {
        "MAPE": intervals.absolute_percent,
        "100 |ln|": lambda r: 100.0 * np.abs(r),
    }
    print(
        "\nSDTM-1 minus | rows | metric | difference | "
        + " | ".join(f"{level} [95 %]" for level in ("day", "year", "storm"))
    )
    found: dict[tuple[str, str, str], list[tuple[float, float, float, int]]] = {}
    for (part, selected), name, (metric, transform) in itertools.product(
        (
            ("all", np.ones(every.sum(), dtype=bool)),
            (f"Ap >= {STORM_AP:.0f}", storm[every]),
        ),
        names[1:],
        metrics.items(),
    ):
        if selected.sum() == 0:
            continue
        other = series[name][every]
        cells = {
            level: paired(
                intervals,
                clusters[level][selected],
                sdtm[selected],
                other[selected],
                transform,
            )
            for level in ("day", "year", "storm")
        }
        found[name, part, metric] = [cells[level] for level in VERDICT_LEVELS]
        point = next(iter(cells.values()))[0]
        print(
            f"{name} | {part} | {metric} | {point:+.2f} | "
            + " | ".join(
                f"[{low:+.2f}, {high:+.2f}] of {n}"
                if n >= MIN_CLUSTERS or every_interval
                else f"n = {n}, too few clusters"
                for _, low, high, n in cells.values()
            )
        )
    if JB2008 not in series:
        print(f"{JB2008} | {JB2008_SKIP}")

    return every, found, clusters


def onset_dayside(
    rows: pd.DataFrame, sdtm: np.ndarray, every: np.ndarray, g, intervals, final: bool
) -> str:
    """B4: SDTM-1 in the onset phase, on the dayside at high magnetic latitude, and the onset contrast by
    family, each with `tools/residual-maps.py`'s interval over whole storms. Its word is read at the first
    read with `B4_STORMS` storms with onset rows (§9 A1); with fewer, it is not read yet, or, once the
    rows complete the last look (`final`), not resolved."""
    maps = load("residual_maps", "residual-maps.py")
    t = rows.t.to_numpy()
    doy = rows.t.dt.day_of_year.to_numpy()
    year = rows.t.dt.year.to_numpy() + (doy - 0.5) / 365.25
    latitude = np.radians(rows.latitude.to_numpy())
    longitude = np.radians(rows.longitude.to_numpy())
    axis = maps.dipole_axis(year)
    mlat = np.abs(np.degrees(maps.magnetic(axis, latitude, longitude)[0]))
    mlt = maps.magnetic_local_time(
        axis, latitude, longitude, rows.lst.to_numpy(), maps.declination(doy)
    )
    phase, storm, _ = maps.storm_phases(rows, g, t)
    onset = every & (phase == ONSET)
    storms = np.unique(storm[onset]).size
    rng = np.random.default_rng(intervals.SEED)
    values = 100.0 * np.where(every, sdtm, 0.0)
    dayside = onset & (mlat >= HIGH_MLAT) & (mlt >= DAYSIDE[0]) & (mlt < DAYSIDE[1])
    cells: list[tuple[str, int, float, float, float]] = []
    if dayside.any():
        cells.append(
            (
                "onset, dayside, |MLat| >= 60",
                dayside.sum(),
                *maps.mean_interval(values, dayside, storm, DRAWS, rng),
            )
        )
    families = sorted(set(rows.family.to_numpy()[onset]))
    for f in families:
        here = onset & (rows.family.to_numpy() == f)
        high, low = here & (mlat >= HIGH_MLAT), here & (mlat < LOW_MLAT)
        if high.any() and low.any():
            cells.append(
                (
                    f"onset contrast, {f}",
                    here.sum(),
                    *maps.contrast(values, high, low, storm, DRAWS, rng),
                )
            )
    for name, n, mean, low, high in cells:
        print(f"  {name}: {n} rows, {mean:+.1f} [{low:+.1f}, {high:+.1f}]")
    if storms < B4_STORMS:
        if final:
            return f"not resolved ({storms} storms with onset rows by {LOOK_ENDS[-1]})"
        return f"not read: {storms} storms with onset rows, the protocol reads B4 at {B4_STORMS}"
    if len(cells) != 1 + len(families):
        return f"not resolved ({storms} storms with onset rows, a family lacks a cell)"
    if all(cell[4] < 0.0 for cell in cells):
        return "holds"
    if all(cell[3] > 0.0 for cell in cells):
        return "fails"
    return "not shown"


def storm_folds(harness) -> str:
    """The rival's storm-out folds beside the shipped form's, pooled, where both have been run."""
    pooled = {}
    for field in ("default", RIVAL):
        parts = [
            np.load(path)["residual"]
            for s in harness.read_splits()
            if s["kind"] == "storm"
            and (path := harness.OUT / field / f"{s['name']}.npz").exists()
        ]
        if not parts:
            return f"storm-out folds of {field} not run"
        pooled[field] = np.concatenate(parts)
    if pooled["default"].size != pooled[RIVAL].size:
        return "the two fields' storm-out folds hold different rows"
    g = harness.g
    return (
        f"storm-out folds, {pooled[RIVAL].size} rows: {RIVAL} {g.mape(pooled[RIVAL]):.2f} %, "
        f"the shipped form {g.mape(pooled['default']):.2f} %"
    )


def rival_residual(harness, rows: pd.DataFrame) -> np.ndarray:
    """B5's rival on `rows`, with the parameters `rival` fitted."""
    path = harness.OUT / RIVAL / "blind-rival.npz"
    if not path.exists():
        sys.exit(f"no {path}; run tools/score-blind.py rival")
    z = np.load(path)
    settings = harness.FIELDS[RIVAL]
    every = np.arange(len(rows))
    model = harness.fold_model(
        harness.g.fit_arguments(rows, settings.get("degree", 2)),
        rows.sat.to_numpy(),
        tuple(z["train_sats"]),
        settings,
        every,
    )
    residual, _ = model.score(z["parameters"], every)
    return residual


def read_verdicts(
    rows: pd.DataFrame,
    series: dict[str, np.ndarray],
    every: np.ndarray,
    found: dict,
    clusters: dict[str, np.ndarray],
    harness,
    intervals,
) -> None:
    """§5 as amended (§9): B1 and B3 once there are `VERDICT_DAYS` days of blind rows, B4 at
    `B4_STORMS` storms with onset rows, B2 and B5 at each complete look of the group-sequential rule."""
    t = rows.t.to_numpy()
    days = np.unique(t[every].astype("datetime64[D]")).size
    looks = complete_looks(t[every], rows.sat.to_numpy()[every])
    sdtm = series["SDTM-1"]
    if days < VERDICT_DAYS:
        print(
            f"\nB1 and B3 not read: {days} days of blind rows, the protocol reads them at {VERDICT_DAYS}"
        )
    else:
        print(
            f"\nB1 and B3 (§5), on {days} days; B1 holds when the interval lies below zero at "
            f"{', '.join(VERDICT_LEVELS)}"
        )
        print(
            f"B1, SDTM-1 against NRLMSISE-00, MAPE, all rows: {verdict(found['NRLMSISE-00', 'all', 'MAPE'])}"
        )
        level = float(np.exp(np.median(sdtm[every])))
        inside = LEVEL_BAND[0] <= level <= LEVEL_BAND[1]
        print(
            f"B3, SDTM-1's pooled median M/O {level:.3f} in [{LEVEL_BAND[0]}, {LEVEL_BAND[1]}]: "
            f"{'holds' if inside else 'fails'}"
        )
    print("B4, SDTM-1 at storm onset, x 100, storms resampled:")
    final = looks == len(LOOK_ENDS)
    print(f"B4: {onset_dayside(rows, sdtm, every, harness.g, intervals, final)}")
    print(
        f"\nB2 and B5 by the group-sequential rule (§9 A2): {looks} of {len(LOOK_ENDS)} looks complete"
    )
    if looks == 0:
        print(f"no look complete; the first reads the rows before {LOOK_ENDS[0]}")
        return
    labels = {level: clusters[level] for level in VERDICT_LEVELS}
    if JB2008 in series:
        look_table(
            f"B2, SDTM-1 minus the {JB2008}, MAPE, all rows",
            t[every],
            intervals.absolute_percent(sdtm[every])
            - intervals.absolute_percent(series[JB2008][every]),
            labels,
            looks,
        )
    else:
        print(f"B2, SDTM-1 against the {JB2008}: {JB2008_SKIP}")
    rival = rival_residual(harness, rows)
    both = every & np.isfinite(rival)
    look_table(
        f"B5, {RIVAL} minus SDTM-1, MAPE, all rows",
        t[both],
        intervals.absolute_percent(rival[both])
        - intervals.absolute_percent(sdtm[both]),
        {level: clusters[level][both[every]] for level in VERDICT_LEVELS},
        looks,
    )
    print(f"  {storm_folds(harness)}")


def command_cut(_args) -> int:
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    fit, g = harness.fit, harness.g
    intervals = load("cluster_intervals", "cluster-intervals.py")
    fields = [harness.OUT / field / f"{SPLIT}.field" for field in FIELDS]
    for path in fields:
        if not path.exists():
            sys.exit(
                f"no {path}; run tools/evaluate-sdtm.py export --split {SPLIT} [--field solar_wind]"
            )
    saved = {field: np.load(harness.OUT / field / f"{SPLIT}.npz") for field in FIELDS}
    test = saved["default"]["test"]
    if not np.array_equal(test, saved["solar_wind"]["test"]):
        sys.exit("the two fields' folds scored different rows")
    corpus = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    rows = corpus.iloc[test].reset_index(drop=True)
    rows.attrs = corpus.attrs
    rust = evaluate(rows, g, fields)
    msis = fit.nrlmsise00(rows) - np.log(rows.rho.to_numpy())
    print(f"{SPLIT}: {len(rows)} rows, the fields {', '.join(str(p) for p in fields)}")
    for path in [*fields, *feeds(g)]:
        print(f"  sha256 {sha256(path)}  {path.name}")
    print(
        "each row against the harness's own residual: model | rows | largest |difference| in ln"
    )
    for name, mine, theirs in (
        ("SDTM-1", rust[:, 0], saved["default"]["residual"]),
        ("SDTM-1 with the solar wind", rust[:, 1], saved["solar_wind"]["residual"]),
        ("NRLMSISE-00", msis, saved["default"]["msis"]),
    ):
        both = np.isfinite(mine) & np.isfinite(theirs)
        print(
            f"  {name} | {both.sum()} | {np.max(np.abs(mine[both] - theirs[both])):.3e}"
        )
    series = {
        "SDTM-1": rust[:, 0],
        "SDTM-1 with the solar wind": rust[:, 1],
        "NRLMSISE-00": msis,
    }
    score(rows, series, g, fit, intervals, every_interval=True)
    return 0


def command_rival(_args) -> int:
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    fit, g = harness.fit, harness.g
    settings = harness.FIELDS[RIVAL]
    rows = fit.load_corpus(harness.STEP, phase=harness.PHASE)
    sat = rows.sat.to_numpy()
    every = np.arange(len(rows))
    train_sats = tuple(s for s in fit.SATELLITES if s in set(sat))
    model = harness.fold_model(
        g.fit_arguments(rows, settings.get("degree", 2)),
        sat,
        train_sats,
        settings,
        every,
    )
    p = model.fit(every)
    residual, _ = model.score(p, every)
    target = harness.OUT / RIVAL / "blind-rival.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, parameters=p, train_sats=np.array(train_sats))
    print(
        f"{target}: {RIVAL} on {len(rows)} rows, in sample MAPE {g.mape(residual):.2f} %"
    )
    return 0


# §9 A2's simulation under the null. Clusters added per look, at the storm level: every combination of
# 10 and 14, and 12 at each look. A storm lasts three to five days; a day holds 36 to 48 scored rows
# (four satellites at one row per 2 h, with gaps).
NULL_SCENARIOS = (*itertools.product((10, 14), repeat=3), (12, 12, 12))
NULL_STORM_DAYS = (3, 5)
NULL_ROWS_PER_DAY = (36, 48)
# Shape of every draw; a storm's sd against a block's 1; a day's sd within its cluster. §10.38's
# paired MAPE differences to NRLMSISE-00 give a storm 0.6 of a block's spread and a day 0.7; the grid
# brackets them. The gate holds the symmetric shapes, which a null of exchangeable errors within a
# cluster implies (§9 A2); the skewed shape is printed beside them, not gated.
NULL_MODELS = tuple(
    itertools.product(("normal", "heavy", "skewed"), (0.6, 1.4), (0.4, 0.8))
)
GATED_SHAPES = ("normal", "heavy")
LEVEL = 0.05


def null_design(per_look: tuple[int, ...], seed: int) -> dict[str, np.ndarray]:
    """The days of the three looks, cut into storm-level clusters as `tools/cluster-intervals.py`
    cuts them (a storm, or the rest of a 27-day block), with storms placed at random until look k
    adds `per_look[k]` clusters; and each day's number of rows."""
    rotation_days = load("cluster_intervals", "cluster-intervals.py").ROTATION_DAYS
    rng = np.random.default_rng(seed)
    days = np.arange(FROZEN_END, LOOK_ENDS[-1], dtype="datetime64[D]")
    look = np.searchsorted(
        np.array(LOOK_ENDS, dtype="datetime64[D]"), days, side="right"
    )
    rotation = (days - np.datetime64("1970-01-01", "D")).astype(int) // rotation_days
    storm = np.full(days.size, -1)

    def labels() -> np.ndarray:
        return np.where(storm >= 0, storm, storm.max() + 1 + rotation - rotation.min())

    def added(k: int) -> int:
        cut = labels()
        return len(set(cut[look == k]) - set(cut[look < k]))

    for k, target in enumerate(per_look):
        window = np.flatnonzero(look == k)
        while added(k) < target:
            length = int(rng.integers(NULL_STORM_DAYS[0], NULL_STORM_DAYS[1] + 1))
            first = int(rng.integers(window[0], window[-1] - length + 2))
            if (storm[first : first + length] < 0).all():
                storm[first : first + length] = storm.max() + 1
        if added(k) != target:
            raise ValueError(f"look {k + 1} adds {added(k)} clusters, not {target}")
    cut = np.unique(labels(), return_inverse=True)[1]
    is_storm = np.zeros(cut.max() + 1, dtype=bool)
    is_storm[cut[storm >= 0]] = True
    low, high = NULL_ROWS_PER_DAY
    return {
        "look": look,
        "labels": cut,
        "storm": is_storm,
        "rows": rng.integers(low, high + 1, days.size).astype(float),
    }


def null_draw(shape: str, rng: np.random.Generator, size) -> np.ndarray:
    """Draws of mean 0 and variance 1: normal; skewed (an exponential less 1, skewness 2); or heavy
    (Student's t with 3 degrees of freedom over sqrt(3))."""
    if shape == "normal":
        return rng.standard_normal(size)
    if shape == "skewed":
        return rng.standard_exponential(size) - 1.0
    if shape == "heavy":
        return rng.standard_t(3, size) / math.sqrt(3.0)
    raise ValueError(f"no shape {shape!r}")


def null_level(
    design: dict[str, np.ndarray],
    shape: str,
    storm_sd: float,
    day_sd: float,
    reps: int,
    seed: int,
    chunk: int = 4000,
) -> dict[str, dict[str, int]]:
    """§9 A2's rule, run as code under the null on `design`. A day's mean difference is its cluster's
    draw (sd `storm_sd` for a storm, 1 for a block) plus its own (sd `day_sd`), all of `shape`, mean 0.

    For each reference law, `student` (the rule) and `normal` (its statistics read against the normal
    bounds, the defect §9 A2 (b) removes), counts the replicates that end in `holds` and in `fails`,
    and those whose storm level alone crosses at some look (§9 A2 (c): the rule's level is at most
    that one's)."""
    rng = np.random.default_rng(seed)
    look, labels, rows = design["look"], design["labels"], design["rows"]
    member = np.zeros((labels.size, labels.max() + 1))
    member[np.arange(labels.size), labels] = 1.0
    scale = np.where(design["storm"], storm_sd, 1.0)
    plan = []
    for k in range(len(LOOK_ENDS)):
        days = int(np.searchsorted(look, k, side="right"))
        present = np.unique(labels[:days])
        counts = rows[:days] @ member[:days][:, present]
        plan.append(
            (days, present, counts, cr2_design(counts), cr2_design(rows[:days]))
        )
    z = {
        "storm": z_bounds([float(p[2] @ p[2]) for p in plan]),
        "day": z_bounds([float(rows[: p[0]] @ rows[: p[0]]) for p in plan]),
    }
    limits = {
        "student": [
            {
                "storm": t_bound(z["storm"][k], p[3][2]),
                "day": t_bound(z["day"][k], p[4][2]),
            }
            for k, p in enumerate(plan)
        ],
        "normal": [{level: z[level][k] for level in z} for k in range(len(plan))],
    }
    found = {
        law: {"reps": reps, "holds": 0, "fails": 0, "storm alone": 0} for law in limits
    }
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        sums = (
            null_draw(shape, rng, (r, scale.size))[:, labels] * scale[labels]
            + day_sd * null_draw(shape, rng, (r, labels.size))
        ) * rows
        statistic = []
        for days, present, counts, storm_design, day_design in plan:
            storm_mean, storm_se = cr2_mean(
                sums[:, :days] @ member[:days][:, present], counts, *storm_design[:2]
            )
            day_mean, day_se = cr2_mean(sums[:, :days], rows[:days], *day_design[:2])
            statistic.append({"storm": storm_mean / storm_se, "day": day_mean / day_se})
        for law, by_look in limits.items():
            going = np.ones(r, dtype=bool)
            alone = np.zeros(r, dtype=bool)
            for t, limit in zip(statistic, by_look, strict=True):
                below = (t["storm"] <= -limit["storm"]) & (t["day"] <= -limit["day"])
                above = (t["storm"] >= limit["storm"]) & (t["day"] >= limit["day"])
                found[law]["holds"] += int((going & below).sum())
                found[law]["fails"] += int((going & above).sum())
                going &= ~(below | above)
                alone |= np.abs(t["storm"]) >= limit["storm"]
            found[law]["storm alone"] += int(alone.sum())
        done += r
    return found


def mc_interval(hits: int, reps: int, cells: int = 1) -> tuple[float, float]:
    """Clopper and Pearson's interval on a simulated level, 95 % jointly over `cells` levels
    (Bonferroni). A gate over many cells needs the joint one: at a true level of 0.05, one of 72
    per-cell 95 % intervals lies wholly above it with probability 1 - 0.975^72 = 0.84."""
    tail = 0.025 / cells
    low = 0.0 if hits == 0 else float(stats.beta.ppf(tail, hits, reps - hits + 1))
    high = (
        1.0
        if hits == reps
        else float(stats.beta.ppf(1.0 - tail, hits + 1, reps - hits))
    )
    return low, high


def command_level(args) -> int:
    """§9 A2's S2 gate: the rule's level under the null over `NULL_SCENARIOS` x `NULL_MODELS`. The
    gate passes when every Monte Carlo interval of the rule's level on a `GATED_SHAPES` model, 95 %
    jointly over those cells, reaches down to `LEVEL`; the skewed model's rows say how far a skewed
    null moves it, and to which word."""
    cells = len(NULL_SCENARIOS) * sum(m[0] in GATED_SHAPES for m in NULL_MODELS)
    print(
        f"§9 A2 under the null, {args.reps} replicates a cell, seed {args.seed}: clusters added per look "
        f"| fractions | shape, storm sd, day sd | level [MC, 95 % jointly over the {cells} gated cells] "
        "| holds | fails | storm level alone | normal bounds (the defect)"
    )
    worst = 0.0
    passed = True
    for i, per_look in enumerate(NULL_SCENARIOS):
        design = null_design(per_look, args.seed + i)
        totals = np.cumsum(per_look)
        fractions = ", ".join(f"{n / totals[-1]:.2f}" for n in totals)
        for j, (shape, storm_sd, day_sd) in enumerate(NULL_MODELS):
            found = null_level(
                design, shape, storm_sd, day_sd, args.reps, args.seed + 1000 * i + j
            )
            rule, defect = found["student"], found["normal"]
            hits = rule["holds"] + rule["fails"]
            low, high = mc_interval(hits, args.reps, cells)
            gated = shape in GATED_SHAPES
            if gated:
                passed &= low <= LEVEL
                worst = max(worst, hits / args.reps)
            print(
                f"{per_look} | {fractions} | {shape}, {storm_sd}, {day_sd} | {hits / args.reps:.4f} "
                f"[{low:.4f}, {high:.4f}] | {rule['holds'] / args.reps:.4f} | "
                f"{rule['fails'] / args.reps:.4f} | {rule['storm alone'] / args.reps:.4f} | "
                f"{(defect['holds'] + defect['fails']) / args.reps:.4f}"
                + ("" if gated else " | not gated")
            )
    print(
        f"largest gated level {worst:.4f}; every gated interval reaches {LEVEL}: "
        f"{'yes' if passed else 'NO'}"
    )
    return 0 if passed else 1


def blind_refusal(t: np.ndarray, sat: np.ndarray) -> str | None:
    """§2: why `blind` refuses its rows, or `None`. A row before `FROZEN_END` lies inside the frozen
    corpus's span; Swarm's 2026-03 is one of them, and `gap` scores it apart (§9 A4)."""
    n = int((t < FROZEN_END).sum())
    if n:
        return f"{n} rows before {FROZEN_END}: they are the frozen corpus's, or §9 A4's month for gap"
    return None


def gap_refusal(t: np.ndarray, sat: np.ndarray) -> str | None:
    """§9 A4: why `gap` refuses its rows, or `None`. It reads Swarm A, B and C of 2026-03 only."""
    outside = int((t.astype("datetime64[M]") != GAP_MONTH).sum())
    others = sorted(set(sat) - set(GAP_SATELLITES))
    if outside or others:
        return (
            f"{outside} rows outside {GAP_MONTH}, satellites {others or 'none'} outside "
            f"{', '.join(GAP_SATELLITES)}: gap reads only §9 A4's month"
        )
    return None


def read_rows(
    directory: pathlib.Path,
    satellites: tuple[str, ...],
    refusal: Callable[[np.ndarray, np.ndarray], str | None],
    harness,
    title: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """The rows of `directory` under the shipped fields, refused unless the frozen baseline passes and
    `refusal` finds nothing; prints the runner's commit, every input's sha256 and the months."""
    fit, g = harness.fit, harness.g
    frozen = subprocess.run(
        [sys.executable, "-I", str(HERE / "check-sdtm-freeze.py"), "--data"],
        capture_output=True,
        text=True,
    )
    if frozen.returncode != 0:
        sys.exit(f"the frozen baseline does not pass:\n{frozen.stdout}{frozen.stderr}")
    named = {
        p.name.removeprefix("corpus_").removesuffix(".npz")
        for p in directory.glob("corpus_*.npz")
    }
    if named - set(satellites):
        sys.exit(
            f"{directory} holds {sorted(named - set(satellites))}: this read takes {satellites}"
        )
    present = tuple(s for s in satellites if s in named)
    if not present:
        sys.exit(f"no corpus_<SAT>.npz in {directory}")
    rows = fit.load_corpus(
        harness.STEP, phase=harness.PHASE, corpus=directory, satellites=present
    )
    refused = refusal(rows.t.to_numpy(), rows.sat.to_numpy())
    if refused:
        sys.exit(f"{directory}: {refused}")
    rust = evaluate(rows, g, [])
    msis = fit.nrlmsise00(rows) - np.log(rows.rho.to_numpy())
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(
        f"{title} of {', '.join(present)} in {directory}, the shipped fields; runner at {head}"
        + (" (uncommitted changes)" if dirty else "")
    )
    for path in sorted(directory.glob("*")):
        print(f"  sha256 {sha256(path)}  {path.name}")
    for path in feeds(g):
        print(f"  sha256 {sha256(path)}  {path.name}")
    months = rows.groupby("sat").month.agg(["min", "max", "nunique"])
    print("months by satellite: satellite | first | last | months")
    for s, m in months.iterrows():
        print(f"  {s} | {m['min']} | {m['max']} | {m['nunique']}")
    series = {
        "SDTM-1": rust[:, 0],
        "SDTM-1 with the solar wind": rust[:, 1],
        "NRLMSISE-00": msis,
    }
    return rows, series


def command_blind(args) -> int:
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    rows, series = read_rows(
        args.directory, harness.fit.SATELLITES, blind_refusal, harness, "blind rows"
    )
    every, found, clusters = score(rows, series, harness.g, harness.fit, intervals)
    read_verdicts(rows, series, every, found, clusters, harness, intervals)
    return 0


def command_gap(args) -> int:
    """§9 A4: Swarm A, B and C of 2026-03, which the frozen corpus lacks: its own labelled row, never
    pooled with the blind months; no hypothesis is read on it."""
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    rows, series = read_rows(
        args.directory, GAP_SATELLITES, gap_refusal, harness, f"Swarm {GAP_MONTH}"
    )
    print(
        f"Swarm {GAP_MONTH}: before FROZEN_END and in no fit; its own row, not pooled with the blind "
        "months, no hypothesis (§9 A4)"
    )
    score(rows, series, harness.g, harness.fit, intervals)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    sub.add_parser("cut").set_defaults(func=command_cut)
    sub.add_parser("rival").set_defaults(func=command_rival)
    blind = sub.add_parser("blind")
    blind.add_argument("directory", type=pathlib.Path)
    blind.set_defaults(func=command_blind)
    gap = sub.add_parser("gap")
    gap.add_argument("directory", type=pathlib.Path)
    gap.set_defaults(func=command_gap)
    level = sub.add_parser("level")
    level.add_argument("--reps", type=int, default=100_000)
    level.add_argument("--seed", type=int, default=20261124)
    level.set_defaults(func=command_level)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
