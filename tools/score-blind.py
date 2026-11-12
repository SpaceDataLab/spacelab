#!/usr/bin/env python3
"""The scorer of SDTM-1's prospective blind test: `docs/sdtm1-blind-protocol.md`, `plans/17` Ф30.J.

Two reads, one code path:

    tools/score-blind.py cut          # the 2020 cut of the frozen corpus, under the fields frozen at 2020
    tools/score-blind.py rival        # B5's rival, fitted on every row of the frozen corpus
    tools/score-blind.py blind DIR    # the blind rows in DIR, under the shipped fields

`cut` is the protocol's §7 check. It scores the rows `tools/evaluate-sdtm.py run --kind frozen` scored,
under the fields `tools/evaluate-sdtm.py export --split frozen-2020 [--field solar_wind]` wrote, and
must print `docs/verification.md` §10.38's public table to its printed digits. It also compares each
row with the harness's own residual, model by model, where the harness kept one.

`blind` reads `corpus_<SAT>.npz` for each satellite DIR holds, built by `tools/build-density-corpus.py
--out DIR`, every 60th row from the first (`STEP, PHASE = 60, 0`). It refuses a row inside the frozen
corpus's span and a field that `tools/check-sdtm-freeze.py --data` does not pass. The shipped fields are
scored as their coefficient tables hold them, through the library (`examples/blind_rows.rs`).

Models: SDTM-1 and its solar-wind field (`blind_rows`), NRLMSISE-00 (`nrlmsise00_rows`), the clean-room
JB2008 (`blind_rows`). Each satellite's log scale from the field's table enters SDTM-1, as the harness
scored §10.38. The metrics are §4's, the paired intervals `tools/cluster-intervals.py`'s, and the
verdicts §5's, read once there are 180 days of blind rows:

- **B1, B2:** SDTM-1 minus NRLMSISE-00, and minus the clean-room JB2008, on MAPE over all rows.
- **B3:** SDTM-1's pooled median `M/O` in [0.95, 1.05], a point check.
- **B4:** onset phase, magnetic local time 9-15 h, `|MLat|` >= 60: SDTM-1's mean `100 ln M/O`; and the
  onset contrast `|MLat|` >= 60 minus `|MLat|` < 40 on each mission family. The phases, the dipole and
  the sectors are `tools/residual-maps.py`'s, the intervals resample whole storms.
- **B5:** the field `flux_boundary_latitude` minus SDTM-1, on MAPE over all rows. `rival` fits it on
  every row of the frozen corpus at the harness's cadence, once, at the runner's commit; the fit is
  deterministic. Its storm-out folds, where `tools/evaluate-sdtm.py run --kind storm --field
  flux_boundary_latitude` has written them, are printed beside it.

Run under `uv run --project tools --locked python -I`, after
`cargo build -p spacelab-propagation --release --example blind_rows --example nrlmsise00_rows`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import pathlib
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd

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
    "JB2008, clean-room": (
        "SET's F10/S10/M10/Y10 at their lags, centred 81-day means, hourly ΔTc from Dst",
        "our code: public",
    ),
}


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
            "SOLFSMY.TXT",
            "DTCFILE.TXT",
        )
    ]


def evaluate(rows: pd.DataFrame, g, fields: list[pathlib.Path]) -> np.ndarray:
    """`ln(model / measured)` of SDTM-1, its solar-wind field and the clean-room JB2008: rows x 3."""
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
    out = np.array(done.stdout.split(), dtype=float).reshape(-1, 3)
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

    return every, found, clusters


def onset_dayside(
    rows: pd.DataFrame, sdtm: np.ndarray, every: np.ndarray, g, intervals
) -> str:
    """B4: SDTM-1 in the onset phase, on the dayside at high magnetic latitude, and the onset contrast by
    family, each with `tools/residual-maps.py`'s interval over whole storms."""
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
    if storms < MIN_CLUSTERS or len(cells) != 1 + len(families):
        return f"not resolved ({storms} storms with onset rows)"
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
    """§5, once there are `VERDICT_DAYS` days of blind rows."""
    days = np.unique(rows.t.to_numpy()[every].astype("datetime64[D]")).size
    if days < VERDICT_DAYS:
        print(
            f"\nverdicts not read: {days} days of blind rows, the protocol reads them at {VERDICT_DAYS}"
        )
        return
    sdtm = series["SDTM-1"]
    print(
        f"\nverdicts (§5), on {days} days; B1, B2 and B5 hold when the interval lies below zero at "
        f"{', '.join(VERDICT_LEVELS)}"
    )
    print(
        f"B1, SDTM-1 against NRLMSISE-00, MAPE, all rows: {verdict(found['NRLMSISE-00', 'all', 'MAPE'])}"
    )
    print(
        "B2, SDTM-1 against the clean-room JB2008, MAPE, all rows: "
        f"{verdict(found['JB2008, clean-room', 'all', 'MAPE'])}"
    )
    level = float(np.exp(np.median(sdtm[every])))
    inside = LEVEL_BAND[0] <= level <= LEVEL_BAND[1]
    print(
        f"B3, SDTM-1's pooled median M/O {level:.3f} in [{LEVEL_BAND[0]}, {LEVEL_BAND[1]}]: "
        f"{'holds' if inside else 'fails'}"
    )
    print("B4, SDTM-1 at storm onset, x 100, storms resampled:")
    print(f"B4: {onset_dayside(rows, sdtm, every, harness.g, intervals)}")
    rival = rival_residual(harness, rows)
    both = every & np.isfinite(rival)
    cells = [
        paired(
            intervals,
            clusters[level][both[every]],
            rival[both],
            sdtm[both],
            intervals.absolute_percent,
        )
        for level in VERDICT_LEVELS
    ]
    print(
        f"B5, {RIVAL} minus SDTM-1, MAPE, all rows: {cells[0][0]:+.2f}, "
        + ", ".join(
            f"{level} [{low:+.2f}, {high:+.2f}] of {n}"
            for level, (_, low, high, n) in zip(VERDICT_LEVELS, cells, strict=True)
        )
        + f": {verdict(cells)}"
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
        "JB2008, clean-room": rust[:, 2],
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


def command_blind(args) -> int:
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    fit, g = harness.fit, harness.g
    intervals = load("cluster_intervals", "cluster-intervals.py")
    frozen = subprocess.run(
        [sys.executable, "-I", str(HERE / "check-sdtm-freeze.py"), "--data"],
        capture_output=True,
        text=True,
    )
    if frozen.returncode != 0:
        sys.exit(f"the frozen baseline does not pass:\n{frozen.stdout}{frozen.stderr}")
    present = tuple(
        s for s in fit.SATELLITES if (args.directory / f"corpus_{s}.npz").exists()
    )
    if not present:
        sys.exit(f"no corpus_<SAT>.npz in {args.directory}")
    rows = fit.load_corpus(
        harness.STEP, phase=harness.PHASE, corpus=args.directory, satellites=present
    )
    if (rows.t.to_numpy() < FROZEN_END).any():
        sys.exit(
            f"{args.directory} holds rows before {FROZEN_END}: they are the frozen corpus's"
        )
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
        f"blind rows of {', '.join(present)} in {args.directory}, the shipped fields; runner at {head}"
        + (" (uncommitted changes)" if dirty else "")
    )
    for path in sorted(args.directory.glob("*")):
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
        "JB2008, clean-room": rust[:, 2],
    }
    every, found, clusters = score(rows, series, g, fit, intervals)
    read_verdicts(rows, series, every, found, clusters, harness, intervals)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    sub.add_parser("cut").set_defaults(func=command_cut)
    sub.add_parser("rival").set_defaults(func=command_rival)
    blind = sub.add_parser("blind")
    blind.add_argument("directory", type=pathlib.Path)
    blind.set_defaults(func=command_blind)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
