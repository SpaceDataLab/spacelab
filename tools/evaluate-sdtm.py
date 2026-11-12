#!/usr/bin/env python3
"""The evaluation harness for SDTM-1, separate from its fit: `plans/17` Ф30.B, D-30.

The fit tool's folds chose the form they report (`docs/sdtm-selection-ledger.md`), so its numbers are
validation. This tool asks the question those folds cannot: the same form, fitted without the time or
the storm it is then scored on. It holds four kinds of split, declared in `docs/sdtm-splits.json` before
any of them is read:

* `family`: the fit tool's folds, kept for continuity. Under `--legacy-penalty` they reproduce its numbers;
* `origin`: rolling-origin by date, training on the rows before the cut and scoring up to the next;
* `frozen`: one cut, training before it and scoring every row after it, with the comparators of
  `plans/17` Ф30.D read on the same rows. Its own kind, so that it never enters the pooled `origin`;
* `storm`: one storm out, the rows of its window and its recovery masked. The indices before and during it
  stay as input, because the storm state needs them and they are not what is being predicted;
* `joint`: a storm and a mission family out together;
* `windows`: another benchmark's cases, read from its manifest (`plans/17` Ф25.E). Every case's span, padded
  by `pad_hours`, is out of training on every satellite; the `hold_out` satellites are out entirely, and
  with `until` so is every row from then on. The scored rows are each case's own satellite inside its span.

Inside each fold everything derived from the data is rebuilt from the training rows or the training hours:
the ridge's scale and, for the solar-wind field, the coupling map. `guard` and `guard_map` refuse a fold
that breaks this, and `tools/test_evaluate_sdtm.py` holds them to it with deliberate leaks. This is
reconstruction: every fold reads the measured indices, as `plans/17` rule 12 says it must state.

    tools/evaluate-sdtm.py splits                      # declare the splits
    tools/evaluate-sdtm.py run --kind family --legacy-penalty
    tools/evaluate-sdtm.py run --kind origin [--field solar_wind]
    tools/evaluate-sdtm.py summary [--field default]
    tools/evaluate-sdtm.py export --split frozen-2020 [--field solar_wind]   # for a Rust caller (Ф30.H, J)

Run under `uv run --project tools --locked python -I`. Results stay under
`data/`, which is never committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(
    "fit_sdtm_corpus", HERE / "fit-sdtm-corpus.py"
)
assert spec is not None and spec.loader is not None
fit = importlib.util.module_from_spec(spec)
sys.modules["fit_sdtm_corpus"] = fit
spec.loader.exec_module(fit)
g = fit.g

ROOT = HERE.parent
SPLITS = ROOT / "docs" / "sdtm-splits.json"
OUT = ROOT / "data" / "validation" / "sdtm-eval"

STEP, PHASE = 60, 0
ORIGINS = ("2005", "2008", "2011", "2014", "2017", "2020", "2023")
FROZEN = "2020"
# The storms large enough to carry a peak the gate names: the first-peak `ap` of the G4-G5 storms.
STORM_PEAK_AP = 300.0
JOINT_NAMED = ("2003-10-29 07:30:00", "2024-05-10 19:30:00", "2024-10-10 22:30:00")
# The recovery is masked with the storm: 72 h past the last phase the protocol scores.
RECOVERY_HOURS = 72.0
FORM: dict[str, Any] = {
    "caps": "storm-log1-lin",
    "ridge": 1.0e-3,
    "sigma": None,
    "cooling": True,
    "excess": "fit",
    "gain": "input-power",
    "ap60": False,
    "wind": None,
    "limit": None,
}
# The ablation's rungs below the shipped form (Ф30.F, `derivations/sdtm_ablation.mac`), each the rung above
# in a limit: no flux gain or saturation, then each storm state, then no storm term, then no geography.
RUNG = {**FORM, "cooling": False, "excess": None, "gain": None}
# The shipped form, and that form with one mechanism taken away or added: `plans/17` Ф30.C and Ф30.F.
FIELDS: dict[str, dict[str, Any]] = {
    "default": FORM,
    "solar_wind": {**FORM, "ap60": True, "wind": "fit"},
    "ap60": {**FORM, "ap60": True},
    "no_gain": {**FORM, "gain": None},
    "no_saturation": {**FORM, "excess": None},
    "no_cooling": {**FORM, "cooling": False},
    # The centred 81-day flux, which NRLMSISE-00 reads, for the trailing one: reconstruction only,
    # since a centred mean needs 40 days of the future.
    "centred": {**FORM, "centred": True},
    "bates": {**RUNG, "caps": "bates", "limit": "instant"},
    "harmonics": {**RUNG, "caps": "harmonics", "limit": "instant"},
    "instant": {**RUNG, "limit": "instant"},
    "first_order": {**RUNG, "limit": "first-order"},
    "two_constant": RUNG,
    "nitric": {**RUNG, "cooling": True},
    # Every satellite on the field's level, no log scale of its own (Ф30.G): with the shipped free
    # scales, the two ends a hierarchical prior on the scales lies between.
    "no_scales": {**FORM, "scales": False},
    # Ф23.J's rivals for the post-storm drop. The composition rival: the boundary's atomic oxygen carries a
    # quiet factor and a storm depletion that outlasts the heating (`derivations/sdtm_composition.mac`).
    "composition": {**FORM, "composition": True},
    # The delta rival: the cooling carries Weimer's quiet offset, so a quiet drive under a high nitric-oxide
    # state overcools; and the review's linear form, which has no saturation
    # (`derivations/sdtm_overcooling.mac`).
    "delta": {**FORM, "delta": True},
    "linear": {**FORM, "linear": True},
    # The nitric oxide's time scaled by the flux, `tn (F10.7a/150)^(-kn)` (`derivations/sdtm_nitric_flux.mac`).
    "nitric_flux": {**FORM, "nitric_flux": "fit"},
    # The same with `kn` held at Wang et al.'s middle ratio.
    "nitric_flux_wang": {**FORM, "nitric_flux": fit.WANG_KN},
    # Ф23.L's memory arm: the daily F10.7 filtered before it reaches T_inf (`derivations/sdtm_flux_memory.mac`),
    # with the fitted heating time and with a time of its own.
    "flux_memory": {**FORM, "flux_memory": "heating"},
    "flux_memory_fit": {**FORM, "flux_memory": "fit"},
    # Ф23.V's latitude rows: the mean field at degree 3, a richer latitude structure of T_inf alone; and the
    # boundary's ln rho with Y20 and an annual Y10, a latitude structure below T_inf.
    "degree3": {**FORM, "degree": 3},
    "boundary_latitude": {**FORM, "boundary_latitude": True},
    # Read after the split by flux: the flux's response at degree 2, alone and with the boundary's structure.
    "flux_latitude": {**FORM, "caps": "storm-log1-lin-f2"},
    # Read after §10.49's smoothing table: no Swarm row in any fold's training set, so no row whose density
    # comes from GPS accelerations at about 20 min resolution. The Swarm fold is the shipped one.
    "no_swarm": {**FORM, "drop_families": ("Swarm",)},
    "flux_boundary_latitude": {
        **FORM,
        "caps": "storm-log1-lin-f2",
        "boundary_latitude": True,
    },
    # The shipped form on GRACE-FO C's v02 densities where v02 exists, v02c elsewhere, the rows unchanged
    # (`tools/corpus-variant.py`; Ф30.G-2, F72).
    "gc_v02": {**FORM, "corpus": "corpus-gc-v02"},
}


class LeakError(ValueError):
    """A fold's training set holds something of what it is scored on."""


def stamp(text: str) -> np.datetime64:
    return np.datetime64(pd.Timestamp(text))


def declared_splits(
    ap: np.ndarray, epoch: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp
):
    """The splits, in a fixed order, from the 3-hourly `ap` and the corpus's span alone."""
    out: list[dict[str, str]] = [
        {"name": f"family-{f}", "kind": "family", "family": f}
        for f in sorted(set(fit.FAMILY.values()))
    ]
    cuts = [*ORIGINS, f"{last.year + 1}"]
    out += [
        {
            "name": f"origin-{cut}",
            "kind": "origin",
            "cut": f"{cut}-01-01",
            "until": f"{following}-01-01",
        }
        for cut, following in itertools.pairwise(cuts)
    ]
    out.append(
        {
            "name": f"frozen-{FROZEN}",
            "kind": "frozen",
            "cut": f"{FROZEN}-01-01",
            "until": f"{last.year + 1}-01-01",
        }
    )
    start_index = (first.normalize() - epoch).days * 8
    found = g.storm_onsets(
        ap[start_index:], epoch + pd.Timedelta(days=start_index // 8)
    )
    for s in found:
        if s["peak_ap"] < STORM_PEAK_AP:
            continue
        start = s["t0"] + pd.Timedelta(hours=g.PHASES[0][0])
        end = s["t0"] + pd.Timedelta(hours=s["phase3_end"] + RECOVERY_HOURS)
        name = s["t0"].strftime("%Y-%m-%dT%H")
        window = {"start": str(start), "end": str(end)}
        out.append({"name": f"storm-{name}", "kind": "storm", **window})
        if str(s["t0"]) in JOINT_NAMED:
            out += [
                {"name": f"joint-{name}-{f}", "kind": "joint", "family": f, **window}
                for f in sorted(set(fit.FAMILY.values()))
            ]
    return out + list(BENCHMARKS)


# Other benchmarks' cases, each manifest written by `tools/benchmark-splits.py`. STORM-AI trained without
# Swarm B; its orbit means read half an orbit past each end of the 72 h. AETHER-P3 Nowcast trained on
# densities up to 2023 and on neither Swarm A nor Swarm B; its scores are at points.
BENCHMARKS: tuple[dict[str, Any], ...] = (
    {
        "name": "storm-ai",
        "kind": "windows",
        "manifest": "docs/storm-ai-split.json",
        "hold_out": ["SB"],
        "pad_hours": 1.0,
    },
    {
        "name": "aether-p3",
        "kind": "windows",
        "manifest": "docs/aether-p3-cases.json",
        "hold_out": ["SA", "SB"],
        "pad_hours": 0.0,
        "until": "2024-01-01",
    },
)


def case_spans(split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(satellite, start, end)` of a `windows` split's cases: inline `cases`, else its manifest's."""
    cases = split.get("cases")
    if cases is None:
        manifest = json.loads((ROOT / split["manifest"]).read_text(encoding="utf-8"))
        cases = [
            dict(zip(manifest["columns"], c, strict=True)) for c in manifest["cases"]
        ]
    sat = np.array([c["sat"] for c in cases])
    start, end = (
        np.array([stamp(c[key]) for c in cases], dtype="datetime64[ns]")
        for key in ("start", "end")
    )
    return sat, start, end


def inside_spans(t: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Whether each `t` lies in some `[start, end)`: inside one exactly when it is before the latest end
    of the spans that begin at or before it."""
    order = np.argsort(start, kind="stable")
    begins, reach = start[order], np.maximum.accumulate(end[order])
    i = np.searchsorted(begins, t, side="right") - 1
    return (i >= 0) & (t < reach[np.maximum(i, 0)])


def window_masks(
    t: np.ndarray, sat: np.ndarray, split: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """A `windows` split's training and scored rows, as boolean masks."""
    case_sat, start, end = case_spans(split)
    pad = np.timedelta64(round(split["pad_hours"] * 3600), "s")
    train = ~inside_spans(t, start - pad, end + pad) & ~np.isin(sat, split["hold_out"])
    if "until" in split:
        train &= t < stamp(split["until"])
    test = np.zeros(t.size, dtype=bool)
    for s in np.unique(case_sat):
        mine = case_sat == s
        test |= (sat == s) & inside_spans(t, start[mine], end[mine])
    return train, test


def masks(
    t: np.ndarray,
    family: np.ndarray,
    split: dict[str, Any],
    sat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The training rows and the scored rows of one split, as indices. A `windows` split needs `sat`."""
    kind = split["kind"]
    if kind == "family":
        test = family == split["family"]
        train = ~test
    elif kind in ("origin", "frozen"):
        train = t < stamp(split["cut"])
        test = (t >= stamp(split["cut"])) & (t < stamp(split["until"]))
    elif kind in ("storm", "joint"):
        inside = (t >= stamp(split["start"])) & (t < stamp(split["end"]))
        if kind == "storm":
            train, test = ~inside, inside
        else:
            held = family == split["family"]
            train, test = ~(inside | held), inside & held
    elif kind == "windows":
        if sat is None:
            raise ValueError(
                f"{split['name']}: a windows split needs each row's satellite"
            )
        train, test = window_masks(t, sat, split)
    else:
        raise ValueError(f"no such split kind: {kind}")
    return np.flatnonzero(train), np.flatnonzero(test)


def guard(
    t: np.ndarray,
    family: np.ndarray,
    split: dict[str, Any],
    train: np.ndarray,
    test: np.ndarray,
    sat: np.ndarray | None = None,
) -> None:
    """Refuse a fold whose training rows hold anything it is scored on."""
    if np.intersect1d(train, test).size:
        raise LeakError(f"{split['name']}: a row is in both sets")
    kind = split["kind"]
    if kind == "windows":
        if sat is None:
            raise ValueError(
                f"{split['name']}: a windows split needs each row's satellite"
            )
        _, start, end = case_spans(split)
        pad = np.timedelta64(round(split["pad_hours"] * 3600), "s")
        if inside_spans(t[train], start - pad, end + pad).any():
            raise LeakError(f"{split['name']}: a training row from inside a case")
        if np.isin(sat[train], split["hold_out"]).any():
            raise LeakError(
                f"{split['name']}: a held-out satellite is in the training rows"
            )
        if "until" in split and (t[train] >= stamp(split["until"])).any():
            raise LeakError(f"{split['name']}: a training row from after its end")
    if kind in ("family", "joint") and (family[train] == split["family"]).any():
        raise LeakError(f"{split['name']}: the held-out family is in the training rows")
    if kind in ("origin", "frozen") and (t[train] >= stamp(split["cut"])).any():
        raise LeakError(f"{split['name']}: a training row from after the cut")
    if kind in ("storm", "joint"):
        inside = (t[train] >= stamp(split["start"])) & (t[train] < stamp(split["end"]))
        if inside.any():
            raise LeakError(f"{split['name']}: a training row from inside the storm")


def training_hours(hours: np.ndarray, split: dict[str, str]) -> np.ndarray:
    """The hours a fold may read a whole-series table from: its training time, not its test time."""
    kind = split["kind"]
    if kind in ("origin", "frozen"):
        return hours < stamp(split["cut"])
    if kind in ("storm", "joint"):
        return (hours < stamp(split["start"])) | (hours >= stamp(split["end"]))
    if kind == "windows":
        _, start, end = case_spans(split)
        pad = np.timedelta64(round(split["pad_hours"] * 3600), "s")
        allowed = ~inside_spans(hours, start - pad, end + pad)
        return (
            allowed & (hours < stamp(split["until"])) if "until" in split else allowed
        )
    return np.ones(hours.size, dtype=bool)


def guard_map(hours_used: np.ndarray, split: dict[str, str]) -> None:
    """Refuse a coupling map built on any hour the fold is scored on."""
    if not training_hours(hours_used, split).all():
        raise LeakError(
            f"{split['name']}: the coupling map read an hour of the test span"
        )


def fold_map(
    on_grid: np.ndarray, hourly: np.ndarray, epoch: pd.Timestamp, split: dict[str, str]
):
    """The coupling map rebuilt from the fold's training hours only."""
    present = np.isfinite(on_grid)
    hours = np.asarray(
        epoch + pd.to_timedelta(np.flatnonzero(present), unit="h"),
        dtype="datetime64[ns]",
    )
    allowed = training_hours(hours, split)
    guard_map(hours[allowed], split)
    return fit.coupling_map(
        on_grid[present][allowed],
        hourly[present][allowed],
        pd.DatetimeIndex(hours[allowed]),
    )


def fold_drive(
    on_grid: np.ndarray, hourly: np.ndarray, epoch: pd.Timestamp, split: dict[str, str]
):
    """The solar-wind drive with the coupling map rebuilt from the fold's training hours only."""
    present = np.isfinite(on_grid)
    out = hourly.copy()
    out[present] = fold_map(on_grid, hourly, epoch, split)(on_grid[present])
    return out


def read_splits() -> list[dict[str, str]]:
    return json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]


def command_splits(_args) -> int:
    rows = fit.load_corpus(STEP, phase=PHASE)
    splits = declared_splits(
        rows.attrs["ap"], rows.attrs["ap_epoch"], rows.t.min(), rows.t.max()
    )
    SPLITS.write_text(
        json.dumps(
            {
                "step": STEP,
                "phase": PHASE,
                "recovery_hours": RECOVERY_HOURS,
                "splits": splits,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    kinds = pd.Series([s["kind"] for s in splits]).value_counts().to_dict()
    print(f"{len(splits)} splits: {kinds}")
    return 0


def fold_model(
    arg: dict,
    sat: np.ndarray,
    train_sats: tuple[str, ...],
    settings: dict[str, Any],
    penalty_rows: np.ndarray | None,
):
    """A field's `CorpusModel` on `arg`'s rows, with a slot for each satellite of `train_sats`: a fold's
    fitted parameters score any rows through it."""
    return fit.CorpusModel(
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
        penalty_rows=penalty_rows,
        limit=settings["limit"],
        scales=settings.get("scales", True),
        composition=settings.get("composition", False),
        delta=settings.get("delta", False),
        linear=settings.get("linear", False),
        nitric_flux=settings.get("nitric_flux"),
        flux_memory=settings.get("flux_memory"),
        boundary_latitude=settings.get("boundary_latitude", False),
    )


def command_run(args) -> int:
    settings = FIELDS[args.field]
    variant = settings.get("corpus")
    rows = fit.load_corpus(
        STEP,
        trailing_mean=not settings.get("centred", False),
        phase=PHASE,
        ap60=settings["ap60"],
        wind=settings["wind"] is not None,
        corpus=fit.CORPUS if variant is None else fit.CORPUS.parent / variant,
    )
    arg = g.fit_arguments(rows, settings.get("degree", 2))
    arg["wind"] = rows.attrs.get("wind")
    arg["wind_map"] = rows.attrs.get("wind_map")
    sat = rows.sat.to_numpy()
    family = rows.family.to_numpy()
    t = rows.t.to_numpy()
    day_ap = rows.attrs["ap"].reshape(-1, 8).mean(axis=1)
    ap_daily = day_ap[
        (rows.t.dt.normalize() - rows.attrs["ap_epoch"]).dt.days.to_numpy()
    ]
    suffix = "-legacy" if args.legacy_penalty else ""
    out_dir = OUT / f"{args.field}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # NRLMSISE-00 against the measured densities: a corpus variant measures differently.
    msis_path = OUT / ("msis.npy" if variant is None else f"msis-{variant}.npy")
    if not msis_path.exists():
        np.save(msis_path, fit.nrlmsise00(rows) - arg["ln_rho"])
    msis = np.load(msis_path)
    if settings["wind"] is not None:
        hourly, hourly_epoch = rows.attrs["driver"], rows.attrs["driver_epoch"]
        on_grid = fit.coupling_on_grid(hourly, hourly_epoch)
    for split in read_splits():
        if split["kind"] != args.kind or (args.only and split["name"] != args.only):
            continue
        target = out_dir / f"{split['name']}.npz"
        if target.exists():
            continue
        train, test = masks(t, family, split, sat)
        train = train[~np.isin(family[train], settings.get("drop_families", ()))]
        guard(t, family, split, train, test, sat)
        if test.size == 0:
            print(f"{split['name']}: no rows to score", flush=True)
            continue
        fold_arg = arg
        if settings["wind"] is not None and not args.legacy_penalty:
            # `wind_map` is the whole span's, and only the emitter reads it: a stray read here must fail.
            fold_arg = dict(
                arg,
                wind=fold_drive(on_grid, hourly, hourly_epoch, split),
                wind_map=None,
            )
        seen = set(sat[train])
        train_sats = tuple(s for s in fit.SATELLITES if s in seen)
        began = time.time()
        model = fold_model(
            fold_arg,
            sat,
            train_sats,
            settings,
            None if args.legacy_penalty else train,
        )
        p = model.fit(train)
        residual, _ = model.score(p, test)
        np.savez_compressed(
            target,
            test=test,
            residual=residual,
            msis=msis[test],
            sat=sat[test].astype("U2"),
            ap_daily=ap_daily[test],
            train_rows=train.size,
            parameters=p,
            train_sats=np.array(train_sats),
        )
        print(
            f"{split['name']}: {train.size} train, {test.size} test, MAPE {g.mape(residual):.1f} %, "
            f"{time.time() - began:.0f} s",
            flush=True,
        )
    return 0


def command_summary(args) -> int:
    in_dir = OUT / args.field
    print(
        "split | train | test | MAPE | debiased | NRLMSISE-00 | Ap >= 50 | NRLMSISE-00 | median model/measured"
    )
    pooled: dict[str, list[np.ndarray]] = {}
    for split in read_splits():
        path = in_dir / f"{split['name']}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        residual, msis = z["residual"], z["msis"]
        done = np.isfinite(msis)
        storm = z["ap_daily"] >= 50.0
        pooled.setdefault(split["kind"], []).append(
            np.column_stack([residual, fit.debias(residual, z["sat"]), msis])
        )
        stormy = (done & storm).any()
        print(
            f"{split['name']} | {int(z['train_rows'])} | {residual.size} | {g.mape(residual[done]):.1f} % | "
            f"{g.mape(fit.debias(residual, z['sat'])[done]):.1f} % | {g.mape(msis[done]):.1f} % | "
            + (
                f"{g.mape(residual[done & storm]):.1f} % | {g.mape(msis[done & storm]):.1f} % | "
                if stormy
                else "— | — | "
            )
            + f"{np.exp(np.median(residual[done])):.3f}"
        )
    for kind, parts in pooled.items():
        stacked = np.vstack(parts)
        done = np.isfinite(stacked[:, 2])
        print(
            f"pooled {kind}: {done.sum()} rows, MAPE {g.mape(stacked[done, 0]):.1f} %, "
            f"debiased by fold {g.mape(stacked[done, 1]):.1f} %, NRLMSISE-00 {g.mape(stacked[done, 2]):.1f} %"
        )
    return 0


def command_export(args) -> int:
    """Write a fold's fitted field as `Thermosphere::from_coefficients`'s arguments, and the generator's own
    values at 64 of the fold's scored rows, so a Rust caller can rebuild the field and check it (Ф30.H,
    Ф30.J).

    The field file also holds each satellite's log scale against the fold's reference (`scale_<SAT>`),
    and for the solar-wind field the share and the coupling map the fold rebuilt from its training hours
    (`share`, `knot_coupling`, `knot_ap60`). The second file has the columns of
    `crates/spacelab-propagation/tests/fixtures/sdtm_reference.csv`: the field alone, with no satellite's
    log scale, as `fit-sdtm-corpus.py`'s `emit` writes it.
    """
    settings = FIELDS[args.field]
    if args.field not in ("default", "solar_wind"):
        sys.exit(
            "export writes the shipped forms only: the 3-hourly field and the solar-wind one"
        )
    saved = np.load(OUT / args.field / f"{args.split}.npz")
    wind = settings["wind"] is not None
    rows = fit.load_corpus(
        STEP, trailing_mean=True, phase=PHASE, ap60=settings["ap60"], wind=wind
    )
    arg = g.fit_arguments(rows, 2)
    split = next(s for s in read_splits() if s["name"] == args.split)
    sat = rows.sat.to_numpy()
    # The fold's training satellites, as `command_run` read them; a fold saved before it kept them
    # has none to compare with.
    train, _ = masks(rows.t.to_numpy(), rows.family.to_numpy(), split, sat)
    train_sats = tuple(s for s in fit.SATELLITES if s in set(sat[train]))
    if "train_sats" in saved.files and tuple(saved["train_sats"]) != train_sats:
        sys.exit(f"{args.split}: the fold trained on {tuple(saved['train_sats'])}")
    wind_lines = []
    if wind:
        hourly, epoch = rows.attrs["driver"], rows.attrs["driver_epoch"]
        on_grid = fit.coupling_on_grid(hourly, epoch)
        wind_map = fold_map(on_grid, hourly, epoch, split)
        # As `command_run` scores the fold: the drive on the fold's own map, and no whole-span map.
        arg["wind"] = fold_drive(on_grid, hourly, epoch, split)
        arg["wind_map"] = None
        wind_lines = [
            f"knot_coupling {' '.join(g.literal(c) for c in wind_map.coupling)}",
            f"knot_ap60 {' '.join(g.literal(a) for a in wind_map.ap60)}",
        ]
    p = saved["parameters"]
    model = fold_model(arg, sat, train_sats, settings, None)
    if p.size != model.pack.size:
        sys.exit(
            f"{args.split}: {p.size} parameters saved, the form has {model.pack.size}"
        )
    pk = model.pack
    kernel = model.kernel(p)
    heating, nitric, saturation = kernel[:3]
    grid = np.zeros((g.DRIVERS, arg["k"]))
    for (d, k), c in zip(model.pairs, p[: pk.n_c], strict=True):
        grid[d, k] = c
    boundary = p[pk.i_boundary]
    lines = [
        f"# SDTM-1 field fitted by tools/evaluate-sdtm.py, field {args.field}, split {args.split}.",
        f"sigma {g.literal(model.shape(p))}",
        f"boundary_scale {g.literal(np.exp(boundary))}",
        f"heating_s {g.literal(heating * 3600.0)}",
        f"nitric_oxide_s {g.literal(nitric * 3600.0)}",
        f"saturation {g.literal(saturation)}",
        f"excess_scale {g.literal(model.excess(p))}",
        f"gain_exponent {g.literal(model.gain(p))}",
        *(f"coefficients {' '.join(g.literal(c) for c in row)}" for row in grid),
        *(
            f"scale_{sat} {g.literal(s)}"
            for sat, s in zip(
                pk.scaled,
                p[pk.i_scale : pk.i_scale + len(pk.scaled)],
                strict=True,
            )
        ),
    ]
    share = model.share(kernel)
    if share is not None:
        lines += [f"share {g.literal(share)}", *wind_lines]
    stem = OUT / args.field / args.split
    stem.with_suffix(".field").write_text("\n".join(lines) + "\n")

    test = saved["test"]
    index = test[np.linspace(0, test.size - 1, 64).astype(int)]
    exponent = (
        model.design(*kernel, excess=model.excess(p), gain=model.gain(p))[index]
        @ p[: pk.n_c]
    )
    t_inf = g.BOUNDARY_TEMPERATURE + np.exp(exponent)
    ln_rho, _ = g.profile(t_inf, model.shape(p), arg["zeta"][index])
    storm = fit.cooling_grid(
        model.drive(kernel), heating, nitric, saturation, arg["cadence"], model.first
    )[arg["interval"]]
    fixture = [
        "# altitude_m,latitude_rad,local_solar_time_h,day_of_year,f107,f107a,storm_ap,t_inf_K,density_kg_m3"
    ]
    for j, i in enumerate(index):
        fixture.append(
            ",".join(
                g.literal(x)
                for x in (
                    rows.altitude.iat[i],
                    np.deg2rad(rows.latitude.iat[i]),
                    rows.lst.iat[i],
                    arg["doy"][i],
                    arg["f107"][i],
                    arg["f107a"][i],
                    storm[i],
                    t_inf[j],
                    np.exp(ln_rho[j] + boundary),
                )
            )
        )
    stem.with_suffix(".fixture.csv").write_text("\n".join(fixture) + "\n")
    print(f"wrote {stem.with_suffix('.field')} and {stem.with_suffix('.fixture.csv')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True)
    sub.add_parser("splits").set_defaults(func=command_splits)
    runner = sub.add_parser("run")
    runner.add_argument(
        "--kind",
        required=True,
        choices=("family", "origin", "frozen", "storm", "joint", "windows"),
    )
    runner.add_argument("--only")
    runner.add_argument("--field", default="default", choices=sorted(FIELDS))
    runner.add_argument(
        "--legacy-penalty",
        action="store_true",
        help="scale the ridge over every row, as the fit tool does: the reproduction gate only",
    )
    runner.set_defaults(func=command_run)
    summary = sub.add_parser("summary")
    summary.add_argument("--field", default="default")
    summary.set_defaults(func=command_summary)
    export = sub.add_parser("export")
    export.add_argument("--split", required=True)
    export.add_argument("--field", default="default", choices=sorted(FIELDS))
    export.set_defaults(func=command_export)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
