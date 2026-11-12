#!/usr/bin/env python3
"""SDTM-1 on other benchmarks' cases, through the harness's `windows` folds (`plans/17` Ф25.E).

Each benchmark is a manifest under `docs/` and a split of `tools/evaluate-sdtm.py`, which fits the shipped
form without any row of a case's span, without the satellites the benchmark held out of its own training,
and, where it says so, without any row after its training ended. This tool writes the manifests and scores
the fitted folds on the benchmarks' own terms:

* **STORM-AI** (Sanchez-Hurtado et al. 2026, doi:10.1029/2026SW004994; data CC BY 4.0, Harvard Dataverse
  doi:10.7910/DVN/U6K6MJ): 4 537 evaluation cases in Phase 1.1, Phase 1.2 and the private set, each the
  orbit-mean density at 10 min over the 72 h after an initial state. Swarm B is never in its training. The
  target is TU Delft's running orbit mean at UTC ticks, so each model here is averaged the same way: along
  the corpus's 120 s rows, over one orbital period centred on each tick. Two input modes: reconstruction,
  the measured indices through the 72 h, as every harness fold reads; and persistence, the 3-hourly `ap`
  held from the last whole interval before the initial state and F10.7 and its 81-day mean from the day
  before it, which is all a STORM-AI entry knew. NRLMSISE-00 and the clean-room JB2008 read the measured
  indices, an advantage over persistence.
* **AETHER-P3 Nowcast** (Wang & Bai 2026, arXiv:2609.06777, CC BY 4.0): twelve cases in February 2015 and
  May 2024, Table 7, dates only and read as whole UTC days. Its RE is the mean absolute relative error at
  points, then the unweighted mean over cases (Table 8). Reconstruction only: it is a driver-based nowcast.

    tools/benchmark-splits.py storm-ai-manifest DIR   # DIR holds tools/fetch-storm-ai.sh's archives
    tools/benchmark-splits.py aether-manifest
    tools/evaluate-sdtm.py run --kind windows --only storm-ai
    tools/benchmark-splits.py score storm-ai [--field default]
    tools/benchmark-splits.py table storm-ai [--field default]

Run under `uv run --no-project --with numpy --with pandas --with scipy python -I`. The targets and the scores
stay under `data/`, which is never committed; the manifests name cases, not densities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "evaluate_sdtm", HERE / "evaluate-sdtm.py"
)
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
sys.modules["evaluate_sdtm"] = harness
spec.loader.exec_module(harness)
fit, g = harness.fit, harness.g

ROOT = HERE.parent
TRUTH = ROOT / "data" / "validation" / "storm-ai" / "truth.npz"
SPLIT = {s["name"]: s for s in harness.BENCHMARKS}
STORM_AI_DOI = "doi:10.7910/DVN/U6K6MJ"
HORIZON = pd.Timedelta(hours=72)
TICK = pd.Timedelta(minutes=10)
# STORM-AI's mission folders, as the corpus names the satellites. GRACE-FO 1 is GRACE-C.
MISSION = {
    "CHAMP": "CH",
    "GRACE 1": "GA",
    "GRACE 2": "GB",
    "GRACE-FO 1": "GC",
    "SWARM A": "SA",
    "SWARM B": "SB",
    "SWARM C": "SC",
}
SUBSETS = {
    "initial_states_phase_1_1.csv": "1.1",
    "initial_states_phase_1_2.csv": "1.2",
    "initial_states_private_eval.csv": "private",
}
# The evaluation set's composition, STORM-AI's documentation (`docs/dataset.md` of its devkit).
COUNTS = {
    "1.1": {"CH": 1129, "GA": 29, "GB": 949, "GC": 640},
    "1.2": {"SB": 844, "SC": 483},
    "private": {"CH": 10, "GB": 10, "SA": 102, "SB": 122, "SC": 109, "GC": 130},
}
MU_KM3_S2 = 398_600.4418
# A tick's mean needs nine-tenths of the period's rows; fewer and the corpus has a gap there.
COVERAGE = 0.9
# STORM-AI's propagation score weights a tick at `exp(-ln(1/eps) dt/T)` over the horizon `T`.
EPSILON = 1.0e-5
# AETHER-P3 Nowcast's Table 7, and its Table 9 RE for each model: AETHER-P3 (seed 20), HASDM, JB2008,
# NRLMSISE-00, WAM-IPE (May 2024 only).
AETHER_CASES = (
    ("SA", "2024-05-24", "2024-05-31", "High-solar quiet"),
    ("SC", "2024-05-24", "2024-05-31", "High-solar quiet"),
    ("SA", "2015-02-10", "2015-02-15", "Medium-solar quiet"),
    ("SC", "2015-02-10", "2015-02-15", "Medium-solar quiet"),
    ("SA", "2015-02-16", "2015-02-20", "Moderate event I"),
    ("SC", "2015-02-16", "2015-02-20", "Moderate event I"),
    ("SA", "2015-02-23", "2015-02-25", "Moderate event II"),
    ("SC", "2015-02-23", "2015-02-25", "Moderate event II"),
    ("GC", "2024-05-10", "2024-05-13", "Extreme storm"),
    ("SA", "2024-05-10", "2024-05-13", "Extreme storm"),
    ("SB", "2024-05-10", "2024-05-13", "Extreme storm"),
    ("SC", "2024-05-10", "2024-05-13", "Extreme storm"),
)
AETHER_RE = {
    "AETHER-P3": (
        *(0.143036, 0.141980, 0.090956, 0.086915, 0.081684, 0.085234),
        *(0.073291, 0.081610, 0.164603, 0.136497, 0.167655, 0.135025),
    ),
    "HASDM": (
        *(0.113216, 0.134148, 0.125945, 0.118162, 0.128879, 0.129354),
        *(0.138669, 0.139972, 0.206805, 0.175488, 0.184123, 0.172525),
    ),
    "JB2008": (
        *(0.290732, 0.280593, 0.134046, 0.127988, 0.134813, 0.136542),
        *(0.135201, 0.139370, 0.334363, 0.305439, 0.304762, 0.309680),
    ),
    "NRLMSISE-00": (
        *(0.470256, 0.494741, 0.206978, 0.210512, 0.146154, 0.152656),
        *(0.126469, 0.133006, 0.826327, 0.799140, 0.842948, 0.802217),
    ),
    "WAM-IPE": (
        *(1.012565, 1.076193),
        *(np.nan,) * 6,
        *(0.907220, 0.804714, 1.005889, 0.815642),
    ),
}


def write_manifest(split: dict, cases: list[dict], source: dict) -> None:
    columns = list(cases[0])
    body = ",\n".join(
        "  " + json.dumps([c[k] for k in columns], ensure_ascii=False) for c in cases
    )
    head = json.dumps({"source": source, "columns": columns}, indent=1)[:-2]
    (ROOT / split["manifest"]).write_text(
        f'{head},\n "cases": [\n{body}\n ]\n}}\n', encoding="utf-8"
    )


def command_storm_ai_manifest(args) -> int:
    """The cases and their targets, from the Dataverse files: initial states, then each mission's targets."""
    source = pathlib.Path(args.dir)
    archives = sorted(source.glob("private_eval__*.zip"))
    states = source / "private_eval__initial_states.zip"
    if states not in archives or len(archives) < 12:
        sys.exit(
            f"{source}: need the private_eval initial states and eleven mission archives"
        )
    md5 = {}
    for a in archives:
        with a.open("rb") as f:
            md5[a.name] = hashlib.file_digest(f, "md5").hexdigest()
    # Twenty cases, ten CHAMP and ten GRACE-B in October 2003, are in Phase 1.1 and the private set both, at
    # the same instant and with the same target. The private set gives the GRACE-B ten other elements, 12 km
    # lower in semi-major axis; the first list's, which are the mission archive's own, set the period.
    subset: dict[int, tuple[list[str], pd.Timestamp, float]] = {}
    moved = 0
    with zipfile.ZipFile(states) as z:
        for member, name in SUBSETS.items():
            table = pd.read_csv(io.BytesIO(z.read(f"initial_states/{member}")))
            for file_id, t0, sma in zip(
                table["File ID"],
                pd.to_datetime(table["Timestamp"]),
                table["Semi-major Axis (km)"],
                strict=True,
            ):
                if int(file_id) not in subset:
                    subset[int(file_id)] = ([name], t0, float(sma))
                    continue
                names, first_t0, first_sma = subset[int(file_id)]
                if t0 != first_t0:
                    sys.exit(f"case {file_id} starts at {first_t0} and at {t0}")
                names.append(name)
                moved += float(sma) != first_sma
    # The archives hold every case of each mission's span, of which the evaluation is a subset; the
    # 2021-2024 GRACE-FO archive also holds 23 cases twice, a day apart, none of them evaluated.
    target = re.compile(
        r"^([^/]+)/sat_density/[a-z0-9_-]+-(\d{5})-\d{8}_to_\d{8}\.csv$"
    )
    cases, ids, period, offsets, ticks, rho, lead = [], [], [], [0], [], [], []
    for archive in archives:
        if archive == states:
            continue
        with zipfile.ZipFile(archive) as z:
            for member in sorted(z.namelist()):
                found = target.match(member)
                if not found or int(found.group(2)) not in subset:
                    continue
                folder = re.sub(r"^\d{4}(-\d{4})? ", "", found.group(1))
                file_id = int(found.group(2))
                names, t0, sma = subset[file_id]
                frame = pd.read_csv(io.BytesIO(z.read(member)))
                stamps = pd.to_datetime(frame["Timestamp"])
                values = frame["Orbit Mean Density (kg/m^3)"].to_numpy(
                    dtype=float, copy=True
                )
                values[(values <= 0.0) | (values > 1.0e30)] = np.nan
                cases.append(
                    {
                        "id": file_id,
                        "sat": MISSION[folder],
                        "start": str(min(t0, stamps.min())),
                        "end": str(
                            max(t0 + HORIZON, stamps.max() + pd.Timedelta(seconds=1))
                        ),
                        "subsets": names,
                    }
                )
                lead.append((stamps.min() - t0).total_seconds())
                ids.append(file_id)
                period.append(2.0 * np.pi * np.sqrt(sma**3 / MU_KM3_S2))
                ticks.append(
                    (
                        (stamps - fit.EPOCH_2000) // pd.Timedelta(milliseconds=1)
                    ).to_numpy()
                )
                rho.append(values)
                offsets.append(offsets[-1] + len(values))
    order = np.argsort(ids, kind="stable")
    cases = [cases[i] for i in order]
    if len({c["id"] for c in cases}) != len(cases):
        sys.exit("a case has two targets")
    if len(cases) != len(subset):
        sys.exit(f"{len(subset) - len(cases)} evaluation cases have no target")
    counted: dict[str, dict[str, int]] = {}
    for c in cases:
        for name in c["subsets"]:
            counted.setdefault(name, {}).setdefault(c["sat"], 0)
            counted[name][c["sat"]] += 1
    if counted != COUNTS:
        sys.exit(f"the composition differs from STORM-AI's documentation: {counted}")
    write_manifest(
        SPLIT["storm-ai"],
        cases,
        {
            "benchmark": "STORM-AI evaluation split (Phase 1.1, Phase 1.2, private)",
            "dataset": STORM_AI_DOI,
            "licence": "CC BY 4.0",
            "md5": md5,
        },
    )
    TRUTH.parent.mkdir(parents=True, exist_ok=True)
    pieces = [(ticks[i], rho[i]) for i in order]
    lengths = np.array([len(p[1]) for p in pieces])
    np.savez_compressed(
        TRUTH,
        id=np.array(ids)[order],
        period_s=np.array(period)[order],
        offset=np.concatenate([[0], np.cumsum(lengths)]),
        t_ms=np.concatenate([p[0] for p in pieces]),
        rho=np.concatenate([p[1] for p in pieces]),
    )
    print(
        f"{len(cases)} cases {counted}; {lengths.sum()} targets in {TRUTH}; {moved} with other "
        f"elements in a second subset; first target {min(lead):+.0f} to {max(lead):+.0f} s from t0"
    )
    return 0


def command_aether_manifest(_args) -> int:
    cases = [
        {
            "id": i + 1,
            "sat": sat,
            "start": f"{first} 00:00:00",
            "end": str(pd.Timestamp(last) + pd.Timedelta(days=1)),
            "subset": label,
        }
        for i, (sat, first, last, label) in enumerate(AETHER_CASES)
    ]
    write_manifest(
        SPLIT["aether-p3"],
        cases,
        {
            "benchmark": "AETHER-P3 Nowcast's twelve benchmark cases (Table 7; dates as whole UTC days)",
            "paper": "Wang & Bai 2026, arXiv:2609.06777",
            "licence": "CC BY 4.0",
        },
    )
    print(f"{len(cases)} cases")
    return 0


def manifest_cases(split: dict) -> pd.DataFrame:
    manifest = json.loads((ROOT / split["manifest"]).read_text(encoding="utf-8"))
    cases = pd.DataFrame(manifest["cases"], columns=manifest["columns"])
    cases["start"], cases["end"] = (pd.to_datetime(cases[k]) for k in ("start", "end"))
    return cases


def fold(field: str, name: str) -> tuple[dict, np.ndarray, tuple[str, ...]]:
    settings = harness.FIELDS[field]
    unsupported = [
        key
        for key in ("ap60", "wind", "centred", "nitric_flux", "flux_memory", "corpus")
        if settings.get(key)
    ]
    if unsupported:
        sys.exit(f"field {field}: persistence does not hold {unsupported}")
    path = harness.OUT / field / f"{name}.npz"
    if not path.exists():
        sys.exit(
            f"no {path}; run tools/evaluate-sdtm.py run --kind windows --only {name}"
        )
    z = np.load(path)
    return settings, z["parameters"], tuple(str(s) for s in z["train_sats"])


def satellite_rows(cases: pd.DataFrame, sat: str, pad: pd.Timedelta) -> pd.DataFrame:
    spans = [(s - pad, e + pad) for s, e in zip(cases.start, cases.end, strict=True)]
    rows = fit.load_corpus(1, windows=spans)
    keep = (rows.sat == sat).to_numpy()
    out = rows[keep].reset_index(drop=True)
    out.attrs = dict(rows.attrs)
    return out


def sliced(
    rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, int]:
    """The rows in `[start, end)` and where they begin in `rows`, which are one satellite's, in time."""
    t = rows.t.to_numpy()
    a, b = (
        np.searchsorted(t, np.datetime64(start)),
        np.searchsorted(t, np.datetime64(end)),
    )
    out = rows.iloc[a:b].reset_index(drop=True)
    out.attrs = dict(rows.attrs)
    return out, int(a)


def persisted(rows: pd.DataFrame, t0: pd.Timestamp, flux: pd.DataFrame) -> pd.DataFrame:
    """`rows` with every index after `t0` held at what was known at `t0`: `ap` of the last whole 3 h
    interval, F10.7 and its trailing 81-day mean of the day before."""
    out = rows.copy()
    out.attrs = dict(rows.attrs)
    epoch = rows.attrs["driver_epoch"]
    last = (t0.normalize() - epoch).days * 8 + t0.hour // 3 - 1
    drive = rows.attrs["driver"].copy()
    drive[last + 1 :] = drive[last]
    out.attrs["driver"] = drive
    known = flux.loc[t0.normalize() - pd.Timedelta(days=1)]
    after = (out.t >= t0).to_numpy()
    out.loc[after, "f107"] = known["F10.7_OBS"]
    out.loc[after, "f107a"] = known["F10.7_OBS_LAST81"]
    return out


def model_ln_rho(rows, p, train_sats, settings) -> np.ndarray:
    arg = g.fit_arguments(rows, settings.get("degree", 2))
    model = harness.fold_model(arg, rows.sat.to_numpy(), train_sats, settings, None)
    residual, _ = model.score(p, np.arange(len(rows)))
    return residual + arg["ln_rho"]


def centred_means(
    t_ms: np.ndarray, rho: np.ndarray, ticks_ms: np.ndarray, period_s: float
):
    """Each column of `rho` averaged over `[tick - P/2, tick + P/2)`, `NaN` where a gap leaves fewer than
    `COVERAGE` of the period's rows."""
    half = period_s * 500.0
    lo = np.searchsorted(t_ms, ticks_ms - half)
    hi = np.searchsorted(t_ms, ticks_ms + half)
    sums = np.vstack([np.zeros((1, rho.shape[1])), np.cumsum(rho, axis=0)])
    n = hi - lo
    means = (sums[hi] - sums[lo]) / np.maximum(n, 1)[:, None]
    full = n >= COVERAGE * period_s / 120.0
    return np.where(full[:, None], means, np.nan)


def comparator_columns(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """`ln rho` of each comparator on `rows`, from the measured indices."""
    columns = {"NRLMSISE-00": fit.nrlmsise00(rows)}
    return columns


def score_storm_ai(field: str) -> int:
    split = SPLIT["storm-ai"]
    settings, p, train_sats = fold(field, "storm-ai")
    cases = manifest_cases(split)
    truth = dict(np.load(TRUTH))
    if not np.array_equal(truth["id"], cases["id"].to_numpy()):
        sys.exit(f"{TRUTH} does not hold the manifest's cases; rerun storm-ai-manifest")
    flux = pd.read_csv(
        g.DENSITY / "SW-All.csv", usecols=["DATE", "F10.7_OBS", "F10.7_OBS_LAST81"]
    )
    flux = flux.set_index(pd.to_datetime(flux.DATE)).sort_index()
    pad = pd.Timedelta(hours=split["pad_hours"])
    names = ["measured", "SDTM-1 persistence", "SDTM-1 reconstruction"]
    out: dict[str, list[np.ndarray]] = {"case": [], "tick": [], "target": []}
    for sat in sorted(cases.sat.unique()):
        mine = cases[cases.sat == sat]
        rows = satellite_rows(mine, sat, pad)
        comparators = comparator_columns(rows)
        print(f"{sat}: {len(mine)} cases, {len(rows)} rows", flush=True)
        for i, case in mine.iterrows():
            here, first = sliced(rows, case.start - pad, case.end + pad)
            if len(here) == 0:
                continue
            t_ms = (
                (here.t - fit.EPOCH_2000) // pd.Timedelta(milliseconds=1)
            ).to_numpy()
            columns = [
                np.log(here.rho.to_numpy()),
                model_ln_rho(
                    persisted(here, case.start, flux), p, train_sats, settings
                ),
                model_ln_rho(here, p, train_sats, settings),
            ] + [c[first : first + len(here)] for c in comparators.values()]
            lo, hi = truth["offset"][i], truth["offset"][i + 1]
            ticks = truth["t_ms"][lo:hi]
            means = centred_means(
                t_ms, np.exp(np.column_stack(columns)), ticks, truth["period_s"][i]
            )
            out["case"].append(np.full(ticks.size, i))
            out["tick"].append(np.arange(ticks.size))
            out["target"].append(truth["rho"][lo:hi])
            for name, m in zip(names + list(comparators), means.T, strict=True):
                out.setdefault(name, []).append(m)
    target = harness.OUT / field / "storm-ai-orbit-means.npz"
    np.savez_compressed(target, **{k: np.concatenate(v) for k, v in out.items()})
    print(f"{target}: {sum(len(v) for v in out['case'])} ticks")
    return 0


def score_aether(field: str) -> int:
    split = SPLIT["aether-p3"]
    settings, p, train_sats = fold(field, "aether-p3")
    cases = manifest_cases(split)
    rows_by_sat = {
        sat: satellite_rows(cases[cases.sat == sat], sat, pd.Timedelta(0))
        for sat in sorted(cases.sat.unique())
    }
    out: dict[str, list[float]] = {"rows": []}
    for _, case in cases.iterrows():
        here, _ = sliced(rows_by_sat[case.sat], case.start, case.end)
        measured = np.log(here.rho.to_numpy())
        columns = {"SDTM-1": model_ln_rho(here, p, train_sats, settings)}
        columns.update(comparator_columns(here))
        out["rows"].append(len(here))
        for name, ln_model in columns.items():
            out.setdefault(name, []).append(
                float(np.mean(np.abs(np.expm1(ln_model - measured))))
            )
    target = harness.OUT / field / "aether-p3-re.json"
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{target}")
    return 0


def command_score(args) -> int:
    return (score_storm_ai if args.benchmark == "storm-ai" else score_aether)(
        args.field
    )


def weighted_skill(
    ticks: np.ndarray, error: np.ndarray, reference: np.ndarray
) -> float:
    """STORM-AI's propagation score with `reference` in its MSIS's place: per tick, `1 - RMSE/RMSE_ref`
    over the cases, weighted by `exp(-ln(1/eps) tick/last)`."""
    frame = pd.DataFrame({"tick": ticks, "e": error**2, "r": reference**2})
    rmse = np.sqrt(frame.groupby("tick").mean())
    skill = 1.0 - rmse.e / rmse.r
    span = max(rmse.index.max() - rmse.index.min(), 1)
    weights = np.exp(-np.log(1.0 / EPSILON) * (rmse.index - rmse.index.min()) / span)
    return float(np.average(skill, weights=weights))


def table_storm_ai(field: str) -> int:
    z = dict(np.load(harness.OUT / field / "storm-ai-orbit-means.npz"))
    cases = manifest_cases(SPLIT["storm-ai"])
    models = [k for k in z if k not in ("case", "tick", "target", "measured")]
    every = np.isfinite(z["target"]) & np.all(
        [np.isfinite(z[k]) for k in models], axis=0
    )
    gap = z["measured"][every] / z["target"][every] - 1.0
    print(
        f"orbit-mean operator against the target: {every.sum()} ticks, median |ratio - 1| "
        f"{np.median(np.abs(gap)):.2e}, 99th percentile {np.percentile(np.abs(gap), 99):.2e}"
    )
    sat = cases.sat.to_numpy()[z["case"]]
    member = {
        s: cases.subsets.map(lambda names, s=s: s in names).to_numpy()[z["case"]]
        for s in SUBSETS.values()
    }
    groups = [("all", every)] + [(s, every & m) for s, m in member.items()]
    groups += [
        (f"private {s}", every & member["private"] & (sat == s)) for s in np.unique(sat)
    ]
    print(
        "group | cases | ticks | "
        + " | ".join(f"{m} RMSE" for m in models)
        + " | 1 - RMSE/JB2008 (persistence, reconstruction) | skill against NRLMSISE-00 (persistence)"
    )
    for label, use in groups:
        if not use.any():
            continue
        target = z["target"][use]
        rmse = {m: np.sqrt(np.mean((z[m][use] - target) ** 2)) for m in models}
        jb = rmse.get("JB2008")
        versus = (
            f"{1 - rmse['SDTM-1 persistence'] / jb:+.3f}, {1 - rmse['SDTM-1 reconstruction'] / jb:+.3f}"
            if jb
            else "—"
        )
        skill = weighted_skill(
            z["tick"][use],
            z["SDTM-1 persistence"][use] - target,
            z["NRLMSISE-00"][use] - target,
        )
        print(
            f"{label} | {len(np.unique(z['case'][use]))} | {use.sum()} | "
            + " | ".join(f"{rmse[m] * 1e13:.3f}" for m in models)
            + f" | {versus} | {skill:+.3f}"
        )
    print("RMSE in 1e-13 kg/m^3, on the ticks every model and the target answered.")
    return 0


def table_aether(field: str) -> int:
    ours = json.loads(
        (harness.OUT / field / "aether-p3-re.json").read_text(encoding="utf-8")
    )
    names = [k for k in ours if k != "rows"]
    print(
        "case | satellite | label | rows | "
        + " | ".join(names)
        + " | "
        + " | ".join(AETHER_RE)
    )
    for i, (sat, first, last, label) in enumerate(AETHER_CASES):
        print(
            f"{i + 1} | {sat} | {label} {first}..{last} | {ours['rows'][i]} | "
            + " | ".join(f"{ours[n][i]:.4f}" for n in names)
            + " | "
            + " | ".join(f"{v[i]:.4f}" for v in AETHER_RE.values())
        )
    print(
        "mean |  |  |  | "
        + " | ".join(f"{np.mean(ours[n]):.4f}" for n in names)
        + " | "
        + " | ".join(f"{np.nanmean(v):.4f}" for v in AETHER_RE.values())
    )
    return 0


def command_table(args) -> int:
    return (table_storm_ai if args.benchmark == "storm-ai" else table_aether)(
        args.field
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True)
    storm = sub.add_parser("storm-ai-manifest")
    storm.add_argument("dir")
    storm.set_defaults(func=command_storm_ai_manifest)
    sub.add_parser("aether-manifest").set_defaults(func=command_aether_manifest)
    for name, func in (("score", command_score), ("table", command_table)):
        command = sub.add_parser(name)
        command.add_argument("benchmark", choices=sorted(SPLIT))
        command.add_argument(
            "--field", default="default", choices=sorted(harness.FIELDS)
        )
        command.set_defaults(func=func)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
