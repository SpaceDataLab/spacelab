#!/usr/bin/env python3
"""Orbit prediction under SDTM-1 and NRLMSISE-00 on TU Delft's positions: the pilot of `plans/17` Ф30.H (F64).

A 20.8 % density error says nothing by itself about where a satellite is three days later. This tool
asks that question with everything but the density held fixed. Each arc is one day of positions
before an epoch and three days after it. `orbit_prediction` (`crates/spacelab/examples`) fits the state
and a drag scale to the day, freezes the scale (the persistence of the last 24 h correction, F64's
baseline), predicts, and prints the radial, along-track and cross-track errors at 6, 12, 24, 48 and
72 h. Same integrator, gravity, Sun, Moon and solar pressure for both models; `docs/verification.md`
§10.56 is the gate on the error this measures.

The positions are TU Delft's v02 density files' (CC BY 4.0, `docs/provenance.md`): GRACE-FO C at
0.001° (110 m), Swarm at 1e-8° (1 mm), every 120 s. They are a pilot, not POD of independent origin:
the Swarm and GRACE-FO orbit products themselves wait on their licence steps.

**The field is SDTM-1 frozen at 2020** (`tools/evaluate-sdtm.py export --split frozen-2020`), so every
arc, from 2020-01-01 on, is outside its fit. `run` holds the exported field to the Python fit's own values
before it starts.

**The rule, fixed before the first run.**

* Arcs: GC, SA, SB, SC, an epoch at 00:00 UTC every 14 days from 2020-01-01 while the corpus covers
  72 h after it. An arc qualifies with its first row at most 600 s after `epoch − 24 h`, at least 684
  rows (95 %) in the fit window, no gap above 600 s in it or before the epoch, and a row within 60 s of
  every horizon.
* Modes: `forecast` reads only what is known at the epoch — the indices held at their state one
  second before it, and the trailing 81-day flux in the fit and the prediction alike, for both models;
  `observed` reads the measured indices, centred flux included, and is shown apart.
* Exclusion, symmetric: an arc leaves every table when any of its four fits fails, or has an RMS above
  five times its satellite's median fit RMS (a manoeuvre or an unmodelled event in the window).
* Primary: forecast mode, 72 h, along-track. Per arc `d = ln(|s_SDTM| / |s_MSIS|)`; its median over arcs,
  with a 95 % interval from 4000 bootstrap draws of epochs (arcs at one epoch share the drivers, so they
  are drawn together). SDTM-1 is ahead when the interval lies below zero, behind when above, and no
  difference is shown otherwise.
* Everything else is description: other horizons, the observed mode, the total error, each satellite,
  low and high flux, and the drag scale's spread.
* The calibrated mode, shown apart (`run --calibrated`): state and scale fitted to the whole 96 h arc
  under the measured indices. What is left is how each model's density varies over four days, its
  level calibrated out. A 96 h fit costs about six times the four 24 h fits, so the mode runs on every
  fourth epoch (56 days apart): of those, the arcs kept above whose calibrated fits both succeed and stay
  within five times the satellite's median calibrated RMS. Per arc `ln(RMS_SDTM / RMS_MSIS)`; its median and
  interval as above.

    tools/orbit-pilot.py arcs                  # writes the arcs and jobs.tsv, prints the counts
    tools/orbit-pilot.py run [--processes N]   # runs the jobs not yet in results.tsv
    tools/orbit-pilot.py run --calibrated      # the same into calibrated.tsv
    tools/orbit-pilot.py score                 # prints the tables

Run with `uv run --project tools --locked python -I`, after
`cargo build -p spacelab --release --example orbit_prediction`.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import csv
import datetime as dt
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "validation" / "tudelft-v02" / "corpus"
FEED = ROOT / "data" / "validation" / "density" / "SW-All.csv"
FIELD = ROOT / "data" / "validation" / "sdtm-eval" / "default" / "frozen-2020.field"
FIXTURE = FIELD.with_suffix(".fixture.csv")
WORKER = ROOT / "target" / "release" / "examples" / "orbit_prediction"
OUT = ROOT / "data" / "validation" / "orbit-pilot"
JOBS = OUT / "jobs.tsv"
RESULTS = OUT / "results.tsv"
CALIBRATED = OUT / "calibrated.tsv"
CALIBRATED_EVERY = 4

SATELLITES = ("GC", "SA", "SB", "SC")
FIRST_EPOCH = dt.date(2020, 1, 1)
STEP_DAYS = 14
FIT_SPAN = 86_400.0
HORIZONS = (6, 12, 24, 48, 72)
MIN_ROWS = 684
MAX_GAP = 600.0
TARGET_SLACK = 60.0
OUTLIER = 5.0
MODELS = ("sdtm", "msis")
MODES = ("forecast", "observed")
DRAWS = 4000
SEED = 20261112
# Corpus time: UTC milliseconds since 2000-01-01T00:00:00, leap seconds not counted.
CORPUS_UNIX = 946_684_800.0


def unix(day: dt.date) -> float:
    return float(calendar.timegm(day.timetuple()))


def qualifies(t: np.ndarray, epoch: float) -> str | None:
    """Why an arc does not qualify, or `None`."""
    window = t[(t >= epoch - FIT_SPAN) & (t <= epoch)]
    if len(window) == 0 or window[0] - (epoch - FIT_SPAN) > MAX_GAP:
        return "fit window starts late"
    if len(window) < MIN_ROWS:
        return "fewer than 684 rows in the fit window"
    if np.diff(window).max() > MAX_GAP or epoch - window[-1] > MAX_GAP:
        return "a gap in the fit window"
    for hours in HORIZONS:
        if np.abs(t - (epoch + hours * 3_600.0)).min() > TARGET_SLACK:
            return "a horizon without a row"
    return None


def arcs(_args) -> int:
    (OUT / "arcs").mkdir(parents=True, exist_ok=True)
    jobs, reasons = [], {}
    for sat in SATELLITES:
        corpus = np.load(CORPUS / f"corpus_{sat}.npz")
        t = corpus["t_ms"] / 1000.0 + CORPUS_UNIX
        columns = [corpus[k] for k in ("latitude", "longitude", "altitude_m")]
        day = FIRST_EPOCH
        while (epoch := unix(day)) + HORIZONS[-1] * 3_600.0 + TARGET_SLACK <= t[-1]:
            reason = qualifies(t, epoch)
            if reason is None:
                keep = (t >= epoch - FIT_SPAN) & (
                    t <= epoch + HORIZONS[-1] * 3_600.0 + TARGET_SLACK
                )
                path = OUT / "arcs" / f"{sat}-{day.isoformat()}.txt"
                with path.open("w") as out:
                    for row in zip(t[keep], *(c[keep] for c in columns), strict=True):
                        out.write(" ".join(repr(float(v)) for v in row) + "\n")
                jobs.append(
                    (
                        f"{sat}-{day.isoformat()}",
                        str(path.relative_to(ROOT)),
                        f"{epoch:.0f}",
                    )
                )
            else:
                reasons[(sat, reason)] = reasons.get((sat, reason), 0) + 1
            day += dt.timedelta(days=STEP_DAYS)
    with JOBS.open("w") as out:
        out.writelines("\t".join(job) + "\n" for job in jobs)
    for sat in SATELLITES:
        print(f"{sat}: {sum(j[0].startswith(sat) for j in jobs)} arcs")
        for (s, reason), n in sorted(reasons.items()):
            if s == sat:
                print(f"  left out: {reason}: {n}")
    print(f"{len(jobs)} arcs in {JOBS.relative_to(ROOT)}")
    return 0


def check_field() -> None:
    done = subprocess.run(
        [WORKER, "--check-field", FIELD, FIXTURE],
        capture_output=True,
        text=True,
        check=False,
    )
    print(done.stdout.strip() or done.stderr.strip())
    if done.returncode != 0:
        sys.exit("the exported field does not reproduce the Python fit")


def run_one(job: str, calibrated: bool) -> str:
    done = subprocess.run(
        [WORKER, *(["--calibrated"] if calibrated else []), FEED, FIELD],
        input=job.replace(" ", "\t") + "\n",
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if done.returncode != 0:
        name = job.split("\t")[0]
        return f"{name}\terror\t{done.stderr.strip().splitlines()[-1] if done.stderr else ''}\n"
    return done.stdout


def in_calibrated_subset(arc: str) -> bool:
    days = (dt.date.fromisoformat(arc[3:]) - FIRST_EPOCH).days
    return (days // STEP_DAYS) % CALIBRATED_EVERY == 0


def run(args) -> int:
    check_field()
    results = CALIBRATED if args.calibrated else RESULTS
    finished = set()
    if results.exists():
        finished = {line.split("\t")[0] for line in results.read_text().splitlines()}
    jobs = [
        line
        for line in JOBS.read_text().splitlines()
        if line.split("\t")[0] not in finished
        and (not args.calibrated or in_calibrated_subset(line.split("\t")[0]))
    ]
    print(f"{len(jobs)} jobs to run, {len(finished)} done", flush=True)
    with (
        results.open("a") as out,
        concurrent.futures.ThreadPoolExecutor(args.processes) as pool,
    ):
        runs = pool.map(run_one, jobs, [args.calibrated] * len(jobs))
        for k, text in enumerate(runs, 1):
            out.write(text)
            out.flush()
            if k % 20 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    return 0


def read_results() -> tuple[dict, dict]:
    """`errors[(arc, model, mode, horizon)] = (r, s, w)` and `fits[(arc, model, mode)] = (scale, rms)`."""
    errors, fits, failed = {}, {}, set()
    for line in RESULTS.read_text().splitlines():
        cells = line.split("\t")
        if len(cells) < 10:
            failed.add(cells[0])
            continue
        arc, model, mode, hours = cells[0], cells[1], cells[2], int(float(cells[3]))
        r, s, w, scale, rms = (float(c) for c in cells[4:9])
        errors[arc, model, mode, hours] = (r, s, w)
        fits[arc, model, mode] = (scale, rms)
    arcs = {key[0] for key in fits} | failed
    complete = {
        arc
        for arc in arcs - failed
        if all(
            (arc, m, o, h) in errors for m in MODELS for o in MODES for h in HORIZONS
        )
    }
    return {k: v for k, v in errors.items() if k[0] in complete}, {
        k: v for k, v in fits.items() if k[0] in complete
    }


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def clustered(values: dict[str, float], stat=statistics.median) -> str:
    """`stat` over arcs and its 95 % interval, epochs drawn with their arcs."""
    by_epoch: dict[str, list[float]] = {}
    for arc, value in values.items():
        by_epoch.setdefault(arc[3:], []).append(value)
    epochs = sorted(by_epoch)
    rng = random.Random(SEED)
    spread = []
    for _ in range(DRAWS):
        sample = [
            v for _ in epochs for v in by_epoch[epochs[rng.randrange(len(epochs))]]
        ]
        spread.append(stat(sample))
    low, high = quantile(spread, 0.025), quantile(spread, 0.975)
    return f"{stat(list(values.values())):+.3f} [{low:+.3f}, {high:+.3f}]"


def verdict(text: str) -> str:
    low, high = (float(v) for v in text.split("[")[1].rstrip("]").split(","))
    return (
        "SDTM-1 ahead"
        if high < 0
        else "SDTM-1 behind"
        if low > 0
        else "no difference shown"
    )


def flux_at_epochs() -> dict[str, float]:
    """The trailing 81-day F10.7 of the day before each epoch, by ISO date of the epoch."""
    flux = {}
    with FEED.open() as feed:
        for row in csv.DictReader(feed):
            if row.get("F10.7_OBS_LAST81"):
                day = dt.date.fromisoformat(row["DATE"]) + dt.timedelta(days=1)
                flux[day.isoformat()] = float(row["F10.7_OBS_LAST81"])
    return flux


def table(errors, arcs, mode, hours, component) -> tuple[dict[str, float], str]:
    def size(arc, model):
        r, s, w = errors[arc, model, mode, hours]
        return abs(s) if component == "along" else math.sqrt(r * r + s * s + w * w)

    d = {arc: math.log(size(arc, "sdtm") / size(arc, "msis")) for arc in arcs}
    medians = [statistics.median(size(a, m) for a in arcs) for m in MODELS]
    better = sum(v < 0 for v in d.values()) / len(d)
    line = (
        f"{mode} | {hours} | {component} | {len(arcs)} | {medians[0]:.0f} | {medians[1]:.0f} "
        f"| {clustered(d)} | {better:.2f}"
    )
    return d, line


def score(_args) -> int:
    errors, fits = read_results()
    arcs = sorted({k[0] for k in fits})
    per_sat = {
        sat: statistics.median(v[1] for k, v in fits.items() if k[0].startswith(sat))
        for sat in SATELLITES
        if any(k[0].startswith(sat) for k in fits)
    }
    kept = [
        arc
        for arc in arcs
        if all(
            fits[arc, m, o][1] <= OUTLIER * per_sat[arc[:2]]
            for m in MODELS
            for o in MODES
        )
    ]
    total = len({line.split("\t")[0] for line in RESULTS.read_text().splitlines()})
    print(f"{total} arcs run, {len(arcs)} with all four fits, {len(kept)} kept")
    for sat, rms in per_sat.items():
        n = sum(a.startswith(sat) for a in kept)
        print(
            f"  {sat}: {n} kept, median fit RMS {rms:.1f} m, limit {OUTLIER * rms:.1f} m"
        )
    print()
    head = (
        "mode | h | error | arcs | median SDTM-1, m | median MSIS, m "
        "| median ln(SDTM/MSIS) [95 %] | share SDTM-1 smaller"
    )
    print(head)
    for mode in MODES:
        for component in ("along", "total"):
            for hours in HORIZONS:
                print(table(errors, kept, mode, hours, component)[1])
    _, primary = table(errors, kept, "forecast", 72, "along")
    print(
        f"\nprimary (forecast, 72 h, along-track): {primary.split('|')[6].strip()} -> ",
        end="",
    )
    print(verdict(primary.split("|")[6].strip()))

    print("\nforecast, along-track, by satellite\n" + head)
    for sat in per_sat:
        group = [a for a in kept if a.startswith(sat)]
        if len(group) >= 10:
            for hours in (24, 72):
                print(f"{sat} " + table(errors, group, "forecast", hours, "along")[1])

    flux = flux_at_epochs()
    split = statistics.median(flux[a[3:]] for a in kept)
    print(
        f"\nforecast, along-track, by trailing F10.7a at the epoch (split {split:.1f})\n"
        + head
    )
    for name, group in (
        ("low", [a for a in kept if flux[a[3:]] < split]),
        ("high", [a for a in kept if flux[a[3:]] >= split]),
    ):
        for hours in (24, 72):
            print(f"{name} " + table(errors, group, "forecast", hours, "along")[1])

    # Where drag is weakest, the scale also takes up the along-track part of every unmodelled force
    # and can fall to zero or below; the logarithm is taken of the positive ones, the rest counted.
    print(
        "\nln(drag scale), forecast-mode fit, by satellite: median and spread (MAD) of the positive ones"
    )
    print(
        "satellite | arcs | scale <= 0 | SDTM-1 median | SDTM-1 MAD | MSIS median | MSIS MAD"
    )
    for sat in per_sat:
        group = [a for a in kept if a.startswith(sat)]
        positive = [
            a for a in group if all(fits[a, m, "forecast"][0] > 0 for m in MODELS)
        ]
        cells = []
        for model in MODELS:
            values = [math.log(fits[a, model, "forecast"][0]) for a in positive]
            centre = statistics.median(values)
            cells += [centre, statistics.median(abs(v - centre) for v in values)]
        print(
            f"{sat} | {len(group)} | {len(group) - len(positive)} | "
            + " | ".join(f"{c:+.3f}" for c in cells)
        )
    positive = [
        a for a in kept if all(fits[a, m, o][0] > 0 for m in MODELS for o in MODES)
    ]
    _, line = table(errors, positive, "forecast", 72, "along")
    print(
        "\nsensitivity, not part of the rule: the primary on the arcs where all four scales are"
        f" positive\n{line}"
    )
    if CALIBRATED.exists():
        score_calibrated(kept)
    return 0


def score_calibrated(kept: list[str]) -> None:
    """The calibrated mode's table, on the arcs `score` kept."""
    fits, errors = {}, {}
    for line in CALIBRATED.read_text().splitlines():
        cells = line.split("\t")
        if len(cells) >= 10:
            r, s, w, _scale, rms = (float(c) for c in cells[4:9])
            fits[cells[0], cells[1]] = rms
            errors[cells[0], cells[1], int(float(cells[3]))] = math.sqrt(
                r * r + s * s + w * w
            )
    subset = [a for a in kept if in_calibrated_subset(a)]
    both = [a for a in subset if all((a, m) in fits for m in MODELS)]
    limit = {
        sat: OUTLIER
        * statistics.median(
            fits[a, m] for a in both if a.startswith(sat) for m in MODELS
        )
        for sat in SATELLITES
        if any(a.startswith(sat) for a in both)
    }
    arcs = [a for a in both if all(fits[a, m] <= limit[a[:2]] for m in MODELS)]
    if not arcs:
        print("\ncalibrated: no arc yet")
        return
    print(
        f"\ncalibrated: {len(arcs)} of the {len(subset)} kept arcs at every fourth epoch (both fits, within 5x the median RMS)"
    )
    print(
        "group | arcs | median RMS SDTM-1, m | median RMS MSIS, m | median ln(SDTM/MSIS) [95 %] | share SDTM-1 smaller"
    )
    groups = [("all", arcs)] + [
        (sat, [a for a in arcs if a.startswith(sat)]) for sat in limit
    ]
    for name, group in groups:
        if len(group) < 10:
            continue
        d = {a: math.log(fits[a, "sdtm"] / fits[a, "msis"]) for a in group}
        medians = [statistics.median(fits[a, m] for a in group) for m in MODELS]
        share = sum(v < 0 for v in d.values()) / len(d)
        print(
            f"{name} | {len(group)} | {medians[0]:.1f} | {medians[1]:.1f} | {clustered(d)} | {share:.2f}"
        )
    print(
        "total residual at the horizon, median m: "
        + ", ".join(
            f"{h} h {statistics.median(errors[a, 'sdtm', h] for a in arcs):.0f} / "
            f"{statistics.median(errors[a, 'msis', h] for a in arcs):.0f}"
            for h in HORIZONS
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("arcs").set_defaults(func=arcs)
    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    run_parser.add_argument("--calibrated", action="store_true")
    run_parser.set_defaults(func=run)
    sub.add_parser("score").set_defaults(func=score)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
