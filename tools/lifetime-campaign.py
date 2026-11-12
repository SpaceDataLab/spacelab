#!/usr/bin/env python3
"""Score lifetime predictions on real re-entries: `docs/verification.md` §10.37, Ф23.F.

Each object is an unpowered, near-circular body with a decay date in GCAT and element sets up to it.
From each one and each lead `H` it takes the state `H` days before decay, fits a ballistic coefficient
to the decay already seen, and predicts the rest under the measured indices, under persistence, and
under an ensemble of past solar cycles at the same phase. The ensemble's median and a conformal window
around it are scored on the true remaining time.

The sample is conditioned on a known decay date, and on element sets that reach it. That is the only
way to know the truth, and it means the coverage below is of objects that did come down.

    tools/lifetime-campaign.py select   # writes data/validation/reentry/jobs.tsv, prints the counts
    tools/lifetime-campaign.py converge # the tolerance against one a hundred times tighter
    tools/lifetime-campaign.py run [--max-lead DAYS]   # runs the jobs not yet in results.tsv
    tools/lifetime-campaign.py score    # prints the tables
    tools/lifetime-campaign.py cases    # the three named cases
    tools/lifetime-campaign.py spheres  # Ф23.T: spheres of known area-to-mass, no fitted coefficient
    tools/lifetime-campaign.py halves   # Ф23.S (1): B above and below 225 km on the calibration decays

Standard library only. Raw element sets stay under `data/`.
"""

from __future__ import annotations

import argparse
import bisect
import calendar
import concurrent.futures
import csv
import datetime as dt
import itertools
import math
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "validation" / "reentry"
FEED = ROOT / "data" / "validation" / "density" / "SW-All.csv"
WORKER = ROOT / "target" / "release" / "examples" / "lifetime_campaign"
CPF_STATE = ROOT / "target" / "release" / "examples" / "cpf_mean_state"

DAY = 86_400.0
LEADS_DAYS = (7, 30, 90, 365, 1095)
MODELS = ("sdtm", "msis")
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
UNPOWERED = re.compile(r"^(D|R\d|C)\b")

MAX_ECCENTRICITY = 0.02
MAX_LAST_GAP_DAYS = 5.0
MAX_STATE_GAP_DAYS = 5.0
MAX_UPWARD_JUMP_M = 3_000.0
MIN_DROP_M = 1_000.0
MIN_FIT_DAYS = 30.0
MIN_PERIGEE_M = 180_000.0
HORIZON_FACTOR = 6.0
# One launch's fragments share an orbit and a Sun, so they are one draw and not many.
# A hundred draws a lead, spread evenly in time, is 60 to calibrate and 40 to test.
# Fixed before any long lead was run: a job at 1095 days is about 150 of those at 7 days.
OBJECTS_PER_LEAD = {7: 100, 30: 100, 90: 100, 365: 60, 1095: 40}
# NRLMSISE-00 is the slower model by an order of magnitude. Beyond this lead it runs the measured
# indices and persistence, which is the comparison, and not the ensemble.
MSIS_ENSEMBLE_MAX_LEAD_DAYS = 30
# Declared before any error was seen: the earlier 60 % of launches calibrate, the later 40 % test.
CALIBRATION_FRACTION = 0.6
ALPHA = 0.1
# Everything scored starts here. The analog blend's `tau` is chosen on flux before it, and only there.
TEST_ERA = dt.datetime(1990, 1, 1, tzinfo=dt.UTC).timestamp()
PHASE_OFFSETS_DAYS = (-183, 0, 183)
# SDTM-1's fitted envelope starts here; below it the field is the Bates profile extrapolated (Ф23.S).
SPLIT_PERIGEE_M = 225_000.0
TAU_GRID_DAYS = (30, 60, 90, 180, 365, 730, 1460)

# WGS-72, the constants a TLE's mean motion is defined with (Hoots & Roehrich 1980).
WGS72_RE_M = 6_378_135.0
WGS72_MU_KM3_S2 = 398_600.8
WGS72_J2 = 0.001082616


def mean_motion_to_semi_major_axis(n_rev_day: float, e: float, i: float) -> float:
    """A TLE's Kozai mean motion to Brouwer's mean semi-major axis, m: Spacetrack Report 3's recovery."""
    xke = 60.0 / math.sqrt(6378.135**3 / WGS72_MU_KM3_S2)
    k2 = 0.5 * WGS72_J2
    n0 = n_rev_day * 2.0 * math.pi / 1440.0
    a1 = (xke / n0) ** (2.0 / 3.0)
    shape = 1.5 * k2 * (3.0 * math.cos(i) ** 2 - 1.0) / (1.0 - e * e) ** 1.5
    d1 = shape / (a1 * a1)
    a0 = a1 * (1.0 - d1 / 3.0 - d1 * d1 - 134.0 / 81.0 * d1**3)
    d0 = shape / (a0 * a0)
    return a0 / (1.0 - d0) * WGS72_RE_M


def tle_epoch(field: str) -> float:
    year = int(field[:2])
    year += 2000 if year < 57 else 1900
    start = dt.datetime(year, 1, 1, tzinfo=dt.UTC).timestamp()
    return start + (float(field[2:]) - 1.0) * DAY


def read_elements(text: str) -> list[tuple[float, list[float]]]:
    """Element sets by epoch, oldest first, as `(unix, [a m, e, i, raan, argp, M])`."""
    sets: dict[float, list[float]] = {}
    first = None
    for line in text.splitlines():
        if line.startswith("1 "):
            first = line
        elif line.startswith("2 ") and first is not None:
            try:
                epoch = tle_epoch(first[18:32].strip())
                i = math.radians(float(line[8:16]))
                raan = math.radians(float(line[17:25]))
                e = float("0." + line[26:33].strip())
                argp = math.radians(float(line[34:42]))
                mean_anomaly = math.radians(float(line[43:51]))
                n = float(line[52:63])
            except ValueError:
                first = None
                continue
            first = None
            if n > 0.0 and e < 1.0:
                a = mean_motion_to_semi_major_axis(n, e, i)
                sets[epoch] = [a, e, i, raan, argp, mean_anomaly]
    return sorted(sets.items())


def decay_time(text: str) -> float | None:
    """GCAT's decay date as Unix seconds; a date with no time is taken at noon."""
    match = re.match(r"(\d{4}) (\w{3})\s+(\d+)(?:\s+(\d{2})(\d{2}))?", text)
    if not match:
        return None
    month = MONTHS.index(match[2]) + 1
    if not 1 <= int(match[3]) <= calendar.monthrange(int(match[1]), month)[1]:
        return None
    hour = int(match[4]) if match[4] else 12
    minute = int(match[5]) if match[5] else 0
    moment = dt.datetime(
        int(match[1]), month, int(match[3]), hour, minute, tzinfo=dt.UTC
    )
    return moment.timestamp()


class Feed:
    """The daily 81-day trailing flux, and the solar-cycle minima of its 13-month mean."""

    def __init__(self, path: pathlib.Path) -> None:
        self.days: list[float] = []
        self.flux: list[float] = []
        self.f81: list[float] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                f, f81 = row["F10.7_OBS"].strip(), row["F10.7_OBS_LAST81"].strip()
                if not row["AP_AVG"].strip() or not f or not f81:
                    continue
                date = dt.datetime.fromisoformat(row["DATE"]).replace(tzinfo=dt.UTC)
                self.days.append(date.timestamp())
                self.flux.append(float(f))
                self.f81.append(float(f81))
        self.months: list[float] = []
        self.smooth: list[float] = []
        step = 30
        half = 6 * step
        for k in range(half, len(self.flux) - half, step):
            self.months.append(self.days[k])
            self.smooth.append(statistics.fmean(self.flux[k - half : k + half + 1]))

    def f81_at(self, t: float) -> float:
        k = bisect.bisect_right(self.days, t) - 1
        return self.f81[max(k, 0)]

    def minima_known_at(self, t0: float) -> list[float]:
        """Minima of the 13-month mean a forecaster at `t0` can already name, oldest first.

        The centred mean reaches six months before `t0`. A month counts when it is the lowest of the
        known means within four years either side and lies at least six months before their end, so at
        least twelve months before `t0`.
        """
        end = bisect.bisect_right(self.months, t0 - 182.5 * DAY)
        found: list[float] = []
        for k in range(end - 6):
            lo, hi = max(k - 48, 0), min(k + 49, end)
            if self.smooth[k] == min(self.smooth[lo:hi]) and hi - lo > 24:
                found.append(self.months[k])
        return found

    def analog_starts(self, t0: float, horizon_days: float) -> list[float]:
        """The shifts, days, of past cycles at the phase of `t0`, each wholly known before `t0`."""
        minima = self.minima_known_at(t0)
        if len(minima) < 2:
            return []
        phase = t0 - minima[-1]
        shifts = []
        for past in minima[:-1]:
            for offset in PHASE_OFFSETS_DAYS:
                start = past + phase + offset * DAY
                shift = round((t0 - start) / DAY)
                if shift >= horizon_days and start >= self.days[0] + 366.0 * DAY:
                    shifts.append(shift)
        return shifts


def blend_tau_days(feed: Feed) -> tuple[int, dict[int, float]]:
    """The analog blend's `tau`, chosen by flux error on dates before the test era alone."""
    errors: dict[int, list[float]] = {tau: [] for tau in TAU_GRID_DAYS}
    leads = range(30, 1096, 30)
    t0 = dt.datetime(1970, 1, 1, tzinfo=dt.UTC).timestamp()
    while t0 < TEST_ERA:
        ratio_at = feed.f81_at(t0)
        for shift in feed.analog_starts(t0, 0.0):
            ratio = ratio_at / feed.f81_at(t0 - shift * DAY)
            for lead in leads:
                t = t0 + lead * DAY
                if t >= TEST_ERA:
                    break
                truth = feed.f81_at(t)
                past = feed.f81_at(t - shift * DAY)
                for tau in TAU_GRID_DAYS:
                    factor = 1.0 + (ratio - 1.0) * math.exp(-lead / tau)
                    errors[tau].append(abs(math.log(truth / (past * factor))))
        t0 += 30.0 * DAY
    means = {tau: statistics.fmean(v) for tau, v in errors.items() if v}
    return min(means, key=lambda tau: means[tau]), means


def state_before(sets, t: float):
    """The set nearest before `t` within the allowed gap."""
    k = bisect.bisect_right([s[0] for s in sets], t) - 1
    if k < 0 or t - sets[k][0] > MAX_STATE_GAP_DAYS * DAY:
        return None
    return k


def job_for(sets, decay: float, lead: int, feed: Feed, tau_days: int):
    """One object at one lead, or the reason it is left out."""
    k0 = state_before(sets, decay - lead * DAY)
    if k0 is None:
        return "no state at the lead"
    t0, state = sets[k0]
    if t0 < TEST_ERA:
        return "before the test era"
    if state[0] * (1.0 - state[1]) - 6_378_137.0 < MIN_PERIGEE_M:
        return "perigee below 180 km at the state"
    window = max(MIN_FIT_DAYS, lead / 2.0)
    kh = state_before(sets, t0 - window * DAY)
    if kh is None:
        return "no state to fit from"
    th, start = sets[kh]
    if start[0] - state[0] < MIN_DROP_M:
        return "no real drop to fit"
    span = [s[1][0] for s in sets[kh:] if s[0] <= decay]
    if any(b - a > MAX_UPWARD_JUMP_M for a, b in itertools.pairwise(span)):
        return "an upward jump in a"
    horizon_days = HORIZON_FACTOR * lead
    shifts = feed.analog_starts(t0, horizon_days)
    if not shifts:
        return "no analog"
    ratio_at = feed.f81_at(t0)
    members = ["O", "P"]
    members += [
        f"A:{shift}:{ratio_at / feed.f81_at(t0 - shift * DAY):.17g}:{tau_days * DAY:.1f}"
        for shift in shifts
    ]
    return {
        "lead": lead,
        "th": th,
        "start": start,
        "t0": t0,
        "state": state,
        "horizon": horizon_days * DAY,
        "members": ";".join(members),
    }


def read_catalogue() -> dict[int, dict[str, str]]:
    header: list[str] = []
    rows = {}
    text = (OUT / "satcat.tsv").read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("#JCAT"):
            header = [h.strip() for h in line[1:].split("\t")]
        if line.startswith("#") or not line.strip():
            continue
        row = dict(
            zip(header, (cell.strip() for cell in line.split("\t")), strict=False)
        )
        if row.get("Satcat", "").isdigit():
            rows[int(row["Satcat"])] = row
    return rows


def select(_args) -> int:
    feed = Feed(FEED)
    tau_days, means = blend_tau_days(feed)
    print("tau, days -> mean |ln(F81 / blended analog)| on flux before the test era:")
    for tau, mean in means.items():
        print(f"  {tau:5d} {mean:.4f}{'  chosen' if tau == tau_days else ''}")
    catalogue = read_catalogue()
    reasons: dict[str, int] = {}
    by_lead: dict[int, dict[str, tuple[int, dict]]] = {lead: {} for lead in LEADS_DAYS}
    for path in sorted((OUT / "elements").iterdir()):
        number = int(path.name[1:])
        row = catalogue.get(number)
        if row is None or path.stat().st_size == 0:
            reasons["no elements"] = reasons.get("no elements", 0) + 1
            continue
        decay = decay_time(row["DDate"])
        sets = read_elements(path.read_text(encoding="utf-8", errors="replace"))
        sets = [s for s in sets if decay is not None and s[0] <= decay]
        if decay is None or not sets or not UNPOWERED.match(row["Type"]):
            reasons["no elements"] = reasons.get("no elements", 0) + 1
            continue
        if decay - sets[-1][0] > MAX_LAST_GAP_DAYS * DAY:
            reasons["last set too old"] = reasons.get("last set too old", 0) + 1
            continue
        if sets[-1][1][1] > MAX_ECCENTRICITY:
            reasons["not near-circular"] = reasons.get("not near-circular", 0) + 1
            continue
        for lead in LEADS_DAYS:
            if row["Launch_Tag"] in by_lead[lead]:
                reasons["launch has its object"] = (
                    reasons.get("launch has its object", 0) + 1
                )
                continue
            job = job_for(sets, decay, lead, feed, tau_days)
            if isinstance(job, str):
                reasons[job] = reasons.get(job, 0) + 1
                continue
            by_lead[lead][row["Launch_Tag"]] = (number, job)
    jobs = 0
    with (OUT / "jobs.tsv").open("w") as out:
        for lead, launches in by_lead.items():
            ordered = sorted(launches.values(), key=lambda entry: entry[1]["t0"])
            step = max(len(ordered) - 1, 1) / max(OBJECTS_PER_LEAD[lead] - 1, 1)
            count = min(OBJECTS_PER_LEAD[lead], len(ordered))
            kept = [ordered[round(k * step)] for k in range(count)]
            print(f"lead {lead} d: {len(ordered)} launches, {len(kept)} kept")
            for number, job in kept:
                for model in MODELS:
                    members = job["members"]
                    if model == "msis" and lead > MSIS_ENSEMBLE_MAX_LEAD_DAYS:
                        members = "O;P"
                    cells = [
                        f"S{number:05d}|{lead}|{model}",
                        model,
                        f"{job['th']:.3f}",
                        *(f"{x:.17g}" for x in job["start"]),
                        f"{job['t0']:.3f}",
                        *(f"{x:.17g}" for x in job["state"]),
                        f"{job['horizon']:.1f}",
                        members,
                    ]
                    out.write("\t".join(cells) + "\n")
                    jobs += 1
    print(f"{jobs} jobs")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  left out: {reason}: {count}")
    return 0


def read_done() -> set[str]:
    done_path = OUT / "results.tsv"
    if not done_path.exists():
        return set()
    with done_path.open() as handle:
        return {line.split("\t", 1)[0] for line in handle if line.strip()}


def run_chunk(jobs: list[str], *extra: str) -> str:
    done = subprocess.run(
        [str(WORKER), str(FEED), *extra],
        input="\n".join(jobs) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def converge(args) -> int:
    """The worst change in a lifetime when the propagation's tolerance tightens from 1e-9 to 1e-11."""
    jobs = []
    for line in (OUT / "jobs.tsv").read_text().splitlines():
        cells = line.split("\t")
        if cells[0].split("|")[1] == str(args.lead):
            cells[17] = "O;P"
            jobs.append("\t".join(cells))
    jobs = jobs[: 2 * args.objects]

    def seconds(text: str) -> dict[tuple[str, str], float]:
        found = {}
        for line in text.splitlines():
            key, index, _b, value = line.split("\t")
            if index != "fit" and value not in ("", "nan"):
                found[key, index] = float(value)
        return found

    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        loose = seconds("".join(pool.map(run_chunk, ([j] for j in jobs))))
        tight = seconds(
            "".join(pool.map(run_chunk, ([j] for j in jobs), ["1e-11"] * len(jobs)))
        )
    worst = max(abs(loose[k] / tight[k] - 1.0) for k in loose if k in tight)
    print(
        f"{len(loose)} lifetimes at lead {args.lead} d; worst relative change {worst:.3g}"
    )
    return 0


def run(args) -> int:
    done = read_done()
    jobs = (OUT / "jobs.tsv").read_text().splitlines()

    def lead_of(line: str) -> int:
        return int(line.split("\t", 1)[0].split("|")[1])

    todo = [
        line
        for line in jobs
        if line.split("\t", 1)[0] not in done and lead_of(line) <= args.max_lead
    ]
    # The cheap leads first, so an interrupted run still has whole classes.
    todo.sort(key=lead_of)
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    with (
        (OUT / "results.tsv").open("a") as out,
        concurrent.futures.ProcessPoolExecutor(args.processes) as pool,
    ):
        # As they finish, not in order: one slow object must not hold back the rows of the others,
        # and a run stopped at its time limit keeps every job that had finished.
        futures = [pool.submit(run_chunk, [job]) for job in todo]
        for k, future in enumerate(concurrent.futures.as_completed(futures)):
            out.write(future.result())
            out.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1} done", flush=True)
    return 0


def quantile(sorted_values: list[float], q: float) -> float:
    """The nearest-rank quantile; a censored member sorts as +inf."""
    return sorted_values[max(math.ceil(q * len(sorted_values)) - 1, 0)]


def conformal_radius(scores: list[float]) -> float:
    done = subprocess.run(
        [str(WORKER), "--conformal", str(ALPHA)],
        input="\n".join(f"{s:.17g}" for s in scores) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return float(done.stdout)


def beta_binomial_band(m: int, n: int, alpha: float) -> tuple[int, int]:
    """The 5–95 % range of the hits among `m` test points given `n` calibration points."""
    k = math.ceil((n + 1) * (1.0 - alpha))
    a, b = k, n + 1 - k

    def log_beta(x: float, y: float) -> float:
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    total = 0.0
    lo = None
    for x in range(m + 1):
        total += math.exp(
            math.lgamma(m + 1)
            - math.lgamma(x + 1)
            - math.lgamma(m - x + 1)
            + log_beta(x + a, m - x + b)
            - log_beta(a, b)
        )
        if lo is None and total >= 0.05:
            lo = x
        if total >= 0.95:
            return lo if lo is not None else x, x
    return lo if lo is not None else m, m


def score(_args) -> int:
    catalogue = read_catalogue()
    members: dict[str, dict[int, tuple[float, float | None]]] = {}
    with (OUT / "results.tsv").open() as handle:
        for line in handle:
            key, index, b, seconds = line.rstrip("\n").split("\t")
            if index == "fit":
                members.setdefault(key, {})
                continue
            t = None if seconds == "" else float(seconds)
            members.setdefault(key, {})[int(index)] = (float(b), t)
    jobs = {}
    for line in (OUT / "jobs.tsv").read_text().splitlines():
        cells = line.split("\t")
        jobs[cells[0]] = (float(cells[9]), float(cells[16]))
    # A rerun of `select` on more objects can drop one a launch cap now fills with earlier numbers.
    members = {key: runs for key, runs in members.items() if key in jobs}
    rows: dict[tuple[int, str], list[dict]] = {}
    broke = 0
    for key, runs in members.items():
        name, lead, model = key.split("|")
        row = catalogue[int(name[1:])]
        decay = decay_time(row["DDate"])
        t0, horizon = jobs[key]
        assert decay is not None, (
            key
        )  # `select` made no job for an object without a decay date
        if not runs or any(t is not None and math.isnan(t) for _, t in runs.values()):
            broke += 1
            continue
        truth = decay - t0
        ensemble = sorted(
            math.inf if t is None else t for i, (_, t) in runs.items() if i >= 2
        )
        rows.setdefault((int(lead), model), []).append(
            {
                "launch": row["Launch_Tag"],
                "t0": t0,
                "truth": truth,
                "observed": math.inf if runs[0][1] is None else runs[0][1],
                "persistence": math.inf if runs[1][1] is None else runs[1][1],
                "median": min(quantile(ensemble, 0.5), horizon) if ensemble else None,
                "lo": quantile(ensemble, 0.05) if ensemble else None,
                "hi": quantile(ensemble, 0.95) if ensemble else None,
                "horizon": horizon,
                "b": runs[0][0],
            }
        )
    print(f"{broke} jobs broke down or had no fit\n")
    header = (
        "lead d | model | n cal | n test | launches | median |ln| O | median ln(T_O/T) | P | ens | "
        "5-95 % coverage | radius | conformal hits/m | 5-95 % band"
    )
    print(header)
    with (OUT / "scores.tsv").open("w") as out:
        out.write(
            "lead\tmodel\tobject_launch\tsplit\tln_observed\tln_persistence\tln_median\tcovered_raw\tcovered_conformal\n"
        )
        for (lead, model), group in sorted(rows.items()):
            first = {}
            for r in group:
                first[r["launch"]] = min(first.get(r["launch"], math.inf), r["t0"])
            ordered = sorted(first, key=lambda launch: first[launch])
            cut = round(CALIBRATION_FRACTION * len(ordered))
            early = set(ordered[:cut])
            cal = [r for r in group if r["launch"] in early]
            test = [r for r in group if r["launch"] not in early]

            def err(r: dict, field: str) -> float:
                return abs(math.log(min(r[field], r["horizon"]) / r["truth"]))

            if not cal or not test:
                print(f"{lead} | {model} | too few: {len(group)}")
                continue
            observed_error = statistics.median(err(r, "observed") for r in test)
            observed_bias = statistics.median(
                math.log(min(r["observed"], r["horizon"]) / r["truth"]) for r in test
            )
            persistence_error = statistics.median(err(r, "persistence") for r in test)
            ensemble = group[0]["median"] is not None
            radius = (
                conformal_radius([err(r, "median") for r in cal])
                if ensemble
                else math.inf
            )
            # Every object, so that two models pair on the same launches; a model run without the
            # ensemble leaves the ensemble's three columns empty.
            for split, part in (("cal", cal), ("test", test)):
                for r in part:
                    cells = ["", "", ""]
                    if ensemble:
                        inside = abs(math.log(r["median"] / r["truth"])) <= radius
                        raw_inside = r["lo"] <= r["truth"] <= r["hi"]
                        cells = [
                            f"{err(r, 'median'):.6g}",
                            f"{int(raw_inside)}",
                            f"{int(inside)}",
                        ]
                    out.write(
                        f"{lead}\t{model}\t{r['launch']}\t{split}\t{err(r, 'observed'):.6g}\t"
                        f"{err(r, 'persistence'):.6g}\t" + "\t".join(cells) + "\n"
                    )
            if not ensemble:
                print(
                    f"{lead} | {model} | {len(cal)} | {len(test)} | {len(ordered)} | "
                    f"{observed_error:.3f} | {observed_bias:+.3f} | {persistence_error:.3f} | — | — | — | — | —"
                )
                continue
            hits = 0
            raw = 0
            for r in test:
                inside = abs(math.log(r["median"] / r["truth"])) <= radius
                raw_inside = r["lo"] <= r["truth"] <= r["hi"]
                hits += inside
                raw += raw_inside
            lo, hi = (0, len(test))
            if not math.isinf(radius):
                lo, hi = beta_binomial_band(len(test), len(cal), ALPHA)
            print(
                f"{lead} | {model} | {len(cal)} | {len(test)} | {len(ordered)} | "
                f"{observed_error:.3f} | {observed_bias:+.3f} | {persistence_error:.3f} | "
                f"{statistics.median(err(r, 'median') for r in test):.3f} | "
                f"{raw}/{len(test)} | {radius:.3f} | {hits}/{len(test)} | {lo}–{hi}"
            )
    paired_table()
    return 0


def paired_table(draws: int = 4000, seed: int = 20261111) -> None:
    """SDTM-1 minus NRLMSISE-00 in `|ln T/T_true|` on the objects both fitted, launches resampled.

    One object per launch, so resampling objects is resampling launches. Mean and median of the
    difference's two sides, each with its 95 % percentile interval, by part and by lead.
    """
    by: dict[tuple[str, int, str], dict[str, dict[str, str]]] = {}
    with (OUT / "scores.tsv").open() as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            by.setdefault((r["split"], int(r["lead"]), r["object_launch"]), {})[
                r["model"]
            ] = r
    print(
        "\nSDTM-1 minus NRLMSISE-00, paired by launch: part | leads | objects | "
        "mean, observed | median, observed | mean, persistence"
    )
    groups = {
        "<= 90 d": lambda lead: lead <= 90,
        "365 d": lambda lead: lead == 365,
        "1095 d": lambda lead: lead == 1095,
        "all": lambda lead: True,
    }
    for part in ("test", "cal"):
        for name, wanted in groups.items():
            pairs = [
                (v["sdtm"], v["msis"])
                for (split, lead, _), v in by.items()
                if split == part and wanted(lead) and "sdtm" in v and "msis" in v
            ]
            if not pairs:
                continue
            cells = []
            for field, stat in (
                ("ln_observed", statistics.mean),
                ("ln_observed", statistics.median),
                ("ln_persistence", statistics.mean),
            ):
                a = [float(x[field]) for x, _ in pairs]
                b = [float(y[field]) for _, y in pairs]
                rng = random.Random(seed)
                spread = []
                for _ in range(draws):
                    pick = [rng.randrange(len(a)) for _ in a]
                    spread.append(
                        stat([a[i] for i in pick]) - stat([b[i] for i in pick])
                    )
                low, high = (
                    quantile(sorted(spread), 0.025),
                    quantile(sorted(spread), 0.975),
                )
                cells.append(f"{stat(a) - stat(b):+.3f} [{low:+.3f}, {high:+.3f}]")
            print(f"{part} | {name} | {len(pairs)} | " + " | ".join(cells))


def unix(*when: int) -> float:
    return dt.datetime(*when, tzinfo=dt.UTC).timestamp()


def case_job(key, model, fit, t0, state, horizon, members) -> str:
    """A job line; `fit` is `(t_h, start_state)` or a given `B`."""
    if isinstance(fit, tuple):
        head = [f"{fit[0]:.3f}", *(f"{x:.17g}" for x in fit[1])]
    else:
        head = [f"B={fit}", *["0"] * 6]
    return "\t".join(
        [
            key,
            model,
            *head,
            f"{t0:.3f}",
            *(f"{x:.17g}" for x in state),
            f"{horizon:.1f}",
            members,
        ]
    )


def worker_lines(jobs: list[str]) -> list[list[str]]:
    return [line.split("\t") for line in run_chunk(jobs).splitlines()]


def days(value: str) -> str:
    return "survives" if value == "" else f"{float(value) / DAY:.3f}"


def case_ssw() -> None:
    """January 2009: what 2 sfu moves, beside Yue et al.'s 0.2–0.3 d. SDTM-1 has no SSW in it."""
    t0 = unix(2009, 1, 15)
    flux = Feed(FEED).f81_at(t0)
    print(
        f"Jan 2009: F10.7a {flux:.1f}; +2 sfu held throughout; B fixed at each row's value"
    )
    print("height km | B m2/kg | model | lifetime d | with +2 sfu | shift d")
    for height, b in ((200, 0.02), (230, 0.02), (260, 0.02), (260, 0.05), (300, 0.05)):
        state = [6_378_137.0 + height * 1e3, 0.001, math.radians(50.0), 0.0, 0.0, 0.0]
        members = f"O;A:0:{(flux + 2.0) / flux:.17g}:1e15"
        for model in MODELS:
            job = case_job(f"ssw|{height}", model, b, t0, state, 200 * DAY, members)
            lines = worker_lines([job])
            base, more = lines[0][3], lines[1][3]
            shift = (
                ""
                if not base or not more
                else f"{(float(base) - float(more)) / DAY:.3f}"
            )
            print(f"{height} | {b} | {model} | {days(base)} | {days(more)} | {shift}")


def case_trisat() -> None:
    """May 2024: a 0.4 km/day quiet decay across the storm, beside Kosanič & Kramberger's numbers."""
    quiet = (unix(2024, 4, 21), unix(2024, 5, 9))
    print(
        "May 2024: B set to 0.4 km/day over 21 April – 9 May under the measured indices"
    )
    print(
        "height km | model | B m2/kg | peak rate km/day | drop 10-13 May km | excess km | excess / quiet rate d | drop / quiet rate d"
    )
    rate = 400.0 / DAY
    for height in (350, 400, 450):
        a = 6_378_137.0 + height * 1e3
        start = [
            a + rate * (quiet[1] - quiet[0]),
            0.001,
            math.radians(97.5),
            0.0,
            0.0,
            0.0,
        ]
        state = [a, *start[1:]]
        for model in MODELS:
            job = case_job(
                f"trisat|{height}",
                model,
                (quiet[0], start),
                quiet[1],
                state,
                16 * DAY,
                "T:21600",
            )
            lines = worker_lines([job])
            if lines[0][1] == "fit":
                print(f"{height} | {model} | no fit")
                continue
            trace = [(float(c[1][2:]), float(c[3])) for c in lines]
            at = dict(trace)
            step = 4
            peak = max(
                (trace[k][1] - trace[k + step][1]) / 1e3
                for k in range(len(trace) - step)
            )
            storm = (unix(2024, 5, 10) - quiet[1], unix(2024, 5, 13) - quiet[1])
            drop = (at[storm[0]] - at[storm[1]]) / 1e3
            excess = drop - 0.4 * (storm[1] - storm[0]) / DAY
            print(
                f"{height} | {model} | {float(lines[0][2]):.5g} | {peak:.2f} | {drop:.2f} | "
                f"{excess:.2f} | {excess / 0.4:.1f} | {drop / 0.4:.1f}"
            )


def case_starlink() -> None:
    """February 2022: 209 x 335 km at 53.22 deg, GCAT's own numbers, against GCAT's decay dates."""
    launch = unix(2022, 2, 3, 18, 13)
    # The storm had begun by the launch hour, so the quiet baseline holds the indices two days before.
    t0 = launch + 3600.0
    perigee, apogee = 209e3, 335e3
    a = 6_378_137.0 + 0.5 * (perigee + apogee)
    state = [a, 0.5 * (apogee - perigee) / a, math.radians(53.22), 0.0, 0.0, 0.0]
    decays = []
    for row in read_catalogue().values():
        if row["LDate"].startswith("2022 Feb  3") and row["Name"].startswith(
            "Starlink"
        ):
            moment = decay_time(row["DDate"])
            if moment is not None and moment < unix(2022, 3, 1):
                decays.append((moment - launch) / DAY)
    decays.sort()
    print(
        f"Feb 2022: GCAT's catalogued decays, d after launch: {[round(d, 1) for d in decays]}"
    )
    print(
        "B m2/kg | model | measured indices d | quiet, the indices of 1 Feb held d | shortened by"
    )
    for b in (0.015, 0.037, 0.074, 0.148):
        for model in MODELS:
            job = case_job(f"starlink|{b}", model, b, t0, state, 60 * DAY, "O;Q:2")
            lines = worker_lines([job])
            base, kept = lines[0][3], lines[1][3]
            ratio = (
                "" if not base or not kept else f"{1.0 - float(base) / float(kept):.1%}"
            )
            print(f"{b} | {model} | {days(base)} | {days(kept)} | {ratio}")


# Spheres of known area-to-mass (`plans/17` Ф23.T): catalogue number, name, mass kg, diameter m, source.
# A sphere's drag does not depend on its attitude, so its ballistic coefficient is `C_d A / m` with
# only `C_d` unknown. Their element sets come from `fetch-reentry-elements.py --objects`.
SPHERES = (
    (23558, "GFZ-1", 20.630, 0.215, "GFZ, research infrastructure page"),
    (25769, "Starshine 1", 39.0, 0.475, "eoPortal, STARSHINE"),
    (26996, "Starshine 2", 39.0, 0.475, "eoPortal, STARSHINE: as Starshine 1"),
    (26929, "Starshine 3", 91.0, 0.94, "eoPortal, STARSHINE; ILRS, Starshine 3"),
    # Ф23.U: the 1990s arm. The 6-inch spheres are 5.000 kg of aluminium in two press kits. The
    # 4-inch ones are 1.482 kg of aluminium in Bowman & Moe (AAS 05-257, citing JSC-27242) but 4.256 kg
    # of steel in the STS-53 kit and GCAT; they decayed before the 6-inch ones, which only aluminium
    # explains, so they enter as aluminium.
    (22994, "ODERACS E (6 in)", 5.000, 0.1524, "STS-60 and STS-63 press kits"),
    (22995, "ODERACS F (6 in)", 5.000, 0.1524, "STS-60 and STS-63 press kits"),
    (23471, "ODERACS 2A (6 in)", 5.000, 0.1524, "STS-60 and STS-63 press kits"),
    (22990, "ODERACS A (4 in)", 1.482, 0.1016, "Bowman & Moe, AAS 05-257"),
    (22991, "ODERACS B (4 in)", 1.482, 0.1016, "Bowman & Moe, AAS 05-257"),
    (23472, "ODERACS 2B (4 in)", 1.482, 0.1016, "Bowman & Moe, AAS 05-257"),
    # The 2007-2010 arm, as flown: NRL, "ANDE-RR Flight Hardware Details" (2007) and "ANDE-2 Flight
    # Hardware Details" (2009), hosted by the ILRS. GCAT's 75 kg and 0.4 m for FCal are the design's.
    (29664, "ANDE-RR MAA", 52.04, 0.4826, "NRL, ANDE-RR Flight Hardware Details"),
    (29667, "ANDE-RR FCal", 62.70, 0.4445, "NRL, ANDE-RR Flight Hardware Details"),
    (35694, "ANDE-2 Castor", 47.45, 0.4826, "NRL, ANDE-2 Flight Hardware Details"),
    (35693, "ANDE-2 Pollux", 27.442, 0.4826, "NRL, ANDE-2 Flight Hardware Details"),
)
# Spheres with no element sets in the archive, and their ILRS target names: their states are the
# laser predictions `fetch-reentry-elements.py --cpf` fetches, turned into mean elements by
# `cargo run -p spacelab --release --example cpf_mean_state`.
CPF_TARGETS = {29664: "anderra", 29667: "anderrp", 35694: "andec", 35693: "andep"}
SPHERE_CD = (1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6)


def cpf_states(target: str) -> list[tuple[float, list[float]]]:
    """Mean element sets, as `read_elements` returns them, from the target's prediction files."""
    files = sorted(str(f) for f in (OUT / "cpf" / target).glob("*"))
    if not files:
        return []
    out = subprocess.run(
        [str(CPF_STATE), *files], capture_output=True, text=True, check=True
    ).stdout
    rows = [line.split("\t")[1:] for line in out.splitlines()]
    return sorted((float(r[0]), [float(x) for x in r[1:]]) for r in rows)


def needed_cd(lifetimes: dict[float, float], truth: float) -> float | None:
    """The `C_d` at which the predicted lifetime equals the true one, log-linear between grid points.

    The lifetime falls as `C_d` rises; it is not exactly `1/C_d`, because the decay reaches each height
    at another time, under another Sun. `None` when the truth is outside the grid's lifetimes.
    """
    grid = sorted(cd for cd, t in lifetimes.items() if math.isfinite(t))
    for lo, hi in itertools.pairwise(grid):
        t_lo, t_hi = lifetimes[lo], lifetimes[hi]
        if t_hi <= truth <= t_lo:
            w = math.log(t_lo / truth) / math.log(t_lo / t_hi)
            return math.exp(math.log(lo) + w * math.log(hi / lo))
    return None


def spheres(args) -> int:
    catalogue = read_catalogue()
    jobs = []
    truth = {}
    chosen = [s for s in SPHERES if not args.objects or s[0] in args.objects]
    for number, name, mass, diameter, _ in chosen:
        decay = decay_time(catalogue[number]["DDate"])
        sets = (
            cpf_states(CPF_TARGETS[number])
            if number in CPF_TARGETS
            else read_elements(
                (OUT / "elements" / f"S{number:05d}").read_text(
                    encoding="utf-8", errors="replace"
                )
            )
        )
        if decay is None or not sets:
            print(
                f"{name}: no decay date or no element sets; run fetch-reentry-elements.py --objects"
            )
            continue
        area_over_mass = math.pi * diameter**2 / 4.0 / mass
        for lead in LEADS_DAYS:
            k = state_before(sets, decay - lead * DAY)
            if k is None:
                continue
            t0, state = sets[k]
            truth[(number, lead)] = decay - t0
            horizon = (4.0 * lead + 60.0) * DAY
            for model in MODELS:
                jobs += [
                    case_job(
                        f"{number}|{lead}|{cd}",
                        model,
                        cd * area_over_mass,
                        t0,
                        state,
                        horizon,
                        "O",
                    )
                    for cd in SPHERE_CD
                ]
    print(f"{len(jobs)} jobs", flush=True)
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        lines = [
            line.split("\t")
            for chunk in pool.map(run_chunk, ([job] for job in jobs))
            for line in chunk.splitlines()
        ]
    # A worker line is `id member B seconds`; the job lines name the model in the same order.
    lifetimes: dict[tuple[int, int, str], dict[float, float]] = {}
    for job, line in zip(jobs, lines, strict=True):
        key, model = job.split("\t")[:2]
        number, lead, cd = key.split("|")
        seconds = float(line[3]) if line[3] not in ("", "nan") else math.inf
        lifetimes.setdefault((int(number), int(lead), model), {})[float(cd)] = seconds
    print(
        "sphere | A/m m2/kg | lead d | true remaining d | model | lifetime at C_d 2.1 / 2.4 / 2.7, d | "
        "C_d it needs"
    )
    for number, name, mass, diameter, _ in chosen:
        for lead in LEADS_DAYS:
            if (number, lead) not in truth:
                continue
            t_true = truth[(number, lead)]
            for model in MODELS:
                by_cd = lifetimes[(number, lead, model)]
                cd = needed_cd(by_cd, t_true)
                print(
                    f"{name} | {math.pi * diameter**2 / 4.0 / mass:.5f} | {lead} | {t_true / DAY:.2f} | {model} | "
                    + " / ".join(f"{by_cd[c] / DAY:.2f}" for c in (2.1, 2.4, 2.7))
                    + f" | {'outside 1.2-3.6' if cd is None else f'{cd:.3f}'}"
                )
    return 0


def calibration_objects(catalogue) -> dict[str, int]:
    """One object per launch of §10.37's calibration parts, from launches that entered no test part.

    The shortest lead's object stands for its launch, the lowest catalogue number breaking a tie.
    """
    splits: dict[str, set[str]] = {}
    with (OUT / "scores.tsv").open() as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            splits.setdefault(r["object_launch"], set()).add(r["split"])
    calibration = {launch for launch, parts in splits.items() if parts == {"cal"}}
    chosen: dict[str, tuple[int, int]] = {}
    for line in (OUT / "jobs.tsv").read_text().splitlines():
        name, lead, _model = line.split("\t", 1)[0].split("|")
        number = int(name[1:])
        launch = catalogue[number]["Launch_Tag"]
        if launch in calibration:
            entry = (int(lead), number)
            chosen[launch] = min(chosen.get(launch, entry), entry)
    return {launch: number for launch, (_, number) in chosen.items()}


def split_windows(sets):
    """The fit windows `((start, end) above, (start, end) below)` around `SPLIT_PERIGEE_M`.

    The lower window runs from the first set whose perigee is below the split to the last set. The
    upper one ends where the lower one starts and begins at the latest set at least as far above in
    the mean `a`, so both windows span the same height. Returns the reason when there is none.
    """
    perigee = [s[1][0] * (1.0 - s[1][1]) - 6_378_137.0 for s in sets]
    k_mid = next((k for k, h in enumerate(perigee) if h <= SPLIT_PERIGEE_M), None)
    if not k_mid:
        return "no set on both sides of the split"
    a = [s[1][0] for s in sets]
    drop = a[k_mid] - a[-1]
    if drop < MIN_DROP_M:
        return "no real drop below the split"
    k_up = next((k for k in range(k_mid - 1, -1, -1) if a[k] - a[k_mid] >= drop), None)
    if k_up is None:
        return "no drop as long above the split"
    if any(y - x > MAX_UPWARD_JUMP_M for x, y in itertools.pairwise(a[k_up:])):
        return "an upward jump in a"
    return (sets[k_up], sets[k_mid]), (sets[k_mid], sets[-1])


def interval(values: list[float], stat, draws: int = 4000, seed: int = 20261111) -> str:
    """`stat` of `values` and its 95 % percentile interval, objects (so launches) resampled."""
    rng = random.Random(seed)
    spread = sorted(
        stat([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws)
    )
    return f"{stat(values):+.3f} [{quantile(spread, 0.025):+.3f}, {quantile(spread, 0.975):+.3f}]"


def halves(args) -> int:
    """Ф23.S (1): does `B` fitted below 225 km agree with `B` fitted above it, on the same decays?

    Below 225 km SDTM-1 is its Bates profile extrapolated, and no measured density constrains it. A
    decay does: a model whose density falls with height in the wrong shape needs another `B` below
    the split than above it. Rule, fixed before the first run: SDTM-1's shape below 225 km disagrees
    with the decays when SDTM-1 minus NRLMSISE-00 in `ln(B_below/B_above)`, paired by object, has a
    95 % interval clear of zero on both its mean and its median; a negative sign means SDTM-1 is too
    dense below the split relative to above it. A change of `C_d` with height enters both models
    alike, so only the difference is free of it; each model's own ratio is printed beside it.
    Calibration objects only: §10.37's test part stays closed.
    """
    catalogue = read_catalogue()
    feed = Feed(FEED)
    reasons: dict[str, int] = {}
    windows = {}
    for _launch, number in sorted(calibration_objects(catalogue).items()):
        decay = decay_time(catalogue[number]["DDate"])
        assert decay is not None, number  # `select` made no job without a decay date
        text = (OUT / "elements" / f"S{number:05d}").read_text(
            encoding="utf-8", errors="replace"
        )
        found = split_windows([s for s in read_elements(text) if s[0] <= decay])
        if isinstance(found, str):
            reasons[found] = reasons.get(found, 0) + 1
            continue
        windows[number] = found
    jobs = [
        case_job(f"{number}|{half}|{model}", model, start, end[0], end[1], 1.0, "O")
        for number, pair in windows.items()
        for half, (start, end) in zip(("above", "below"), pair, strict=True)
        for model in MODELS
    ]
    print(f"{len(windows)} objects, {len(jobs)} fits", flush=True)
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  left out: {reason}: {count}")
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        lines = [
            line.split("\t")
            for chunk in pool.map(run_chunk, ([job] for job in jobs))
            for line in chunk.splitlines()
        ]
    fitted: dict[tuple[int, str, str], float] = {}
    for line in lines:
        number, half, model = line[0].split("|")
        if line[1] != "fit":
            fitted[int(number), half, model] = float(line[2])
    ratio = {
        (number, model): math.log(
            fitted[number, "below", model] / fitted[number, "above", model]
        )
        for number in windows
        for model in MODELS
        if (number, "below", model) in fitted and (number, "above", model) in fitted
    }
    both = [n for n in windows if all((n, m) in ratio for m in MODELS)]
    print(f"{len(both)} objects with all four fits\n")

    def height(state) -> float:
        return (state[0] * (1.0 - state[1]) - 6_378_137.0) / 1_000.0

    for label, pick in (
        ("perigee at the upper start, km", lambda w: height(w[0][0][1])),
        ("perigee at the split, km", lambda w: height(w[1][0][1])),
        ("perigee at the last set, km", lambda w: height(w[1][1][1])),
        ("upper window, days", lambda w: (w[0][1][0] - w[0][0][0]) / DAY),
        ("lower window, days", lambda w: (w[1][1][0] - w[1][0][0]) / DAY),
    ):
        values = sorted(pick(windows[n]) for n in both)
        print(
            f"{label}: median {statistics.median(values):.1f}, "
            f"10-90 % {quantile(values, 0.1):.1f}-{quantile(values, 0.9):.1f}"
        )
    flux = {n: feed.f81_at(windows[n][1][0][0]) for n in both}
    cut = statistics.median(flux.values())
    groups = {
        "all": both,
        f"F10.7a at the split < {cut:.0f}": [n for n in both if flux[n] < cut],
        f"F10.7a at the split >= {cut:.0f}": [n for n in both if flux[n] >= cut],
    }
    with (OUT / "halves.tsv").open("w") as out:
        out.write(
            "object\tf107a_split\t"
            + "\t".join(
                f"b_{half}_{model}" for half in ("above", "below") for model in MODELS
            )
            + "\n"
        )
        for n in both:
            out.write(
                f"{n}\t{flux[n]:.1f}\t"
                + "\t".join(
                    f"{fitted[n, half, model]:.6e}"
                    for half in ("above", "below")
                    for model in MODELS
                )
                + "\n"
            )
    print("\nln(B below / B above) | objects | mean | median | median |ln|, the spread")
    for name, group in groups.items():
        for model in MODELS:
            values = [ratio[n, model] for n in group]
            print(
                f"{name}, {model} | {len(values)} | {interval(values, statistics.mean)} | "
                f"{interval(values, statistics.median)} | "
                f"{interval([abs(v) for v in values], statistics.median)}"
            )
    print("\nSDTM-1 minus NRLMSISE-00 | objects | mean | median | |ln| of each, mean")
    for name, group in groups.items():
        values = [ratio[n, "sdtm"] - ratio[n, "msis"] for n in group]
        spread = [abs(ratio[n, "sdtm"]) - abs(ratio[n, "msis"]) for n in group]
        print(
            f"{name} | {len(values)} | {interval(values, statistics.mean)} | "
            f"{interval(values, statistics.median)} | {interval(spread, statistics.mean)}"
        )
    print("\nln(B_SDTM / B_MSIS) by window | objects | mean | median")
    for half in ("above", "below"):
        values = [
            math.log(fitted[n, half, "sdtm"] / fitted[n, half, "msis"]) for n in both
        ]
        print(
            f"{half} | {len(values)} | {interval(values, statistics.mean)} | "
            f"{interval(values, statistics.median)}"
        )
    return 0


def cases(_args) -> int:
    case_ssw()
    print()
    case_trisat()
    print()
    case_starlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    sub.add_parser("select").set_defaults(func=select)
    runner = sub.add_parser("run")
    runner.add_argument("--processes", type=int, default=(os.cpu_count() or 2) // 2)
    runner.add_argument(
        "--max-lead",
        type=int,
        default=max(LEADS_DAYS),
        help="days: run the leads up to this one, so each stage fits a bounded run",
    )
    runner.set_defaults(func=run)
    check = sub.add_parser("converge")
    check.add_argument("--lead", type=int, default=30)
    check.add_argument("--objects", type=int, default=8)
    check.add_argument("--processes", type=int, default=(os.cpu_count() or 2) // 2)
    check.set_defaults(func=converge)
    sub.add_parser("score").set_defaults(func=score)
    sub.add_parser("cases").set_defaults(func=cases)
    sphere = sub.add_parser("spheres")
    sphere.add_argument("--processes", type=int, default=(os.cpu_count() or 2) // 2)
    sphere.add_argument(
        "--objects",
        type=int,
        nargs="+",
        metavar="N",
        help="only these catalogue numbers",
    )
    sphere.set_defaults(func=spheres)
    split = sub.add_parser("halves")
    split.add_argument("--processes", type=int, default=(os.cpu_count() or 2) // 2)
    split.set_defaults(func=halves)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
