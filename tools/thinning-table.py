#!/usr/bin/env python3
"""SDTM-1 fitted at 120 s, 10 min, 30 min and 2 h, compared on the same rows and storms (`plans/17` Ф30.E).

Each run is one family fold of `tools/fit-sdtm-corpus.py --step K --score-step 60 --storms`, logged to
`data/validation/sdtm-eval/thinning/step{K}-{family}.log`. This tool reads the logs: the held-out MAPE on the
rows a run at step 60 reads, the same for every K, and the storm assessment's per-storm onset error (h), peak
ratio (model over measured) and half-recovery error (h), on the 120 s storm windows. It prints them by step
and by the storm's first-peak `ap`, and the change from step 60 storm by storm, on the storms both steps scored.

    uv run --project tools --locked python -I tools/thinning-table.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
THINNING = HERE.parent / "data/validation/sdtm-eval/thinning"
FAMILIES = ("CHAMP", "GOCE", "GRACE", "GRACE-FO", "Swarm")
CLASSES = ((80.0, 150.0), (150.0, 300.0), (300.0, np.inf))
SCORE = re.compile(
    r"on the rows of --step 60: (\d+) rows, SDTM-1 held out ([\d.]+) %, debiased ([\d.]+) %, "
    r"Ap >= 50 ([\d.]+) % \((\d+) rows\); NRLMSISE-00 ([\d.]+) %"
)
STORM = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d)\s+(\d+)\s+(MP|SP)\s+(\S+)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+"
    r"\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)$"
)


def read(path: pathlib.Path) -> tuple[dict, pd.DataFrame]:
    """The run's step-60 score and its SDTM-1 storm assessment, one row per storm and satellite."""
    text = path.read_text()
    score = SCORE.search(text)
    if score is None:
        raise ValueError(f"{path.name}: no --score-step 60 line")
    rows, sdtm, debiased, storm, n_storm, msis = score.groups()
    head = text.index("== storm assessment on the SDTM-1 held-out-mission residual ==")
    tail = text.index("comparisons:", head)
    storms = [
        STORM.match(line.strip()).groups()
        for line in text[head:tail].splitlines()
        if STORM.match(line.strip())
    ]
    table = pd.DataFrame(
        storms,
        columns=["t0", "peak_ap", "type", "mission", "n", "onset", "peak", "half"],
    ).astype({"peak_ap": float, "n": int, "onset": float, "peak": float, "half": float})
    return {
        "rows": int(rows),
        "sdtm": float(sdtm),
        "debiased": float(debiased),
        "storm": float(storm),
        "storm_rows": int(n_storm),
        "msis": float(msis),
    }, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    runs: dict[int, dict[str, tuple[dict, pd.DataFrame]]] = {}
    for path in sorted(THINNING.glob("step*-*.log")):
        step, family = re.match(r"step(\d+)-(.+)\.log", path.name).groups()
        try:
            runs.setdefault(int(step), {})[family] = read(path)
        except ValueError as error:
            print(f"skipped: {error}", file=sys.stderr)
    steps = sorted(runs)
    if 60 not in runs:
        print("no step-60 run yet", file=sys.stderr)
        return 1
    print(
        "held-out MAPE on the step-60 rows, %: SDTM-1 / Ap >= 50; NRLMSISE-00 on the same rows"
    )
    print("family | " + " | ".join(f"step {k}" for k in steps))
    for family in FAMILIES:
        cells = []
        for k in steps:
            got = runs[k].get(family)
            cells.append(
                "—"
                if got is None
                else f"{got[0]['sdtm']:.2f} / {got[0]['storm']:.2f}; {got[0]['msis']:.2f}"
            )
        print(f"{family} | " + " | ".join(cells))
    print(
        "\nstorms by first-peak ap, on the families every step ran: mean |onset| h / mean |peak M/O - 1| / mean |half| h"
    )
    common = [f for f in FAMILIES if all(f in runs[k] for k in steps)]
    print(f"families: {', '.join(common) or 'none'}")
    if not common:
        return 0
    tables = {
        k: pd.concat([runs[k][f][1] for f in common], ignore_index=True) for k in steps
    }
    key = ["t0", "mission"]
    print("first-peak ap | storms | " + " | ".join(f"step {k}" for k in steps))
    for low, high in ((80.0, np.inf), *CLASSES):
        cells = []
        for k in steps:
            t = tables[k][(tables[k].peak_ap >= low) & (tables[k].peak_ap < high)]
            cells.append(
                f"{t.onset.abs().mean():.2f} / {(t.peak - 1).abs().mean():.3f} / {t.half.abs().mean():.2f}"
            )
        n = int(((tables[60].peak_ap >= low) & (tables[60].peak_ap < high)).sum())
        label = f"{low:g}-{high:g}" if np.isfinite(high) else f">= {low:g}"
        print(f"{label} | {n} | " + " | ".join(cells))
    base = tables[60].set_index(key)
    print(
        "\nstorm by storm against step 60: mean change in |peak M/O - 1| and in |onset| h, storms both scored"
    )
    for k in steps:
        if k == 60:
            continue
        other = tables[k].set_index(key)
        both = base.merge(
            other, left_index=True, right_index=True, suffixes=("_60", "_k")
        )
        d_peak = (both.peak_k - 1).abs() - (both.peak_60 - 1).abs()
        d_onset = both.onset_k.abs() - both.onset_60.abs()
        print(
            f"step {k}: {len(both)} storms; |peak - 1| {d_peak.mean():+.3f} (better on {(d_peak < 0).sum()}), "
            f"|onset| {d_onset.mean():+.2f} h (better on {(d_onset < 0).sum()})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
