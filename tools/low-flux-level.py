#!/usr/bin/env python3
"""Two fields' held-out level at low solar activity, on the same family-fold rows (`plans/17` Ф30.G-2).

Reads the family folds of `tools/evaluate-sdtm.py` (`data/validation/sdtm-eval/FIELD/family-*.npz`) for the
two `--fields`, checks they hold the same test rows, and keeps the rows whose 81-day F10.7 is below each
`--below`. Per family it prints the rows kept, each field's MAPE, and each field's median model/measured,
pooled and per satellite. A level shift shows as one ratio between the medians in every family.

    uv run --project tools --locked python -I tools/low-flux-level.py [--fields default gc_v02] [--below 80 70]
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

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

FAMILIES = ("CHAMP", "GOCE", "GRACE", "Swarm", "GRACE-FO")
EVAL = HERE.parent / "data" / "validation" / "sdtm-eval"
# `tools/evaluate-sdtm.py`'s corpus step and phase: its `test` indices are rows of this load.
STEP, PHASE = 60, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fields", nargs=2, default=["default", "gc_v02"])
    parser.add_argument("--below", type=float, nargs="+", default=[80.0, 70.0])
    args = parser.parse_args()
    first, second = args.fields
    f107a = fit.load_corpus(STEP, phase=PHASE).f107a.to_numpy()
    for cut in args.below:
        print(f"F10.7a < {cut:g}")
        print(
            f"family | rows | MAPE {first} | MAPE {second} | median model/measured {first} | {second} | "
            f"per satellite, {first} -> {second}"
        )
        for family in FAMILIES:
            a, b = (np.load(EVAL / f / f"family-{family}.npz") for f in args.fields)
            if not np.array_equal(a["test"], b["test"]):
                sys.exit(f"{family}: the two fields' folds hold different test rows")
            low = f107a[a["test"]] < cut
            if not low.any():
                print(f"{family} | 0 | — | — | — | — | —")
                continue
            ra, rb, sat = a["residual"][low], b["residual"][low], a["sat"][low]
            per = " ".join(
                f"{s} {np.exp(np.median(ra[sat == s])):.3f} -> {np.exp(np.median(rb[sat == s])):.3f}"
                for s in np.unique(sat)
            )
            print(
                f"{family} | {low.sum()} | {g.mape(ra):.2f} | {g.mape(rb):.2f} | "
                f"{np.exp(np.median(ra)):.3f} | {np.exp(np.median(rb)):.3f} | {per}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
