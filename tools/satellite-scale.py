#!/usr/bin/env python3
"""How much of SDTM-1's held-out error is one scale per satellite (`plans/17` Ф30.G).

Reads the family folds of `tools/evaluate-sdtm.py` (`data/validation/sdtm-eval/FIELD/family-*.npz`). A
held-out family has no scale of its own, so it sits on the level of the fold's reference satellite (GRACE-FO
C, or Swarm A where GRACE-FO is held out). Per family and pooled, it prints the held-out MAPE:

* as scored, at the reference's level;
* with the best constant per satellite, the shift that minimises that satellite's MAPE on its own test rows:
  no scale chosen without the test rows can do better, so this bounds what a per-satellite scale can buy;
* at the training satellites' mean level, the reference's zero included: what a scale drawn from the
  training satellites' spread gives a satellite it has never seen;

and NRLMSISE-00's as scored and with its best constant. It prints each fold's fitted training scales, and
with `--field no_scales` scores the fit that gives no satellite a scale (Ф30.G's other end).

    uv run --no-project --with numpy --with scipy python -I tools/satellite-scale.py [--field default]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "validation" / "sdtm-eval"
# `tools/fit-sdtm-corpus.py`'s order: a fold's reference is the first of these it trains on.
SATELLITES = ("GC", "SA", "SB", "SC", "GA", "GB", "CH", "GO")
FAMILY = {
    "CH": "CHAMP",
    "GA": "GRACE",
    "GB": "GRACE",
    "GC": "GRACE-FO",
    "SA": "Swarm",
    "SB": "Swarm",
    "SC": "Swarm",
    "GO": "GOCE",
}
FAMILIES = ("CHAMP", "GOCE", "GRACE", "GRACE-FO", "Swarm")


def mape(residual: np.ndarray) -> float:
    return 100.0 * float(np.mean(np.abs(np.expm1(residual))))


def best_shift(residual: np.ndarray) -> float:
    return float(
        minimize_scalar(
            lambda c: mape(residual - c), bounds=(-1.0, 1.0), method="bounded"
        ).x
    )


def per_satellite_best(residual: np.ndarray, sat: np.ndarray) -> np.ndarray:
    out = residual.copy()
    for s in set(sat):
        here = sat == s
        out[here] -= best_shift(residual[here])
    return out


def training_scales(parameters: np.ndarray, family: str, i_scale: int) -> dict:
    train = [s for s in SATELLITES if FAMILY[s] != family]
    scales = dict(
        zip(train[1:], parameters[i_scale : i_scale + len(train) - 1], strict=True)
    )
    return {train[0]: 0.0, **scales}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", default="default")
    # The shipped form's slots before the scales: the admitted pairs, three kernel slots, the boundary.
    parser.add_argument("--i-scale", type=int, default=40)
    args = parser.parse_args()
    scaled = args.field != "no_scales"
    print(
        "family | SDTM-1 as scored | best constant per satellite | training mean level | "
        "NRLMSISE-00 as scored | its best constant | fitted training scales, %"
    )
    pooled: dict[str, list[np.ndarray]] = {}
    for family in FAMILIES:
        z = np.load(EVAL / args.field / f"family-{family}.npz")
        done = np.isfinite(z["msis"])
        residual, msis, sat = z["residual"][done], z["msis"][done], z["sat"][done]
        scales = (
            training_scales(z["parameters"], family, args.i_scale) if scaled else {}
        )
        mean_level = float(np.mean(list(scales.values()))) if scales else 0.0
        columns = {
            "scored": residual,
            "best": per_satellite_best(residual, sat),
            "mean level": residual + mean_level,
            "msis": msis,
            "msis best": per_satellite_best(msis, sat),
        }
        for key, value in columns.items():
            pooled.setdefault(key, []).append(value)
        print(
            f"{family} | "
            + " | ".join(f"{mape(v):.2f} %" for v in columns.values())
            + " | "
            + (
                " ".join(f"{s} {100.0 * v:+.1f}" for s, v in scales.items())
                if scales
                else "none"
            )
        )
    print(
        "pooled | "
        + " | ".join(f"{mape(np.concatenate(v)):.2f} %" for v in pooled.values())
        + " |"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
