#!/usr/bin/env python3
"""The frozen-N step of SDTM-1's nitric-oxide cooling against its exact solution, on every storm (`plans/17` Ф23.Q).

`cooling_grid`'s step freezes N at its exact mean over each 3 h interval (`derivations/sdtm_no_cooling.mac`).
The exact step of the same system under the held drive is `derivations/sdtm_no_step.mac`'s: N exact, D by its
integrating factor, whose one integral is taken here by Gauss-Legendre at 32 nodes. Both run over the 3-hourly
`ap` from rest, with the shipped constants, and so do 2, 4 and 8 sub-steps of the frozen step. On every storm
the protocol of Bruinsma & Laurens (2024) finds from the corpus's first month on, the tool prints the error of
the storm's peak and the largest error over its window, both relative to the exact peak. Beside them it sets the
spread of the same exact peak over the constants the five family folds fitted: how far the fit itself is from
knowing the state.

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/no-step-accuracy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
NODES = 32
SUBSTEPS = (1, 2, 4, 8)
PEAK_CLASSES = ((80.0, 150.0), (150.0, 300.0), (300.0, np.inf))


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_coefficients(
    n0: np.ndarray,
    drive: np.ndarray,
    heating: float,
    nitric: float,
    saturation: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`(alpha, beta)` with the exact `D(h) = alpha D(0) + beta`, N starting at `n0`, times in hours.

    `A(x) = x/tc + g (H x + (N0 - H) tn (1 - e^(-x/tn)))`, `g = 1/(tc H_s)`: alpha is `e^(-A(h))` and beta
    `(H/tc) int_0^h e^(A(x) - A(h)) dx`, whose integrand lies in (0, 1] and is analytic.
    """
    g = 1.0 / (heating * saturation)

    def a(x):
        return x / heating + g * (
            drive[:, None] * x + (n0 - drive)[:, None] * nitric * -np.expm1(-x / nitric)
        )

    node, weight = np.polynomial.legendre.leggauss(NODES)
    x = 0.5 * h * (node + 1.0)
    end = a(np.array([h]))[:, 0]
    beta = drive / heating * (0.5 * h * np.exp(a(x) - end[:, None]) @ weight)
    return np.exp(-end), beta


def exact_grid(
    ap: np.ndarray,
    heating: float,
    nitric: float,
    saturation: float,
    cadence: float,
    first: int,
) -> np.ndarray:
    """The exact D after each interval, `cooling_grid`'s conventions: from rest at `first`, zeros before."""
    h = cadence / 3600.0
    decay = np.exp(-h / nitric)
    drive = ap[first:]
    n_after = np.empty(drive.size)
    n = 0.0
    for i, u in enumerate(drive):
        n = u + (n - u) * decay
        n_after[i] = n
    n0 = np.concatenate([[0.0], n_after[:-1]])
    alpha, beta = exact_coefficients(n0, drive, heating, nitric, saturation, h)
    out = np.zeros(ap.size)
    d = 0.0
    for i in range(drive.size):
        d = alpha[i] * d + beta[i]
        out[first + i] = d
    return out


def frozen_grid(
    ap: np.ndarray,
    heating: float,
    nitric: float,
    saturation: float,
    cadence: float,
    first: int,
    substeps: int,
) -> np.ndarray:
    """`cooling_grid`'s frozen-N step taken `substeps` times across each interval, its conventions."""
    h = cadence / 3600.0 / substeps
    decay = np.exp(-h / nitric)
    mean_weight = nitric / h * (1.0 - decay)
    out = np.zeros(ap.size)
    d = n = 0.0
    for i in range(first, ap.size):
        drive = ap[i]
        for _ in range(substeps):
            n_mean = drive + (n - drive) * mean_weight
            rate = (1.0 + n_mean / saturation) / heating
            target = drive / (heating * rate)
            d = target + (d - target) * np.exp(-rate * h)
            n = drive + (n - drive) * decay
        out[i] = d
    return out


def fold_kernels(harness) -> list[tuple[float, float, float]]:
    """`(heating h, nitric h, saturation ap)` each family fold of the shipped form fitted."""
    g = harness.g
    n_c = int(g.harmonic_mask(g.CAPS[harness.FORM["caps"]], (2 + 1) ** 2).sum())
    out = []
    for split in harness.read_splits():
        if split["kind"] != "family":
            continue
        p = np.load(harness.OUT / "default" / f"{split['name']}.npz")["parameters"]
        out.append(tuple(float(v) for v in np.exp(p[n_c : n_c + 3])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    fit, g = harness.fit, harness.g
    flux = pd.read_csv(
        g.DENSITY / "SW-All.csv", usecols=["DATE"] + [f"AP{i}" for i in range(1, 9)]
    )
    epoch = pd.Timestamp(flux.DATE.iloc[0])
    ap = flux[[f"AP{i}" for i in range(1, 9)]].to_numpy().reshape(-1).astype(float)
    ap = np.where(np.isfinite(ap), ap, 0.0)
    manifest = json.loads(
        (HERE.parent / "data/validation/tudelft-v02/corpus/manifest.json").read_text()
    )
    corpus_start = pd.Timestamp(min(m["month"] for m in manifest["months"]) + "-01")
    start = (corpus_start - epoch).days * 8
    storms = g.storm_onsets(ap[start:], epoch + pd.Timedelta(days=start // 8))
    cadence = g.AP_CADENCE
    # From rest 120 days before the first storm, as the fit warms its state up.
    first = start - 120 * 8
    shipped = fit.shipped_kernel()
    print(
        f"{len(storms)} storms from {corpus_start:%Y-%m}; shipped heating {shipped[0]:.3f} h, "
        f"nitric oxide {shipped[1]:.3f} h, saturation {shipped[2]:.3f} ap; exact by Gauss-Legendre at {NODES} nodes",
        flush=True,
    )
    exact = exact_grid(ap, *shipped, cadence, first)
    steps = {k: frozen_grid(ap, *shipped, cadence, first, k) for k in SUBSTEPS}
    reference = fit.cooling_grid(ap, *shipped, cadence, first)
    print(
        "cooling_grid against this tool's single frozen step, largest relative difference: "
        f"{np.max(np.abs(reference[first:] - steps[1][first:]) / np.maximum(exact[first:], 1e-300)):.1e}",
        flush=True,
    )
    folds = [exact_grid(ap, *k, cadence, first) for k in fold_kernels(harness)]
    rows = []
    for s in storms:
        t0 = (pd.Timestamp(s["t0"]) - epoch) // pd.Timedelta(hours=3)
        lo = t0 + int(g.PHASES[0][0] // 3)
        hi = t0 + int(np.ceil((s["phase3_end"] + 12.0) / 3.0)) + 1
        window = slice(lo, hi)
        peak = exact[window].max()
        row = {"ap": ap[t0], "exact peak": peak}
        for k, grid in steps.items():
            row[f"peak x{k}"] = 100.0 * (grid[window].max() - peak) / peak
            row[f"window x{k}"] = (
                100.0 * np.abs(grid[window] - exact[window]).max() / peak
            )
        peaks = np.array([f[window].max() for f in folds])
        row["fold spread"] = 100.0 * (peaks.max() - peaks.min()) / np.median(peaks)
        rows.append(row)
    table = pd.DataFrame(rows)
    print("\nper storm, % of the exact peak: median / 90th percentile / largest")
    print(
        "storms | first-peak ap | "
        + " | ".join(f"peak x{k}" for k in SUBSTEPS)
        + " | window x1 | fold spread"
    )
    for low, high in ((80.0, np.inf), *PEAK_CLASSES):
        part = table[(table.ap >= low) & (table.ap < high)]
        if part.empty:
            continue

        def cell(column: str, part: pd.DataFrame = part) -> str:
            v = part[column].to_numpy()
            return f"{np.median(v):+.2f} / {np.percentile(v, 90):+.2f} / {v[np.argmax(np.abs(v))]:+.2f}"

        label = f"{low:g}-{high:g}" if np.isfinite(high) else f">= {low:g}"
        print(
            f"{len(part)} | {label} | "
            + " | ".join(cell(f"peak x{k}") for k in SUBSTEPS)
            + f" | {cell('window x1')} | {cell('fold spread')}"
        )
    ratio = table["peak x1"].abs() / table["fold spread"]
    print(
        f"\nthe shipped step's peak error over the folds' spread: median {ratio.median():.3f}, "
        f"largest {ratio.max():.3f}; storms where it exceeds a tenth of the spread: {(ratio > 0.1).sum()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
