#!/usr/bin/env python3
"""The 90 % interval's coverage by regime on the family folds, for several fields (`plans/17` Ф23.L).

Reads the family folds of `tools/evaluate-sdtm.py` (`data/validation/sdtm-eval/FIELD/family-*.npz`); the five
families partition the corpus, so each row has one held-out residual. For each family the uncertainty table is
built from the other four families' residuals (`gen_sdtm_coefficients.uncertainty`), as `--emit` checks the
shipped table, and the family's rows are covered where `|r| <= 1.645 sigma` of their cell. The cells are
`gen_sdtm_coefficients.regime`'s, the 81-day flux by the storm state, and the storm state is the shipped
kernel's (`coefficients.rs`) for every field, so all fields are scored on the same cells. Regime 5 is the
high-flux storm cell. A second line per field covers the rows of §10.2's campaign, GRACE-FO C and Swarm A in
its six months, held out the same way: where the shipped table under-covers regime 5 in sample; then the
campaign's regime-5 rows by month, with their median model/measured.

Then the split of those rows' `ln(model/measured)` into the two parts the storm protocol reports (`plans/18`
Ф23.Z): the pre-storm level, model/measured over the satellite's rows in the pre-storm phase of the storm
whose window holds the row, and the debiased storm ratio, measured over the model divided by that level,
which is `<O/C>`'s orientation; `ln(model/measured) = ln(level) - ln(measured/debiased model)`. A row in two
windows belongs to the later storm. Each row the interval misses is printed with its two parts and whether
it falls inside its cell's band once the level is removed.

    uv run --project tools --locked python -I tools/regime-coverage.py [--fields default flux_memory]
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

FAMILIES = ("CHAMP", "GOCE", "GRACE", "GRACE-FO", "Swarm")
EVAL = HERE.parent / "data" / "validation" / "sdtm-eval"
# `tools/evaluate-sdtm.py`'s corpus step and phase: its `test` indices are rows of this load.
STEP, PHASE = 60, 0
# §10.2's campaign: the satellites and months `tools/fetch-density-data.sh` fetches.
CAMPAIGN_SATS = ("GC", "SA")
CAMPAIGN_MONTHS = ("2019-01", "2021-11", "2022-04", "2023-04", "2024-05", "2025-06")


def held_out(field: str, size: int) -> np.ndarray:
    """Each row's residual from the family fold that held it out."""
    held = np.full(size, np.nan)
    for family in FAMILIES:
        z = np.load(EVAL / field / f"family-{family}.npz")
        held[z["test"]] = z["residual"]
    if not np.isfinite(held).all():
        sys.exit(f"{field}: the family folds do not cover every row")
    return held


def storm_split(
    t: np.ndarray,
    sat: np.ndarray,
    held: np.ndarray,
    keep: np.ndarray,
    storms: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `keep`, the index in `storms` of the latest storm whose window holds it (-1 for
    none), and that storm's pre-storm level for the row's satellite: the mean `ln(model/measured)` of the
    satellite's rows in phase 1, `nan` where it has none."""
    owner = np.full(t.size, -1)
    level = np.full(t.size, np.nan)
    hour = np.timedelta64(3_600_000_000, "us")
    for j, storm in enumerate(storms):
        t0 = np.datetime64(storm["t0"], "us")
        start, pre_end = (t0 + h * hour for h in g.PHASES[0])
        inside = keep & (t >= start) & (t < t0 + (storm["phase3_end"] + 12.0) * hour)
        owner[inside] = j
        level[inside] = np.nan
        for s in set(sat[inside]):
            pre = (sat == s) & (t >= start) & (t < pre_end)
            if pre.any():
                level[inside & (sat == s)] = held[pre].mean()
    return owner, level


def print_split(
    field: str,
    rows,
    held: np.ndarray,
    band: np.ndarray,
    high: np.ndarray,
    storms: list[dict],
) -> None:
    """`high`'s rows by storm window and satellite, then each miss, with the level split (Ф23.Z)."""
    t = rows.t.to_numpy().astype("datetime64[us]")
    sat = rows.sat.to_numpy()
    owner, level = storm_split(t, sat, held, high, storms)
    covered = np.abs(held) <= band
    print(
        f"  {field}, campaign regime-5 rows by storm window; level: model/measured over the "
        "pre-storm phase; ln(model/measured) = ln(level) - ln(measured/debiased model)"
    )
    print(
        "    storm t0          peak ap  sat  rows  covered %   level  median model/measured"
        "  median measured/debiased model"
    )
    for j in sorted(set(owner[high & (owner >= 0)])):
        for s in sorted(set(sat[high & (owner == j)])):
            here = high & (owner == j) & (sat == s)
            print(
                f"    {storms[j]['t0']:%Y-%m-%d %H:%M} {storms[j]['peak_ap']:8.0f}  {s:<3}"
                f"{here.sum():6d}{100.0 * covered[here].mean():11.1f}{np.exp(level[here][0]):8.3f}"
                f"{np.exp(np.median(held[here])):23.3f}"
                f"{np.exp(np.median(level[here] - held[here])):32.3f}"
            )
    outside = high & (owner < 0)
    if outside.any():
        print(
            f"    outside every storm window: {outside.sum()} rows, covered "
            f"{100.0 * covered[outside].mean():.1f} %, median model/measured "
            f"{np.exp(np.median(held[outside])):.3f}; no pre-storm level"
        )
    miss = high & ~covered
    # Inside the band once the level is removed: `nan` where there is no level, which compares false.
    inside = np.abs(held - level) <= band
    print(
        f"  {field}, the {miss.sum()} regime-5 misses: time, satellite, model/measured, storm t0, level, "
        "measured/debiased model, inside the band without the level"
    )
    for i in np.where(miss)[0]:
        head = f"    {rows.t.iat[i]:%Y-%m-%d %H:%M} {sat[i]:<3}{np.exp(held[i]):7.3f}"
        if owner[i] < 0 or not np.isfinite(level[i]):
            print(head + "  outside every storm window or no pre-storm row: no split")
            continue
        print(
            head + f"  {storms[owner[i]]['t0']:%Y-%m-%d %H:%M}{np.exp(level[i]):7.3f}"
            f"{np.exp(level[i] - held[i]):8.3f}  {'yes' if inside[i] else 'no'}"
        )
    split = miss & (owner >= 0) & np.isfinite(level)
    print(
        f"  {field}: of {miss.sum()} misses, {split.sum()} have a pre-storm level; once it is removed, "
        + "; ".join(
            f"{int((inside & side)[split].sum())} of the {int(side[split].sum())} with the model "
            f"{name} the measurement fall inside the band"
            for name, side in (("above", held > 0.0), ("below", held < 0.0))
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fields", nargs="+", default=["default", "flux_memory"])
    args = parser.parse_args()
    rows = fit.load_corpus(STEP, phase=PHASE)
    arg = g.fit_arguments(rows, 2)
    heating, nitric, saturation = fit.shipped_kernel()
    per_day = round(86_400.0 / arg["cadence"])
    first = max(int(arg["interval"].min()) - 120 * per_day, 0)
    storm = fit.cooling_grid(
        arg["ap"], heating, nitric, saturation, arg["cadence"], first
    )[arg["interval"]]
    cells = g.regime(arg["f107a"], storm)
    storms = g.storm_onsets(rows.attrs["ap"], rows.attrs["ap_epoch"])
    family = rows.family.to_numpy()
    campaign = (
        rows.sat.isin(CAMPAIGN_SATS).to_numpy()
        & rows.month.isin(CAMPAIGN_MONTHS).to_numpy()
    )
    n_cells = (len(g.UNCERTAINTY_FLUX) + 1) * (len(g.UNCERTAINTY_STORM) + 1)
    print(f"rows per regime: {np.bincount(cells, minlength=n_cells).tolist()}")
    print(
        "campaign rows per regime: "
        f"{np.bincount(cells[campaign], minlength=n_cells).tolist()}"
    )
    print(
        "field | covered by regime 0..5, %, the table built without the row's family | pooled | MAPE"
    )
    for field in args.fields:
        held = held_out(field, len(rows))
        band = np.zeros(len(rows))
        for fam in FAMILIES:
            here = family == fam
            table, _ = g.uncertainty(held[~here], arg["f107a"][~here], storm[~here])
            band[here] = g.NORMAL_90 * table[cells[here]]
        covered = np.abs(held) <= band
        for name, keep in (
            ("corpus", np.ones(len(rows), dtype=bool)),
            ("campaign", campaign),
        ):
            by_regime = [
                100.0 * covered[keep & (cells == c)].mean() for c in range(n_cells)
            ]
            print(
                f"{field}, {name} | "
                + " / ".join(f"{v:.1f}" for v in by_regime)
                + f" | {100.0 * covered[keep].mean():.1f} | {g.mape(held[keep]):.2f}"
            )
        high = campaign & (cells == len(by_regime) - 1)
        for month in CAMPAIGN_MONTHS:
            here = high & (rows.month.to_numpy() == month)
            if here.any():
                print(
                    f"  {field}, campaign regime {len(by_regime) - 1}, {month}: {here.sum()} rows, "
                    f"covered {100.0 * covered[here].mean():.1f} %, "
                    f"median model/measured {np.exp(np.median(held[here])):.3f}"
                )
        print_split(field, rows, held, band, high, storms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
