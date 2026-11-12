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
        covered = np.zeros(len(rows), dtype=bool)
        for fam in FAMILIES:
            here = family == fam
            table, _ = g.uncertainty(held[~here], arg["f107a"][~here], storm[~here])
            covered[here] = np.abs(held[here]) <= g.NORMAL_90 * table[cells[here]]
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
