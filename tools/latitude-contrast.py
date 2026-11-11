#!/usr/bin/env python3
"""SDTM-1's quiet latitude contrast, by season and by height: `plans/17` Ф23.V.

§10.40 found SDTM-1's held-out residual higher at high latitude than at low on CHAMP and GOCE, before, during and
after storms, and as large in geographic as in magnetic latitude: a property of the quiet model. This tool reads
it on the quiet rows of the family folds (`tools/evaluate-sdtm.py`), daily `Ap < 15`, in geographic latitude.

The contrast is `100 x` (mean `ln(model/measured)` at `|lat| >= 60` minus that at `|lat| < 40`), with a 95 %
interval resampling 27-day blocks. It is split, as fixed before the residuals were read:

* by season: in the solstice months (May-August, November-February) the high rows of the summer hemisphere and of
  the winter hemisphere, each against the low rows of the same months; in the equinox months, all high rows
  against the low rows;
* by height, every family pooled: below 300 km, 300-400, 400-475 and from 475 km.

The reading, fixed with it: a contrast of one sign in summer and in winter belongs to the mean field's latitude
structure, not to the season's hemispheric term; one that grows as the height falls belongs to the profile's
shape below `T_inf`, not to `T_inf`. NRLMSISE-00 on the same rows is printed beside each field.

The pooled height bins mix families, eras and flux levels. A last table, added after the first run had been read
and so descriptive only, splits each family by height and by F10.7a (below and from 100 sfu). With `--smoothing`
it adds, per window, the shift a centred boxcar over the measured density would put into the contrast: Swarm's
densities come from GPS accelerations with an effective resolution near 20 min (van den IJssel et al., EGU21-5580),
every other family's from an accelerometer.

Ф23.V's latitude rows (`tools/evaluate-sdtm.py`'s `degree3` and `boundary_latitude`), the rule written before
their folds were read: a row explains the structure if its `all` contrast on CHAMP, on GOCE and on Swarm is half
the shipped field's or less in size, and its paired family-fold MAPE (`tools/ablation-table.py`) is no worse.

    uv run --no-project --with numpy --with pandas --with scipy python -I tools/latitude-contrast.py [--fields default]
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
HIGH, LOW = 60.0, 40.0
QUIET_AP = 15.0
HEIGHTS = ((0.0, 300.0), (300.0, 400.0), (400.0, 475.0), (475.0, 1000.0))
SUMMER_NORTH = (5, 6, 7, 8)
SUMMER_SOUTH = (11, 12, 1, 2)
FAMILIES = ("CHAMP", "GOCE", "GRACE", "GRACE-FO", "Swarm")
MIN_ROWS = 300


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def smoothing_width(minutes: float) -> float:
    """The boxcar's width in minutes: an odd number of the corpus's 120 s samples, centred on the row."""
    return (2 * round(minutes / 4.0) + 1) * 2.0


def smoothing_shift(rows, fit, minutes: float) -> np.ndarray:
    """Per row, `ln` of the measured density averaged over the centred boxcar minus `ln` of the row's own,
    from the corpus at its full 120 s cadence; NaN where a sample of the window is missing."""
    half = round(minutes / 4.0)
    width = 2 * half + 1
    out = np.full(len(rows), np.nan)
    sat = rows.sat.to_numpy()
    t_row = ((rows.t - fit.EPOCH_2000) // np.timedelta64(1, "ms")).to_numpy()
    for code in np.unique(sat):
        z = np.load(fit.CORPUS / f"corpus_{code}.npz")
        t, rho = z["t_ms"], z["rho"]
        # Months in GPS time sit whole seconds off the first month's grid, the leap seconds between them.
        k = np.rint((t - t[0]) / 120_000.0).astype(np.int64)
        if np.any(np.abs(t - t[0] - k * 120_000) > 30_000) or np.any(np.diff(k) <= 0):
            raise ValueError(f"{code}: corpus samples not on one 120 s grid")
        grid = np.full(int(k[-1]) + 1 + 2 * half, np.nan)
        grid[k + half] = rho
        present = np.isfinite(grid)
        total = np.concatenate([[0.0], np.cumsum(np.where(present, grid, 0.0))])
        count = np.concatenate([[0], np.cumsum(present)])
        here = np.flatnonzero(sat == code)
        i = np.searchsorted(t, t_row[here])
        if not np.array_equal(t[np.minimum(i, t.size - 1)], t_row[here]):
            raise ValueError(f"{code}: a row not in its corpus file")
        start, stop = k[i], k[i] + width
        mean = (total[stop] - total[start]) / width
        out[here] = np.where(
            count[stop] - count[start] == width, np.log(mean / rho[i]), np.nan
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fields", nargs="+", default=["default"])
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--smoothing", nargs="*", type=float, default=[])
    args = parser.parse_args()
    harness = load("evaluate_sdtm", "evaluate-sdtm.py")
    maps = load("residual_maps", "residual-maps.py")
    intervals = load("cluster_intervals", "cluster-intervals.py")
    rows = harness.fit.load_corpus(harness.STEP, phase=harness.PHASE)
    series: dict[str, np.ndarray] = {}
    ap_daily = np.full(len(rows), np.nan)
    for field in args.fields:
        held = np.full(len(rows), np.nan)
        msis = np.full(len(rows), np.nan)
        for family in FAMILIES:
            z = np.load(harness.OUT / field / f"family-{family}.npz")
            held[z["test"]] = z["residual"]
            msis[z["test"]] = z["msis"]
            ap_daily[z["test"]] = z["ap_daily"]
        series[field] = held
        series.setdefault("NRLMSISE-00", msis)
    done = np.isfinite(series["NRLMSISE-00"]) & (ap_daily < QUIET_AP)
    for values in series.values():
        done &= np.isfinite(values)
    latitude = rows.latitude.to_numpy()
    month = rows.t.dt.month.to_numpy()
    altitude = rows.altitude.to_numpy() / 1000.0
    family = rows.family.to_numpy()
    days = (rows.t - rows.t.min()).dt.days.to_numpy()
    block = days // 27
    high, low = np.abs(latitude) >= HIGH, np.abs(latitude) < LOW
    north_summer = np.isin(month, SUMMER_NORTH)
    south_summer = np.isin(month, SUMMER_SOUTH)
    solstice = north_summer | south_summer
    summer = high & (
        (north_summer & (latitude > 0.0)) | (south_summer & (latitude < 0.0))
    )
    winter = high & (
        (north_summer & (latitude < 0.0)) | (south_summer & (latitude > 0.0))
    )
    groups = {
        "summer hemisphere": (summer, low & solstice),
        "winter hemisphere": (winter, low & solstice),
        "equinox": (high & ~solstice, low & ~solstice),
        "all": (high, low),
    }
    rng = np.random.default_rng(intervals.SEED)
    print(
        f"quiet rows (daily Ap < {QUIET_AP:g}) of the family folds: {done.sum()} of {len(rows)}; "
        f"contrast x 100, |lat| >= {HIGH:g} minus |lat| < {LOW:g}, 27-day blocks resampled"
    )
    print(
        "\nby family and season: family | group | rows high / low | "
        + " | ".join(series)
    )
    for fam in FAMILIES:
        here = done & (family == fam)
        for name, (a, b) in groups.items():
            a_, b_ = here & a, here & b
            if not (a_.any() and b_.any()):
                continue
            cells = []
            for values in series.values():
                c = maps.contrast(100.0 * values, a_, b_, block, args.draws, rng)
                cells.append(f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")
            print(f"{fam} | {name} | {a_.sum()} / {b_.sum()} | " + " | ".join(cells))
    print(
        "\nby height, every family pooled: km | rows high / low | " + " | ".join(series)
    )
    for lo, hi in HEIGHTS:
        here = done & (altitude >= lo) & (altitude < hi)
        a_, b_ = here & high, here & low
        if not (a_.any() and b_.any()):
            continue
        cells = []
        for values in series.values():
            c = maps.contrast(100.0 * values, a_, b_, block, args.draws, rng)
            cells.append(f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")
        top = "" if hi >= 1000.0 else f"{hi:g}"
        print(f"{lo:g}-{top} | {a_.sum()} / {b_.sum()} | " + " | ".join(cells))
    f107a = rows.f107a.to_numpy()
    shifts = {
        f"boxcar {smoothing_width(m):g} min": -smoothing_shift(rows, harness.fit, m)
        for m in args.smoothing
    }
    print(
        "\nread after the data, descriptive: family | km | F10.7a | rows high / low | "
        + " | ".join([*series, *shifts])
    )
    for fam in FAMILIES:
        for lo, hi in HEIGHTS:
            for flux, keep in (("< 100", f107a < 100.0), (">= 100", f107a >= 100.0)):
                here = (
                    done & (family == fam) & keep & (altitude >= lo) & (altitude < hi)
                )
                a_, b_ = here & high, here & low
                if a_.sum() < MIN_ROWS or b_.sum() < MIN_ROWS:
                    continue
                cells = []
                for values in series.values():
                    c = maps.contrast(100.0 * values, a_, b_, block, args.draws, rng)
                    cells.append(f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")
                for values in shifts.values():
                    full = np.isfinite(values)
                    c = maps.contrast(
                        100.0 * values, a_ & full, b_ & full, block, args.draws, rng
                    )
                    cells.append(f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")
                top = "" if hi >= 1000.0 else f"{hi:g}"
                print(
                    f"{fam} | {lo:g}-{top} | {flux} | {a_.sum()} / {b_.sum()} | "
                    + " | ".join(cells)
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
