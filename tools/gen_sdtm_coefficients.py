#!/usr/bin/env python3
"""Fit SDTM-1's exospheric-temperature coefficients to measured densities, and write them as Rust.

`plans/16` Ф15.D. The form is fixed and already verified -- `crates/spacelab-propagation/src/sdtm/`,
`derivations/sdtm_bates.mac`, `docs/verification.md` §2.6. What is fitted here is only what that form
leaves free: the `DRIVERS x K` coefficients of `ln(T_inf - T_0)`, the profile shape `sigma`, the storm
kernel's two time constants, and one log-scale nuisance per mission.

## What this is fitted on, and what that does not licence

The corpus is the twelve satellite-months already cached under `data/validation/density/` (git-ignored,
226 MB): GRACE-FO C from the accelerometer and Swarm A from precise orbit determination, for 2019-01,
2021-11, 2022-04, 2023-04, 2024-05 and 2025-06, spanning F10.7a 71 to 181 and including the Gannon
superstorm. TU Delft's, CC BY 4.0 (`docs/provenance.md`, `tudelft-thermosphere`).

**It is a twelve-month fit, not the corpus fit.** Ф15.A's full archive (CHAMP, GRACE A/B, GOCE, Swarm
A/B/C, GRACE-FO C) is blocked: `thermosphere.tudelft.nl` has refused connections on five probes since
2026-09-05. Every number this script prints is therefore labelled by its split and none of them is
Ф15.E's gate, which requires held-out *missions*. Two missions cannot produce a held-out-mission number
no matter how the folds are arranged, and this script does not pretend otherwise.

## Drivers

F10.7 (previous day), its 81-day **trailing** mean, and the 3-hourly `ap` series, all from CelesTrak's
`SW-All.csv`. Trailing and not centred, which is not a detail: the two differ by an rms of 8.99 sfu over
this corpus, 29.1 K of `T_inf` and about 11 % of density at 400 km (`docs/verification.md` §10.9), and
the centred mean is not a quantity an operational caller has. `--centred-mean` is there to reproduce the
literature convention, not to fit on. That is the set Ф15.B admitted over the whole corpus (`docs/verification.md` §10.9,
decision D-15): the EUV channels that would add 24.6-52.0 K of `T_inf` are licence-clean only from
2010-04 and 2017-02, and the two composites that would cover CHAMP state no terms at all.

The storm regressor is the kernel's response to the `ap` history, computed **causally** -- the value at
the end of the interval *before* the sample's own, so that nothing in the regressor is unpublished at
the sample's epoch. That is also exactly what `StormKernel::response` computes when a caller hands it a
history ending at the previous interval, which is what makes the Rust and this script the same model.

## Method

For fixed `sigma` and kernel, `ln rho` depends on the coefficients only through the scalar
`e = X c`, so the Jacobian with respect to `c` is `(d ln rho / d T_inf) (T_inf - T_0) X` -- one row
scaling of a matrix that never changes. That is supplied analytically; the four remaining scalars
(`sigma`, the two time constants, the per-mission scales) go by finite difference. Levenberg-Marquardt
on the rest.

## Run

    uv run --no-project --with numpy --with pandas --with scipy \\
        python tools/gen_sdtm_coefficients.py --degree 2

Nothing is written: this is the twelve-month pilot and the experiments run on it. The shipped
coefficients are the corpus fit's, written by `tools/fit-sdtm-corpus.py --emit`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parent.parent
DENSITY = ROOT / "data" / "validation" / "density"

# The physics, identical to the Rust: Jacchia (1971) 120 km boundary, helium's alpha = -0.38.
BOUNDARY_ALTITUDE = 120_000.0
BOUNDARY_TEMPERATURE = 380.0
REFERENCE_RADIUS = 6_356_766.0
BOUNDARY_GRAVITY = (
    9.80665 * (REFERENCE_RADIUS / (REFERENCE_RADIUS + BOUNDARY_ALTITUDE)) ** 2
)
BOLTZMANN = 1.380649e-23
ATOMIC_MASS = 1.66053906892e-27
SPECIES = np.array(
    [
        [28.0134, 4.0e17, 0.0],
        [31.9988, 7.5e16, 0.0],
        [15.9994, 7.6e16, 0.0],
        [39.948, 1.8e15, 0.0],
        [4.0026, 3.4e13, -0.38],
    ]
)
MASS = SPECIES[:, 0] * ATOMIC_MASS
NUMBER = SPECIES[:, 1]
ALPHA = SPECIES[:, 2]

# Regime edges of the shipped uncertainty table, in the model's own drivers: the 81-day mean flux, and
# the storm kernel's response in units of `ap`. Both are quantities a caller already has to have in
# hand to evaluate the model at all, which is what keeps the uncertainty a function of the model's
# inputs rather than of anything a caller would have to fetch separately. The flux edges split this
# corpus into three comparable thirds (it reaches 69.8 to 165.7, so the literature's 175 is empty
# here); the storm edge is the literature's `Ap` 15, the quiet/active boundary.
# The standard-normal 90 % two-sided quantile, `sqrt(2)*inverse_erf(0.9)`. Maxima at 30 digits:
# `fpprec:30$ bfloat(sqrt(2)*inverse_erf(9/10));` -> 1.64485362695147271486384890798, whose nearest
# double is 1.6448536269514726 (7.9e-17 away). Written with the digit that round-trips to it; the Rust
# carries the same constant.
NORMAL_90 = 1.6448536269514727
UNCERTAINTY_FLUX = (100.0, 150.0)
UNCERTAINTY_STORM = (15.0,)

DRIVERS = 16
REFERENCE_F107 = 150.0
F107_SCALE = 100.0
STORM_SCALE = 100.0
# The log-activity form of the storm regressor, taken from our own `wind_model::disturbance_drivers`
# rather than invented here: `ap` is quasi-exponential in `Kp` (4, 7, 15, 27, 48, 80, 132, 207, 400 for
# Kp = 1..9), so a regressor linear in `ap` gives one storm hour the leverage of a hundred quiet ones,
# and a least-squares fit over a corpus that is mostly quiet puts its slope where the quiet data is.
# `ln(1 + ap/15)` is near-linear in `Kp` over 3..8, which is the scale the storm response is published
# in. The wind model has carried this since it was fitted; the density model did not inherit it.
REFERENCE_AP = 15.0
AP_CADENCE = 3.0 * 3600.0  # the CelesTrak ap series' own cadence
AP60_CADENCE = 3600.0  # the GFZ Hp60/ap60 series' own cadence
# The USGS hourly Dst screen (`docs/verification.md` §10.7): the `variation` product carries fill
# values averaged into hours, multi-week offsets and spikes to 1e4 nT, none of them geomagnetic. An
# hour is kept when Dst4 lies inside the bounds and, where Dst3 exists, the two agree to within
# DST_AGREEMENT; the 99th percentile of |Dst4 - Dst3| over in-range hours is 66 nT. A rejected or
# missing hour holds the last kept value, the zero-order hold the kernel already assumes.
DST_BOUNDS = (-600.0, 100.0)
DST_AGREEMENT = 150.0
# Dst4's median over quiet hours (Hp60 <= 1), 2015-09 to 2026-10: USGS's zero is near zero, Kyoto's
# is not. The disturbance the kernel is driven by is the depression below it, never negative.
DST_QUIET = 5.72
EPOCH = dt.date(1957, 10, 1)  # the first day of SW-All.csv


def geopotential(altitude: np.ndarray) -> np.ndarray:
    return (
        (altitude - BOUNDARY_ALTITUDE)
        * (REFERENCE_RADIUS + BOUNDARY_ALTITUDE)
        / (REFERENCE_RADIUS + altitude)
    )


def basis(
    latitude: np.ndarray, local_solar_time: np.ndarray, degree: int
) -> np.ndarray:
    """`(n, (degree+1)^2)` sun-fixed real spherical harmonics, the recursion `density_calibration` uses.

    Fully normalised `P_nm`, no Condon-Shortley, `[Y00, Y10, Y11c, Y11s, Y20, ...]`, and the longitude
    is the local solar time in radians. Ported from the Rust rather than from a textbook, because what
    has to match is the shipped basis and not a convention.
    """
    x = np.sin(latitude)
    s = np.cos(latitude)
    lam = local_solar_time * (2.0 * np.pi / 24.0)
    p = np.zeros((degree + 1, degree + 1, latitude.size))
    p[0][0] = 1.0
    if degree >= 1:
        p[1][1] = np.sqrt(3.0) * s
        p[1][0] = np.sqrt(3.0) * x
    for n in range(2, degree + 1):
        p[n][n] = np.sqrt((2.0 * n + 1.0) / (2.0 * n)) * s * p[n - 1][n - 1]
        for m in range(n):
            a = np.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
            second = 0.0
            if m <= n - 2:
                b = np.sqrt(
                    (2.0 * n + 1.0)
                    * (n - m - 1.0)
                    * (n + m - 1.0)
                    / ((2.0 * n - 3.0) * (n - m) * (n + m))
                )
                second = b * p[n - 2][m]
            p[n][m] = a * x * p[n - 1][m] - second
    out = []
    for n in range(degree + 1):
        out.append(p[n][0])
        for m in range(1, n + 1):
            out.append(np.sqrt(2.0) * p[n][m] * np.cos(m * lam))
            out.append(np.sqrt(2.0) * p[n][m] * np.sin(m * lam))
    return np.column_stack(out)


def driver_matrix(
    f107: np.ndarray, f107a: np.ndarray, storm: np.ndarray, doy: np.ndarray
) -> np.ndarray:
    """`(n, DRIVERS)`, the same sixteen regressors and the same order as `sdtm::driver_terms`."""
    f = (f107a - REFERENCE_F107) / F107_SCALE
    d = (f107 - f107a) / F107_SCALE
    s = storm / STORM_SCALE
    a = np.log1p(np.maximum(storm, 0.0) / REFERENCE_AP)
    omega = 2.0 * np.pi * doy / 365.25
    cos1, sin1 = np.cos(omega), np.sin(omega)
    cos2, sin2 = np.cos(2.0 * omega), np.sin(2.0 * omega)
    return np.column_stack(
        [
            np.ones_like(f),
            f,
            f * f,
            d,
            s,
            s * s,
            a,
            a * a,
            cos1,
            sin1,
            cos2,
            sin2,
            f * cos1,
            f * sin1,
            f * cos2,
            f * sin2,
        ]
    )


# Per-driver maximum harmonic degree: how much *geography* each global regressor is allowed. -1 drops
# the driver. The full outer product is 16 x 9 = 144 coefficients on twelve satellite-months, and the
# measured cost of that is in `docs/verification.md` §10.10 -- the held-out error of the unrestricted
# fit is dominated by variance, not bias. Restricting it is expressible in the shipped evaluator
# exactly as written, because a dropped (driver, harmonic) pair is a zero in the coefficient array.
CAPS: dict[str, tuple[int, ...]] = {
    # Everything, at the model's own degree: the reference point the restrictions are measured against.
    "full": (9,) * DRIVERS,
    # The mean field and the activity response get the full geography; the daily excess and the
    # quadratic are global corrections; storms and the season get a hemispheric asymmetry (sin phi) and
    # a diurnal one, which is what the literature models carry for them.
    "graded": (9, 9, 0, 0, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0),
    # As `graded`, but activity crossed only with degree 1 and the activity-by-season products dropped.
    "lean": (9, 1, 0, 0, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1),
    # Degree 1 throughout, semiannual global: the smallest set that still has a diurnal bulge.
    "tight": (1, 1, 0, 0, 1, -1, -1, -1, 1, 1, 0, 0, -1, -1, -1, -1),
    # No season at all. Six calendar months cannot identify an annual and a semiannual harmonic that a
    # caller will evaluate on the other six, and this is the set that does not try.
    "aseasonal": (9, 9, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    # The three sets below are the storm-response candidates of §10.15. The linear-only form
    # under-reaches the measured peak enhancement by about half *in sample* and also on a fit that sees
    # nothing but storms, so neither the corpus nor the loss is what caps it -- the response has
    # curvature the regressor does not.
    #
    # `lean` plus the quadratic in `storm/100`, global.
    "storm2": (9, 1, 0, 0, 1, 0, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1),
    # And the same with the quadratic given the storm term's own geography.
    "storm2-geo": (9, 1, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1),
    # `lean` with the storm response in the log-activity form instead: `[a, a^2]` with
    # `a = ln(1 + storm/15)`, which is what `wind_model::disturbance_drivers` already carries.
    "storm-log": (9, 1, 0, 0, -1, -1, 1, 0, 1, 1, 1, 1, -1, -1, -1, -1),
    # And the log scale *without* the curvature, which is what separates the two effects: `storm2`
    # shows that a quadratic in the linear index destroys the held-out extrapolation (Gannon R falls to
    # 0.29 and 0.04), so the question this set answers is whether the log scale alone is enough.
    "storm-log1": (9, 1, 0, 0, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1),
    # `storm-log1` with the storm term given degree 2, so `Y20` -- the pole-to-equator gradient that
    # auroral heating puts into a storm -- can carry it (Ф23.B, the minor-storm overshoot). It does not:
    # 228 of 630 storm peaks within 25 % against `storm-log1`'s 246 (`docs/verification.md` §10.27).
    "storm-log1-geo": (9, 1, 0, 0, -1, -1, 2, -1, 1, 1, 1, 1, -1, -1, -1, -1),
    # `storm-log1` with the linear `storm/100` beside `a`, global: a term that keeps growing where the
    # log has flattened, for the large storms the log-plus-cooling form under-reaches at every flux.
    # Neutral on its own; with the storm state scaled by `(F10.7a/150)^kappa` it takes the eight
    # storms' mean peak error from 30.5 to 26.7 % (§10.27).
    "storm-log1-lin": (9, 1, 0, 0, 0, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1),
}


def harmonic_mask(caps: tuple[int, ...], k: int) -> np.ndarray:
    """Which of the `DRIVERS*k` (driver, harmonic) pairs a cap set admits, flat and driver-major.

    `basis` emits the harmonics in degree blocks, so "at most degree d" is the first `(d+1)**2` columns.
    """
    out = np.zeros((DRIVERS, k), dtype=bool)
    for g, cap in enumerate(caps):
        if cap >= 0:
            out[g, : min((cap + 1) ** 2, k)] = True
    return out.reshape(-1)


def profile(
    t_inf: np.ndarray, sigma: float, zeta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(ln rho, d ln rho / d T_inf)` of the shipped profile, both analytic."""
    decay = np.exp(-sigma * zeta)
    t = t_inf - (t_inf - BOUNDARY_TEMPERATURE) * decay
    integral = (sigma * zeta + np.log(t / BOUNDARY_TEMPERATURE)) / (sigma * t_inf)
    ratio = BOUNDARY_TEMPERATURE / t
    barometric = MASS * BOUNDARY_GRAVITY / BOLTZMANN
    partial = (
        MASS
        * NUMBER
        * ratio[:, None] ** (1.0 + ALPHA)
        * np.exp(-np.outer(integral, barometric))
    )
    rho = partial.sum(axis=1)
    # dT/dT_inf = 1 - e^{-sigma zeta}; dI1/dT_inf follows from I1's own closed form.
    dt_dinf = 1.0 - decay
    di_dinf = (dt_dinf / t) / (sigma * t_inf) - integral / t_inf
    per_species = -(1.0 + ALPHA) * (dt_dinf / t)[:, None] - np.outer(
        di_dinf, barometric
    )
    return np.log(rho), (partial * per_species).sum(axis=1) / rho


def storm_grid(
    ap: np.ndarray, onset: float, recovery: float, cadence: float
) -> np.ndarray:
    """The kernel's response on the driver's own grid, by the two exact recursions `StormKernel` uses.

    `y <- a + (y - a) e^{-h/tau}` is `y[i] = (1-d) a[i] + d y[i-1]`, a first-order IIR filter, so it is
    `lfilter` and not a Python loop: identical arithmetic to 5e-16 relative, measured, and two hundred
    times the speed, which matters because the fit rebuilds this on every Jacobian.

    `cadence` is the series' step in seconds, not a constant, because the same kernel runs on
    CelesTrak's 3-hourly `ap` and on GFZ's hourly `ap60` and the step is what distinguishes them.
    """
    smoothed = []
    for tau in (onset, recovery):
        decay = np.exp(-cadence / tau)
        numerator, denominator = [1.0 - decay], [1.0, -decay]
        state = signal.lfiltic(numerator, denominator, y=[ap[0]], x=[ap[0]])
        value, _ = signal.lfilter(numerator, denominator, ap, zi=state)
        smoothed.append(value)
    return (recovery * smoothed[1] - onset * smoothed[0]) / (recovery - onset)


def load_ap60() -> tuple[np.ndarray, pd.Timestamp]:
    """GFZ's hourly `ap60` series and its first interval's start.

    The file is a contiguous hourly grid from 1985-01-01 with no gaps and no missing values over this
    corpus (checked on load), so the sample's interval is an index and needs no search.
    """
    path = DENSITY / "Hp60_ap60_complete_series.txt"
    if not path.exists():
        sys.exit(f"no {path.name}; run tools/fetch-density-data.sh")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 8],
        engine="c",
        names=["year", "month", "day", "hour", "ap60"],
    )
    start = pd.to_datetime(frame[["year", "month", "day"]]) + pd.to_timedelta(
        frame.hour, unit="h"
    )
    step = start.diff().iloc[1:].unique()
    if step.size != 1 or step[0] != pd.Timedelta(hours=1):
        sys.exit(f"{path.name} is not a contiguous hourly grid: steps {step}")
    if (frame.ap60 < 0).any():
        sys.exit(f"{path.name} has missing ap60 (-1) inside the span")
    return frame.ap60.to_numpy().astype(float), start.iloc[0]


def load_dst() -> tuple[np.ndarray, pd.Timestamp]:
    """The USGS hourly Dst4 depression `max(0, DST_QUIET - Dst4)`, screened, and its first hour.

    `tools/fetch-usgs-dst.sh` writes one row per hour, stamped at the hour's centre; the grid is
    rebuilt from the stamps and checked contiguous, so the sample's interval is an index.
    """
    path = DENSITY / "usgs_dst_hourly.txt"
    if not path.exists():
        sys.exit(f"no {path.name}; run tools/fetch-usgs-dst.sh")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["t", "dst4", "dst3"],
        na_values=["null"],
        engine="c",
    )
    start = pd.to_datetime(frame.t.str.slice(0, 13) + ":00:00")
    step = start.diff().iloc[1:].unique()
    if step.size != 1 or step[0] != pd.Timedelta(hours=1):
        sys.exit(f"{path.name} is not a contiguous hourly grid: steps {step}")
    dst4, dst3 = frame.dst4.to_numpy(), frame.dst3.to_numpy()
    keep = (dst4 > DST_BOUNDS[0]) & (dst4 < DST_BOUNDS[1])
    keep &= ~(np.abs(dst4 - dst3) > DST_AGREEMENT)
    held = pd.Series(np.where(keep, dst4, np.nan)).ffill().fillna(DST_QUIET).to_numpy()
    print(
        f"{path.name}: {keep.sum()} of {keep.size} hours kept by the screen", flush=True
    )
    return np.maximum(DST_QUIET - held, 0.0), start.iloc[0]


def load(trailing_mean: bool = True, storm_driver: str = "ap") -> pd.DataFrame:
    files = sorted(glob.glob(f"{DENSITY}/GC_DNS_ACC_*_v02c.txt")) + sorted(
        glob.glob(f"{DENSITY}/SA_DNS_POD_*_v02.txt")
    )
    if not files:
        sys.exit(f"no density files under {DENSITY}; run tools/fetch-density-data.sh")
    frames = []
    for path in files:
        mission = "GRACE-FO-C" if "/GC_" in path else "SWARM-A"
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            usecols=range(9),
            engine="c",
            names=[
                "date",
                "time",
                "tsys",
                "altitude",
                "longitude",
                "latitude",
                "lst",
                "arglat",
                "rho",
            ],
        )
        frame = frame[frame.rho > 0]
        frame = frame.iloc[:: 12 if mission.startswith("GRACE") else 4].copy()
        frame["mission"] = mission
        frame["month"] = path.split("_v0")[0][-7:]
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows["t"] = pd.to_datetime(rows.date + " " + rows.time)

    mean_column = "F10.7_OBS_LAST81" if trailing_mean else "F10.7_OBS_CENTER81"
    flux = pd.read_csv(
        DENSITY / "SW-All.csv",
        usecols=["DATE", "F10.7_OBS", mean_column] + [f"AP{i}" for i in range(1, 9)],
    )
    flux["DATE"] = pd.to_datetime(flux.DATE)
    flux = flux.set_index("DATE").sort_index()
    day = rows.t.dt.normalize()
    rows["f107"] = flux.loc[day - pd.Timedelta(days=1), "F10.7_OBS"].to_numpy()
    rows["f107a"] = flux.loc[day, mean_column].to_numpy()

    # One 3-hourly ap series for the whole file. Storms are *detected* on this one whatever drives
    # the kernel, because Bruinsma & Laurens define a storm by the 3-hourly ap reaching 80 and moving
    # the detection grid would change which storms are found.
    ap = flux[[f"AP{i}" for i in range(1, 9)]].to_numpy().reshape(-1).astype(float)
    ap = np.where(np.isfinite(ap), ap, 0.0)
    first = flux.index[0].date()
    offset = (
        np.array([(d.date() - first).days for d in day]) * 8
        + rows.t.dt.hour.to_numpy() // 3
    )
    rows.attrs["ap"] = ap
    rows.attrs["ap_epoch"] = flux.index[0]

    # And the kernel's own driver, with each row's *previous* interval in it.
    if storm_driver in ("ap60", "dst"):
        driver, driver_epoch = load_ap60() if storm_driver == "ap60" else load_dst()
        elapsed = (rows.t - driver_epoch).dt.total_seconds().to_numpy()
        offset = np.floor(elapsed / AP60_CADENCE).astype(int)
        if offset.min() < 1 or offset.max() >= driver.size:
            sys.exit(f"the {storm_driver} series does not span the corpus")
        rows.attrs["driver_cadence"] = AP60_CADENCE
    else:
        driver, driver_epoch = ap, flux.index[0]
        rows.attrs["driver_cadence"] = AP_CADENCE
    rows["interval"] = np.maximum(offset - 1, 0)
    rows.attrs["driver"] = driver
    rows.attrs["driver_epoch"] = driver_epoch
    return rows


def fit_arguments(rows: pd.DataFrame, degree: int) -> dict:
    harmonics = basis(np.deg2rad(rows.latitude.to_numpy()), rows.lst.to_numpy(), degree)
    return {
        "zeta": geopotential(rows.altitude.to_numpy()),
        "harmonics": harmonics,
        "f107": rows.f107.to_numpy(),
        "f107a": rows.f107a.to_numpy(),
        "doy": rows.t.dt.dayofyear.to_numpy() + rows.t.dt.hour.to_numpy() / 24.0,
        "interval": rows.interval.to_numpy(),
        "ap": rows.attrs["driver"],
        "cadence": rows.attrs["driver_cadence"],
        "mission": (rows.mission.to_numpy() == "SWARM-A").astype(float),
        "ln_rho": np.log(rows.rho.to_numpy()),
        "k": harmonics.shape[1],
    }


@dataclasses.dataclass(frozen=True)
class Fitted:
    """What a fit returns, with names instead of offsets."""

    coefficients: np.ndarray
    sigma: float
    onset: float
    recovery: float
    mission: float
    boundary: float
    """Log scale on the total boundary density.

    Jacchia's (1971) 120 km number densities are literature values, and the total they sum to is not
    a fitted quantity anywhere in this model: without this one parameter the fit can only match the
    measured level by distorting `T_inf`, which then distorts the altitude dependence as well. It is
    bounded to a factor of two either way, because past that it is no longer a correction to a
    published boundary but a free gain.
    """


class Model:
    """`ln rho` of the shipped form, as a function of the packed parameter vector.

    Packing: the admitted coefficients (see `CAPS`), then `ln(onset/h)` and `ln((recovery - onset)/h)`
    if the kernel is being fitted, then the second mission's log scale, then the boundary's log scale if
    it is being fitted, and `sigma/1e-5` last if it is. A slot held fixed is absent from the vector, not
    pinned inside it, so a fixed parameter costs the optimiser no flat direction to search -- and a
    bound interval of width zero is not a way to pin one, because `least_squares` refuses it.

    Four choices in that sentence are the whole difference between a fit that generalises and one
    that does not, and each was measured rather than assumed:

    * **the kernel goes in as logarithms of onset and of the gap**, which makes positivity and the
      separation `recovery > onset` structural. Fitting the two constants directly walks into
      `onset -> 0`, where `exp(-h/tau)` underflows and the storm regressor becomes noise -- measured,
      as a first-fold held-out MAPE of 290 % against the pilot's 32 %;
    * **`sigma` is held fixed by default.** Both satellites here fly between 423 and 537 km, and the
      Bates shape parameter is what a single altitude band sees worst. Unrestricted, the free fit moved
      it to 0.0074 /km and traded it against `T_inf`, whose range then reached 2280 K; under the
      restriction below it moves only to 0.0251 /km and costs 0.2 points of held-out MAPE by
      satellite-month and 1.1 by calendar month, so fitting it buys nothing. The corpus of Ф15.A has
      CHAMP at 300-450 km and GOCE at 250, and that is what would identify it;
    * **not every driver gets every harmonic.** The full outer product is 16 x 9 = 144 coefficients on
      twelve satellite-months and it is a variance disaster: 16.1 % in sample against 3823 % held out
      by satellite-month, measured. `CAPS` is the restriction, and `--sweep` is what chooses between
      the cap sets on the held-out splits rather than on the fit;
    * **ridge on everything but the intercept**, scaled by each column's own rms so the penalty is on
      the *effect* of a coefficient and not on its units, and by `sqrt(n)` so that it means the same
      thing on every fold -- without which it means nothing at all, as an identical held-out MAPE to
      four digits across `ridge` 0 to 0.1.
    """

    def __init__(
        self,
        arg: dict,
        ridge: float = 0.0,
        sigma: float | None = 2.0e-5,
        caps: tuple[int, ...] = CAPS["full"],
        boundary: bool = True,
        kernel: tuple[float, float] | None = None,
    ):
        self.a = arg
        self.k = arg["k"]
        self.caps = caps
        self.mask = harmonic_mask(caps, self.k)
        self.n_c = int(self.mask.sum())
        self.fixed_sigma = sigma
        self.fixed_kernel = kernel
        # Named slots, because an offset arithmetic bug here is silent: it fits a different model.
        # A slot of -1 is a parameter this model holds fixed rather than fits.
        self.i_onset = -1 if kernel else self.n_c
        self.i_gap = -1 if kernel else self.n_c + 1
        tail = self.n_c + (0 if kernel else 2)
        self.i_mission = tail
        tail += 1
        self.i_boundary = tail if boundary else -1
        tail += int(boundary)
        self.i_sigma = -1 if sigma is not None else tail
        self.size = tail + int(sigma is None)
        self.ridge = ridge
        self.penalty = np.zeros(self.n_c)
        if ridge > 0.0:
            reference = kernel or (3.0 * 3600.0, 18.0 * 3600.0)
            rms = np.sqrt(np.mean(self.design(*reference) ** 2, axis=0))
            self.penalty = np.sqrt(ridge) * rms
            self.penalty[0] = (
                0.0  # the intercept carries the mid-cycle temperature, not structure
            )

    def unpack(self, p: np.ndarray) -> Fitted:
        if self.fixed_kernel is None:
            onset = np.exp(p[self.i_onset]) * 3600.0
            recovery = onset + np.exp(p[self.i_gap]) * 3600.0
        else:
            onset, recovery = self.fixed_kernel
        return Fitted(
            coefficients=p[: self.n_c],
            sigma=self.fixed_sigma if self.i_sigma < 0 else p[self.i_sigma] * 1.0e-5,
            onset=onset,
            recovery=recovery,
            mission=p[self.i_mission],
            boundary=0.0 if self.i_boundary < 0 else p[self.i_boundary],
        )

    def design(self, onset: float, recovery: float) -> np.ndarray:
        storm = storm_grid(self.a["ap"], onset, recovery, self.a["cadence"])[
            self.a["interval"]
        ]
        drivers = driver_matrix(self.a["f107"], self.a["f107a"], storm, self.a["doy"])
        product = (drivers[:, :, None] * self.a["harmonics"][:, None, :]).reshape(
            -1, DRIVERS * self.k
        )
        return product[:, self.mask]

    def scatter(self, coefficients: np.ndarray) -> np.ndarray:
        """The fitted coefficients back in the `DRIVERS x K` grid the evaluator indexes."""
        grid = np.zeros(DRIVERS * self.k)
        grid[self.mask] = coefficients
        return grid.reshape(DRIVERS, self.k)

    def evaluate(
        self, p: np.ndarray, index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = self.unpack(p)
        design = self.design(v.onset, v.recovery)[index]
        exponent = design @ v.coefficients
        t_inf = BOUNDARY_TEMPERATURE + np.exp(np.clip(exponent, -50.0, 50.0))
        ln_rho, d_ln_d_inf = profile(t_inf, v.sigma, self.a["zeta"][index])
        return (
            ln_rho + v.boundary + v.mission * self.a["mission"][index],
            design,
            d_ln_d_inf * (t_inf - BOUNDARY_TEMPERATURE),
        )

    def score(self, p: np.ndarray, index: np.ndarray) -> np.ndarray:
        """`ln(model / measured)` on those rows -- what every reported number is computed from."""
        return self.evaluate(p, index)[0] - self.a["ln_rho"][index]

    def ridge_rows(self, n: int) -> np.ndarray:
        """The penalty per coefficient on a fold of `n` rows.

        Scaled by `sqrt(n)` because the data term is a *sum* over rows while the penalty is not: without
        it `ridge` would mean something different on every fold and, at these sizes, nothing at all --
        measured, as an identical held-out MAPE to four digits across `ridge` 0 to 0.1.
        """
        return np.sqrt(n) * self.penalty

    def residual(self, p: np.ndarray, index: np.ndarray) -> np.ndarray:
        """What the optimiser sees: [`Self::score`], with the ridge rows appended when there is one."""
        data = self.score(p, index)
        if self.ridge <= 0.0:
            return data
        return np.concatenate([data, self.ridge_rows(index.size) * p[: self.n_c]])

    def jacobian(self, p: np.ndarray, index: np.ndarray) -> np.ndarray:
        _, design, chain = self.evaluate(p, index)
        rows = index.size + (self.n_c if self.ridge > 0.0 else 0)
        out = np.zeros((rows, p.size))
        out[: index.size, : self.n_c] = design * chain[:, None]
        step = 1.0e-4
        base = self.residual(p, index)
        for j in range(self.n_c, p.size):
            shifted = p.copy()
            shifted[j] += step
            out[:, j] = (self.residual(shifted, index) - base) / step
        if self.ridge > 0.0:
            out[index.size :, : self.n_c] = np.diag(self.ridge_rows(index.size))
        return out


def fit(model: Model, index: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    if start is None:
        start = np.zeros(model.size)
        start[0] = np.log(700.0)
        if model.i_onset >= 0:
            start[model.i_onset] = np.log(3.0)  # onset 3 h
            start[model.i_gap] = np.log(15.0)  # recovery 18 h
        if model.i_sigma >= 0:
            start[model.i_sigma] = 2.0
    lower = np.full(start.size, -np.inf)
    upper = np.full(start.size, np.inf)
    # The floor on the onset constant is numerical, not physical: the response is exact for a
    # piecewise-constant input at any tau, so the only thing that breaks is `exp(-cadence/tau)`
    # underflowing, which for an hourly driver happens near tau = 5 s. One minute is four orders above
    # that, and it is deliberately *not* binding -- the 3-hourly fit lands at 0.94 h. A floor that the
    # search sits on is a wall the report would read as a preference, which is what the previous 0.25 h
    # became the moment the driver went hourly.
    if model.i_onset >= 0:
        lower[model.i_onset], upper[model.i_onset] = np.log(1.0 / 60.0), np.log(12.0)
        lower[model.i_gap], upper[model.i_gap] = np.log(1.0), np.log(72.0)
    # The mission scale is what two *measurement methods* disagree by -- an accelerometer against
    # precise orbit determination, published at a few per cent (Siemes et al. 2023) -- so a factor of
    # 1.35 either way is already generous. It is bounded because it is otherwise unidentified whenever a
    # whole mission is held out: with one mission in the training set the indicator is constant, the
    # scale becomes collinear with the intercept, and the held-out residual is then the nuisance rather
    # than the model. Measured, as a transfer MAPE of 1e290 % before the bound.
    lower[model.i_mission], upper[model.i_mission] = -0.3, 0.3
    if model.i_boundary >= 0:
        # A correction to a literature number, not a free gain: +/- a factor of two.
        lower[model.i_boundary], upper[model.i_boundary] = -0.7, 0.7
    if model.i_sigma >= 0:
        lower[model.i_sigma], upper[model.i_sigma] = 0.5, 6.0
    out = least_squares(
        model.residual,
        start,
        jac=model.jacobian,
        args=(index,),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=80,
        verbose=0,
    )
    return out.x


def regime(f107a: np.ndarray, storm: np.ndarray) -> np.ndarray:
    """Which cell of the uncertainty table each row falls in, flattened `flux * storm + storm`."""
    flux = np.searchsorted(UNCERTAINTY_FLUX, f107a, side="right")
    disturbed = np.searchsorted(UNCERTAINTY_STORM, storm, side="right")
    return flux * (len(UNCERTAINTY_STORM) + 1) + disturbed


def uncertainty(
    residual: np.ndarray, f107a: np.ndarray, storm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(sigma, n)` of `ln(model/measured)` per regime cell, from a *held-out* residual.

    The spread is taken about zero rather than about the cell's own mean: a caller who is handed an
    interval wants it to cover the measurement, and a cell whose residual is biased has that bias as
    part of its error. `sqrt(mean(r**2))` is therefore the right statistic and `std` is not.
    """
    cells = regime(f107a, storm)
    count = (len(UNCERTAINTY_FLUX) + 1) * (len(UNCERTAINTY_STORM) + 1)
    sigma = np.zeros(count)
    n = np.zeros(count, dtype=int)
    for c in range(count):
        rows = residual[cells == c]
        n[c] = rows.size
        sigma[c] = np.sqrt(np.mean(rows**2)) if rows.size else np.nan
    return sigma, n


STORM_AP = 80.0
"""`ap` at or above which an interval is storm time -- `Kp = 6`, Bruinsma & Laurens (2024)."""

PHASES = ((-30.0, -18.0), (-18.0, 0.0), (0.0, 36.0), (36.0, 48.0))
"""Storm phases in hours from `t0` for a single-peak storm: pre-storm, onset, main, post-storm."""


def storm_onsets(ap: np.ndarray, epoch: pd.Timestamp) -> list[dict]:
    """Every storm in a 3-hourly `ap` series, on the protocol of Bruinsma & Laurens (2024) 2.

    `t0` is the *first peak* of a contiguous run at or above `STORM_AP` -- not the first crossing,
    which on a storm that keeps rising would sit hours early and drag every phase boundary with it.
    A second interval above the threshold between 9 h and 48 h after `t0` makes the storm
    multiple-peaked and extends phase 3 to 36 h past the last such interval.

    Independent of the same logic in `crates/spacelab-propagation/tests/measured_density.rs`, and the
    two are meant to agree on `t0` to the interval -- which is the cross-check, since everything the
    assessment reports hangs off it.
    """
    out: list[dict] = []
    i = 0
    while i < ap.size:
        if ap[i] < STORM_AP:
            i += 1
            continue
        start = i
        while i < ap.size and ap[i] >= STORM_AP:
            i += 1
        run = ap[start:i]
        peak = next(
            k
            for k in range(run.size)
            if (k == 0 or run[k] >= run[k - 1])
            and (k + 1 == run.size or run[k] >= run[k + 1])
        )
        index = start + peak
        t0 = epoch + pd.Timedelta(hours=3.0 * (index + 0.5))
        later = [
            j
            for j in range(ap.size)
            if ap[j] >= STORM_AP and 9.0 <= 3.0 * (j - index) <= 48.0
        ]
        phase3_end = 36.0 if not later else 3.0 * (later[-1] - index) + 36.0
        out.append(
            {
                "t0": t0,
                "index": index,
                "peak_ap": float(run.max()),
                "multiple_peak": bool(later),
                "phase3_end": phase3_end,
            }
        )
        while i < ap.size and 3.0 * (i - index) <= phase3_end:
            i += 1
    return out


def storm_windows(storm: dict) -> list[tuple[float, float]]:
    """The four phase windows of one storm, in hours from `t0`."""
    return [
        PHASES[0],
        PHASES[1],
        (0.0, storm["phase3_end"]),
        (storm["phase3_end"], storm["phase3_end"] + 12.0),
    ]


def orbit_bins(
    hours: np.ndarray, measured: np.ndarray, modelled: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Density against time in bins of one nominal orbit, 90 minutes.

    Along one orbit the density swings by a factor of two with latitude and local solar time, so the
    maximum of the raw series says where the satellite was rather than when the storm peaked. The bin
    centre is accurate to +/-45 min, well inside the +/-3 h the verdict asks for. Bins with fewer than
    four samples are dropped -- at this thinning a full orbit is about six.
    """
    key = np.floor(hours / 1.5).astype(int)
    counts = np.array([(key == k).sum() for k in np.unique(key)])
    # A bin holding part of an orbit says where the satellite was rather than when the storm peaked,
    # so the partial ones go -- against half the median bin rather than a fixed count, because the two
    # products' cadences differ and a data gap leaves ragged ends either way.
    floor = np.median(counts) / 2.0
    out = []
    for k, n in zip(np.unique(key), counts, strict=True):
        here = key == k
        if n >= floor:
            out.append(
                (
                    (k + 0.5) * 1.5,
                    float(measured[here].mean()),
                    float(modelled[here].mean()),
                )
            )
    if not out:
        return np.empty(0), np.empty(0), np.empty(0)
    table = np.array(out)
    return table[:, 0], table[:, 1], table[:, 2]


def recovery_half_time(
    hours: np.ndarray, density: np.ndarray, quiet: float, peak: int
) -> float:
    """Hours from the density peak to the first time the storm *enhancement* has halved.

    The enhancement is `rho - quiet`, with `quiet` the pre-storm level, so this is a property of what
    the storm added rather than of the absolute level. A least-squares e-folding time over the whole
    main phase was tried first and is not an estimator at all: these storms re-intensify, the fit is
    then over a series that is not a decay, and the resulting `tau` ranged over four orders of
    magnitude with sign changes. The first half-crossing is insensitive to what happens after it.

    `nan` when the enhancement never halves inside the window -- which is a fact about the storm and
    is reported rather than filled in.
    """
    target = quiet + (density[peak] - quiet) / 2.0
    after = np.where((np.arange(density.size) > peak) & (density <= target))[0]
    return float(hours[after[0]] - hours[peak]) if after.size else float("nan")


def storm_window_rows(centred_mean: bool, storm_driver: str) -> pd.DataFrame | None:
    """The corpus at its own cadence, restricted to the storm windows.

    `load` thins the 10 s and 30 s products to 120 s, which is about 45 samples to an orbit -- enough
    to average one. The `--thin` of the fit is *not* applied: a fit needs decorrelated rows and an
    orbit average needs all of them, and those are different requirements on the same data.
    """
    rows = load(trailing_mean=not centred_mean, storm_driver=storm_driver)
    time = rows.t.to_numpy()
    inside = np.zeros(len(rows), dtype=bool)
    for storm in storm_onsets(rows.attrs["ap"], rows.attrs["ap_epoch"]):
        hours = (time - np.datetime64(storm["t0"])) / np.timedelta64(1, "h")
        inside |= (hours >= PHASES[0][0]) & (hours < storm["phase3_end"] + 12.0)
    if not inside.any():
        return None
    out = rows[inside].reset_index(drop=True)
    out.attrs = rows.attrs
    return out


def storm_assessment(
    rows: pd.DataFrame,
    residual: np.ndarray,
    label: str,
    missions: tuple[str, ...] = ("GRACE-FO-C", "SWARM-A"),
) -> None:
    """The storm statistics of Bruinsma & Laurens (2024) Eq. (1) on a *held-out* residual.

    `residual` is `ln(rho_model/rho_measured)`, so the paper's observed-to-computed ratio is
    `exp(-residual)` and its debiasing -- a scale factor fitted on phase 1 and applied to the model in
    every phase -- is a constant shift of the residual. The scatter is therefore unchanged by the
    debiasing and the mean ratio is what moves, which is the point of reporting both.
    """
    storms = storm_onsets(rows.attrs["ap"], rows.attrs["ap_epoch"])
    time = rows.t.to_numpy()
    print(f"\n== storm assessment on the {label} residual ==")
    print(
        "t0                   peak ap  type  mission      "
        "n   <O/C>   StD %     R   P2 <O/C>  P3 <O/C>  P4 <O/C>"
        "  onset h  peak M/O  d(half) h"
    )
    seen = 0
    collected: dict[str, list[float]] = {
        k: [] for k in ("mean_ratio", "scatter", "R", "onset", "peak", "half")
    }
    for storm in storms:
        hours = (time - np.datetime64(storm["t0"])) / np.timedelta64(1, "h")
        windows = storm_windows(storm)
        for mission in missions:
            is_mission = rows.mission.to_numpy() == mission
            phases = [(hours >= a) & (hours < b) & is_mission for a, b in windows]
            if any(p.sum() == 0 for p in phases):
                continue
            seen += 1
            # The debiasing scale factor, in the log: the mean residual over the pre-storm window.
            shift = float(residual[phases[0]].mean())
            inside = phases[0] | phases[1] | phases[2] | phases[3]
            log_ratio = -(residual - shift)
            keep = inside & (np.abs(log_ratio) < np.log(10.0))
            mean_ratio = float(np.exp(log_ratio[keep].mean()))
            scatter = 100.0 * float(np.std(log_ratio[keep]))
            o = rows.rho.to_numpy()[keep]
            c = o * np.exp(residual[keep] - shift)
            correlation = float(np.corrcoef(o, c)[0, 1]) if o.size > 1 else float("nan")
            per_phase = [
                float(np.exp(log_ratio[p & keep].mean()))
                if (p & keep).any()
                else float("nan")
                for p in phases[1:]
            ]
            # Timing, on the orbit-averaged series. The pre-storm phase is binned too, because the
            # quiet level it fixes is what the enhancement below is measured against.
            window = (phases[0] | phases[1] | phases[2]) & keep
            centre, measured, modelled = orbit_bins(
                hours[window],
                rows.rho.to_numpy()[window],
                rows.rho.to_numpy()[window] * np.exp(residual[window] - shift),
            )
            quiet_bins = centre < PHASES[0][1]
            storm_bins = centre >= PHASES[1][0]
            if not quiet_bins.any() or storm_bins.sum() < 4:
                onset = peak_ratio = delta_tau = float("nan")
            else:
                quiet = (
                    float(measured[quiet_bins].mean()),
                    float(modelled[quiet_bins].mean()),
                )
                t, o, c = centre[storm_bins], measured[storm_bins], modelled[storm_bins]
                i_measured, i_modelled = int(o.argmax()), int(c.argmax())
                onset = t[i_modelled] - t[i_measured]
                peak_ratio = (c[i_modelled] - quiet[1]) / (o[i_measured] - quiet[0])
                delta_tau = recovery_half_time(
                    t, c, quiet[1], i_modelled
                ) - recovery_half_time(t, o, quiet[0], i_measured)
            for key, value in (
                ("mean_ratio", mean_ratio),
                ("scatter", scatter),
                ("R", correlation),
                ("onset", onset),
                ("peak", peak_ratio),
                ("half", delta_tau),
            ):
                collected[key].append(value)
            print(
                f"{storm['t0']:%Y-%m-%d %H:%M}  {storm['peak_ap']:7.0f}  "
                f"{'MP' if storm['multiple_peak'] else 'SP'}    {mission:<12}"
                f"{int(keep.sum()):6d}{mean_ratio:8.3f}{scatter:8.1f}{correlation:7.3f}"
                + "".join(f"{x:10.3f}" for x in per_phase)
                + f"{onset:9.1f}{peak_ratio:10.3f}{delta_tau:10.1f}"
            )
    if seen == 0:
        print("no storm in the corpus has all four phases covered by either mission")
        return
    # The aggregate, computed here rather than by hand off the table above: the mean of the per-storm
    # mean ratios in log space and the spread of those means, which is the pair the literature reports,
    # then the three timing numbers the verdict is stated in.
    ratios = np.array(collected["mean_ratio"])
    centre = float(np.exp(np.log(ratios).mean()))
    spread = 100.0 * float(np.std(np.log(ratios)))
    print(
        f"{seen} comparisons: <O/C> {centre:.3f}  StD<O/C> {spread:.1f}%  "
        f"mean StD {np.mean(collected['scatter']):.1f}%  mean R {np.mean(collected['R']):.3f}"
    )
    for name, values, bound in (
        ("|onset|", np.abs(collected["onset"]), 3.0),
        ("|peak M/O - 1|", np.abs(np.array(collected["peak"]) - 1.0), 0.25),
        ("|d(half)|", np.abs(collected["half"]), 6.0),
    ):
        good = np.isfinite(values)
        print(
            f"  {name:<16} mean {np.mean(values[good]):7.2f}  worst {np.max(values[good]):7.2f}  "
            f"within {bound:g}: {int((values[good] <= bound).sum())} of {int(good.sum())}"
        )


def mape(residual: np.ndarray) -> float:
    return 100.0 * float(np.mean(np.abs(np.exp(residual) - 1.0)))


def literal(value: float) -> str:
    """A Rust `f64` literal that parses back to exactly this double.

    `repr` is the shortest decimal that round-trips, and Rust's `str::parse` is correctly rounded, so
    the two sides hold the same bits. Printing ten digits instead would move the evaluated density by
    more than `sdtm_reference.rs` allows -- the test compares against this fit's own values, not
    against a physical tolerance -- and the tree's rule is that a generated constant is never truncated.
    """
    return repr(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degree", type=int, default=2, help="spherical-harmonic degree (K = (d+1)^2)"
    )
    parser.add_argument(
        "--thin", type=int, default=2, help="keep every n-th row of the loaded sample"
    )
    parser.add_argument(
        "--ridge", type=float, default=1.0e-3, help="ridge strength on the coefficients"
    )
    parser.add_argument(
        "--fit-sigma",
        action="store_true",
        help="fit the Bates shape parameter instead of holding it at 2e-5 /m",
    )
    parser.add_argument(
        "--centred-mean",
        action="store_true",
        help="fit on the 81-day *centred* F10.7 mean (NRLMSISE-00's convention) instead of "
        "the trailing one, which is the only one an operational caller has",
    )
    parser.add_argument(
        "--caps",
        choices=sorted(CAPS),
        default="graded",
        help="which (driver, harmonic) pairs to fit; see CAPS",
    )
    parser.add_argument(
        "--fix-boundary",
        action="store_true",
        help="hold the boundary density at Jacchia's published total instead of fitting a scale on it",
    )
    parser.add_argument(
        "--ap60",
        action="store_true",
        help="drive the storm kernel from GFZ's hourly ap60 instead of CelesTrak's 3-hourly ap; "
        "storms are still *detected* on the 3-hourly ap, which is how the protocol defines them",
    )
    parser.add_argument(
        "--pin-kernel",
        metavar="ONSET_H,RECOVERY_H",
        help="hold the storm kernel's two time constants at these hours instead of fitting them, "
        "so a ladder of memories can be scored against each other with nothing else moving",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="report LOSMO over a ladder of ridge strengths and cap sets, then stop",
    )
    parser.add_argument(
        "--dst",
        action="store_true",
        help="drive the storm kernel from the USGS hourly Dst4 depression below its quiet level, "
        "screened (DST_BOUNDS, DST_AGREEMENT); storms are still detected on the 3-hourly ap",
    )
    argument = parser.parse_args()
    if argument.ap60 and argument.dst:
        sys.exit("--ap60 and --dst choose the same driver; give one")
    driver = "ap60" if argument.ap60 else "dst" if argument.dst else "ap"

    began = time.time()
    rows = (
        load(trailing_mean=not argument.centred_mean, storm_driver=driver)
        .iloc[:: argument.thin]
        .reset_index(drop=True)
    )
    counts = rows.groupby(["mission", "month"]).size()
    print(
        f"{len(rows)} rows from {len(counts)} satellite-months in {time.time() - began:.1f}s"
    )
    arg = fit_arguments(rows, argument.degree)
    arg["_altitude"] = rows.altitude.to_numpy()
    arg["_latitude"] = np.deg2rad(rows.latitude.to_numpy())
    arg["_lst"] = rows.lst.to_numpy()
    sigma = None if argument.fit_sigma else 2.0e-5
    pinned = None
    if argument.pin_kernel:
        hours = tuple(float(x) for x in argument.pin_kernel.split(","))
        if len(hours) != 2 or not 0.0 < hours[0] < hours[1]:
            sys.exit("--pin-kernel wants ONSET_H,RECOVERY_H with 0 < onset < recovery")
        pinned = (hours[0] * 3600.0, hours[1] * 3600.0)
    everything = np.arange(len(rows))
    key = (rows.mission + " " + rows.month).to_numpy()
    month = rows.month.to_numpy()

    def leave_out(
        model: Model,
        fold_of: np.ndarray,
        label: str,
        keep: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        out = np.zeros(len(rows))
        for fold in np.unique(fold_of):
            test = fold_of == fold
            parameters = fit(model, np.where(~test)[0])
            if keep is not None:
                keep[str(fold)] = parameters
            out[test] = model.score(parameters, np.where(test)[0])
            if label:
                print(
                    f"  {label} {fold:<20} n={test.sum():6d}  MAPE {mape(out[test]):5.1f}%",
                    flush=True,
                )
        return out

    if argument.sweep:
        print(
            "caps       coeffs  ridge  bnd  in-sample   LOSMO  worst    LOMO   worst  "
            " transfer  debiased",
            flush=True,
        )
        for name in (
            "graded",
            "lean",
            "tight",
            "aseasonal",
            "storm2",
            "storm2-geo",
            "storm-log",
            "storm-log1",
        ):
            for scale in (True, False):
                for strength in (0.0, 1.0e-1):
                    candidate = Model(
                        arg,
                        ridge=strength,
                        sigma=sigma,
                        caps=CAPS[name],
                        boundary=scale,
                    )
                    inside = candidate.score(fit(candidate, everything), everything)
                    satellite_month = leave_out(candidate, key, "")
                    calendar_month = leave_out(candidate, month, "")
                    across = [
                        candidate.score(
                            fit(candidate, np.where(~(rows.mission == m))[0]),
                            np.where(rows.mission == m)[0],
                        )
                        for m in ("SWARM-A", "GRACE-FO-C")
                    ]
                    transfer = max(mape(r) for r in across)
                    # The same residual with its own mean removed: a held-out mission's scale is not
                    # identified by a training set that does not contain it, so the debiased number is
                    # what the *shape* of the model is worth there and the raw one is what a caller
                    # with no calibration would see.
                    debiased = max(mape(r - r.mean()) for r in across)
                    print(
                        f"{name:<10} {candidate.n_c:4d} {strength:>7g}"
                        f" {'fit' if scale else ' 1 '}"
                        f" {mape(inside):9.1f}% {mape(satellite_month):7.1f}%"
                        f" {max(mape(satellite_month[key == f]) for f in np.unique(key)):7.1f}%"
                        f" {mape(calendar_month):7.1f}%"
                        f" {max(mape(calendar_month[month == f]) for f in np.unique(month)):7.1f}%"
                        f" {transfer:9.1f}% {debiased:7.1f}%",
                        flush=True,
                    )
        # Second stage: the ridge is the one knob that moved the calendar-month split by more than a
        # factor of two above, so the cap set the first stage picks gets a finer ladder.
        print("", flush=True)
        for name in ("lean",):
            for scale in (True, False):
                for strength in (0.03, 0.3, 1.0):
                    candidate = Model(
                        arg,
                        ridge=strength,
                        sigma=sigma,
                        caps=CAPS[name],
                        boundary=scale,
                    )
                    inside = candidate.score(fit(candidate, everything), everything)
                    satellite_month = leave_out(candidate, key, "")
                    calendar_month = leave_out(candidate, month, "")
                    across = [
                        candidate.score(
                            fit(candidate, np.where(~(rows.mission == m))[0]),
                            np.where(rows.mission == m)[0],
                        )
                        for m in ("SWARM-A", "GRACE-FO-C")
                    ]
                    transfer = max(mape(r) for r in across)
                    # The same residual with its own mean removed: a held-out mission's scale is not
                    # identified by a training set that does not contain it, so the debiased number is
                    # what the *shape* of the model is worth there and the raw one is what a caller
                    # with no calibration would see.
                    debiased = max(mape(r - r.mean()) for r in across)
                    print(
                        f"{name:<10} {candidate.n_c:4d} {strength:>7g}"
                        f" {'fit' if scale else ' 1 '}"
                        f" {mape(inside):9.1f}% {mape(satellite_month):7.1f}%"
                        f" {max(mape(satellite_month[key == f]) for f in np.unique(key)):7.1f}%"
                        f" {mape(calendar_month):7.1f}%"
                        f" {max(mape(calendar_month[month == f]) for f in np.unique(month)):7.1f}%"
                        f" {transfer:9.1f}% {debiased:7.1f}%",
                        flush=True,
                    )
        return

    model = Model(
        arg,
        ridge=argument.ridge,
        sigma=sigma,
        caps=CAPS[argument.caps],
        boundary=not argument.fix_boundary,
        kernel=pinned,
    )
    print(
        f"degree {argument.degree}: K = {arg['k']}, caps {argument.caps} "
        f"({model.n_c} coefficients, {model.size} parameters), "
        f"ridge {argument.ridge:g}, sigma {'fitted' if sigma is None else f'{sigma:g} /m'}, "
        f"kernel {'fitted' if pinned is None else f'pinned {pinned[0] / 3600:g} / {pinned[1] / 3600:g} h'}",
        flush=True,
    )

    losmo = leave_out(model, key, "LOSMO")
    print(
        f"LOSMO pooled MAPE {mape(losmo):.1f}%  rms ln {np.std(losmo):.3f}", flush=True
    )

    by_month: dict[str, np.ndarray] = {}
    lomo = leave_out(model, month, "LOMO ", keep=by_month)
    print(f"LOMO pooled MAPE {mape(lomo):.1f}%  rms ln {np.std(lomo):.3f}", flush=True)

    for train, test in (("GRACE-FO-C", "SWARM-A"), ("SWARM-A", "GRACE-FO-C")):
        is_test = rows.mission.to_numpy() == test
        parameters = fit(model, np.where(~is_test)[0])
        residual = model.score(parameters, np.where(is_test)[0])
        print(
            f"  train {train:<11} -> test {test:<11} MAPE {mape(residual):5.1f}%  rms ln {np.std(residual):.3f}"
        )

    parameters = fit(model, everything)
    residual = model.score(parameters, everything)
    v = model.unpack(parameters)
    exponent = model.design(v.onset, v.recovery) @ v.coefficients
    t_inf = BOUNDARY_TEMPERATURE + np.exp(exponent)
    print(
        f"in-sample MAPE {mape(residual):.1f}%  rms ln {np.std(residual):.3f}\n"
        f"sigma {v.sigma:.4e} /m ({v.sigma * 1000:.4f} /km)  onset {v.onset / 3600:.2f} h  "
        f"recovery {v.recovery / 3600:.2f} h  boundary scale {np.exp(v.boundary):.4f}  "
        f"SWARM-A/GRACE-FO-C scale {np.exp(v.mission):.4f}\n"
        f"T_inf {t_inf.min():.0f}-{t_inf.max():.0f} K"
    )

    # The shipped uncertainty is built from the *calendar-month* residual, not the satellite-month
    # one: the leaky split would publish an interval a fifth too narrow, and an interval that does not
    # cover is worse than no interval. What it cannot cover is the transfer bias, which is why the
    # coverage of both splits is reported and only one of them is a claim.
    kernel_storm = storm_grid(arg["ap"], v.onset, v.recovery, arg["cadence"])[
        arg["interval"]
    ]
    cells = regime(arg["f107a"], kernel_storm)
    table = {}
    # Named `spread` and not `sigma`: `sigma` is the Bates shape parameter everywhere else in this
    # function, and rebinding it here would hand a six-element table to every model built afterwards.
    for label, held in (("LOSMO", losmo), ("LOMO ", lomo)):
        spread, n = uncertainty(held, arg["f107a"], kernel_storm)
        table[label] = spread
        print(
            f"{label} sigma of ln(model/measured) by regime  "
            + "  ".join(
                f"[{c}] {s_c:.3f} (n={n_c})"
                for c, (s_c, n_c) in enumerate(zip(spread, n, strict=True))
            )
        )
    # Which split an interval is built from is the whole decision, so it is measured both ways round.
    for source, spread in table.items():
        for label, held in (("LOSMO", losmo), ("LOMO ", lomo)):
            inside = np.abs(held) <= NORMAL_90 * spread[cells]
            print(
                f"  the {source} 90 % interval covers {100.0 * inside.mean():5.1f}% "
                f"of the {label} residual"
            )
    lomo_sigma = table["LOMO "]
    mission_coverage: dict[str, float] = {}
    for test in ("SWARM-A", "GRACE-FO-C"):
        is_test = rows.mission.to_numpy() == test
        across = model.score(fit(model, np.where(~is_test)[0]), np.where(is_test)[0])
        inside = np.abs(across) <= NORMAL_90 * lomo_sigma[cells[is_test]]
        mission_coverage[test] = 100.0 * float(inside.mean())
        print(
            f"  the LOMO interval covers {mission_coverage[test]:5.1f}% of {test} "
            f"held out entirely ({is_test.sum()} rows)"
        )

    # The storm view of the same model, on the protocol the literature assesses storms with. It runs
    # on the *unthinned* rows and not on the fitted sample: at --thin 32 there is one observation per
    # 64 minutes, which cannot average an orbit, and every timing number below is read off an orbit
    # average. The parameters are the ones already fitted -- the whole set for the in-sample pass, and
    # each calendar month's own leave-one-out fold for the held-out pass.
    storm_rows = storm_window_rows(argument.centred_mean, driver)
    if storm_rows is not None:
        storm_arg = fit_arguments(storm_rows, argument.degree)
        storm_model = Model(
            storm_arg,
            ridge=argument.ridge,
            sigma=sigma,
            caps=CAPS[argument.caps],
            boundary=not argument.fix_boundary,
            kernel=pinned,
        )
        whole = np.arange(len(storm_rows))
        storm_assessment(storm_rows, storm_model.score(parameters, whole), "in-sample")
        held_out = np.zeros(len(storm_rows))
        fine_month = storm_rows.month.to_numpy()
        missing = sorted(set(fine_month) - set(by_month))
        for m, p_m in by_month.items():
            here = np.where(fine_month == m)[0]
            if here.size:
                held_out[here] = storm_model.score(p_m, here)
        if missing:
            print(f"no leave-one-out fold for {missing}; those rows are in sample")
        storm_assessment(storm_rows, held_out, "calendar-month held-out")

    for name, mask in (
        ("F10.7a < 100", arg["f107a"] < 100),
        ("100-175", (arg["f107a"] >= 100) & (arg["f107a"] < 175)),
        (">= 175", arg["f107a"] >= 175),
        ("alt < 350 km", arg["_altitude"] < 350e3),
        ("alt >= 350 km", arg["_altitude"] >= 350e3),
        ("day (9-15 LST)", (arg["_lst"] >= 9) & (arg["_lst"] < 15)),
        ("night (21-3 LST)", (arg["_lst"] >= 21) | (arg["_lst"] < 3)),
    ):
        print(
            f"  bin {name:<16} n={mask.sum():6d}  LOSMO MAPE {mape(losmo[mask]):5.1f}%"
        )

    print(f"total {time.time() - began:.0f}s")


if __name__ == "__main__":
    main()
