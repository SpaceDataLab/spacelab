#!/usr/bin/env python3
"""Fit SDTM-1 on the whole TU Delft corpus and score it on held-out missions (`plans/16` Ф15.E).

The form, the drivers and the storm kernel are `tools/gen_sdtm_coefficients.py`'s, imported rather
than copied, so this fits the shipped model and nothing else. What differs is the corpus — the eight
satellites `tools/build-density-corpus.py` writes (`docs/verification.md` §10.20) instead of twelve
satellite-months — and what that forces:

* **one log scale per satellite**, not one for "the second mission". Each accelerometer is calibrated
  on its own, and orbit determination (Swarm) is a different method again; one satellite of the
  training set is the reference and carries none;
* **folds by mission family**, not by satellite: GRACE-A and -B fly the same orbit 220 km apart, and
  Swarm A and C side by side, so holding out one of a pair would score the model on its twin;
* **a held-out family's scales are not fitted** — there is nothing to fit them on — so its residual is
  reported twice: raw (the model's absolute level transfers) and debiased per satellite (only the
  structure transfers), the two readings `gen_sdtm_coefficients.py` already prints for one mission.

The design matrix holds the admitted (driver, harmonic) pairs only: the full 16 x K product is 840 MB
at the default thinning, for 35 columns of which the rest are zeros.

The comparator is NRLMSISE-00 on the same rows with §10.2's drivers (storm-time `ap[]`), evaluated by
the library itself through `examples/nrlmsise00_rows.rs`. `--fit-sigma` frees the Bates shape, which
the pilot held fixed because two satellites in one altitude band could not identify it; GOCE at 250 km
and CHAMP at 300-450 km can.

`--cooling` replaces the two-constant storm kernel by heating with nitric-oxide cooling
(`cooling_grid`, `derivations/sdtm_no_cooling.mac`), the Ф23.B state equation. `--cooling-mix`
drives its nitric oxide by `mix H + (1 - mix) D`, `mix` fixed or fitted, which is that equation
at `mix = 1` (`mixed_cooling_grid`). `--fast-heating` passes a share `w` of the heating through a
fast first-order mode that the nitric oxide does not cool (`fast_grid`). `--ap60` drives the storm
state by GFZ's hourly `ap60` instead of the 3-hourly `ap`, and `--solar-wind` mixes into it the
hourly solar-wind coupling of NASA's OMNI (`newell_driver`). `--storms` then scores every archived
storm on Bruinsma & Laurens (2024)'s protocol (Ф15.H), and per storm phase on Wang et al. (2026)'s
storms and satellites too (Ф23.C). `--storm-table` writes those per-phase statistics, and
`--scorecard` pools the tables of runs that held out different families.

`--emit` (with `--cooling`, every fold) writes the shipped field: the same form fitted on all eight
satellites, GRACE-FO C the reference, into `crates/spacelab-propagation/src/sdtm/coefficients.rs`,
and the Rust test's fixture. Its uncertainty table is the held-out-family residual's spread; its
storm-history length is the shortest whole number of days whose truncation moves the driver by
less than `HISTORY_TOLERANCE` anywhere in the corpus span. Nothing from the density files is in
either output: the coefficients are ours, the fixture is our own model's values. With `--ap60
--solar-wind fit` it writes the solar-wind field instead, `solar_wind_coefficients.rs` with its
coupling map, its fixture, and an hourly fixture of closed-form synthetic hours (`emit_hours`).

Run:

    cargo build -p spacelab-propagation --release --example nrlmsise00_rows
    uv run --no-project --with numpy --with pandas --with scipy python -I tools/fit-sdtm-corpus.py --fit-sigma
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_sdtm_coefficients as g

CORPUS = g.ROOT / "data" / "validation" / "tudelft-v02" / "corpus"
SATELLITES = ("GC", "SA", "SB", "SC", "GA", "GB", "CH", "GO")
FAMILY = {
    "CH": "CHAMP",
    "GA": "GRACE",
    "GB": "GRACE",
    "GO": "GOCE",
    "GC": "GRACE-FO",
    "SA": "Swarm",
    "SB": "Swarm",
    "SC": "Swarm",
}
NAME = {
    "CH": "CHAMP",
    "GA": "GRACE A",
    "GB": "GRACE B",
    "GO": "GOCE",
    "GC": "GRACE-FO C",
    "SA": "Swarm A",
    "SB": "Swarm B",
    "SC": "Swarm C",
}
WANG_2026 = (
    pd.Timestamp("2001-01-01"),
    pd.Timestamp("2024-01-01"),
    ("CH", "GA", "GO", "GC", "SA"),
)
"""Wang et al. (2026)'s storms and satellites: `t0` in 2001-2023, and CHAMP, GRACE, GOCE, GRACE-FO and
Swarm A. Of the GRACE pair only A: the two fly one orbit 220 km apart, so both would count GRACE twice."""
EPOCH_2000 = pd.Timestamp("2000-01-01")
OUT_RS = g.ROOT / "crates" / "spacelab-propagation" / "src" / "sdtm" / "coefficients.rs"
OUT_FIXTURE = (
    g.ROOT
    / "crates"
    / "spacelab-propagation"
    / "tests"
    / "fixtures"
    / "sdtm_reference.csv"
)
OUT_RS_WIND = OUT_RS.with_name("solar_wind_coefficients.rs")
OUT_FIXTURE_WIND = OUT_FIXTURE.with_name("sdtm_solar_wind_reference.csv")
OUT_FIXTURE_HOURS = OUT_FIXTURE.with_name("sdtm_solar_wind_hours.csv")
HISTORY_TOLERANCE = 0.01
"""Largest change in the storm driver, in `ap`, that the shipped history length may make: a
hundredth of the index's own resolution."""


def load_corpus(
    step: int,
    trailing_mean: bool = True,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    phase: int = 0,
    ap60: bool = False,
    wind: bool = False,
) -> pd.DataFrame:
    """Every `step`-th corpus row of each satellite, with `gen_sdtm_coefficients.load`'s drivers.

    `windows`, when given, keeps only the rows inside one of those `[start, end)` spans, before any
    driver is joined: the storm windows of the whole archive are a few per cent of its 22 million rows.
    `ap60` drives the storm state by GFZ's hourly series, as `gen_sdtm_coefficients.load` does; the
    storms are still found on the 3-hourly `ap`, so the same storms are scored. Measured and not
    shipped: better peak timing and a fifth of the large-storm gap (`docs/verification.md` §10.30).
    `wind`, with `ap60`, adds `newell_driver`'s series on the same hourly grid as `attrs["wind"]`,
    and the map that put it on `ap60`'s scale as `attrs["wind_map"]`.
    """
    frames = []
    for sat in SATELLITES:
        path = CORPUS / f"corpus_{sat}.npz"
        if not path.exists():
            sys.exit(f"no {path.name}; run tools/build-density-corpus.py")
        z = np.load(path)
        pick = np.arange(phase, z["t_ms"].size, step)
        if windows is not None:
            t_ms = z["t_ms"][pick]
            inside = np.zeros(pick.size, dtype=bool)
            for start, end in windows:
                lo, hi = (
                    (x - EPOCH_2000) // pd.Timedelta(milliseconds=1)
                    for x in (start, end)
                )
                inside[np.searchsorted(t_ms, lo) : np.searchsorted(t_ms, hi)] = True
            pick = pick[inside]
        frames.append(
            pd.DataFrame(
                {
                    "t": EPOCH_2000 + pd.to_timedelta(z["t_ms"][pick], unit="ms"),
                    "altitude": z["altitude_m"][pick],
                    "latitude": z["latitude"][pick],
                    "longitude": z["longitude"][pick],
                    "lst": z["lst"][pick],
                    "rho": z["rho"][pick],
                    "sat": sat,
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    rows["family"] = rows.sat.map(FAMILY)
    rows["month"] = rows.t.dt.strftime("%Y-%m")
    # Unused by this script's model, which carries its own per-satellite scales; `fit_arguments`
    # reads the column, so it is present and constant.
    rows["mission"] = "corpus"

    # The drivers exactly as `gen_sdtm_coefficients.load` joins them.
    mean_column = "F10.7_OBS_LAST81" if trailing_mean else "F10.7_OBS_CENTER81"
    flux = pd.read_csv(
        g.DENSITY / "SW-All.csv",
        usecols=["DATE", "F10.7_OBS", mean_column] + [f"AP{i}" for i in range(1, 9)],
    )
    flux["DATE"] = pd.to_datetime(flux.DATE)
    flux = flux.set_index("DATE").sort_index()
    day = rows.t.dt.normalize()
    rows["f107"] = flux.loc[day - pd.Timedelta(days=1), "F10.7_OBS"].to_numpy()
    rows["f107a"] = flux.loc[day, mean_column].to_numpy()
    ap = flux[[f"AP{i}" for i in range(1, 9)]].to_numpy().reshape(-1).astype(float)
    ap = np.where(np.isfinite(ap), ap, 0.0)
    offset = (
        day - flux.index[0]
    ).dt.days.to_numpy() * 8 + rows.t.dt.hour.to_numpy() // 3
    rows["interval"] = np.maximum(offset - 1, 0)
    rows.attrs.update(
        ap=ap,
        ap_epoch=flux.index[0],
        driver=ap,
        driver_epoch=flux.index[0],
        driver_cadence=g.AP_CADENCE,
    )
    if ap60:
        hourly, hourly_epoch = g.load_ap60()
        elapsed = (rows.t - hourly_epoch).dt.total_seconds().to_numpy()
        offset = np.floor(elapsed / g.AP60_CADENCE).astype(int)
        if offset.min() < 1 or offset.max() >= hourly.size:
            sys.exit("the ap60 series does not span the corpus")
        rows["interval"] = offset - 1
        rows.attrs.update(
            driver=hourly, driver_epoch=hourly_epoch, driver_cadence=g.AP60_CADENCE
        )
        if wind:
            rows.attrs["wind"], rows.attrs["wind_map"] = newell_driver(
                hourly, hourly_epoch
            )
    if not (np.isfinite(rows.f107).all() and np.isfinite(rows.f107a).all()):
        sys.exit("SW-All.csv lacks a flux for some corpus day")
    return rows


def shipped_kernel() -> tuple[float, float, float]:
    """The shipped field's `(heating h, nitric h, saturation ap)`, as `coefficients.rs` holds them."""
    text = OUT_RS.read_text()
    heating, nitric, saturation = (
        float(re.search(rf"\b{name}: ([0-9.eE+-]+),", text).group(1))
        for name in ("heating", "nitric_oxide", "saturation")
    )
    return heating / 3600.0, nitric / 3600.0, saturation


def newell_coupling(speed: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """Newell et al. (2007)'s `V^(4/3) B_T^(2/3) sin^(8/3)(theta/2)`, with `B_T` and the clock angle
    `theta` from the IMF's GSM `By` and `Bz`: `sdtm::SolarWind::coupling`'s arithmetic, in its order.
    """
    coupling = speed ** (4.0 / 3.0) * np.hypot(by, bz) ** (2.0 / 3.0)
    return coupling * np.abs(np.sin(np.arctan2(by, bz) / 2.0)) ** (8.0 / 3.0)


@dataclasses.dataclass(frozen=True)
class CouplingMap:
    """The map from Newell's coupling onto `ap60`'s scale, and the hours it was read from."""

    coupling: np.ndarray
    """The knots' coupling, strictly increasing."""
    ap60: np.ndarray
    """The knots' `ap60`: one class of the series each."""
    hours: int
    first: pd.Timestamp
    last: pd.Timestamp

    def __call__(self, coupling: np.ndarray) -> np.ndarray:
        """Linear between the knots and held at the end values outside them, as `np.interp` and
        `sdtm::CouplingMap::ap60` read them."""
        return np.interp(coupling, self.coupling, self.ap60)


def coupling_map(
    coupling: np.ndarray, ap60: np.ndarray, hours: np.ndarray
) -> CouplingMap:
    """One knot per `ap60` class over `hours`: the class's value at the coupling's quantile at the
    class's mid-probability.

    The quantile map this replaces is a step at every class boundary and is read off 240 000 hours,
    so it cannot ship; the table is continuous and monotone, and it is what the solar-wind field
    ships.
    """
    values, counts = np.unique(ap60, return_counts=True)
    knots = np.quantile(coupling, (np.cumsum(counts) - 0.5 * counts) / counts.sum())
    if not (np.diff(knots) > 0.0).all():
        sys.exit("the coupling's quantiles at ap60's classes do not strictly increase")
    return CouplingMap(knots, values, hours.size, hours.min(), hours.max())


def newell_driver(
    hourly: np.ndarray, epoch: pd.Timestamp
) -> tuple[np.ndarray, CouplingMap]:
    """`newell_coupling` on `ap60`'s grid and scale, and the map that put it there.

    The speed and the IMF are OMNI's hourly averages from their stamp, as `ap60`'s are
    (`tools/fetch-omni.sh`). `coupling_map` puts the coupling onto `ap60`'s scale over every hour both
    series have, so the two share one distribution, to the table's interpolation, and differ in
    which hours carry which value; an hour without the solar wind takes its `ap60`. Of three
    couplings it agrees best with `ap60` by the hour, Spearman 0.547 against Kan & Lee's 0.542 and
    `V Bs`'s 0.407 over 2000-2026.
    """
    path = g.DENSITY / "omni2_sw_hourly.txt"
    if not path.exists():
        sys.exit(f"no {path.name}; run tools/fetch-omni.sh")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 5, 6, 9],
        names=["hour", "by", "bz", "v"],
        na_values=["null"],
        engine="c",
    )
    start = pd.to_datetime(frame.hour, format="%Y-%m-%dT%H")
    step = start.diff().iloc[1:].unique()
    if step.size != 1 or step[0] != pd.Timedelta(hours=1):
        sys.exit(f"{path.name} is not a contiguous hourly grid: steps {step}")
    by, bz, v = (frame[c].to_numpy() for c in ("by", "bz", "v"))
    coupling = newell_coupling(v, by, bz)
    lead = int((start.iloc[0] - epoch) / pd.Timedelta(hours=1))
    if lead < 0:
        sys.exit(f"{path.name} starts before the ap60 series")
    on_grid = np.full(hourly.size, np.nan)
    span = min(coupling.size, hourly.size - lead)
    on_grid[lead : lead + span] = coupling[:span]
    present = np.isfinite(on_grid)
    hours = epoch + pd.to_timedelta(np.flatnonzero(present), unit="h")
    wind_map = coupling_map(on_grid[present], hourly[present], hours)
    out = hourly.copy()
    out[present] = wind_map(on_grid[present])
    print(
        f"{path.name}: the solar wind fills {present.sum()} of the {span} hours it spans, "
        f"{wind_map.first:%Y-%m-%dT%H} to {wind_map.last:%Y-%m-%dT%H}; "
        f"{wind_map.coupling.size} knots",
        flush=True,
    )
    return out, wind_map


STORM_COLUMNS = [4, 5, 6, 7]
"""`s`, `s^2`, `a`, `a^2`: the columns of `g.driver_matrix` the storm state enters."""


@dataclasses.dataclass(frozen=True)
class Packing:
    n_c: int
    scaled: tuple[str, ...]
    """Satellites with a fitted log scale: the training set less its reference."""
    free_sigma: bool
    cooling: bool
    """The storm response is `cooling_grid`'s two-state equation, with a third kernel slot."""
    capped: bool = False
    """And the cooling's largest factor `q` is fitted, in a fourth kernel slot (Ф23.B)."""
    mixed: bool = False
    """Or the nitric oxide's drive mixes the index and the response, `ln mix` in that slot."""
    fast: bool = False
    """And a fast, uncooled heating mode: `ln tau_f` and `ln w` in two kernel slots after those."""
    wind: bool = False
    """Or the solar wind's share of the drive, `ln share` in the fourth kernel slot."""
    free_excess: bool = False
    """The daily flux excess's saturation scale is fitted, after `sigma` (Ф23.G)."""
    free_gain: bool = False
    """The storm term's flux exponent `kappa` is fitted, after the excess (Ф23.B)."""
    free_knee: bool = False
    """And a constant `mu` beside it, `exp(mu + kappa f)`: the log regressor's knee, fitted."""

    @property
    def i_onset(self) -> int:
        return self.n_c

    @property
    def i_gap(self) -> int:
        return self.n_c + 1

    @property
    def i_cool(self) -> int:
        """`ln(H_s / ap)`, the cooling's saturation level; absent without cooling."""
        return self.n_c + 2 if self.cooling else -1

    @property
    def i_cap(self) -> int:
        """`ln q`, the cooling's largest factor; absent unless the cooling is capped."""
        return self.n_c + 3 if self.capped else -1

    @property
    def i_mix(self) -> int:
        """`ln mix`, the index's share of the nitric oxide's drive; absent unless it is mixed."""
        return self.n_c + 3 if self.mixed else -1

    @property
    def i_wind(self) -> int:
        """`ln share`, the solar wind's share of the drive; absent unless it is fitted."""
        return self.n_c + 3 if self.wind else -1

    @property
    def i_fast(self) -> int:
        """`ln tau_f`, the fast mode's time in hours, then `ln w`; absent without the mode."""
        return self.n_c + 3 + int(self.capped or self.mixed) if self.fast else -1

    @property
    def n_kernel(self) -> int:
        if not self.cooling:
            return 2
        return 3 + int(self.capped or self.mixed or self.wind) + 2 * int(self.fast)

    @property
    def i_boundary(self) -> int:
        return self.n_c + self.n_kernel

    @property
    def i_scale(self) -> int:
        return self.n_c + self.n_kernel + 1

    @property
    def i_sigma(self) -> int:
        """`sigma / 1e-5 m^-1`, last, and absent when `sigma` is held fixed."""
        return self.i_scale + len(self.scaled) if self.free_sigma else -1

    @property
    def i_excess(self) -> int:
        """`ln D` of `D tanh(d/D)`, after `sigma`, and absent when the excess is not fitted."""
        return (
            self.i_scale + len(self.scaled) + int(self.free_sigma)
            if self.free_excess
            else -1
        )

    @property
    def i_gain(self) -> int:
        """`kappa`, after the excess, and absent when the storm term does not scale with the flux."""
        return self.size - 1 - int(self.free_knee) if self.free_gain else -1

    @property
    def i_knee(self) -> int:
        """`mu`, last, and absent unless the knee is fitted."""
        return self.size - 1 if self.free_knee else -1

    @property
    def size(self) -> int:
        return (
            self.i_scale
            + len(self.scaled)
            + int(self.free_sigma)
            + int(self.free_excess)
            + int(self.free_gain)
            + int(self.free_knee)
        )


def cooling_grid(
    ap: np.ndarray,
    heating: float,
    nitric: float,
    saturation: float,
    cadence: float,
    first: int,
    cap: float = np.inf,
) -> np.ndarray:
    """Heating with nitric-oxide cooling, on the driver's grid (`derivations/sdtm_no_cooling.mac`).

    `dD/dt = (H - D)/heating - D N/(heating saturation)`, `dN/dt = (H - N)/nitric`, `H` the driver
    held over each interval. `N` takes its exact step; `D` the exact step of the system with `N`
    frozen at its exact mean over the interval, whose local error starts at `h^3`. Element `i` is
    the state after interval `i`, the convention `storm_grid` has, so the same `interval` index
    reads it. Without cooling (`saturation` -> infinity) it is the unit-gain first-order filter;
    in steady state it is `H / (1 + H/saturation)`.

    A finite `cap` = `q` bounds the cooling: `N/saturation` becomes `(N/saturation)/(1 + N/(q
    saturation))`, the same below `q saturation` and at most `q` above it, so the steady state grows as
    `H/(1 + q)` instead of stopping at `saturation`. Refuted, not derived: the fit sends `q` to
    its 1e4 bound in every fold (`docs/verification.md` §10.27).

    The recursion starts at `first` from rest; entries before it are zero and are never read.
    """
    h = cadence / 3600.0
    decay = np.exp(-h / nitric)
    out = np.zeros(ap.size)
    d = n = 0.0
    mean_weight = nitric / h * (1.0 - decay)
    for i in range(first, ap.size):
        drive = ap[i]
        n_mean = drive + (n - drive) * mean_weight
        cooling = n_mean / saturation / (1.0 + n_mean / (cap * saturation))
        rate = (1.0 + cooling) / heating
        target = drive / (heating * rate)
        d = target + (d - target) * np.exp(-rate * h)
        n = drive + (n - drive) * decay
        out[i] = d
    return out


def mixed_cooling_grid(
    ap: np.ndarray,
    heating: float,
    nitric: float,
    saturation: float,
    mix: float,
    cadence: float,
    first: int,
) -> np.ndarray:
    """`cooling_grid` with the nitric oxide driven by `mix H + (1 - mix) D` rather than `H`.

    `dN/dt = (mix H + (1 - mix) D - N)/nitric`. At `mix = 1` it is `cooling_grid` bit for bit; at
    `mix = 0` the nitric oxide follows the response, and the steady state `D (1 + D/saturation) =
    H` grows as `sqrt(H saturation)` instead of stopping at `saturation`. The step is
    `cooling_grid`'s, with the drive of `N` frozen at the interval mean of `D` from the frozen-`N`
    step and the pair iterated once. Against RK4 on a two-day `ap` 300 storm at the shipped
    constants it errs by 0.46 % of the peak at `mix = 0` and 0.75 % at `mix = 0.5`, and at
    `mix = 1` by `cooling_grid`'s own 1.28 % (Maxima). Refuted, not derived: the fit sends `mix`
    to 1 in every fold, and fixed at 0 or 0.5 it leaves the largest storms' peaks where they
    were (`docs/verification.md` §10.30).
    """
    h = cadence / 3600.0
    decay = np.exp(-h / nitric)
    mean_weight = nitric / h * (1.0 - decay)
    out = np.zeros(ap.size)
    d = n = 0.0
    for i in range(first, ap.size):
        index = ap[i]
        d_mean = d
        for _ in range(2):
            drive = mix * index + (1.0 - mix) * d_mean
            n_mean = drive + (n - drive) * mean_weight
            rate = (1.0 + n_mean / saturation) / heating
            target = index / (heating * rate)
            relax = np.exp(-rate * h)
            d_mean = target + (d - target) * (1.0 - relax) / (rate * h)
        d = target + (d - target) * relax
        n = drive + (n - drive) * decay
        out[i] = d
    return out


def fast_grid(ap: np.ndarray, fast: float, cadence: float, first: int) -> np.ndarray:
    """The unit-gain first-order filter `dF/dt = (H - F)/fast`, `fast` in hours, on the ap grid.

    Its exact step for the driver held over each interval is the nitric-oxide step of
    `derivations/sdtm_no_cooling.mac` 1, with `fast` for `tn`. `--fast-heating` makes the storm
    state `(1 - w) D + w F`: the share `w` of the heating that reaches the density before the
    nitric oxide builds and is not cooled by it, so the steady state grows as `w H` past
    `cooling_grid`'s saturation. Conventions as `cooling_grid`'s. Refuted: the fit sends `fast`
    to its 0.25 h bound with `w` 0.004-0.014, and fixed at 1.5 h and 0.1 it leaves the largest
    storms' peaks where they were (`docs/verification.md` §10.30).
    """
    decay = np.exp(-cadence / 3600.0 / fast)
    out = np.zeros(ap.size)
    f = 0.0
    for i in range(first, ap.size):
        f = ap[i] + (f - ap[i]) * decay
        out[i] = f
    return out


class CorpusModel:
    """`ln rho` of the shipped form with one log scale per satellite; `sigma` fitted when `None`."""

    def __init__(
        self,
        arg: dict,
        sat: np.ndarray,
        train: tuple[str, ...],
        caps,
        ridge: float,
        sigma: float | None,
        cooling: bool | str = False,
        excess: float | str | None = None,
        gain: str | None = None,
        mix: float | str | None = None,
        fast: tuple[float, float] | str | None = None,
        wind: float | str | None = None,
    ):
        """`excess` saturates the daily flux excess: `None` keeps it linear, a number is a fixed
        scale `D`, `"fit"` fits `D` with the other nonlinear parameters. `gain`, with `kappa`
        fitted and `f` the 81-day flux regressor: `"amplitude"` multiplies every storm regressor by
        `exp(kappa f)`; `"input"` multiplies the storm state by it before the regressors are formed,
        so inside `a = ln(1 + storm/15)` it is a gain on a minor storm and a shift on a large one;
        `"input-knee"` is `"input"` with `exp(mu + kappa f)`, `mu` fitted too; `"input-power"` is
        `"input"` with `(F10.7a/150)^kappa`, flatter than `exp(kappa f)` at high flux. `mix`, with
        cooling, drives the nitric oxide by `mix H + (1 - mix) D`: `None` keeps `H`, a number is a
        fixed share, `"fit"` fits it in a fourth kernel slot. `fast`, with cooling: `None` has no
        fast mode, `(tau_f h, w)` a fixed one, `"fit"` fits both after the other kernel slots.
        `wind`, with cooling and an hourly driver: the drive is `(1 - share) ap60 + share N`, `N` in
        `arg["wind"]`; `None` has no solar wind, a number is a fixed share, `"fit"` fits it."""
        self.a = arg
        self.sat = sat
        self.sigma = sigma
        self.excess_scale = excess if isinstance(excess, float) else None
        self.mix_share = mix if isinstance(mix, float) else None
        self.fast_mode = fast if isinstance(fast, tuple) else None
        self.wind_share = wind if isinstance(wind, float) else None
        self.pairs = np.argwhere(
            g.harmonic_mask(caps, arg["k"]).reshape(g.DRIVERS, arg["k"])
        )
        self.pack = Packing(
            n_c=len(self.pairs),
            scaled=train[1:],
            free_sigma=sigma is None,
            cooling=bool(cooling),
            capped=cooling == "capped",
            mixed=mix == "fit",
            fast=fast == "fit",
            wind=wind == "fit",
            free_excess=excess == "fit",
            free_gain=gain is not None,
            free_knee=gain == "input-knee",
        )
        self.gain_form = gain
        # 120 days before the first sample: the slowest admitted state (480 h) has forgotten its
        # start by e^-6 there.
        per_day = round(86_400.0 / arg["cadence"])
        self.first = max(int(arg["interval"].min()) - 120 * per_day, 0)
        self.indicator = (
            np.column_stack([(sat == s).astype(float) for s in self.pack.scaled])
            if self.pack.scaled
            else np.zeros((sat.size, 0))
        )
        self.ridge = ridge
        self.root_weight: np.ndarray | None = None
        """Per-row `sqrt(weight)` of the loss, or `None` for every row alike."""
        self._design_key: tuple[float | None, ...] | None = None
        self._design: np.ndarray | None = None
        reference = self.design(
            *(
                (
                    (6.0, 20.0, 300.0)
                    + ((10.0,) if cooling == "capped" else ())
                    + ((0.5,) if mix == "fit" or wind == "fit" else ())
                    + ((2.0, 0.1) if fast == "fit" else ())
                )
                if cooling
                else (3.0 * 3600.0, 18.0 * 3600.0)
            ),
            excess=0.5 if self.pack.free_excess else self.excess_scale,
            gain=0.0 if gain else None,
        )
        self.penalty = np.sqrt(ridge) * np.sqrt(np.mean(reference**2, axis=0))
        self.penalty[0] = 0.0

    def shape(self, p: np.ndarray) -> float:
        return p[self.pack.i_sigma] * 1.0e-5 if self.sigma is None else self.sigma

    def kernel(self, p: np.ndarray) -> tuple[float, ...]:
        """`(onset s, recovery s)`, or with cooling `(heating h, nitric h, saturation ap)` and then
        `q` or `mix` where either is fitted."""
        pk = self.pack
        if pk.cooling:
            slots = (pk.i_onset, pk.i_gap, pk.i_cool) + (
                (pk.i_cap,) if pk.capped else ((pk.i_mix,) if pk.mixed else ())
            )
            slots += (pk.i_wind,) if pk.wind else ()
            slots += (pk.i_fast, pk.i_fast + 1) if pk.fast else ()
            return tuple(float(np.exp(p[j])) for j in slots)
        onset = np.exp(p[pk.i_onset]) * 3600.0
        return onset, onset + np.exp(p[pk.i_gap]) * 3600.0

    def excess(self, p: np.ndarray) -> float | None:
        """The saturation scale `D` of the daily flux excess, or `None` where it is linear."""
        return (
            float(np.exp(p[self.pack.i_excess]))
            if self.pack.free_excess
            else self.excess_scale
        )

    def gain(self, p: np.ndarray) -> float | None:
        """`kappa` of the storm term's flux gain, or `None` where the term ignores the flux."""
        return float(p[self.pack.i_gain]) if self.pack.free_gain else None

    def _gain_argument(self, f107a: np.ndarray) -> np.ndarray:
        """`f`, or `ln(F10.7a/150)` for `"input-power"`, whose gain is `(F10.7a/150)^kappa`."""
        if self.gain_form == "input-power":
            return np.log(f107a / g.REFERENCE_F107)
        return (f107a - g.REFERENCE_F107) / g.F107_SCALE

    def knee(self, p: np.ndarray) -> float | None:
        """`mu` of `exp(mu + kappa f)`, or `None` where the knee is held at `ln(1 + storm/15)`'s."""
        return float(p[self.pack.i_knee]) if self.pack.free_knee else None

    def share(self, kernel: tuple[float, ...]) -> float | None:
        """The solar wind's share of the drive, or `None` without it."""
        return kernel[3] if self.pack.wind else self.wind_share

    def drive(self, kernel: tuple[float, ...]) -> np.ndarray:
        """The series the storm state integrates: the index, or `(1 - share) ap60 + share N`."""
        share = self.share(kernel)
        if share is None:
            return self.a["ap"]
        return (1.0 - share) * self.a["ap"] + share * self.a["wind"]

    def design(
        self,
        *kernel: float,
        excess: float | None = None,
        gain: float | None = None,
        knee: float | None = None,
    ) -> np.ndarray:
        key = (*kernel, excess, gain, knee)
        if self._design_key == key and self._design is not None:
            return self._design
        drive = self.drive(kernel)
        if self.pack.mixed or self.mix_share is not None:
            share = kernel[3] if self.pack.mixed else self.mix_share
            grid = mixed_cooling_grid(
                self.a["ap"], *kernel[:3], share, self.a["cadence"], self.first
            )
        elif self.pack.cooling:
            grid = cooling_grid(
                drive,
                *kernel[:3],
                self.a["cadence"],
                self.first,
                cap=kernel[3] if self.pack.capped else np.inf,
            )
        else:
            grid = g.storm_grid(self.a["ap"], *kernel, self.a["cadence"])
        if self.pack.fast or self.fast_mode is not None:
            fast, share = kernel[-2:] if self.pack.fast else self.fast_mode
            grid = (1.0 - share) * grid + share * fast_grid(
                self.a["ap"], fast, self.a["cadence"], self.first
            )
        storm = grid[self.a["interval"]]
        f107, f107a = self.a["f107"], self.a["f107a"]
        if excess is not None:
            span = g.F107_SCALE * excess
            f107 = f107a + span * np.tanh((f107 - f107a) / span)
        scale = (
            None
            if gain is None
            else np.exp((knee or 0.0) + gain * self._gain_argument(f107a))
        )
        if scale is not None and self.gain_form in (
            "input",
            "input-knee",
            "input-power",
        ):
            storm = storm * scale
        drivers = g.driver_matrix(f107, f107a, storm, self.a["doy"])
        if scale is not None and self.gain_form == "amplitude":
            drivers[:, STORM_COLUMNS] *= scale[:, None]
        out = np.empty((drivers.shape[0], len(self.pairs)))
        for j, (d, k) in enumerate(self.pairs):
            out[:, j] = drivers[:, d] * self.a["harmonics"][:, k]
        self._design_key, self._design = key, out
        return out

    def score(self, p: np.ndarray, index: np.ndarray, design: np.ndarray | None = None):
        """`(ln(model/measured), d/d coefficients factor)` on `index`."""
        if design is None:
            design = self.design(
                *self.kernel(p),
                excess=self.excess(p),
                gain=self.gain(p),
                knee=self.knee(p),
            )
        exponent = design[index] @ p[: self.pack.n_c]
        t_inf = g.BOUNDARY_TEMPERATURE + np.exp(np.clip(exponent, -50.0, 50.0))
        ln_rho, d_ln = g.profile(t_inf, self.shape(p), self.a["zeta"][index])
        scales = (
            self.indicator[index]
            @ p[self.pack.i_scale : self.pack.i_scale + len(self.pack.scaled)]
        )
        residual = ln_rho + p[self.pack.i_boundary] + scales - self.a["ln_rho"][index]
        return residual, d_ln * (t_inf - g.BOUNDARY_TEMPERATURE)

    def residual(self, p: np.ndarray, index: np.ndarray) -> np.ndarray:
        r, _ = self.score(p, index)
        if self.root_weight is not None:
            r = r * self.root_weight[index]
        return np.concatenate(
            [r, np.sqrt(index.size) * self.penalty * p[: self.pack.n_c]]
        )

    def jacobian(self, p: np.ndarray, index: np.ndarray) -> np.ndarray:
        n, pk = index.size, self.pack
        out = np.zeros((n + pk.n_c, pk.size))
        design = self.design(
            *self.kernel(p), excess=self.excess(p), gain=self.gain(p), knee=self.knee(p)
        )
        base, chain = self.score(p, index, design)
        out[:n, : pk.n_c] = design[index] * chain[:, None]
        out[:n, pk.i_boundary] = 1.0
        out[:n, pk.i_scale : pk.i_scale + len(pk.scaled)] = self.indicator[index]
        step = 1.0e-4
        slots = (pk.i_onset, pk.i_gap) + ((pk.i_cool,) if pk.cooling else ())
        slots += (pk.i_cap,) if pk.capped else ()
        slots += (pk.i_mix,) if pk.mixed else ()
        slots += (pk.i_wind,) if pk.wind else ()
        slots += (pk.i_fast, pk.i_fast + 1) if pk.fast else ()
        slots += (pk.i_sigma,) if pk.free_sigma else ()
        slots += (pk.i_excess,) if pk.free_excess else ()
        slots += (pk.i_gain,) if pk.free_gain else ()
        for j in slots + ((pk.i_knee,) if pk.free_knee else ()):
            shifted = p.copy()
            shifted[j] += step
            out[:n, j] = (self.score(shifted, index)[0] - base) / step
        if self.root_weight is not None:
            out[:n] *= self.root_weight[index][:, None]
        # restore the cache to the unshifted kernel
        self.design(
            *self.kernel(p), excess=self.excess(p), gain=self.gain(p), knee=self.knee(p)
        )
        out[n:, : pk.n_c] = np.diag(np.sqrt(n) * self.penalty)
        return out

    def fit(self, index: np.ndarray) -> np.ndarray:
        pk = self.pack
        start = np.zeros(pk.size)
        start[0] = np.log(700.0)
        lower, upper = np.full(pk.size, -np.inf), np.full(pk.size, np.inf)
        if pk.cooling:
            # Heating 0.5-72 h and nitric oxide 1-480 h, in hours; the saturation from 5 ap, where
            # cooling halves the response of a minor storm, to 1e5, where there is none.
            start[pk.i_onset], start[pk.i_gap], start[pk.i_cool] = np.log(
                [6.0, 20.0, 300.0]
            )
            lower[pk.i_onset], upper[pk.i_onset] = np.log(0.5), np.log(72.0)
            lower[pk.i_gap], upper[pk.i_gap] = np.log(1.0), np.log(480.0)
            lower[pk.i_cool], upper[pk.i_cool] = np.log(5.0), np.log(1.0e5)
        else:
            start[pk.i_onset], start[pk.i_gap] = np.log(3.0), np.log(15.0)
            lower[pk.i_onset], upper[pk.i_onset] = np.log(1.0 / 60.0), np.log(12.0)
            lower[pk.i_gap], upper[pk.i_gap] = np.log(1.0), np.log(72.0)
        if pk.capped:
            # From cooling that at most doubles the decay rate to one that is never reached.
            start[pk.i_cap] = np.log(10.0)
            lower[pk.i_cap], upper[pk.i_cap] = np.log(1.0), np.log(1.0e4)
        if pk.mixed:
            # From a drive that is almost all response to the shipped one, all index.
            start[pk.i_mix] = np.log(0.5)
            lower[pk.i_mix], upper[pk.i_mix] = np.log(1.0e-3), 0.0
        if pk.wind:
            # From a drive that is almost all ap60 to one that is all solar wind.
            start[pk.i_wind] = np.log(0.5)
            lower[pk.i_wind], upper[pk.i_wind] = np.log(1.0e-3), 0.0
        if pk.fast:
            # Faster than a 3 h interval to a quarter day; a share from none to half the heating.
            start[pk.i_fast], start[pk.i_fast + 1] = np.log(2.0), np.log(0.1)
            lower[pk.i_fast], upper[pk.i_fast] = np.log(0.25), np.log(12.0)
            lower[pk.i_fast + 1], upper[pk.i_fast + 1] = np.log(1.0e-3), np.log(0.5)
        lower[pk.i_boundary], upper[pk.i_boundary] = -0.7, 0.7
        # The accelerometer-to-accelerometer and accelerometer-to-POD scales, bounded as
        # `gen_sdtm_coefficients.fit` bounds its one mission scale and for the same reason.
        scales = slice(pk.i_scale, pk.i_scale + len(pk.scaled))
        lower[scales], upper[scales] = -0.3, 0.3
        if pk.free_sigma:
            start[pk.i_sigma] = 2.0
            lower[pk.i_sigma], upper[pk.i_sigma] = 0.5, 6.0
        if pk.free_excess:
            # 2 to 1000 sfu: from a term that is a sign to one that is linear across the corpus.
            start[pk.i_excess] = np.log(0.5)
            lower[pk.i_excess], upper[pk.i_excess] = np.log(0.02), np.log(10.0)
        if pk.free_gain:
            # A factor e^3 either way across one unit of f, 100 sfu of the 81-day mean.
            lower[pk.i_gain], upper[pk.i_gain] = -3.0, 3.0
        if pk.free_knee:
            lower[pk.i_knee], upper[pk.i_knee] = -3.0, 3.0
        out = least_squares(
            self.residual,
            start,
            jac=self.jacobian,
            args=(index,),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=80,
            verbose=0,
        )
        if pk.wind or self.wind_share is not None:
            # With the solar wind in the drive a second basin, nitric oxide at its 1 h bound and
            # a training cost 1.2 % higher, catches the start above in one fold of five
            # (`docs/verification.md` §10.31): start again from the shipped field's kernel and
            # keep the lower cost.
            start = start.copy()
            start[[pk.i_onset, pk.i_gap, pk.i_cool]] = np.log(shipped_kernel())
            again = least_squares(
                self.residual,
                start,
                jac=self.jacobian,
                args=(index,),
                bounds=(lower, upper),
                x_scale="jac",
                max_nfev=80,
                verbose=0,
            )
            if again.cost < out.cost:
                out = again
        return out.x


def truncated_cooling(
    ap: np.ndarray,
    heating: float,
    nitric: float,
    saturation: float,
    cadence: float,
    ends: np.ndarray,
    length: int,
) -> np.ndarray:
    """`NitricOxideCooling::response` over the `length` intervals ending at each of `ends`.

    The Rust's convention, not `cooling_grid`'s: each window starts in steady state at its own first
    value instead of from rest far back, so the difference between the two is what a finite history
    costs.
    """
    start = ends - length + 1
    h = cadence / 3600.0
    decay = np.exp(-h / nitric)
    mean_weight = nitric / h * (1.0 - decay)
    first = ap[start]
    d, n = first / (1.0 + first / saturation), first.copy()
    for j in range(length):
        drive = ap[start + j]
        n_mean = drive + (n - drive) * mean_weight
        rate = (1.0 + n_mean / saturation) / heating
        target = drive / (heating * rate)
        d = target + (d - target) * np.exp(-rate * h)
        n = drive + (n - drive) * decay
    return d


def history_length(model: CorpusModel, p: np.ndarray) -> tuple[int, float]:
    """The shortest whole-day history whose response is within `HISTORY_TOLERANCE` of the fit's own.

    Scored on every interval of the drive's cadence from the corpus's first row to its last,
    against `cooling_grid` warmed up from rest 120 days before the first. With a storm gain the
    regressors read the gained driver, so the change is scaled by the largest gain any corpus row
    has. Returns the length in intervals.
    """
    kernel = model.kernel(p)
    gain = model.gain(p)
    largest = (
        1.0
        if gain is None
        else float(np.exp(gain * model._gain_argument(model.a["f107a"])).max())
    )
    drive, cadence = model.drive(kernel), model.a["cadence"]
    per_day = round(86_400.0 / cadence)
    reference = cooling_grid(drive, *kernel[:3], cadence, model.first)
    interval = model.a["interval"]
    ends = np.arange(int(interval.min()), int(interval.max()) + 1)
    for days in range(1, 61):
        worst = largest * float(
            np.abs(
                truncated_cooling(drive, *kernel[:3], cadence, ends, per_day * days)
                - reference[ends]
            ).max()
        )
        print(
            f"  history {days:2d} d: worst storm-driver change {worst:.3g} ap",
            flush=True,
        )
        if worst < HISTORY_TOLERANCE:
            return per_day * days, worst
    sys.exit("no history up to 60 days reaches HISTORY_TOLERANCE")


def wind_note(wind_map: CouplingMap, share: float) -> list[str]:
    """The solar-wind field's header lines: its drive, and the attribution its map's sources ask."""
    return [
        f"It runs hourly on (1 - s) ap60 + s N, s = {share:.3f} fitted (`DRIVE`): GFZ's ap60, and N",
        "Newell et al. (2007)'s coupling of NASA's OMNI solar wind on ap60's scale, by a table whose",
        f"{wind_map.coupling.size} knots are read off the {wind_map.hours} hours both series have, "
        f"{wind_map.first:%Y-%m-%dT%H} to",
        f"{wind_map.last:%Y-%m-%dT%H}: OMNI from NASA/GSFC's SPDF, CC0 1.0 (King & Papitashvili 2005,",
        "doi:10.1029/2004JA010649), and ap60 from GFZ Helmholtz Centre for Geosciences, CC BY 4.0",
        "(Matzka et al. 2024, doi:10.5880/Hpo.0003), which is the attribution that licence asks of",
        "adapted material. An hour without the solar wind takes its ap60.",
    ]


def drive_rs(wind_map: CouplingMap, share: float) -> str:
    """`KNOTS` and `DRIVE` for `solar_wind_coefficients.rs`."""

    def rows(values: np.ndarray, per_line: int) -> str:
        return "".join(
            "            "
            + ", ".join(g.literal(v) for v in values[i : i + per_line])
            + ",\n"
            for i in range(0, values.size, per_line)
        )

    return (
        "/// Number of knots in [`DRIVE`]'s coupling map.\n"
        f"pub const KNOTS: usize = {wind_map.coupling.size};\n\n"
        "/// How [`FIELD`]'s storm state is driven: hourly, by `(1 - share) ap60 + share N` with `N` the\n"
        "/// solar-wind coupling on `ap60`'s scale, and by `ap60` alone in an hour without the solar\n"
        "/// wind. Built by the unchecked constructors for the reason [`FIELD`] is;\n"
        "/// `sdtm_solar_wind_reference.rs` puts the literals through `CouplingMap::new` and\n"
        "/// `SolarWindDrive::new`.\n"
        "#[rustfmt::skip]\n"
        "pub const DRIVE: SolarWindDrive<KNOTS> = SolarWindDrive::from_fitted(\n"
        f"    {g.literal(share)},\n"
        "    CouplingMap::from_fitted(\n"
        "        [\n" + rows(wind_map.coupling, 4) + "        ],\n"
        "        [\n" + rows(wind_map.ap60, 12) + "        ],\n"
        "    ),\n"
        ");\n\n"
    )


def synthetic_hours(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(ap60, speed km/s, By nT, Bz nT)` over `n` hours: a minor storm, a major one peaking 36 h
    before the end, and the cases `sdtm::SolarWindDrive` and `sdtm::CouplingMap` branch on.

    Closed-form rather than measured, so the fixture carries nothing of OMNI's or GFZ's: gaps of a
    day, of four hours and of one (all three values NaN), a northward field with `By = 0` and a zero
    field (coupling zero, below the first knot), the clock angle at `-pi/2` and at `pi`, and a coupling
    above the last knot.
    """
    t = np.arange(n, dtype=float)
    major = np.exp(-0.5 * ((t - (n - 36)) / 10.0) ** 2)
    minor = np.exp(-0.5 * ((t - n / 2) / 6.0) ** 2)
    ap60 = np.round(2.0 + 300.0 * major + 60.0 * minor)
    speed = (
        350.0 + 450.0 * major + 120.0 * minor + 15.0 * np.sin(2.0 * np.pi * t / 29.0)
    )
    transverse = 3.0 + 35.0 * major + 8.0 * minor
    clock = 2.0 * np.pi * t / 41.0 + np.pi * major
    by, bz = transverse * np.sin(clock), transverse * np.cos(clock)
    for hour, (v, y, z) in {
        n - 30: (1100.0, 0.0, -50.0),
        n - 10: (400.0, 0.0, 5.0),
        n - 9: (400.0, 0.0, 0.0),
        n - 8: (500.0, -6.0, 0.0),
        n - 7: (500.0, 0.0, -6.0),
    }.items():
        speed[hour], by[hour], bz[hour] = v, y, z
    for gap in (np.arange(100, 124), np.arange(n - 40, n - 36), np.array([n - 20])):
        speed[gap] = by[gap] = bz[gap] = np.nan
    return ap60, speed, by, bz


def emit_hours(model: CorpusModel, kernel: tuple[float, ...], history: int) -> None:
    """The solar-wind field's hourly fixture: `synthetic_hours` through `newell_coupling`, the map,
    the drive, and `truncated_cooling` over each `history`-hour window that ends in the last 48."""
    share, wind_map = model.share(kernel), model.a["wind_map"]
    ap60, speed, by, bz = synthetic_hours(history + 48)
    coupling = newell_coupling(speed, by, bz)
    present = np.isfinite(coupling)
    drive = np.where(present, (1.0 - share) * ap60 + share * wind_map(coupling), ap60)
    ends = np.arange(history - 1, ap60.size)
    storm = np.full(ap60.size, np.nan)
    storm[ends] = truncated_cooling(drive, *kernel[:3], 3600.0, ends, history)
    lines = [
        "# SDTM-1 solar-wind field hourly reference -- generated by tools/fit-sdtm-corpus.py.",
        "# Closed-form synthetic hours (`synthetic_hours`), not measured data. NaN marks an hour without",
        f"# the solar wind; `storm` is the driver over the {history} hours ending at the row.",
        "# ap60,speed_km_s,by_nT,bz_nT,coupling,drive,storm",
    ]
    for row in zip(ap60, speed, by, bz, coupling, drive, storm, strict=True):
        lines.append(",".join("NaN" if np.isnan(x) else g.literal(x) for x in row))
    OUT_FIXTURE_HOURS.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_FIXTURE_HOURS.relative_to(g.ROOT)} ({ap60.size} hours)")


def emit(
    model: CorpusModel,
    p: np.ndarray,
    rows: pd.DataFrame,
    argument: argparse.Namespace,
    held: np.ndarray,
) -> None:
    """Write the shipped field and the fixture: `p` fitted on every satellite, `held` the residual
    each row had in the fold that held its family out. With the solar wind in the drive the field
    is the solar-wind one, and `emit_hours` writes its hourly fixture beside it."""
    pk, arg = model.pack, model.a
    kernel = model.kernel(p)
    heating, nitric, saturation = kernel[:3]
    share = model.share(kernel)
    out_rs, out_fixture = (
        (OUT_RS, OUT_FIXTURE) if share is None else (OUT_RS_WIND, OUT_FIXTURE_WIND)
    )
    grid = np.zeros((g.DRIVERS, arg["k"]))
    for (d, k), c in zip(model.pairs, p[: pk.n_c], strict=True):
        grid[d, k] = c
    storm = cooling_grid(
        model.drive(kernel), heating, nitric, saturation, arg["cadence"], model.first
    )[arg["interval"]]
    f107a = arg["f107a"]

    sigma, count = g.uncertainty(held, f107a, storm)
    if not (count > 0).all():
        sys.exit(f"an uncertainty cell has no held-out rows: {count}")
    cells = g.regime(f107a, storm)
    family = rows.family.to_numpy()
    coverage = {}
    covered = np.zeros(len(held), dtype=bool)
    for fam in sorted(set(family)):
        here = family == fam
        table, _ = g.uncertainty(held[~here], f107a[~here], storm[~here])
        covered[here] = np.abs(held[here]) <= g.NORMAL_90 * table[cells[here]]
        coverage[fam] = 100.0 * float(covered[here].mean())
        print(
            f"  the 90 % interval built without {fam} covers {coverage[fam]:.1f} % of it"
        )
    # The table's claim is per regime, so it is checked per regime too: a pooled family figure can
    # hide one cell, and one family's cell is the floor a single campaign cell is held to.
    for c in range(len(sigma)):
        cell = cells == c
        worst = min(covered[cell & (family == f)].mean() for f in set(family[cell]))
        print(
            f"  regime {c}: held out, the 90 % interval covers {100.0 * covered[cell].mean():.1f} %"
            f" of {cell.sum()} rows, {100.0 * worst:.1f} % of its worst family"
        )
    history, worst = history_length(model, p)

    scales = p[pk.i_scale : pk.i_scale + len(pk.scaled)]
    boundary = p[pk.i_boundary]
    at_bound = " (on its bound)" if abs(abs(boundary) - 0.7) < 1e-9 else ""
    note = [
        f"Fitted {pd.Timestamp.today():%Y-%m-%d} by `tools/fit-sdtm-corpus.py` on {len(rows)} rows,",
        f"every {argument.step}th of the TU Delft v02 corpus (`docs/verification.md` §10.20):",
        "CHAMP, GRACE A/B, GOCE, GRACE-FO C and Swarm A/B/C, "
        f"{rows.t.min():%Y-%m} to {rows.t.max():%Y-%m}, from TU Delft,",
        "Siemes et al. (2023), CC BY 4.0 -- which is the attribution that licence asks of adapted",
        "material, and `docs/provenance.md` (`sdtm-reference`) is the rest of it.",
        "",
        f"Degree {argument.degree}, harmonic caps `{argument.caps}`: {pk.n_c} of the "
        f"{g.DRIVERS * arg['k']} possible",
        f"coefficients, ridge {argument.ridge:g}, the Bates shape fitted, the boundary scale fitted"
        f"{at_bound}.",
        "The storm driver is heating with nitric-oxide cooling (`derivations/sdtm_no_cooling.mac`).",
        *([] if share is None else wind_note(model.a["wind_map"], share)),
        *(
            [
                "The daily flux excess saturates as D tanh(d/D), D fitted "
                "(`derivations/sdtm_flux_excess.mac`)."
            ]
            if model.excess(p) is not None
            else []
        ),
        *(
            [
                "The storm driver scales as (F10.7a/150)^kappa, kappa fitted "
                "(`derivations/sdtm_storm_gain.mac`)."
            ]
            if model.gain(p) is not None
            else []
        ),
        f"Held out by mission family, MAPE {g.mape(held):.1f} % pooled.",
        "",
        "The uncertainty table below is that held-out residual's spread about zero, per regime. A",
        "table built without one family's rows has a 90 % interval that covers, of that family:",
        *(f"{k} {v:.1f} %," for k, v in coverage.items()),
    ]
    satellite_rows = "".join(
        f'    ("{NAME[s]}", {g.literal(v)}),\n'
        for s, v in zip(pk.scaled, scales, strict=True)
    )
    rows_rs = ["    [" + ", ".join(g.literal(c) for c in row) + "]," for row in grid]
    out_rs.write_text(
        (
            "//! Fitted SDTM-1 coefficients -- generated by `tools/fit-sdtm-corpus.py`, do not edit.\n"
            if share is None
            else "//! Fitted SDTM-1 solar-wind field -- generated by `tools/fit-sdtm-corpus.py`, do not "
            "edit.\n"
        )
        + "//!\n"
        + "".join(f"//! {line}\n".replace("//! \n", "//!\n") for line in note)
        + (
            "//!\n"
            "//! The fit's own verdict, the folds it was scored on and what it does not claim are in\n"
            f"//! `docs/verification.md` §10.21-§10.{28 if share is None else 32}.\n"
            "#![allow(clippy::unreadable_literal)]\n\n"
            "use super::{\n"
            + (
                "    DRIVERS, DensityUncertainty, Envelope, FLUX_EDGES, FluxExcess, NitricOxideCooling, STORM_EDGES,\n"
                "    StormGain, StormResponse, Thermosphere,\n"
                if share is None
                else "    CouplingMap, DRIVERS, DensityUncertainty, Envelope, FLUX_EDGES, FluxExcess, NitricOxideCooling,\n"
                "    STORM_EDGES, SolarWindDrive, StormGain, StormResponse, Thermosphere,\n"
            )
            + "};\n\n"
            "/// Degree of the sun-fixed spherical-harmonic expansion this fit used.\n"
            f"pub const DEGREE: usize = {argument.degree};\n"
            "/// Number of harmonics, `(DEGREE + 1)^2`.\n"
            f"pub const HARMONICS: usize = {arg['k']};\n\n"
            "/// `ln(T_inf - T_0)` coefficients, driver-major: `[driver][harmonic]`. A zero is a pair the\n"
            "/// fit did not admit, not a pair that came out small.\n"
            "#[rustfmt::skip]\n"
            "const COEFFICIENTS: [[f64; HARMONICS]; DRIVERS] = [\n"
            + "\n".join(rows_rs)
            + "\n];\n\n"
            "/// The fitted field.\n"
            "///\n"
            "/// Built by the unchecked constructor because every argument is a literal in this file;\n"
            "/// `sdtm_reference.rs` puts the same six through `Thermosphere::from_coefficients`,\n"
            "/// `with_flux_excess` and `with_storm_gain` and asserts they accept them, which is where the\n"
            "/// refusals bind.\n"
            "pub const FIELD: Thermosphere<HARMONICS> = Thermosphere::from_fitted(\n"
            "    COEFFICIENTS,\n"
            f"    {g.literal(model.shape(p))},\n"
            f"    {g.literal(np.exp(boundary))},\n"
            "    StormResponse::Cooling(NitricOxideCooling {\n"
            f"        heating: {g.literal(heating * 3600.0)},\n"
            f"        nitric_oxide: {g.literal(nitric * 3600.0)},\n"
            f"        saturation: {g.literal(saturation)},\n"
            "    }),\n"
            + (
                "    FluxExcess::Saturated {\n"
                f"        scale: {g.literal(model.excess(p))},\n"
                "    },\n"
                if model.excess(p) is not None
                else "    FluxExcess::Linear,\n"
            )
            + (
                "    StormGain::FluxPower {\n"
                f"        exponent: {g.literal(model.gain(p))},\n"
                "    },\n"
                if model.gain(p) is not None
                else "    StormGain::Unity,\n"
            )
            + ");\n\n"
            + ("" if share is None else drive_rs(model.a["wind_map"], share))
            + "/// Where [`FIELD`] was fitted: the lowest and the highest geodetic altitude, m, and 81-day mean\n"
            "/// F10.7 among its rows.\n"
            "pub const ENVELOPE: Envelope = Envelope {\n"
            f"    altitude: {g.literal(rows.altitude.min())}..={g.literal(rows.altitude.max())},\n"
            f"    f107a: {g.literal(rows.f107a.min())}..={g.literal(rows.f107a.max())},\n"
            "};\n\n"
            + (
                "/// Number of 3-hourly `ap` intervals the storm state is to be run over: the shortest whole\n"
                "/// number of days whose truncation moves the driver by less than a hundredth of an `ap` at\n"
                f"/// every interval of the corpus span (worst {worst:.2g}).\n"
                if share is None
                else "/// Number of hours the storm state is to be run over: the shortest whole number of days\n"
                "/// whose truncation moves the driver by less than a hundredth of an `ap` at every hour of\n"
                f"/// the corpus span (worst {worst:.2g}).\n"
            )
            + f"pub const HISTORY: usize = {history};\n\n"
            "/// Log-scale offset of each satellite's densities against GRACE-FO C's, reported rather\n"
            "/// than applied: the model is the physics, and these are what the calibrations disagree by.\n"
            f"pub const SATELLITE_SCALE: [(&str, f64); {len(pk.scaled)}] = [\n"
            + satellite_rows
            + "];\n\n"
            "/// Spread of `ln(rho_model/rho_measured)` by regime, from the residual each row had with its\n"
            "/// mission family held out -- see `DensityUncertainty` for why a held-out split.\n"
            "pub const UNCERTAINTY: DensityUncertainty = DensityUncertainty::from_fitted([\n"
            + "".join(f"    {g.literal(x)},\n" for x in sigma)
            + "]);\n\n"
            "// The regime edges the table above was measured on. If the crate's edges move and this\n"
            "// file is not regenerated, every entry is indexed by a bin it was not measured in -- and\n"
            "// silently, because the table would still be the right length. So it is a compile error.\n"
            + "".join(
                f"const _: () = assert!(FLUX_EDGES[{i}] == {g.literal(e)});\n"
                for i, e in enumerate(g.UNCERTAINTY_FLUX)
            )
            + "".join(
                f"const _: () = assert!(STORM_EDGES[{i}] == {g.literal(e)});\n"
                for i, e in enumerate(g.UNCERTAINTY_STORM)
            )
        )
    )
    print(f"wrote {out_rs.relative_to(g.ROOT)}")

    index = np.linspace(0, len(rows) - 1, 64).astype(int)
    exponent = (
        model.design(*kernel, excess=model.excess(p), gain=model.gain(p))[index]
        @ p[: pk.n_c]
    )
    t_inf = g.BOUNDARY_TEMPERATURE + np.exp(exponent)
    ln_rho, _ = g.profile(t_inf, model.shape(p), arg["zeta"][index])
    lines = [
        f"# SDTM-1{'' if share is None else ' solar-wind field'} reference values -- generated by "
        "tools/fit-sdtm-corpus.py.",
        "# Our own model evaluated at our own sample of geometries: nothing here is measured data.",
        "# altitude_m,latitude_rad,local_solar_time_h,day_of_year,f107,f107a,storm_ap,t_inf_K,density_kg_m3",
    ]
    for j, i in enumerate(index):
        field = [
            rows.altitude.iat[i],
            np.deg2rad(rows.latitude.iat[i]),
            rows.lst.iat[i],
            arg["doy"][i],
            arg["f107"][i],
            f107a[i],
            storm[i],
            t_inf[j],
            np.exp(ln_rho[j] + boundary),
        ]
        lines.append(",".join(g.literal(x) for x in field))
    out_fixture.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_fixture.relative_to(g.ROOT)} ({index.size} rows)")
    if share is not None:
        emit_hours(model, kernel, history)


def debias(residual: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Each satellite's mean log residual removed: what is left when only the structure transfers."""
    out = residual.copy()
    for s in set(sat):
        here = sat == s
        out[here] -= residual[here].mean()
    return out


def nrlmsise00(rows: pd.DataFrame) -> np.ndarray:
    """`ln rho` of NRLMSISE-00 with the storm-time `ap[]`, from the Rust evaluator (§10.2's drivers)."""
    program = g.ROOT / "target" / "release" / "examples" / "nrlmsise00_rows"
    if not program.exists():
        sys.exit(
            "no NRLMSISE-00 evaluator; run cargo build -p spacelab-propagation --release "
            "--example nrlmsise00_rows"
        )
    t = rows.t
    dates = np.column_stack([t.dt.year, t.dt.month, t.dt.day])
    values = np.column_stack(
        [
            (t - t.dt.normalize()).dt.total_seconds(),
            rows.altitude / 1000.0,
            rows.latitude,
            rows.longitude,
            rows.lst,
        ]
    )
    # `repr` is the shortest string that round-trips, so the evaluator sees the corpus's own doubles.
    text = "".join(
        f"{d[0]} {d[1]} {d[2]} " + " ".join(repr(float(x)) for x in v) + "\n"
        for d, v in zip(dates, values, strict=True)
    )
    done = subprocess.run(
        [str(program), str(g.DENSITY / "SW-All.csv")],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    rho = np.array(done.stdout.split(), dtype=float)
    if rho.size != len(rows):
        sys.exit(f"NRLMSISE-00 answered {rho.size} rows of {len(rows)}")
    return np.log(rho)


def report(
    rows: pd.DataFrame, ap_daily: np.ndarray, columns: dict[str, np.ndarray]
) -> None:
    f107a = rows.f107a.to_numpy()
    bins = {
        "all": np.ones(len(rows), dtype=bool),
        "F10.7a < 100": f107a < 100,
        "100-150": (f107a >= 100) & (f107a < 150),
        ">= 150": f107a >= 150,
        "Ap < 15": ap_daily < 15,
        "15-50": (ap_daily >= 15) & (ap_daily < 50),
        "Ap >= 50": ap_daily >= 50,
    }
    # The daily excess enters `T∞` linearly (`driver_matrix`'s `d`); these bins are where that
    # linearity is tested (Ф23.G), so they also print the median O/C, which says which way it errs.
    excess = (rows.f107.to_numpy() - f107a) / g.F107_SCALE
    excess_bins = {
        "d < 0": excess < 0.0,
        "0 <= d < 0.3": (excess >= 0.0) & (excess < 0.3),
        "0.3 <= d < 0.7": (excess >= 0.3) & (excess < 0.7),
        "d >= 0.7": excess >= 0.7,
    }
    bins.update(excess_bins)
    for fam in sorted(set(rows.family)):
        bins[fam] = rows.family.to_numpy() == fam
    print("== held out by mission family, MAPE")
    print(f"{'bin':<14} {'n':>8}" + "".join(f" {name:>16}" for name in columns))
    for name, mask in bins.items():
        if mask.any():
            cells = "".join(f" {g.mape(r[mask]):15.1f}%" for r in columns.values())
            print(f"{name:<14} {mask.sum():8d}{cells}")
    print("== by daily flux excess, median model/measured")
    for name, mask in excess_bins.items():
        if mask.any():
            cells = "".join(
                f" {np.exp(np.median(r[mask])):16.3f}" for r in columns.values()
            )
            print(f"{name:<14} {mask.sum():8d}{cells}")


def excess_option(argument: argparse.Namespace) -> float | str | None:
    """`--excess-saturation` as `CorpusModel` takes it."""
    value = argument.excess_saturation
    return value if value in (None, "fit") else float(value)


def fast_option(argument: argparse.Namespace) -> tuple[float, float] | str | None:
    """`--fast-heating` as `CorpusModel` takes it."""
    value = argument.fast_heating
    if value in (None, "fit"):
        return value
    tau, share = (float(x) for x in value.split(","))
    return tau, share


def wind_option(argument: argparse.Namespace) -> float | str | None:
    """`--solar-wind` as `CorpusModel` takes it."""
    value = argument.solar_wind
    return value if value in (None, "fit") else float(value)


def mix_option(argument: argparse.Namespace) -> float | str | None:
    """`--cooling-mix` as `CorpusModel` takes it."""
    value = argument.cooling_mix
    return value if value in (None, "fit") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--step", type=int, default=60, help="keep every n-th 120 s corpus row"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="start the thinning at this row (0 <= phase < step)",
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--caps", choices=sorted(g.CAPS), default="storm-log1")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument("--sigma", type=float, default=2.0e-5, help="Bates shape, /m")
    shape.add_argument(
        "--fit-sigma",
        action="store_true",
        help="fit the Bates shape: the corpus spans 250-540 km, which the pilot did not",
    )
    parser.add_argument("--folds", nargs="*", help="only these held-out families")
    parser.add_argument(
        "--storms",
        action="store_true",
        help="then score every storm of the archive on Bruinsma & Laurens (2024)'s protocol, "
        "each satellite by the fold that held its family out (Ф15.H)",
    )
    parser.add_argument(
        "--storm-table",
        type=pathlib.Path,
        help="with --storms, write every comparison's per-phase statistics to this CSV",
    )
    parser.add_argument(
        "--scorecard",
        nargs="+",
        type=pathlib.Path,
        metavar="CSV",
        help="print the per-phase scorecards of these --storm-table files pooled, and fit nothing",
    )
    parser.add_argument(
        "--emit",
        action="store_true",
        help="then fit on every satellite and write coefficients.rs and the fixture, or with "
        "--ap60 --solar-wind fit the solar-wind field's; needs --cooling and every fold",
    )
    parser.add_argument(
        "--cooling",
        action="store_true",
        help="drive the storm term by heating with nitric-oxide cooling (Ф23.B) instead of the "
        "two-constant kernel",
    )
    parser.add_argument(
        "--storm-weight",
        type=float,
        help="weight the fold fits' rows with daily Ap >= 50 by this factor; scoring is unweighted",
    )
    parser.add_argument(
        "--cooling-cap",
        action="store_true",
        help="bound the cooling's factor at a fitted q, so the steady response keeps growing",
    )
    parser.add_argument(
        "--cooling-mix",
        metavar="MIX|fit",
        help="drive the nitric oxide by mix H + (1 - mix) D, mix fixed in [0, 1] or fitted in "
        "every fold: the shipped closure at mix = 1, the response's own at mix = 0",
    )
    parser.add_argument(
        "--ap60",
        action="store_true",
        help="drive the storm state by GFZ's hourly ap60 instead of CelesTrak's 3-hourly ap; "
        "storms are still found on the 3-hourly ap",
    )
    parser.add_argument(
        "--solar-wind",
        metavar="SHARE|fit",
        help="with --ap60, drive the storm state by (1 - SHARE) ap60 + SHARE N, N the Newell "
        "coupling of OMNI's hourly solar wind on ap60's scale, SHARE fixed in [0, 1] or fitted "
        "in every fold",
    )
    parser.add_argument(
        "--fast-heating",
        metavar="TAU,W|fit",
        help="pass a share W of the heating through a fast uncooled mode of TAU hours, fixed or "
        "fitted in every fold: the storm state becomes (1 - W) D + W F",
    )
    parser.add_argument(
        "--storm-gain",
        choices=("amplitude", "input", "input-knee", "input-power"),
        help="scale the storm term (amplitude) or the storm state inside it (input) by "
        "exp(kappa f), by exp(mu + kappa f) (input-knee) or by (F10.7a/150)^kappa (input-power), "
        "kappa and mu fitted in every fold (Ф23.B); --emit writes input-power",
    )
    parser.add_argument(
        "--excess-saturation",
        metavar="D|fit",
        help="saturate the daily flux excess as D tanh(d/D) (Ф23.G), D fixed or fitted in every "
        "fold; the comparator and the bins read the raw flux",
    )
    argument = parser.parse_args()
    if argument.scorecard:
        scorecards(
            pd.concat(
                (pd.read_csv(path, parse_dates=["t0"]) for path in argument.scorecard),
                ignore_index=True,
            )
        )
        return
    if argument.storm_table and not argument.storms:
        sys.exit("--storm-table writes what --storms scores: give both")
    if argument.emit and argument.excess_saturation not in (None, "fit"):
        sys.exit(
            "a fixed --excess-saturation is a measurement, not a shippable field: fit it"
        )
    if argument.emit and argument.storm_weight is not None:
        sys.exit("--storm-weight is a fold experiment: --emit fits every row alike")
    if argument.cooling_cap:
        if not argument.cooling or argument.emit:
            sys.exit("--cooling-cap bounds --cooling, and --emit does not write it yet")
        argument.cooling = "capped"
    if argument.cooling_mix is not None:
        if not argument.cooling or argument.emit or argument.cooling_cap:
            sys.exit(
                "--cooling-mix changes --cooling's closure, alone: --emit does not write it yet"
            )
        if (
            argument.cooling_mix != "fit"
            and not 0.0 <= float(argument.cooling_mix) <= 1.0
        ):
            sys.exit(
                f"--cooling-mix is a share in [0, 1] or fit, not {argument.cooling_mix}"
            )
    if argument.ap60 and argument.emit and argument.solar_wind is None:
        sys.exit(
            "--ap60 alone is a fold experiment: --emit writes the 3-hourly field, or the "
            "solar-wind one with --solar-wind fit"
        )
    if argument.solar_wind is not None:
        if not (argument.cooling and argument.ap60):
            sys.exit("--solar-wind mixes into --ap60's hourly drive under --cooling")
        if argument.emit and argument.solar_wind != "fit":
            sys.exit(
                "a fixed --solar-wind share is a measurement, not a shippable field: fit it"
            )
        if (
            argument.cooling_cap
            or argument.cooling_mix is not None
            or argument.fast_heating
        ):
            sys.exit(
                "--solar-wind takes --cooling alone, without a cap, a mix or a fast mode"
            )
        if (
            argument.solar_wind != "fit"
            and not 0.0 <= float(argument.solar_wind) <= 1.0
        ):
            sys.exit(
                f"--solar-wind is a share in [0, 1] or fit, not {argument.solar_wind}"
            )
    if argument.fast_heating is not None:
        if (
            not argument.cooling
            or argument.emit
            or argument.cooling_cap
            or argument.cooling_mix is not None
        ):
            sys.exit(
                "--fast-heating adds to --cooling alone, without a cap or a mix: --emit does not "
                "write it yet"
            )
        if argument.fast_heating != "fit":
            tau, share = fast_option(argument)
            if not (tau > 0.0 and 0.0 <= share <= 1.0):
                sys.exit(
                    f"--fast-heating wants TAU > 0 h, W in [0, 1] or fit: {argument.fast_heating}"
                )
    if argument.emit and argument.storm_gain not in (None, "input-power"):
        sys.exit(
            "--emit writes the storm gain as sdtm::StormGain::FluxPower: give input-power"
        )
    if argument.emit and (argument.folds or not argument.cooling):
        sys.exit(
            "--emit writes the cooling field and its held-out table: give --cooling, no --folds"
        )
    sigma = None if argument.fit_sigma else argument.sigma

    began = time.time()
    rows = load_corpus(
        argument.step,
        phase=argument.phase,
        ap60=argument.ap60,
        wind=argument.solar_wind is not None,
    )
    day_ap = rows.attrs["ap"].reshape(-1, 8).mean(axis=1)
    ap_daily = day_ap[
        (rows.t.dt.normalize() - rows.attrs["ap_epoch"]).dt.days.to_numpy()
    ]
    print(
        f"{len(rows)} rows, {rows.sat.nunique()} satellites, {time.time() - began:.0f}s",
        flush=True,
    )
    arg = g.fit_arguments(rows, argument.degree)
    arg["wind"] = rows.attrs.get("wind")
    arg["wind_map"] = rows.attrs.get("wind_map")
    excess = excess_option(argument)
    mix = mix_option(argument)
    fast = fast_option(argument)
    wind = wind_option(argument)
    sat = rows.sat.to_numpy()
    caps = g.CAPS[argument.caps]

    families = argument.folds or sorted(set(FAMILY.values()))
    held = np.full(len(rows), np.nan)
    fitted: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for fam in families:
        test = rows.family.to_numpy() == fam
        train_sats = tuple(s for s in SATELLITES if s in set(sat[~test]))
        model = CorpusModel(
            arg,
            sat,
            train_sats,
            caps,
            argument.ridge,
            sigma,
            argument.cooling,
            excess,
            argument.storm_gain,
            mix,
            fast,
            wind,
        )
        if argument.storm_weight is not None:
            model.root_weight = np.sqrt(
                np.where(ap_daily >= 50.0, argument.storm_weight, 1.0)
            )
        t0 = time.time()
        p = model.fit(np.where(~test)[0])
        r, _ = model.score(p, np.where(test)[0])
        held[test] = r
        fitted[fam] = (train_sats, p)
        kernel = model.kernel(p)
        pk = model.pack
        scales = " ".join(
            f"{s} {np.exp(v):.3f}"
            for s, v in zip(
                pk.scaled, p[pk.i_scale : pk.i_scale + len(pk.scaled)], strict=True
            )
        )
        print(
            f"held out {fam:<9} {test.sum():8d} rows: raw {g.mape(r):5.1f}%  "
            f"debiased {g.mape(debias(r, sat[test])):5.1f}%  bias {np.exp(r.mean()):.3f}  "
            + (
                f"heating {kernel[0]:.2f} h  NO {kernel[1]:.1f} h  saturation {kernel[2]:.0f} ap  "
                + (f"cap q {kernel[3]:.2f}  " if pk.capped else "")
                + (f"mix {kernel[3]:.3f}  " if pk.mixed else "")
                + (f"solar wind {kernel[3]:.3f}  " if pk.wind else "")
                + (
                    f"fast {kernel[-2]:.2f} h share {kernel[-1]:.3f}  "
                    if pk.fast
                    else ""
                )
                if pk.cooling
                else f"kernel {kernel[0] / 3600:.2f}/{kernel[1] / 3600:.2f} h  "
            )
            + f"boundary {np.exp(p[pk.i_boundary]):.3f}  sigma {model.shape(p) * 1e3:.4f}/km  "
            + (
                f"excess D {model.excess(p):.3f}  "
                if model.excess(p) is not None
                else ""
            )
            + (
                f"storm gain kappa {model.gain(p):+.3f}  "
                if model.gain(p) is not None
                else ""
            )
            + (f"knee mu {model.knee(p):+.3f}  " if model.knee(p) is not None else "")
            + f"scales vs {train_sats[0]}: {scales}  {time.time() - t0:.0f}s",
            flush=True,
        )
    msis = np.full(len(rows), np.nan)
    scored = np.isfinite(held)
    msis[scored] = nrlmsise00(rows[scored]) - arg["ln_rho"][scored]
    # Both models are scored on the same rows. NRLMSISE-00 has no answer where the feed lacks a day
    # it reads, or where its own temperature profile fails: on 2005-09-10 the previous day's F10.7 is
    # the 707.6 sfu of the 2005-09-09 X-class flare, and the profile returns NaN on part of that day.
    done = scored & np.isfinite(msis)
    for day, n in rows[scored & ~done].t.dt.date.value_counts().sort_index().items():
        print(
            f"NRLMSISE-00 has no answer on {n} rows of {day}; dropped from both models"
        )
    report(
        rows[done].reset_index(drop=True),
        ap_daily[done],
        {
            "SDTM-1 raw": held[done],
            "SDTM-1 debiased": debias(held[done], sat[done]),
            "NRLMSISE-00": msis[done],
            "MSIS debiased": debias(msis[done], sat[done]),
        },
    )
    if argument.emit:
        print("\nthe shipped field, fitted on every satellite", flush=True)
        model = CorpusModel(
            arg,
            sat,
            SATELLITES,
            caps,
            argument.ridge,
            sigma,
            argument.cooling,
            excess,
            argument.storm_gain,
            mix,
            fast,
            wind,
        )
        p = model.fit(np.arange(len(rows)))
        r, _ = model.score(p, np.arange(len(rows)))
        names = ("heating h", "NO h", "saturation ap") + (
            ("solar wind",) if model.pack.wind else ()
        )
        print(
            f"  in sample MAPE {g.mape(r):.1f}%  "
            + "  ".join(
                f"{name} {v:.4g}"
                for name, v in zip(names, model.kernel(p), strict=True)
            ),
            flush=True,
        )
        emit(model, p, rows, argument, held)
    if argument.storms:
        storms(rows, fitted, argument, sigma)


def storms(
    rows: pd.DataFrame,
    fitted: dict[str, tuple[tuple[str, ...], np.ndarray]],
    argument: argparse.Namespace,
    sigma: float | None,
) -> None:
    """Ф15.H: every storm the archive covers, each satellite scored by the model that never saw it.

    The fit is on thinned rows; the storm protocol averages over orbits, so it reads the corpus at
    its full 120 s inside the storm windows only. Storms are found on the 3-hourly `ap` from the
    corpus's first day, so `storm_onsets` does not scan sixty years the corpus does not cover.
    """
    ap, epoch = rows.attrs["ap"], rows.attrs["ap_epoch"]
    first = (rows.t.min().normalize() - epoch).days * 8
    found = g.storm_onsets(ap[first:], epoch + pd.Timedelta(days=first // 8))
    windows = [
        (
            s["t0"] + pd.Timedelta(hours=g.PHASES[0][0]),
            s["t0"] + pd.Timedelta(hours=s["phase3_end"] + 12.0),
        )
        for s in found
    ]
    window_rows = load_corpus(
        1, windows=windows, ap60=argument.ap60, wind=argument.solar_wind is not None
    )
    window_rows = window_rows[window_rows.family.isin(fitted)].reset_index(drop=True)
    window_rows["mission"] = window_rows.sat
    print(
        f"\n{len(found)} storms with ap >= {g.STORM_AP:g} since {rows.t.min():%Y-%m}, "
        f"{sum(WANG_2026[0] <= s['t0'] < WANG_2026[1] for s in found)} with t0 in 2001-2023; "
        f"{len(window_rows)} rows at 120 s inside their windows",
        flush=True,
    )
    excess = excess_option(argument)
    mix = mix_option(argument)
    fast = fast_option(argument)
    wind = wind_option(argument)
    arg = g.fit_arguments(window_rows, argument.degree)
    arg["wind"] = window_rows.attrs.get("wind")
    sat = window_rows.sat.to_numpy()
    caps = g.CAPS[argument.caps]
    held = np.full(len(window_rows), np.nan)
    for fam, (train_sats, p) in fitted.items():
        model = CorpusModel(
            arg,
            sat,
            train_sats,
            caps,
            argument.ridge,
            sigma,
            argument.cooling,
            excess,
            argument.storm_gain,
            mix,
            fast,
            wind,
        )
        here = np.where(window_rows.family.to_numpy() == fam)[0]
        held[here] = model.score(p, here)[0]
    msis = nrlmsise00(window_rows) - arg["ln_rho"]
    done = np.isfinite(held) & np.isfinite(msis)
    print(
        f"{int((~done).sum())} window rows without an NRLMSISE-00 answer; dropped from both"
    )
    scored = window_rows[done].reset_index(drop=True)
    scored.attrs = {
        **rows.attrs,
        "ap": ap[first:],
        "ap_epoch": epoch + pd.Timedelta(days=first // 8),
    }
    present = tuple(s for s in SATELLITES if s in set(scored.sat))
    table = pd.concat(
        [
            g.storm_assessment(scored, residual, model, present).assign(model=model)
            for model, residual in (
                ("SDTM-1 held-out-mission", held[done]),
                ("NRLMSISE-00 ap[]", msis[done]),
            )
        ],
        ignore_index=True,
    )
    scorecards(table)
    if argument.storm_table:
        table.to_csv(argument.storm_table, index=False)
        print(f"wrote {len(table)} per-phase rows to {argument.storm_table}")


def scorecards(table: pd.DataFrame) -> None:
    """Each model's per-phase scorecard over every comparison, then over Wang et al. (2026)'s scope."""
    start, end, satellites = WANG_2026
    scope = (table.t0 >= start) & (table.t0 < end) & table.mission.isin(satellites)
    for model in table.model.unique():
        mine = table.model == model
        g.phase_scorecard(table[mine], f"{model}, every comparison")
        g.phase_scorecard(
            table[mine & scope],
            f"{model}, Wang et al. (2026)'s scope: 2001-2023, {' '.join(satellites)}",
        )


if __name__ == "__main__":
    main()
