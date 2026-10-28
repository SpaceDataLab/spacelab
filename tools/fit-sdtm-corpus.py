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

Run:

    cargo build -p spacelab-propagation --release --example nrlmsise00_rows
    uv run --no-project --with numpy --with pandas --with scipy python -I tools/fit-sdtm-corpus.py --fit-sigma
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
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
EPOCH_2000 = pd.Timestamp("2000-01-01")


def load_corpus(step: int, trailing_mean: bool = True) -> pd.DataFrame:
    """Every `step`-th corpus row of each satellite, with `gen_sdtm_coefficients.load`'s drivers."""
    frames = []
    for sat in SATELLITES:
        path = CORPUS / f"corpus_{sat}.npz"
        if not path.exists():
            sys.exit(f"no {path.name}; run tools/build-density-corpus.py")
        z = np.load(path)
        pick = slice(None, None, step)
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
    if not (np.isfinite(rows.f107).all() and np.isfinite(rows.f107a).all()):
        sys.exit("SW-All.csv lacks a flux for some corpus day")
    return rows


@dataclasses.dataclass(frozen=True)
class Packing:
    n_c: int
    scaled: tuple[str, ...]
    """Satellites with a fitted log scale: the training set less its reference."""
    free_sigma: bool

    @property
    def i_onset(self) -> int:
        return self.n_c

    @property
    def i_gap(self) -> int:
        return self.n_c + 1

    @property
    def i_boundary(self) -> int:
        return self.n_c + 2

    @property
    def i_scale(self) -> int:
        return self.n_c + 3

    @property
    def i_sigma(self) -> int:
        """`sigma / 1e-5 m^-1`, last, and absent when `sigma` is held fixed."""
        return self.n_c + 3 + len(self.scaled) if self.free_sigma else -1

    @property
    def size(self) -> int:
        return self.n_c + 3 + len(self.scaled) + int(self.free_sigma)


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
    ):
        self.a = arg
        self.sat = sat
        self.sigma = sigma
        self.pairs = np.argwhere(
            g.harmonic_mask(caps, arg["k"]).reshape(g.DRIVERS, arg["k"])
        )
        self.pack = Packing(
            n_c=len(self.pairs), scaled=train[1:], free_sigma=sigma is None
        )
        self.indicator = (
            np.column_stack([(sat == s).astype(float) for s in self.pack.scaled])
            if self.pack.scaled
            else np.zeros((sat.size, 0))
        )
        self.ridge = ridge
        self._design_key: tuple[float, float] | None = None
        self._design: np.ndarray | None = None
        reference = self.design(3.0 * 3600.0, 18.0 * 3600.0)
        self.penalty = np.sqrt(ridge) * np.sqrt(np.mean(reference**2, axis=0))
        self.penalty[0] = 0.0

    def shape(self, p: np.ndarray) -> float:
        return p[self.pack.i_sigma] * 1.0e-5 if self.sigma is None else self.sigma

    def kernel(self, p: np.ndarray) -> tuple[float, float]:
        onset = np.exp(p[self.pack.i_onset]) * 3600.0
        return onset, onset + np.exp(p[self.pack.i_gap]) * 3600.0

    def design(self, onset: float, recovery: float) -> np.ndarray:
        if self._design_key == (onset, recovery) and self._design is not None:
            return self._design
        storm = g.storm_grid(self.a["ap"], onset, recovery, self.a["cadence"])[
            self.a["interval"]
        ]
        drivers = g.driver_matrix(self.a["f107"], self.a["f107a"], storm, self.a["doy"])
        out = np.empty((drivers.shape[0], len(self.pairs)))
        for j, (d, k) in enumerate(self.pairs):
            out[:, j] = drivers[:, d] * self.a["harmonics"][:, k]
        self._design_key, self._design = (onset, recovery), out
        return out

    def score(self, p: np.ndarray, index: np.ndarray, design: np.ndarray | None = None):
        """`(ln(model/measured), d/d coefficients factor)` on `index`."""
        if design is None:
            design = self.design(*self.kernel(p))
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
        return np.concatenate(
            [r, np.sqrt(index.size) * self.penalty * p[: self.pack.n_c]]
        )

    def jacobian(self, p: np.ndarray, index: np.ndarray) -> np.ndarray:
        n, pk = index.size, self.pack
        out = np.zeros((n + pk.n_c, pk.size))
        design = self.design(*self.kernel(p))
        base, chain = self.score(p, index, design)
        out[:n, : pk.n_c] = design[index] * chain[:, None]
        out[:n, pk.i_boundary] = 1.0
        out[:n, pk.i_scale : pk.i_scale + len(pk.scaled)] = self.indicator[index]
        step = 1.0e-4
        for j in (pk.i_onset, pk.i_gap) + ((pk.i_sigma,) if pk.free_sigma else ()):
            shifted = p.copy()
            shifted[j] += step
            out[:n, j] = (self.score(shifted, index)[0] - base) / step
        self.design(*self.kernel(p))  # restore the cache to the unshifted kernel
        out[n:, : pk.n_c] = np.diag(np.sqrt(n) * self.penalty)
        return out

    def fit(self, index: np.ndarray) -> np.ndarray:
        pk = self.pack
        start = np.zeros(pk.size)
        start[0] = np.log(700.0)
        start[pk.i_onset], start[pk.i_gap] = np.log(3.0), np.log(15.0)
        lower, upper = np.full(pk.size, -np.inf), np.full(pk.size, np.inf)
        lower[pk.i_onset], upper[pk.i_onset] = np.log(1.0 / 60.0), np.log(12.0)
        lower[pk.i_gap], upper[pk.i_gap] = np.log(1.0), np.log(72.0)
        lower[pk.i_boundary], upper[pk.i_boundary] = -0.7, 0.7
        # The accelerometer-to-accelerometer and accelerometer-to-POD scales, bounded as
        # `gen_sdtm_coefficients.fit` bounds its one mission scale and for the same reason.
        scales = slice(pk.i_scale, pk.i_scale + len(pk.scaled))
        lower[scales], upper[scales] = -0.3, 0.3
        if pk.free_sigma:
            start[pk.i_sigma] = 2.0
            lower[pk.i_sigma], upper[pk.i_sigma] = 0.5, 6.0
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
        return out.x


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
    for fam in sorted(set(rows.family)):
        bins[fam] = rows.family.to_numpy() == fam
    print("== held out by mission family, MAPE")
    print(f"{'bin':<14} {'n':>8}" + "".join(f" {name:>16}" for name in columns))
    for name, mask in bins.items():
        if mask.any():
            cells = "".join(f" {g.mape(r[mask]):15.1f}%" for r in columns.values())
            print(f"{name:<14} {mask.sum():8d}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--step", type=int, default=60, help="keep every n-th 120 s corpus row"
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
    argument = parser.parse_args()
    sigma = None if argument.fit_sigma else argument.sigma

    began = time.time()
    rows = load_corpus(argument.step)
    day_ap = rows.attrs["ap"].reshape(-1, 8).mean(axis=1)
    ap_daily = day_ap[
        (rows.t.dt.normalize() - rows.attrs["ap_epoch"]).dt.days.to_numpy()
    ]
    print(
        f"{len(rows)} rows, {rows.sat.nunique()} satellites, {time.time() - began:.0f}s",
        flush=True,
    )
    arg = g.fit_arguments(rows, argument.degree)
    sat = rows.sat.to_numpy()
    caps = g.CAPS[argument.caps]

    families = argument.folds or sorted(set(FAMILY.values()))
    held = np.full(len(rows), np.nan)
    for fam in families:
        test = rows.family.to_numpy() == fam
        train_sats = tuple(s for s in SATELLITES if s in set(sat[~test]))
        model = CorpusModel(arg, sat, train_sats, caps, argument.ridge, sigma)
        t0 = time.time()
        p = model.fit(np.where(~test)[0])
        r, _ = model.score(p, np.where(test)[0])
        held[test] = r
        onset, recovery = model.kernel(p)
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
            f"kernel {onset / 3600:.2f}/{recovery / 3600:.2f} h  "
            f"boundary {np.exp(p[pk.i_boundary]):.3f}  sigma {model.shape(p) * 1e3:.4f}/km  "
            f"scales vs {train_sats[0]}: {scales}  {time.time() - t0:.0f}s",
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


if __name__ == "__main__":
    main()
