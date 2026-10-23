"""Validate the spacelab FES ocean-tide admittance weights against pyfes (the FES oracle).

The Rust runtime infers each minor wave's geopotential coefficient as a *real*-weighted sum of the
reference majors, `coef(minor) = Σ Wₖ·coef(majorₖ)` (weights from `ocean_tide_admittance.rs`, generated
by `derivations/ocean_tide_admittance.py`). Those same real weights are what FES/pyfes applies to the
complex *elevation* amplitudes, so we validate them by driving pyfes with a fixed set of reference-major
complex amplitudes, running its inference, and recovering each inferred minor's complex amplitude.

pyfes exposes no standalone "apply inference" call — inference runs inside
`evaluate_tide_from_constituents`. So we recover each minor amplitude by least squares:

  1. Run the prediction twice: once with only the reference majors provided (inference fills the minors),
     once with the minors additionally provided as (0,0) — marking them *modeled*, which makes inference
     skip them. The difference time series is exactly the sum of the inferred minors.
  2. Build the per-wave design matrix from `WaveTable.compute_nodal_modulations` (nodal amplitude f and
     argument v+u), and least-squares solve each minor's complex amplitude. The recovery is exact
     (condition number ~1e2, fit residual ~1e-13) because the arguments are analytic and independent.

The recovered pyfes amplitudes are compared to the algorithm weights (`coef(minor)=Σ Wₖ·Zₖ`). SPLINE runs
on the Darwin engine, LINEAR on the Doodson (PERTH) engine — matching aviso-fes' `SplineInference` /
`PerthInference`. The script prints the max |Δ| per scheme and the Rust-test golden (minor Doodson, Z).

    uv run --with pyfes python experiments/ocean_tide_admittance_verify.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pyfes

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "derivations"))
import ocean_tide_admittance as gen

# Fixed reference-major complex amplitudes (arbitrary distinct values; the map is linear in them).
# Q1..S2 reuse the aviso-fes tests/library/inference.cpp inputs so LINEAR short-period == that golden.
MAJOR_TIDE = {
    "Q1": 2.044581413269043 - 2.3776917457580566j,
    "O1": 7.7220735549926758 + 1.2257133722305298j,
    "K1": -7.9487228393554688 + 5.2526679039001465j,
    "N2": -18.638496398925781 + 3.0053455829620361j,
    "M2": -90.521110534667969 - 20.603012084960938j,
    "S2": -24.581066131591797 - 25.664165496826172j,
    "K2": -6.0 - 7.0j,
    "2N2": -2.5 + 1.3j,
    "Node": 1.2 + 0.0j,
    "Mm": 3.1 - 1.1j,
    "Mf": 5.7 + 2.2j,
}

FORMULAE = pyfes.Formulae.SCHUREMAN_ORDER_1


def _amp_phase(z: complex) -> tuple[float, float]:
    """Complex tide Z = A·e^{-iG} → (amplitude A, phase G in degrees), pyfes' convention."""
    return abs(z), np.degrees(np.arctan2(-z.imag, z.real)) % 360.0


def _recover(engine, inference, weight_tables) -> dict[str, complex]:
    """Least-squares-recover each inferred minor's complex amplitude from pyfes."""
    wt = pyfes.wave_table_factory(engine)
    names = list(wt.constituents)
    ref_names = sorted({r for refs in weight_tables.values() for r in refs if r in names})
    minor_names = [m for m in weight_tables if m in names]

    epochs = np.datetime64("2024-01-01") + np.arange(0, 24 * 90) * np.timedelta64(1, "h")
    settings = (
        pyfes.Settings()
        .with_engine_type(engine)
        .with_inference_type(inference)
        .with_compute_long_period_equilibrium(False)
    )
    provided = {r: _amp_phase(MAJOR_TIDE[r]) for r in ref_names}
    tide_inf, lp_inf = pyfes.evaluate_tide_from_constituents(provided, epochs, 0.0, settings=settings)
    blocked = dict(provided)
    blocked.update(dict.fromkeys(minor_names, (0.0, 0.0)))
    tide_blk, lp_blk = pyfes.evaluate_tide_from_constituents(blocked, epochs, 0.0, settings=settings)

    f, vu = wt.compute_nodal_modulations(epochs, FORMULAE)
    # short-period minors live in the `tide` channel, long-period in the `lp` channel; use pyfes'
    # own wave-type classification (the routing it actually applies), not our H-map membership.
    def band(name: str) -> str:
        return "long_period" if wt[name].type == pyfes.WaveType.LONG_PERIOD else "short"

    out: dict[str, complex] = {}
    for chan_name, chan_diff in (("short", tide_inf - tide_blk), ("long_period", lp_inf - lp_blk)):
        chan_minors = [m for m in minor_names if band(m) == chan_name]
        if not chan_minors:
            continue
        cols = []
        for m in chan_minors:
            i = names.index(m)
            cols.append(f[i] * np.cos(vu[i]))
            cols.append(-f[i] * np.sin(vu[i]))
        design = np.column_stack(cols)
        sol, _res, _rk, sv = np.linalg.lstsq(design, chan_diff, rcond=None)
        fit = np.max(np.abs(design @ sol - chan_diff))
        print(f"    [{inference} / {chan_name}] recover cond={sv[0] / sv[-1]:.1e} "
              f"fit_resid={fit:.1e}  ({len(chan_minors)} minors)")
        for j, m in enumerate(chan_minors):
            out[m] = sol[2 * j] + 1j * sol[2 * j + 1]
    return out


def _algo_minor(weights: dict[str, float]) -> complex:
    return sum(w * MAJOR_TIDE[ref] for ref, w in weights.items())


def main() -> None:
    freq = gen._load_pyfes()[0]
    doodson6 = gen._load_pyfes()[1]

    # Build the algorithm weight tables exactly as the Rust generator does.
    d_w = gen.linear_weights(freq, gen.H_DIURNAL, gen.LINEAR_DIURNAL_REFS, "diurnal")
    s_w = gen.linear_weights(freq, gen.H_SEMIDIURNAL, gen.LINEAR_SEMIDIURNAL_REFS, "semidiurnal")
    lp_w = gen.linear_weights(freq, gen.H_LONG_PERIOD, gen.LINEAR_LONG_PERIOD_REFS, "long_period")
    linear_all = {**d_w, **s_w, **lp_w}
    linear_tbl = {m: w for m, w in linear_all.items() if doodson6(m) not in gen.FES2004_EXPLICIT}
    spline_tbl = {m: w for m, w in gen.SPLINE_WEIGHTS.items() if doodson6(m) not in gen.FES2004_EXPLICIT}

    goldens = {}
    for label, engine, inference, tbl in (
        ("SPLINE", pyfes.EngineType.DARWIN, pyfes.InferenceType.SPLINE, spline_tbl),
        ("LINEAR", pyfes.EngineType.DOODSON, pyfes.InferenceType.LINEAR, linear_tbl),
    ):
        print(f"\n=== {label} ===")
        recovered = _recover(engine, inference, tbl)
        max_err = 0.0
        golden = []
        for minor in sorted(tbl, key=lambda m: doodson6(m)):
            algo = _algo_minor(tbl[minor])
            pyf = recovered.get(minor)
            if pyf is None:
                print(f"    {minor:8s} NOT RECOVERED (absent from engine table)")
                continue
            err = abs(algo - pyf)
            max_err = max(max_err, err)
            golden.append((minor, doodson6(minor), pyf))
        print(f"    max |algo - pyfes| = {max_err:.3e}  over {len(golden)} inferred minors")
        goldens[label] = golden

    print("\n// ---- Rust-test golden (minor, Doodson, pyfes-recovered Re, Im) ----")
    for label in ("SPLINE", "LINEAR"):
        print(f"// {label}")
        for minor, doo, z in goldens[label]:
            print(f"({doo}, {z.real:+.10e}, {z.imag:+.10e}), // {minor}")


if __name__ == "__main__":
    main()
