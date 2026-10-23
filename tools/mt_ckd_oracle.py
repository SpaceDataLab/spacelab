#!/usr/bin/env python3
"""How much of the water absorption the gas table is missing, measured against MT_CKD.

`gen_gas_transmittance.py` truncates every strong line's Lorentz wing at 25 cm^-1 and prints what
that truncation costs: at 940 nm the channel transmittance moves 1.90e-1 against a 500 cm^-1
reference, 7.29e-2 at 50 cm^-1 and 2.15e-2 at 100. That is not a convergence parameter set too
small. Every line-by-line code truncates at exactly 25 cm^-1 -- LBLRTM does, and MODTRAN 5.3.3,
which is what produces RadCalNet's TOA -- and then adds a **continuum** calibrated against that
truncation, because the far Lorentz wing is known to be the wrong shape: super-Lorentzian for
self-broadening, sub-Lorentzian for foreign. So the 25 cm^-1 cutoff is the correct choice and the
missing piece is the continuum, not more wing.

**This file measures the gap; it does not close it.** See the licence note below for why.

## Licence

MT_CKD is AER's. The coefficient file carries this notice, reproduced here as its licence requires:

    Copyright (c), Atmospheric and Environmental Research, Inc., 2022. All rights reserved. This
    source code was developed as part of the LBLRTM software and is designed for scientific and
    research purposes. Atmospheric and Environmental Research Inc. (AER) grants USER the right to
    download, install, use and copy this software and data for scientific and research purposes
    only. This software and data may be redistributed as long as this copyright notice is reproduced
    on any copy made and appropriate acknowledgment is given to AER. This software and data or any
    modified version of this software and data may not be incorporated into proprietary software or
    data or commercial software or data offered for sale without the express written consent of AER.
    This software and data are provided as is without any expressed or implied warranties.

Two clauses decide what may be built on it. Redistribution *is* granted -- unlike HITRAN's line
parameters -- but the grant is **field-of-use limited to scientific and research purposes**, and
**any modified version may not be incorporated into commercial software offered for sale**. A
continuum coefficient interpolated onto this repository's channel grid is plainly a modified version
of that data, so the escape hatch that works for HITRAN -- fetch at generation time, commit only our
own fit -- does **not** work here: what would be committed is still AER's numbers, rearranged.

So MT_CKD is an **oracle, never a dependency**, the same standing NRLMSIS 2.x has. The coefficient
file is cached outside the repository, nothing derived from it is written to a fixture, and what this
script emits is a measurement of *our* error, which is ours. Closing the gap needs either AER's
express written consent or a continuum whose licence permits redistribution without a field-of-use
limit.

## Provenance and method

Coefficients: `absco-ref_wv-mt-ckd.nc` from the MT_CKD_H2O 4.3 release (AER-RC/MT_CKD_H2O, tag 4.3),
2003 points on a 10 cm^-1 grid from -20 to 20000 cm^-1, self and foreign at 1013 mb and 296 K plus a
per-wavenumber temperature exponent for the self term. General reference: E. J. Mlawer et al.,
*Development and recent evaluation of the MT_CKD model of continuum absorption*, Phil. Trans. R. Soc.
A **370**, 2520 (2012), doi:10.1098/rsta.2011.0295. Acknowledgment to AER is given here and in
`docs/verification.md`.

Scaling, transcribed from the published formulation in `mt_ckd_h2o_module.f90` rather than from
memory -- the physics, not the code:

    rho_rat = (P / 1013 mb) (296 K / T)
    rad(nu) = nu (1 - e^-x) / (1 + e^-x),   x = hc nu / (k T),   -> nu when x >> 1
    k_self  = C_self^296(nu) (296/T)^texp(nu) . x_H2O    . rho_rat . rad(nu)
    k_for   = C_for^296(nu)                   (1 - x_H2O) . rho_rat . rad(nu)

both in cm^2 per molecule of water, so the layer's continuum optical depth is `(k_self + k_for)`
times that layer's water column. In the visible and near infrared `x` runs past 50, so the radiation
term is `nu` to every digit that matters and is not the interesting part; the density scaling is.

The atmosphere is `gen_gas_transmittance.profile`, imported rather than re-derived, so the layering,
the water shape and the column normalisation are the same ones the shipped table was built on. The
continuum is averaged over the same 10 nm triangular slit, which at the 10 cm^-1 coefficient spacing
is a formality -- the continuum is smooth by construction -- but it costs nothing and removes the
question.

## Invocation

    timeout 600 uv run --with numpy --with scipy python tools/mt_ckd_oracle.py

`--with` rather than `uv add`: this is a verification tool, and neither scipy nor the netCDF reader
belongs in the project's lockfile for it.
"""

from __future__ import annotations

import math
import pathlib
import sys
import urllib.error
import urllib.request

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gen_gas_transmittance import (
    LAYERS,
    PRESSURE_NODES,
    SLIT_FWHM_NM,
    WATER,
    WAVELENGTHS_NM,
    profile,
)

# The 4.3 release, pinned: a continuum is only reproducible against the version it was measured from.
RELEASE = "4.3"
SOURCE = (
    f"https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/{RELEASE}"
    "/data/absco-ref_wv-mt-ckd.nc"
)

# Outside the repository, deliberately and not incidentally -- see the licence note above.
CACHE = pathlib.Path.home() / ".cache" / "spacelab" / "mt_ckd"

# hc/k, cm K. AER's RADCN2, carried at their value so the radiation term is theirs and not a second
# opinion about physical constants.
RADCN2 = 1.4387752

# Precipitable centimetres to report at. RadCalNet's own record runs about 0.1 to 5 cm and its median
# is near 1; the shipped table's line widths are frozen at 2 cm.
COLUMNS_CM = (0.5, 1.0, 2.0, 5.0)

# The nine wavelengths the RadCalNet gates admit, nm.
WINDOWS_NM = (410.0, 440.0, 490.0, 550.0, 600.0, 660.0, 680.0, 870.0, 1030.0)


def coefficients() -> dict[str, np.ndarray]:
    """Fetch the release's coefficient file into the cache if it is not there, and read it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"absco-ref_wv-mt-ckd-{RELEASE}.nc"
    if not path.exists():
        try:
            with urllib.request.urlopen(SOURCE, timeout=60.0) as response:
                path.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            raise SystemExit(f"{SOURCE}: {error}") from error
    with netcdf_file(str(path), "r", mmap=False) as data:
        return {
            "wavenumber": np.asarray(data.variables["wavenumbers"][:], dtype=float),
            "self": np.asarray(data.variables["self_absco_ref"][:], dtype=float),
            "foreign": np.asarray(data.variables["for_absco_ref"][:], dtype=float),
            "texp": np.asarray(data.variables["self_texp"][:], dtype=float),
            "reference_hpa": float(data.variables["ref_press"][()]),
            "reference_k": float(data.variables["ref_temp"][()]),
        }


def slit_nodes(wavelength_nm: float, points: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Wavenumbers and normalised weights of the 10 nm triangular slit centred on `wavelength_nm`."""
    offsets = np.linspace(-SLIT_FWHM_NM, SLIT_FWHM_NM, points)
    weight = np.maximum(0.0, 1.0 - np.abs(offsets) / SLIT_FWHM_NM)
    weight /= weight.sum()
    return 1.0e7 / (wavelength_nm + offsets), weight


def radiation(wavenumber: np.ndarray, temperature: float) -> np.ndarray:
    """`nu tanh(hc nu / 2kT)`, in AER's own branch form so the small-`x` limit is theirs too."""
    x = wavenumber * RADCN2 / temperature
    small = x <= 0.01
    out = np.where(small, 0.5 * x * wavenumber, wavenumber)
    warm = (~small) & (x <= 10.0)
    if warm.any():
        decay = np.exp(-x[warm])
        out[warm] = wavenumber[warm] * (1.0 - decay) / (1.0 + decay)
    return out


def continuum_depth(
    table: dict[str, np.ndarray],
    surface_hpa: float,
    column_cm: float,
    wavelength_nm: float,
) -> float | None:
    """Slit-averaged continuum optical depth of a vertical column at one channel.

    `None` where the channel falls outside the coefficients' own spectral domain, which is the answer
    for every wavelength below 500 nm and is a finding rather than a gap to paper over.
    """
    nodes, weight = slit_nodes(wavelength_nm)
    lo, hi = table["wavenumber"][0], table["wavenumber"][-1]
    if nodes.min() < lo or nodes.max() > hi:
        return None

    layers = profile(LAYERS, surface_hpa, reference_cm=column_cm)
    total = 0.0
    for pressure_hpa, temperature, air_column, _share, water_vmr in layers:
        density = (pressure_hpa / table["reference_hpa"]) * (
            table["reference_k"] / temperature
        )
        warm = (table["reference_k"] / temperature) ** table["texp"]
        self_ref = np.interp(nodes, table["wavenumber"], table["self"] * warm)
        foreign_ref = np.interp(nodes, table["wavenumber"], table["foreign"])
        rad = radiation(nodes, temperature)
        cross_section = (
            (self_ref * water_vmr + foreign_ref * (1.0 - water_vmr)) * density * rad
        )
        total += float(np.dot(weight, cross_section)) * air_column * water_vmr
    return total


def main() -> None:
    table = coefficients()
    print(
        f"# MT_CKD_H2O {RELEASE} as an oracle: what the 25 cm^-1 line-wing cutoff leaves out.\n"
        f"# Coefficients (c) Atmospheric and Environmental Research, Inc., 2022, used under their\n"
        f"# research grant and cached outside this repository. Nothing derived from them is written\n"
        f"# to a fixture -- see the licence note in this file.\n"
        f"# Domain: {table['wavenumber'][0]:.0f} to {table['wavenumber'][-1]:.0f} cm^-1 "
        f"({1.0e7 / table['wavenumber'][-1]:.0f} nm and longer), reference "
        f"{table['reference_hpa']:.0f} hPa / {table['reference_k']:.0f} K."
    )

    surface = PRESSURE_NODES[1]
    total_column = WATER.molecules_per_unit
    print(
        f"\n# vertical continuum optical depth at {surface:.2f} hPa, and the relative change it\n"
        f"# makes to a channel transmittance of unity -- the same quantity the cutoff ladder reports"
    )
    header = "  ".join(f"{c:.1f} cm".rjust(11) for c in COLUMNS_CM)
    print(f"{'nm':>8}  {header}")
    outside = []
    for wavelength_nm in WAVELENGTHS_NM:
        depths = [
            continuum_depth(table, surface, column_cm, wavelength_nm)
            for column_cm in COLUMNS_CM
        ]
        if any(depth is None for depth in depths):
            outside.append(wavelength_nm)
            continue
        if max(depths) < 1.0e-6:
            continue
        cells = "  ".join(f"{1.0 - math.exp(-d):11.4e}" for d in depths)
        mark = " <- gate window" if wavelength_nm in WINDOWS_NM else ""
        print(f"{wavelength_nm:8.0f}  {cells}{mark}")

    if outside:
        print(
            f"# {len(outside)} channels lie above the coefficients' 20000 cm^-1 ceiling and have no\n"
            f"# MT_CKD continuum at all: {outside[0]:.0f}-{outside[-1]:.0f} nm, which includes the\n"
            f"# {sum(1 for w in WINDOWS_NM if w in outside)} gate windows "
            f"{', '.join(f'{w:.0f}' for w in WINDOWS_NM if w in outside)} nm"
        )
    print(f"# one precipitable cm is {total_column:.4e} molecules/cm^2")

    print("\n# the same channels across the table's own surface-pressure axis, at 2 cm")
    nodes = "  ".join(f"{p:.0f}".rjust(11) for p in PRESSURE_NODES)
    print(f"{'nm':>8}  {nodes}")
    for wavelength_nm in WINDOWS_NM:
        depths = [
            continuum_depth(table, node_hpa, 2.0, wavelength_nm)
            for node_hpa in PRESSURE_NODES
        ]
        if any(depth is None for depth in depths):
            continue
        cells = "  ".join(f"{1.0 - math.exp(-d):11.4e}" for d in depths)
        print(f"{wavelength_nm:8.0f}  {cells}")

    summary(table, surface)


def summary(table: dict[str, np.ndarray], surface_hpa: float) -> None:
    """The two numbers the verdict rests on, computed here rather than read off the tables above."""
    covered = [
        (w, continuum_depth(table, surface_hpa, 2.0, w))
        for w in WINDOWS_NM
        if continuum_depth(table, surface_hpa, 2.0, w) is not None
    ]
    worst_nm, worst_depth = max(covered, key=lambda pair: pair[1])
    print(
        f"\n# worst gate window, at 2 cm: {worst_nm:.0f} nm, "
        f"{1.0 - math.exp(-worst_depth):.4e} of channel transmittance"
    )
    # The cutoff ladder the fixture header already charges, at the same channel, from #353: the
    # worst relative change in channel transmittance against a 500 cm^-1 reference.
    ladder = {25.0: 1.90e-1, 50.0: 7.29e-2, 100.0: 2.15e-2}
    print(
        "# against the cutoff ladder the header already charges at 940 nm -- if the continuum is\n"
        "# what the truncation stands in for, these are the same size, and that is the check"
    )
    for column_cm in (2.0, 5.0):
        depth = continuum_depth(table, surface_hpa, column_cm, 940.0)
        assert depth is not None
        missing = 1.0 - math.exp(-depth)
        nearest = min(ladder, key=lambda c: abs(math.log(ladder[c] / missing)))
        print(
            f"#   940 nm at {column_cm:.0f} cm: continuum {missing:.4e}, "
            f"nearest rung {ladder[nearest]:.2e} at {nearest:.0f} cm^-1 "
            f"({missing / ladder[nearest]:.2f}x)"
        )


if __name__ == "__main__":
    main()
