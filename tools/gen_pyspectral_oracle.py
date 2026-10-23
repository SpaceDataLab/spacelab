#!/usr/bin/env python3
"""libRadtran's Rayleigh reflectance, as an oracle for `radiative_transfer::path_reflectance`.

pyspectral (Apache-2.0, pytroll) does not compute Rayleigh scattering; it *tabulates* it, from
libRadtran runs somebody else made. That is what makes it worth having: libRadtran's DISORT is a
discrete-ordinate solver, ours is a photon Monte Carlo with a next-event estimator, and the two share
no line, no method and no author. Where they agree the agreement is evidence.

**Provenance, since a table that appeared by `pip install` is not an oracle.**

  * File: `rayleigh_lut_us-standard.h5` from the `rayleigh_only` set, as shipped by
    `pyspectral.rayleigh` and pinned by `PYSPECTRAL_ATM_CORR_LUT_RO` to **v1.0.1**.
  * Source: Zenodo, DOI **10.5281/zenodo.1288441**, "Look-up tables for atmospheric correction of
    satellite imagery in the visible spectral range" (Adam Dybbroe et al.), **CC-BY-4.0** — so the
    numbers may be quoted and redistributed with attribution, which is what this header is.
  * Content: `reflectance[81 wavelength, 96 sun-zenith secant, 19 azimuth difference,
    9 satellite-zenith secant]`, 400–800 nm in 5 nm steps, from libRadtran simulations of a
    molecular atmosphere over a black surface on the US Standard profile.
  * Units: a reflectance factor, `π L / (μ_s E_0)`, **not** a percentage. Checked against the
    single-scattering law rather than assumed: at 450 nm, `μ_s = μ_v = 1`, the table reads 0.079 265
    and the closed form at `τ_R = 0.2277` gives 0.069, the difference being the multiple scattering
    the closed form omits. A percentage would be off by two orders of magnitude.

**The table is not the pure Rayleigh atmosphere its name suggests.** `rayleigh_only` is a value of
pyspectral's *`aerosol_type`* argument — its siblings are `marine_clean_aerosol` and the rest — so it
says there is no aerosol, not that there is no molecular absorption. The atmosphere is `us-standard`,
whose profile carries the standard ozone column, and libRadtran does not drop it unless the input deck
says so. Read as a conservative Rayleigh slab the table comes back low by 1.6 % at 400 nm, **16.5 % at
600 nm** and 2.3 % at 800 nm: the Chappuis band peaks at 602 nm. `docs/verification.md` §5.1 has the
three measurements that pin it, including the out-of-sample one.

**What the comparison can and cannot control.** The table has no optical depth axis, so feeding our
model a `τ` means choosing a Rayleigh optical-depth model — a *third* implementation, whose error
would land in the residual and be read as ours. This script writes the Bodhaine et al. (1999) value
beside each wavelength and the Rust test uses it **unfitted**; the single fitted scalar per wavelength
is the ozone slant depth above the scattering layer, which is a named absorber rather than a free
amplitude, and is checked by predicting the high-air-mass half of the grid from the low. What is left
being tested is the angular shape, which is the part our radiative transfer actually computes.

Bodhaine's is the standard sea-level column depth,

    τ_R(λ) = 0.0021520 · (1.0455996 − 341.29061 λ⁻² − 0.90230850 λ²)
                       / (1 + 0.0027059889 λ⁻² − 85.968563 λ²),   λ in µm,

which is their equation for 1013.25 hPa, and the US Standard profile's surface pressure is 1013.25
hPa, so the two are directly comparable.

**A difference to expect rather than to be surprised by.** DISORT is a *scalar* solver and so is
ours: neither carries polarization. Real Rayleigh scattering is polarized, and a vector solution
differs from a scalar one by several percent in the molecular atmosphere. Both sides make the same
approximation, so it cancels here — this comparison cannot see it, and neither implementation should
be quoted as absolute truth for a polarized measurement.

    uv run --with h5py python tools/gen_pyspectral_oracle.py

writes `crates/spacelab-earthvision/tests/fixtures/pyspectral_rayleigh.txt`.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np

LUT = (
    pathlib.Path.home()
    / ".local/share/pyspectral/rayleigh_only/rayleigh_lut_us-standard.h5"
)
OUT = pathlib.Path(__file__).resolve().parents[1] / (
    "crates/spacelab-earthvision/tests/fixtures/pyspectral_rayleigh.txt"
)

# Eight wavelengths across the table's range. 400 nm is where the optical depth is largest and
# multiple scattering hardest; 800 nm is where it is smallest and the single-scattering law nearly
# exact. A comparison at one wavelength could not tell a wrong phase function from a wrong tau. 600 nm
# is here because the Chappuis band peaks at 602 nm: with it, the ozone above is a shape across eight
# points instead of one unexplained wavelength.
WAVELENGTHS_NM = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 800.0]

# Secants rather than angles because that is the table's own axis; these are exact grid points, so
# nothing here interpolates. 1.0 is overhead, 2.5 is 66.4 degrees.
SUN_SECANTS = [1.0, 1.25, 1.75, 2.5]
VIEW_SECANTS = [1.0, 1.5, 2.0, 2.5]
AZIMUTHS_DEG = [0.0, 60.0, 120.0, 180.0]


def bodhaine_1999(wavelength_um: float) -> float:
    """Rayleigh optical depth of the whole atmosphere at 1013.25 hPa (Bodhaine et al. 1999)."""
    w2 = wavelength_um * wavelength_um
    numerator = 1.0455996 - 341.29061 / w2 - 0.90230850 * w2
    denominator = 1.0 + 0.0027059889 / w2 - 85.968563 * w2
    return 0.0021520 * numerator / denominator


def index_of(axis: np.ndarray, value: float) -> int:
    """The exact grid index, refusing anything that would silently interpolate."""
    hits = np.flatnonzero(np.isclose(axis, value, rtol=0.0, atol=1e-9))
    if hits.size != 1:
        raise SystemExit(
            f"{value} is not a grid point of {axis[:4]}... ({axis.size} values)"
        )
    return int(hits[0])


def main() -> None:
    if not LUT.exists():
        raise SystemExit(
            f"{LUT} is absent. It is pyspectral's Zenodo download, not a repository file:\n"
            "  uv run --with pyspectral python -c "
            "\"from pyspectral.rayleigh import Rayleigh; Rayleigh('Suomi-NPP', 'viirs')\""
        )

    with h5py.File(LUT, "r") as handle:
        wavelengths = handle["wavelengths"][:]
        sun_secants = handle["sun_zenith_secant"][:]
        azimuths = handle["azimuth_difference"][:]
        view_secants = handle["satellite_zenith_secant"][:]
        reflectance = handle["reflectance"][:]

    lines = [
        "# libRadtran Rayleigh reflectance over a black surface, US Standard atmosphere.",
        "# pyspectral rayleigh_only LUT v1.0.1, Zenodo DOI 10.5281/zenodo.1288441, CC-BY-4.0.",
        "# Generated by tools/gen_pyspectral_oracle.py -- do not edit.",
        "#",
        "# Reflectance factor pi*L/(mu_s*E_0), dimensionless. mu = 1/secant.",
        "#",
        "# wavelength_nm  tau_bodhaine1999  mu_sun  mu_view  azimuth_deg  reflectance",
    ]
    for wavelength in WAVELENGTHS_NM:
        wi = index_of(wavelengths, wavelength)
        tau = bodhaine_1999(wavelength / 1000.0)
        for sun in SUN_SECANTS:
            si = index_of(sun_secants, sun)
            for view in VIEW_SECANTS:
                vi = index_of(view_secants, view)
                for azimuth in AZIMUTHS_DEG:
                    ai = index_of(azimuths, azimuth)
                    lines.append(
                        f"{wavelength:9.1f} {tau:17.12f} {1.0 / sun:8.6f} {1.0 / view:8.6f} "
                        f"{azimuth:11.1f} {reflectance[wi, si, ai, vi]:16.12f}"
                    )

    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(lines) - 7} rows")


if __name__ == "__main__":
    main()
