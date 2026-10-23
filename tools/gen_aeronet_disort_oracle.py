#!/usr/bin/env python3
"""DISORT's coupling terms on the AERONET atmospheres, as the reference for `coupling_from_slab`.

`tools/gen_aeronet_oracle.py` supplies the measured atmosphere; this supplies the independent solve
on it. DISORT (Stamnes et al.) decomposes the transfer equation azimuthally and solves it on a Gauss
quadrature by eigen-decomposition, where `spacelab_earthvision` traces photons -- no shared line, no
shared method, so the residual is solver-vs-solver on an atmosphere neither of them chose.

Run ephemerally -- pydisort is an oracle, never a dependency of this workspace:

    timeout 1800 uv run --with pydisort python tools/gen_aeronet_disort_oracle.py

**The scatterer is analytic on both sides.** AERONET reports an asymmetry parameter, so the phase
function is Henyey-Greenstein, whose Legendre moments are `g**l` exactly -- DISORT takes the moments,
the Rust takes the closed form, and neither has to imitate the other's representation. The fixture's
largest asymmetry is `g = 0.8255`, above the 0.85 case that set `gen_disort_oracle.py`'s moment count
only in the sense that it is close to it, so the truncation was measured rather than assumed: summing
`(2l+1) g**l P_l(mu)/2` over 20 001 points of `mu` against the closed form gives **4.9e-6 absolute
and 1.7e-5 relative** at 96 moments there (1.8e-15 at `g = 0.45`, 5.1e-14 at `g = 0.68`), falling to
3.8e-11 at 160 moments. Four orders below the residual this file is used to measure.

**The radiance convention was pinned by measurement, not by reading the manual.** With
`usrtau,usrang` and no `onlyfl`, `Disort.gather_rad()` returns the intensity at the user angles, and
`pi * L / mu_0` is the reflectance factor `path_reflectance` computes. Against the analytic
single-scattering limit at `mu_0 = 0.8`, `mu_v = 0.6` and azimuths 0, 90 and 180 degrees, the ratio
goes to 1 as the layer thins -- 1.0509 at `tau = 0.02`, 1.0072 at 0.002 and 1.00086 at 0.0002 for
isotropic scattering, and 1.0887 / 1.0119 / 1.0014 at `g = 0.7` -- which is the multiple scattering
DISORT includes and the single-scattering formula omits, and it also fixes the azimuth origin: a
convention off by 180 degrees would not converge at all three azimuths at once.

**pydisort 1.8.5's `intensity_correction` flag returns NaN** at every stream count tried (16, 32,
96), so it is not used, and DISORT prints its own warning that intensities may be less accurate
without it. That warning is the reason the stream count below was measured rather than argued.

**96 streams is not a guess (#327).** The flag that would correct a truncated forward peak is off, so
the reference was re-solved on all of these atmospheres at 32, 64, 96, 160 and 256 streams and the
path reflectance compared against the 256-stream answer:

    32 streams   worst 3.022e-03   rms 5.875e-04
    64 streams   worst 8.542e-06   rms 9.009e-07
    96 streams   worst 1.666e-07   rms 2.094e-08
    160 streams  worst 2.420e-08   rms 2.643e-09

The oracle is converged four orders of magnitude below the Monte Carlo noise it is compared against,
at the fixture's hardest column (`tau = 3.778`, `g = 0.826`) as much as anywhere else. Whatever a
residual against this file contains, it is not the stream count -- and 32 streams would have hidden a
3e-3 error inside one, which is why the number is written down.

**Two spherical albedos, because one of them measures our quadrature rather than our solver.**
`coupling_from_slab` integrates the flux reflectance as `int_0^1 R(mu) 2 mu dmu` on a four-node
Gauss-Legendre rule. The reference is computed on *the same four nodes*, so the residual is the
Monte Carlo's; a sixteen-node value is written beside it so the four-node truncation can be read
separately instead of being blamed on the sampler.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
from pydisort import Disort, DisortOptions, kIDN, kIUP

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / (
    "crates/spacelab-earthvision/tests/fixtures"
)
SOURCE = FIXTURES / "aeronet_atmosphere.txt"
OUT = FIXTURES / "aeronet_coupling_disort.txt"

NSTR = 96
NMOM = 96
DTYPE = torch.float64

# The view direction the coupling is reported at: a 36.9 degree view zenith, and the three azimuths
# that separate the forward-scattering half-plane from the backscattering one.
MU_VIEW = 0.8
AZIMUTHS_DEG = (0.0, 90.0, 180.0)

# The four-node Gauss-Legendre rule on [0, 1] that `coupling_from_slab` uses, verbatim.
GAUSS_01 = (
    (0.06943184420297367, 0.1739274225687269),
    (0.33000947820757185, 0.32607257743127305),
    (0.6699905217924281, 0.32607257743127305),
    (0.9305681557970263, 0.1739274225687269),
)


def gauss_01(nodes: int) -> list[tuple[float, float]]:
    """Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    return [
        (0.5 * (node + 1.0), 0.5 * weight) for node, weight in zip(x, w, strict=True)
    ]


def moments(g: float) -> list[float]:
    return [g**level for level in range(1, NMOM + 1)]


def solve(
    tau: float, ssa: float, g: float, mu0: float, azimuths_deg: tuple[float, ...]
) -> tuple[float, float, list[float]]:
    """Flux reflectance, total transmittance, and the TOA reflectance factor at each azimuth.

    One run answers all three: the fluxes at the two user optical depths give `R` and `T`, and the
    upward intensity at the top gives the path reflectance.
    """
    options = (
        DisortOptions()
        .flags("usrtau,usrang,lamber,quiet")
        .nwave(1)
        .ncol(1)
        .user_tau([0.0, tau])
        .user_mu([MU_VIEW])
        .user_phi(list(azimuths_deg))
    )
    state = options.ds()
    state.nlyr = 1
    state.nstr = NSTR
    state.nmom = NMOM
    state.nphase = NMOM
    disort = Disort(options)

    prop = torch.zeros((1, 1, 1, 2 + NMOM), dtype=DTYPE)
    prop[0, 0, 0, 0] = tau
    prop[0, 0, 0, 1] = ssa
    for index, moment in enumerate(moments(g)):
        prop[0, 0, 0, 2 + index] = moment

    flux = disort.forward(
        prop,
        umu0=torch.tensor([mu0], dtype=DTYPE),
        phi0=torch.tensor([0.0], dtype=DTYPE),
        fbeam=torch.tensor([[1.0]], dtype=DTYPE),
        albedo=torch.tensor([[0.0]], dtype=DTYPE),
    )
    radiance = disort.gather_rad().reshape(len(azimuths_deg), -1)[:, 0].tolist()

    incident = mu0
    reflectance = float(flux[0, 0, 0, kIUP]) / incident
    transmittance = float(flux[0, 0, -1, kIDN]) / incident
    return reflectance, transmittance, [math.pi * value / mu0 for value in radiance]


def flux_reflectance(tau: float, ssa: float, g: float, mu0: float) -> float:
    return solve(tau, ssa, g, mu0, (0.0,))[0]


def spherical_albedo(
    tau: float, ssa: float, g: float, rule: list[tuple[float, float]]
) -> float:
    return sum(
        weight * flux_reflectance(tau, ssa, g, mu) * 2.0 * mu for mu, weight in rule
    )


def rows() -> list[tuple[str, ...]]:
    parsed: list[tuple[str, ...]] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parsed.append(tuple(line.split()))
    return parsed


def main() -> None:
    gauss16 = gauss_01(16)
    header = [
        "# DISORT coupling terms on the AERONET Level 2.0 atmospheres of aeronet_atmosphere.txt.",
        "# Generated by tools/gen_aeronet_disort_oracle.py -- do not edit.",
        f"# pydisort, {NSTR} streams and {NMOM} Legendre moments; Henyey-Greenstein moments g**l.",
        f"# View cosine {MU_VIEW}; azimuths "
        + ", ".join(f"{azimuth:g}" for azimuth in AZIMUTHS_DEG)
        + " degrees from the forward-scattering half-plane.",
        "# spherical_albedo_g4 uses the same four Gauss-Legendre nodes as coupling_from_slab, so the",
        "# residual against it is the Monte Carlo's; spherical_albedo_g16 bounds the four-node",
        "# truncation separately.",
        "#",
        "# site  date  time_utc  wavelength_nm  transmittance_down  transmittance_up"
        "  spherical_albedo_g4  spherical_albedo_g16  path_reflectance_"
        + "  path_reflectance_".join(f"{azimuth:g}deg" for azimuth in AZIMUTHS_DEG),
    ]

    out: list[str] = []
    for (
        site,
        _regime,
        date,
        time,
        sza_deg,
        nm,
        aod,
        ssa,
        asymmetry,
        _water,
        _ozone,
    ) in rows():
        tau = float(aod)
        omega = float(ssa)
        g = float(asymmetry)
        mu_sun = math.cos(math.radians(float(sza_deg)))

        _, t_down, path = solve(tau, omega, g, mu_sun, AZIMUTHS_DEG)
        _, t_up, _ = solve(tau, omega, g, MU_VIEW, (0.0,))
        albedo4 = spherical_albedo(tau, omega, g, list(GAUSS_01))
        albedo16 = spherical_albedo(tau, omega, g, gauss16)

        out.append(
            f"{site:16s} {date} {time} {int(nm):6d} {t_down:.9f} {t_up:.9f} "
            f"{albedo4:.9f} {albedo16:.9f} "
            + " ".join(f"{value:.9f}" for value in path)
        )

    OUT.write_text("\n".join(header + out) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(out)} rows")


if __name__ == "__main__":
    main()
