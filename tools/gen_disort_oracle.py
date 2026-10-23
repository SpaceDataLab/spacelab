"""Slab reflectance and transmittance from DISORT, as an oracle for the earthvision Monte Carlo.

DISORT (Stamnes et al.) is the discrete-ordinates reference implementation of plane-parallel
radiative transfer.  It shares no method with `spacelab_earthvision::monte_carlo_slab`: it solves the
azimuthally decomposed transfer equation on a Gauss quadrature by eigen-decomposition, where the
Monte Carlo traces photons.  A Monte Carlo agreeing with itself is not a verification; this is what
makes the comparison one.

Run ephemerally -- pydisort is an oracle, never a dependency of this workspace:

    timeout 900 uv run --with pydisort python tools/gen_disort_oracle.py

The phase function is given to DISORT as Legendre moments, which is the only form it accepts, and the
same moments are what the Rust test builds its phase function from, so the two codes see the
identical scatterer and the residual is solver-vs-solver.
"""

import torch
from pydisort import Disort, DisortOptions, kIDN, kIUP

# 96 streams and 96 moments.  The moment count is set by the sharpest case: Henyey-Greenstein
# moments are g**l, so g = 0.85 needs l > 85 before the truncated expansion is within 1e-6 of the
# exact phase function.  That matters because the Rust side uses the *analytic* phase function, not a
# truncated expansion -- keeping the truncation below the comparison tolerance is what makes the two
# codes see the same scatterer without either having to imitate the other's representation.
NSTR = 96
NMOM = 96


def henyey_greenstein_moments(g: float, n: int) -> list[float]:
    """Legendre moments of the Henyey-Greenstein phase function are g**l, exactly."""
    return [g**level for level in range(1, n + 1)]


def two_lobe_moments(forward: float, backward: float, weight: float, n: int) -> list[float]:
    return [
        weight * forward**level + (1.0 - weight) * backward**level for level in range(1, n + 1)
    ]


def slab(tau: float, ssa: float, moments: list[float], mu0: float) -> tuple[float, float]:
    """Flux reflectance and total transmittance of one homogeneous layer over a black surface."""
    options = DisortOptions().flags("onlyfl,lamber,quiet")
    state = options.ds()
    state.nlyr = 1
    state.nstr = NSTR
    state.nmom = len(moments)
    state.nphase = len(moments)
    disort = Disort(options)

    prop = torch.zeros((1, 1, 1, 2 + len(moments)), dtype=torch.float64)
    prop[0, 0, 0, 0] = tau
    prop[0, 0, 0, 1] = ssa
    for index, moment in enumerate(moments):
        prop[0, 0, 0, 2 + index] = moment

    flux = disort.forward(
        prop,
        umu0=torch.tensor([mu0], dtype=torch.float64),
        fbeam=torch.tensor([[1.0]], dtype=torch.float64),
        albedo=torch.tensor([[0.0]], dtype=torch.float64),
    )
    # With `onlyfl` the last axis is (upward, downward) flux and the level axis is (top, bottom);
    # the downward flux at the bottom already carries the unscattered beam, which is what the Monte
    # Carlo's transmittance counts too.  The incident vertical flux is `mu0 * fbeam`.
    incident = mu0
    up = float(flux[0, 0, 0, kIUP])
    down = float(flux[0, 0, -1, kIDN])
    return up / incident, down / incident


CASES = [
    ("isotropic", 0.0, 1.0, 1.0, [0.0] * NMOM),
    ("isotropic", 1.0, 1.0, 1.0, [0.0] * NMOM),
    ("isotropic", 1.0, 0.9, 0.6, [0.0] * NMOM),
    ("hg g=0.5", 2.0, 1.0, 0.8, henyey_greenstein_moments(0.5, NMOM)),
    ("hg g=0.85", 4.0, 0.99, 0.7, henyey_greenstein_moments(0.85, NMOM)),
    ("hg g=-0.4", 1.5, 0.85, 1.0, henyey_greenstein_moments(-0.4, NMOM)),
    ("two-lobe", 3.0, 0.95, 0.75, two_lobe_moments(0.85, -0.6, 0.6, NMOM)),
]


def main() -> None:
    print(f"# NSTR = {NSTR}, NMOM = {NMOM}")
    for name, tau, ssa, mu0, moments in CASES:
        reflectance, transmittance = slab(tau, ssa, moments, mu0)
        print(
            f"{name:12s} tau={tau:<5} ssa={ssa:<5} mu0={mu0:<5} "
            f"R={reflectance:.9f} T={transmittance:.9f} sum={reflectance + transmittance:.9f}"
        )


if __name__ == "__main__":
    main()
