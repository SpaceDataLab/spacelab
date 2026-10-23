#!/usr/bin/env python3
"""A numerically integrated air mass and refraction, as an oracle for the published fits.

`spacelab_coverage::observability` carries three published fits to the same two physical
quantities: Kasten & Young (1989) and Young (1994) for the relative optical air mass, and
Sæmundsson (1986) / Bennett (1982) for astronomical refraction. They were verified against each
other, which is worth something — Kasten & Young and Young 1994 are independent fits in *different*
arguments, and joining them through Sæmundsson makes a real cross-check. What it cannot do is
detect an error the fits share, and all four were fitted to the same atmosphere.

This computes both quantities from first principles instead, so a shared error has somewhere to
show. Nothing here is a fit and nothing here is a table lookup:

  * **U.S. Standard Atmosphere 1976**, the seven-layer piecewise-linear-lapse-rate hydrostatic
    formulation up to 86 km geopotential, evaluated from its published constants. The layers are
    the standard's own; the density comes out of the ideal gas law rather than out of a table.
  * **Gladstone–Dale**, `n(r) − 1 = (n₀ − 1)·ρ(r)/ρ₀`. The sea-level refractivity is Ciddor's
    (1996) value for dry air at 550 nm, 15 °C and 101 325 Pa.
  * **The ray invariant of a spherically symmetric medium**, `n r sin ψ = const` — the Bouguer/Snell
    generalisation. Everything else is one quadrature each:

        X(z) = ∫ ρ ds / ∫ ρ dr|_zenith,   ds/dr = [1 − (n₀r₀ sin z / n r)²]^(−1/2)
        R(z) = −∫ (1/n)(dn/dr) tan ψ dr

`z` here is the **apparent** zenith angle at the observer, which is the direction a telescope is
pointed; Kasten & Young's fit is in that argument. Young 1994's is in the true zenith angle, which
is `z + R(z)` — so the same integrated ray serves both fits, indexed two ways, and that is the
whole point of computing the refraction alongside the air mass rather than borrowing it from one of
the fits under test.

**The singular integrand.** At `z = 90°` the ray is horizontal at the observer, `ds/dr` diverges as
`(r − r₀)^(−1/2)`, and the integral is finite but no quadrature rule will find it. The substitution
`r = r₀ + u²` cancels the singularity exactly: `dr = 2u du` and the integrand goes as `1/u`, so the
product is bounded. Both quadratures use it at every zenith angle, not only at the horizon, so
there is no branch that runs at one angle and not another.

**Two choices this cannot make for itself**, both reported in the fixture header rather than
buried:

  * The Earth radius. A refracted ray's path length depends on the curvature it is bent around, and
    "the Earth's radius" is not one number. This uses the mean 6 371.0 km and reports what the
    equatorial and the standard's own geopotential radius would have given at the horizon.
  * Where the atmosphere ends. Integrating to 86 km leaves out a column; the fraction is measured
    and printed, not assumed negligible.

    uv run --with numpy --with scipy python tools/gen_airmass_oracle.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.integrate import quad

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crates/spacelab-coverage/tests/fixtures/airmass_integrated.txt"
)

# U.S. Standard Atmosphere 1976, Part 1 Table 4 and §1.3.1.
G0 = 9.80665  # m s^-2, standard gravity
R_STAR = 8.314_32e3  # N m kmol^-1 K^-1
M0 = 28.964_4  # kg kmol^-1, mean molecular weight of dry air
T0 = 288.15  # K
P0 = 101_325.0  # Pa
R_GEOPOTENTIAL = 6_356_766.0  # m, the standard's effective Earth radius

# (base geopotential height m, base temperature K, lapse rate K/m)
LAYERS = [
    (0.0, 288.15, -6.5e-3),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 1.0e-3),
    (32_000.0, 228.65, 2.8e-3),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -2.8e-3),
    (71_000.0, 214.65, -2.0e-3),
]
TOP_GEOPOTENTIAL = (
    84_852.0  # m, the top of the 1976 standard's lower-atmosphere formulation
)

# Ciddor (1996) for dry air, 550 nm, 15 °C, 101 325 Pa, 0 ppm CO2.
REFRACTIVITY_SEA_LEVEL = 2.778_7e-4

EARTH_RADIUS = 6_371_000.0  # m, mean
EARTH_RADIUS_VARIANTS = {
    "mean 6371.0": 6_371_000.0,
    "equatorial 6378.137": 6_378_137.0,
    "USSA geopotential 6356.766": 6_356_766.0,
}


def layer_base_pressures() -> list[float]:
    """Pressure at the base of each layer, integrated up from sea level."""
    pressures = [P0]
    for i in range(len(LAYERS) - 1):
        h_b, t_b, lapse = LAYERS[i]
        h_next = LAYERS[i + 1][0]
        p_b = pressures[-1]
        if lapse == 0.0:
            p = p_b * np.exp(-G0 * M0 * (h_next - h_b) / (R_STAR * t_b))
        else:
            p = p_b * (t_b / (t_b + lapse * (h_next - h_b))) ** (
                G0 * M0 / (R_STAR * lapse)
            )
        pressures.append(p)
    return pressures


BASE_PRESSURES = layer_base_pressures()


def density(geometric_height: float) -> float:
    """Mass density, kg m^-3, at a geometric height above sea level."""
    if geometric_height < 0.0:
        return density(0.0)
    # Geometric to geopotential, the standard's own conversion.
    h = R_GEOPOTENTIAL * geometric_height / (R_GEOPOTENTIAL + geometric_height)
    if h >= TOP_GEOPOTENTIAL:
        return 0.0
    index = 0
    for i, (h_b, _, _) in enumerate(LAYERS):
        if h >= h_b:
            index = i
    h_b, t_b, lapse = LAYERS[index]
    p_b = BASE_PRESSURES[index]
    t = t_b + lapse * (h - h_b)
    if lapse == 0.0:
        p = p_b * np.exp(-G0 * M0 * (h - h_b) / (R_STAR * t_b))
    else:
        p = p_b * (t_b / t) ** (G0 * M0 / (R_STAR * lapse))
    return p * M0 / (R_STAR * t)


DENSITY_SEA_LEVEL = density(0.0)
TOP_GEOMETRIC = R_GEOPOTENTIAL * TOP_GEOPOTENTIAL / (R_GEOPOTENTIAL - TOP_GEOPOTENTIAL)


def refractive_index(geometric_height: float) -> float:
    return 1.0 + REFRACTIVITY_SEA_LEVEL * density(geometric_height) / DENSITY_SEA_LEVEL


def zenith_column(earth_radius: float) -> float:
    """`∫ ρ dr` straight up — the normalisation the relative air mass is relative to."""
    del earth_radius  # the vertical column does not see the curvature
    value, _ = quad(density, 0.0, TOP_GEOMETRIC, limit=400)
    return value


def airmass(apparent_zenith: float, earth_radius: float, column: float) -> float:
    """Relative optical air mass along the refracted ray leaving at `apparent_zenith` radians."""
    r0 = earth_radius
    invariant = refractive_index(0.0) * r0 * np.sin(apparent_zenith)

    def integrand(u: float) -> float:
        height = u * u
        r = r0 + height
        n = refractive_index(height)
        ratio = invariant / (n * r)
        # Rounding can put `ratio` a few ulp past 1 at the horizon, where the true value is exactly
        # 1 at `u = 0`. Clamping is the continuation of the ray, not a repair of the physics.
        cos_squared = max(1.0 - ratio * ratio, 0.0)
        if cos_squared == 0.0:
            # The `2u` from the substitution and the `u` in the expansion cancel; the limit is
            # finite and this endpoint has measure zero for the rule, so it never actually runs.
            return 0.0
        return density(height) * 2.0 * u / np.sqrt(cos_squared)

    top = np.sqrt(TOP_GEOMETRIC)
    value, _ = quad(integrand, 0.0, top, limit=800)
    return value / column


def refraction(apparent_zenith: float, earth_radius: float) -> float:
    """Total astronomical refraction, radians: `z_true − z_apparent`."""
    r0 = earth_radius
    invariant = refractive_index(0.0) * r0 * np.sin(apparent_zenith)
    scale = REFRACTIVITY_SEA_LEVEL / DENSITY_SEA_LEVEL

    def dn_dr(height: float) -> float:
        step = (
            1.0  # metre; the density profile is piecewise smooth on a kilometre scale
        )
        return (
            scale
            * (density(height + step) - density(max(height - step, 0.0)))
            / (2.0 * step)
        )

    def integrand(u: float) -> float:
        height = u * u
        r = r0 + height
        n = refractive_index(height)
        ratio = invariant / (n * r)
        cos_squared = max(1.0 - ratio * ratio, 0.0)
        if cos_squared == 0.0:
            return 0.0
        tan_psi = ratio / np.sqrt(cos_squared)
        return -(dn_dr(height) / n) * tan_psi * 2.0 * u

    top = np.sqrt(TOP_GEOMETRIC)
    value, _ = quad(integrand, 0.0, top, limit=800)
    return value


APPARENT_ALTITUDES_DEG = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    85.0,
    90.0,
]


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    column = zenith_column(EARTH_RADIUS)

    # What the truncation at 86 km costs, measured against an exponential tail with the local scale
    # height rather than assumed away.
    top_density = density(TOP_GEOMETRIC * 0.999)
    scale_height = 6_000.0
    truncated = top_density * scale_height / column

    horizon_by_radius = {
        name: airmass(np.pi / 2.0, radius, zenith_column(radius))
        for name, radius in EARTH_RADIUS_VARIANTS.items()
    }

    rows = [
        "# Numerically integrated relative optical air mass and astronomical refraction.",
        "# U.S. Standard Atmosphere 1976 + Gladstone-Dale + the spherical ray invariant.",
        "# Generated by tools/gen_airmass_oracle.py -- do not edit.",
        "#",
        f"# Earth radius {EARTH_RADIUS / 1e3:.3f} km; sea-level refractivity {REFRACTIVITY_SEA_LEVEL:.6e}",
        f"# zenith column {column:.9e} kg m^-2; truncated above 86 km: {truncated:.3e} of it",
        "# Horizon air mass by Earth radius, the one choice this cannot make for itself:",
    ]
    for name, value in horizon_by_radius.items():
        rows.append(f"#   {name:30s} X(90 deg apparent) = {value:.6f}")
    rows.append("#")
    rows.append(
        "# apparent_altitude_deg  airmass  refraction_arcmin  true_altitude_deg"
    )

    for altitude in APPARENT_ALTITUDES_DEG:
        z_apparent = np.radians(90.0 - altitude)
        x = airmass(z_apparent, EARTH_RADIUS, column)
        r = refraction(z_apparent, EARTH_RADIUS)
        true_altitude = altitude - np.degrees(r)
        rows.append(
            f"{altitude:22.15e} {x:22.15e} "
            f"{np.degrees(r) * 60.0:22.15e} {true_altitude:22.15e}"
        )

    FIXTURE.write_text("\n".join(rows) + "\n")
    print(f"airmass_integrated.txt: {len(APPARENT_ALTITUDES_DEG)} rows")
    print(
        f"  X(90 deg) = {horizon_by_radius['mean 6371.0']:.4f}, "
        f"R(90 deg) = {np.degrees(refraction(np.pi / 2.0, EARTH_RADIUS)) * 60.0:.3f} arcmin"
    )


if __name__ == "__main__":
    main()
