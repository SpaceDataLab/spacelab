//! Jacchia static diffusion model (SR313 Sec. 2-4; census Sec. 1 for the edition of each part).
//!
//! 90 km to the homopause (105 km): barometric equation (5) with the mean molecular mass of
//! eq. (1), composition from eqs. (2)-(4). Above 105 km: diffusion equation (6) per species,
//! thermal diffusion factor -0.38 for He. Hydrogen from 500 km up: eq. (7) at 500 km with the
//! profile temperature T(500) as argument (SR332 p. 9), then eq. (6). No hydrogen below 500 km
//! (SR313 p. 7). Gravity from eq. (8).
//!
//! Gate: `derivations/jacchia_bowman_static_reference.mac`, the `(G2)` tests in
//! `tests/jacchia_bowman_g1_g2_static.rs` and SR313 Table 5 in `tests/jacchia_bowman_g5_sr313_table5.rs`.

use super::constants::{
    ALPHA_HE, AVOGADRO, G_PER_KG, G0, H500_COEFFS, LOG10_CM3_PER_M3, M_AR, M_BAR_0, M_BAR_COEFFS,
    M_BAR_Z_REF, M_H, M_HE, M_N2, M_O, M_O2, M_PER_KM, Q0_AR, Q0_HE, Q0_N2, Q0_O2, R_E, R_STAR,
    RHO0, T0, Z_HOMOPAUSE, Z_HYDROGEN, Z0, ZX,
};
use super::profile::Profile;
use super::quadrature::boole;

/// Default largest quadrature panel (four Boole subintervals), km. The error falls as the
/// sixth power of the step; halving it changes every log10 n_i and log10 rho by less than
/// 5e-12 at 90-2500 km and T_inf 500-2500 K (test `quadrature_converges`).
pub const DEFAULT_MAX_STEP_KM: f64 = 1.0;

/// Number densities, m^-3.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NumberDensities {
    /// Molecular nitrogen.
    pub n2: f64,
    /// Molecular oxygen.
    pub o2: f64,
    /// Atomic oxygen.
    pub o: f64,
    /// Argon.
    pub ar: f64,
    /// Helium.
    pub he: f64,
    /// Atomic hydrogen; 0 below 500 km (SR313 p. 7).
    pub h: f64,
}

impl NumberDensities {
    /// Mass density, kg m^-3.
    pub fn mass_density(&self) -> f64 {
        (self.n2 * M_N2
            + self.o2 * M_O2
            + self.o * M_O
            + self.ar * M_AR
            + self.he * M_HE
            + self.h * M_H)
            / AVOGADRO
    }

    /// Total particle number density, m^-3.
    pub fn total(&self) -> f64 {
        self.n2 + self.o2 + self.o + self.ar + self.he + self.h
    }

    /// Mean molar mass, kg/mol.
    pub fn mean_molar_mass(&self) -> f64 {
        self.mass_density() * AVOGADRO / self.total()
    }
}

/// Result of the static model at one height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticState {
    /// Local temperature T(z), K.
    pub temperature: f64,
    /// Number densities, m^-3.
    pub n: NumberDensities,
    /// Mass density, kg m^-3.
    pub density: f64,
}

/// Gravity g(z), m s^-2, z in km (SR313 p. 8, eq. (8)).
pub fn gravity(z: f64) -> f64 {
    let r = 1.0 + z / R_E;
    G0 / (r * r)
}

/// Mean molecular mass M(z), kg/mol, 90 <= z <= 105 km (SR313 p. 4-5, eq. (1)).
pub fn mean_molar_mass_mixing(z: f64) -> f64 {
    let x = z - M_BAR_Z_REF;
    M_BAR_COEFFS.iter().rev().fold(0.0, |acc, c| acc * x + c) / G_PER_KG
}

/// log10 of n(H) at 500 km in m^-3, from the profile temperature at 500 km
/// (SR332 p. 9, eq. (7); SR313 p. 7 numbers).
pub fn log10_hydrogen_500(t500: f64) -> f64 {
    let l = t500.log10();
    H500_COEFFS[0] + H500_COEFFS[1] * l + H500_COEFFS[2] * l * l + LOG10_CM3_PER_M3
}

/// Integral of g/T dz from `a` to `b` (km), in m s^-2 K^-1 m, split at z_x where the profile
/// changes form.
fn g_over_t_integral(p: &Profile, a: f64, b: f64, max_step: f64) -> f64 {
    let f = |z: f64| gravity(z) / p.temperature(z);
    let integral = if a < ZX && b > ZX {
        boole(f, a, ZX, max_step) + boole(f, ZX, b, max_step)
    } else {
        boole(f, a, b, max_step)
    };
    integral * M_PER_KM
}

/// Composition at z in the mixing region, 90 <= z <= 105 km.
fn mixing_region(p: &Profile, z: f64, max_step: f64) -> NumberDensities {
    // eq. (5): d ln rho = d ln(M/T) - M g/(R* T) dz, integrated from 90 km.
    let m90 = mean_molar_mass_mixing(Z0);
    let mz = mean_molar_mass_mixing(z);
    let integral = boole(
        |s| mean_molar_mass_mixing(s) * gravity(s) / p.temperature(s),
        Z0,
        z,
        max_step,
    ) * M_PER_KM;
    let rho = RHO0 * (mz / m90) * (T0 / p.temperature(z)) * (-integral / R_STAR).exp();
    // eqs. (2)-(4)
    let n_total = AVOGADRO * rho / mz;
    let ratio = mz / M_BAR_0;
    NumberDensities {
        n2: Q0_N2 * ratio * n_total,
        o2: n_total * (ratio * (1.0 + Q0_O2) - 1.0),
        o: 2.0 * n_total * (1.0 - ratio),
        ar: Q0_AR * ratio * n_total,
        he: Q0_HE * ratio * n_total,
        h: 0.0,
    }
}

/// Static model at height `z` (km, >= 90) for the profile `p`, with quadrature panels of at most
/// `max_step` km.
pub fn static_state_with_step(p: &Profile, z: f64, max_step: f64) -> StaticState {
    let temperature = p.temperature(z);
    let n = if z <= Z_HOMOPAUSE {
        mixing_region(p, z, max_step)
    } else {
        let base = mixing_region(p, Z_HOMOPAUSE, max_step);
        // eq. (6): n(z) = n(105) (T(105)/T(z))^(1 + alpha) exp(-M/R* int g/T dz)
        let t_ratio = p.temperature(Z_HOMOPAUSE) / temperature;
        let integral = g_over_t_integral(p, Z_HOMOPAUSE, z, max_step) / R_STAR;
        let diffuse =
            |n0: f64, m: f64, alpha: f64| n0 * t_ratio.powf(1.0 + alpha) * (-m * integral).exp();
        let h = if z >= Z_HYDROGEN {
            let t500 = p.temperature(Z_HYDROGEN);
            let n500 = 10f64.powf(log10_hydrogen_500(t500));
            let integral_h = g_over_t_integral(p, Z_HYDROGEN, z, max_step) / R_STAR;
            n500 * (t500 / temperature) * (-M_H * integral_h).exp()
        } else {
            0.0
        };
        // alpha = 0 for all constituents but He (SR313 p. 7).
        NumberDensities {
            n2: diffuse(base.n2, M_N2, 0.0),
            o2: diffuse(base.o2, M_O2, 0.0),
            o: diffuse(base.o, M_O, 0.0),
            ar: diffuse(base.ar, M_AR, 0.0),
            he: diffuse(base.he, M_HE, ALPHA_HE),
            h,
        }
    };
    StaticState {
        temperature,
        n,
        density: n.mass_density(),
    }
}

/// Static model with the default quadrature step.
pub fn static_state(p: &Profile, z: f64) -> StaticState {
    static_state_with_step(p, z, DEFAULT_MAX_STEP_KM)
}
