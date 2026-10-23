//! The DSST **Gaussian** (drag / solar-radiation-pressure) short-period corrections (Danielson
//! §2.5.3), turning DSST mean equinoctial elements into osculating ones — the numerically-averaged
//! analogue of the [`super::short_period`] (zonal) and [`super::third_body_short_period`] ports.
//!
//! Direct port of Orekit's `AbstractGaussianContribution` short-period machinery, shared by
//! `DSSTSolarRadiationPressure` and `DSSTAtmosphericDrag` (both extend `AbstractGaussianContribution`);
//! only the per-node acceleration and the integration limits differ between the two, and both are
//! already ported ([`super::srp`], [`super::drag`]). Mirrors Orekit's inner-class split:
//! * `FourierCjSjCoefficients` — the Fourier Cᵢʲ/Sᵢʲ of the six equinoctial variational integrands,
//!   obtained by **Gauss–Legendre quadrature** of the same per-node `(δaᵢ/δv)·q` used for the mean
//!   rates, weighted by `cos(jL)`/`sin(jL)` instead of `roa²` (Orekit's `IntegrableFunction.value`
//!   in short-periodic mode), divided by π, for `j = 0 … JMAX`;
//! * `computeRhoSigmaCoefficients` — ρⱼ = (1+jB)(−b)ʲCⱼ(k,h), σⱼ = (1+jB)(−b)ʲSⱼ(k,h) (Danielson
//!   2.5.3-(4)), `j = 1 … 3·JMAX`;
//! * `UijVijCoefficients` — the U₁ʲ/V₁ʲ (Danielson 2.5.3-(21,22)) built from the Fourier coefficients
//!   and ρ/σ, and the U₂ʲ/V₂ʲ (only the Fourier index i = 0 is needed);
//! * `GaussianShortPeriodicCoefficients.computeCoefficients` — the stored Cᵢ⁰/Cᵢʲ/Sᵢʲ and the two
//!   polynomial coefficients Dᵢ¹/Dᵢ² (only the mean-longitude i = 5 carries them), plus the k₂⁰
//!   free term (Danielson 2.5.3-(9a)); and
//! * `GaussianShortPeriodicCoefficients.value` — the Fourier sum over the **true** longitude L = Lᵥ
//!   plus the `(L − λ)`, `(L − λ)²` polynomial part.
//!
//! Variable names follow Orekit for auditability (`JMAX`, `rhoSigma`, `u1ij`/`v1ij`/`u2ij`/`v2ij`,
//! `k20`, `to2an`/`to4an`, `getLLimits`).
//!
//! # What differs from the zonal / third-body short-period
//!
//! * The Fourier Cᵢʲ/Sᵢʲ are **numerical Gauss-quadrature integrals** of the per-node acceleration
//!   (there is no closed-form disturbing function), reusing [`super::gaussian`]'s node kinematics and
//!   the same integration limits `[Lₗₒw, Lₕᵢgₕ]` the mean rates use — the SRP cylindrical shadow arc
//!   or the drag atmospheric arc. The Gauss order is the one Orekit's `isDirty` pass fixes for this
//!   state (SRP → 48, drag → 12); the same [`super::gaussian::select_order`] the mean rates expose.
//! * The series is in the **true** longitude L = Lᵥ (`meanOrbit.getLv()`), like the mean-rate
//!   averaging variable — not the eccentric longitude the third-body series uses — and carries an
//!   extra polynomial `(L − λ)·Dᵢ¹ + (L − λ)²·Dᵢ²` (Danielson 2.5.3-(26)).
//!
//! # What is *not* ported
//!
//! As in the sibling ports, Orekit's `ShortPeriodicsInterpolatedCoefficient` (Hermite interpolation of
//! the coefficients between integration-step grid points) is a propagator detail: this module computes
//! the coefficients directly at the output mean state, which the interpolation approximates and is
//! exact at every grid point. The oracle therefore seeds a `DSSTPropagator` OSCULATING run from the
//! identical mean state and compares `osc − mean` against the port's short-period δ.

use std::vec::Vec;

use libm::{cos, sin};

use super::auxiliary::DsstAuxiliary;
use super::gaussian::{self, NodeState};
use super::srp::mean_to_true;

/// Maximum value of the Fourier index `j` (`AbstractGaussianContribution.JMAX`).
const JMAX: usize = 12;

/// The six equinoctial short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` (Orekit ordering) of a
/// Gaussian contribution at the given **mean** state, for the true-longitude integration window `ll`
/// (`getLLimits`), the fixed Gauss order `order` and the per-node perturbing acceleration `accel`.
///
/// Returns zeros when `ll[0] ≥ ll[1]` — Orekit skips the Fourier integral there, and every stored
/// coefficient then vanishes (the whole orbit is eclipsed, for SRP, or above the atmosphere, for drag).
#[must_use]
pub(crate) fn short_period_correction<F>(
    aux: &DsstAuxiliary,
    ll: [f64; 2],
    order: usize,
    accel: F,
) -> [f64; 6]
where
    F: Fn(&NodeState) -> [f64; 3],
{
    // Orekit skips the Fourier integral (and every stored coefficient then vanishes) when the sunlit /
    // atmospheric arc is empty — the whole orbit is eclipsed (SRP) or above the atmosphere (drag).
    if ll[0] < ll[1] {
        // ρⱼ, σⱼ for j = 1 … 3·JMAX (Danielson 2.5.3-(4)).
        let (rho, sigma) = compute_rho_sigma(aux);

        // Fourier Cᵢʲ / Sᵢʲ for j = 0 … JMAX via Gauss quadrature of the per-node integrand.
        let (ccoef, scoef) = fourier_coefficients(aux, ll, order, &accel);

        // U₁ʲ / V₁ʲ / U₂ʲ / V₂ʲ (Danielson 2.5.3-(21,22)).
        let uv = UijVij::new(&ccoef, &scoef, &rho, &sigma);

        // Stored Cᵢ⁰ / Cᵢʲ / Sᵢʲ and the polynomial Dᵢ¹ / Dᵢ² (Danielson 2.5.3-(26)).
        let sp = StoredCoefficients::new(aux, &ccoef, &uv, &rho, &sigma);

        // Fourier sum over the true longitude L = Lᵥ plus the (L − λ) polynomial part.
        sp.value(aux)
    } else {
        [0.0; 6]
    }
}

/// The ρⱼ = (1+jB)(−b)ʲCⱼ(k,h) and σⱼ = (1+jB)(−b)ʲSⱼ(k,h) coefficients for j = 0 … 3·JMAX, where
/// Cⱼ(k,h) + i·Sⱼ(k,h) = (k + i·h)ʲ and b = 1/(1+B) (Orekit `computeRhoSigmaCoefficients`, Danielson
/// 2.5.3-(4)). Index 0 is unused (ρ₀ = σ₀ = 0), matching Orekit's 1-based fill.
fn compute_rho_sigma(aux: &DsstAuxiliary) -> (Vec<f64>, Vec<f64>) {
    let n = 3 * JMAX;
    // Cⱼ(k,h), Sⱼ(k,h) = real / imaginary parts of (k + i·h)ʲ.
    let mut cj = std::vec![0.0; n + 1];
    let mut sj = std::vec![0.0; n + 1];
    cj[0] = 1.0;
    for j in 1..=n {
        cj[j] = cj[j - 1] * aux.k - sj[j - 1] * aux.h;
        sj[j] = cj[j - 1] * aux.h + sj[j - 1] * aux.k;
    }

    let b = 1.0 / (1.0 + aux.big_b);
    let mut rho = std::vec![0.0; n + 1];
    let mut sigma = std::vec![0.0; n + 1];
    let mut mbtj = 1.0; // (−b)ʲ
    for j in 1..=n {
        mbtj *= -b;
        let coef = (1.0 + j as f64 * aux.big_b) * mbtj;
        rho[j] = coef * cj[j];
        sigma[j] = coef * sj[j];
    }
    (rho, sigma)
}

/// The Fourier Cᵢʲ / Sᵢʲ coefficients (Orekit `FourierCjSjCoefficients`) as `ccoef[j][i]` / `scoef[j][i]`
/// with `i` the equinoctial element (0:a, 1:k, 2:h, 3:q, 4:p, 5:λ) and `j = 0 … JMAX`.
///
/// For each `j` the twelve-component integrand `[cos(jL)·derivᵢ, sin(jL)·derivᵢ]` is Gauss-quadrature
/// integrated over `[ll[0], ll[1]]` at the fixed `order` and divided by π. The per-node six `derivᵢ`
/// are the shared [`gaussian::element_derivatives`] (`(δaᵢ/δv)·q`, without the mean-mode `roa²`),
/// computed once per node and reused across all `j` — bit-identical to Orekit re-evaluating the
/// integrand per `j`, since `derivᵢ` is deterministic in the node. The compensated (Kahan) accumulation
/// and the `w·(cos(jL)·derivᵢ)` multiplication order mirror `GaussQuadrature.integrate` exactly.
fn fourier_coefficients<F>(
    aux: &DsstAuxiliary,
    ll: [f64; 2],
    order: usize,
    accel: &F,
) -> (Vec<[f64; 6]>, Vec<[f64; 6]>)
where
    F: Fn(&NodeState) -> [f64; 3],
{
    let (points, weights) = gaussian::nodes_weights(order);
    let scale = (ll[1] - ll[0]) / 2.0;
    let shift = ll[0] + scale;
    let m = points.len();

    // Per-node true longitude, quadrature weight and the six raw δaᵢ/δv·q integrands.
    let mut xs = std::vec![0.0; m];
    let mut ws = std::vec![0.0; m];
    let mut derivs = std::vec![[0.0f64; 6]; m];
    for i in 0..m {
        xs[i] = points[i] * scale + shift;
        ws[i] = weights[i] * scale;
        derivs[i] = gaussian::element_derivatives(aux, xs[i], accel).0;
    }

    let oo_pi = 1.0 / core::f64::consts::PI;
    let mut ccoef = std::vec![[0.0f64; 6]; JMAX + 1];
    let mut scoef = std::vec![[0.0f64; 6]; JMAX + 1];
    for j in 0..=JMAX {
        let jf = j as f64;
        // Seed the compensated sums with the first node (GaussQuadrature.basicIntegrate).
        let cos0 = cos(jf * xs[0]);
        let sin0 = sin(jf * xs[0]);
        let mut cs = [0.0f64; 6]; // Σ w·cos(jL)·derivᵢ
        let mut ss = [0.0f64; 6]; // Σ w·sin(jL)·derivᵢ
        for e in 0..6 {
            cs[e] = ws[0] * (cos0 * derivs[0][e]);
            ss[e] = ws[0] * (sin0 * derivs[0][e]);
        }
        let mut cc = [0.0f64; 6]; // Kahan compensations
        let mut sc = [0.0f64; 6];
        for i in 1..m {
            let cos_j = cos(jf * xs[i]);
            let sin_j = sin(jf * xs[i]);
            for e in 0..6 {
                let yc = ws[i] * (cos_j * derivs[i][e]) - cc[e];
                let tc = cs[e] + yc;
                cc[e] = (tc - cs[e]) - yc;
                cs[e] = tc;

                let ys = ws[i] * (sin_j * derivs[i][e]) - sc[e];
                let ts = ss[e] + ys;
                sc[e] = (ts - ss[e]) - ys;
                ss[e] = ts;
            }
        }
        for e in 0..6 {
            ccoef[j][e] = oo_pi * cs[e];
            scoef[j][e] = oo_pi * ss[e];
        }
    }
    (ccoef, scoef)
}

/// The U₁ʲ / V₁ʲ / U₂ʲ / V₂ʲ coefficients (Orekit `UijVijCoefficients`, Danielson 2.5.3-(21,22)).
/// `u1[i][j]` / `v1[i][j]` are indexed by the equinoctial element `i` (0…5) and `j` (0…2·JMAX, though
/// only i = 0 is filled for j > JMAX); `u2[j]` / `v2[j]` (j = 0…JMAX) use only the Fourier index i = 0.
struct UijVij {
    u1: Vec<[f64; 6]>,
    v1: Vec<[f64; 6]>,
    u2: Vec<f64>,
    v2: Vec<f64>,
}

impl UijVij {
    fn new(ccoef: &[[f64; 6]], scoef: &[[f64; 6]], rho: &[f64], sigma: &[f64]) -> Self {
        // ccoef/scoef are [0..=JMAX][0..6]; getCij(i,j) = ccoef[j][i], getSij(i,j) = scoef[j][i].
        let cij = |i: usize, j: usize| ccoef[j][i];
        let sij = |i: usize, j: usize| scoef[j][i];

        let mut u1 = std::vec![[0.0f64; 6]; 2 * JMAX + 1];
        let mut v1 = std::vec![[0.0f64; 6]; 2 * JMAX + 1];
        let mut u2 = std::vec![0.0f64; JMAX + 1];
        let mut v2 = std::vec![0.0f64; JMAX + 1];

        // computeU1V1Coefficients: for j = 1 … JMAX, all six Fourier indices i.
        u1[0][0] = 0.0;
        for j in 1..=JMAX {
            let ooj = 1.0 / j as f64;
            for i in 0..6 {
                let mut u = sij(i, j);
                let mut v = cij(i, j);

                if j > 1 {
                    for k in 1..j {
                        u += cij(i, j - k) * sigma[k] + sij(i, j - k) * rho[k];
                        v += cij(i, j - k) * rho[k] - sij(i, j - k) * sigma[k];
                    }
                }
                if j != JMAX {
                    for k in 1..=JMAX - j {
                        u += -cij(i, j + k) * sigma[k] + sij(i, j + k) * rho[k];
                        v += cij(i, j + k) * rho[k] + sij(i, j + k) * sigma[k];
                    }
                }
                for k in 1..=JMAX {
                    u += -cij(i, k) * sigma[j + k] - sij(i, k) * rho[j + k];
                    v += cij(i, k) * rho[j + k] + sij(i, k) * sigma[j + k];
                }

                u1[j][i] = -ooj * u;
                v1[j][i] = ooj * v;

                if i == 0 {
                    u1[0][0] += -u1[j][0] * rho[j] - v1[j][0] * sigma[j];
                }
            }
        }

        // Terms j = JMAX+1 … 2·JMAX (only Fourier index i = 0), needed by U₂ʲ / V₂ʲ.
        for j in JMAX + 1..=2 * JMAX {
            let ooj = 1.0 / j as f64;
            let mut u = 0.0f64;
            let mut v = 0.0f64;
            for k in (j - JMAX)..j {
                u += cij(0, j - k) * sigma[k] + sij(0, j - k) * rho[k];
                v += cij(0, j - k) * rho[k] - sij(0, j - k) * sigma[k];
            }
            for k in 1..=JMAX {
                u += -cij(0, k) * sigma[j + k] - sij(0, k) * rho[j + k];
                v += cij(0, k) * rho[j + k] + sij(0, k) * sigma[j + k];
            }
            u1[j][0] = -ooj * u;
            v1[j][0] = ooj * v;
        }

        // computeU2V2Coefficients (only Fourier index i = 0). Note: Orekit's j+l loop accumulates
        // BOTH terms into u2ij[j] (the second statement is not v2ij[j]); ported verbatim so the
        // osculating output matches Orekit.
        for j in 1..=JMAX {
            let ooj = 1.0 / j as f64;
            let mut u = v1[j][0];
            let mut v = u1[j][0];
            if j > 1 {
                for l in 1..j {
                    u += u1[j - l][0] * sigma[l] + v1[j - l][0] * rho[l];
                    v += u1[j - l][0] * rho[l] - v1[j - l][0] * sigma[l];
                }
            }
            for l in 1..=JMAX {
                u += -u1[j + l][0] * sigma[l] + u1[l][0] * sigma[j + l] + v1[j + l][0] * rho[l]
                    - v1[l][0] * rho[j + l];
                u += u1[j + l][0] * rho[l]
                    + u1[l][0] * rho[j + l]
                    + v1[j + l][0] * sigma[l]
                    + v1[l][0] * sigma[j + l];
            }
            u2[j] = -ooj * u;
            v2[j] = ooj * v;
        }

        Self { u1, v1, u2, v2 }
    }
}

/// The stored short-period coefficients (Orekit `GaussianShortPeriodicCoefficients.computeCoefficients`):
/// the free term Cᵢ⁰, the per-frequency Cᵢʲ / Sᵢʲ (j = 1…JMAX) and the polynomial Dᵢ¹ / Dᵢ² (only the
/// mean-longitude i = 5 is non-zero). Danielson 2.5.3-(26).
struct StoredCoefficients {
    cij0: [f64; 6],
    cij: Vec<[f64; 6]>,
    sij: Vec<[f64; 6]>,
    di1: [f64; 6],
    di2: [f64; 6],
}

impl StoredCoefficients {
    fn new(
        aux: &DsstAuxiliary,
        ccoef: &[[f64; 6]],
        uv: &UijVij,
        rho: &[f64],
        sigma: &[f64],
    ) -> Self {
        // k₂⁰ = Σ_{k=1}^{JMAX} (2/k²)(σ_k² + ρ_k²) (Danielson 2.5.3-(9a)).
        let mut k20 = 0.0;
        for k in 1..=JMAX {
            let term = sigma[k] * sigma[k] + rho[k] * rho[k];
            k20 += term * 2.0 / (k * k) as f64;
        }

        let n = aux.n;
        let a = aux.sma;
        let oon = 1.0 / n;
        let to2an = 1.5 * oon / a; // 3 / (2·a·n)
        let to4an = to2an / 2.0; // 3 / (4·a·n)

        let mut cij0 = [0.0f64; 6];
        let mut cij = std::vec![[0.0f64; 6]; JMAX + 1];
        let mut sij = std::vec![[0.0f64; 6]; JMAX + 1];
        let mut di1 = [0.0f64; 6];
        let mut di2 = [0.0f64; 6];

        for i in 0..6 {
            // Dᵢ¹, Dᵢ² (all j-orders above 2 are 0; only the mean longitude i = 5 is non-zero).
            di1[i] = -oon * ccoef[0][i]; // getCij(i, 0)
            if i == 5 {
                di1[i] += to2an * uv.u1[0][0]; // getU1(0, 0)
            }
            di2[i] = 0.0;
            if i == 5 {
                di2[i] += -to4an * ccoef[0][0]; // getCij(0, 0)
            }

            // Cᵢ⁰ accumulates from all Cᵢʲ / Sᵢʲ (Danielson 2.5.2-(15a)).
            cij0[i] = -di2[i] * k20;

            for j in 1..=JMAX {
                let mut c = oon * uv.u1[j][i]; // getU1(j, i)
                if i == 5 {
                    c += -to2an * uv.u2[j]; // getU2(j)
                }
                let mut s = oon * uv.v1[j][i]; // getV1(j, i)
                if i == 5 {
                    s += -to2an * uv.v2[j]; // getV2(j)
                }
                cij[j][i] = c;
                sij[j][i] = s;

                cij0[i] -= c * rho[j] + s * sigma[j];
            }
        }

        Self {
            cij0,
            cij,
            sij,
            di1,
            di2,
        }
    }

    /// The six short-period corrections evaluated at the mean state (Orekit
    /// `GaussianShortPeriodicCoefficients.value`): the Fourier sum over the **true** longitude L = Lᵥ
    /// plus the `(L − λ)·Dᵢ¹ + (L − λ)²·Dᵢ²` polynomial part.
    fn value(&self, aux: &DsstAuxiliary) -> [f64; 6] {
        // L = Lᵥ (meanOrbit.getLv()); center = L − λ.
        let l = mean_to_true(aux.k, aux.h, aux.lambda);
        let center = l - aux.lambda;
        let center2 = center * center;

        let mut spv = [0.0f64; 6];
        for i in 0..6 {
            spv[i] = self.cij0[i] + center * self.di1[i] + center2 * self.di2[i];
        }
        for j in 1..=JMAX {
            let jl = j as f64 * l;
            let (c, s) = (cos(jl), sin(jl));
            for i in 0..6 {
                spv[i] += self.cij[j][i] * c + self.sij[j][i] * s;
            }
        }
        spv
    }
}

#[cfg(test)]
mod tests {
    use spacelab_testkit::nan_max;

    use super::super::drag::{Atmosphere, DsstAtmosphericDrag};
    use super::super::srp::DsstSolarRadiationPressure;
    use super::{DsstAuxiliary, JMAX, compute_rho_sigma, cos, gaussian, sin};

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`, orekit_dsst_srp_sp.csv /
    // orekit_dsst_drag_sp.csv): paired MEAN and OSCULATING DSSTPropagator runs seeded from the same MEAN
    // equinoctial LEO, ONE Gaussian force each, sampled at t > 0 (t = 0 excluded — a DSST OSCULATING
    // propagator returns the raw mean state at the initial epoch). Feeding the port Orekit's own MEAN
    // state at each sample isolates the short-period assembly: osc − mean must equal the port's δ.
    const MU: f64 = 3.98600441500000000e+14;

    // ---- SRP (cr = 1.5, area = 10 m², mass = 1000 kg ⇒ S/m = 0.01, spherical Earth of radius AE) ----
    const AE: f64 = 6.37813646000000000e+06;
    const CR: f64 = 1.50000000000000000e+00;
    const SRP_AOM: f64 = 1.00000000000000000e+01 / 1.00000000000000000e+03;

    #[rustfmt::skip]
    const SRP_MEAN: [[f64; 6]; 4] = [
        [7.19999999995021300e+06, 3.09030844107721940e-03, 9.51052081350051800e-03, 8.54890341105058600e-01, 7.69746746923938300e-01, 2.37526966973964560e+01],
        [7.19999999987309400e+06, 3.09044655160035320e-03, 9.51047695433844200e-03, 8.54890352810721800e-01, 7.69746783698135100e-01, 4.60742234075780400e+01],
        [7.19999999976902600e+06, 3.09058427639166650e-03, 9.51043358724990200e-03, 8.54890364330943900e-01, 7.69746820164667400e-01, 6.83957501172855000e+01],
        [7.19999999963844100e+06, 3.09072161675843240e-03, 9.51039071400774400e-03, 8.54890375660562700e-01, 7.69746856304788900e-01, 9.07172768266210600e+01],
    ];
    #[rustfmt::skip]
    const SRP_OSC: [[f64; 6]; 4] = [
        [7.19999993772446400e+06, 3.09030835252220360e-03, 9.51052168450583700e-03, 8.54890349410882400e-01, 7.69746747929201000e-01, 2.37526967036784000e+01],
        [7.20000005075473400e+06, 3.09044274553558950e-03, 9.51047814252891900e-03, 8.54890345947319000e-01, 7.69746782781918400e-01, 4.60742233977678740e+01],
        [7.19999997195553000e+06, 3.09058449963069140e-03, 9.51043275124962200e-03, 8.54890369547576200e-01, 7.69746825289596100e-01, 6.83957501381545800e+01],
        [7.20000000696213700e+06, 3.09071930125067600e-03, 9.51039023863294000e-03, 8.54890372841655800e-01, 7.69746850401728300e-01, 9.07172768026218400e+01],
    ];
    #[rustfmt::skip]
    const SRP_SUN: [[f64; 3]; 4] = [
        [2.54573948706984800e+10, -1.32930226786026440e+11, -5.76233810999046550e+10],
        [2.61012496643221800e+10, -1.32824745443904500e+11, -5.75776788441440700e+10],
        [2.67446099015337100e+10, -1.32716684101772370e+11, -5.75308590366453860e+10],
        [2.73874632760801120e+10, -1.32606044237044450e+11, -5.74829222629517900e+10],
    ];

    // ---- Drag (cd = 2.2, area = 10 m², mass = 1000 kg, constant co-rotating atmosphere) ----
    const CD: f64 = 2.20000000000000000e+00;
    const DRAG_AOM: f64 = 1.00000000000000000e+01 / 1.00000000000000000e+03;
    const RHO: f64 = 1.00000000000000000e-12;
    const OMEGA: f64 = 7.29211514670697900e-05;
    const RBAR: f64 = 7.37813700000000000e+06;

    #[rustfmt::skip]
    const DRAG_MEAN: [[f64; 6]; 4] = [
        [6.99997439400475800e+06, 3.09016460346352250e-03, 9.51054858646200500e-03, 8.54890276800989500e-01, 7.69746662650970400e-01, 2.47161982984143200e+01],
        [6.99994878805948800e+06, 3.09015926317054340e-03, 9.51053200996174700e-03, 8.54890224383209200e-01, 7.69746615442262000e-01, 4.80013543758131400e+01],
        [6.99992318216418900e+06, 3.09015392288265870e-03, 9.51051543346615800e-03, 8.54890171965635100e-01, 7.69746568233717600e-01, 7.12866382197508400e+01],
        [6.99989757631886000e+06, 3.09014858258770860e-03, 9.51049885696273000e-03, 8.54890119548262000e-01, 7.69746521025365900e-01, 9.45720498311465600e+01],
    ];
    #[rustfmt::skip]
    const DRAG_OSC: [[f64; 6]; 4] = [
        [6.99997438315856200e+06, 3.09023197057150570e-03, 9.51069744585146700e-03, 8.54890278380545900e-01, 7.69746662460085400e-01, 2.47161983059336660e+01],
        [6.99994879201833500e+06, 3.09028455863667550e-03, 9.51042857814017400e-03, 8.54890222913846000e-01, 7.69746616363346100e-01, 4.80013543925342460e+01],
        [6.99992319112775850e+06, 3.09001908335829700e-03, 9.51042174969843200e-03, 8.54890172869865200e-01, 7.69746566882164100e-01, 7.12866382213419000e+01],
        [6.99989756759482500e+06, 3.09009732840400450e-03, 9.51065475648114400e-03, 8.54890119671008300e-01, 7.69746522577778600e-01, 9.45720498354101100e+01],
    ];

    /// A constant-density, purely co-rotating atmosphere: `ρ = RHO` everywhere and `v_atm = ω·ẑ × r`.
    /// The cross product is Hipparchus `Vector3D.crossProduct((0,0,ω), r)` (extended-precision
    /// `linearCombination`), bit-identical to the Java oracle's overridden `getVelocity`.
    struct CoRotatingConstant {
        rho: f64,
        omega: f64,
    }

    impl Atmosphere for CoRotatingConstant {
        fn density(&self, _position: [f64; 3]) -> f64 {
            self.rho
        }
        fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
            gaussian::cross([0.0, 0.0, self.omega], position)
        }
    }

    /// Compare an osculating state to Orekit's: `a` to `a_tol` (m, abs) and the equinoctial e/i
    /// components + mean longitude to `comp_tol` (abs). Returns `(|Δa|, max |Δcomp|)`.
    fn assert_osc(
        got: [f64; 6],
        want: [f64; 6],
        a_tol: f64,
        comp_tol: f64,
        label: &str,
    ) -> (f64, f64) {
        let da = (got[0] - want[0]).abs();
        assert!(da < a_tol, "{label}: a abs {da:.3e} m (tol {a_tol:.0e})");
        let mut max_comp = 0.0_f64;
        for i in 1..6 {
            let d = (got[i] - want[i]).abs();
            max_comp = max_comp.max(d);
            assert!(
                d < comp_tol,
                "{label} comp[{i}]: {d:.3e} (tol {comp_tol:.0e})"
            );
        }
        (da, max_comp)
    }

    /// The SRP osculating state `mean + short-period δ` matches Orekit's OSCULATING DSSTPropagator at
    /// every sample: a ~62 mm short-period `a` swing recovered to `2.0e-6 m` and the equinoctial e/i
    /// components + mean longitude to `2.4e-12`. Orekit's own mean state is fed in, so this gates only the
    /// Gaussian short-period assembly.
    ///
    /// `Δa` used to sit at ~9e-9 m, the terminator/penumbra `acos` ULP sensitivity documented for the
    /// SRP mean-rate `da`. It is now `2.0e-6 m`, and the worst component `2.4e-12` (the mean
    /// longitude), because the quadrature runs over a different arc: `DsstSolarRadiationPressure::
    /// l_limits` casts a shadow cylinder of radius `ae`, where Orekit's is `ae·B` — see
    /// `srp::tests::integration_limits_match_orekit` for the measurement. The shift stays four
    /// orders below the `~62 mm` swing being reproduced, which is what says the short-period
    /// assembly itself is unchanged; only the arc it integrates over is.
    #[test]
    fn srp_short_period_matches_orekit() {
        let srp = DsstSolarRadiationPressure::new(CR, SRP_AOM, MU, AE);
        let (mut max_da, mut max_comp) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            let osc = srp.osculating(SRP_MEAN[k], SRP_SUN[k]);
            let (da, comp) = assert_osc(osc, SRP_OSC[k], 5e-6, 5e-12, "srp");
            max_da = max_da.max(da);
            max_comp = max_comp.max(comp);
        }
        std::println!(
            "SRP short-period vs Orekit: max |Δa| {max_da:.2e} m, max comp {max_comp:.2e}"
        );
    }

    /// The drag osculating state `mean + short-period δ` matches Orekit's OSCULATING DSSTPropagator at
    /// every sample (full-circle quadrature, no shadow): `a` to < 5e-6 m and the equinoctial e/i
    /// components + mean longitude to < 5e-11. Drag's `da` is well-conditioned (no terminator
    /// cancellation); the slightly looser `a` bound vs SRP absorbs the strongly-decaying orbit's mean-
    /// propagation residual (achieved ~6e-7 m).
    #[test]
    fn drag_short_period_matches_orekit() {
        let drag = DsstAtmosphericDrag::with_rbar(CD, DRAG_AOM, MU, RBAR);
        let atmosphere = CoRotatingConstant {
            rho: RHO,
            omega: OMEGA,
        };
        let (mut max_da, mut max_comp) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            let osc = drag.osculating(DRAG_MEAN[k], &atmosphere);
            let (da, comp) = assert_osc(osc, DRAG_OSC[k], 5e-6, 5e-11, "drag");
            max_da = max_da.max(da);
            max_comp = max_comp.max(comp);
        }
        std::println!(
            "drag short-period vs Orekit: max |Δa| {max_da:.2e} m, max comp {max_comp:.2e}"
        );
    }

    /// `ρⱼ + iσⱼ = (1 + jB)(−b)ʲ(k + ih)ʲ`, `b = 1/(1 + B)` — Danielson 2.5.3-(4).
    ///
    /// The port raises `k + ih` to the `j`-th power by the two-term real recursion Orekit uses, and
    /// nothing downstream can see one of its terms go wrong: the short-period corrections these feed
    /// are asserted against Orekit at `5e-6 m` in `a`, and a slip in a high-`j` coefficient moves
    /// that by far less. Dropping the recursion's own previous imaginary part, or dividing by `B`
    /// where the coefficient multiplies, both survived on that basis. The oracle here is the polar
    /// form `rʲ(cos jθ, sin jθ)`, which shares no arithmetic with the recursion.
    #[test]
    #[allow(clippy::float_cmp)] // index 0 is an unused slot written as an exact zero
    fn rho_sigma_match_the_closed_form() {
        let mut worst = 0.0_f64;
        for &(k, h) in &[
            (0.0, 0.0),
            (1.0e-4, -1.0e-4),
            (0.05, 0.02),
            (0.3, -0.6),
            (0.7, 0.21),
        ] {
            let aux = DsstAuxiliary::new(7.2e6, k, h, 0.3, -0.15, 1.2, MU);
            let (rho, sigma) = compute_rho_sigma(&aux);
            assert_eq!(rho[0], 0.0, "index 0 is unused");
            assert_eq!(sigma[0], 0.0, "index 0 is unused");
            let (r, theta) = (libm::hypot(k, h), libm::atan2(h, k));
            let b = 1.0 / (1.0 + aux.big_b);
            for j in 1..=3 * JMAX {
                let jf = j as f64;
                let coef = (1.0 + jf * aux.big_b) * libm::pow(-b, jf) * libm::pow(r, jf);
                let scale = libm::fabs(coef).max(f64::MIN_POSITIVE);
                let miss = nan_max(
                    libm::fabs(rho[j] - coef * cos(jf * theta)),
                    libm::fabs(sigma[j] - coef * sin(jf * theta)),
                ) / (f64::EPSILON * scale);
                worst = nan_max(worst, miss);
                assert!(
                    miss < 256.0,
                    "k={k} h={h} j={j}: rho/sigma off by {miss:.1} ulps of their own magnitude"
                );
            }
        }
        std::println!("rho/sigma vs the closed form: worst {worst:.1} ulps");
    }
}
