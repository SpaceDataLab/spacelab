//! The DSST **third-body** (Sun / Moon) mean-element rates.
//!
//! Direct port of Orekit's `DSSTThirdBody.getMeanElementRate` and the inner `UAnddU`
//! potential-derivative helper, restricted to the analytic **mean-element rates** — the short-period
//! corrections are out of scope (a follow-on). Variable names (Vₙₛ, Qₙₛ, GₛHₛ, the third-body Hansen
//! kernel K₀ⁿˢ, the U derivatives dU/da …, `maxAR3Pow` / `maxEccPow`) follow Orekit for auditability.
//!
//! # Independent validation
//!
//! Beyond the bit-exact Orekit gate (`fixed_state_{sun,moon}_rates_match_orekit`), the Sun and Moon
//! mean rates are cross-checked against a first-principles numerical orbit-average that shares no code
//! with this port or Orekit — `experiments/dsst_thirdbody_independent_verify.py` finite-differences
//! `∂(element)/∂v` from a textbook rv↔classical conversion and averages `∂(element)/∂v · a_body` over
//! one revolution (the Keplerian part is secularly stationary). The Sun's dominant `dq/dp/dλ` agree to
//! ~3e-9…6e-9 and the Moon's to ~3e-7 — the looser Moon match being the expected signature of the
//! truncated `a/r³` expansion (Moon `a/r ≈ 0.018` vs Sun `5e-5`), while `da ≈ 0` reproduces the
//! semimajor-axis theorem. This validates Orekit's own values, not merely that the port matches them.
//!
//! `derivations/dsst_third_body_average.mac` checks the expansion itself exactly, in Maxima. The
//! kernel recurrence and its roots equal the definition K₀ⁿˢ = ⟨(r/a)ⁿ cos sf⟩/eˢ for s ≤ 6 and
//! n ≤ s+8, and the derivative recurrence equals d/dχ. For n = 2..6 the Vₙₛ Qₙₛ K₀ⁿˢ Gₛ sum equals the
//! orbit average of (r/a)ⁿ Pₙ(cos ψ), the degree-n term of the point-mass potential. Each partial
//! `potential_derivatives` returns is the derivative of the returned U.
//!
//! # What differs from the zonal contribution
//!
//! The final `[da, dk, dh, dq, dp, dλ]` assembly (the equinoctial Lagrange brackets, Danielson
//! 3.1-(1)) and the shape of the `UAnddU` double sum are **identical** to [`super::zonal`]. The
//! third-body specifics are:
//! * the direction cosines α/β/γ are the *normalized body direction* dotted with the equinoctial
//!   f/g/w vectors (not the pole z-components used by the axisymmetric zonal field);
//! * the radial factor is `(a/R₃)ⁿ` (not `(Rₑ/a)ⁿ`) and the potential scale is `µ₃/R₃`
//!   (not `−µ/a`), with `R₃ = |body_pos|` the central-body→third-body distance;
//! * there is no gravity coefficient `Cₙ₀`; `coef0 = (2−δ₀ₛ)·(a/R₃)ⁿ·Vₙₛ`;
//! * `dU/da` carries the factor `n` (not `n+1`);
//! * the sum runs `n = max(2, s)..=maxAR3Pow`, so `n = s` occurs (with dQₙₛ/dγ set to 0 there,
//!   Cefola & Broucke 1975), unlike the zonal `n = s+2..`;
//! * the truncation fixes **two** orders — `maxAR3Pow` (power of a/R₃) and `maxEccPow` (power of e) —
//!   from the a/R₃ ratio and the eccentricity (`DSSTThirdBodyStaticContext`);
//! * the Hansen kernel is the third-body K₀ⁿˢ(χ) (`HansenThirdBodyLinear`), a positive-`n` recurrence
//!   distinct from the zonal K₀^{−n−1,s}.

use std::vec::Vec;

use super::auxiliary::{DsstAuxiliary, I};
use super::coefficients;

/// Max power for the summation (`DSSTThirdBody.MAX_POWER`). Bounds the Vₙₛ table and the truncation
/// search.
const MAX_POWER: usize = 22;

/// Maximum eccentricity power used by the short-period computation (`MAX_ECCPOWER_SP`). The mean path
/// only reads Qₙₛ up to column `maxEccPow + 1`, but Orekit sizes the shared Qₙₛ to
/// `max(maxEccPow, MAX_ECCPOWER_SP)`; mirrored so the (identical) Qₙₛ values are byte-for-byte the
/// same as Orekit's.
const MAX_ECCPOWER_SP: usize = 4;

/// Truncation tolerance for big, eccentric orbits (`BIG_TRUNCATION_TOLERANCE`).
const BIG_TRUNCATION_TOLERANCE: f64 = 1.0e-1;

/// Truncation tolerance for small orbits (`SMALL_TRUNCATION_TOLERANCE`).
const SMALL_TRUNCATION_TOLERANCE: f64 = 1.9e-6;

/// Third-body Hansen kernels K₀ⁿˢ(χ) and their χ-derivatives, evaluated at a fixed χ⁻¹ = B over the
/// degrees `n = 0..=n_max` for one `s`.
///
/// Ported from Orekit's `HansenThirdBodyLinear`. Orekit precomputes, per `s`, a set of polynomials
/// in χ⁻¹ (composed linear transformations, `HansenUtilities.generateThirdBodyPolynomials`) and
/// evaluates them at run time; those polynomials are a *factored* form of the three-term recurrence
/// Danielson 2.7.3-(7c) / Collins 4-254. This module evaluates that same recurrence directly at the
/// working value of `B = √(1 − e²)`, which is mathematically identical and far shorter (mirroring the
/// zonal [`super::hansen::HansenZonal`] approach). It is validated against Orekit through the
/// mean-element-rate gate.
///
/// Recurrence, with `bb = B²`, `bbb = B³` and `K(n) ≡ K₀ⁿˢ`:
/// * roots (Collins 4-255/4-256/4-259): `K(s) = (−1)ˢ·(2s+1)!!/(s+1)!`,
///   `K(s+1) = (−1)ˢ·((2s+1)!!/(s+2)!)·(2s+3 − B²)`;
/// * step (n ≥ s+2): `K(n) = a·K(n−1) + b·B²·K(n−2)` with `a = (2n+1)/(n+1)`,
///   `b = −(n+s)(n−s)/(n(n+1))`.
///
/// The χ-derivative is the exact χ-derivative of that recurrence (`dB/dχ = −B²`):
/// `K'(s) = 0`, `K'(s+1) = (−1)ˢ·2·((2s+1)!!/(s+2)!)·B³`, and for n ≥ s+2
/// `K'(n) = a·K'(n−1) + b·B²·K'(n−2) − 2b·B³·K(n−2)`.
#[derive(Debug, Clone)]
pub struct HansenThirdBody {
    values: Vec<f64>,
    derivs: Vec<f64>,
}

impl HansenThirdBody {
    /// Compute all third-body kernels and χ-derivatives for order `s` up to degree `n_max` at the
    /// given `b = B = √(1 − e²)` (Orekit's `chitm1 = χ⁻¹`).
    #[must_use]
    pub fn new(n_max: usize, s: usize, b: f64) -> Self {
        let mut values = std::vec![0.0; n_max + 1];
        let mut derivs = std::vec![0.0; n_max + 1];
        let bb = b * b;
        let bbb = bb * b;

        // (-1)^s * (2s+1)!!/(s+1)! via Orekit's exact product (`twosp1dfosp1f`).
        let mut twosp1dfosp1f = if s.is_multiple_of(2) { 1.0 } else { -1.0 };
        for i in (1..=s).rev() {
            twosp1dfosp1f *= (2.0 * i as f64 + 1.0) / (i as f64 + 1.0);
        }
        let twosp1dfosp2f = twosp1dfosp1f / (s as f64 + 2.0);
        let twosp3 = (2 * s + 3) as f64;
        let two2sp1dfosp2f = 2.0 * twosp1dfosp2f;

        // Roots (Collins 4-255/4-256/4-259).
        values[s] = twosp1dfosp1f;
        derivs[s] = 0.0;
        if s < n_max {
            values[s + 1] = twosp1dfosp2f * (twosp3 - bb);
            derivs[s + 1] = two2sp1dfosp2f * bbb;
        }

        for n in (s + 2)..=n_max {
            let nf = n as f64;
            let sf = s as f64;
            let a = (2.0 * nf + 1.0) / (nf + 1.0);
            let b_coef = -((nf + sf) * (nf - sf)) / (nf * (nf + 1.0));
            values[n] = a * values[n - 1] + b_coef * bb * values[n - 2];
            derivs[n] = a * derivs[n - 1] + b_coef * bb * derivs[n - 2]
                - 2.0 * b_coef * bbb * values[n - 2];
        }

        Self { values, derivs }
    }

    /// The third-body Hansen kernel K₀ⁿˢ.
    #[must_use]
    pub fn value(&self, n: usize) -> f64 {
        self.values[n]
    }

    /// The χ-derivative dK₀ⁿˢ/dχ.
    #[must_use]
    pub fn derivative(&self, n: usize) -> f64 {
        self.derivs[n]
    }
}

/// The third-body disturbing potential U and its partial derivatives (Orekit's `UAnddU`):
/// `[U, dU/da, dU/dk, dU/dh, dU/dα, dU/dβ, dU/dγ]`. `u` is needed by the short-period corrections;
/// the mean-element rates use only the derivatives.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ThirdBodyPotential {
    pub(crate) u: f64,
    pub(crate) du_da: f64,
    pub(crate) du_dk: f64,
    pub(crate) du_dh: f64,
    pub(crate) du_dal: f64,
    pub(crate) du_dbe: f64,
    pub(crate) du_dga: f64,
}

/// The eccentricity/`a/R₃` truncation orders fixed by `DSSTThirdBodyStaticContext`.
#[derive(Debug, Clone, Copy)]
pub struct ThirdBodyTruncation {
    /// Max power of `a/R₃` in the series (`maxAR3Pow`), the highest harmonic degree retained.
    pub max_ar3_pow: usize,
    /// Max power of the eccentricity in the series (`maxEccPow`).
    pub max_ecc_pow: usize,
}

/// One third body's (Sun or Moon) contribution to the DSST mean-element rates.
///
/// Holds the central-body gravitational parameter `µ` and the perturbing body's `µ₃`. The Vₙₛ table
/// is precomputed once (as Orekit does in the `DSSTThirdBody` constructor). Sun and Moon are two
/// independent instances whose rate vectors the caller sums.
#[derive(Debug, Clone)]
pub struct DsstThirdBody {
    /// Central-body gravitational parameter `µ`, m³/s².
    pub mu: f64,
    /// Perturbing third body's gravitational parameter `µ₃`, m³/s².
    pub third_body_mu: f64,
    /// Precomputed Vₙₛ table (`CoefficientsFactory.computeVns(MAX_POWER)`).
    vns: Vec<Vec<f64>>,
}

impl DsstThirdBody {
    /// Build for one third body from the central-body `µ` and the third body's `µ₃`.
    #[must_use]
    pub fn new(mu: f64, third_body_mu: f64) -> Self {
        Self {
            mu,
            third_body_mu,
            vns: coefficients::compute_vns(MAX_POWER),
        }
    }

    /// The precomputed Vₙₛ table (`CoefficientsFactory.computeVns`). Shared with the short-period
    /// `Gₙₛ` coefficients.
    #[must_use]
    pub(crate) fn vns(&self) -> &[Vec<f64>] {
        &self.vns
    }

    /// The `(maxAR3Pow, maxEccPow)` truncation Orekit fixes at this state and geometry
    /// (`DSSTThirdBodyStaticContext`). Exposed for validation against Orekit's own values.
    ///
    /// `x = χ = 1/√(1 − e²)` and `r3 = |body_pos|`.
    #[must_use]
    pub fn truncation(&self, aux: &DsstAuxiliary, r3: f64) -> ThirdBodyTruncation {
        let x = aux.chi;

        // Factorials up to 2·MAX_POWER − 1 (Orekit's `fact` array, dim = 2·MAX_POWER).
        let dim = 2 * MAX_POWER;
        let mut fact = std::vec![0.0; dim];
        fact[0] = 1.0;
        for i in 1..dim {
            fact[i] = i as f64 * fact[i - 1];
        }

        // Truncation tolerance from the a/R₃ ratio and eccentricity.
        let aor = aux.sma / r3;
        let tol = if aor > 0.3 || (aor > 0.15 && aux.ecc > 0.25) {
            BIG_TRUNCATION_TOLERANCE
        } else {
            SMALL_TRUNCATION_TOLERANCE
        };

        // Utilities for truncation. Lower bound for eccentricity.
        let eo2 = 0.0025_f64.max(0.5 * aux.ecc);
        let x2o2 = 0.5 * x * x;
        let mut ecc_pwr = std::vec![0.0; MAX_POWER];
        let mut chi_pwr = std::vec![0.0; MAX_POWER];
        ecc_pwr[0] = 1.0;
        chi_pwr[0] = x;
        for i in 1..MAX_POWER {
            ecc_pwr[i] = ecc_pwr[i - 1] * eo2;
            chi_pwr[i] = chi_pwr[i - 1] * x2o2;
        }

        let ao2rxx = aor / (2.0 * x * x);
        let mut xmuarn = ao2rxx * ao2rxx * self.third_body_mu / (x * r3);

        let mut max_ar3_pow: i64 = 2;
        let mut max_ecc_pow: i64 = 0;
        let mut n: i64 = 2;
        let mut m: i64 = 2;
        let mut nsmd2: i64 = 0;

        loop {
            // Upper bound for Tₙₘ.
            let term = xmuarn
                * (fact[(n + m) as usize] / (fact[nsmd2 as usize] * fact[(nsmd2 + m) as usize]))
                * (fact[(n + m + 1) as usize] / (fact[m as usize] * fact[(n + 1) as usize]))
                * (fact[(n - m + 1) as usize] / fact[(n + 1) as usize])
                * ecc_pwr[m as usize]
                * coefficients::get_dnl(x * x, chi_pwr[m as usize], n + 2, m);

            if term < tol {
                if m == 0 {
                    break;
                } else if m < 2 {
                    xmuarn *= ao2rxx;
                    m = 0;
                    n += 1;
                    nsmd2 += 1;
                } else {
                    m -= 2;
                    nsmd2 += 1;
                }
            } else {
                max_ar3_pow = n;
                max_ecc_pow = m.max(max_ecc_pow);
                xmuarn *= ao2rxx;
                m += 1;
                n += 1;
            }

            if n >= MAX_POWER as i64 {
                break;
            }
        }

        max_ecc_pow = max_ar3_pow.min(max_ecc_pow);

        ThirdBodyTruncation {
            max_ar3_pow: max_ar3_pow as usize,
            max_ecc_pow: max_ecc_pow as usize,
        }
    }

    /// The six mean-element rates `[da/dt, dk/dt, dh/dt, dq/dt, dp/dt, dλ/dt]` (Orekit equinoctial
    /// ordering) from this third body at geocentric inertial position `body_pos` (m), at the given
    /// auxiliary state.
    ///
    /// The truncation `(maxAR3Pow, maxEccPow)` is computed from this state and geometry, matching a
    /// fixed-state Orekit `initializeShortPeriodTerms` + `getMeanElementRate` call.
    #[must_use]
    pub fn mean_element_rates(&self, aux: &DsstAuxiliary, body_pos: [f64; 3]) -> [f64; 6] {
        let trunc = self.truncation(aux, norm(body_pos));
        self.mean_element_rates_with_trunc(aux, body_pos, trunc)
    }

    /// Mean-element rates with an externally fixed truncation — the form used inside a propagation,
    /// where Orekit computes `(maxAR3Pow, maxEccPow)` once at the initial state and reuses it for
    /// every derivative evaluation.
    #[must_use]
    pub fn mean_element_rates_with_trunc(
        &self,
        aux: &DsstAuxiliary,
        body_pos: [f64; 3],
        trunc: ThirdBodyTruncation,
    ) -> [f64; 6] {
        let r3 = norm(body_pos);
        let inv_r3 = 1.0 / r3;
        let dir = [
            body_pos[0] * inv_r3,
            body_pos[1] * inv_r3,
            body_pos[2] * inv_r3,
        ];

        // Direction cosines: normalized body direction dotted with the equinoctial f/g/w vectors.
        let alpha = dot(dir, aux.f_vec);
        let beta = dot(dir, aux.g_vec);
        let gamma = dot(dir, aux.w_vec);

        let udu = self.potential_derivatives(aux, alpha, beta, gamma, r3, trunc);

        // Cross derivatives [Danielson 2.2-(8)], using the third-body α/β/γ.
        let u_alpha_gamma = alpha * udu.du_dga - gamma * udu.du_dal;
        let u_beta_gamma = beta * udu.du_dga - gamma * udu.du_dbe;
        let p_uag_m_iq_ubg_o_ab = (aux.p * u_alpha_gamma - I * aux.q * u_beta_gamma) * aux.ooab;

        // Mean-element rates [Danielson 3.1-(1)] — identical assembly to the zonal contribution.
        // Orekit's mCo2AB = −C/(2AB) = −aux.co2ab and m2aoA = −2a/A = −aux.ax2oa.
        let da = 0.0;
        let dh = aux.boa * udu.du_dk + aux.k * p_uag_m_iq_ubg_o_ab;
        let dk = -aux.boa * udu.du_dh - aux.h * p_uag_m_iq_ubg_o_ab;
        let dp = -aux.co2ab * u_beta_gamma;
        let dq = -aux.co2ab * u_alpha_gamma * I;
        let dm = -aux.ax2oa * udu.du_da
            + aux.boabpo * (aux.h * udu.du_dh + aux.k * udu.du_dk)
            + p_uag_m_iq_ubg_o_ab;

        [da, dk, dh, dq, dp, dm]
    }

    /// Third-body potential derivatives (Orekit `UAnddU`): the sum over `s = 0..=maxEccPow` and
    /// `n = max(2, s)..=maxAR3Pow` (with `n − s` even) of the third-body disturbing-potential terms.
    ///
    /// `u` (the potential value) is used only by the short-period corrections; the mean-element rates
    /// read the derivatives.
    pub(crate) fn potential_derivatives(
        &self,
        aux: &DsstAuxiliary,
        alpha: f64,
        beta: f64,
        gamma: f64,
        r3: f64,
        trunc: ThirdBodyTruncation,
    ) -> ThirdBodyPotential {
        let max_ar3_pow = trunc.max_ar3_pow;
        let max_ecc_pow = trunc.max_ecc_pow;
        let muo_r3 = self.third_body_mu / r3;

        // a / R₃ up to power maxAR3Pow.
        let aor3 = aux.sma / r3;
        let mut aor3_pow = std::vec![0.0; max_ar3_pow + 1];
        aor3_pow[0] = 1.0;
        for i in 1..=max_ar3_pow {
            aor3_pow[i] = aor3 * aor3_pow[i - 1];
        }

        let (gs_arr, hs_arr) = coefficients::compute_gs_hs(aux.k, aux.h, alpha, beta, max_ecc_pow);
        let qns = coefficients::compute_qns(gamma, max_ar3_pow, max_ecc_pow.max(MAX_ECCPOWER_SP));

        let b = aux.big_b;
        let hxxx = aux.h * aux.chi3;
        let kxxx = aux.k * aux.chi3;

        let mut u = 0.0;
        let mut du_da = 0.0;
        let mut du_dk = 0.0;
        let mut du_dh = 0.0;
        let mut du_dal = 0.0;
        let mut du_dbe = 0.0;
        let mut du_dga = 0.0;

        for s in 0..=max_ecc_pow {
            let hansen = HansenThirdBody::new(max_ar3_pow, s, b);
            let gs = gs_arr[s];

            // Gs partial derivatives [Danielson 3.1-(9)].
            let (mut dgsdh, mut dgsdk, mut dgsdal, mut dgsdbe) = (0.0, 0.0, 0.0, 0.0);
            if s > 0 {
                let sxgsm1 = s as f64 * gs_arr[s - 1];
                let sxhsm1 = s as f64 * hs_arr[s - 1];
                dgsdh = beta * sxgsm1 - alpha * sxhsm1;
                dgsdk = alpha * sxgsm1 + beta * sxhsm1;
                dgsdal = aux.k * sxgsm1 - aux.h * sxhsm1;
                dgsdbe = aux.h * sxgsm1 + aux.k * sxhsm1;
            }

            // Kronecker symbol (2 − δ₀ₛ).
            let delta0s = if s == 0 { 1.0 } else { 2.0 };

            for n in s.max(2)..=max_ar3_pow {
                if (n - s) % 2 == 0 {
                    let kns = hansen.value(n);
                    let dkns = hansen.derivative(n);
                    let vns = self.vns[n][s];
                    let coef0 = delta0s * aor3_pow[n] * vns;
                    let coef1 = coef0 * qns[n][s];
                    let coef2 = coef1 * kns;
                    // dQns/dGamma = Q(n, s+1) [Danielson 3.1-(8)]; for n = s, Q(n, n+1) = 0
                    // (Cefola & Broucke 1975).
                    let dqns = if n == s { 0.0 } else { qns[n][s + 1] };

                    u += coef2 * gs;
                    du_da += coef2 * n as f64 * gs;
                    du_dh += coef1 * (kns * dgsdh + hxxx * gs * dkns);
                    du_dk += coef1 * (kns * dgsdk + kxxx * gs * dkns);
                    du_dal += coef2 * dgsdal;
                    du_dbe += coef2 * dgsdbe;
                    du_dga += coef0 * kns * dqns * gs;
                }
            }
        }

        // Multiply by µ₃ / R₃ (dU/da carries the extra 1/a).
        ThirdBodyPotential {
            u: u * muo_r3,
            du_da: du_da * muo_r3 / aux.sma,
            du_dk: du_dk * muo_r3,
            du_dh: du_dh * muo_r3,
            du_dal: du_dal * muo_r3,
            du_dbe: du_dbe * muo_r3,
            du_dga: du_dga * muo_r3,
        }
    }
}

/// Euclidean norm of a 3-vector.
fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// Dot product of two 3-vectors.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{Controls, integrate_dop853};
    use crate::{DynamicalSystem, Tolerance};

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`, orekit_dsst_thirdbody.csv): the same
    // MEAN equinoctial LEO as the zonal case (a = 7.2e6, e = 0.01, i = 98°), with the Sun and the Moon
    // as third bodies, at epoch 2024-01-01T00:00:00 UTC in GCRF. Direct getMeanElementRate per body,
    // and a MEAN DSSTPropagator run with both bodies.
    const MU: f64 = 3.98600441500000000e+14;
    const SUN_GM: f64 = 1.32712440041279460e+20;
    const MOON_GM: f64 = 4.90280011845755100e+12;

    const SUN_MAX_AR3: usize = 3;
    const SUN_MAX_ECC: usize = 2;
    const MOON_MAX_AR3: usize = 4;
    const MOON_MAX_ECC: usize = 2;

    // Initial MEAN equinoctial state (a, ex=k, ey=h, hx=q, hy=p, lM=λ).
    const A0: f64 = 7.20000000000000000e+06;
    const EX0: f64 = 3.09016994374947470e-03;
    const EY0: f64 = 9.51056516295153600e-03;
    const HX0: f64 = 8.54890329218965000e-01;
    const HY0: f64 = 7.69746709859868000e-01;
    const LM0: f64 = 1.43116998663535020e+00;
    const INIT: [f64; 6] = [A0, EX0, EY0, HX0, HY0, LM0];

    // Geocentric body positions in the propagation frame (GCRF) at the epoch, m.
    const SUN_POS: [f64; 3] = [
        2.48130578062755620e+10,
        -1.33033126695834660e+11,
        -5.76679652360023650e+10,
    ];
    const MOON_POS: [f64; 3] = [
        -3.67980874244052000e+08,
        1.42721023286279650e+08,
        8.93144218169177900e+07,
    ];

    // getMeanElementRate for the Sun: [da, dk, dh, dq, dp, dM].
    const SUN_RATES: [f64; 6] = [
        0.00000000000000000e+00,
        2.16956373398949160e-14,
        -1.00495702004443560e-14,
        -9.93909076449685000e-12,
        -3.09118088926159100e-11,
        2.72444232947602650e-11,
    ];

    // getMeanElementRate for the Moon: [da, dk, dh, dq, dp, dM].
    const MOON_RATES: [f64; 6] = [
        0.00000000000000000e+00,
        1.72603527350549330e-12,
        -1.13161201468628720e-12,
        4.36466413189831100e-11,
        2.46913560075631850e-11,
        8.89245323174147000e-11,
    ];

    fn sun() -> DsstThirdBody {
        DsstThirdBody::new(MU, SUN_GM)
    }

    fn moon() -> DsstThirdBody {
        DsstThirdBody::new(MU, MOON_GM)
    }

    /// The `(maxAR3Pow, maxEccPow)` truncation the port fixes at the initial state and geometry must
    /// equal Orekit's `DSSTThirdBodyStaticContext` values (Sun: (3, 2); Moon: (4, 2)) — else different
    /// terms are retained and the rates drift.
    #[test]
    fn truncation_matches_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let ts = sun().truncation(&aux, norm(SUN_POS));
        assert_eq!((ts.max_ar3_pow, ts.max_ecc_pow), (SUN_MAX_AR3, SUN_MAX_ECC));
        let tm = moon().truncation(&aux, norm(MOON_POS));
        assert_eq!(
            (tm.max_ar3_pow, tm.max_ecc_pow),
            (MOON_MAX_AR3, MOON_MAX_ECC)
        );
    }

    // The sweep the truncation rule is checked over. `truncation` reads exactly four numbers — µ₃,
    // R₃, a/R₃ and e — so the grid is the outer product of the two bodies' µ₃, four distances from
    // GEO radius to 1 AU, and these ratios and eccentricities. `AOR_GRID` holds 0.15 and 0.3 and
    // `ECC_GRID` holds 0.25: the knees of the tolerance rule, which the sweep must reach exactly and
    // not merely straddle (asserted in the test — `sma = aor · R₃` reproduces them bit-exactly).
    const R3_GRID: [f64; 4] = [1.4959787e11, 3.844e8, 1.0e8, 4.2164e7];
    const AOR_GRID: [f64; 11] = [
        5.0e-5, 1.0e-3, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
    ];
    const ECC_GRID: [f64; 5] = [0.0, 0.05, 0.25, 0.5, 0.7];

    /// The Danielson upper bound Tₙₘ on a third-body term of degree `n` and eccentricity power `m`,
    /// in closed form, plus the rule that turns it into the retained orders.
    ///
    /// [`DsstThirdBody::truncation`] never evaluates that bound as a function of `(n, m)`: it walks a
    /// path through the lattice carrying `xmuarn` and `nsmd2` as running accumulators, and reads the
    /// eccentricity and χ powers out of tables. It is a function of `(n, m)` all the same, because two
    /// invariants hold at every
    /// point of every path the loop can take — `xmuarn = (a/(2χ²R₃))ⁿ · µ₃/(χR₃)` and
    /// `nsmd2 = (n − m)/2`. Both hold at entry `(n, m) = (2, 2)`; the accept branch and the `m == 1`
    /// branch each advance `n` and multiply `xmuarn` by `a/(2χ²R₃)` together, the first leaving
    /// `(n − m)/2` fixed and the second raising it by one (`n − m` even and `m = 1` make `n` odd, so
    /// `(n+1)/2 = (n−1)/2 + 1`); the `m ≥ 2` branch drops `m` by two and raises `nsmd2` by one with
    /// `n` and `xmuarn` untouched. Substituting both collapses the walk into [`TermBound::at`], which
    /// is what makes an independent oracle possible: the scan below owns no accumulator to get wrong.
    ///
    /// `get_dnl` is shared with the module under test, but lives in [`super::coefficients`] and is
    /// mutation-tested there.
    struct TermBound {
        third_body_mu: f64,
        chi: f64,
        ecc: f64,
        aor: f64,
        r3: f64,
        /// `i!` for `i = 0..2·MAX_POWER`; the largest index the lattice needs is `n + m + 1`.
        fact: [f64; 2 * MAX_POWER],
    }

    /// What the independent scan of the lattice yields: the orders the rule selects, and the smallest
    /// `|ln(Tₙₘ/tol)|` it saw — the margin the tightest accept/reject decision on the grid was taken
    /// with, which is what says whether the sweep resolves the rule or merely brushes past it.
    struct LatticeScan {
        max_ar3_pow: usize,
        max_ecc_pow: usize,
        tie_margin: f64,
    }

    impl TermBound {
        fn new(third_body_mu: f64, aux: &DsstAuxiliary, r3: f64) -> Self {
            let mut fact = [0.0; 2 * MAX_POWER];
            fact[0] = 1.0;
            for i in 1..fact.len() {
                fact[i] = i as f64 * fact[i - 1];
            }
            Self {
                third_body_mu,
                chi: aux.chi,
                ecc: aux.ecc,
                aor: aux.sma / r3,
                r3,
                fact,
            }
        }

        /// The truncation tolerance (`DSSTThirdBodyStaticContext`): the coarse `1e-1` once the orbit
        /// is big, or big-and-eccentric enough that the fine one would run the series past
        /// `MAX_POWER`; the fine `1.9e-6` otherwise.
        fn tol(&self) -> f64 {
            if self.aor > 0.3 || (self.aor > 0.15 && self.ecc > 0.25) {
                BIG_TRUNCATION_TOLERANCE
            } else {
                SMALL_TRUNCATION_TOLERANCE
            }
        }

        /// Tₙₘ for `n ≥ m ≥ 0` with `n − m` even.
        fn at(&self, n: i64, m: i64) -> f64 {
            let x2 = self.chi * self.chi;
            let eo2 = 0.0025_f64.max(0.5 * self.ecc);
            let xmuarn = libm::pow(self.aor / (2.0 * x2), n as f64) * self.third_body_mu
                / (self.chi * self.r3);
            let d2 = ((n - m) / 2) as usize;
            let f = &self.fact;
            xmuarn
                * (f[(n + m) as usize] / (f[d2] * f[d2 + m as usize]))
                * (f[(n + m + 1) as usize] / (f[m as usize] * f[(n + 1) as usize]))
                * (f[(n - m + 1) as usize] / f[(n + 1) as usize])
                * libm::pow(eo2, m as f64)
                * coefficients::get_dnl(x2, self.chi * libm::pow(0.5 * x2, m as f64), n + 2, m)
        }

        /// The orders the rule prescribes: the highest degree, and the highest eccentricity power,
        /// whose bound still reaches the tolerance anywhere on the lattice `2 ≤ n < MAX_POWER`,
        /// `0 ≤ m ≤ n`, `n − m` even. Degree 2 is retained unconditionally.
        fn scan(&self) -> LatticeScan {
            let tol = self.tol();
            let (mut max_n, mut max_m) = (2_i64, 0_i64);
            let mut tie_margin = f64::INFINITY;
            for n in 2..(MAX_POWER as i64) {
                let mut m = n % 2;
                while m <= n {
                    let bound = self.at(n, m);
                    if bound > 0.0 {
                        tie_margin = tie_margin.min(libm::log(bound / tol).abs());
                    }
                    if bound >= tol {
                        max_n = max_n.max(n);
                        max_m = max_m.max(m);
                    }
                    m += 2;
                }
            }
            LatticeScan {
                max_ar3_pow: max_n as usize,
                max_ecc_pow: max_m as usize,
                tie_margin,
            }
        }
    }

    /// The orders `truncation` selects equal, at every point of a 440-point sweep of the four numbers
    /// it reads (µ₃, R₃, a/R₃, e), the orders the series-bound rule prescribes: the highest degree and
    /// eccentricity power whose Danielson upper bound Tₙₘ still reaches the truncation tolerance. (G2)
    ///
    /// [`truncation_matches_orekit`] pins one fixture state per body, where the answer is (3, 2) and
    /// (4, 2) and every accept/reject decision is taken with orders of magnitude to spare. Nothing in
    /// the loop has to be right for that to pass — not the tolerance rule, not the term formula, not
    /// the `xmuarn` / `nsmd2` / `m` / `n` accumulators that walk the lattice — which is why 24 mutants
    /// of `truncation` survived it. The sweep drives the answer across 121 distinct order pairs
    /// spanning `n = 2..21` and `m = 0..21`, and lands on all three knees of the tolerance rule
    /// (`a/R₃ = 0.15`, `a/R₃ = 0.3`, `e = 0.25`) bit-exactly, so each comparison is forced on both
    /// sides and at the boundary itself.
    #[test]
    #[allow(clippy::float_cmp)] // the tolerance knees must be landed on exactly, not near
    fn truncation_matches_the_series_bound_rule() {
        assert!(
            DsstAuxiliary::new(A0, 0.25, 0.0, HX0, HY0, LM0, MU).ecc == 0.25,
            "the e = 0.25 knee of the tolerance rule is not reached exactly"
        );

        let mut seen: std::vec::Vec<(usize, usize)> = std::vec::Vec::new();
        let mut tie_margin = f64::INFINITY;
        for third_body_mu in [SUN_GM, MOON_GM] {
            let body = DsstThirdBody::new(MU, third_body_mu);
            for r3 in R3_GRID {
                for aor in AOR_GRID {
                    let sma = aor * r3;
                    if aor == 0.15 || aor == 0.3 {
                        assert!(
                            sma / r3 == aor,
                            "the a/R₃ = {aor} knee is not reached exactly at R₃ = {r3:e}"
                        );
                    }
                    for ecc in ECC_GRID {
                        let aux = DsstAuxiliary::new(sma, ecc, 0.0, HX0, HY0, LM0, MU);
                        let got = body.truncation(&aux, r3);
                        let want = TermBound::new(third_body_mu, &aux, r3).scan();
                        assert_eq!(
                            (got.max_ar3_pow, got.max_ecc_pow),
                            (want.max_ar3_pow, want.max_ecc_pow),
                            "µ₃ = {third_body_mu:e}, R₃ = {r3:e}, a/R₃ = {aor}, e = {ecc}"
                        );
                        tie_margin = tie_margin.min(want.tie_margin);
                        let orders = (got.max_ar3_pow, got.max_ecc_pow);
                        if !seen.contains(&orders) {
                            seen.push(orders);
                        }
                    }
                }
            }
        }

        // The sweep is only evidence if it moved the answer. Both orders must reach the ends of their
        // range, and the tolerance must be resolved by a wide margin rather than by a coin flip.
        let n_lo = seen.iter().map(|o| o.0).min().unwrap_or(0);
        let n_hi = seen.iter().map(|o| o.0).max().unwrap_or(0);
        let m_lo = seen.iter().map(|o| o.1).min().unwrap_or(0);
        let m_hi = seen.iter().map(|o| o.1).max().unwrap_or(0);
        assert_eq!((n_lo, n_hi), (2, MAX_POWER - 1), "a/R₃ order span");
        assert_eq!((m_lo, m_hi), (0, MAX_POWER - 1), "eccentricity order span");
        assert!(
            seen.len() >= 100,
            "only {} distinct order pairs",
            seen.len()
        );
        assert!(
            tie_margin > 1e-5,
            "tolerance decided at a tie: {tie_margin:.3e}"
        );
        std::println!(
            "truncation sweep: {} distinct (maxAR3Pow, maxEccPow) over n {n_lo}..{n_hi}, m {m_lo}..{m_hi}, closest |ln(T/tol)| {tie_margin:.3e}",
            seen.len()
        );
    }

    /// Compare a rate vector to Orekit's, returning the max relative error (`da` is identically zero,
    /// checked absolutely) and asserting each component below `1e-10`.
    fn assert_rates(got: [f64; 6], want: [f64; 6], label: &str) -> f64 {
        let mut max_rel = 0.0_f64;
        for i in 0..6 {
            let err = if want[i] == 0.0 {
                got[i].abs()
            } else {
                (got[i] - want[i]).abs() / want[i].abs()
            };
            max_rel = max_rel.max(err);
            assert!(
                err < 1e-10,
                "{label} rate[{i}] got {} want {} rel {err:.3e}",
                got[i],
                want[i]
            );
        }
        max_rel
    }

    /// The six fixed-state Sun mean-element rates match Orekit's `getMeanElementRate` to < 1e-10
    /// relative.
    #[test]
    fn fixed_state_sun_rates_match_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let got = sun().mean_element_rates(&aux, SUN_POS);
        let max_rel = assert_rates(got, SUN_RATES, "sun");
        std::println!("max fixed-state Sun rate rel error: {max_rel:.3e}");
    }

    /// The six fixed-state Moon mean-element rates match Orekit's `getMeanElementRate` to < 1e-10
    /// relative.
    #[test]
    fn fixed_state_moon_rates_match_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let got = moon().mean_element_rates(&aux, MOON_POS);
        let max_rel = assert_rates(got, MOON_RATES, "moon");
        std::println!("max fixed-state Moon rate rel error: {max_rel:.3e}");
    }

    // MEAN DSSTPropagator samples with BOTH bodies (Sun + Moon): sample times, mean equinoctial
    // states, and the Sun/Moon geocentric positions (GCRF) at each sample time. The port re-flies the
    // rates against this ephemeris (a 5-point Lagrange interpolation of the sampled body positions
    // feeds the integrator — its sub-metre position error is negligible on the ~1e-11 rates).
    const SAMPLE_T: [f64; 5] = [0.0, 21600.0, 43200.0, 64800.0, 86400.0];

    #[rustfmt::skip]
    const SAMPLE_STATE: [[f64; 6]; 5] = [
        [7.20000000000000000e+06, 3.09016994374947470e-03, 9.51056516295153600e-03, 8.54890329218965000e-01, 7.69746709859868000e-01, 1.43116998663535020e+00],
        [7.20000000000000000e+06, 3.09020748491940300e-03, 9.51054208941464400e-03, 8.54891070342577800e-01, 7.69746621381429700e-01, 2.37526990889750850e+01],
        [7.20000000000000000e+06, 3.09024402501579800e-03, 9.51052251625396700e-03, 8.54891832985364100e-01, 7.69746619321749300e-01, 4.60742280727071150e+01],
        [7.20000000000000000e+06, 3.09027869449430380e-03, 9.51050690465078200e-03, 8.54892610222829100e-01, 7.69746695153976900e-01, 6.83957569240684800e+01],
        [7.20000000000000000e+06, 3.09031063744916660e-03, 9.51049567015242400e-03, 8.54893395057535300e-01, 7.69746839728008200e-01, 9.07172856305794500e+01],
    ];

    #[rustfmt::skip]
    const SUN_EPH: [[f64; 3]; 5] = [
        [2.48130578062755620e+10, -1.33033126695834660e+11, -5.76679652360023650e+10],
        [2.54573948706984800e+10, -1.32930226786026440e+11, -5.76233810999046550e+10],
        [2.61012496643221800e+10, -1.32824745443904500e+11, -5.75776788441440700e+10],
        [2.67446099015337100e+10, -1.32716684101772370e+11, -5.75308590366453860e+10],
        [2.73874632760801120e+10, -1.32606044237044450e+11, -5.74829222629517900e+10],
    ];

    #[rustfmt::skip]
    const MOON_EPH: [[f64; 3]; 5] = [
        [-3.67980874244052000e+08, 1.42721023286279650e+08, 8.93144218169177900e+07],
        [-3.76299454181501150e+08, 1.25685986178895600e+08, 8.04947667791853800e+07],
        [-3.83545231476985500e+08, 1.08302982671580910e+08, 7.14499683252973000e+07],
        [-3.89698023132602750e+08, 9.06209838891624700e+07, 6.22056075512975160e+07],
        [-3.94740176669762300e+08, 7.26895945018838500e+07, 5.27876996025825440e+07],
    ];

    /// Component-wise 5-point Lagrange interpolation of a sampled body ephemeris at time `t`.
    fn lagrange5(t: f64, times: &[f64; 5], eph: &[[f64; 3]; 5]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for i in 0..5 {
            let mut li = 1.0;
            for j in 0..5 {
                if i != j {
                    li *= (t - times[j]) / (times[i] - times[j]);
                }
            }
            for c in 0..3 {
                out[c] += li * eph[i][c];
            }
        }
        out
    }

    /// Reduce an angle difference to (−π, π] so an accumulated mean longitude compares wrap-safe.
    fn wrap(d: f64) -> f64 {
        use core::f64::consts::TAU;
        d - TAU * libm::round(d / TAU)
    }

    /// The MEAN mean-element ODE with both bodies: `dy/dt = sun rates + moon rates + [0,0,0,0,0,n]`,
    /// with each body's truncation fixed at the initial state and its position from the interpolated
    /// ephemeris.
    struct MeanThirdBodyOde<'a> {
        sun: &'a DsstThirdBody,
        moon: &'a DsstThirdBody,
        sun_trunc: ThirdBodyTruncation,
        moon_trunc: ThirdBodyTruncation,
    }

    impl DynamicalSystem<6> for MeanThirdBodyOde<'_> {
        fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
            let aux = DsstAuxiliary::new(y[0], y[1], y[2], y[3], y[4], y[5], self.sun.mu);
            let sp = lagrange5(t, &SAMPLE_T, &SUN_EPH);
            let mp = lagrange5(t, &SAMPLE_T, &MOON_EPH);
            let rs = self
                .sun
                .mean_element_rates_with_trunc(&aux, sp, self.sun_trunc);
            let rm = self
                .moon
                .mean_element_rates_with_trunc(&aux, mp, self.moon_trunc);
            for i in 0..6 {
                dydt[i] = rs[i] + rm[i];
            }
            // Keplerian contribution (DSSTNewtonianAttraction): only the mean longitude advances at n.
            dydt[5] += aux.n;
        }
    }

    /// The MEAN semi-analytical propagation with both third bodies matches Orekit's `DSSTPropagator`
    /// at every 6 h sample: `a` to < 1e-9 relative, the equinoctial e/i components to < 1e-9 absolute,
    /// and the mean longitude to < 1 m along-track. The Sun/Moon ephemeris is a 5-point Lagrange
    /// interpolation of Orekit's sampled positions (sub-metre accurate, negligible on these rates); a
    /// production propagation would feed a real ephemeris instead.
    #[test]
    #[allow(clippy::expect_used)]
    fn mean_propagation_matches_orekit() {
        let aux0 = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let s = sun();
        let m = moon();
        let ode = MeanThirdBodyOde {
            sun: &s,
            moon: &m,
            sun_trunc: s.truncation(&aux0, norm(SUN_POS)),
            moon_trunc: m.truncation(&aux0, norm(MOON_POS)),
        };
        let tol = Tolerance::new(1e-14, 1e-12);
        let (mut max_a_rel, mut max_eh_abs, mut max_along) = (0.0_f64, 0.0_f64, 0.0_f64);
        for k in 0..5 {
            let t = SAMPLE_T[k];
            let want = SAMPLE_STATE[k];
            let got = if t == 0.0 {
                INIT
            } else {
                integrate_dop853(&ode, 0.0, &INIT, t, 60.0, Controls::new(tol, 1e-6))
                    .expect("DSST third-body mean propagation")
                    .final_state()
            };
            let a_rel = (got[0] - want[0]).abs() / want[0].abs();
            max_a_rel = max_a_rel.max(a_rel);
            for i in 1..=4 {
                max_eh_abs = max_eh_abs.max((got[i] - want[i]).abs());
            }
            let along = want[0] * wrap(got[5] - want[5]).abs();
            max_along = max_along.max(along);

            assert!(a_rel < 1e-9, "t={t}: a rel {a_rel:.3e}");
            for i in 1..=4 {
                let e = (got[i] - want[i]).abs();
                assert!(e < 1e-9, "t={t}: comp[{i}] abs {e:.3e}");
            }
            assert!(along < 1.0, "t={t}: along-track {along:.3e} m");
        }
        std::println!(
            "max third-body mean-prop errors: a_rel {max_a_rel:.3e}, e/i_abs {max_eh_abs:.3e}, along {max_along:.3e} m"
        );
    }
}
