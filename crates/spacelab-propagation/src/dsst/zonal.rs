//! The DSST central-body **zonal** mean-element rates (J2..Jₙ) and their semi-analytical
//! propagation.
//!
//! Direct port of Orekit's `DSSTZonal.getMeanElementRate` / `computeMeanElementRates` and the inner
//! `UAnddU` potential-derivative helper, restricted to the analytic **mean-element rates** — the
//! short-period corrections are out of scope. Variable names (χ, Qₙₛ, Vₙₛ, the Hansen K, the U
//! derivatives dU/da …) follow Orekit for auditability.

use std::vec::Vec;

use super::auxiliary::{DsstAuxiliary, I};
use super::coefficients;
use super::hansen::HansenZonal;
use crate::driver::{Controls, integrate_dop853};
use crate::{DynamicalSystem, Tolerance};

/// The gravitational potential U and its partial derivatives (Orekit's `UAnddU`):
/// `[U, dU/da, dU/dk, dU/dh, dU/dα, dU/dβ, dU/dγ]`. `u` (the potential value itself) is needed by
/// the short-period corrections; the mean-element rates use only the derivatives.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PotentialDerivatives {
    pub(crate) u: f64,
    du_da: f64,
    du_dk: f64,
    du_dh: f64,
    du_dal: f64,
    du_dbe: f64,
    du_dga: f64,
}

/// Zonal contribution to the central-body gravitational perturbation.
///
/// Holds the gravitational parameter `µ`, the reference radius `Rₑ`, and the **unnormalized** zonal
/// coefficients `Cₙ₀` indexed by degree `n` (`c_n0[0]`, `c_n0[1]` unused). The maximum degree is
/// `c_n0.len() - 1`. Note `Jₙ = −Cₙ₀`; [`DsstZonal::from_j`] builds from `Jₙ` directly.
#[derive(Debug, Clone)]
pub struct DsstZonal {
    /// Gravitational parameter `µ`, m³/s².
    pub mu: f64,
    /// Reference (equatorial) radius `Rₑ`, m.
    pub re: f64,
    /// Unnormalized zonal coefficients `Cₙ₀`, indexed by degree `n`.
    pub c_n0: Vec<f64>,
    /// Maximum harmonic degree.
    max_degree: usize,
    /// Maximum harmonic order (always 0 for a pure zonal field).
    max_order: usize,
    /// Precomputed Vₙₛ table.
    vns: Vec<Vec<f64>>,
}

impl DsstZonal {
    /// Build from `µ`, `Rₑ` and the **unnormalized** zonal coefficients `Cₙ₀` indexed by degree
    /// (`c_n0[0]`, `c_n0[1]` ignored; length ≥ 3, giving `max_degree = c_n0.len() - 1`).
    ///
    /// # Panics
    ///
    /// If `c_n0.len() < 3` (need at least J2).
    #[must_use]
    pub fn new(mu: f64, re: f64, c_n0: Vec<f64>) -> Self {
        assert!(c_n0.len() >= 3, "DsstZonal needs at least degree 2 (J2)");
        let max_degree = c_n0.len() - 1;
        let vns = coefficients::compute_vns(max_degree + 1);
        Self {
            mu,
            re,
            c_n0,
            max_degree,
            max_order: 0,
            vns,
        }
    }

    /// Build from `µ`, `Rₑ` and the zonal `Jₙ` coefficients `[J2, J3, …, Jₙ]` (slice indexed from
    /// degree 2), setting `Cₙ₀ = −Jₙ`.
    ///
    /// # Panics
    ///
    /// If `j` is empty.
    #[must_use]
    pub fn from_j(mu: f64, re: f64, j: &[f64]) -> Self {
        assert!(!j.is_empty(), "DsstZonal needs at least J2");
        let mut c_n0 = std::vec![0.0; j.len() + 2];
        for (i, &jn) in j.iter().enumerate() {
            c_n0[i + 2] = -jn;
        }
        Self::new(mu, re, c_n0)
    }

    /// The maximum harmonic degree of the field.
    #[must_use]
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// The precomputed Vₙₛ table (shared with the short-period coefficient generation).
    pub(crate) fn vns(&self) -> &[Vec<f64>] {
        &self.vns
    }

    /// The eccentricity-power truncation order Orekit would fix at this state
    /// (`maxEccPowMeanElements`). Exposed for validation against Orekit's own value.
    #[must_use]
    pub fn max_ecc_pow(&self, aux: &DsstAuxiliary) -> usize {
        coefficients::compute_max_ecc_pow(
            self.max_degree,
            self.max_order,
            &self.c_n0,
            self.re,
            self.mu,
            aux,
        )
    }

    /// The six mean-element rates `[da/dt, dk/dt, dh/dt, dq/dt, dp/dt, dλ/dt]` (Orekit equinoctial
    /// ordering) for the zonal disturbing potential, at the given auxiliary state.
    ///
    /// The eccentricity-power truncation is computed from this state, matching a fixed-state Orekit
    /// `initializeShortPeriodTerms` + `getMeanElementRate` call.
    #[must_use]
    pub fn mean_element_rates(&self, aux: &DsstAuxiliary) -> [f64; 6] {
        let max_ecc_pow = self.max_ecc_pow(aux);
        self.mean_element_rates_with_pow(aux, max_ecc_pow)
    }

    /// Mean-element rates with an externally fixed truncation order — the form used inside a
    /// propagation, where Orekit computes `maxEccPowMeanElements` once at the initial state and
    /// reuses it for every derivative evaluation.
    pub(crate) fn mean_element_rates_with_pow(
        &self,
        aux: &DsstAuxiliary,
        max_ecc_pow: usize,
    ) -> [f64; 6] {
        let udu = self.potential_derivatives(aux, max_ecc_pow);

        // Cross derivatives [Danielson 2.2-(8)].
        let u_alpha_gamma = aux.alpha * udu.du_dga - aux.gamma * udu.du_dal;
        let u_beta_gamma = aux.beta * udu.du_dga - aux.gamma * udu.du_dbe;
        let p_uag_m_iq_ubg_o_ab = (aux.p * u_alpha_gamma - I * aux.q * u_beta_gamma) * aux.ooab;

        // Mean-element rates [Danielson 3.1-(1)].
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

    /// Potential derivatives (Orekit `UAnddU`): the sum over `s = 0..=max_ecc_pow` and
    /// `n = s+2..=max_degree` (with `n − s` even) of the zonal disturbing-potential terms.
    pub(crate) fn potential_derivatives(
        &self,
        aux: &DsstAuxiliary,
        max_ecc_pow: usize,
    ) -> PotentialDerivatives {
        let (gs_arr, hs_arr) =
            coefficients::compute_gs_hs(aux.k, aux.h, aux.alpha, aux.beta, max_ecc_pow);
        let qns = coefficients::compute_qns(aux.gamma, self.max_degree, max_ecc_pow);

        let roa = self.re / aux.sma;
        let mut roa_pow = std::vec![0.0; self.max_degree + 1];
        roa_pow[0] = 1.0;
        for i in 1..=self.max_degree {
            roa_pow[i] = roa * roa_pow[i - 1];
        }

        let mut u = 0.0;
        let mut du_da = 0.0;
        let mut du_dk = 0.0;
        let mut du_dh = 0.0;
        let mut du_dal = 0.0;
        let mut du_dbe = 0.0;
        let mut du_dga = 0.0;

        for s in 0..=max_ecc_pow {
            let hansen = HansenZonal::new(self.max_degree, s, aux.chi);
            let gs = gs_arr[s];

            // Gs partial derivatives [Danielson 3.1-(9)].
            let (mut dgsdh, mut dgsdk, mut dgsdal, mut dgsdbe) = (0.0, 0.0, 0.0, 0.0);
            if s > 0 {
                let sxgsm1 = s as f64 * gs_arr[s - 1];
                let sxhsm1 = s as f64 * hs_arr[s - 1];
                dgsdh = aux.beta * sxgsm1 - aux.alpha * sxhsm1;
                dgsdk = aux.alpha * sxgsm1 + aux.beta * sxhsm1;
                dgsdal = aux.k * sxgsm1 - aux.h * sxhsm1;
                dgsdbe = aux.h * sxgsm1 + aux.k * sxhsm1;
            }

            // Kronecker symbol (2 − δ₀ₛ).
            let d0s = if s == 0 { 1.0 } else { 2.0 };

            for n in (s + 2)..=self.max_degree {
                if (n - s) % 2 == 0 {
                    let kns = hansen.value(n);
                    let dkns = hansen.derivative(n);
                    let vns = self.vns[n][s];
                    let coef0 = d0s * roa_pow[n] * vns * -self.c_n0[n];
                    let coef1 = coef0 * qns[n][s];
                    let coef2 = coef1 * kns;
                    let coef3 = coef2 * gs;
                    // dQns/dGamma = Q(n, s+1) [Danielson 3.1-(8)].
                    let dqns = qns[n][s + 1];

                    u += coef3;
                    du_da += coef3 * (n + 1) as f64;
                    du_dk += coef1 * (kns * dgsdk + aux.k * aux.chi3 * gs * dkns);
                    du_dh += coef1 * (kns * dgsdh + aux.h * aux.chi3 * gs * dkns);
                    du_dal += coef2 * dgsdal;
                    du_dbe += coef2 * dgsdbe;
                    du_dga += coef0 * kns * dqns * gs;
                }
            }
        }

        // Multiply by −(µ / a) (dU/da carries the extra 1/a).
        PotentialDerivatives {
            u: -aux.muoa * u,
            du_da: du_da * aux.muoa / aux.sma,
            du_dk: du_dk * -aux.muoa,
            du_dh: du_dh * -aux.muoa,
            du_dal: du_dal * -aux.muoa,
            du_dbe: du_dbe * -aux.muoa,
            du_dga: du_dga * -aux.muoa,
        }
    }

    /// Semi-analytically propagate a MEAN equinoctial state `[a, k, h, q, p, λ]` forward by `dt`
    /// seconds, integrating the mean-element rates with DOP853.
    ///
    /// This is the DSST mean propagation: `dy/dt = zonal mean-element rates + Keplerian [0,0,0,0,0,n]`
    /// (Orekit's auto-added `DSSTNewtonianAttraction` supplies the mean-longitude rate `n`). The
    /// eccentricity-power truncation is fixed once from the initial state, exactly as Orekit does.
    ///
    /// # Panics
    ///
    /// If the DOP853 driver fails (only on non-finite controls or a step underflow, which a smooth
    /// mean-element ODE with `dt > 0` does not produce).
    #[must_use]
    #[allow(clippy::expect_used)] // the invariant is documented above; a smooth ODE cannot fail here
    pub fn propagate_mean(&self, initial: [f64; 6], dt: f64) -> [f64; 6] {
        if dt == 0.0 {
            return initial;
        }
        let aux0 = DsstAuxiliary::new(
            initial[0], initial[1], initial[2], initial[3], initial[4], initial[5], self.mu,
        );
        let max_ecc_pow = self.max_ecc_pow(&aux0);
        let ode = MeanZonalOde {
            zonal: self,
            max_ecc_pow,
        };
        let tol = Tolerance::new(1e-14, 1e-12);
        let traj = integrate_dop853(&ode, 0.0, &initial, dt, 60.0, Controls::new(tol, 1e-6))
            .expect("DSST zonal mean propagation");
        traj.final_state()
    }
}

/// The mean-element ODE: zonal rates plus the Keplerian mean-longitude rate `n`, with the
/// eccentricity-power truncation fixed at the initial state.
struct MeanZonalOde<'a> {
    zonal: &'a DsstZonal,
    max_ecc_pow: usize,
}

impl DynamicalSystem<6> for MeanZonalOde<'_> {
    fn derivative(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let aux = DsstAuxiliary::new(y[0], y[1], y[2], y[3], y[4], y[5], self.zonal.mu);
        let rates = self
            .zonal
            .mean_element_rates_with_pow(&aux, self.max_ecc_pow);
        dydt.copy_from_slice(&rates);
        // Keplerian contribution (DSSTNewtonianAttraction): only the mean longitude advances at n.
        dydt[5] += aux.n;
    }
}
