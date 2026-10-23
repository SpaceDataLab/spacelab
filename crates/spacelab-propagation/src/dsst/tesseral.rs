//! The DSST central-body **tesseral** (non-zonal) mean-element rates.
//!
//! Direct port of Orekit's `DSSTTesseral.getMeanElementRate` and its inner `UAnddU` /
//! `computeNSum` machinery, restricted to the analytic **mean-element rates** — the short-period
//! corrections (the m-daily and non-resonant Fourier coefficients built in
//! `updateShortPeriodTerms`) are out of scope.
//!
//! # Scope: resonant terms only
//!
//! In Orekit the tesseral **mean** rate is a sum over *resonant* orders only (`resOrders`): a
//! tesseral harmonic of order `m` contributes to the mean drift solely when `m·θ̇ ≈ j·n` for some
//! integer `j` (θ̇ the central-body rotation rate, `n` the mean motion). Away from a resonance the
//! `resOrders` list is empty and every mean rate is identically zero — the m-daily and other
//! non-resonant terms are purely short-period. This module reproduces that path faithfully.
//!
//! # The disturbing-function machinery (Orekit / Danielson names)
//!
//! * `gamma_mns_value` — the Γᵐₙₛ(γ) inclination function (Danielson 2.7.1-(13)) and its
//!   γ-derivative `gamma_mns_deriv`;
//! * `GHmsj` / `CjSj` — the Gᵐₛʲ, Hᵐₛʲ polynomials in `(k, h, α, β)` (Danielson 2.7.5) via the
//!   `Cⱼ(x, y) + i·Sⱼ(x, y) = (x + i·y)ʲ` series;
//! * `jacobi_value_deriv` — the Jacobi polynomial `Pₗ^{v,w}(γ)` (Hipparchus `PolynomialsUtils`);
//! * `hansen_tesseral` — the tesseral Hansen kernel `Kⱼ^{-n-1,s}(χ)` and `dK/de²` from the
//!   modified-Newcomb power series (Danielson 2.7.3-(10));
//! * `newcomb` — the modified Newcomb operators `Yᵨ,σ^{n,s}` (Danielson 2.7.3-(12));
//! * `vmns` — the Vₘₙₛ coefficient (Danielson 2.8.2) (reuses the zonal Vₙₛ table).
//!
//! # Central-body rotation angle
//!
//! Orekit's mean rate depends on the central-body rotation angle θ only through the resonant phase
//! `j·λ − m·θ`. θ is Orekit's `DSSTTesseralContext` angle, which for a body-fixed frame that is a
//! pure rotation of the orbit frame about its +Z equals the Greenwich sidereal angle. The port takes
//! it as the `gmst` argument. The direction cosines α/β/γ are then the (p, q)-only equinoctial
//! values `DsstAuxiliary` carries — identical to the zonal convention.

use std::collections::HashMap;
use std::vec::Vec;

use libm::{cos, round, scalbn, sin};

use super::auxiliary::{DsstAuxiliary, I};
use super::coefficients::{compute_vns, factorial, jacobi_value_deriv};

/// Orekit `DSSTTesseral`: minimum resonant period thresholds for the resonance search.
const MIN_PERIOD_IN_SAT_REV: f64 = 10.0;
/// Orekit `DSSTTesseral`: minimum resonant period in seconds (10 days).
const MIN_PERIOD_IN_SECONDS: f64 = 864_000.0;

/// Tesseral contribution to the central-body gravitational perturbation.
///
/// Holds `µ`, the reference radius `Rₑ`, the central-body rotation rate `θ̇`, and the
/// **unnormalized** `Cₙₘ`/`Sₙₘ` coefficients as ragged `[n][m]` tables (`n = 0..=max_degree`,
/// `m = 0..=n`). `max_frequency` mirrors Orekit's `maxFrequencyShortPeriodics` default
/// `min(12, max_degree + 4)`, the upper bound `j` may take in the resonance search.
#[derive(Debug, Clone)]
pub struct DsstTesseral {
    /// Gravitational parameter `µ`, m³/s².
    pub mu: f64,
    /// Reference (equatorial) radius `Rₑ`, m.
    pub re: f64,
    /// Central-body rotation rate `θ̇`, rad/s.
    pub central_body_rotation_rate: f64,
    /// Unnormalized `Cₙₘ`, ragged `[n][m]`.
    pub cnm: Vec<Vec<f64>>,
    /// Unnormalized `Sₙₘ`, ragged `[n][m]`.
    pub snm: Vec<Vec<f64>>,
    /// Maximum harmonic degree.
    pub(crate) max_degree: usize,
    /// Maximum harmonic order.
    pub(crate) max_order: usize,
    /// Upper bound on the resonant index `j` (`maxFrequencyShortPeriodics`).
    pub(crate) max_frequency: usize,
    /// Precomputed Vₙₛ table (for the Vₘₙₛ coefficient), dense `[n][s]`.
    vns: Vec<Vec<f64>>,
}

impl DsstTesseral {
    /// Build from `µ`, `Rₑ`, the central-body rotation rate `θ̇`, and the **unnormalized**
    /// `Cₙₘ`/`Sₙₘ` coefficients (ragged `[n][m]`, `n = 0..=max_degree`, row `n` holding `m = 0..=n`).
    ///
    /// `max_degree = cnm.len() - 1` and `max_order = max_degree` (a full square field). The
    /// resonance-search bound `max_frequency` is Orekit's default `min(12, max_degree + 4)`.
    ///
    /// # Panics
    ///
    /// If `cnm`/`snm` disagree in shape or the field is below degree 2.
    #[must_use]
    pub fn new(
        mu: f64,
        re: f64,
        central_body_rotation_rate: f64,
        cnm: Vec<Vec<f64>>,
        snm: Vec<Vec<f64>>,
    ) -> Self {
        assert!(cnm.len() >= 3, "DsstTesseral needs at least degree 2");
        assert_eq!(cnm.len(), snm.len(), "Cnm/Snm degree mismatch");
        let max_degree = cnm.len() - 1;
        let max_order = max_degree;
        let max_frequency = (max_degree + 4).min(12);
        let vns = compute_vns(max_degree + 1);
        Self {
            mu,
            re,
            central_body_rotation_rate,
            cnm,
            snm,
            max_degree,
            max_order,
            max_frequency,
            vns,
        }
    }

    /// The six mean-element rates `[da/dt, dk/dt, dh/dt, dq/dt, dp/dt, dλ/dt]` (Orekit equinoctial
    /// ordering) for the tesseral disturbing potential, at the given auxiliary state.
    ///
    /// `gmst` is the central-body rotation angle θ at the epoch (the Greenwich sidereal angle for a
    /// pure-Z body frame); it enters only through the resonant phase `j·λ − m·θ`. Returns all zeros
    /// away from a tesseral resonance, exactly as Orekit does.
    #[must_use]
    pub fn mean_element_rates(&self, aux: &DsstAuxiliary, gmst: f64) -> [f64; 6] {
        let udu = self.potential_derivatives(aux, gmst);
        let [du_da, du_dk, du_dh, du_dl, du_dal, du_dbe, du_dga] = udu;

        // Cross-derivative operators (Orekit getMeanElementRate).
        let u_alpha_gamma = aux.alpha * du_dga - aux.gamma * du_dal;
        let u_alpha_beta = aux.alpha * du_dbe - aux.beta * du_dal;
        let u_beta_gamma = aux.beta * du_dga - aux.gamma * du_dbe;
        let uhk = aux.h * du_dk - aux.k * du_dh;
        let p_uag_m_iq_ubg_o_ab = (aux.p * u_alpha_gamma - I * aux.q * u_beta_gamma) * aux.ooab;
        let uhkm_uab_m_dudl = uhk - u_alpha_beta - du_dl;

        let da = aux.ax2oa * du_dl;
        let dh = aux.boa * du_dk + aux.k * p_uag_m_iq_ubg_o_ab - aux.h * aux.boabpo * du_dl;
        let dk = -(aux.boa * du_dh + aux.h * p_uag_m_iq_ubg_o_ab + aux.k * aux.boabpo * du_dl);
        let dp = aux.co2ab * (aux.p * uhkm_uab_m_dudl - u_beta_gamma);
        let dq = aux.co2ab * (aux.q * uhkm_uab_m_dudl - I * u_alpha_gamma);
        let dm =
            -aux.ax2oa * du_da + aux.boabpo * (aux.h * du_dh + aux.k * du_dk) + p_uag_m_iq_ubg_o_ab;

        [da, dk, dh, dq, dp, dm]
    }

    /// The resonant `(m, j)` orders for this field and orbit (Orekit `getResonantAndNonResonantTerms`
    /// natural-resonance branch): `θ̇·m ≈ j·n` within the sat-rev / period tolerance, `1 ≤ j ≤
    /// max_frequency`.
    pub(crate) fn resonant_orders(&self, aux: &DsstAuxiliary) -> Vec<(usize, i64)> {
        let ratio = self.central_body_rotation_rate / aux.n; // = period / bodyPeriod
        let orbit_period = core::f64::consts::TAU / aux.n;
        let tolerance = 1.0 / MIN_PERIOD_IN_SAT_REV.max(MIN_PERIOD_IN_SECONDS / orbit_period);
        let mut res = Vec::new();
        for m in 1..=self.max_order {
            let resonance = ratio * m as f64;
            let j_res = round(resonance) as i64;
            if j_res > 0
                && j_res <= self.max_frequency as i64
                && (resonance - j_res as f64).abs() <= tolerance
            {
                res.push((m, j_res));
            }
        }
        res
    }

    /// The potential derivatives `[dU/da, dU/dk, dU/dh, dU/dl, dU/dα, dU/dβ, dU/dγ]` (Orekit's
    /// tesseral `UAnddU`): the resonant sum over `{m, j}` of the `s`- and `n`-sums.
    fn potential_derivatives(&self, aux: &DsstAuxiliary, gmst: f64) -> [f64; 7] {
        let res_orders = self.resonant_orders(aux);
        if res_orders.is_empty() {
            return [0.0; 7];
        }

        let max_ecc_pow = get_max_ecc_pow(aux.ecc);
        let max_hansen = (max_ecc_pow / 2) as i64;
        let e2 = aux.ecc * aux.ecc;
        let opig = 1.0 + I * aux.gamma;

        // R / a up to power max_degree.
        let roa = self.re / aux.sma;
        let mut roa_pow = std::vec![0.0; self.max_degree + 1];
        roa_pow[0] = 1.0;
        for i in 1..=self.max_degree {
            roa_pow[i] = roa * roa_pow[i - 1];
        }

        let gh = GHmsj::new(aux.k, aux.h, aux.alpha, aux.beta, self.cjsj_max());
        let mut newcomb_cache: HashMap<(i64, i64, i64, i64), f64> = HashMap::new();

        let mut du = [0.0_f64; 7];
        for (m, j) in res_orders {
            let mi = m as i64;
            // Phase angle j·λ − m·θ.
            let jl_mmt = j as f64 * aux.lambda - mi as f64 * gmst;
            let sin_phi = sin(jl_mmt);
            let cos_phi = cos(jl_mmt);

            // Per-{m, j} cos/sin accumulators for the seven derivative components.
            let mut acc = [[0.0_f64; 2]; 7];

            let s_min = (max_ecc_pow as i64 - j).min(self.max_degree as i64);
            let s_max = (max_ecc_pow as i64 + j).min(self.max_degree as i64);
            let mut s = 0_i64;
            while s <= s_max {
                let n_pos = self.compute_n_sum(
                    j,
                    mi,
                    s,
                    opig,
                    aux,
                    &roa_pow,
                    &gh,
                    max_hansen,
                    e2,
                    &mut newcomb_cache,
                );
                for r in 0..7 {
                    acc[r][0] += n_pos[r][0];
                    acc[r][1] += n_pos[r][1];
                }
                if s > 0 && s <= s_min {
                    let n_neg = self.compute_n_sum(
                        j,
                        mi,
                        -s,
                        opig,
                        aux,
                        &roa_pow,
                        &gh,
                        max_hansen,
                        e2,
                        &mut newcomb_cache,
                    );
                    for r in 0..7 {
                        acc[r][0] += n_neg[r][0];
                        acc[r][1] += n_neg[r][1];
                    }
                }
                s += 1;
            }

            for r in 0..7 {
                du[r] += cos_phi * acc[r][0] + sin_phi * acc[r][1];
            }
        }

        // Scale by µ/a (dU/da carries the extra −1/a; Orekit assembles the leading minus into dM).
        let mu_on_a = aux.muoa;
        [
            du[0] * (-mu_on_a / aux.sma), // dU/da
            du[2] * mu_on_a,              // dU/dk
            du[1] * mu_on_a,              // dU/dh
            du[3] * mu_on_a,              // dU/dl
            du[4] * mu_on_a,              // dU/dα
            du[5] * mu_on_a,              // dU/dβ
            du[6] * mu_on_a,              // dU/dγ
        ]
    }

    /// The `n`-sum for the potential-derivative components at fixed `{j, m, s}` (Orekit
    /// `computeNSum`). Returns `[[cos, sin]; 7]` for `dU/da, dU/dh, dU/dk, dU/dl, dU/dα, dU/dβ,
    /// dU/dγ` (Orekit's internal ordering).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_n_sum(
        &self,
        j: i64,
        m: i64,
        s: i64,
        opig: f64,
        aux: &DsstAuxiliary,
        roa_pow: &[f64],
        gh: &GHmsj,
        max_hansen: i64,
        e2: f64,
        newcomb_cache: &mut HashMap<(i64, i64, i64, i64), f64>,
    ) -> [[f64; 2]; 7] {
        let mut out = [[0.0_f64; 2]; 7];

        // Jacobi v, w indices [Danielson 2.7.1-(15)]; I^m = 1 (I = +1).
        let v = (m - s).unsigned_abs() as i64;
        let w = (m + s).unsigned_abs() as i64;
        let nmin = 2.max(m).max(s.abs());

        let mut n = nmin;
        while n <= self.max_degree as i64 {
            if (n - s) % 2 == 0 {
                let v_mns = self.vmns(m, n, s);
                let ga_mns = gamma_mns_value(m, n, s, opig);
                let dga_mns = gamma_mns_deriv(m, n, s, opig);
                let (kjns, dkjns) = hansen_tesseral(
                    -n - 1,
                    s,
                    j,
                    max_hansen,
                    e2,
                    aux.chi,
                    aux.chi2,
                    newcomb_cache,
                );

                let g_msj = gh.gmsj(m, s, j);
                let h_msj = gh.hmsj(m, s, j);
                let dgdh = gh.dgmsdh(m, s, j);
                let dgdk = gh.dgmsdk(m, s, j);
                let dgda = gh.dgmsd_alpha(m, s, j);
                let dgdb = gh.dgmsd_beta(m, s, j);
                let dhdh = gh.dhmsdh(m, s, j);
                let dhdk = gh.dhmsdk(m, s, j);
                let dhda = gh.dhmsd_alpha(m, s, j);
                let dhdb = gh.dhmsd_beta(m, s, j);

                let l = (n - m).min(n - s.abs());
                let (jac_v, jac_d) = jacobi_value_deriv(l as usize, v, w, aux.gamma);

                let cnm = self.cnm[n as usize][m as usize];
                let snm = self.snm[n as usize][m as usize];

                // Common factors [Danielson 3.3-(4) expansion].
                let cf_0 = roa_pow[n as usize] * v_mns;
                let cf_1 = cf_0 * ga_mns * jac_v;
                let cf_2 = cf_1 * kjns;
                let gc_phs = g_msj * cnm + h_msj * snm;
                let gs_mhc = g_msj * snm - h_msj * cnm;
                let dk_gc_phsx2 = 2.0 * dkjns * gc_phs;
                let dk_gs_mhcx2 = 2.0 * dkjns * gs_mhc;
                let duda_coef = (n + 1) as f64 * cf_2;
                let dudl_coef = j as f64 * cf_2;
                let dudga_coef = cf_0 * kjns * (jac_v * dga_mns + ga_mns * jac_d);

                // dU/da
                out[0][0] += duda_coef * gc_phs;
                out[0][1] += duda_coef * gs_mhc;
                // dU/dh
                out[1][0] += cf_1 * (kjns * (cnm * dgdh + snm * dhdh) + aux.h * dk_gc_phsx2);
                out[1][1] += cf_1 * (kjns * (snm * dgdh - cnm * dhdh) + aux.h * dk_gs_mhcx2);
                // dU/dk
                out[2][0] += cf_1 * (kjns * (cnm * dgdk + snm * dhdk) + aux.k * dk_gc_phsx2);
                out[2][1] += cf_1 * (kjns * (snm * dgdk - cnm * dhdk) + aux.k * dk_gs_mhcx2);
                // dU/dλ
                out[3][0] += dudl_coef * gs_mhc;
                out[3][1] += -dudl_coef * gc_phs;
                // dU/dα
                out[4][0] += cf_2 * (dgda * cnm + dhda * snm);
                out[4][1] += cf_2 * (dgda * snm - dhda * cnm);
                // dU/dβ
                out[5][0] += cf_2 * (dgdb * cnm + dhdb * snm);
                out[5][1] += cf_2 * (dgdb * snm - dhdb * cnm);
                // dU/dγ
                out[6][0] += dudga_coef * gc_phs;
                out[6][1] += dudga_coef * gs_mhc;
            }
            n += 1;
        }
        out
    }

    /// The Vₘₙₛ coefficient from the zonal Vₙₛ table (Orekit `CoefficientsFactory.getVmns`).
    fn vmns(&self, m: i64, n: i64, s: i64) -> f64 {
        if (n - s) % 2 != 0 {
            return 0.0;
        }
        let fns = factorial(n + s.abs());
        let fnm = factorial(n - m);
        if s >= 0 {
            fns * self.vns[n as usize][s as usize] / fnm
        } else {
            // Vmn,-s = (−1)^s Vmns.
            let mops = if s & 1 == 0 { 1.0 } else { -1.0 };
            mops * fns * self.vns[n as usize][(-s) as usize] / fnm
        }
    }

    /// Upper bound for the `Cⱼ`/`Sⱼ` series order needed by `GHmsj`.
    pub(crate) fn cjsj_max(&self) -> usize {
        2 * self.max_order + self.max_degree + self.max_frequency + 2
    }
}

/// The maximum power of the eccentricity kept in the summation over `s` (Orekit
/// `DSSTTesseral.getMaxEccPow`), a step function of the eccentricity.
pub(crate) fn get_max_ecc_pow(e: f64) -> usize {
    if e <= 0.005 {
        3
    } else if e <= 0.02 {
        4
    } else if e <= 0.1 {
        7
    } else if e <= 0.2 {
        10
    } else if e <= 0.3 {
        12
    } else if e <= 0.4 {
        15
    } else {
        20
    }
}

/// `base^exp` for an integer exponent (repeated multiplication; `exp` small, `|exp| ≤ max_degree`).
fn powi(base: f64, exp: i64) -> f64 {
    let mut acc = 1.0;
    let n = exp.unsigned_abs();
    for _ in 0..n {
        acc *= base;
    }
    if exp < 0 { 1.0 / acc } else { acc }
}

/// The Γᵐₙₛ(γ) inclination function value (Danielson 2.7.1-(13)). `opig = 1 + I·γ`, `I = +1`.
fn gamma_mns_value(m: i64, n: i64, s: i64, opig: f64) -> f64 {
    let sign = if (m - s) & 1 == 0 { 1.0 } else { -1.0 };
    if s <= -m {
        sign * scalbn(powi(opig, -m), s as i32)
    } else if s <= m {
        sign * scalbn(powi(opig, s), (-m) as i32) * gamma_ratio(m, n, s)
    } else {
        scalbn(powi(opig, m), (-s) as i32)
    }
}

/// The dΓᵐₙₛ/dγ derivative (Orekit `GammaMnsFunction.getDerivative`), `I = +1`.
fn gamma_mns_deriv(m: i64, n: i64, s: i64, opig: f64) -> f64 {
    let value = gamma_mns_value(m, n, s, opig);
    if s <= -m {
        -(m as f64) * value / opig
    } else if s >= m {
        (m as f64) * value / opig
    } else {
        s as f64 * value / opig
    }
}

/// The factorial ratio used by the middle branch of Γᵐₙₛ (Orekit `GammaMnsFunction.getRatios`),
/// evaluated directly for `(m, n, s)`; symmetric in `s → |s|`.
fn gamma_ratio(m: i64, n: i64, s: i64) -> f64 {
    let mut r = 1.0;
    // R(m, n, 0) = Π_{i=1}^{m} (n + i) / (n − i + 1).
    for i in 1..=m {
        r *= (n + i) as f64 / (n - i + 1) as f64;
    }
    // R(m, n, |s|) = R(m, n, 0) · Π_{a=1}^{|s|} (n − a + 1) / (n + a).
    for a in 1..=s.abs() {
        r *= (n - a + 1) as f64 / (n + a) as f64;
    }
    r
}

/// The tesseral Hansen kernel `Kⱼ^{mnm1,s}(χ)` and its `de²` derivative, from the modified-Newcomb
/// power series (Danielson 2.7.3-(10)) with `mnm1 = −n−1`. Valid for `max_degree ≤ 5`, where
/// Orekit's slice/linear-recurrence extension is not exercised and every kernel is a series root.
#[allow(clippy::too_many_arguments)]
fn hansen_tesseral(
    mnm1: i64,
    s: i64,
    j: i64,
    max_hansen: i64,
    e2: f64,
    chi: f64,
    chi2: f64,
    cache: &mut HashMap<(i64, i64, i64, i64), f64>,
) -> (f64, f64) {
    let a_ht = (j - s).max(0);
    let b_ht = (s - j).max(0);

    // Σ_α Y^{n,s}_{α+aHT, α+bHT} e^{2α}, and its e²-derivative.
    let mut serie = 0.0;
    let mut dserie = 0.0;
    let mut e2_pow = 1.0; // e²^α
    for alpha in 0..=max_hansen {
        let c = newcomb(cache, alpha + a_ht, alpha + b_ht, mnm1, s);
        serie += c * e2_pow;
        if alpha >= 1 {
            dserie += c * alpha as f64 * powi(e2, alpha - 1);
        }
        e2_pow *= e2;
    }

    let chi_pow = powi(chi2, -mnm1 - 1); // χ2^{−mnm1−1} = χ2^n
    let value = chi_pow * serie / chi;
    let coef = -(mnm1 as f64 + 1.5);
    let derivative = coef * chi2 * value + chi_pow * dserie / chi;
    (value, derivative)
}

/// The modified Newcomb operator `Yᵨ,σ^{n,s}` (Danielson 2.7.3-(12)), memoized scalar recurrence.
/// `Y₀,₀ = 1`, `Y = 0` for `ρ < 0` or `σ < 0`.
#[allow(clippy::too_many_arguments)]
fn newcomb(
    cache: &mut HashMap<(i64, i64, i64, i64), f64>,
    rho: i64,
    sigma: i64,
    n: i64,
    s: i64,
) -> f64 {
    if rho < 0 || sigma < 0 {
        return 0.0;
    }
    if rho == 0 && sigma == 0 {
        return 1.0;
    }
    let key = (rho, sigma, n, s);
    if let Some(&v) = cache.get(&key) {
        return v;
    }
    let val = (2.0 * (2 * s - n) as f64 * newcomb(cache, rho - 1, sigma, n, s + 1)
        + (s - n) as f64 * newcomb(cache, rho - 2, sigma, n, s + 2)
        - 2.0 * (2 * s + n) as f64 * newcomb(cache, rho, sigma - 1, n, s - 1)
        - (s + n) as f64 * newcomb(cache, rho, sigma - 2, n, s - 2)
        + 2.0
            * (2 * rho + 2 * sigma + 2 + 3 * n) as f64
            * newcomb(cache, rho - 1, sigma - 1, n, s))
        / (4.0 * (rho + sigma) as f64);
    cache.insert(key, val);
    val
}

/// The `Cⱼ(x, y)`, `Sⱼ(x, y)` series (Danielson 2.5.3-(5)): `Cⱼ + i·Sⱼ = (x + i·y)ʲ`, with the
/// `k`/`h` partial derivatives.
struct CjSj {
    cj: Vec<f64>,
    sj: Vec<f64>,
}

impl CjSj {
    fn new(x: f64, y: f64, jmax: usize) -> Self {
        let mut cj = std::vec![0.0; jmax + 1];
        let mut sj = std::vec![0.0; jmax + 1];
        cj[0] = 1.0;
        sj[0] = 0.0;
        for jj in 1..=jmax {
            cj[jj] = cj[jj - 1] * x - sj[jj - 1] * y;
            sj[jj] = cj[jj - 1] * y + sj[jj - 1] * x;
        }
        Self { cj, sj }
    }

    fn cj(&self, jj: i64) -> f64 {
        self.cj[jj as usize]
    }
    fn sj(&self, jj: i64) -> f64 {
        self.sj[jj as usize]
    }
    fn dcj_dk(&self, jj: i64) -> f64 {
        if jj == 0 {
            0.0
        } else {
            jj as f64 * self.cj(jj - 1)
        }
    }
    fn dsj_dk(&self, jj: i64) -> f64 {
        if jj == 0 {
            0.0
        } else {
            jj as f64 * self.sj(jj - 1)
        }
    }
    fn dcj_dh(&self, jj: i64) -> f64 {
        if jj == 0 {
            0.0
        } else {
            -(jj as f64) * self.sj(jj - 1)
        }
    }
    fn dsj_dh(&self, jj: i64) -> f64 {
        if jj == 0 {
            0.0
        } else {
            jj as f64 * self.cj(jj - 1)
        }
    }
}

/// Sign of an integer as `±1.0` (Orekit `GHmsjPolynomials.sgn`; note `sgn(0) = +1`).
fn sgn(i: i64) -> f64 {
    if i < 0 { -1.0 } else { 1.0 }
}

/// The Gᵐₛʲ, Hᵐₛʲ polynomials in `(k, h, α, β)` and their `(k, h, α, β)` partials
/// (Danielson 2.7.5-(1)(2)), Orekit `GHmsjPolynomials` (retrograde factor `I = +1`).
pub(crate) struct GHmsj {
    kh: CjSj,
    ab: CjSj,
}

impl GHmsj {
    pub(crate) fn new(k: f64, h: f64, alpha: f64, beta: f64, jmax: usize) -> Self {
        Self {
            kh: CjSj::new(k, h, jmax),
            ab: CjSj::new(alpha, beta, jmax),
        }
    }

    fn gmsj(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s; // I = +1
            self.kh.cj(s_mj) * self.ab.cj(m_mis) - sgn(s - j) * self.kh.sj(s_mj) * self.ab.sj(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            self.kh.cj(s_mj) * self.ab.cj(s_mim)
                + sgn(s - j) * sgn(s - m) * self.kh.sj(s_mj) * self.ab.sj(s_mim)
        }
    }

    fn hmsj(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            I * self.kh.cj(s_mj) * self.ab.sj(m_mis)
                + sgn(s - j) * self.kh.sj(s_mj) * self.ab.cj(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            -sgn(s - m) * self.kh.cj(s_mj) * self.ab.sj(s_mim)
                + sgn(s - j) * self.kh.sj(s_mj) * self.ab.cj(s_mim)
        }
    }

    fn dgmsdk(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            self.kh.dcj_dk(s_mj) * self.ab.cj(m_mis)
                - sgn(s - j) * self.kh.dsj_dk(s_mj) * self.ab.sj(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            self.kh.dcj_dk(s_mj) * self.ab.cj(s_mim)
                + sgn(s - j) * sgn(s - m) * self.kh.dsj_dk(s_mj) * self.ab.sj(s_mim)
        }
    }

    fn dgmsdh(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            self.kh.dcj_dh(s_mj) * self.ab.cj(m_mis)
                - sgn(s - j) * self.kh.dsj_dh(s_mj) * self.ab.sj(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            self.kh.dcj_dh(s_mj) * self.ab.cj(s_mim)
                + sgn(s - j) * sgn(s - m) * self.kh.dsj_dh(s_mj) * self.ab.sj(s_mim)
        }
    }

    fn dgmsd_alpha(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            self.kh.cj(s_mj) * self.ab.dcj_dk(m_mis)
                - sgn(s - j) * self.kh.sj(s_mj) * self.ab.dsj_dk(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            self.kh.cj(s_mj) * self.ab.dcj_dk(s_mim)
                + sgn(s - j) * sgn(s - m) * self.kh.sj(s_mj) * self.ab.dsj_dk(s_mim)
        }
    }

    fn dgmsd_beta(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            self.kh.cj(s_mj) * self.ab.dcj_dh(m_mis)
                - sgn(s - j) * self.kh.sj(s_mj) * self.ab.dsj_dh(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            self.kh.cj(s_mj) * self.ab.dcj_dh(s_mim)
                + sgn(s - j) * sgn(s - m) * self.kh.sj(s_mj) * self.ab.dsj_dh(s_mim)
        }
    }

    fn dhmsdk(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            I * self.kh.dcj_dk(s_mj) * self.ab.sj(m_mis)
                + sgn(s - j) * self.kh.dsj_dk(s_mj) * self.ab.cj(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            -sgn(s - m) * self.kh.dcj_dk(s_mj) * self.ab.sj(s_mim)
                + sgn(s - j) * self.kh.dsj_dk(s_mj) * self.ab.cj(s_mim)
        }
    }

    fn dhmsdh(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            I * self.kh.dcj_dh(s_mj) * self.ab.sj(m_mis)
                + sgn(s - j) * self.kh.dsj_dh(s_mj) * self.ab.cj(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            -sgn(s - m) * self.kh.dcj_dh(s_mj) * self.ab.sj(s_mim)
                + sgn(s - j) * self.kh.dsj_dh(s_mj) * self.ab.cj(s_mim)
        }
    }

    fn dhmsd_alpha(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            I * self.kh.cj(s_mj) * self.ab.dsj_dk(m_mis)
                + sgn(s - j) * self.kh.sj(s_mj) * self.ab.dcj_dk(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            -sgn(s - m) * self.kh.cj(s_mj) * self.ab.dsj_dk(s_mim)
                + sgn(s - j) * self.kh.sj(s_mj) * self.ab.dcj_dk(s_mim)
        }
    }

    fn dhmsd_beta(&self, m: i64, s: i64, j: i64) -> f64 {
        let s_mj = (s - j).abs();
        if s.abs() <= m {
            let m_mis = m - I as i64 * s;
            I * self.kh.cj(s_mj) * self.ab.dsj_dh(m_mis)
                + sgn(s - j) * self.kh.sj(s_mj) * self.ab.dcj_dh(m_mis)
        } else {
            let s_mim = (s - I as i64 * m).abs();
            -sgn(s - m) * self.kh.cj(s_mj) * self.ab.dsj_dh(s_mim)
                + sgn(s - j) * self.kh.sj(s_mj) * self.ab.dcj_dh(s_mim)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden fixtures from Orekit 13 (tools/orekit-oracle, orekit_dsst_tesseral.csv): unnormalized
    // 4x4 field, WGS84 Earth rotation rate, body frame = pure +Z rotation of the orbit frame by
    // gmst = 1.234 (so theta = gmst exactly and alpha/beta/gamma reduce to the (p,q)-only values).
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const RATE: f64 = 7.29211500000000000e-05;
    const GMST: f64 = 1.234;

    // Unnormalized Cnm/Snm (n, m) for n = 2..=4.
    #[rustfmt::skip]
    const CS: [(usize, usize, f64, f64); 12] = [
        (2, 0, -1.08262689216735860e-03,  0.00000000000000000e+00),
        (2, 1, -7.74637991057574400e-10,  2.13693885436337300e-09),
        (2, 2,  1.57460625724152070e-06, -9.03896492990687600e-07),
        (3, 0,  2.53209950653743350e-06,  0.00000000000000000e+00),
        (3, 1,  2.19321948061808240e-06,  2.68492359875363950e-07),
        (3, 2,  3.09070220068959660e-07, -2.11471885565559140e-07),
        (3, 3,  1.00581879094907070e-07,  1.97234537055279140e-07),
        (4, 0,  1.62004577259847830e-06,  0.00000000000000000e+00),
        (4, 1, -5.08710241363811900e-07, -4.49279874110026540e-07),
        (4, 2,  7.83670455129086500e-08,  1.48157820133437310e-07),
        (4, 3,  5.92141380826491500e-08, -1.20029237873928450e-08),
        (4, 4, -3.98333574448002600e-09,  6.52507740576946850e-09),
    ];

    // Cases: (a, ex=k, ey=h, hx=q, hy=p, lM=λ) and the six Orekit rates [da, dk, dh, dq, dp, dM].
    #[rustfmt::skip]
    const LEO_STATE: [f64; 6] = [
        7.20000000000000000e+06, 3.09016994374947470e-03, 9.51056516295153600e-03,
        8.54890329218965000e-01, 7.69746709859868000e-01, 1.43116998663535020e+00,
    ];

    #[rustfmt::skip]
    const GEO_STATE: [f64; 6] = [
        4.21641730000000000e+07, 3.09016994374947470e-03, 9.51056516295153600e-03,
        9.78368723432483700e-02, 8.80927155393123200e-02, 1.43116998663535020e+00,
    ];
    #[rustfmt::skip]
    const GEO_RATES: [f64; 6] = [
        -1.06733871520185040e-03,  6.04536719151678900e-13, -7.50725497563041100e-13,
         8.20860240597892300e-13,  3.71876383825889560e-13,  3.18754236634667500e-11,
    ];

    // The same GEO geometry at e = 0.004, below Orekit's 0.005 maxEccPow step.
    #[rustfmt::skip]
    const GEO_LOW_ECC_STATE: [f64; 6] = [
        4.21641730000000000e+07, 1.23606797749978980e-03, 3.80422606518061440e-03,
        9.78368723432483700e-02, 8.80927155393123200e-02, 1.43116998663535020e+00,
    ];
    #[rustfmt::skip]
    const GEO_LOW_ECC_RATES: [f64; 6] = [
        -1.06685122946418300e-03,  3.81370893742825900e-13, -7.21261432137843700e-13,
         8.12884379310904900e-13,  3.70083796331287900e-13,  3.19109969492273800e-11,
    ];

    #[rustfmt::skip]
    const GPS_STATE: [f64; 6] = [
        2.65600000000000000e+07, 3.09016994374947470e-03, 9.51056516295153600e-03,
        3.86856709931559350e-01, 3.48327346186132370e-01, 1.43116998663535020e+00,
    ];
    #[rustfmt::skip]
    const GPS_RATES: [f64; 6] = [
        -5.14526727073451100e-05, -1.22171328238994320e-11,  5.82552533402288400e-13,
         1.26617543091296620e-13,  1.48281208638392410e-12, -2.94808861626080470e-12,
    ];

    fn tesseral() -> DsstTesseral {
        tesseral_with_rate(RATE)
    }

    fn tesseral_with_rate(rate: f64) -> DsstTesseral {
        let deg = 4usize;
        let mut cnm = std::vec![std::vec![0.0; 1]; deg + 1];
        let mut snm = std::vec![std::vec![0.0; 1]; deg + 1];
        for n in 0..=deg {
            cnm[n] = std::vec![0.0; n + 1];
            snm[n] = std::vec![0.0; n + 1];
        }
        for &(n, m, c, s) in &CS {
            cnm[n][m] = c;
            snm[n][m] = s;
        }
        DsstTesseral::new(MU, AE, rate, cnm, snm)
    }

    fn aux(state: &[f64; 6]) -> DsstAuxiliary {
        DsstAuxiliary::new(
            state[0], state[1], state[2], state[3], state[4], state[5], MU,
        )
    }

    /// Score every rate component against its own Orekit magnitude (< 1e-9 relative). No component
    /// needs the near-zero-cancellation relaxation: even the ~1e-13 e/i-vector rates track Orekit to
    /// ~1e-14 relative, since the resonant potential and its derivatives are smooth here.
    fn check(name: &str, state: &[f64; 6], want: &[f64; 6]) -> f64 {
        let got = tesseral().mean_element_rates(&aux(state), GMST);
        let mut max_rel = 0.0_f64;
        for i in 0..6 {
            let err = if want[i] == 0.0 {
                got[i].abs()
            } else {
                (got[i] - want[i]).abs() / want[i].abs()
            };
            max_rel = max_rel.max(err);
            assert!(
                err < 1e-9,
                "{name} rate[{i}] got {} want {} rel {err:.3e}",
                got[i],
                want[i]
            );
        }
        max_rel
    }

    /// LEO (a = 7.2e6): no tesseral resonance -> all six mean rates are identically zero.
    #[test]
    fn leo_no_resonance_is_zero() {
        let got = tesseral().mean_element_rates(&aux(&LEO_STATE), GMST);
        for (i, &g) in got.iter().enumerate() {
            assert!(g == 0.0, "LEO rate[{i}] = {g} (expected exact zero)");
        }
    }

    /// GEO (a ~ 42164 km): 1:1 resonance, resonant orders {1,2,3,4}. Exercises the full resonant
    /// path (Hansen tesseral, GHmsj, GammaMns, Jacobi, Vmns) for j = 1..4.
    #[test]
    fn geo_resonant_matches_orekit() {
        let max_rel = check("geo", &GEO_STATE, &GEO_RATES);
        std::println!("GEO max tesseral rate rel error: {max_rel:.3e}");
    }

    /// GPS-like (a ~ 26560 km): 2:1 resonance, resonant orders {2,4} with j = {1,2} — a different
    /// resonant structure and j-selection than GEO.
    #[test]
    fn gps_resonant_matches_orekit() {
        let max_rel = check("gps", &GPS_STATE, &GPS_RATES);
        std::println!("GPS max tesseral rate rel error: {max_rel:.3e}");
    }

    /// GEO at e = 0.004, the same 1:1 resonance one eccentricity band lower. Both older fixtures
    /// sit at e = 0.01, where `maxEccPow = 4 = max_degree` makes `min(maxEccPow ± j, max_degree)`
    /// saturate at `max_degree` for every resonant `j`: the s-range is `[-4, 4]` however the two
    /// bounds are written. At e = 0.004 `maxEccPow` drops to 3 and the clamp stops biting — the
    /// j = 1 range is `s ∈ [-2, 4]`, asymmetric — so this is the only case in the suite where the
    /// truncation bounds themselves are observable.
    #[test]
    fn geo_low_ecc_resonant_matches_orekit() {
        let max_rel = check("geo_low_ecc", &GEO_LOW_ECC_STATE, &GEO_LOW_ECC_RATES);
        std::println!("GEO e=0.004 max tesseral rate rel error: {max_rel:.3e}");
    }

    /// `get_max_ecc_pow` against Orekit 13.1.7's own step table, read off the shipped
    /// `DSSTTesseral.getMaxEccPow` bytecode: thresholds 0.005/0.02/0.1/0.2/0.3/0.4 returning
    /// 3/4/7/10/12/15, 20 above. Every rate fixture sits at e ≤ 0.01, so only the first two steps
    /// were ever evaluated and the four upper thresholds could each move a whole band unnoticed.
    /// Both the threshold itself (the comparison is `<=`) and an interior point are checked.
    #[test]
    fn max_ecc_pow_matches_orekit_step_table() {
        #[rustfmt::skip]
        const TABLE: [(f64, usize); 13] = [
            (0.0,  3), (0.005, 3),
            (0.01, 4), (0.02,  4),
            (0.05, 7), (0.1,   7),
            (0.15, 10), (0.2,  10),
            (0.25, 12), (0.3,  12),
            (0.35, 15), (0.4,  15),
            (0.5,  20),
        ];
        for (e, want) in TABLE {
            assert_eq!(get_max_ecc_pow(e), want, "max_ecc_pow at e = {e}");
        }
    }

    /// The resonance acceptance window: Orekit keeps order `m` when `|m·θ̇/n − j| ≤ 1 / max(10,
    /// 864000 / period)`, i.e. one part in ten satellite revolutions once the period exceeds
    /// 24 000 s (`getResonantAndNonResonantTerms`, constants 10.0 and 864000.0 in the shipped
    /// bytecode). Both rate fixtures are commensurate to ~1e-8, hundreds of times deeper than the
    /// window, so the window's *width* was never observed — only that it accepts an exact
    /// resonance. Two GEO-like orbits placed either side of it are: at θ̇/n = 1.09 only m = 1 is
    /// inside (0.09 ≤ 0.1) while m = 2..4 are 0.18, 0.27, 0.36 outside; at 1.11 nothing is.
    #[test]
    fn resonance_window_is_one_part_in_ten_revolutions() {
        // n = √(µ/a³) ⇒ a = ∛(µ · (ratio/θ̇)²) for a prescribed ratio = θ̇/n.
        let state_for_ratio = |ratio: f64| {
            let x = ratio / RATE;
            let mut s = GEO_STATE;
            s[0] = libm::cbrt(MU * x * x);
            s
        };

        let inside = state_for_ratio(1.09);
        assert_eq!(
            tesseral().resonant_orders(&aux(&inside)),
            std::vec![(1, 1)],
            "0.09 of a cycle off the 1:1 commensurability is inside the 0.1 window for m = 1 only"
        );

        let outside = state_for_ratio(1.11);
        assert!(
            tesseral().resonant_orders(&aux(&outside)).is_empty(),
            "0.11 of a cycle off is outside the 0.1 window for every order"
        );
        let got = tesseral().mean_element_rates(&aux(&outside), GMST);
        for (i, &g) in got.iter().enumerate() {
            assert!(g == 0.0, "off-window rate[{i}] = {g} (expected exact zero)");
        }
    }

    /// `round(m·θ̇/n) = 0` is not a resonance. Orekit admits an order only for `j ≥ 1`, because the
    /// `j = 0` term is the m-daily one and is purely short-period; the mean rate it would
    /// contribute is spurious. Around Earth the `j ≥ 1` test never decides anything on its own —
    /// an order that rounds to `j = 0` is always rejected by the tolerance as well, since
    /// `m·θ̇/n ≥ θ̇/n = period / 86164 s` and the window is at most 0.1. A body rotating 7000×
    /// slower separates the two: `m·θ̇/n ≤ 5.5e-4` rounds to 0 and sits well inside the window.
    #[test]
    fn zero_frequency_order_is_not_resonant() {
        let slow = tesseral_with_rate(1.0e-8);
        assert!(
            slow.resonant_orders(&aux(&GEO_STATE)).is_empty(),
            "j = 0 is the m-daily term, not a resonance"
        );
        let got = slow.mean_element_rates(&aux(&GEO_STATE), GMST);
        for (i, &g) in got.iter().enumerate() {
            assert!(
                g == 0.0,
                "slow-rotator rate[{i}] = {g} (expected exact zero)"
            );
        }
    }
}
