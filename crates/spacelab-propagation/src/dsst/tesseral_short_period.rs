//! The DSST central-body **tesseral** short-period corrections (mean → osculating), the tesseral
//! analogue of the [`super::short_period`] zonal and [`super::third_body_short_period`] third-body
//! ports. This completes the osculating DSST: it is the last of the five short-period forces.
//!
//! Direct port of Orekit's `DSSTTesseral` short-period machinery (`updateShortPeriodTerms`, the inner
//! `FourierCjSjCoefficients` / `buildCoefficients` / `TesseralShortPeriodicCoefficients.value`),
//! mirroring its class split:
//! * `FourierCjSjCoefficients` — for each order `m` and frequency `j`, the six equinoctial Fourier
//!   coefficients Cᵢ^{jm}/Sᵢ^{jm} (the cos/sin parts of the equinoctial rate operator applied to the
//!   `{j, m}` harmonic's potential derivatives ∂R/∂x), built from the **shared** per-`{j, m, s}`
//!   `DsstTesseral::compute_n_sum` and `GHmsj` that the validated mean-rate path also uses;
//! * `buildCoefficients` — the resonance denominator: Cᵢ^{jm} = −Sᵢ_F^{jm}/(jn − mθ̇), Sᵢ^{jm} =
//!   Cᵢ_F^{jm}/(jn − mθ̇), plus the `3n/2a` δ₆ᵢ mean-longitude term;
//! * `value(meanOrbit)` — the Fourier sum Σ Cᵢ^{jm} cos(jλ − mθ) + Sᵢ^{jm} sin(jλ − mθ) over the
//!   m-daily (`j = 0`) and non-resonant `(j, m)` terms.
//!
//! # m-daily vs resonant/non-resonant short-period
//!
//! Unlike the tesseral **mean** rate — nonzero only at a resonance (`resOrders`) — the tesseral
//! short-period is present for **every** orbit. It splits into
//! * the **m-daily** terms (`j = 0`, orders `m = 1..=maxOrderMdailyTesseralSP`), whose phase is
//!   `−mθ` — always present; on a non-resonant LEO these are the whole tesseral short-period; and
//! * the **non-resonant** `(j, m)` terms (`m = 1..=maxOrderTesseralSP`, every `j` in
//!   `[−maxFrequency, maxFrequency]` except `0` and the resonant `jRes` of that order), phase
//!   `jλ − mθ`.
//!
//! The resonant frequency `jRes` of a resonant order is **excluded** from the short-period (its
//! `1/(jn − mθ̇)` denominator vanishes at resonance): that harmonic is carried by the mean rate
//! instead. Away from any resonance `jRes = 0` for every order and only `j = 0` is skipped.
//!
//! # Truncations (Orekit default constructor)
//!
//! The port builds the coefficients with Orekit's `new DSSTTesseral(bodyFrame, rate, provider)`
//! defaults, so for a degree/order-`d` field: `maxDegree{,Mdaily}TesseralSP = d`,
//! `maxOrder{,Mdaily}TesseralSP = d`, `maxEccPowTesseralSP = min(4, d)`,
//! `maxEccPowMdailyTesseralSP = min(4, d − 2)`, `maxFrequency = min(12, d + 4)`. The eccentricity
//! power bound differs between the m-daily (`maxEccPowMdailyTesseralSP`) and non-resonant
//! (`maxEccPowTesseralSP`) s-sums; the n-sum upper bound is the field degree (the default sets both
//! short-period degrees equal to it), reusing `DsstTesseral::compute_n_sum`'s own degree bound.
//!
//! # Central-body rotation angle
//!
//! As in the mean-rate port, `gmst` is the central-body rotation angle θ (Orekit's tesseral
//! `value(...)` `currentTheta`, the Greenwich sidereal angle for a body frame that is a pure +Z
//! rotation of the orbit frame). It enters only through the phases `−mθ` and `jλ − mθ`; the rotation
//! **rate** θ̇ (the resonance denominator) is the separate [`DsstTesseral::central_body_rotation_rate`].
//!
//! # What is *not* ported
//!
//! As with the other short-period ports, Orekit's `ShortPeriodicsInterpolatedCoefficient` (Hermite
//! interpolation between integration-step grid points) is a propagator detail: this module evaluates
//! the coefficients directly at the output mean state — exact at every grid point, and what the
//! interpolation approximates between them. The oracle compares against Orekit's OSCULATING
//! `DSSTPropagator` output seeded from the identical mean state.

use std::collections::HashMap;

use libm::{cos, sin};

use super::auxiliary::{DsstAuxiliary, I};
use super::tesseral::{DsstTesseral, GHmsj, get_max_ecc_pow};

/// The short-period truncation orders Orekit's `DSSTTesseral` default constructor derives from the
/// field degree/order. Orekit field names are kept for auditability.
struct SpTruncation {
    /// `maxOrderTesseralSP` — highest order of the non-resonant terms.
    max_order_tesseral_sp: usize,
    /// `maxEccPowTesseralSP` — eccentricity-power bound of the non-resonant s-sum.
    max_ecc_pow_tesseral_sp: usize,
    /// `maxOrderMdailyTesseralSP` — highest order of the m-daily terms.
    max_order_mdaily_sp: usize,
    /// `maxEccPowMdailyTesseralSP` — eccentricity-power bound of the m-daily s-sum.
    max_ecc_pow_mdaily_sp: usize,
    /// `maxFrequencyShortPeriodics` — absolute bound on the frequency index `j`.
    max_frequency: usize,
}

impl SpTruncation {
    fn from(tess: &DsstTesseral) -> Self {
        let d = tess.max_degree;
        let o = tess.max_order;
        Self {
            max_order_tesseral_sp: o,
            max_ecc_pow_tesseral_sp: 4.min(o),
            max_order_mdaily_sp: o,
            max_ecc_pow_mdaily_sp: 4.min(d - 2),
            max_frequency: tess.max_frequency,
        }
    }
}

/// The equinoctial short-period rate operator (Orekit `getMeanElementRate` / the `buildFourier
/// Coefficients` assembly), applied to one `{cos or sin}` set of µ/a-scaled potential derivatives
/// `dr = [∂R/∂a, ∂R/∂h, ∂R/∂k, ∂R/∂λ, ∂R/∂α, ∂R/∂β, ∂R/∂γ]`. Returns the six equinoctial coefficients
/// `[a, k, h, q, p, λ]`.
fn assemble_element_row(aux: &DsstAuxiliary, dr: [f64; 7]) -> [f64; 6] {
    let (alpha, beta, gamma) = (aux.alpha, aux.beta, aux.gamma);
    let (k, h, p, q) = (aux.k, aux.h, aux.p, aux.q);

    let r_alpha_gamma = alpha * dr[6] - gamma * dr[4];
    let r_alpha_beta = alpha * dr[5] - beta * dr[4];
    let r_beta_gamma = beta * dr[6] - gamma * dr[5];
    let rhk = h * dr[2] - k * dr[1];
    let p_rag = (p * r_alpha_gamma - I * q * r_beta_gamma) * aux.ooab;
    let rhkm = rhk - r_alpha_beta - dr[3];

    [
        aux.ax2oa * dr[3],
        -(aux.boa * dr[1] + h * p_rag + k * aux.boabpo * dr[3]),
        aux.boa * dr[2] + k * p_rag - h * aux.boabpo * dr[3],
        aux.co2ab * (q * rhkm - I * r_alpha_gamma),
        aux.co2ab * (p * rhkm - r_beta_gamma),
        -aux.ax2oa * dr[0] + aux.boabpo * (h * dr[1] + k * dr[2]) + p_rag,
    ]
}

/// The six equinoctial Fourier coefficients `(cCoef, sCoef)` for order `m` and frequency `j` (Orekit
/// `FourierCjSjCoefficients.buildFourierCoefficients`): the cos/sin parts of the equinoctial rate
/// operator on the `{j, m}` harmonic's potential derivatives, summed over `s ∈ [−s_bound, s_bound]`.
#[allow(clippy::too_many_arguments)]
fn fourier_coefficients(
    tess: &DsstTesseral,
    aux: &DsstAuxiliary,
    j: i64,
    m: i64,
    s_bound: usize,
    roa_pow: &[f64],
    gh: &GHmsj,
    max_hansen: i64,
    e2: f64,
    opig: f64,
    cache: &mut HashMap<(i64, i64, i64, i64), f64>,
) -> ([f64; 6], [f64; 6]) {
    // Per-derivative [cos, sin] accumulators for ∂R/∂{a, h, k, λ, α, β, γ} (Orekit's computeNSum
    // ordering: h before k), summed over s.
    let mut acc = [[0.0_f64; 2]; 7];
    for s in 0..=s_bound as i64 {
        let np = tess.compute_n_sum(j, m, s, opig, aux, roa_pow, gh, max_hansen, e2, cache);
        for r in 0..7 {
            acc[r][0] += np[r][0];
            acc[r][1] += np[r][1];
        }
        if s > 0 {
            let nn = tess.compute_n_sum(j, m, -s, opig, aux, roa_pow, gh, max_hansen, e2, cache);
            for r in 0..7 {
                acc[r][0] += nn[r][0];
                acc[r][1] += nn[r][1];
            }
        }
    }

    // Scale by µ/a (∂R/∂a carries the extra −1/a; Orekit folds the leading minus into the λ term).
    let muoa = aux.muoa;
    let scale = [-muoa / aux.sma, muoa, muoa, muoa, muoa, muoa, muoa];
    let mut dr_cos = [0.0_f64; 7];
    let mut dr_sin = [0.0_f64; 7];
    for r in 0..7 {
        dr_cos[r] = acc[r][0] * scale[r];
        dr_sin[r] = acc[r][1] * scale[r];
    }

    (
        assemble_element_row(aux, dr_cos),
        assemble_element_row(aux, dr_sin),
    )
}

/// Apply the resonance denominator `1/(jn − mθ̇)` (Orekit `buildCoefficients`): the short-period
/// Cᵢ^{jm}/Sᵢ^{jm} from the Fourier `(cCoef, sCoef)`, with the `3n/2a` mean-longitude (δ₆ᵢ) term.
fn build_coefficients(
    ccoef: &[f64; 6],
    scoef: &[f64; 6],
    j: i64,
    m: i64,
    tnota: f64,
    n: f64,
    theta_dot: f64,
) -> ([f64; 6], [f64; 6]) {
    let oojnmt = 1.0 / (j as f64 * n - m as f64 * theta_dot);
    let mut cijm = [0.0_f64; 6];
    let mut sijm = [0.0_f64; 6];
    for i in 0..6 {
        cijm[i] = -scoef[i];
        sijm[i] = ccoef[i];
    }
    // Separate part for the δ₆ᵢ (mean-longitude) element.
    cijm[5] += tnota * oojnmt * ccoef[0];
    sijm[5] += tnota * oojnmt * scoef[0];
    for i in 0..6 {
        cijm[i] *= oojnmt;
        sijm[i] *= oojnmt;
    }
    (cijm, sijm)
}

/// The six equinoctial tesseral short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` at the given mean
/// state (Orekit `TesseralShortPeriodicCoefficients.value`), `gmst` the central-body rotation angle θ.
fn short_period_correction(tess: &DsstTesseral, aux: &DsstAuxiliary, gmst: f64) -> [f64; 6] {
    let trunc = SpTruncation::from(tess);

    let max_ecc_pow = get_max_ecc_pow(aux.ecc);
    let max_hansen = (max_ecc_pow / 2) as i64;
    let e2 = aux.ecc * aux.ecc;
    let opig = 1.0 + I * aux.gamma;

    // R / a up to the field degree.
    let roa = tess.re / aux.sma;
    let mut roa_pow = std::vec![0.0; tess.max_degree + 1];
    roa_pow[0] = 1.0;
    for i in 1..=tess.max_degree {
        roa_pow[i] = roa * roa_pow[i - 1];
    }

    let gh = GHmsj::new(aux.k, aux.h, aux.alpha, aux.beta, tess.cjsj_max());
    let mut cache: HashMap<(i64, i64, i64, i64), f64> = HashMap::new();

    // 3n / 2a and the rotation rate θ̇ (the resonance denominator, distinct from the phase angle θ).
    let tnota = 1.5 * aux.n / aux.sma;
    let theta_dot = tess.central_body_rotation_rate;

    let mut sp = [0.0_f64; 6];

    // m-daily terms (j = 0), phase −mθ.
    for m in 1..=trunc.max_order_mdaily_sp as i64 {
        let (ccoef, scoef) = fourier_coefficients(
            tess,
            aux,
            0,
            m,
            trunc.max_ecc_pow_mdaily_sp,
            &roa_pow,
            &gh,
            max_hansen,
            e2,
            opig,
            &mut cache,
        );
        let (cijm, sijm) = build_coefficients(&ccoef, &scoef, 0, m, tnota, aux.n, theta_dot);
        let phase = -(m as f64) * gmst;
        let (s, c) = (sin(phase), cos(phase));
        for i in 0..6 {
            sp[i] += cijm[i] * c + sijm[i] * s;
        }
    }

    // Non-resonant (j, m) terms, phase jλ − mθ. The resonant jRes of a resonant order is excluded
    // (its denominator vanishes at resonance); away from resonance jRes = 0 and only j = 0 is skipped.
    let res = tess.resonant_orders(aux);
    let max_f = trunc.max_frequency as i64;
    for m in 1..=trunc.max_order_tesseral_sp as i64 {
        let j_res = res
            .iter()
            .find(|(rm, _)| *rm as i64 == m)
            .map_or(0, |(_, jr)| *jr);
        for j in -max_f..=max_f {
            if j == 0 || j == j_res {
                continue;
            }
            let (ccoef, scoef) = fourier_coefficients(
                tess,
                aux,
                j,
                m,
                trunc.max_ecc_pow_tesseral_sp,
                &roa_pow,
                &gh,
                max_hansen,
                e2,
                opig,
                &mut cache,
            );
            let (cijm, sijm) = build_coefficients(&ccoef, &scoef, j, m, tnota, aux.n, theta_dot);
            let phase = j as f64 * aux.lambda - m as f64 * gmst;
            let (s, c) = (sin(phase), cos(phase));
            for i in 0..6 {
                sp[i] += cijm[i] * c + sijm[i] * s;
            }
        }
    }

    sp
}

impl DsstTesseral {
    /// The six equinoctial tesseral short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` at the given
    /// **mean** state, with `gmst` the central-body rotation angle θ at the epoch (Orekit's
    /// `TesseralShortPeriodicCoefficients.value`). Includes the m-daily and non-resonant terms.
    #[must_use]
    pub fn short_period_correction(&self, aux: &DsstAuxiliary, gmst: f64) -> [f64; 6] {
        short_period_correction(self, aux, gmst)
    }

    /// The osculating equinoctial state `mean + short-period` for a **mean** equinoctial state
    /// `[a, k, h, q, p, λ]`, with `gmst` the central-body rotation angle θ at the epoch. Mirrors
    /// [`super::third_body::DsstThirdBody::osculating`].
    #[must_use]
    pub fn osculating(&self, mean: [f64; 6], gmst: f64) -> [f64; 6] {
        let aux = DsstAuxiliary::new(
            mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], self.mu,
        );
        let sp = self.short_period_correction(&aux, gmst);
        let mut osc = mean;
        for i in 0..6 {
            osc[i] += sp[i];
        }
        osc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden fixtures from Orekit 13 (tools/orekit-oracle, orekit_dsst_tesseral_sp.csv): the same
    // unnormalized 4x4 field, WGS84 rotation rate and idealized non-precessing body frame (a fixed +Z
    // rotation of GCRF by gmst = 1.234) as the tesseral mean case. From a fixed MEAN equinoctial LEO
    // (a = 7.2e6, e = 0.01, i = 98 deg) a MEAN DSSTPropagator carrying only DSSTTesseral emits, at each
    // t > 0 sample, the mean equinoctial state; Orekit's tesseral short-period delta is evaluated
    // DIRECTLY at that mean state via ShortPeriodTerms.value(meanOrbit) (no second propagator, no
    // Hermite interpolation of the coefficients). The port's delta at the same mean state with the
    // emitted GMST is validated against it. On this LEO there is no tesseral resonance, so the mean
    // tesseral drift is identically zero (a, k, h, q, p are frozen across the four samples; only the
    // mean longitude advances) and the delta is the whole tesseral short-period (m-daily + non-resonant).
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const RATE: f64 = 7.29211500000000000e-05;
    const GMST: f64 = 1.23400000000000020e+00;

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

    // (t, MEAN[6]) samples at t = 21600, 43200, 64800, 86400 s; a/k/h/q/p frozen, λ advancing.
    #[rustfmt::skip]
    const MEAN: [[f64; 6]; 4] = [
        [7.20000000000000000e+06, 3.09016994374947470e-03, 9.51056516295153600e-03, 8.54890329218965000e-01, 7.69746709859868000e-01, 2.37526966341341900e+01],
        [7.20000000000000000e+06, 3.09016994374947470e-03, 9.51056516295153600e-03, 8.54890329218965000e-01, 7.69746709859868000e-01, 4.60742232816330240e+01],
        [7.20000000000000000e+06, 3.09016994374947470e-03, 9.51056516295153600e-03, 8.54890329218965000e-01, 7.69746709859868000e-01, 6.83957499291318700e+01],
        [7.20000000000000000e+06, 3.09016994374947470e-03, 9.51056516295153600e-03, 8.54890329218965000e-01, 7.69746709859868000e-01, 9.07172765766307000e+01],
    ];

    // Orekit's direct value() tesseral short-period delta [Δa, Δk, Δh, Δq, Δp, Δλ] at each MEAN state.
    #[rustfmt::skip]
    const DELTA: [[f64; 6]; 4] = [
        [-6.02195587428524000e+01, -1.49103948657550400e-05, 3.20171812843484160e-06, 2.97627495529018060e-05, 8.27924411007973300e-06, 3.93463761120039800e-05],
        [-9.45144919841273700e+00, -1.60704276137262550e-05, 2.67814885743164960e-06, 2.36106977103193450e-05, 4.53240450962886100e-06, 3.26372150154445500e-05],
        [ 5.10318576443830000e+00, -1.24254497576050420e-05, -4.23988227789483700e-06, 2.86313110674540060e-05, 1.37480204126678140e-05, 3.87189350691100600e-05],
        [-3.11655774085851820e+01, -1.43617994589285700e-05, 1.91903368826202370e-06, 2.45749620495773330e-05, 4.53537898048493840e-06, 2.86068667255713600e-05],
    ];

    fn tesseral() -> DsstTesseral {
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
        DsstTesseral::new(MU, AE, RATE, cnm, snm)
    }

    /// The unnormalized 4x4 field takes Orekit's default short-period truncations: order 4, non-resonant
    /// eccentricity power 4, m-daily eccentricity power 2, frequency bound 8.
    #[test]
    fn truncation_matches_orekit() {
        let trunc = SpTruncation::from(&tesseral());
        assert_eq!(trunc.max_order_tesseral_sp, 4);
        assert_eq!(trunc.max_ecc_pow_tesseral_sp, 4);
        assert_eq!(trunc.max_order_mdaily_sp, 4);
        assert_eq!(trunc.max_ecc_pow_mdaily_sp, 2);
        assert_eq!(trunc.max_frequency, 8);
    }

    /// Orekit's m-daily eccentricity bound is `min(4, degree − 2)`, but the 4×4 fixture sits on the one
    /// degree where `degree − 2` and `degree / 2` agree, so it pins the rule at a single point. Walk the
    /// degrees the truncation is defined over instead.
    #[test]
    fn the_mdaily_eccentricity_bound_follows_the_degree() {
        for (deg, want) in [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (8, 4)] {
            let empty = std::vec![std::vec![0.0; deg + 1]; deg + 1];
            let field = DsstTesseral::new(MU, AE, RATE, empty.clone(), empty);
            let trunc = SpTruncation::from(&field);
            assert_eq!(
                trunc.max_ecc_pow_mdaily_sp, want,
                "degree {deg}: m-daily eccentricity bound"
            );
            assert_eq!(trunc.max_ecc_pow_tesseral_sp, 4.min(deg), "degree {deg}");
        }
    }

    /// At a resonant order the short-period denominator `j·n − m·θ̇` vanishes: that term belongs to the
    /// mean rates and has to be dropped here. Nothing asserted it — the Orekit fixture is a non-resonant
    /// LEO, where `resonant_orders` is empty, `j_res` is `0`, and the exclusion collapses onto the `j = 0`
    /// skip that the m-daily loop already owns. A GEO placed on the 1:1 resonance separates them: with
    /// `n = θ̇` the reciprocal `1/(j·n − m·θ̇)` runs to ~1e19 for `j = m`, so keeping any resonant term
    /// moves the correction by ~1e15 over its true size.
    #[test]
    fn the_resonant_term_is_excluded_from_the_short_period_sum() {
        use libm::{cbrt, fabs};

        let tess = tesseral();
        let a_res = cbrt(MU / (RATE * RATE));
        let aux = DsstAuxiliary::new(a_res, 1.0e-4, 2.0e-4, 1.0e-3, -2.0e-3, 0.7, MU);
        assert_eq!(
            tess.resonant_orders(&aux),
            std::vec![(1, 1), (2, 2), (3, 3), (4, 4)],
            "the sweep must actually be resonant"
        );

        let sp = tess.short_period_correction(&aux, GMST);
        assert!(
            fabs(sp[0]) < 1.0e3,
            "resonant Δa {:.3e} m — the singular term is in the sum",
            sp[0]
        );
        for (i, &c) in sp.iter().enumerate().skip(1) {
            assert!(fabs(c) < 1.0e-3, "resonant sp[{i}] = {c:.3e}");
        }
        std::println!("GEO-resonant tesseral short period: {sp:?}");
    }

    /// `osculating` is the public entry point and no test called it: the sign with which the correction
    /// is applied was free. Against Orekit's own mean/δ pair it is not.
    #[test]
    fn osculating_adds_the_short_period_correction_to_the_mean() {
        let tess = tesseral();
        for k in 0..4 {
            let osc = tess.osculating(MEAN[k], GMST);
            for i in 0..6 {
                let want = MEAN[k][i] + DELTA[k][i];
                let miss = (osc[i] - want).abs() / f64::EPSILON.mul_add(want.abs(), 1e-15);
                assert!(miss < 8.0, "k={k} osc[{i}]: {miss:.2} ulps off mean + δ");
            }
        }
    }

    /// A non-resonant LEO has no resonant orders, so every order's non-resonant sum spans all `j` in
    /// `[−8, 8] \ {0}` (nothing extra excluded) and the tesseral mean rate is zero.
    #[test]
    fn leo_has_no_resonance() {
        let aux = DsstAuxiliary::new(
            MEAN[0][0], MEAN[0][1], MEAN[0][2], MEAN[0][3], MEAN[0][4], MEAN[0][5], MU,
        );
        assert_eq!(tesseral().resonant_orders(&aux).len(), 0);
    }

    /// The tesseral short-period δ [Δa, Δk, Δh, Δq, Δp, Δλ] matches Orekit's direct `value(meanOrbit)`
    /// at every sample to round-off: `a` to a tight absolute bound and the equinoctial e/i components +
    /// mean longitude to `comp_tol`. Orekit's own mean state and a field frozen at the epoch (so its
    /// harmonics do not drift over the propagation) isolate the short-period assembly (m-daily +
    /// non-resonant); the port reproduces it to ~1e-15 relative.
    #[test]
    fn tesseral_short_period_matches_orekit() {
        let tess = tesseral();
        let (mut max_da, mut max_comp) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            let aux = DsstAuxiliary::new(
                MEAN[k][0], MEAN[k][1], MEAN[k][2], MEAN[k][3], MEAN[k][4], MEAN[k][5], MU,
            );
            let sp = tess.short_period_correction(&aux, GMST);
            let da = (sp[0] - DELTA[k][0]).abs();
            assert!(da < 1e-9, "k={k}: a abs {da:.3e} m");
            max_da = max_da.max(da);
            for i in 1..6 {
                let d = (sp[i] - DELTA[k][i]).abs();
                max_comp = max_comp.max(d);
                assert!(d < 1e-15, "k={k} comp[{i}]: {d:.3e}");
            }
        }
        std::println!(
            "Tesseral short-period vs Orekit: max |Δa| {max_da:.2e} m, max comp {max_comp:.2e}"
        );
    }
}
