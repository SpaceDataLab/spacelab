//! DSST scalar coefficients and the eccentricity-power truncation.
//!
//! Ported from Orekit's `CoefficientsFactory` (Vₙₛ, Qₙₛ, GₛHₛ), `UpperBounds` (Dₙˡ, Rₙₘˡ) and the
//! `DSSTZonal.computeMeanElementsTruncations` logic. All names follow Danielson / Orekit.
//! `derivations/dsst_coefficients.mac` proves each recurrence here its closed form and writes the
//! values the tests hold it to.

use core::cmp::Ordering;
use std::vec::Vec;

use super::auxiliary::DsstAuxiliary;

/// Truncation tolerance for the mean-element eccentricity-power series (`DSSTZonal`).
const TRUNCATION_TOLERANCE: f64 = 1e-4;

/// `n!` as an `f64` (Orekit `CombinatoricsUtils.factorialDouble`). `n ≥ 0`.
pub(crate) fn factorial(n: i64) -> f64 {
    let mut acc = 1.0;
    let mut i = 2;
    while i <= n {
        acc *= i as f64;
        i += 1;
    }
    acc
}

/// Compute the Vₙₛ coefficients (Danielson 2.8.2-(1)(2)) up to `order`, as a dense `vns[n][s]` table
/// for `n, s < order`. Mirrors `CoefficientsFactory.computeVns` (fresh, uncached).
#[must_use]
pub fn compute_vns(order: usize) -> Vec<Vec<f64>> {
    let mut vns = std::vec![std::vec![0.0; order]; order];
    vns[0][0] = 1.0;
    // The reference seeds `vns[1][0] = 0` and `vns[1][1] = 0.5` here. Both are redundant: the very
    // first iteration below, `n = s = 0`, writes `vns[1][1] = vns[0][0] / 2`, and `vns[1][0]` is
    // reached by the parity branch, which stores the `0.0` the vector already holds.
    for n in 0..order {
        for s in 0..=n {
            if (n - s) % 2 != 0 {
                vns[n][s] = 0.0;
            } else {
                if n == s && (s + 1) < order {
                    vns[s + 1][s + 1] = vns[s][s] / (2 * s + 2) as f64;
                }
                if (n + 2) < order {
                    vns[n + 2][s] =
                        vns[n][s] * (-(n as f64) + s as f64 - 1.0) / (n as f64 + s as f64 + 2.0);
                }
            }
        }
    }
    vns
}

/// Compute the Qₙₛ coefficients evaluated at γ (Danielson 2.8.3-(2) recurrence), for `n = 0..=n_max`
/// and `s = 0..=s_max+1` (the extra column gives dQₙₛ/dγ = Q(n, s+1)). Ragged, exactly as Orekit's
/// `CoefficientsFactory.computeQns`.
#[must_use]
pub fn compute_qns(gamma: f64, n_max: usize, s_max: usize) -> Vec<Vec<f64>> {
    let s_dim = (s_max + 1).min(n_max) + 1;
    let mut qns: Vec<Vec<f64>> = Vec::with_capacity(n_max + 1);
    for i in 0..=n_max {
        let sn_dim = (i + 1).min(s_dim);
        qns.push(std::vec![0.0; sn_dim]);
    }
    qns[0][0] = 1.0;
    for n in 1..=n_max {
        let sn_dim = (n + 1).min(s_dim);
        for s in 0..sn_dim {
            // n ≥ s throughout, so match on the (non-negative) gap n − s [Danielson 2.8.3-(2)].
            qns[n][s] = match n - s {
                0 => (2.0 * s as f64 - 1.0) * qns[s - 1][s - 1],
                1 => (2.0 * s as f64 + 1.0) * gamma * qns[s][s],
                _ => {
                    ((2.0 * n as f64 - 1.0) * gamma * qns[n - 1][s]
                        - (n as f64 + s as f64 - 1.0) * qns[n - 2][s])
                        / (n - s) as f64
                }
            };
        }
    }
    qns
}

/// Compute the Gₛ and Hₛ polynomials (Danielson 3.1-(5) recurrence) for `s = 0..=order`. Returns
/// `(gs, hs)`. Mirrors `CoefficientsFactory.computeGsHs`.
#[must_use]
pub fn compute_gs_hs(k: f64, h: f64, alpha: f64, beta: f64, order: usize) -> (Vec<f64>, Vec<f64>) {
    let hamkb = h * alpha - k * beta;
    let kaphb = k * alpha + h * beta;
    let mut gs = std::vec![0.0; order + 1];
    let mut hs = std::vec![0.0; order + 1];
    gs[0] = 1.0;
    hs[0] = 0.0;
    for s in 1..=order {
        gs[s] = kaphb * gs[s - 1] - hamkb * hs[s - 1];
        hs[s] = hamkb * gs[s - 1] + kaphb * hs[s - 1];
    }
    (gs, hs)
}

/// The value and γ-derivative of the Jacobi polynomial `Pₗ^{v,w}(x)` (Hipparchus
/// `PolynomialsUtils.createJacobiPolynomial` + `JacobiPolynomials.getValueAndDerivative`). Shared by
/// the tesseral and third-body short-period contributions.
pub(crate) fn jacobi_value_deriv(l: usize, v: i64, w: i64, x: f64) -> (f64, f64) {
    // Build P₀..Pₗ by the Hipparchus recurrence P_{k+1} = (a₀ + a₁·X)·P_k − a₂·P_{k−1}.
    let mut polys: Vec<Vec<f64>> = Vec::with_capacity(l + 1);
    polys.push(std::vec![1.0]);
    if l >= 1 {
        polys.push(std::vec![(v - w) as f64 / 2.0, (2 + v + w) as f64 / 2.0]);
    }
    for k in 1..l {
        let kk = k as i64 + 1;
        let kvw = kk + v + w;
        let two_kvw = kvw + kk;
        let two_kvw_m1 = two_kvw - 1;
        let two_kvw_m2 = two_kvw - 2;
        let den = (2 * kk * kvw * two_kvw_m2) as f64;
        let a0 = (two_kvw_m1 * (v * v - w * w)) as f64 / den;
        let a1 = (two_kvw_m1 * two_kvw * two_kvw_m2) as f64 / den;
        let a2 = (2 * (kk + v - 1) * (kk + w - 1) * two_kvw) as f64 / den;

        let mut next = std::vec![0.0; k + 2];
        {
            let pk = &polys[k];
            for (i, &c) in pk.iter().enumerate() {
                next[i] += a0 * c;
                next[i + 1] += a1 * c;
            }
        }
        {
            let pkm1 = &polys[k - 1];
            for (i, &c) in pkm1.iter().enumerate() {
                next[i] -= a2 * c;
            }
        }
        polys.push(next);
    }

    // Horner evaluation of value and first derivative (Orekit getValueAndDerivative).
    let a = &polys[l];
    let degree = l;
    let mut value = a[degree];
    let mut derivative = value * degree as f64;
    let mut jj = degree as i64 - 1;
    while jj >= 1 {
        let coef = a[jj as usize];
        value = value * x + coef;
        derivative = derivative * x + coef * jj as f64;
        jj -= 1;
    }
    if degree > 0 {
        value = value * x + a[0];
    }
    (value, derivative)
}

/// Upper bound Dₙˡ(χ) (`UpperBounds.getDnl`). `xx = χ²`, `xpl = χ · (χ²/2)ˡ`.
pub(crate) fn get_dnl(xx: f64, xpl: f64, n: i64, l: i64) -> f64 {
    let lp2 = l + 2;
    match n.cmp(&lp2) {
        Ordering::Greater => {
            let ll = l * l;
            // Sliding three-term window (Orekit dL/dM/dB); dL is captured per iteration.
            let mut d_m = xpl;
            let mut d_b = (l + 1) as f64 * xx * xpl;
            for j in (l + 3)..=n {
                let jm1 = j - 1;
                let d_l = d_m;
                d_m = d_b;
                d_b = jm1 as f64 * xx * ((2 * j - 3) as f64 * d_m - (j - 2) as f64 * d_l)
                    / (jm1 * jm1 - ll) as f64;
            }
            d_b
        }
        Ordering::Equal => (l + 1) as f64 * xx * xpl,
        Ordering::Less => xpl,
    }
}

/// Upper bound Rₙₘˡ^ε(γ) (`UpperBounds.getRnml`). `eps` is ε (±1), `irf` the retrograde factor I.
fn get_rnml(gamma: f64, n: i64, l: i64, m: i64, eps: i64, irf: i64) -> f64 {
    let mei = m * eps * irf;
    let sinisq = 1.0 - gamma * gamma;
    let sininc = 0.03_f64.max(libm::sqrt(sinisq));
    let onepig = 1.0 + gamma * irf as f64;
    let sininc_pow = libm::pow(sininc, (l - mei) as f64);
    let onepig_pow = libm::pow(onepig, mei as f64);
    let mut r_bound = sininc_pow * onepig_pow;

    if n > l {
        let lp1 = l + 1;
        let mut dpnml = (lp1 * eps) as f64;
        let mut pnml = dpnml * gamma - m as f64;
        if n > l + 1 {
            let ll = l * l;
            let ml = m * l;
            let mm = m * m;
            // Sliding window (Orekit pn1ml/pn2ml); the n-2 terms are captured per iteration.
            let mut pn1ml = 1.0;
            let mut dpn1ml = 0.0;
            for in_ in (l + 2)..=n {
                let nm1 = in_ - 1;
                let tnm1 = in_ + nm1;
                let nnlnl = nm1 * (in_ * in_ - ll);
                let nnmnm = in_ * (nm1 * nm1 - mm);
                let c2nne = tnm1 * in_ * nm1 * eps;
                let c2nml = tnm1 * ml;
                let coef = c2nne as f64 * gamma - c2nml as f64;
                let pn2ml = pn1ml;
                let dpn2ml = dpn1ml;
                pn1ml = pnml;
                dpn1ml = dpnml;
                pnml = (coef * pn1ml - nnmnm as f64 * pn2ml) / nnlnl as f64;
                dpnml =
                    (coef * dpn1ml - nnmnm as f64 * dpn2ml + c2nne as f64 * pn1ml) / nnlnl as f64;
            }
        }
        r_bound *= libm::sqrt(pnml * pnml + dpnml * dpnml * sinisq / ((n - l) * (n + lp1)) as f64);
    }
    r_bound
}

/// The highest power of the eccentricity retained in the mean-element rate expansion
/// (`maxEccPowMeanElements`), from `DSSTZonal.computeMeanElementsTruncations`.
///
/// This is computed once at the initial state (as Orekit does inside `initializeShortPeriodTerms`)
/// and reused for every rate evaluation of a propagation. Zonal field ⇒ order 0, so `Sₙₘ = 0` and
/// the coefficient magnitude reduces to `|Cₙ₀|`.
#[must_use]
pub fn compute_max_ecc_pow(
    max_degree: usize,
    max_order: usize,
    c_n0: &[f64],
    ae: f64,
    mu: f64,
    aux: &DsstAuxiliary,
) -> usize {
    if max_degree == 2 {
        return 0;
    }

    let max_deg_top = max_degree as i64;
    let ax2or = 2.0 * aux.sma / ae;
    let mut xmuran = mu / aux.sma;
    let eo2 = 0.0025_f64.max(aux.ecc / 2.0);
    let x2o2 = aux.chi2 / 2.0;

    let mut ecc_pwr = std::vec![0.0; max_degree + 1];
    let mut chi_pwr = std::vec![0.0; max_degree + 1];
    let mut haf_pwr = std::vec![0.0; max_degree + 1];
    ecc_pwr[0] = 1.0;
    chi_pwr[0] = aux.chi;
    haf_pwr[0] = 1.0;
    for i in 1..=max_degree {
        ecc_pwr[i] = ecc_pwr[i - 1] * eo2;
        chi_pwr[i] = chi_pwr[i - 1] * x2o2;
        haf_pwr[i] = haf_pwr[i - 1] * 0.5;
        xmuran /= ax2or;
    }

    let gamma = aux.gamma;
    let chi2 = aux.chi2;
    let irf: i64 = 1; // retrograde factor I (always +1, see auxiliary::I)

    let mut max_ecc: i64 = 0;
    let mut max_deg = max_deg_top;
    loop {
        let mut m: i64 = 0;
        loop {
            // Zonal: Cₙₘ nonzero only at m = 0, Sₙₘ = 0.
            let cnm = if m == 0 { c_n0[max_deg as usize] } else { 0.0 };
            let csnm = libm::fabs(cnm);
            if csnm == 0.0 {
                break;
            }
            let mut last_term = 0.0;
            let mut nsld2 = (max_deg - max_ecc - 1) / 2;
            let mut l = max_deg - 2 * nsld2;
            let mut term;
            loop {
                if m < l {
                    term = csnm
                        * xmuran
                        * (factorial(max_deg - l) / factorial(max_deg - m))
                        * (factorial(max_deg + l) / (factorial(nsld2) * factorial(nsld2 + l)))
                        * ecc_pwr[l as usize]
                        * get_dnl(chi2, chi_pwr[l as usize], max_deg, l)
                        * (get_rnml(gamma, max_deg, l, m, 1, irf)
                            + get_rnml(gamma, max_deg, l, m, -1, irf));
                } else {
                    term = csnm
                        * xmuran
                        * (factorial(max_deg + m) / (factorial(nsld2) * factorial(nsld2 + l)))
                        * ecc_pwr[l as usize]
                        * haf_pwr[(m - l) as usize]
                        * get_dnl(chi2, chi_pwr[l as usize], max_deg, l)
                        * (get_rnml(gamma, max_deg, m, l, 1, irf)
                            + get_rnml(gamma, max_deg, m, l, -1, irf));
                }
                if term >= TRUNCATION_TOLERANCE {
                    max_ecc = l;
                } else if term < last_term {
                    break;
                }
                last_term = term;
                l += 2;
                nsld2 -= 1;
                if l >= max_deg {
                    break;
                }
            }
            if term >= TRUNCATION_TOLERANCE {
                return max_ecc.min(max_deg_top - 2) as usize;
            }
            m += 1;
            if m > max_deg.min(max_order as i64) {
                break;
            }
        }
        xmuran *= ax2or;
        max_deg -= 1;
        if max_deg <= max_ecc + 2 {
            break;
        }
    }

    max_ecc.min(max_deg_top - 2) as usize
}

#[cfg(test)]
// `unwrap`/`expect` on a malformed fixture is the right failure: the file is checked in beside the
// test and a parse error in it is a broken checkout, not a runtime condition to recover from. The
// exact float comparisons are deliberate: a closed form's exact zeros and its poles are values, not
// tolerances.
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use spacelab_testkit::nan_max;
    use std::string::String;
    use std::vec::Vec;

    use super::{compute_gs_hs, compute_qns, compute_vns, get_dnl, get_rnml, jacobi_value_deriv};

    /// The fixtures' `(k, h)` and `(alpha, beta)` pairs, which their `gs`/`hs` rows index.
    const KH: [(f64, f64); 4] = [(0.0, 0.0), (0.05, -0.02), (-0.3, 0.4), (0.6, 0.6)];
    const AB: [(f64, f64); 4] = [(1.0, 0.0), (0.6, -0.8), (-0.28, 0.96), (0.0, 1.0)];
    /// `(gs, hs)` as `compute_gs_hs` returns them, and the ragged `qns[n][s]` table.
    type GsHs = (Vec<f64>, Vec<f64>);
    type QnsTable = Vec<Vec<f64>>;

    const VNS_ORDER: usize = 14;
    const QNS_N: usize = 12;
    const GS_ORDER: usize = 12;

    /// Per-kind distance from the closed forms: `(kind, relative bound, absolute bound at an exact
    /// zero)`, each 3–5× its measured worst.
    ///
    /// Separate bounds, because one number would hide what the measurement says. `Vₙₛ` lands within
    /// `1.9e-16` of its exact value, `Dₙˡ` within `2.3e-15`, `Gₛ` `5.5e-15` and `Qₙₛ` `6.5e-15`.
    /// `Hₛ` at `6.5e-14` and the Jacobi polynomials at `1.1e-13` (value) and `5.3e-14`
    /// (derivative) are cancellation: `Hₛ` of `(0.6 − 0.6i)ˢ` and `P₆⁽⁴,⁰⁾(−0.8)` sum terms a hundred
    /// times their result. `Rₙₘˡ` is the outlier at `4.3e-9`, and only where `m > l` and
    /// `εγ = 0.9995`: there `P⁽ˡ⁻ᵐ,ˡ⁺ᵐ⁾ₙ₋ₗ` has a zero of order `m − l` at `εγ = 1`, which the
    /// three-term recurrence reaches by cancellation, so its digits go as `(1 − εγ)^(m−l)`. No
    /// caller reaches it — the zonal truncation passes `m = 0` in both argument orders — and at
    /// every other row the recurrence holds `2.7e-13`.
    const BOUNDS: [(&str, f64, f64); 8] = [
        ("dnl", 1.0e-14, 0.0),
        ("rnml", 1.5e-8, 0.0),
        ("vns", 8.0e-16, 0.0),
        ("qns", 2.5e-14, 0.0),
        ("gs", 2.0e-14, 0.0),
        ("hs", 2.5e-13, 0.0),
        ("jacobi_value", 4.0e-13, 0.0),
        ("jacobi_deriv", 2.0e-13, 8.0e-16),
    ];

    /// What each kind is, for the comparison rows.
    const QUANTITY: [(&str, &str); 8] = [
        ("dnl", "DSST upper bound Dₙˡ(χ)"),
        ("rnml", "DSST upper bound Rₙₘˡ(γ)"),
        ("vns", "DSST coefficient Vₙₛ"),
        ("qns", "DSST coefficient Qₙₛ(γ)"),
        ("gs", "DSST coefficient Gₛ"),
        ("hs", "DSST coefficient Hₛ"),
        ("jacobi_value", "DSST Jacobi polynomial Pₗ⁽ᵛ,ʷ⁾(γ)"),
        ("jacobi_deriv", "DSST Jacobi polynomial derivative"),
    ];

    #[derive(Default)]
    struct Worst {
        rel: f64,
        rel_row: String,
        abs: f64,
        abs_row: String,
    }

    /// One fixture row, `kind,a,b,c,d,e,value`; unused columns are 0.
    struct Row<'a> {
        line: &'a str,
        kind: &'a str,
        a: f64,
        b: i64,
        c: i64,
        d: i64,
        e: i64,
        want: f64,
    }

    fn fixture(name: &str) -> String {
        let path = std::format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(path).expect("fixture")
    }

    fn rows(text: &str) -> Vec<Row<'_>> {
        text.lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let f: Vec<&str> = line.split(',').collect();
                assert_eq!(f.len(), 7, "malformed row: {line}");
                Row {
                    line,
                    kind: f[0],
                    a: f[1].parse().unwrap(),
                    b: f[2].parse().unwrap(),
                    c: f[3].parse().unwrap(),
                    d: f[4].parse().unwrap(),
                    e: f[5].parse().unwrap(),
                    want: f[6].parse().unwrap(),
                }
            })
            .collect()
    }

    /// This module's value for a row, with the tables a kind needs built once.
    struct Module {
        vns: Vec<Vec<f64>>,
        qns: Vec<(f64, QnsTable)>,
        gs_hs: Vec<((usize, usize), GsHs)>,
    }

    impl Module {
        fn new() -> Self {
            Self {
                vns: compute_vns(VNS_ORDER),
                qns: Vec::new(),
                gs_hs: Vec::new(),
            }
        }

        fn value(&mut self, row: &Row) -> f64 {
            let (a, b, c, d, e) = (row.a, row.b, row.c, row.d, row.e);
            match row.kind {
                // `a` is chi, and the caller always forms the two arguments this way.
                "dnl" => get_dnl(a * a, a * libm::pow(a * a / 2.0, c as f64), b, c),
                "rnml" => get_rnml(a, b, c, d, e, 1),
                "vns" => self.vns[b as usize][c as usize],
                "qns" => qns_for(&mut self.qns, a)[b as usize][c as usize],
                "gs" => gs_hs_for(&mut self.gs_hs, (c as usize, d as usize)).0[b as usize],
                "hs" => gs_hs_for(&mut self.gs_hs, (c as usize, d as usize)).1[b as usize],
                "jacobi_value" => jacobi_value_deriv(b as usize, c, d, a).0,
                "jacobi_deriv" => jacobi_value_deriv(b as usize, c, d, a).1,
                other => panic!("unknown kind {other}"),
            }
        }
    }

    /// Every scalar coefficient and both Danielson bounds, against their closed forms at the
    /// inputs' doubles. `derivations/dsst_coefficients.mac` proves in exact arithmetic that each
    /// recurrence here is its closed form at every index the fixture holds (G1): `Vₙₛ` Danielson's
    /// double-factorial quotient, `Qₙₛ = dˢPₙ/dγˢ`, `Gₛ + iHₛ` the complex power
    /// `(k + ih)ˢ(α − iβ)ˢ`, the Jacobi polynomial its explicit sum, `Dₙˡ(χ) =
    /// χⁿ⁺ˡ (n−1)!/(n+l−1)! · dˡPₙ₋₁/dχˡ` (at `l = 0` the mean of `(a/r)ⁿ⁺¹`, integrated there), and
    /// `Rₙₘˡ` built on `P⁽ˡ⁻ᵐ,ˡ⁺ᵐ⁾ₙ₋ₗ(εγ)`. It then writes the closed forms' values, exact or at 60
    /// digits, to 25 (G3).
    ///
    /// These six functions decide *where the eccentricity series is truncated*, and until this test
    /// nothing in the suite looked at them: the DSST tests compare a propagated orbit, and by then
    /// the series has converged well enough that a wrong bound moves the truncation without moving
    /// the answer past the end-to-end tolerance. That is not harmless — a bound that comes out too
    /// small truncates too early and costs accuracy silently — and it is exactly the shape a
    /// mutation survives: `cargo mutants` left `34%` of this file alive against `2.2%` for
    /// [`super::super::zonal`], which is checked term by term.
    #[test]
    fn the_scalar_coefficients_and_upper_bounds_are_their_closed_forms() {
        let text = fixture("dsst_coefficients.csv");
        let mut module = Module::new();
        let mut worst: Vec<(&str, Worst)> =
            BOUNDS.iter().map(|b| (b.0, Worst::default())).collect();
        let mut count = 0usize;

        for row in rows(&text) {
            let (got, want, line) = (module.value(&row), row.want, row.line);
            // Three regimes, because one measure does not fit them. Where the closed form has a pole
            // — `Rₙₘˡ` at `γ = −1` with `m·ε < 0`, a perfectly retrograde equatorial orbit — there is
            // nothing to divide by, and the row holds IEEE's reading of the product, `∞` or, where
            // the other factor is an exact zero, NaN; only that same value counts. Where the closed
            // form is an exact zero the miss is absolute, because a polynomial that cancels to zero
            // exactly lands a rounding away in floating point. Otherwise it is relative, and a NaN
            // has to be admitted rather than dropped: `NaN > worst` is false, so the plain
            // comparison would report perfect agreement for the one row that has none.
            let w = &mut worst.iter_mut().find(|(k, _)| *k == row.kind).unwrap().1;
            let exact = got == want || (got.is_nan() && want.is_nan());
            if !want.is_finite() {
                assert!(exact, "{line} -> got {got:.17e}, no tolerance at a pole");
            } else if want == 0.0 {
                if libm::fabs(got) > w.abs {
                    w.abs = libm::fabs(got);
                    w.abs_row = std::format!("{line} -> got {got:.17e}");
                }
            } else {
                let miss = libm::fabs((got - want) / want);
                if miss.is_nan() || miss > w.rel {
                    w.rel = nan_max(w.rel, miss);
                    w.rel_row = std::format!("{line} -> got {got:.17e}");
                }
            }
            count += 1;
        }

        assert_eq!(count, 7712, "fixture row count");
        let mut report = String::new();
        for ((kind, rel_bound, abs_bound), (_, w)) in BOUNDS.iter().zip(worst.iter()) {
            use core::fmt::Write as _;
            if exceeds(w.rel, *rel_bound) {
                let _ = writeln!(report, "{kind}: relative {:.3e} at {}", w.rel, w.rel_row);
            }
            if exceeds(w.abs, *abs_bound) {
                let _ = writeln!(report, "{kind}: absolute {:.3e} at {}", w.abs, w.abs_row);
            }
        }
        assert!(report.is_empty(), "\n{report}");
    }

    /// `compute_qns` keeps one column past `s_max`, `Qₙ,ₛₘₐₓ₊₁ = dQₙ,ₛₘₐₓ/dγ`, which the zonal rates
    /// read, and a narrower table is the wider one's prefix: each column's recurrence reads only its
    /// own column and the diagonal. The closed forms decide the full table above; this pins the
    /// width the callers ask for. (S2)
    #[test]
    fn a_narrower_qns_table_is_the_wider_ones_prefix_with_its_derivative_column() {
        for gamma in [-0.8, 0.4, 0.9995] {
            let wide = compute_qns(gamma, QNS_N, QNS_N);
            for s_max in 0..QNS_N {
                for (n, row) in compute_qns(gamma, QNS_N, s_max).iter().enumerate() {
                    assert_eq!(
                        row.len(),
                        (n + 1).min(s_max + 2),
                        "n = {n}, s_max = {s_max}"
                    );
                    assert_eq!(row[..], wide[n][..row.len()], "n = {n}, s_max = {s_max}");
                }
            }
        }
    }

    /// The distance of each quantity to Orekit 13.1.7's `CoefficientsFactory` and `UpperBounds`,
    /// whose values `tools/orekit-oracle/DsstCoefficientsOracle.java` writes to the fixture that
    /// pairs line by line with the closed forms': relative, over the rows where Orekit's value is
    /// finite and not zero.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn distance_to_orekit() {
        let mut module = Module::new();
        let mut distances: Vec<(&str, Vec<f64>)> =
            QUANTITY.iter().map(|q| (q.0, Vec::new())).collect();
        for row in rows(&fixture("orekit_dsst_coefficients.csv")) {
            if row.want.is_finite() && row.want != 0.0 {
                let got = module.value(&row);
                let d = &mut distances
                    .iter_mut()
                    .find(|(k, _)| *k == row.kind)
                    .unwrap()
                    .1;
                d.push(libm::fabs((got - row.want) / row.want));
            }
        }
        for ((_, quantity), (_, d)) in QUANTITY.iter().zip(distances.iter()) {
            crate::dsst::comparison_row(quantity, "relative", d);
        }
    }

    /// Does the worst miss break its bound? A NaN does, and a bare `worst > bound` says it does
    /// not: IEEE-754 makes every ordered comparison against NaN false, so the one measurement that
    /// has no meaning reads as the tightest. [`nan_max`] keeps a NaN in the accumulator; this
    /// keeps it in the verdict. Measured, not argued: with `pnml * pnml` mutated to `pnml / pnml`
    /// in `get_rnml` the worst relative miss is NaN, and the test passed until this existed.
    fn exceeds(worst: f64, bound: f64) -> bool {
        worst.is_nan() || worst > bound
    }

    fn qns_for(cache: &mut Vec<(f64, QnsTable)>, gamma: f64) -> &QnsTable {
        if let Some(i) = cache.iter().position(|(k, _)| *k == gamma) {
            return &cache[i].1;
        }
        cache.push((gamma, compute_qns(gamma, QNS_N, QNS_N)));
        &cache.last().unwrap().1
    }

    fn gs_hs_for(cache: &mut Vec<((usize, usize), GsHs)>, key: (usize, usize)) -> &GsHs {
        if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
            return &cache[i].1;
        }
        let ((k, h), (alpha, beta)) = (KH[key.0], AB[key.1]);
        cache.push((key, compute_gs_hs(k, h, alpha, beta, GS_ORDER)));
        &cache.last().unwrap().1
    }
}
