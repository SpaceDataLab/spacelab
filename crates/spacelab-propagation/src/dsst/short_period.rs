//! The DSST central-body **zonal** short-period corrections (Danielson §4.1), turning DSST mean
//! equinoctial elements into osculating ones.
//!
//! Direct port of Orekit's `DSSTZonal` short-period machinery: the `FourierCjSjCoefficients`
//! (`cij`/`sij` and their derivatives, Danielson 4.1-(13)…(16b)), `computeCijSijCoefficients`
//! (4.1-(20)…(25)), `computeRhoSigmaCoefficients` (2.5.3-(4)), `computeDiCoefficients`, and the
//! `ZonalShortPeriodicCoefficients.value(meanOrbit)` assembly. Variable names follow Orekit for
//! auditability.
//!
//! # What is *not* ported
//!
//! Orekit computes these coefficients at integration-step grid points and Hermite-interpolates them
//! between steps (`ShortPeriodicsInterpolatedCoefficient`). That interpolation is a propagator
//! implementation detail, not part of the theory: at any output date the osculating state is
//! `mean + Σ value(meanOrbit)` where the coefficients are those of the mean state itself. This module
//! evaluates the coefficients *directly* at the mean state, which is what the interpolation
//! approximates and is exact at every grid point (Orekit seeds one at the initial state). The oracle
//! therefore calls Orekit's `ShortPeriodTerms.value(meanOrbit)` at the mean state directly (see
//! `tools/orekit-oracle/DsstZonalShortPeriodOracle.java`), not the interpolated propagator output.
//!
//! # Truncation (Orekit defaults for `new DSSTZonal(provider)`)
//!
//! * `maxDegreeShortPeriodics` = `maxDegree`;
//! * `maxEccPowShortPeriodics` = `min(4, maxDegree − 1)` (the `s` sum bound);
//! * `maxFrequencyShortPeriodics` = `2·maxDegree + 1` (the `j` true-longitude frequency bound).
//!
//! The `U`-potential value and the mean-element rates that feed the `Dᵢ` coefficients use
//! `maxEccPowMeanElements` (the same truncation as the mean rates), exactly as Orekit's `UAnddU`.

use std::vec::Vec;

use libm::{atan, cos, fabs, sin, sqrt};

use super::auxiliary::{DsstAuxiliary, I};
use super::coefficients;
use super::hansen::HansenZonal;
use super::zonal::DsstZonal;

/// Convergence tolerance for the equinoctial Kepler solve (Orekit
/// `EquinoctialLongitudeArgumentUtility.TOLERANCE_CONVERGENCE`).
const KEPLER_TOLERANCE: f64 = 1.0e-11;
/// Maximum iterations for the equinoctial Kepler solve (Orekit `MAXIMUM_ITERATION`).
const KEPLER_MAX_ITER: usize = 50;

/// Cⱼ(x, y) + i·Sⱼ(x, y) = (x + i·y)ʲ (Danielson 2.5.3-(5)) with the partial derivatives
/// ∂Cⱼ/∂x = j·Cⱼ₋₁, ∂Sⱼ/∂x = j·Sⱼ₋₁, ∂Cⱼ/∂y = −j·Sⱼ₋₁, ∂Sⱼ/∂y = j·Cⱼ₋₁. Orekit's `CjSjCoefficient`.
struct CjSj {
    cj: Vec<f64>,
    sj: Vec<f64>,
}

impl CjSj {
    fn new(x: f64, y: f64, j_max: usize) -> Self {
        let mut cj = std::vec![0.0; j_max + 1];
        let mut sj = std::vec![0.0; j_max + 1];
        cj[0] = 1.0;
        sj[0] = 0.0;
        for j in 1..=j_max {
            // (x + i·y)ʲ = (x + i·y)ʲ⁻¹ · (x + i·y).
            cj[j] = cj[j - 1] * x - sj[j - 1] * y;
            sj[j] = cj[j - 1] * y + sj[j - 1] * x;
        }
        Self { cj, sj }
    }

    fn cj(&self, j: usize) -> f64 {
        self.cj[j]
    }
    fn sj(&self, j: usize) -> f64 {
        self.sj[j]
    }
    fn dcj_dx(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            j as f64 * self.cj[j - 1]
        }
    }
    fn dsj_dx(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            j as f64 * self.sj[j - 1]
        }
    }
    fn dcj_dy(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            -(j as f64) * self.sj[j - 1]
        }
    }
    fn dsj_dy(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            j as f64 * self.cj[j - 1]
        }
    }
}

/// The Gⱼₛ, Hⱼₛ, Iⱼₛ, Jⱼₛ polynomials and their h/k/α/β derivatives (Danielson 4.1-(10)), built from
/// Cⱼ,Sⱼ series in (k, h) and in (α, β). Orekit's `GHIJjsPolynomials`. The first argument indexes the
/// (k, h) series, the second the (α, β) series (matching the Orekit call sites `getGjs(s, jps)` etc.).
struct Ghij {
    kh: CjSj,
    ab: CjSj,
}

impl Ghij {
    fn new(k: f64, h: f64, alpha: f64, beta: f64, kh_max: usize, ab_max: usize) -> Self {
        Self {
            kh: CjSj::new(k, h, kh_max),
            ab: CjSj::new(alpha, beta, ab_max),
        }
    }

    fn g(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.cj(s) + self.kh.sj(j) * self.ab.sj(s)
    }
    fn dg_dk(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dx(j) * self.ab.cj(s) + self.kh.dsj_dx(j) * self.ab.sj(s)
    }
    fn dg_dh(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dy(j) * self.ab.cj(s) + self.kh.dsj_dy(j) * self.ab.sj(s)
    }
    fn dg_dalpha(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dcj_dx(s) + self.kh.sj(j) * self.ab.dsj_dx(s)
    }
    fn dg_dbeta(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dcj_dy(s) + self.kh.sj(j) * self.ab.dsj_dy(s)
    }

    fn h(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.sj(s) - self.kh.sj(j) * self.ab.cj(s)
    }
    fn dh_dk(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dx(j) * self.ab.sj(s) - self.kh.dsj_dx(j) * self.ab.cj(s)
    }
    fn dh_dh(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dy(j) * self.ab.sj(s) - self.kh.dsj_dy(j) * self.ab.cj(s)
    }
    fn dh_dalpha(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dsj_dx(s) - self.kh.sj(j) * self.ab.dcj_dx(s)
    }
    fn dh_dbeta(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dsj_dy(s) - self.kh.sj(j) * self.ab.dcj_dy(s)
    }

    fn i(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.sj(s) + self.kh.sj(j) * self.ab.cj(s)
    }
    fn di_dk(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dx(j) * self.ab.sj(s) + self.kh.dsj_dx(j) * self.ab.cj(s)
    }
    fn di_dh(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dy(j) * self.ab.sj(s) + self.kh.dsj_dy(j) * self.ab.cj(s)
    }
    fn di_dalpha(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dsj_dx(s) + self.kh.sj(j) * self.ab.dcj_dx(s)
    }
    fn di_dbeta(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dsj_dy(s) + self.kh.sj(j) * self.ab.dcj_dy(s)
    }

    fn jj(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.cj(s) - self.kh.sj(j) * self.ab.sj(s)
    }
    fn djj_dk(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dx(j) * self.ab.cj(s) - self.kh.dsj_dx(j) * self.ab.sj(s)
    }
    fn djj_dh(&self, j: usize, s: usize) -> f64 {
        self.kh.dcj_dy(j) * self.ab.cj(s) - self.kh.dsj_dy(j) * self.ab.sj(s)
    }
    fn djj_dalpha(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dcj_dx(s) - self.kh.sj(j) * self.ab.dsj_dx(s)
    }
    fn djj_dbeta(&self, j: usize, s: usize) -> f64 {
        self.kh.cj(j) * self.ab.dcj_dy(s) - self.kh.sj(j) * self.ab.dsj_dy(s)
    }
}

/// Lₙˢ(γ) = (R/a)ⁿ·Vₙₛ·Qₙˢ(γ) and dLₙˢ/dγ = (R/a)ⁿ·Vₙₛ·Qₙ,ₛ₊₁ (0 when n = s). Orekit's
/// `LnsCoefficients`. Dense `[n][s]` over `0..=n_max`; the odd-parity and n < s entries are zero.
struct Lns {
    lns: Vec<Vec<f64>>,
    dlns: Vec<Vec<f64>>,
}

impl Lns {
    fn new(n_max: usize, vns: &[Vec<f64>], qns: &[Vec<f64>], roa: f64) -> Self {
        let mut lns = std::vec![std::vec![0.0; n_max + 1]; n_max + 1];
        let mut dlns = std::vec![std::vec![0.0; n_max + 1]; n_max + 1];
        let mut roa_pow = std::vec![0.0; n_max + 1];
        roa_pow[0] = 1.0;
        for i in 1..=n_max {
            roa_pow[i] = roa * roa_pow[i - 1];
        }
        for s in 0..=n_max {
            for n in s..=n_max {
                if (n - s) % 2 == 0 {
                    let coef = roa_pow[n] * vns[n][s];
                    lns[n][s] = coef * qns[n][s];
                    dlns[n][s] = if n == s { 0.0 } else { coef * qns[n][s + 1] };
                }
            }
        }
        Self { lns, dlns }
    }

    fn lns(&self, n: usize, s: usize) -> f64 {
        self.lns[n][s]
    }
    fn dlns(&self, n: usize, s: usize) -> f64 {
        self.dlns[n][s]
    }
}

/// The mean-element-derived scalar factors of `DSSTZonalContext` used by the short-period
/// assembly. Holds a reference to the auxiliary elements plus the derived combinations.
struct Ctx<'a> {
    aux: &'a DsstAuxiliary,
    roa: f64,
    /// h·k.
    hk: f64,
    /// k² − h².
    k2mh2: f64,
    /// (k² − h²) / 2.
    k2mh2o2: f64,
    /// 1 / (n²·a²) = 1 / (A·n).
    oon2a2: f64,
    /// χ³ / (n²·a).
    x3on2a: f64,
    /// χ / (n²·a²).
    xon2a2: f64,
    /// (C·χ) / (2·n²·a²).
    cxo2n2a2: f64,
    /// χ² / (n²·a²·(χ + 1)).
    x2on2a2xp1: f64,
    /// B·B.
    bb: f64,
}

impl<'a> Ctx<'a> {
    fn new(aux: &'a DsstAuxiliary, roa: f64) -> Self {
        let oon2a2 = 1.0 / (aux.big_a * aux.n);
        let oon2a = aux.sma * oon2a2;
        let x3on2a = aux.chi3 * oon2a;
        let xon2a2 = aux.chi * oon2a2;
        let cxo2n2a2 = xon2a2 * aux.big_c / 2.0;
        let x2on2a2xp1 = xon2a2 * aux.chi / (aux.chi + 1.0);
        let k2mh2 = aux.k * aux.k - aux.h * aux.h;
        Self {
            aux,
            roa,
            hk: aux.h * aux.k,
            k2mh2,
            k2mh2o2: k2mh2 / 2.0,
            oon2a2,
            x3on2a,
            xon2a2,
            cxo2n2a2,
            x2on2a2xp1,
            bb: aux.big_b * aux.big_b,
        }
    }
}

/// `index ∈ [lo, hi]`, on `i64` to keep the Orekit bounds (`2N−3`, …) safe near zero.
fn is_between(index: i64, lo: i64, hi: i64) -> bool {
    index >= lo && index <= hi
}

/// Accumulate one Gⱼₛ/Hⱼₛ term into the C/S rows (first and second double sums, and the leading
/// `j ∈ [2, N]` term). `c_sign` flips the C-coefficient sign between the two double sums; the
/// S-coefficient is always additive. Orekit `generateCoefficients`, the G/H branch.
#[allow(clippy::too_many_arguments)]
fn accumulate_hg(
    crow: &mut [f64; 7],
    srow: &mut [f64; 7],
    s: usize,
    n: usize,
    ab: usize,
    d0: f64,
    c_sign: f64,
    ghij: &Ghij,
    lns: &Lns,
    hansen: &[HansenZonal],
    c_n0: &[f64],
    h_xxx: f64,
    k_xxx: f64,
) {
    let lns_v = lns.lns(n, ab);
    let dlns_v = lns.dlns(n, ab);
    let hjs = ghij.h(s, ab);
    let gjs = ghij.g(s, ab);
    let jn = -c_n0[n];
    let kns = hansen[s].value(n);
    let dkns = hansen[s].derivative(n);
    let coef0 = d0 * jn;
    let coef1 = coef0 * lns_v;
    let coef2 = coef1 * kns;
    let coef3 = coef2 * hjs;
    let coef4 = coef2 * gjs;
    let np1 = (n + 1) as f64;

    crow[0] += c_sign * coef3;
    crow[1] += c_sign * coef3 * np1;
    crow[2] += c_sign * coef1 * (kns * ghij.dh_dh(s, ab) + hjs * h_xxx * dkns);
    crow[3] += c_sign * coef1 * (kns * ghij.dh_dk(s, ab) + hjs * k_xxx * dkns);
    crow[4] += c_sign * coef2 * ghij.dh_dalpha(s, ab);
    crow[5] += c_sign * coef2 * ghij.dh_dbeta(s, ab);
    crow[6] += c_sign * coef0 * dlns_v * kns * hjs;

    srow[0] += coef4;
    srow[1] += coef4 * np1;
    srow[2] += coef1 * (kns * ghij.dg_dh(s, ab) + gjs * h_xxx * dkns);
    srow[3] += coef1 * (kns * ghij.dg_dk(s, ab) + gjs * k_xxx * dkns);
    srow[4] += coef2 * ghij.dg_dalpha(s, ab);
    srow[5] += coef2 * ghij.dg_dbeta(s, ab);
    srow[6] += coef0 * dlns_v * kns * gjs;
}

/// Accumulate one Iⱼₛ/Jⱼₛ term into the C/S rows (the third double sum, the `j ∈ [2, N]` simple sum,
/// and the `j ∈ [3, 2N−1]` Σ expressions). C is always subtractive, S additive. Orekit
/// `generateCoefficients`, the I/J branch.
#[allow(clippy::too_many_arguments)]
fn accumulate_ij(
    crow: &mut [f64; 7],
    srow: &mut [f64; 7],
    s: usize,
    n: usize,
    jms: usize,
    d0: f64,
    ghij: &Ghij,
    lns: &Lns,
    hansen: &[HansenZonal],
    c_n0: &[f64],
    h_xxx: f64,
    k_xxx: f64,
) {
    let lns_v = lns.lns(n, jms);
    let dlns_v = lns.dlns(n, jms);
    let ijs = ghij.i(s, jms);
    let jjs = ghij.jj(s, jms);
    let jn = -c_n0[n];
    let kns = hansen[s].value(n);
    let dkns = hansen[s].derivative(n);
    let coef0 = d0 * jn;
    let coef1 = coef0 * lns_v;
    let coef2 = coef1 * kns;
    let coef3 = coef2 * ijs;
    let coef4 = coef2 * jjs;
    let np1 = (n + 1) as f64;

    crow[0] -= coef3;
    crow[1] -= coef3 * np1;
    crow[2] -= coef1 * (kns * ghij.di_dh(s, jms) + ijs * h_xxx * dkns);
    crow[3] -= coef1 * (kns * ghij.di_dk(s, jms) + ijs * k_xxx * dkns);
    crow[4] -= coef2 * ghij.di_dalpha(s, jms);
    crow[5] -= coef2 * ghij.di_dbeta(s, jms);
    crow[6] -= coef0 * dlns_v * kns * ijs;

    srow[0] += coef4;
    srow[1] += coef4 * np1;
    srow[2] += coef1 * (kns * ghij.djj_dh(s, jms) + jjs * h_xxx * dkns);
    srow[3] += coef1 * (kns * ghij.djj_dk(s, jms) + jjs * k_xxx * dkns);
    srow[4] += coef2 * ghij.djj_dalpha(s, jms);
    srow[5] += coef2 * ghij.djj_dbeta(s, jms);
    srow[6] += coef0 * dlns_v * kns * jjs;
}

/// The Fourier Cʲ, Sʲ coefficients and their (a, h, k, α, β, γ) derivatives (Danielson 4.1-(13)…),
/// as `c_coef[j][row]` / `s_coef[j][row]` with `row` = 0:value, 1:∂/∂a, 2:∂/∂h, 3:∂/∂k, 4:∂/∂α,
/// 5:∂/∂β, 6:∂/∂γ. Direct port of `FourierCjSjCoefficients.generateCoefficients`.
fn fourier_cj_sj(
    ctx: &Ctx,
    n_max: usize,
    s_max: usize,
    j_max: usize,
    c_n0: &[f64],
    vns: &[Vec<f64>],
) -> (Vec<[f64; 7]>, Vec<[f64; 7]>) {
    let aux = ctx.aux;
    let ghij = Ghij::new(aux.k, aux.h, aux.alpha, aux.beta, s_max, n_max);
    let qns = coefficients::compute_qns(aux.gamma, n_max, n_max);
    let lns = Lns::new(n_max, vns, &qns, ctx.roa);
    let hansen: Vec<HansenZonal> = (0..=s_max)
        .map(|s| HansenZonal::new(n_max, s, aux.chi))
        .collect();
    let h_xxx = aux.h * aux.chi3;
    let k_xxx = aux.k * aux.chi3;

    let n_max_i = n_max as i64;
    let mut cc = std::vec![[0.0f64; 7]; j_max + 1];
    let mut sc = std::vec![[0.0f64; 7]; j_max + 1];

    for j in 1..=j_max {
        let ji = j as i64;
        let (mut crow, mut srow) = ([0.0f64; 7], [0.0f64; 7]);

        if is_between(ji, 1, n_max_i - 1) {
            // First double sum: s: j → min(N−1, sMax), n: s+1 → N.
            for s in j..=(n_max - 1).min(s_max) {
                let ab = s - j; // −(j − s) ≥ 0
                let d0smj = if s == j { 1.0 } else { 2.0 };
                for n in (s + 1)..=n_max {
                    if (n + j - s) % 2 == 0 {
                        accumulate_hg(
                            &mut crow, &mut srow, s, n, ab, d0smj, 1.0, &ghij, &lns, &hansen, c_n0,
                            h_xxx, k_xxx,
                        );
                    }
                }
            }
            // Second double sum: s: 0 → min(N−j, sMax), n: max(j+s, j+1) → N.
            for s in 0..=(n_max - j).min(s_max) {
                let jps = j + s;
                for n in (j + s).max(j + 1)..=n_max {
                    if (n + jps) % 2 == 0 {
                        accumulate_hg(
                            &mut crow, &mut srow, s, n, jps, 2.0, -1.0, &ghij, &lns, &hansen, c_n0,
                            h_xxx, k_xxx,
                        );
                    }
                }
            }
            // Third double sum: s: 1 → min(j, sMax), n: j+1 → N.
            for s in 1..=j.min(s_max) {
                let jms = j - s;
                let d0smj = if s == j { 1.0 } else { 2.0 };
                for n in (j + 1)..=n_max {
                    if (n + jms) % 2 == 0 {
                        accumulate_ij(
                            &mut crow, &mut srow, s, n, jms, d0smj, &ghij, &lns, &hansen, c_n0,
                            h_xxx, k_xxx,
                        );
                    }
                }
            }
        }

        if is_between(ji, 2, n_max_i) {
            // Leading G/H term at s = 0, n = j (dLns = 0 since n = s = j).
            accumulate_hg(
                &mut crow, &mut srow, 0, j, j, 2.0, -1.0, &ghij, &lns, &hansen, c_n0, h_xxx, k_xxx,
            );
            // Simple sum s: 1 → min(j−1, sMax), even s only, n = j.
            for s in 1..=(j - 1).min(s_max) {
                if s % 2 == 0 {
                    let jms = j - s;
                    let d0smj = if s == j { 1.0 } else { 2.0 };
                    accumulate_ij(
                        &mut crow, &mut srow, s, j, jms, d0smj, &ghij, &lns, &hansen, c_n0, h_xxx,
                        k_xxx,
                    );
                }
            }
        }

        if is_between(ji, 3, 2 * n_max_i - 1) {
            let minjm1on = (j - 1).min(n_max);
            if j % 2 == 0 {
                // First Σ: s: j−min(j−1,N) → min(j/2−1, sMax), n: j−s → min(j−1,N).
                for s in (j - minjm1on)..=(j / 2 - 1).min(s_max) {
                    let jms = j - s;
                    let d0smj = if s == j { 1.0 } else { 2.0 };
                    for n in (j - s)..=minjm1on {
                        if (n + jms) % 2 == 0 {
                            accumulate_ij(
                                &mut crow, &mut srow, s, n, jms, d0smj, &ghij, &lns, &hansen, c_n0,
                                h_xxx, k_xxx,
                            );
                        }
                    }
                }
                // Second Σ: s: j/2 → min(min(j−1,N)−1, sMax), n: s+1 → min(j−1,N).
                for s in (j / 2)..=(minjm1on - 1).min(s_max) {
                    let jms = j - s;
                    let d0smj = if s == j { 1.0 } else { 2.0 };
                    for n in (s + 1)..=minjm1on {
                        if (n + jms) % 2 == 0 {
                            accumulate_ij(
                                &mut crow, &mut srow, s, n, jms, d0smj, &ghij, &lns, &hansen, c_n0,
                                h_xxx, k_xxx,
                            );
                        }
                    }
                }
            } else {
                // First Σ: s: (j−1)/2 → min(min(j−1,N)−1, sMax), n: s+1 → min(j−1,N).
                for s in ((j - 1) / 2)..=(minjm1on - 1).min(s_max) {
                    let jms = j - s;
                    let d0smj = if s == j { 1.0 } else { 2.0 };
                    for n in (s + 1)..=minjm1on {
                        if (n + jms) % 2 == 0 {
                            accumulate_ij(
                                &mut crow, &mut srow, s, n, jms, d0smj, &ghij, &lns, &hansen, c_n0,
                                h_xxx, k_xxx,
                            );
                        }
                    }
                }
                // Second Σ (only if N ≥ 4 and j ∈ [5, 2N−3]): s: j−min(j−1,N) → min((j−3)/2, sMax).
                if n_max >= 4 && is_between(ji, 5, 2 * n_max_i - 3) {
                    for s in (j - minjm1on)..=((j - 3) / 2).min(s_max) {
                        let jms = j - s;
                        let d0smj = if s == j { 1.0 } else { 2.0 };
                        for n in (j - s)..=minjm1on {
                            if (n + jms) % 2 == 0 {
                                accumulate_ij(
                                    &mut crow, &mut srow, s, n, jms, d0smj, &ghij, &lns, &hansen,
                                    c_n0, h_xxx, k_xxx,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Common factors (Orekit): row 1 (∂/∂a) carries μ/(a·j), the rest −μ/(a·j) · a = −μoa/j.
        let muoa = aux.muoa;
        let jf = j as f64;
        crow[0] *= -muoa / jf;
        crow[1] *= muoa / (jf * aux.sma);
        crow[2] *= -muoa / jf;
        crow[3] *= -muoa / jf;
        crow[4] *= -muoa / jf;
        crow[5] *= -muoa / jf;
        crow[6] *= -muoa / jf;
        srow[0] *= -muoa / jf;
        srow[1] *= muoa / (jf * aux.sma);
        srow[2] *= -muoa / jf;
        srow[3] *= -muoa / jf;
        srow[4] *= -muoa / jf;
        srow[5] *= -muoa / jf;
        srow[6] *= -muoa / jf;

        cc[j] = crow;
        sc[j] = srow;
    }

    (cc, sc)
}

/// ρⱼ = (1 + jB)(−b)ʲCⱼ(k, h), σⱼ = (1 + jB)(−b)ʲSⱼ(k, h) with b = 1/(1+B) (Danielson 2.5.3-(4)).
/// Orekit `computeRhoSigmaCoefficients`. Returns `[ρ, σ]` per `j`, index 0 unused.
fn compute_rho_sigma(aux: &DsstAuxiliary, j_max: usize) -> Vec<[f64; 2]> {
    let cjsj = CjSj::new(aux.k, aux.h, j_max);
    let b = 1.0 / (1.0 + aux.big_b);
    let mut mbtj = 1.0;
    let mut rho_sigma = std::vec![[0.0f64; 2]; j_max + 1];
    for j in 1..=j_max {
        mbtj *= -b;
        let coef = (1.0 + j as f64 * aux.big_b) * mbtj;
        rho_sigma[j][0] = coef * cjsj.cj(j);
        rho_sigma[j][1] = coef * cjsj.sj(j);
    }
    rho_sigma
}

/// The Dᵢ coefficients: mean-element rate / n, with the λ component carrying the extra
/// −3·U/(n²·a²) (Orekit `computeDiCoefficients`, `−1.5·2·U·oon2a2`).
fn compute_di(ctx: &Ctx, mean_rates: &[f64; 6], u: f64) -> [f64; 6] {
    let n = ctx.aux.n;
    let mut di = [0.0f64; 6];
    for i in 0..6 {
        di[i] = mean_rates[i] / n;
    }
    di[5] += -3.0 * u * ctx.oon2a2;
    di
}

/// The Cᵢⱼ, Sᵢⱼ short-period coefficients and Cᵢ⁰ (Danielson 4.1-(20)…(25)). Returns
/// `(cij, sij)` with `cij[0]` = Cᵢ⁰ and `cij[j][i]` for the equinoctial element `i` (0:a, 1:k, 2:h,
/// 3:q, 4:p, 5:λ). Direct port of `computeCijSijCoefficients`.
#[allow(clippy::needless_range_loop)]
fn compute_cij_sij(
    ctx: &Ctx,
    cc: &[[f64; 7]],
    sc: &[[f64; 7]],
    rho_sigma: &[[f64; 2]],
    u: f64,
    n_max: usize,
    j_max: usize,
) -> (Vec<[f64; 6]>, Vec<[f64; 6]>) {
    let aux = ctx.aux;
    let (k, h, p, q, sma) = (aux.k, aux.h, aux.p, aux.q, aux.sma);
    let n_max_i = n_max as i64;
    let mut cij = std::vec![[0.0f64; 6]; j_max + 1];
    let mut sij = std::vec![[0.0f64; 6]; j_max + 1];
    let mut ci0 = [0.0f64; 6];

    for j in 1..=j_max {
        let ji = j as i64;
        let (mut cur_c, mut cur_s) = ([0.0f64; 6], [0.0f64; 6]);

        if j == 1 {
            let coef1 = 4.0 * k * u - ctx.hk * cc[1][0] + ctx.k2mh2o2 * sc[1][0];
            let coef2 = 4.0 * h * u + ctx.k2mh2o2 * cc[1][0] + ctx.hk * sc[1][0];
            let coef3 = (k * cc[1][0] + h * sc[1][0]) / 4.0;
            let coef4 = (8.0 * u - h * cc[1][0] + k * sc[1][0]) / 4.0;
            cur_c[0] += coef1;
            cur_s[0] += coef2;
            cur_c[1] += coef4;
            cur_s[1] += coef3;
            cur_c[2] -= coef3;
            cur_s[2] += coef4;
            cur_c[5] -= coef2 / 2.0;
            cur_s[5] += coef1 / 2.0;
        }

        if j == 2 {
            let coef1 = ctx.k2mh2 * u;
            let coef2 = 2.0 * ctx.hk * u;
            let coef3 = h * u / 2.0;
            let coef4 = k * u / 2.0;
            cur_c[0] += coef1;
            cur_s[0] += coef2;
            cur_c[1] += coef4;
            cur_s[1] += coef3;
            cur_c[2] -= coef3;
            cur_s[2] += coef4;
            cur_c[5] -= coef2 / 2.0;
            cur_s[5] += coef1 / 2.0;
        }

        if is_between(ji, 1, 2 * n_max_i - 3) && j + 2 < j_max {
            let jp2 = (j + 2) as f64;
            let (cj, sj) = (cc[j + 2][0], sc[j + 2][0]);
            let coef1 = jp2 * (-ctx.hk * cj + ctx.k2mh2o2 * sj);
            let coef2 = jp2 * (ctx.k2mh2o2 * cj + ctx.hk * sj);
            let coef3 = jp2 * (k * cj + h * sj) / 4.0;
            let coef4 = jp2 * (h * cj - k * sj) / 4.0;
            cur_c[0] += coef1;
            cur_s[0] -= coef2;
            cur_c[1] -= coef4;
            cur_s[1] -= coef3;
            cur_c[2] -= coef3;
            cur_s[2] += coef4;
            cur_c[5] -= coef2 / 2.0;
            cur_s[5] += -coef1 / 2.0;
        }

        if is_between(ji, 1, 2 * n_max_i - 2) && j + 1 < j_max {
            let jp1 = (j + 1) as f64;
            let (cj, sj) = (cc[j + 1][0], sc[j + 1][0]);
            let coef1 = 2.0 * jp1 * (-h * cj + k * sj);
            let coef2 = 2.0 * jp1 * (k * cj + h * sj);
            let coef3 = jp1 * cj;
            let coef4 = jp1 * sj;
            cur_c[0] += coef1;
            cur_s[0] -= coef2;
            cur_c[1] += coef4;
            cur_s[1] -= coef3;
            cur_c[2] -= coef3;
            cur_s[2] -= coef4;
            cur_c[5] -= coef2 / 2.0;
            cur_s[5] += -coef1 / 2.0;
        }

        if is_between(ji, 2, 2 * n_max_i) && j - 1 < j_max {
            let jm1 = (j - 1) as f64;
            let (cj, sj) = (cc[j - 1][0], sc[j - 1][0]);
            let coef1 = 2.0 * jm1 * (h * cj + k * sj);
            let coef2 = 2.0 * jm1 * (k * cj - h * sj);
            let coef3 = jm1 * cj;
            let coef4 = jm1 * sj;
            cur_c[0] += coef1;
            cur_s[0] -= coef2;
            cur_c[1] += coef4;
            cur_s[1] -= coef3;
            cur_c[2] += coef3;
            cur_s[2] += coef4;
            cur_c[5] += coef2 / 2.0;
            cur_s[5] += coef1 / 2.0;
        }

        if is_between(ji, 3, 2 * n_max_i + 1) && j - 2 < j_max {
            let jm2 = (j - 2) as f64;
            let (cj, sj) = (cc[j - 2][0], sc[j - 2][0]);
            let coef1 = jm2 * (ctx.hk * cj + ctx.k2mh2o2 * sj);
            let coef2 = jm2 * (-ctx.k2mh2o2 * cj + ctx.hk * sj);
            let coef3 = jm2 * (k * cj - h * sj) / 4.0;
            let coef4 = jm2 * (h * cj + k * sj) / 4.0;
            let coef5 = jm2 * (ctx.k2mh2o2 * cj - ctx.hk * sj);
            cur_c[0] += coef1;
            cur_s[0] += coef2;
            cur_c[1] += coef4;
            cur_s[1] -= coef3;
            cur_c[2] += coef3;
            cur_s[2] += coef4;
            cur_c[5] += coef5 / 2.0;
            cur_s[5] += coef1 / 2.0;
        }

        // Multiply by the per-element common factor.
        cur_c[0] *= ctx.x3on2a;
        cur_s[0] *= ctx.x3on2a;
        cur_c[1] *= ctx.xon2a2;
        cur_s[1] *= ctx.xon2a2;
        cur_c[2] *= ctx.xon2a2;
        cur_s[2] *= ctx.xon2a2;
        cur_c[5] *= ctx.x2on2a2xp1;
        cur_s[5] *= ctx.x2on2a2xp1;

        if is_between(ji, 1, 2 * n_max_i - 1) && j < j_max {
            let (al, be, ga, chi) = (aux.alpha, aux.beta, aux.gamma, aux.chi);
            // Cross derivatives (Danielson 2.2-(8) form applied to the Fourier coefficients).
            let cj_ag = al * cc[j][6] - ga * cc[j][4];
            let cj_ab = al * cc[j][5] - be * cc[j][4];
            let cj_bg = be * cc[j][6] - ga * cc[j][5];
            let cj_hk = h * cc[j][3] - k * cc[j][2];
            let sj_ag = al * sc[j][6] - ga * sc[j][4];
            let sj_ab = al * sc[j][5] - be * sc[j][4];
            let sj_bg = be * sc[j][6] - ga * sc[j][5];
            let sj_hk = h * sc[j][3] - k * sc[j][2];
            let jf = j as f64;

            let coef1 = ctx.x3on2a * (3.0 - ctx.bb) * jf;
            cur_c[0] += coef1 * sc[j][0];
            cur_s[0] -= coef1 * cc[j][0];

            let coef2 = p * cj_ag - I * q * cj_bg;
            let coef3 = p * sj_ag - I * q * sj_bg;
            cur_c[1] -= ctx.xon2a2 * (h * coef2 + ctx.bb * cc[j][2] - 1.5 * k * jf * sc[j][0]);
            cur_s[1] -= ctx.xon2a2 * (h * coef3 + ctx.bb * sc[j][2] + 1.5 * k * jf * cc[j][0]);
            cur_c[2] += ctx.xon2a2 * (k * coef2 + ctx.bb * cc[j][3] + 1.5 * h * jf * sc[j][0]);
            cur_s[2] += ctx.xon2a2 * (k * coef3 + ctx.bb * sc[j][3] - 1.5 * h * jf * cc[j][0]);

            let coef4 = cj_hk - cj_ab - jf * sc[j][0];
            let coef5 = sj_hk - sj_ab + jf * cc[j][0];
            cur_c[3] = ctx.cxo2n2a2 * (-I * cj_ag + q * coef4);
            cur_s[3] = ctx.cxo2n2a2 * (-I * sj_ag + q * coef5);
            cur_c[4] = ctx.cxo2n2a2 * (-cj_bg + p * coef4);
            cur_s[4] = ctx.cxo2n2a2 * (-sj_bg + p * coef5);

            let coef6 = h * cc[j][2] + k * cc[j][3];
            let coef7 = h * sc[j][2] + k * sc[j][3];
            cur_c[5] += ctx.oon2a2
                * (-2.0 * sma * cc[j][1] + coef6 / (chi + 1.0) + chi * coef2 - 3.0 * cc[j][0]);
            cur_s[5] += ctx.oon2a2
                * (-2.0 * sma * sc[j][1] + coef7 / (chi + 1.0) + chi * coef3 - 3.0 * sc[j][0]);
        }

        for i in 0..6 {
            ci0[i] -= cur_c[i] * rho_sigma[j][0] + cur_s[i] * rho_sigma[j][1];
        }
        cij[j] = cur_c;
        sij[j] = cur_s;
    }

    cij[0] = ci0;
    (cij, sij)
}

/// Eccentric → true longitude argument (Orekit `EquinoctialLongitudeArgumentUtility.eccentricToTrue`).
fn eccentric_to_true(ex: f64, ey: f64, le: f64) -> f64 {
    let epsilon = sqrt(1.0 - ex * ex - ey * ey);
    let (s, c) = (sin(le), cos(le));
    let num = ex * s - ey * c;
    let den = epsilon + 1.0 - ex * c - ey * s;
    le + 2.0 * atan(num / den)
}

/// Mean → eccentric longitude argument via the equinoctial Kepler equation
/// λ = LE − k·sin(LE) + h·cos(LE), solved with Orekit's Danby-style step (`meanToEccentric`).
///
/// # Panics
///
/// If the iteration does not converge within [`KEPLER_MAX_ITER`] (only for e ≈ 1, outside the
/// DSST-zonal LEO regime this module targets).
fn mean_to_eccentric(ex: f64, ey: f64, lm: f64) -> f64 {
    let mut le = lm;
    let mut le_m_lm = 0.0;
    for _ in 0..KEPLER_MAX_ITER {
        let (s, c) = (sin(le), cos(le));
        let f2 = ex * s - ey * c;
        let f1 = 1.0 - ex * c - ey * s;
        let f0 = le_m_lm - f2;
        let f12 = 2.0 * f1;
        let shift = f0 * f12 / (f1 * f12 - f0 * f2);
        le_m_lm -= shift;
        le = lm + le_m_lm;
        if fabs(shift) <= KEPLER_TOLERANCE {
            return le;
        }
    }
    panic!("equinoctial Kepler solve did not converge (e too close to 1)");
}

/// Compute the six equinoctial short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` at the mean state.
pub(crate) fn short_period_correction(zonal: &DsstZonal, mean_aux: &DsstAuxiliary) -> [f64; 6] {
    let n_max = zonal.max_degree();
    let s_max = 4.min(n_max - 1); // maxEccPowShortPeriodics
    let j_max = 2 * n_max + 1; // maxFrequencyShortPeriodics
    let max_ecc_pow_mean = zonal.max_ecc_pow(mean_aux);
    let roa = zonal.re / mean_aux.sma;
    let ctx = Ctx::new(mean_aux, roa);

    // Potential value + mean-element rates use the mean-element truncation (Orekit `UAnddU`).
    let udu = zonal.potential_derivatives(mean_aux, max_ecc_pow_mean);
    let mean_rates = zonal.mean_element_rates_with_pow(mean_aux, max_ecc_pow_mean);

    let (cc, sc) = fourier_cj_sj(&ctx, n_max, s_max, j_max, &zonal.c_n0, zonal.vns());
    let rho_sigma = compute_rho_sigma(mean_aux, j_max);
    let di = compute_di(&ctx, &mean_rates, udu.u);
    let (cij, sij) = compute_cij_sij(&ctx, &cc, &sc, &rho_sigma, udu.u, n_max, j_max);

    // Evaluate at the mean state's true longitude L = Lv(mean); the equation-of-centre term uses
    // L − λ_M (Orekit `ZonalShortPeriodicCoefficients.value`).
    let le = mean_to_eccentric(mean_aux.k, mean_aux.h, mean_aux.lambda);
    let lv = eccentric_to_true(mean_aux.k, mean_aux.h, le);
    let center = lv - mean_aux.lambda;

    let mut sp = cij[0];
    for i in 0..6 {
        sp[i] += center * di[i];
    }
    for j in 1..=j_max {
        let jl = j as f64 * lv;
        let (s, c) = (sin(jl), cos(jl));
        for i in 0..6 {
            sp[i] += cij[j][i] * c + sij[j][i] * s;
        }
    }
    sp
}

impl DsstZonal {
    /// The six equinoctial short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` for the zonal field at
    /// the given **mean** state, evaluated at the mean state's true longitude (Orekit's
    /// `ZonalShortPeriodicCoefficients.value(meanOrbit)`).
    #[must_use]
    pub fn short_period_correction(&self, mean_aux: &DsstAuxiliary) -> [f64; 6] {
        short_period_correction(self, mean_aux)
    }

    /// The osculating equinoctial state `mean + short-period` for a **mean** equinoctial state
    /// `[a, k, h, q, p, λ]` (Orekit's `MeanPlusShortPeriodicMapper`, `PositionAngleType.MEAN`, so the
    /// sixth element is the osculating mean longitude).
    #[must_use]
    pub fn osculating(&self, mean: [f64; 6]) -> [f64; 6] {
        let aux = DsstAuxiliary::new(
            mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], self.mu,
        );
        let sp = self.short_period_correction(&aux);
        let mut osc = mean;
        for i in 0..6 {
            osc[i] += sp[i];
        }
        osc
    }

    /// Semi-analytically propagate a **mean** equinoctial state forward by `dt` seconds and return
    /// the **osculating** equinoctial state `[a, k, h, q, p, λ]` — the DSST OSCULATING output.
    #[must_use]
    pub fn propagate_osculating(&self, initial_mean: [f64; 6], dt: f64) -> [f64; 6] {
        self.osculating(self.propagate_mean(initial_mean, dt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsst::DsstAuxiliary;
    use core::f64::consts::{PI, TAU};

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`), the same degree-6 zonal field and MEAN
    // equinoctial LEO as the `dsst` module tests. `MEAN` is the MEAN DSSTPropagator state
    // (orekit_dsst_zonal.csv); `SP` is `DSSTZonal.ShortPeriodTerms.value(meanOrbit)` evaluated at
    // exactly that state by `DsstZonalShortPeriodOracle`, so it carries no propagation and no
    // interpolation. Feeding the port the same mean state isolates the short-period assembly from
    // the mean propagation, which is what the `dsst::tests` osculating check cannot do.
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const C20: f64 = -1.08262689216735860e-03;
    const C30: f64 = 2.53209950653743350e-06;
    const C40: f64 = 1.62004577259847830e-06;
    const C50: f64 = 2.27515624171733840e-07;
    const C60: f64 = -5.41217294926390600e-07;

    #[rustfmt::skip]
    const MEAN: [[f64; 6]; 2] = [
        [7.20000000000000000e+06, 3.16395301514028070e-03, 9.47328593348677900e-03, 8.51839935608425400e-01, 7.73121044973265600e-01, 2.37304247212616130e+01],
        [7.20000000000000000e+06, 3.23744043048822130e-03, 9.43531196934242000e-03, 8.48776222732037000e-01, 7.76483292346584100e-01, 4.60296794537376600e+01],
    ];
    /// `[degree − 2][state]` over degrees 2…6. The degree sweep is not decoration: the frequency the
    /// `Cʲ`/`Sʲ` arrays die at is `min(2N−1, N + sMax)`, and `2N−1` is the binding half only for
    /// `N ≤ 5`, so several window guards of `compute_cij_sij` are unreachable at degree 6 alone.
    #[rustfmt::skip]
    const SP: [[[f64; 6]; 2]; 5] = [
        [
            [-3.45909460854783360e+03, 2.06105353789709460e-04, 1.05293576782805970e-03, 9.41608053915009100e-05, -4.28116849031910200e-05, 7.61505015450793000e-04],
            [-8.12172778279658500e+03, 5.37438567739130700e-04, -1.45028665590917530e-03, 9.70854999650846000e-05, 2.74060809289003400e-05, 3.94360894081119900e-04],
        ],
        [
            [-3.46360552775002300e+03, 2.07267330343685970e-04, 1.05356195659033530e-03, 9.47258813659257700e-05, -4.33558633538259040e-05, 7.64034454158945600e-04],
            [-8.10172631335961000e+03, 5.37057826959886600e-04, -1.44791627425373740e-03, 9.66434046638209300e-05, 2.75584861136626450e-05, 3.92786659489239900e-04],
        ],
        [
            [-3.46708563705235970e+03, 2.06514548799573260e-04, 1.05339154383918540e-03, 9.44784396440815800e-05, -4.30254888346163100e-05, 7.62951530459554900e-04],
            [-8.09398350515280100e+03, 5.36981911702085100e-04, -1.44643860282979860e-03, 9.64093657742930100e-05, 2.76875728032763730e-05, 3.91809438788189200e-04],
        ],
        [
            [-3.46650180949623560e+03, 2.06598498778590480e-04, 1.05333339554339840e-03, 9.45432244321828900e-05, -4.31067884220069900e-05, 7.63053929037943200e-04],
            [-8.09310297273251500e+03, 5.37007621095447900e-04, -1.44630259785300100e-03, 9.63574226667173100e-05, 2.77298923146529080e-05, 3.91666880664979570e-04],
        ],
        [
            [-3.46480613173448140e+03, 2.06738065327386060e-04, 1.05322825463387870e-03, 9.46425691017533300e-05, -4.32449890172684950e-05, 7.63088909436818400e-04],
            [-8.09409571551390300e+03, 5.36905594242196100e-04, -1.44656589677189840e-03, 9.64615183286401100e-05, 2.76320991724613050e-05, 3.91945908105699200e-04],
        ],
    ];

    fn zonal() -> DsstZonal {
        zonal_of_degree(6)
    }

    /// The same field truncated to `degree`, which is how Orekit's oracle was run for each `SP` row.
    fn zonal_of_degree(degree: usize) -> DsstZonal {
        let all = std::vec![0.0, 0.0, C20, C30, C40, C50, C60];
        DsstZonal::new(MU, AE, all[..=degree].to_vec())
    }

    fn aux_at(mean: [f64; 6]) -> DsstAuxiliary {
        DsstAuxiliary::new(mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], MU)
    }

    /// Reduce an angle difference to (−π, π] so a longitude comparison is wrap-safe.
    fn wrap(d: f64) -> f64 {
        d - TAU * libm::round(d / TAU)
    }

    /// The six short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` evaluated at Orekit's own mean
    /// state reproduce Orekit's own `ShortPeriodTerms.value` there to `9.1e-13 m` in `a` and
    /// `6.5e-19` in the remaining five components, over degrees 2…6.
    ///
    /// `dsst::tests::osculating_short_period_matches_orekit` compares the same physics but through a
    /// full `propagate_osculating` run and asserts `|Δa| < 1e-2 m` / `< 1e-9` per component against an
    /// achieved `1.06e-5 m` / `7.23e-12`: three and two orders of magnitude of slack, so any mutation
    /// that perturbs the assembly below those thresholds passes unseen.
    ///
    /// This fixture used to difference a MEAN and an OSCULATING propagator run, which left `3.03e-6 m`
    /// in `a` — Orekit's Hermite interpolation of its own short-period coefficients between
    /// integration steps, fifteen times the `1.97e-7 m` the surviving `compute_cij_sij` mutants move,
    /// so the comparison could not see them. Asking Orekit for the correction *at* the mean state
    /// removes the interpolation and the mean-propagation drift together, sharpening the residual by
    /// six orders of magnitude, and the bounds here are set at the measured figure.
    #[test]
    fn short_period_at_orekit_mean_state_matches_orekit() {
        let (mut max_da, mut max_comp) = (0.0_f64, 0.0_f64);
        for (d, row) in SP.iter().enumerate() {
            let degree = d + 2;
            let z = zonal_of_degree(degree);
            for k in 0..MEAN.len() {
                let sp = z.short_period_correction(&aux_at(MEAN[k]));
                let da = (sp[0] - row[k][0]).abs();
                assert!(da < 1.0e-12, "degree={degree} k={k}: |Δa| {da:.3e} m");
                max_da = max_da.max(da);
                for i in 1..6 {
                    let e = (sp[i] - row[k][i]).abs();
                    assert!(e < 7.0e-19, "degree={degree} k={k} comp[{i}]: {e:.3e}");
                    max_comp = max_comp.max(e);
                }
            }
        }
        std::println!(
            "zonal short-period at Orekit mean state, degrees 2..6: max |Δa| {max_da:.3e} m, max comp {max_comp:.3e}"
        );
    }

    /// Danielson 2.5.3-(4) defines ρⱼ and σⱼ as the λ-averages ⟨cos jL⟩ and ⟨sin jL⟩, and Cᵢ⁰ as
    /// `−Σⱼ (Cᵢⱼ ρⱼ + Sᵢⱼ σⱼ)` precisely so that the short-period correction has **zero mean over one
    /// revolution** — it is a periodic correction, not a secular one. Sweeping λ over a full turn at a
    /// frozen (a, k, h, q, p) and averaging must therefore return zero for all six components; the
    /// equation-of-centre term `(Lv − λ)·Dᵢ` carries zero mean of its own.
    ///
    /// This is the one property that ties ρ/σ to the assembly they are used in: the two Orekit
    /// comparisons only ever sample a single λ per state, where a wrong mean is indistinguishable from
    /// a wrong amplitude.
    #[test]
    fn short_period_has_zero_mean_over_one_revolution() {
        // A finite Fourier series in the true longitude has harmonics in λ that decay like eʲ, so a
        // uniform 2048-point rule leaves aliasing far below round-off at e = 0.01.
        const NGRID: usize = 2048;
        let z = zonal();
        let m = MEAN[0];
        let (mut acc, mut amp) = ([0.0f64; 6], [0.0f64; 6]);
        for g in 0..NGRID {
            let lambda = TAU * (g as f64) / (NGRID as f64);
            let sp = z.short_period_correction(&aux_at([m[0], m[1], m[2], m[3], m[4], lambda]));
            for i in 0..6 {
                acc[i] += sp[i];
                amp[i] = amp[i].max(sp[i].abs());
            }
        }
        let mut max_rel = 0.0_f64;
        for i in 0..6 {
            let rel = (acc[i] / NGRID as f64).abs() / amp[i];
            assert!(rel < 1e-14, "component {i}: mean/amplitude {rel:.3e}");
            max_rel = max_rel.max(rel);
        }
        std::println!("zonal short-period λ-mean / amplitude: max {max_rel:.3e}");
    }

    /// ρⱼ + i·σⱼ = (1 + jB)·(−b·(k + i·h))ʲ with b = 1/(1 + B) (Danielson 2.5.3-(4)). The port builds
    /// it from the Cartesian Cⱼ/Sⱼ recurrence and a running `(−b)ʲ`; the oracle here is the same
    /// complex number in **polar** form — modulus `(b·e)ʲ` and argument `j·(π + atan2(h, k))` — which
    /// shares no operation with the recurrence.
    ///
    /// The Orekit comparisons reach ρ/σ only through Cᵢ⁰, where the factor `(1 + jB)` is one term of a
    /// thirteen-term sum dominated by j = 1, so a wrong ρⱼ at large j is invisible to them.
    #[test]
    fn rho_sigma_match_the_polar_closed_form() {
        let aux = aux_at(MEAN[0]);
        let j_max = 2 * 6 + 1;
        let rho_sigma = compute_rho_sigma(&aux, j_max);
        let modulus = (aux.k * aux.k + aux.h * aux.h).sqrt() / (1.0 + aux.big_b);
        let arg = libm::atan2(aux.h, aux.k) + PI;
        let mut max_rel = 0.0_f64;
        for j in 1..=j_max {
            let jf = j as f64;
            let coef = (1.0 + jf * aux.big_b) * libm::pow(modulus, jf);
            for (got, want) in [
                (rho_sigma[j][0], coef * cos(jf * arg)),
                (rho_sigma[j][1], coef * sin(jf * arg)),
            ] {
                let rel = (got - want).abs() / want.abs();
                assert!(rel < 1e-13, "j={j}: rel {rel:.3e} (got {got:.6e})");
                max_rel = max_rel.max(rel);
            }
        }
        std::println!("ρⱼ/σⱼ vs polar closed form: max rel {max_rel:.3e}");
    }

    /// The two longitude conversions are checked against the **classical** anomalies, which they never
    /// touch: with e = √(k² + h²) and ϖ = atan2(h, k) the equinoctial arguments are λ = ϖ + M,
    /// LE = ϖ + E, Lv = ϖ + ν, so
    ///
    /// * `mean_to_eccentric` must satisfy its own defining equation λ = LE − k·sin LE + h·cos LE, and
    ///   must agree with a bisection solve of Kepler's equation M = E − e·sin E (200 halvings of a
    ///   ±1 rad bracket — 2⁻²⁰⁰, i.e. exact in double), which shares no step with the Danby iteration;
    /// * `eccentric_to_true` must agree with tan(ν/2) = √((1+e)/(1−e))·tan(E/2).
    ///
    /// The Orekit comparisons see these only through `center = Lv − λ` and the `cos jLv` arguments,
    /// scaled by short-period amplitudes of 1e-3 or less, so a longitude error of ~1e-9 rad is below
    /// their thresholds. Swept over 97 mean longitudes and over e up to 0.99: at the LEO e = 0.01 of
    /// the fixtures the Danby step converges in two iterations from almost any update rule, and only
    /// an eccentric orbit separates a second-order step from a merely convergent one.
    #[test]
    fn longitude_conversions_match_the_classical_anomalies() {
        let (mut max_res, mut max_ecc_anom, mut max_true) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &ecc in &[
            sqrt(MEAN[0][1] * MEAN[0][1] + MEAN[0][2] * MEAN[0][2]),
            0.1,
            0.3,
            0.6,
            0.9,
            0.99,
        ] {
            for &varpi in &[libm::atan2(MEAN[0][2], MEAN[0][1]), 2.0] {
                let (k, h) = (ecc * cos(varpi), ecc * sin(varpi));
                for g in 0..97u16 {
                    let lambda = -PI + TAU * f64::from(g) / 97.0;
                    let le = mean_to_eccentric(k, h, lambda);
                    max_res = max_res.max(((le - k * sin(le) + h * cos(le)) - lambda).abs());

                    let mean_anom = lambda - varpi;
                    let (mut lo, mut hi) = (mean_anom - 1.2, mean_anom + 1.2);
                    for _ in 0..200 {
                        let mid = 0.5 * (lo + hi);
                        if mid - ecc * sin(mid) < mean_anom {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let ecc_anom = 0.5 * (lo + hi);
                    max_ecc_anom = max_ecc_anom.max((le - (varpi + ecc_anom)).abs());

                    let true_anom = 2.0
                        * libm::atan2(
                            sqrt((1.0 + ecc) / (1.0 - ecc)) * sin(0.5 * ecc_anom),
                            cos(0.5 * ecc_anom),
                        );
                    let lv = eccentric_to_true(k, h, le);
                    max_true = max_true.max(wrap(lv - (varpi + true_anom)).abs());
                }
            }
        }
        assert!(max_res < 1e-15, "Kepler residual {max_res:.3e} rad");
        assert!(
            max_ecc_anom < 3e-15,
            "LE vs bisection {max_ecc_anom:.3e} rad"
        );
        // The tan-half-angle oracle carries a √((1+e)/(1−e)) that reaches 14 at e = 0.99, so its own
        // round-off — not the conversion's — sets this bound.
        assert!(max_true < 1e-14, "Lv vs classical ν {max_true:.3e} rad");
        std::println!(
            "longitudes vs classical anomalies: Kepler residual {max_res:.3e}, LE {max_ecc_anom:.3e}, Lv {max_true:.3e} rad"
        );
    }

    /// A zonal field is invariant under rotation about the pole, so the generator of that rotation
    /// annihilates every Fourier coefficient. In equinoctial variables the generator is
    /// `h·∂/∂k − k·∂/∂h − (α·∂/∂β − β·∂/∂α) − ∂/∂L`, giving the pair of identities
    ///
    /// * `h·∂Cʲ/∂k − k·∂Cʲ/∂h − α·∂Cʲ/∂β + β·∂Cʲ/∂α − j·Sʲ = 0`,
    /// * `h·∂Sʲ/∂k − k·∂Sʲ/∂h − α·∂Sʲ/∂β + β·∂Sʲ/∂α + j·Cʲ = 0`.
    ///
    /// These are exactly `coef4` and `coef5` of `compute_cij_sij`, which is why the `q`/`p`
    /// short-period corrections reduce to their `−I·Cʲ_αγ` / `−Cʲ_βγ` parts. Nothing else in the
    /// suite constrains the four derivative rows `∂/∂h`, `∂/∂k`, `∂/∂α`, `∂/∂β` against the value
    /// row: the Orekit comparisons see only their weighted sum.
    #[test]
    fn fourier_coefficients_are_annihilated_by_the_pole_rotation() {
        let z = zonal();
        let aux = aux_at(MEAN[0]);
        let (n_max, s_max, j_max) = (6usize, 4usize, 13usize);
        let ctx = Ctx::new(&aux, AE / aux.sma);
        let (cc, sc) = fourier_cj_sj(&ctx, n_max, s_max, j_max, &z.c_n0, z.vns());
        let mut max_rel = 0.0_f64;
        for j in 1..=j_max {
            let jf = j as f64;
            for (row, other, sign) in [(&cc[j], &sc[j], -1.0), (&sc[j], &cc[j], 1.0)] {
                let hk = aux.h * row[3] - aux.k * row[2];
                let ab = aux.alpha * row[5] - aux.beta * row[4];
                let fast = sign * jf * other[0];
                let residual = hk - ab + fast;
                let scale = hk.abs().max(ab.abs()).max(fast.abs());
                let rel = if scale == 0.0 {
                    residual.abs()
                } else {
                    residual.abs() / scale
                };
                assert!(rel < 1e-14, "j={j}: rotation residual {rel:.3e}");
                max_rel = max_rel.max(rel);
            }
        }
        std::println!("Cʲ/Sʲ pole-rotation residual: max rel {max_rel:.3e}");
    }

    /// The true-longitude frequency content of the coefficient arrays. A degree-`n` zonal term
    /// carrying eccentricity power `s` can reach the harmonic `n + s`, and Danielson's own bound on
    /// the Cʲ series is `2N − 1`, so `Cʲ` and `Sʲ` are non-zero for exactly
    /// `1 ≤ j ≤ min(2N − 1, N + sMax)` and identically zero above. The Cᵢⱼ/Sᵢⱼ assembly (4.1-(20)…(25))
    /// couples `Cʲ⁻²…Cʲ⁺²`, shifting the top harmonic up by exactly two.
    ///
    /// Swept over degrees 2…6 so both truncation regimes appear: `sMax = N − 1` binds below degree 6
    /// and the eccentricity-power clamp `sMax = 4` binds at and above it. Every existing check
    /// weights the harmonics by `eʲ`, so at e = 0.01 everything past `j ≈ 4` is already below the
    /// round-off of the sum and no accuracy assertion can see whether it was computed at all.
    #[test]
    #[allow(clippy::float_cmp)] // structural test: a harmonic is present or exactly absent
    fn coefficient_arrays_have_the_expected_frequency_content() {
        let aux = aux_at(MEAN[0]);
        let ctx = Ctx::new(&aux, AE / aux.sma);
        for n_max in 2..=6usize {
            let z = DsstZonal::new(
                MU,
                AE,
                std::vec![0.0, 0.0, C20, C30, C40, C50, C60][..=n_max].to_vec(),
            );
            let (s_max, j_max) = (4.min(n_max - 1), 2 * n_max + 1);
            let j_top = (2 * n_max - 1).min(n_max + s_max);

            let (cc, sc) = fourier_cj_sj(&ctx, n_max, s_max, j_max, &z.c_n0, z.vns());
            for j in 1..=j_max {
                let present = cc[j][0] != 0.0 || sc[j][0] != 0.0;
                assert_eq!(present, j <= j_top, "N={n_max}: Cʲ at j={j}");
            }

            let udu = z.potential_derivatives(&aux, z.max_ecc_pow(&aux));
            let rho_sigma = compute_rho_sigma(&aux, j_max);
            let (cij, sij) = compute_cij_sij(&ctx, &cc, &sc, &rho_sigma, udu.u, n_max, j_max);
            for j in 1..=j_max {
                let present = (0..6).any(|i| cij[j][i] != 0.0 || sij[j][i] != 0.0);
                assert_eq!(present, j <= j_top + 2, "N={n_max}: Cᵢⱼ at j={j}");
            }
        }
        std::println!("frequency content matches min(2N−1, N+sMax) (+2 for Cᵢⱼ) for N = 2…6");
    }
}
