//! The DSST **third-body** (Sun / Moon) short-period corrections (Danielson §4.2), turning DSST mean
//! equinoctial elements into osculating ones — the third-body analogue of the [`super::short_period`]
//! zonal port.
//!
//! Direct port of Orekit's `DSSTThirdBody` short-period machinery, mirroring its inner-class split:
//! * `GnsCoefficients` — Gₙₛ = (2 − δ₀ₛ)·(a/R₃)ⁿ·Vₙₛ·Qₙˢ·µ₃/R₃ and its a/γ derivatives (Danielson
//!   4.2-17);
//! * `WnsjEtomjmsCoefficient` — e^{−|j−s|}·wⱼ^{n,s}(χ) and its h/k derivatives (Danielson 4.2-9,10,11),
//!   built on the Jacobi polynomial Pₗ^{|j−s|,|j+s|}(χ);
//! * `CjSjAlphaBetaKH` — the Aⱼₛ/Bⱼₛ/Dⱼₛ/Eⱼₛ terms in Cⱼ/Sⱼ of (α, β) and (k, h);
//! * `FourierCjSjCoefficients` — the Fourier Cʲ/Sʲ of the disturbing function and their derivatives
//!   (Danielson 4.2-14,15) plus the Cʲ,λ/Sʲ,λ terms (4.2-21);
//! * `GeneratingFunctionCoefficients` — the generating function S = C⁰ + Σ(Cʲ cos jF + Sʲ sin jF) and
//!   its a/k/h/α/β/γ/λ derivatives (with the j = 1 correction C¹ = C¹_F − hU, S¹ = S¹_F + kU);
//! * `ThirdBodyShortPeriodicCoefficients` — the `value(meanOrbit)` assembly turning the generating
//!   function into the six equinoctial short-period Cᵢⱼ/Sᵢⱼ and the Fourier sum over the **eccentric**
//!   longitude F.
//!
//! Variable names follow Orekit for auditability.
//!
//! # What differs from the zonal short-period
//!
//! * The Fourier series is in the **eccentric** longitude F = Lₑ (`meanOrbit.getLE()`), not the true
//!   longitude Lᵥ the zonal contribution uses.
//! * There is an intermediate **generating function** layer (`GeneratingFunctionCoefficients` over the
//!   Fourier `FourierCjSjCoefficients`); the zonal builds Cᵢⱼ/Sᵢⱼ directly.
//! * The eccentricity sum bound for the Fourier/`Gₙₛ` coefficients is fixed at `MAX_ECCPOWER_SP = 4`,
//!   while the disturbing-potential value U (only needed by the j = 1 term) uses the mean-element
//!   `maxEccPow` — exactly Orekit's split.
//! * The radial factor is `(a/R₃)ⁿ` and the coefficients carry `µ₃/R₃`, reusing the validated
//!   third-body potential `DsstThirdBody::potential_derivatives` and Vₙₛ table.
//!
//! # What is *not* ported
//!
//! As in the zonal port, Orekit's `ShortPeriodicsInterpolatedCoefficient` (Hermite interpolation of the
//! coefficients between integration-step grid points) is a propagator detail: this module evaluates the
//! coefficients directly at the output mean state, which the interpolation approximates and is exact at
//! every grid point. The oracle therefore compares against Orekit's `DSSTPropagator` OSCULATING output
//! seeded from the identical mean state.
//!
//! # The defining equation, checked
//!
//! `tests/dsst_short_period_identity.rs` checks the corrections against n·∂η/∂λ = F − ⟨F⟩. Here F is
//! Gauss's rate from the point-mass tide of a body held fixed over the revolution, as the DSST holds
//! it. The residual is 2.4e-5 in LEO and 1.8e-3 in GEO, where a/R₃ is 0.11. With the tide held
//! fixed, each halving of a/R₃ divides the residual by about 8: it is the distance cut, not an error.

use std::vec::Vec;

use libm::{cos, fabs, sin, sqrt};

use super::auxiliary::{DsstAuxiliary, I};
use super::coefficients::{compute_qns, factorial, jacobi_value_deriv};
use super::third_body::{DsstThirdBody, ThirdBodyPotential};

/// Convergence tolerance for the equinoctial Kepler solve (Orekit
/// `EquinoctialLongitudeArgumentUtility.TOLERANCE_CONVERGENCE`).
const KEPLER_TOLERANCE: f64 = 1.0e-11;
/// Maximum iterations for the equinoctial Kepler solve (Orekit `MAXIMUM_ITERATION`).
const KEPLER_MAX_ITER: usize = 50;
/// Maximum eccentricity power used by the short-period Fourier/`Gₙₛ` sums (`MAX_ECCPOWER_SP`).
const MAX_ECCPOWER_SP: usize = 4;

/// Cⱼ(x, y) + i·Sⱼ(x, y) = (x + i·y)ʲ (Danielson 2.5.3-(5)) with the partial derivatives w.r.t. the
/// first argument (`_dk`) and the second (`_dh`). Orekit's `CjSjCoefficient` (whose `getDcjDk` etc.
/// name the derivatives for the primary (k, h) use; for the (α, β) instance `_dk` = ∂/∂α, `_dh` = ∂/∂β).
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
    fn dcj_dk(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            j as f64 * self.cj[j - 1]
        }
    }
    fn dsj_dk(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            j as f64 * self.sj[j - 1]
        }
    }
    fn dcj_dh(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            -(j as f64) * self.sj[j - 1]
        }
    }
    fn dsj_dh(&self, j: usize) -> f64 {
        if j == 0 {
            0.0
        } else {
            j as f64 * self.cj[j - 1]
        }
    }
}

/// The mean-element-derived scalar factors of `DSSTThirdBodyDynamicContext` used by the short-period
/// assembly. Holds a reference to the auxiliary elements plus the third-body direction cosines and
/// derived combinations.
struct Ctx<'a> {
    aux: &'a DsstAuxiliary,
    /// Direction cosine α = body direction · f.
    alpha: f64,
    /// Direction cosine β = body direction · g.
    beta: f64,
    /// Direction cosine γ = body direction · w.
    gamma: f64,
    /// b = 1 / (1 + B).
    b: f64,
    /// µ₃ / R₃.
    muo_r3: f64,
    /// h · χ³.
    hxxx: f64,
    /// k · χ³.
    kxxx: f64,
}

impl<'a> Ctx<'a> {
    fn new(aux: &'a DsstAuxiliary, alpha: f64, beta: f64, gamma: f64, muo_r3: f64) -> Ctx<'a> {
        Ctx {
            aux,
            alpha,
            beta,
            gamma,
            b: 1.0 / (1.0 + aux.big_b),
            muo_r3,
            hxxx: aux.h * aux.chi3,
            kxxx: aux.k * aux.chi3,
        }
    }
}

/// The Gₙₛ coefficients and their a/γ derivatives (Danielson 4.2-17). Dense `[n][s]` over
/// `n = 0..=n_max`, `s = 0..=s_max`; the odd-parity and `n < max(2, s)` entries are zero. Orekit's
/// `GnsCoefficients`.
struct Gns {
    gns: Vec<Vec<f64>>,
    dgnsda: Vec<Vec<f64>>,
    dgnsdgamma: Vec<Vec<f64>>,
}

impl Gns {
    fn new(
        ctx: &Ctx,
        n_max: usize,
        s_max: usize,
        aor3_pow: &[f64],
        qns: &[Vec<f64>],
        vns: &[Vec<f64>],
    ) -> Self {
        let sma = ctx.aux.sma;
        let muo_r3 = ctx.muo_r3;
        let mut gns = std::vec![std::vec![0.0; s_max + 1]; n_max + 1];
        let mut dgnsda = std::vec![std::vec![0.0; s_max + 1]; n_max + 1];
        let mut dgnsdgamma = std::vec![std::vec![0.0; s_max + 1]; n_max + 1];
        for s in 0..=s_max {
            for n in s.max(2)..=n_max {
                if (n - s) % 2 == 0 {
                    let delta0s = if s == 0 { 1.0 } else { 2.0 };
                    let coef0 = delta0s * aor3_pow[n] * vns[n][s] * muo_r3;
                    let coef1 = coef0 * qns[n][s];
                    // dQns/dGamma = Q(n, s + 1); for n = s, Q(n, n + 1) = 0 (Cefola & Broucke 1975).
                    let dqns = if n == s { 0.0 } else { qns[n][s + 1] };
                    gns[n][s] = coef1;
                    dgnsda[n][s] = coef1 * n as f64 / sma;
                    dgnsdgamma[n][s] = coef0 * dqns;
                }
            }
        }
        Self {
            gns,
            dgnsda,
            dgnsdgamma,
        }
    }

    fn gns(&self, n: usize, s: usize) -> f64 {
        self.gns[n][s]
    }
    fn dgnsda(&self, n: usize, s: usize) -> f64 {
        self.dgnsda[n][s]
    }
    fn dgnsdgamma(&self, n: usize, s: usize) -> f64 {
        self.dgnsdgamma[n][s]
    }
}

/// The coefficient e^{−|j−s|}·wⱼ^{n,s}(χ) and its (k, h) derivatives (Danielson 4.2-9,10,11), Orekit's
/// `WnsjEtomjmsCoefficient`. Precomputes the c/b powers and derivatives at the mean state; `compute`
/// evaluates one (j, s, n) term (with the Jacobi polynomial) returning `[value, ∂/∂k, ∂/∂h]`.
struct Wnsj {
    /// c = e·b.
    c: f64,
    db_dh: f64,
    db_dk: f64,
    dc_dh: f64,
    dc_dk: f64,
    /// (1 − c²)ⁿ.
    omc2tn: Vec<f64>,
    /// (1 + c²)ⁿ.
    opc2tn: Vec<f64>,
    /// b^{|j−s|}.
    btjms: Vec<f64>,
    chi: f64,
    hxxx: f64,
    kxxx: f64,
}

impl Wnsj {
    fn new(ctx: &Ctx, max_ar3_pow: usize, max_freq_f: usize) -> Self {
        let aux = ctx.aux;
        let b = ctx.b;
        let ecc = aux.ecc;
        let c = ecc * b;
        let c2 = c * c;

        // b² · χ.
        let b2chi = b * b * aux.chi;
        let db_dh = aux.h * b2chi;
        let db_dk = aux.k * b2chi;

        // dc/dh, dc/dk (with the perfectly-circular singularity handled as Orekit does).
        let (dc_dh, dc_dk) = if ecc == 0.0 {
            (ecc * db_dh + b, ecc * db_dk)
        } else {
            (ecc * db_dh + b * aux.h / ecc, ecc * db_dk + b * aux.k / ecc)
        };

        let hi = max_ar3_pow + max_freq_f + 1;
        let mut omc2tn = std::vec![0.0; hi + 1];
        let mut opc2tn = std::vec![0.0; hi + 1];
        let omc2 = 1.0 - c2;
        let opc2 = 1.0 + c2;
        omc2tn[0] = 1.0;
        opc2tn[0] = 1.0;
        for i in 1..=hi {
            omc2tn[i] = omc2tn[i - 1] * omc2;
            opc2tn[i] = opc2tn[i - 1] * opc2;
        }

        let bhi = max_ar3_pow + max_freq_f;
        let mut btjms = std::vec![0.0; bhi + 1];
        btjms[0] = 1.0;
        for i in 1..=bhi {
            btjms[i] = btjms[i - 1] * b;
        }

        Self {
            c,
            db_dh,
            db_dk,
            dc_dh,
            dc_dk,
            omc2tn,
            opc2tn,
            btjms,
            chi: aux.chi,
            hxxx: ctx.hxxx,
            kxxx: ctx.kxxx,
        }
    }

    /// Compute e^{−|j−s|}·wⱼ^{n,s} and its (k, h) derivatives for one term, `[value, ∂/∂k, ∂/∂h]`
    /// (Orekit `computeWjnsEmjmsAndDeriv`). `j` may be negative; `s ≥ 0`; `n = n_arg` is the (already
    /// shifted) upper index the caller passes (`n+1` for the main terms, `n` for the λ terms).
    fn compute(&self, j: i64, s: i64, n_arg: i64) -> [f64; 3] {
        let abs_j = j.abs();
        let abs_s = s.abs();
        let abs_jms = (j - s).abs();
        let abs_jps = (j + s).abs();

        // Lower index of P (also the power of 1 − c²) and the factorial-ratio coefficient.
        let (fact_coef, l) = if abs_s > abs_j {
            let fc = (factorial(n_arg + s) / factorial(n_arg + j))
                * (factorial(n_arg - s) / factorial(n_arg - j));
            (fc, n_arg - abs_s)
        } else {
            (1.0, n_arg - abs_j)
        };
        let l = l as usize;

        let sign = if abs_jms % 2 != 0 { -1.0 } else { 1.0 };
        let coef1 = self.omc2tn[l] / self.opc2tn[n_arg as usize];
        let coef2 = sign * self.btjms[abs_jms as usize];
        let (jacobi, jac_der) = jacobi_value_deriv(l, abs_jms, abs_jps, self.chi);

        // ∂coef1/∂c and its h/k propagation.
        let dcoef1dc =
            -coef1 * 2.0 * self.c * (n_arg as f64 / self.opc2tn[1] + l as f64 / self.omc2tn[1]);
        let dcoef1dh = dcoef1dc * self.dc_dh;
        let dcoef1dk = dcoef1dc * self.dc_dk;

        // ∂coef2/∂b and its h/k propagation.
        let dcoef2db = if abs_jms == 0 {
            0.0
        } else {
            sign * abs_jms as f64 * self.btjms[(abs_jms - 1) as usize]
        };
        let dcoef2dh = dcoef2db * self.db_dh;
        let dcoef2dk = dcoef2db * self.db_dk;

        let djacobidh = jac_der * self.hxxx;
        let djacobidk = jac_der * self.kxxx;

        let term1 = fact_coef * coef1 * coef2;
        let term2 = fact_coef * coef1 * jacobi;
        let term3 = fact_coef * coef2 * jacobi;

        [
            term1 * jacobi,
            dcoef1dk * term3 + dcoef2dk * term2 + djacobidk * term1,
            dcoef1dh * term3 + dcoef2dh * term2 + djacobidh * term1,
        ]
    }
}

/// The Aⱼₛ, Bⱼₛ, Dⱼₛ, Eⱼₛ terms in the Cⱼ/Sⱼ series of (α, β) and (k, h) and their (k, h, α, β)
/// derivatives (Orekit `CjSjAlphaBetaKH`). `compute` returns `(A, B, D, E)`, each a 5-vector
/// `[value, ∂/∂k, ∂/∂h, ∂/∂α, ∂/∂β]`.
struct Abde {
    cjsjkh: CjSj,
    cjsjalbe: CjSj,
}

impl Abde {
    fn new(ctx: &Ctx, kh_max: usize, ab_max: usize) -> Self {
        Self {
            cjsjkh: CjSj::new(ctx.aux.k, ctx.aux.h, kh_max),
            cjsjalbe: CjSj::new(ctx.alpha, ctx.beta, ab_max),
        }
    }

    fn compute(&self, j: i64, s: usize) -> ([f64; 5], [f64; 5], [f64; 5], [f64; 5]) {
        let sign = if j < s as i64 { -1.0 } else { 1.0 };
        let m = (j - s as i64).unsigned_abs() as usize;
        let jps = (j + s as i64) as usize;
        let kh = &self.cjsjkh;
        let ab = &self.cjsjalbe;

        let a = [
            sign * ab.cj(s) * kh.sj(m) + ab.sj(s) * kh.cj(m),
            sign * ab.cj(s) * kh.dsj_dk(m) + ab.sj(s) * kh.dcj_dk(m),
            sign * ab.cj(s) * kh.dsj_dh(m) + ab.sj(s) * kh.dcj_dh(m),
            sign * ab.dcj_dk(s) * kh.sj(m) + ab.dsj_dk(s) * kh.cj(m),
            sign * ab.dcj_dh(s) * kh.sj(m) + ab.dsj_dh(s) * kh.cj(m),
        ];
        let b = [
            ab.cj(s) * kh.sj(jps) - ab.sj(s) * kh.cj(jps),
            ab.cj(s) * kh.dsj_dk(jps) - ab.sj(s) * kh.dcj_dk(jps),
            ab.cj(s) * kh.dsj_dh(jps) - ab.sj(s) * kh.dcj_dh(jps),
            ab.dcj_dk(s) * kh.sj(jps) - ab.dsj_dk(s) * kh.cj(jps),
            ab.dcj_dh(s) * kh.sj(jps) - ab.dsj_dh(s) * kh.cj(jps),
        ];
        let d = [
            ab.cj(s) * kh.cj(m) - sign * ab.sj(s) * kh.sj(m),
            ab.cj(s) * kh.dcj_dk(m) - sign * ab.sj(s) * kh.dsj_dk(m),
            ab.cj(s) * kh.dcj_dh(m) - sign * ab.sj(s) * kh.dsj_dh(m),
            ab.dcj_dk(s) * kh.cj(m) - sign * ab.dsj_dk(s) * kh.sj(m),
            ab.dcj_dh(s) * kh.cj(m) - sign * ab.dsj_dh(s) * kh.sj(m),
        ];
        let e = [
            ab.cj(s) * kh.cj(jps) + ab.sj(s) * kh.sj(jps),
            ab.cj(s) * kh.dcj_dk(jps) + ab.sj(s) * kh.dsj_dk(jps),
            ab.cj(s) * kh.dcj_dh(jps) + ab.sj(s) * kh.dsj_dh(jps),
            ab.dcj_dk(s) * kh.cj(jps) + ab.dsj_dk(s) * kh.sj(jps),
            ab.dcj_dh(s) * kh.cj(jps) + ab.dsj_dh(s) * kh.sj(jps),
        ];
        (a, b, d, e)
    }
}

/// The Fourier Cʲ, Sʲ coefficients and their (a, k, h, α, β, γ) derivatives (Danielson 4.2-14,15),
/// as `cj[j][row]` / `sj[j][row]` with `row` = 0:value, 1:∂/∂a, 2:∂/∂k, 3:∂/∂h, 4:∂/∂α, 5:∂/∂β,
/// 6:∂/∂γ, plus the Cʲ,λ/Sʲ,λ coefficients (Danielson 4.2-21). Orekit's `FourierCjSjCoefficients`.
struct Fourier {
    cj: Vec<[f64; 7]>,
    sj: Vec<[f64; 7]>,
    cjlambda: Vec<f64>,
    sjlambda: Vec<f64>,
    jmax: usize,
}

impl Fourier {
    fn cjlambda(&self, j: i64) -> f64 {
        if j < 1 || j >= self.jmax as i64 {
            0.0
        } else {
            self.cjlambda[j as usize]
        }
    }
    fn sjlambda(&self, j: i64) -> f64 {
        if j < 1 || j >= self.jmax as i64 {
            0.0
        } else {
            self.sjlambda[j as usize]
        }
    }
    fn c0lambda(&self) -> f64 {
        self.cjlambda[0]
    }
}

/// Build the Fourier Cʲ/Sʲ coefficients (Orekit `FourierCjSjCoefficients.computeCoefficients`).
/// `n_max = maxAR3Pow`, `s_max = MAX_ECCPOWER_SP`, `j_max = maxAR3Pow + 1`.
fn fourier(
    ctx: &Ctx,
    n_max: usize,
    s_max: usize,
    j_max: usize,
    gns: &Gns,
    wnsj: &Wnsj,
    abde: &Abde,
) -> Fourier {
    let mut cj = std::vec![[0.0f64; 7]; j_max + 1];
    let mut sj = std::vec![[0.0f64; 7]; j_max + 1];
    let mut cjlambda = std::vec![0.0f64; j_max + 1];
    let mut sjlambda = std::vec![0.0f64; j_max + 1];

    for j in 1..=j_max {
        let ji = j as i64;
        let (mut crow, mut srow) = ([0.0f64; 7], [0.0f64; 7]);
        let (mut cjl, mut sjl) = (0.0f64, 0.0f64);

        for s in 0..=s_max {
            let (a, b, d, e) = abde.compute(ji, s);
            let si = s as i64;
            let min_n = 2.max((j - 1).max(s));
            for n in min_n..=n_max {
                if (n - s) % 2 == 0 {
                    let wjnp1 = wnsj.compute(ji, si, (n + 1) as i64);
                    let wmjnp1 = wnsj.compute(-ji, si, (n + 1) as i64);
                    let gns_v = gns.gns(n, s);
                    let dgnsda_v = gns.dgnsda(n, s);
                    let dgnsdg_v = gns.dgnsdgamma(n, s);
                    let coef1 = -(wjnp1[0] * a[0] + wmjnp1[0] * b[0]);
                    let coef2 = wjnp1[0] * d[0] + wmjnp1[0] * e[0];

                    crow[0] += gns_v * coef1;
                    crow[1] += dgnsda_v * coef1;
                    crow[2] += -gns_v
                        * (wjnp1[1] * a[0] + wjnp1[0] * a[1] + wmjnp1[1] * b[0] + wmjnp1[0] * b[1]);
                    crow[3] += -gns_v
                        * (wjnp1[2] * a[0] + wjnp1[0] * a[2] + wmjnp1[2] * b[0] + wmjnp1[0] * b[2]);
                    crow[4] += -gns_v * (wjnp1[0] * a[3] + wmjnp1[0] * b[3]);
                    crow[5] += -gns_v * (wjnp1[0] * a[4] + wmjnp1[0] * b[4]);
                    crow[6] += dgnsdg_v * coef1;

                    srow[0] += gns_v * coef2;
                    srow[1] += dgnsda_v * coef2;
                    srow[2] += gns_v
                        * (wjnp1[1] * d[0] + wjnp1[0] * d[1] + wmjnp1[1] * e[0] + wmjnp1[0] * e[1]);
                    srow[3] += gns_v
                        * (wjnp1[2] * d[0] + wjnp1[0] * d[2] + wmjnp1[2] * e[0] + wmjnp1[0] * e[2]);
                    srow[4] += gns_v * (wjnp1[0] * d[3] + wmjnp1[0] * e[3]);
                    srow[5] += gns_v * (wjnp1[0] * d[4] + wmjnp1[0] * e[4]);
                    srow[6] += dgnsdg_v * coef2;

                    if n >= j && j < j_max {
                        let wjns = wnsj.compute(ji, si, n as i64);
                        let wmjns = wnsj.compute(-ji, si, n as i64);
                        cjl += gns_v * (wjns[0] * d[0] + wmjns[0] * e[0]);
                        sjl += gns_v * (wjns[0] * a[0] + wmjns[0] * b[0]);
                    }
                }
            }
        }

        let jf = j as f64;
        for i in 0..7 {
            crow[i] /= jf;
            srow[i] /= jf;
        }
        cj[j] = crow;
        sj[j] = srow;
        if j < j_max {
            cjlambda[j] = cjl;
            sjlambda[j] = sjl;
        }
    }

    // C⁰,λ (Danielson 4.2-21).
    cjlambda[0] = ctx.aux.k * cjlambda[1] / 2.0 + ctx.aux.h * sjlambda[1] / 2.0;

    Fourier {
        cj,
        sj,
        cjlambda,
        sjlambda,
        jmax: j_max,
    }
}

/// The generating function S coefficients and their (a, k, h, α, β, γ, λ) derivatives (Orekit
/// `GeneratingFunctionCoefficients`). Returns `(cj_coefs, sj_coefs)` as `[j][row]` with `row` =
/// 0:S, 1:∂/∂a, 2:∂/∂k, 3:∂/∂h, 4:∂/∂α, 5:∂/∂β, 6:∂/∂γ, 7:∂/∂λ.
fn generating_function(
    fourier: &Fourier,
    udu: &ThirdBodyPotential,
    aux: &DsstAuxiliary,
    j_max: usize,
) -> (Vec<[f64; 8]>, Vec<[f64; 8]>) {
    let (k, h) = (aux.k, aux.h);
    let mut cj_coefs = std::vec![[0.0f64; 8]; j_max + 1];
    let mut sj_coefs = std::vec![[0.0f64; 8]; j_max + 1];

    for j in 1..=j_max {
        let ji = j as i64;
        let cc = &fourier.cj[j];
        let ss = &fourier.sj[j];

        cj_coefs[j][0] = cc[0];
        cj_coefs[j][1] = cc[1];
        cj_coefs[j][2] = cc[2] - (fourier.sjlambda(ji - 1) - fourier.sjlambda(ji + 1)) / 2.0;
        cj_coefs[j][3] = cc[3] - (fourier.cjlambda(ji - 1) + fourier.cjlambda(ji + 1)) / 2.0;
        cj_coefs[j][4] = cc[4];
        cj_coefs[j][5] = cc[5];
        cj_coefs[j][6] = cc[6];
        cj_coefs[j][7] = fourier.cjlambda(ji);

        sj_coefs[j][0] = ss[0];
        sj_coefs[j][1] = ss[1];
        sj_coefs[j][2] = ss[2] + (fourier.cjlambda(ji - 1) - fourier.cjlambda(ji + 1)) / 2.0;
        sj_coefs[j][3] = ss[3] - (fourier.sjlambda(ji - 1) + fourier.sjlambda(ji + 1)) / 2.0;
        sj_coefs[j][4] = ss[4];
        sj_coefs[j][5] = ss[5];
        sj_coefs[j][6] = ss[6];
        sj_coefs[j][7] = fourier.sjlambda(ji);

        // The j = 1 case carries the extra C¹ = C¹_F − hU, S¹ = S¹_F + kU terms (Danielson 4.2-19,20).
        if j == 1 {
            let u = udu.u;
            let c0l = fourier.c0lambda();
            cj_coefs[1][0] += -h * u;
            cj_coefs[1][1] += -h * udu.du_da;
            cj_coefs[1][2] += -h * udu.du_dk;
            cj_coefs[1][3] += -(h * udu.du_dh + u + c0l);
            cj_coefs[1][4] += -h * udu.du_dal;
            cj_coefs[1][5] += -h * udu.du_dbe;
            cj_coefs[1][6] += -h * udu.du_dga;

            sj_coefs[1][0] += k * u;
            sj_coefs[1][1] += k * udu.du_da;
            sj_coefs[1][2] += k * udu.du_dk + u + c0l;
            sj_coefs[1][3] += k * udu.du_dh;
            sj_coefs[1][4] += k * udu.du_dal;
            sj_coefs[1][5] += k * udu.du_dbe;
            sj_coefs[1][6] += k * udu.du_dga;
        }
    }

    (cj_coefs, sj_coefs)
}

/// The six equinoctial short-period Cᵢⱼ/Sᵢⱼ coefficients from the generating function (Orekit
/// `updateShortPeriodTerms`, Danielson 2.5.5-10). Returns `(cij, sij)` as `[j][elem]` with `elem`
/// 0:a, 1:k, 2:h, 3:q, 4:p, 5:λ; `cij[0]` is the C⁰ free coefficient (Danielson 2.5.2-15a).
fn compute_cij_sij(
    ctx: &Ctx,
    cj_coefs: &[[f64; 8]],
    sj_coefs: &[[f64; 8]],
    j_max: usize,
) -> (Vec<[f64; 6]>, Vec<[f64; 6]>) {
    let aux = ctx.aux;
    let n = aux.n;
    let (alpha, beta, gamma) = (ctx.alpha, ctx.beta, ctx.gamma);
    let (k, h, p, q) = (aux.k, aux.h, aux.p, aux.q);
    let ax2oan = aux.ax2oa / n;
    let boan = aux.boa / n;
    let ooabn = aux.ooab / n;
    let co2abn = aux.co2ab / n;
    let boabpon = aux.boabpo / n;
    let m3ona = -3.0 / (aux.big_a * n);

    let mut cij = std::vec![[0.0f64; 6]; j_max + 1];
    let mut sij = std::vec![[0.0f64; 6]; j_max + 1];

    for j in 1..=j_max {
        let cur_c = one_element_row(
            &cj_coefs[j],
            alpha,
            beta,
            gamma,
            p,
            q,
            h,
            k,
            ax2oan,
            boan,
            ooabn,
            co2abn,
            boabpon,
            m3ona,
        );
        let cur_s = one_element_row(
            &sj_coefs[j],
            alpha,
            beta,
            gamma,
            p,
            q,
            h,
            k,
            ax2oan,
            boan,
            ooabn,
            co2abn,
            boabpon,
            m3ona,
        );
        cij[j] = cur_c;
        sij[j] = cur_s;

        // C⁰ free coefficient from the j = 1 Cᵢ/Sᵢ (Danielson 2.5.2-15a).
        if j == 1 {
            let mut c0 = [0.0f64; 6];
            for i in 0..6 {
                c0[i] = cur_c[i] * k / 2.0 + cur_s[i] * h / 2.0;
            }
            cij[0] = c0;
        }
    }

    (cij, sij)
}

/// One Cᵢⱼ (or Sᵢⱼ) six-element row from a generating-function coefficient column `sd` (rows
/// 0:S, 1:∂a, 2:∂k, 3:∂h, 4:∂α, 5:∂β, 6:∂γ, 7:∂λ). Orekit `updateShortPeriodTerms`, the `currentCij`
/// / `currentSij` block.
#[allow(clippy::too_many_arguments)]
fn one_element_row(
    sd: &[f64; 8],
    alpha: f64,
    beta: f64,
    gamma: f64,
    p: f64,
    q: f64,
    h: f64,
    k: f64,
    ax2oan: f64,
    boan: f64,
    ooabn: f64,
    co2abn: f64,
    boabpon: f64,
    m3ona: f64,
) -> [f64; 6] {
    // Cross derivatives [Danielson 2.2-(8)] of the generating function.
    let s_alpha_gamma = alpha * sd[6] - gamma * sd[4];
    let s_alpha_beta = alpha * sd[5] - beta * sd[4];
    let s_beta_gamma = beta * sd[6] - gamma * sd[5];
    let shk = h * sd[2] - k * sd[3];
    let psag = (p * s_alpha_gamma - I * q * s_beta_gamma) * ooabn;
    let shk_m = shk - s_alpha_beta - sd[7];

    [
        ax2oan * sd[7],
        -(boan * sd[3] + h * psag + k * boabpon * sd[7]),
        boan * sd[2] + k * psag - h * boabpon * sd[7],
        co2abn * (q * shk_m - I * s_alpha_gamma),
        co2abn * (p * shk_m - s_beta_gamma),
        -ax2oan * sd[1] + boabpon * (h * sd[3] + k * sd[2]) + psag + m3ona * sd[0],
    ]
}

/// Mean → eccentric longitude argument via the equinoctial Kepler equation
/// λ = LE − k·sin(LE) + h·cos(LE), solved with Orekit's Danby-style step (`meanToEccentric`). Third
/// bodies expand the short-period series in the eccentric longitude F = LE (`meanOrbit.getLE()`).
///
/// # Panics
///
/// If the iteration does not converge within [`KEPLER_MAX_ITER`] (only for e ≈ 1, outside the
/// third-body LEO regime this module targets).
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

/// Euclidean norm of a 3-vector.
fn norm(v: [f64; 3]) -> f64 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// Dot product of two 3-vectors.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the six equinoctial short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` from one third body
/// at the given mean state and geocentric inertial body position.
pub(crate) fn short_period_correction(
    tb: &DsstThirdBody,
    aux: &DsstAuxiliary,
    body_pos: [f64; 3],
) -> [f64; 6] {
    let r3 = norm(body_pos);
    let trunc = tb.truncation(aux, r3);
    let n_max = trunc.max_ar3_pow;
    let s_max = MAX_ECCPOWER_SP;
    let j_max = n_max + 1; // maxFreqF
    let max_freq_f = j_max;

    // Direction cosines: normalized body direction dotted with the equinoctial f/g/w vectors.
    let inv_r3 = 1.0 / r3;
    let dir = [
        body_pos[0] * inv_r3,
        body_pos[1] * inv_r3,
        body_pos[2] * inv_r3,
    ];
    let alpha = dot(dir, aux.f_vec);
    let beta = dot(dir, aux.g_vec);
    let gamma = dot(dir, aux.w_vec);

    let ctx = Ctx::new(aux, alpha, beta, gamma, tb.third_body_mu / r3);

    // a / R₃ up to power maxAR3Pow.
    let aor3 = aux.sma / r3;
    let mut aor3_pow = std::vec![0.0; n_max + 1];
    aor3_pow[0] = 1.0;
    for i in 1..=n_max {
        aor3_pow[i] = aor3 * aor3_pow[i - 1];
    }

    // Qns to (maxAR3Pow, max(maxEccPow, MAX_ECCPOWER_SP)) — the extra column feeds dQns/dγ.
    let qns = compute_qns(gamma, n_max, trunc.max_ecc_pow.max(MAX_ECCPOWER_SP));

    let gns = Gns::new(&ctx, n_max, s_max, &aor3_pow, &qns, tb.vns());
    let wnsj = Wnsj::new(&ctx, n_max, max_freq_f);
    let abde = Abde::new(&ctx, j_max + s_max, s_max);
    let fr = fourier(&ctx, n_max, s_max, j_max, &gns, &wnsj, &abde);

    // The disturbing-potential value U (only the j = 1 generating-function term needs it) reuses the
    // validated mean-path UAnddU with the mean-element truncation.
    let udu = tb.potential_derivatives(aux, alpha, beta, gamma, r3, trunc);
    let (cj_coefs, sj_coefs) = generating_function(&fr, &udu, aux, j_max);
    let (cij, sij) = compute_cij_sij(&ctx, &cj_coefs, &sj_coefs, j_max);

    // Evaluate the Fourier series at the eccentric longitude F = LE (Orekit `value`).
    let f = mean_to_eccentric(aux.k, aux.h, aux.lambda);
    let mut sp = cij[0];
    for j in 1..=max_freq_f {
        let jf = j as f64 * f;
        let (s, c) = (sin(jf), cos(jf));
        for i in 0..6 {
            sp[i] += cij[j][i] * c + sij[j][i] * s;
        }
    }
    sp
}

impl DsstThirdBody {
    /// The six equinoctial short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` for this third body at
    /// the given **mean** state and geocentric inertial body position `body_pos` (m), evaluated at the
    /// mean state's eccentric longitude (Orekit's `ThirdBodyShortPeriodicCoefficients.value`).
    #[must_use]
    pub fn short_period_correction(&self, aux: &DsstAuxiliary, body_pos: [f64; 3]) -> [f64; 6] {
        short_period_correction(self, aux, body_pos)
    }

    /// The osculating equinoctial state `mean + short-period` for a **mean** equinoctial state
    /// `[a, k, h, q, p, λ]` and this third body's inertial position `body_pos` (m). Mirrors
    /// [`super::zonal::DsstZonal::osculating`]; for both the Sun and the Moon the caller sums each
    /// body's short-period δ (as with the mean rates).
    #[must_use]
    pub fn osculating(&self, mean: [f64; 6], body_pos: [f64; 3]) -> [f64; 6] {
        let aux = DsstAuxiliary::new(
            mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], self.mu,
        );
        let sp = self.short_period_correction(&aux, body_pos);
        let mut osc = mean;
        for i in 0..6 {
            osc[i] += sp[i];
        }
        osc
    }
}

#[cfg(test)]
mod tests {
    use super::super::third_body::ThirdBodyTruncation;
    use super::*;

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`, orekit_dsst_thirdbody_sp.csv): the same
    // MEAN equinoctial LEO as the third-body mean case (a = 7.2e6, e = 0.01, i = 98°), epoch
    // 2024-01-01T00:00:00 UTC in GCRF. For the Sun alone and for the Sun+Moon, a paired MEAN and
    // OSCULATING DSSTPropagator run (seeded from the same MEAN state) emits, at each t > 0 sample,
    // both the mean equinoctial state and the osculating one plus the body positions — so the port's
    // short-period δ is validated as osc − mean, isolated from mean-propagation error. t = 0 is
    // excluded (a DSST OSCULATING propagator returns the raw mean state there).
    const MU: f64 = 3.98600441500000000e+14;
    const SUN_GM: f64 = 1.32712440041279460e+20;
    const MOON_GM: f64 = 4.90280011845755100e+12;

    const SUN_MAX_AR3: usize = 3;
    const SUN_MAX_ECC: usize = 2;
    const MOON_MAX_AR3: usize = 4;
    const MOON_MAX_ECC: usize = 2;

    const A0: f64 = 7.20000000000000000e+06;
    const EX0: f64 = 3.09016994374947470e-03;
    const EY0: f64 = 9.51056516295153600e-03;
    const HX0: f64 = 8.54890329218965000e-01;
    const HY0: f64 = 7.69746709859868000e-01;
    const LM0: f64 = 1.43116998663535020e+00;

    // Sun-only: mean state (MEAN run) at t = 21600, 43200, 64800, 86400 s.
    #[rustfmt::skip]
    const SUN_MEAN: [[f64; 6]; 4] = [
        [7.20000000000000000e+06, 3.09017050357823220e-03, 9.51056489162198900e-03, 8.54890115711952400e-01, 7.69746043093225600e-01, 2.37526972276385500e+01],
        [7.20000000000000000e+06, 3.09017124541791740e-03, 9.51056451233129800e-03, 8.54889904576584500e-01, 7.69745378215610200e-01, 4.60742244786746300e+01],
        [7.20000000000000000e+06, 3.09017216866464900e-03, 9.51056402592101400e-03, 8.54889695838137900e-01, 7.69744715275154100e-01, 6.83957517397179000e+01],
        [7.20000000000000000e+06, 3.09017327269952130e-03, 9.51056343324162100e-03, 8.54889489521691000e-01, 7.69744054319827200e-01, 9.07172790107418000e+01],
    ];

    // Sun-only: osculating state (OSCULATING run) at the same samples.
    #[rustfmt::skip]
    const SUN_OSC: [[f64; 6]; 4] = [
        [7.20000006999222300e+06, 3.09015805437974940e-03, 9.51052444788596700e-03, 8.54890102673326800e-01, 7.69746032767138300e-01, 2.37526972025170660e+01],
        [7.19999998785736200e+06, 3.09018227759085800e-03, 9.51059249507089700e-03, 8.54889902154280500e-01, 7.69745363828441400e-01, 4.60742244391365300e+01],
        [7.19999991727059700e+06, 3.09016285475792330e-03, 9.51054961979884800e-03, 8.54889701502629200e-01, 7.69744699956818200e-01, 6.83957517019681000e+01],
        [7.19999987816585900e+06, 3.09018068851591900e-03, 9.51056474164066100e-03, 8.54889503867869000e-01, 7.69744048756330200e-01, 9.07172789923301200e+01],
    ];

    #[rustfmt::skip]
    const SUN_POS: [[f64; 3]; 4] = [
        [2.54573948706984800e+10, -1.32930226786026440e+11, -5.76233810999046550e+10],
        [2.61012496643221800e+10, -1.32824745443904500e+11, -5.75776788441440700e+10],
        [2.67446099015337100e+10, -1.32716684101772370e+11, -5.75308590366453860e+10],
        [2.73874632760801120e+10, -1.32606044237044450e+11, -5.74829222629517900e+10],
    ];

    // Sun+Moon: mean state (MEAN run) at the same samples.
    #[rustfmt::skip]
    const BOTH_MEAN: [[f64; 6]; 4] = [
        [7.20000000000000000e+06, 3.09020748491940300e-03, 9.51054208941464400e-03, 8.54891070342577800e-01, 7.69746621381429700e-01, 2.37526990889750850e+01],
        [7.20000000000000000e+06, 3.09024402501579800e-03, 9.51052251625396700e-03, 8.54891832985364100e-01, 7.69746619321749300e-01, 4.60742280727071150e+01],
        [7.20000000000000000e+06, 3.09027869449430380e-03, 9.51050690465078200e-03, 8.54892610222829100e-01, 7.69746695153976900e-01, 6.83957569240684800e+01],
        [7.20000000000000000e+06, 3.09031063744916660e-03, 9.51049567015242400e-03, 8.54893395057535300e-01, 7.69746839728008200e-01, 9.07172856305794500e+01],
    ];

    // Sun+Moon: osculating state (OSCULATING run) at the same samples.
    #[rustfmt::skip]
    const BOTH_OSC: [[f64; 6]; 4] = [
        [7.19999993071735700e+06, 3.09017548317691870e-03, 9.51048336649572500e-03, 8.54891063070449800e-01, 7.69746638016494900e-01, 2.37526990543928170e+01],
        [7.19999979049512700e+06, 3.09026529755272500e-03, 9.51054858234412700e-03, 8.54891818243022800e-01, 7.69746627617769500e-01, 4.60742280426284600e+01],
        [7.19999968991479900e+06, 3.09026823088646640e-03, 9.51051557463474700e-03, 8.54892593128391700e-01, 7.69746696340391200e-01, 6.83957569403265200e+01],
        [7.19999972525745500e+06, 3.09030659692474000e-03, 9.51045696123429200e-03, 8.54893379050850700e-01, 7.69746828990780000e-01, 9.07172856782056100e+01],
    ];

    #[rustfmt::skip]
    const BOTH_SUN_POS: [[f64; 3]; 4] = [
        [2.54573948706984800e+10, -1.32930226786026440e+11, -5.76233810999046550e+10],
        [2.61012496643221800e+10, -1.32824745443904500e+11, -5.75776788441440700e+10],
        [2.67446099015337100e+10, -1.32716684101772370e+11, -5.75308590366453860e+10],
        [2.73874632760801120e+10, -1.32606044237044450e+11, -5.74829222629517900e+10],
    ];

    #[rustfmt::skip]
    const BOTH_MOON_POS: [[f64; 3]; 4] = [
        [-3.76299454181501150e+08, 1.25685986178895600e+08, 8.04947667791853800e+07],
        [-3.83545231476985500e+08, 1.08302982671580910e+08, 7.14499683252973000e+07],
        [-3.89698023132602750e+08, 9.06209838891624700e+07, 6.22056075512975160e+07],
        [-3.94740176669762300e+08, 7.26895945018838500e+07, 5.27876996025825440e+07],
    ];

    fn sun() -> DsstThirdBody {
        DsstThirdBody::new(MU, SUN_GM)
    }

    fn moon() -> DsstThirdBody {
        DsstThirdBody::new(MU, MOON_GM)
    }

    /// Compare an osculating state to Orekit's, asserting `a` to `a_tol` metres and the equinoctial
    /// e/i components + mean longitude to `comp_tol`; returns `(max |Δa|, max |Δcomp|)`.
    ///
    /// Both bounds are set a factor of three or four above what the port measurably achieves against
    /// each fixture, not decades above it. The residual is the Hermite interpolation Orekit applies to
    /// the coefficients between grid points and this module does not — a fixed, deterministic
    /// modelling difference over fixed golden data, computed through `libm` rather than the platform
    /// libm, so it does not drift. Three decades of slack here is what let a whole class of
    /// arithmetic faults through: `wnsj.compute(-j, …)` losing its sign moves `max comp` to 3.30e-12
    /// and `max |Δa|` to 1.68e-7 m, both of which the old 1e-9 / 1e-5 bounds accepted.
    fn assert_osc(
        got: [f64; 6],
        want: [f64; 6],
        a_tol: f64,
        comp_tol: f64,
        label: &str,
    ) -> (f64, f64) {
        let da = (got[0] - want[0]).abs();
        assert!(da < a_tol, "{label}: a abs {da:.3e} m");
        let mut max_comp = 0.0_f64;
        for i in 1..6 {
            let d = (got[i] - want[i]).abs();
            max_comp = max_comp.max(d);
            assert!(d < comp_tol, "{label} comp[{i}]: {d:.3e}");
        }
        (da, max_comp)
    }

    /// The `(maxAR3Pow, maxEccPow)` truncation the short-period path fixes must equal Orekit's
    /// `DSSTThirdBodyStaticContext` values (Sun: (3, 2); Moon: (4, 2)).
    #[test]
    fn truncation_matches_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let ts = sun().truncation(&aux, norm(SUN_POS[0]));
        assert_eq!((ts.max_ar3_pow, ts.max_ecc_pow), (SUN_MAX_AR3, SUN_MAX_ECC));
        let tm = moon().truncation(&aux, norm(BOTH_MOON_POS[0]));
        assert_eq!(
            (tm.max_ar3_pow, tm.max_ecc_pow),
            (MOON_MAX_AR3, MOON_MAX_ECC)
        );
    }

    /// The Sun-only osculating state `mean + short-period δ` matches Orekit's OSCULATING DSSTPropagator
    /// at every sample: `a` to a tight abs bound and the equinoctial e/i components + mean longitude to
    /// ~1e-9. Orekit's own mean state is fed in, so this gates only the short-period assembly.
    #[test]
    fn sun_short_period_matches_orekit() {
        let s = sun();
        let (mut max_da, mut max_comp) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            let osc = s.osculating(SUN_MEAN[k], SUN_POS[k]);
            // Achieved: 2.79e-8 m and 2.84e-13.
            let (da, comp) = assert_osc(osc, SUN_OSC[k], 1e-7, 1e-12, "sun");
            max_da = max_da.max(da);
            max_comp = max_comp.max(comp);
        }
        std::println!(
            "Sun short-period vs Orekit: max |Δa| {max_da:.2e} m, max comp {max_comp:.2e}"
        );
    }

    /// The Sun+Moon osculating state `mean + δ_sun + δ_moon` matches Orekit's OSCULATING DSSTPropagator
    /// (both bodies) at every sample. Each body's short-period δ is summed, as with the mean rates.
    #[test]
    fn sun_moon_short_period_matches_orekit() {
        let s = sun();
        let m = moon();
        let (mut max_da, mut max_comp) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            let aux = DsstAuxiliary::new(
                BOTH_MEAN[k][0],
                BOTH_MEAN[k][1],
                BOTH_MEAN[k][2],
                BOTH_MEAN[k][3],
                BOTH_MEAN[k][4],
                BOTH_MEAN[k][5],
                MU,
            );
            let dsun = s.short_period_correction(&aux, BOTH_SUN_POS[k]);
            let dmoon = m.short_period_correction(&aux, BOTH_MOON_POS[k]);
            let mut osc = BOTH_MEAN[k];
            for i in 0..6 {
                osc[i] += dsun[i] + dmoon[i];
            }
            // Achieved: 1.30e-7 m and 1.21e-12.
            let (da, comp) = assert_osc(osc, BOTH_OSC[k], 4e-7, 5e-12, "sun+moon");
            max_da = max_da.max(da);
            max_comp = max_comp.max(comp);
        }
        std::println!(
            "Sun+Moon short-period vs Orekit: max |Δa| {max_da:.2e} m, max comp {max_comp:.2e}"
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Component oracles.
    //
    // The two fixtures above assert only the assembled osculating state, and everything the assembly
    // differentiates — the ∂/∂k, ∂/∂h, ∂/∂α, ∂/∂β rows of `Wnsj`, `Abde` and `Fourier` — reaches that
    // state multiplied by the short-period amplitude itself. Measured over the whole mutation run,
    // the worst any single arithmetic fault in `Wnsj::compute`'s derivative rows does to `max comp`
    // is 9.13e-10: inside the 1e-9 the fixtures assert, and three decades above the 1.21e-12 the port
    // achieves. Tightening the fixture bound is not available, 1.21e-12 being the port-vs-Orekit
    // agreement floor. Each coefficient therefore needs an oracle of its own, and the ones below are
    // closed forms, conservation identities and gradients of the value row — never a number this
    // module produced somewhere else.

    /// A general mean state for the component oracles: `|k| ≠ |h|`, neither zero, `e = 0.176` — the
    /// fixtures sit at `e = 0.01`, where `χ³ = 1.00015` and `1/χ³ = 0.99985` are one number to four
    /// digits and half the χ-derivative chain cannot be told from its own reciprocal — and direction
    /// cosines off every axis with `α² + β² + γ² = 1` exactly.
    const CK: f64 = 0.0925;
    const CH: f64 = -0.15;
    const CQ: f64 = 0.35;
    const CP: f64 = -0.21;
    const CLM: f64 = 1.1;
    const CALPHA: f64 = 0.36;
    const CBETA: f64 = -0.48;
    const CGAMMA: f64 = 0.8;
    /// Geocentric distance of the perturbing body (m), a lunar value.
    const CR3: f64 = 3.9e8;
    /// The Moon's `(maxAR3Pow, maxFreqF)`: the widest grid the port runs, and the one whose largest
    /// reachable `|j − s| = maxFreqF + MAX_ECCPOWER_SP = 9` is exactly the last index of `Wnsj`'s
    /// `btjms` table.
    const C_NMAX: usize = MOON_MAX_AR3;
    const C_JMAX: usize = MOON_MAX_AR3 + 1;

    /// A first-order dual number `a + b·ε`, `ε² = 0`. Pushing `Dual::var(x)` through an expression
    /// carries its exact derivative in the second component, with none of the step-size trade-off a
    /// difference quotient has — and both `Wnsj` and `Abde` have call sites where the true derivative
    /// is fifteen decades below the value, so a difference quotient could only report its own noise
    /// there. Only the operations the two reference formulas below use are implemented.
    #[derive(Clone, Copy)]
    struct Dual(f64, f64);

    impl Dual {
        /// The constant `a`, derivative zero.
        fn c(a: f64) -> Self {
            Self(a, 0.0)
        }
        /// The variable being differentiated against, at `a`.
        fn var(a: f64) -> Self {
            Self(a, 1.0)
        }
        fn sqrt(self) -> Self {
            let r = self.0.sqrt();
            Self(r, self.1 / (2.0 * r))
        }
        fn powi(self, n: i32) -> Self {
            if n == 0 {
                Self(1.0, 0.0)
            } else {
                Self(self.0.powi(n), f64::from(n) * self.0.powi(n - 1) * self.1)
            }
        }
        fn sin(self) -> Self {
            Self(self.0.sin(), self.0.cos() * self.1)
        }
        fn cos(self) -> Self {
            Self(self.0.cos(), -self.0.sin() * self.1)
        }
        fn atan2(self, x: Self) -> Self {
            let r2 = x.0 * x.0 + self.0 * self.0;
            Self(self.0.atan2(x.0), (x.0 * self.1 - self.0 * x.1) / r2)
        }
    }

    impl core::ops::Add for Dual {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Self(self.0 + o.0, self.1 + o.1)
        }
    }
    impl core::ops::Sub for Dual {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Self(self.0 - o.0, self.1 - o.1)
        }
    }
    impl core::ops::Mul for Dual {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            Self(self.0 * o.0, self.1 * o.0 + self.0 * o.1)
        }
    }
    impl core::ops::Div for Dual {
        type Output = Self;
        fn div(self, o: Self) -> Self {
            Self(self.0 / o.0, (self.1 * o.0 - self.0 * o.1) / (o.0 * o.0))
        }
    }
    impl core::ops::Mul<f64> for Dual {
        type Output = Self;
        fn mul(self, c: f64) -> Self {
            Self(self.0 * c, self.1 * c)
        }
    }
    impl core::ops::Div<f64> for Dual {
        type Output = Self;
        fn div(self, c: f64) -> Self {
            Self(self.0 / c, self.1 / c)
        }
    }
    impl core::ops::Sub<f64> for Dual {
        type Output = Self;
        fn sub(self, c: f64) -> Self {
            Self(self.0 - c, self.1)
        }
    }
    impl core::ops::Add<f64> for Dual {
        type Output = Self;
        fn add(self, c: f64) -> Self {
            Self(self.0 + c, self.1)
        }
    }

    /// `C(a, b)` for `0 ≤ b ≤ a`.
    fn binom(a: i64, b: i64) -> f64 {
        factorial(a) / (factorial(b) * factorial(a - b))
    }

    /// `Pₗ^{(v,w)}(x)` from the explicit hypergeometric sum [Abramowitz & Stegun 22.3.2],
    /// `Σ_{m=0..l} C(l+v, m)·C(l+w, l−m)·((x−1)/2)^{l−m}·((x+1)/2)^m`.
    ///
    /// A construction independent of the one under test: [`jacobi_value_deriv`] builds every degree
    /// `P₀..Pₗ` by the Hipparchus three-term recurrence and evaluates the last by Horner, this one
    /// touches no other degree and no coefficient array. At the `x = χ > 1` this module evaluates at,
    /// every term of the sum is positive, so it carries no cancellation either.
    fn jacobi_explicit(l: usize, v: i64, w: i64, x: Dual) -> Dual {
        let li = l as i64;
        let (xm, xp) = ((x - 1.0) / 2.0, (x + 1.0) / 2.0);
        (0..=li).fold(Dual::c(0.0), |acc, m| {
            acc + xm.powi((li - m) as i32)
                * xp.powi(m as i32)
                * (binom(li + v, m) * binom(li + w, li - m))
        })
    }

    /// `e^{−|j−s|}·wⱼ^{n,s}(χ)` written straight from Danielson 4.2-9,10,11 as a function of the mean
    /// `(k, h)` alone: `e² = k²+h²`, `B = √(1−e²)`, `b = 1/(1+B)`, `χ = 1/B`, `c = e·b`,
    /// `l = n − max(|j|,|s|)`, and the factorial ratio only where `|s| > |j|`.
    fn wnsj_ref(k: Dual, h: Dual, j: i64, s: i64, n: i64) -> Dual {
        let one = Dual::c(1.0);
        let e2 = k * k + h * h;
        let bigb = (one - e2).sqrt();
        let b = one / (one + bigb);
        let c2 = e2 * b * b;
        let (abs_j, abs_s) = (j.abs(), s.abs());
        let (abs_jms, abs_jps) = ((j - s).abs(), (j + s).abs());
        let l = n - abs_j.max(abs_s);
        let fact_coef = if abs_s > abs_j {
            (factorial(n + s) / factorial(n + j)) * (factorial(n - s) / factorial(n - j))
        } else {
            1.0
        };
        let sign = if abs_jms % 2 != 0 { -1.0 } else { 1.0 };
        (one - c2).powi(l as i32) / (one + c2).powi(n as i32)
            * b.powi(abs_jms as i32)
            * jacobi_explicit(l as usize, abs_jms, abs_jps, one / bigb)
            * (sign * fact_coef)
    }

    /// Every `(j, s, n)` [`fourier`] hands [`Wnsj::compute`]: the main terms at `n + 1` for `±j`, and
    /// the Cʲ,λ terms at `n`. Read off the loop bounds rather than invented, because `Wnsj` sizes
    /// `btjms` so that the largest reachable `|j − s|` is its last index — a sweep one triple wider
    /// runs off the end of it.
    fn wnsj_call_sites(n_max: usize, s_max: usize, j_max: usize) -> Vec<(i64, i64, i64)> {
        let mut sites = Vec::new();
        for j in 1..=j_max {
            for s in 0..=s_max {
                for n in 2.max((j - 1).max(s))..=n_max {
                    if (n - s) % 2 == 0 {
                        let (ji, si) = (j as i64, s as i64);
                        sites.push((ji, si, (n + 1) as i64));
                        sites.push((-ji, si, (n + 1) as i64));
                        if n >= j && j < j_max {
                            sites.push((ji, si, n as i64));
                            sites.push((-ji, si, n as i64));
                        }
                    }
                }
            }
        }
        sites
    }

    /// The fourth-order central difference `[−f(x+2δ) + 8f(x+δ) − 8f(x−δ) + f(x−2δ)] / 12δ` from four
    /// already-evaluated samples — the coefficient chain is rebuilt once per offset, not once per
    /// coefficient.
    fn stencil5(p2: f64, p1: f64, m1: f64, m2: f64, d: f64) -> f64 {
        (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * d)
    }

    /// A [`Ctx`] at a mean state and direction cosines, built by the constructor
    /// `short_period_correction` uses — repeating its field expressions here would put `b`, `hxxx`
    /// and `kxxx` beyond the reach of every test below, which is how two mutations of `hxxx`/`kxxx`
    /// survived a suite that asserts the derivative rows they feed.
    fn ctx_at(aux: &DsstAuxiliary, alpha: f64, beta: f64, gamma: f64, muo_r3: f64) -> Ctx<'_> {
        Ctx::new(aux, alpha, beta, gamma, muo_r3)
    }

    /// The value row of [`Wnsj::compute`] against Danielson 4.2-9,10,11 evaluated by the explicit
    /// Jacobi sum, over every `(j, s, n)` the Fourier loop reaches and four eccentricities.
    #[test]
    fn wnsj_value_matches_the_danielson_closed_form() {
        let sites = wnsj_call_sites(C_NMAX, MAX_ECCPOWER_SP, C_JMAX);
        let mut worst = 0.0_f64;
        for &(k, h) in &[(EX0, EY0), (CK, CH), (0.31, 0.22), (-0.55, 0.42)] {
            let aux = DsstAuxiliary::new(A0, k, h, CQ, CP, CLM, MU);
            let ctx = ctx_at(&aux, CALPHA, CBETA, CGAMMA, MOON_GM / CR3);
            let w = Wnsj::new(&ctx, C_NMAX, C_JMAX);
            for &(j, s, n) in &sites {
                let want = wnsj_ref(Dual::c(k), Dual::c(h), j, s, n).0;
                let rel = (w.compute(j, s, n)[0] - want).abs() / want.abs();
                assert!(rel < 1e-13, "w({j},{s},{n}) at e={:.3}: {rel:.3e}", aux.ecc);
                worst = worst.max(rel);
            }
        }
        std::println!("Wnsj value vs the explicit Jacobi sum: worst {worst:.2e} relative");
    }

    /// The `∂/∂k` and `∂/∂h` rows of [`Wnsj::compute`] against the exact gradient of the same closed
    /// form, differentiated by [`Dual`] rather than recovered from the value the port itself computed.
    ///
    /// The port reaches these two rows by a hand-written chain rule over `b`, `c` and `χ` — nine
    /// products the fixtures cannot separate, every one of them moving `max comp` by less than the
    /// 1e-9 they assert. `wnsj_ref` reaches them by differentiating the formula.
    #[test]
    fn wnsj_kh_derivatives_match_the_closed_form_gradient() {
        let sites = wnsj_call_sites(C_NMAX, MAX_ECCPOWER_SP, C_JMAX);
        let (mut worst_rel, mut worst_ulp) = (0.0_f64, 0.0_f64);
        for &(k, h) in &[(EX0, EY0), (CK, CH), (0.31, 0.22), (-0.55, 0.42)] {
            let aux = DsstAuxiliary::new(A0, k, h, CQ, CP, CLM, MU);
            let ctx = ctx_at(&aux, CALPHA, CBETA, CGAMMA, MOON_GM / CR3);
            let w = Wnsj::new(&ctx, C_NMAX, C_JMAX);
            for &(j, s, n) in &sites {
                let got = w.compute(j, s, n);
                let want = [
                    got[0],
                    wnsj_ref(Dual::var(k), Dual::c(h), j, s, n).1,
                    wnsj_ref(Dual::c(k), Dual::var(h), j, s, n).1,
                ];
                for r in 1..3 {
                    // A few sites have an identically zero row — `w¹⁰₂` reduces to the constant −1,
                    // the χ, b and c factors cancelling outright — and there both sides are pure
                    // round-off of the value, so the bound is relative to the derivative where it
                    // resolves and to the value's last bits where it does not.
                    let err = (got[r] - want[r]).abs();
                    assert!(
                        err <= 1e-12 * want[r].abs() + 1e-14 * got[0].abs(),
                        "dw{r}({j},{s},{n}) at e={:.3}: {err:.3e} on {:.3e}",
                        aux.ecc,
                        want[r]
                    );
                    if want[r].abs() > 1e-10 * got[0].abs() {
                        worst_rel = worst_rel.max(err / want[r].abs());
                    }
                    worst_ulp = worst_ulp.max(err / (f64::EPSILON * got[0].abs()));
                }
            }
        }
        std::println!(
            "Wnsj (∂/∂k, ∂/∂h) vs the exact closed-form gradient: worst {worst_rel:.2e} relative \
             where the row resolves, {worst_ulp:.1} ulp of the value where it is identically zero"
        );
    }

    /// `(Aⱼₛ, Bⱼₛ, Dⱼₛ, Eⱼₛ)` in polar form. Writing `k + i·h = e·e^{iφ}` and `α + i·β = ρ·e^{iθ}`,
    /// Danielson 2.5.3-(5) gives `Cⱼ + i·Sⱼ = (x + i·y)ʲ`, and the eight products `Abde::compute`
    /// forms collapse by the angle-sum identities to four amplitude-and-phase terms:
    /// `A = ρˢ·e^{|j−s|}·sin(sθ + (j−s)φ)`, `B = ρˢ·e^{j+s}·sin((j+s)φ − sθ)`, and the same two with
    /// `cos` for `D` and `E`. Note this is one expression for `j ≥ s` and `j < s` both — the port's
    /// `sign`/`|j − s|` split is a way of writing it, not part of it.
    fn abde_ref(k: Dual, h: Dual, alpha: Dual, beta: Dual, j: i64, s: i64) -> [Dual; 4] {
        let (ecc, phi) = ((k * k + h * h).sqrt(), h.atan2(k));
        let (rho, theta) = ((alpha * alpha + beta * beta).sqrt(), beta.atan2(alpha));
        let rs = rho.powi(s as i32);
        let em = rs * ecc.powi((j - s).unsigned_abs() as i32);
        let ep = rs * ecc.powi((j + s) as i32);
        let pa = theta * s as f64 + phi * (j - s) as f64;
        let pb = phi * (j + s) as f64 - theta * s as f64;
        [em * pa.sin(), ep * pb.sin(), em * pa.cos(), ep * pb.cos()]
    }

    /// All five rows of all four [`Abde::compute`] terms against the polar closed form and its exact
    /// `(k, h, α, β)` gradient, over every `(j, s)` the Fourier loop asks for — `j < s` included, so
    /// the `sign` branch is exercised in both directions.
    ///
    /// The port reaches `A`, `B`, `D`, `E` through `CjSj`'s complex-power recurrence and eight
    /// hand-expanded products of its entries; [`abde_ref`] reaches them through an amplitude and a
    /// phase, with no recurrence and no product. They agree only if every sign in the expansion is
    /// right — and the fixtures cannot see those signs, the seventeen mutants surviving here moving
    /// `max comp` by at most 3.3e-10 against the 1e-9 they assert.
    #[test]
    fn abde_matches_the_polar_closed_form_and_its_gradient() {
        let aux = DsstAuxiliary::new(A0, CK, CH, CQ, CP, CLM, MU);
        let ctx = ctx_at(&aux, CALPHA, CBETA, CGAMMA, MOON_GM / CR3);
        let abde = Abde::new(&ctx, C_JMAX + MAX_ECCPOWER_SP, MAX_ECCPOWER_SP);
        let (ecc, rho) = (aux.ecc, (CALPHA * CALPHA + CBETA * CBETA).sqrt());
        let mut worst = 0.0_f64;
        for j in 1..=C_JMAX as i64 {
            for s in 0..=MAX_ECCPOWER_SP {
                let si = s as i64;
                let got = abde.compute(j, s);
                let terms = [got.0, got.1, got.2, got.3];
                let d = |k: Dual, h: Dual, a: Dual, b: Dual| abde_ref(k, h, a, b, j, si);
                let (ck, ch) = (Dual::c(CK), Dual::c(CH));
                let (ca, cb) = (Dual::c(CALPHA), Dual::c(CBETA));
                let want = [
                    d(ck, ch, ca, cb),
                    d(Dual::var(CK), ch, ca, cb),
                    d(ck, Dual::var(CH), ca, cb),
                    d(ck, ch, Dual::var(CALPHA), cb),
                    d(ck, ch, ca, Dual::var(CBETA)),
                ];
                // Every entry is a homogeneous polynomial of degree `s` in (α, β) and `|j−s|` or
                // `j+s` in (k, h), so `ρˢ·e^…` is its natural amplitude and the derivatives are that
                // over ρ or e. Several entries are structurally zero — `∂A₁₀/∂k` is `∂h/∂k` — which
                // rules a relative scale out; this is an absolute bound in units of the term itself.
                let amp = [
                    rho.powi(s as i32) * ecc.powi((j - si).unsigned_abs() as i32),
                    rho.powi(s as i32) * ecc.powi((j + si) as i32),
                ];
                for (t, term) in terms.iter().enumerate() {
                    for (r, w) in want.iter().enumerate() {
                        let scale = amp[t % 2] * if r == 0 { 1.0 } else { 1.0 / ecc.min(rho) };
                        // Row 0 wants the dual's value part, rows 1..4 its derivative part.
                        let err = if r == 0 {
                            (term[0] - w[t].0).abs() / scale
                        } else {
                            (term[r] - w[t].1).abs() / scale
                        };
                        assert!(
                            err < 1e-13,
                            "abde[{t}] row {r} at (j={j}, s={s}): {err:.3e}"
                        );
                        worst = worst.max(err);
                    }
                }
            }
        }
        std::println!(
            "Abde vs the polar closed form: worst {worst:.2e}, in units of the term's own amplitude"
        );
    }

    /// The six variables the `Fourier` and generating-function derivative rows are taken against.
    #[derive(Clone, Copy)]
    struct Vars {
        sma: f64,
        k: f64,
        h: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    }

    impl Vars {
        /// The variable row `r` of a coefficient column differentiates against (1:a … 6:γ).
        fn get(self, r: usize) -> f64 {
            [self.sma, self.k, self.h, self.alpha, self.beta, self.gamma][r - 1]
        }
        fn with(mut self, r: usize, x: f64) -> Self {
            match r {
                1 => self.sma = x,
                2 => self.k = x,
                3 => self.h = x,
                4 => self.alpha = x,
                5 => self.beta = x,
                _ => self.gamma = x,
            }
            self
        }
    }

    const CVARS: Vars = Vars {
        sma: A0,
        k: CK,
        h: CH,
        alpha: CALPHA,
        beta: CBETA,
        gamma: CGAMMA,
    };

    /// Difference steps, ~1.4e-5 of each variable's own scale, indexed by the derivative row.
    const CSTEP: [f64; 7] = [0.0, 100.0, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5];

    /// The whole coefficient chain — `Gₙₛ`, `Wnsj`, `Abde`, `fourier` — at one point of the
    /// six-variable space, wired as `short_period_correction` wires it. `α`, `β` and `γ` move
    /// independently here: the port's `∂/∂α`, `∂/∂β`, `∂/∂γ` are formal partials of the series, and
    /// only their recombination into `∂/∂p`, `∂/∂q` puts them back on the unit sphere.
    fn fourier_at(v: Vars, tb: &DsstThirdBody) -> Fourier {
        let aux = DsstAuxiliary::new(v.sma, v.k, v.h, CQ, CP, CLM, MU);
        let ctx = ctx_at(&aux, v.alpha, v.beta, v.gamma, tb.third_body_mu / CR3);
        let mut aor3_pow = std::vec![1.0; C_NMAX + 1];
        for i in 1..=C_NMAX {
            aor3_pow[i] = aor3_pow[i - 1] * (v.sma / CR3);
        }
        let qns = compute_qns(v.gamma, C_NMAX, MAX_ECCPOWER_SP);
        let gns = Gns::new(&ctx, C_NMAX, MAX_ECCPOWER_SP, &aor3_pow, &qns, tb.vns());
        let wnsj = Wnsj::new(&ctx, C_NMAX, C_JMAX);
        let abde = Abde::new(&ctx, C_JMAX + MAX_ECCPOWER_SP, MAX_ECCPOWER_SP);
        fourier(&ctx, C_NMAX, MAX_ECCPOWER_SP, C_JMAX, &gns, &wnsj, &abde)
    }

    /// The generating-function columns at one point of the same space.
    fn generating_at(v: Vars, tb: &DsstThirdBody) -> (Vec<[f64; 8]>, Vec<[f64; 8]>) {
        let aux = DsstAuxiliary::new(v.sma, v.k, v.h, CQ, CP, CLM, MU);
        let trunc = ThirdBodyTruncation {
            max_ar3_pow: C_NMAX,
            max_ecc_pow: MOON_MAX_ECC,
        };
        let udu = tb.potential_derivatives(&aux, v.alpha, v.beta, v.gamma, CR3, trunc);
        generating_function(&fourier_at(v, tb), &udu, &aux, C_JMAX)
    }

    /// Rows 1..6 of every Fourier Cʲ/Sʲ column are `∂/∂(a, k, h, α, β, γ)` of row 0: `Gₙₛ` carries the
    /// `a` and `γ` dependence, `Wnsj` and `Abde` the `k`, `h`, `α`, `β` dependence, and no term of the
    /// accumulation mixes the two. Row 0 is what the Orekit fixtures pin, so its own numerical
    /// gradient is the one oracle for the six rows under it that is not those same accumulations
    /// read back. (S1)
    #[test]
    fn fourier_derivative_rows_are_the_gradient_of_the_value_row() {
        let tb = moon();
        let base = fourier_at(CVARS, &tb);
        let scale = (1..=C_JMAX).fold(0.0_f64, |m, j| {
            m.max(base.cj[j][0].abs()).max(base.sj[j][0].abs())
        });
        let mut worst = 0.0_f64;
        for r in 1..=6 {
            let (x0, d) = (CVARS.get(r), CSTEP[r]);
            let at = |m: f64| fourier_at(CVARS.with(r, x0 + m * d), &tb);
            let (p2, p1, m1, m2) = (at(2.0), at(1.0), at(-1.0), at(-2.0));
            // The stencil's own round-off, `1.5·ε·|C|/δ`, is the floor below which no difference
            // quotient says anything; five times it is the scale the comparison is made on.
            let floor = 7.5 * f64::EPSILON * scale / d;
            for j in 1..=C_JMAX {
                for (name, pick) in [
                    (
                        "C",
                        (|f: &Fourier, j: usize, r: usize| f.cj[j][r])
                            as fn(&Fourier, usize, usize) -> f64,
                    ),
                    ("S", |f: &Fourier, j: usize, r: usize| f.sj[j][r]),
                ] {
                    let fd = stencil5(
                        pick(&p2, j, 0),
                        pick(&p1, j, 0),
                        pick(&m1, j, 0),
                        pick(&m2, j, 0),
                        d,
                    );
                    let err = (pick(&base, j, r) - fd).abs() / fd.abs().max(floor);
                    assert!(err < 1e-8, "{name}{j} row {r}: {err:.3e} on {fd:.6e}");
                    worst = worst.max(err);
                }
            }
        }
        std::println!("Fourier rows 1..6 vs the gradient of row 0: worst {worst:.2e} relative");
    }

    /// The Cʲ,λ/Sʲ,λ series is `∂S/∂λ` for a generating function periodic in λ, so its mean over one
    /// revolution must vanish. With `dλ = (1 − k·cos F − h·sin F)·dF` that gives
    /// `⟨cos jF⟩_λ = −k/2·δ_{j1}` and `⟨sin jF⟩_λ = −h/2·δ_{j1}`, so the mean is
    /// `C⁰,λ − (k·C¹,λ + h·S¹,λ)/2` — and requiring it to be zero *is* Danielson 4.2-21. The identity
    /// comes from the periodicity, not from the line that implements it.
    ///
    /// The fixtures cannot see this: every mutation of that line moves `max comp` by at most 7.4e-10
    /// against the 1e-9 they assert, and dropping `C⁰,λ` to zero outright moves nothing they print.
    #[test]
    fn the_lambda_derivative_series_has_zero_mean_over_one_revolution() {
        const N: usize = 2048;
        let fr = fourier_at(CVARS, &moon());
        let (mut sum, mut amp) = (0.0_f64, 0.0_f64);
        for i in 0..N {
            let lm = core::f64::consts::TAU * i as f64 / N as f64;
            let f = mean_to_eccentric(CK, CH, lm);
            let g = (1..=C_JMAX as i64).fold(fr.c0lambda(), |acc, j| {
                acc + fr.cjlambda(j) * cos(j as f64 * f) + fr.sjlambda(j) * sin(j as f64 * f)
            });
            sum += g;
            amp = amp.max(g.abs());
        }
        let mean = (sum / N as f64).abs() / amp;
        assert!(
            mean < 1e-13,
            "⟨∂S/∂λ⟩ is {mean:.3e} of the series amplitude"
        );
        std::println!("⟨∂S/∂λ⟩ over one revolution: {mean:.2e} of the series amplitude");
    }

    /// Rows 1, 4, 5 and 6 of the generating-function columns are `∂/∂(a, α, β, γ)` of row 0 — the four
    /// variables the eccentric longitude `F` does not depend on, so no λ-mixing term enters them and
    /// the relation is exact. That includes the `j = 1` correction `C¹ = C¹_F − h·U`, `S¹ = S¹_F + k·U`,
    /// whose seven `±h·∂U/∂x` and `±k·∂U/∂x` products the fixtures leave entirely unasserted: every
    /// mutation of that block moves `max comp` by less than the 1e-9 they assert.
    #[test]
    fn generating_function_rows_are_the_gradient_of_the_value_row() {
        let tb = moon();
        let (bc, bs) = generating_at(CVARS, &tb);
        let scale = (1..=C_JMAX).fold(0.0_f64, |m, j| m.max(bc[j][0].abs()).max(bs[j][0].abs()));
        let mut worst = 0.0_f64;
        for r in [1, 4, 5, 6] {
            let (x0, d) = (CVARS.get(r), CSTEP[r]);
            let at = |m: f64| generating_at(CVARS.with(r, x0 + m * d), &tb);
            let (p2, p1, m1, m2) = (at(2.0), at(1.0), at(-1.0), at(-2.0));
            let floor = 7.5 * f64::EPSILON * scale / d;
            for j in 1..=C_JMAX {
                for (name, sel) in [("C", 0usize), ("S", 1usize)] {
                    let pick = |g: &(Vec<[f64; 8]>, Vec<[f64; 8]>), row: usize| {
                        if sel == 0 { g.0[j][row] } else { g.1[j][row] }
                    };
                    let fd = stencil5(pick(&p2, 0), pick(&p1, 0), pick(&m1, 0), pick(&m2, 0), d);
                    let got = if sel == 0 { bc[j][r] } else { bs[j][r] };
                    let err = (got - fd).abs() / fd.abs().max(floor);
                    assert!(err < 1e-8, "{name}{j} row {r}: {err:.3e} on {fd:.6e}");
                    worst = worst.max(err);
                }
            }
        }
        std::println!(
            "Generating-function rows (a, α, β, γ) vs the gradient of row 0: worst {worst:.2e}"
        );
    }

    /// [`mean_to_eccentric`] returns the root of the equinoctial Kepler equation, not merely the point
    /// its iteration stopped moving at: `λ = LE − k·sin LE + h·cos LE` is substituted back, which is
    /// the definition and owes nothing to the Danby step used to reach it.
    ///
    /// Nothing asserted `F` before. It enters the fixtures only through `Σ Cᵢⱼ cos jF + Sᵢⱼ sin jF`,
    /// where the coefficients are of order the short-period amplitude itself, so an `F` wrong in the
    /// fourth decimal moves the printed state by ~1e-12 m — which is why all twelve mutations of the
    /// solver, the convergence test included, left the fixtures' printed figures unchanged.
    #[test]
    fn mean_to_eccentric_inverts_the_equinoctial_kepler_equation() {
        const N: usize = 97;
        // Two pericentre longitudes per eccentricity, so a `(k, h)` asymmetry cannot hide.
        const R2: f64 = core::f64::consts::FRAC_1_SQRT_2;
        let mut worst = 0.0_f64;
        for &(k, h) in &[
            (0.0, 0.0),
            (EX0, EY0),
            (CK, CH),
            (-0.42, 0.31),
            (0.60, 0.55),
            // The Danby step's own arithmetic only becomes observable near the singular limit: the
            // step is a correction proportional to the residual, so a corrupted one still lands on
            // the root wherever the iteration converges at all, and it converges easily below
            // e ≈ 0.8. These four carry e = 0.9 and e = 0.99.
            (0.90, 0.0),
            (0.90 * R2, -0.90 * R2),
            (0.99, 0.0),
            (0.99 * R2, 0.99 * R2),
        ] {
            for i in 0..N {
                // λ is kept inside one revolution so the residual is not measured through the
                // cancellation of two large longitudes: `LE − λ` is of order e whatever λ is.
                let lm = core::f64::consts::TAU * i as f64 / N as f64;
                let le = mean_to_eccentric(k, h, lm);
                let residual = (le - k * sin(le) + h * cos(le) - lm).abs();
                assert!(
                    residual < 1e-15,
                    "kepler residual {residual:.3e} at e={k},{h}"
                );
                worst = worst.max(residual);
            }
        }
        std::println!(
            "Equinoctial Kepler residual λ − (LE − k sin LE + h cos LE): worst {worst:.2e}"
        );
    }

    /// The short-period correction averages to zero over one revolution of the mean longitude, in
    /// every one of the six elements.
    ///
    /// The Cᵢⱼ/Sᵢⱼ do not depend on λ, only `F` does, so with `⟨cos jF⟩_λ = −k/2·δ_{j1}` and
    /// `⟨sin jF⟩_λ = −h/2·δ_{j1}` the average is `C⁰ − (k·Cᵢ₁ + h·Sᵢ₁)/2` — zero exactly when `C⁰` is
    /// Danielson 2.5.2-15a. That is what a *short-period* term means, and it is a property of the
    /// assembled correction rather than a number this module produced: mutating the `C⁰` line takes
    /// `max |Δa|` from 1.30e-7 m to 9.68e-6 m, still inside the 1e-5 the fixtures allow.
    #[test]
    fn the_short_period_correction_has_zero_mean_over_the_mean_longitude() {
        const N: usize = 4096;
        let m = moon();
        let (mut sum, mut amp) = ([0.0_f64; 6], [0.0_f64; 6]);
        for i in 0..N {
            let lm = core::f64::consts::TAU * i as f64 / N as f64;
            let aux = DsstAuxiliary::new(A0, CK, CH, CQ, CP, lm, MU);
            let sp = m.short_period_correction(&aux, BOTH_MOON_POS[0]);
            for e in 0..6 {
                sum[e] += sp[e];
                amp[e] = amp[e].max(sp[e].abs());
            }
        }
        let mut worst = 0.0_f64;
        for e in 0..6 {
            let mean = (sum[e] / N as f64).abs() / amp[e];
            assert!(
                mean < 1e-13,
                "element {e}: mean is {mean:.3e} of the amplitude"
            );
            worst = worst.max(mean);
        }
        std::println!("⟨short-period δ⟩ over one revolution: worst {worst:.2e} of the amplitude");
    }

    /// Rows 2 and 3 of the generating-function columns — the two variables the eccentric longitude
    /// itself depends on — once the λ-mixing is taken back out.
    ///
    /// `F` depends on `k` and `h` through the Kepler equation: differentiating
    /// `λ = F − k·sin F + h·cos F` at fixed λ gives `∂F/∂k = sin F·∂F/∂λ` and
    /// `∂F/∂h = −cos F·∂F/∂λ`, so `∂S/∂k|_λ = ∂S/∂k|_F + sin F·∂S/∂λ` and
    /// `∂S/∂h|_λ = ∂S/∂h|_F − cos F·∂S/∂λ`. Expanding `sin F` and `cos F` against
    /// `∂S/∂λ = C⁰,λ + Σ(Cʲ,λ cos jF + Sʲ,λ sin jF)` by the product-to-sum identities puts
    /// `(S^{j+1,λ} − S^{j−1,λ})/2` on `cos jF` of the `∂/∂k` series, `(C^{j−1,λ} − C^{j+1,λ})/2`
    /// plus a whole `C⁰,λ` at `j = 1` on its `sin jF`, and the two `cos F` counterparts on `∂/∂h` —
    /// exactly the terms rows 2 and 3 carry above the `|_F` partial. Subtracting them leaves the
    /// plain `∂/∂(k, h)` of row 0, which is what is finite-differenced against.
    ///
    /// `C⁰,λ` is taken as `(k·C¹,λ + h·S¹,λ)/2`, the value the zero-mean identity above forces, not
    /// the port's own `c0lambda`.
    #[test]
    fn generating_function_kh_rows_are_the_gradient_once_the_lambda_mixing_is_removed() {
        let tb = moon();
        let fr = fourier_at(CVARS, &tb);
        let (bc, bs) = generating_at(CVARS, &tb);
        let c0l = (CK * fr.cjlambda(1) + CH * fr.sjlambda(1)) / 2.0;
        let scale = (1..=C_JMAX).fold(0.0_f64, |m, j| m.max(bc[j][0].abs()).max(bs[j][0].abs()));
        let mut worst = 0.0_f64;
        for r in [2usize, 3] {
            let (x0, d) = (CVARS.get(r), CSTEP[r]);
            let at = |m: f64| generating_at(CVARS.with(r, x0 + m * d), &tb);
            let (p2, p1, m1, m2) = (at(2.0), at(1.0), at(-1.0), at(-2.0));
            let floor = 7.5 * f64::EPSILON * scale / d;
            for j in 1..=C_JMAX {
                let (ji, at1) = (j as i64, f64::from(j == 1));
                let mix = if r == 2 {
                    [
                        -(fr.sjlambda(ji - 1) - fr.sjlambda(ji + 1)) / 2.0,
                        (fr.cjlambda(ji - 1) - fr.cjlambda(ji + 1)) / 2.0 + at1 * c0l,
                    ]
                } else {
                    [
                        -(fr.cjlambda(ji - 1) + fr.cjlambda(ji + 1)) / 2.0 - at1 * c0l,
                        -(fr.sjlambda(ji - 1) + fr.sjlambda(ji + 1)) / 2.0,
                    ]
                };
                for (name, sel) in [("C", 0usize), ("S", 1usize)] {
                    let row0 = |g: &(Vec<[f64; 8]>, Vec<[f64; 8]>)| {
                        if sel == 0 { g.0[j][0] } else { g.1[j][0] }
                    };
                    let fd = stencil5(row0(&p2), row0(&p1), row0(&m1), row0(&m2), d);
                    let got = if sel == 0 { bc[j][r] } else { bs[j][r] } - mix[sel];
                    let err = (got - fd).abs() / fd.abs().max(floor);
                    assert!(err < 1e-8, "{name}{j} row {r}: {err:.3e} on {fd:.6e}");
                    worst = worst.max(err);
                }
            }
        }
        std::println!(
            "Generating-function rows (k, h), λ-mixing removed, vs the gradient of row 0: \
             worst {worst:.2e}"
        );
    }

    /// The direction cosines and the geocentric distance of one body at a mean state, as both the
    /// mean-rate and the short-period path derive them.
    fn body_geometry(aux: &DsstAuxiliary, body: [f64; 3]) -> (f64, f64, f64, f64) {
        let r3 = norm(body);
        let inv = 1.0 / r3;
        let dir = [body[0] * inv, body[1] * inv, body[2] * inv];
        (
            dot(dir, aux.f_vec),
            dot(dir, aux.g_vec),
            dot(dir, aux.w_vec),
            r3,
        )
    }

    /// [`one_element_row`] applies the equinoctial Poisson bracket `δxᵢ = {xᵢ, S}` [Danielson
    /// 2.5.5-10]; [`DsstThirdBody::mean_element_rates`] applies the same bracket to the averaged
    /// potential [Danielson 3.1-(1)]. The second is written out separately in `third_body.rs` and
    /// validated there against Orekit's own mean rates — a far tighter oracle than the 1e-12 residual
    /// the short-period fixtures leave — so feeding the bracket the potential's own gradient has to
    /// reproduce those rates, times the mean motion the short-period factors divide out. (S1)
    ///
    /// The one term the mean rates do not carry, `C/(2AB)·(q, p)·(h·U_k − k·U_h − U_αβ)`, is the
    /// generator of a simultaneous rotation of `(k, h)` and `(α, β)`. The third-body potential reaches
    /// those four only through `kα + hβ` and `hα − kβ` (`compute_gs_hs`) and through `k² + h²`, all
    /// three rotation invariants, so the generator annihilates `U` identically — asserted first, since
    /// the comparison rests on it.
    #[test]
    fn one_element_row_reproduces_the_mean_element_rates() {
        let tb = moon();
        let body = BOTH_MOON_POS[0];
        let aux = DsstAuxiliary::new(A0, CK, CH, CQ, CP, CLM, MU);
        let (alpha, beta, gamma, r3) = body_geometry(&aux, body);
        let trunc = tb.truncation(&aux, r3);
        let udu = tb.potential_derivatives(&aux, alpha, beta, gamma, r3, trunc);

        let uscale = [udu.du_dk, udu.du_dh, udu.du_dal, udu.du_dbe]
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        let generator =
            aux.h * udu.du_dk - aux.k * udu.du_dh - (alpha * udu.du_dbe - beta * udu.du_dal);
        assert!(
            generator.abs() < 1e-14 * uscale,
            "rotation generator {generator:.3e} against |∇U| {uscale:.3e}"
        );

        let ctx = ctx_at(&aux, alpha, beta, gamma, tb.third_body_mu / r3);
        let sd = [
            0.0, udu.du_da, udu.du_dk, udu.du_dh, udu.du_dal, udu.du_dbe, udu.du_dga, 0.0,
        ];
        let (cij, _) = compute_cij_sij(&ctx, &[[0.0; 8], sd], &[[0.0; 8]; 2], 1);
        let want = tb.mean_element_rates(&aux, body);
        let mut worst = 0.0_f64;
        // Element 0 is left out: `δa = 2a/(An)·∂S/∂λ`, and both sides are exactly zero here because
        // the averaged potential has no λ to be differentiated against.
        for i in 1..6 {
            let rel = (cij[1][i] * aux.n - want[i]).abs() / want[i].abs();
            assert!(rel < 1e-12, "element {i}: {rel:.3e} relative");
            worst = worst.max(rel);
        }
        std::println!(
            "Bracket on ∇U vs the mean-element rates: worst {worst:.2e} relative; \
             rotation generator {:.2e} of |∇U|",
            generator.abs() / uscale
        );
    }

    /// The `(a, k, h, λ)` block of the same bracket is antisymmetric — `{xᵢ, xⱼ} = −{xⱼ, xᵢ}`. Only
    /// those four rows are in coordinates matching their columns; the `q` and `p` rows are driven by
    /// the α, β, γ columns, which reach `p` and `q` only through a chain rule. It is the one relation
    /// that pins the `∂S/∂λ` column of the `k` and `h` rows, which the mean rates cannot reach at all:
    /// an averaged potential has no λ dependence to put there. Extracted a column at a time from unit
    /// inputs, the row being linear in them.
    #[test]
    fn the_bracket_is_antisymmetric_on_the_a_k_h_lambda_block() {
        let aux = DsstAuxiliary::new(A0, CK, CH, CQ, CP, CLM, MU);
        let ctx = ctx_at(&aux, CALPHA, CBETA, CGAMMA, MOON_GM / CR3);
        // (column of `sd`, row of the output) for a, k, h and λ.
        let axes = [(1usize, 0usize), (2, 1), (3, 2), (7, 5)];
        let mut m = [[0.0_f64; 4]; 4];
        for (c, &(col, _)) in axes.iter().enumerate() {
            let mut sd = [0.0_f64; 8];
            sd[col] = 1.0;
            let (cij, _) = compute_cij_sij(&ctx, &[[0.0; 8], sd], &[[0.0; 8]; 2], 1);
            for (r, &(_, row)) in axes.iter().enumerate() {
                m[r][c] = cij[1][row];
            }
        }
        let mut worst = 0.0_f64;
        for r in 0..4 {
            for c in 0..4 {
                let s = m[r][c].abs().max(m[c][r].abs());
                let rel = if s > 0.0 {
                    (m[r][c] + m[c][r]).abs() / s
                } else {
                    0.0
                };
                assert!(rel < 1e-15, "block ({r},{c}): {rel:.3e}");
                worst = worst.max(rel);
            }
        }
        std::println!("Poisson antisymmetry on the (a, k, h, λ) block: worst {worst:.2e}");
    }

    /// The Cʲ,λ/Sʲ,λ series really is the λ-derivative of the generating function it belongs to.
    ///
    /// `S(λ) = Σ(Cʲ cos jF + Sʲ sin jF)` with the coefficients independent of λ, so
    /// `∂S/∂λ = (∂F/∂λ)·Σ j(Sʲ cos jF − Cʲ sin jF)` with `∂F/∂λ = 1/(1 − k cos F − h sin F)` from the
    /// Kepler equation — exact, no difference quotient. The series `C⁰,λ + Σ(Cʲ,λ cos jF + Sʲ,λ sin jF)`
    /// is that same derivative re-expanded, and it is only *approximately* equal because both sides
    /// are truncated at `maxFreqF` while `∂F/∂λ` is not a finite Fourier polynomial. The measured
    /// agreement is the bound: this is the only assertion in the file that reaches the λ-coefficient
    /// *values* rather than the relations between them.
    #[test]
    fn the_lambda_coefficients_reproduce_the_generating_function_derivative() {
        const N: usize = 512;
        let tb = moon();
        let fr = fourier_at(CVARS, &tb);
        let (gc, gs) = generating_at(CVARS, &tb);
        let (mut worst, mut amp) = (0.0_f64, 0.0_f64);
        for i in 0..N {
            let lm = core::f64::consts::TAU * i as f64 / N as f64;
            let f = mean_to_eccentric(CK, CH, lm);
            let dfdl = 1.0 / (1.0 - CK * cos(f) - CH * sin(f));
            let mut exact = 0.0;
            let mut series = fr.c0lambda();
            for j in 1..=C_JMAX {
                let (jf, jr) = (j as f64 * f, j as f64);
                let (c, s) = (cos(jf), sin(jf));
                exact += jr * (gs[j][0] * c - gc[j][0] * s) * dfdl;
                series += fr.cjlambda(j as i64) * c + fr.sjlambda(j as i64) * s;
            }
            worst = worst.max((exact - series).abs());
            amp = amp.max(exact.abs());
        }
        // 1e-4 against a measured 4.35e-5: the truncation is what it is, and this is the whole
        // resolution the identity has to offer.
        let rel = worst / amp;
        assert!(rel < 1e-4, "λ-series vs ∂S/∂λ: {rel:.3e} of the amplitude");
        std::println!(
            "Cʲ,λ/Sʲ,λ series vs ∂S/∂λ: worst {rel:.2e} of the amplitude \
             (the truncation of ∂F/∂λ at maxFreqF)"
        );
    }
}
