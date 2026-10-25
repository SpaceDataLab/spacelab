//! The DSST **Gaussian** (drag / solar-radiation-pressure) short-period corrections (Danielson
//! §2.5.3), turning DSST mean equinoctial elements into osculating ones — the numerically-averaged
//! analogue of the [`super::short_period`] (zonal) and [`super::third_body_short_period`] ones.
//!
//! The correction ηᵢ of element `i` is the zero-mean periodic solution of `n·∂ηᵢ/∂λ = Fᵢ − ⟨Fᵢ⟩`,
//! with `+ (3/2)(n/a)·η_a` on the left for the mean longitude, where `Fᵢ` is the element's Gauss rate
//! `(δaᵢ/δv)·q` on the Keplerian orbit of the mean elements and `⟨Fᵢ⟩` its mean over λ. The force
//! is known only at quadrature nodes, so `F` enters through its Fourier series in the **true**
//! longitude L, and the equation is integrated in L through `dλ/dL = 1 + 2Σₖ(ρₖ cos kL + σₖ sin kL)`:
//! * `fourier_coefficients` — `Cᵢʲ, Sᵢʲ = (1/π)∫Fᵢ·(cos jL, sin jL) dL` over the drag arc at
//!   `SHORT_PERIOD_ORDER` nodes of `quadrature`, or for SRP on the pieces and the rule of its mean
//!   rates;
//! * `compute_rho_sigma` — `ρⱼ + iσⱼ = (1 + jB)(−b)ʲ(k + ih)ʲ`, the means of `cos jL`, `sin jL` over λ
//!   (Danielson 2.5.3-(4)), `j = 1 … 3·JMAX`;
//! * `UijVij` — U₁ʲ, V₁ʲ: the product to sum of `(F − ⟨F⟩)·dλ/dL`, integrated term by term (Danielson
//!   2.5.3-(21, 22)); U₂ʲ, V₂ʲ: the same for the series of η_a, which the λ equation integrates;
//! * `StoredCoefficients` — the stored Cᵢ⁰, Cᵢʲ, Sᵢʲ and the polynomial `(L − λ)·Dᵢ¹ + (L − λ)²·Dᵢ²`
//!   (Danielson 2.5.3-(26)), and their sum at the mean state.
//!
//! A port of Orekit's `AbstractGaussianContribution` (Apache-2.0, in `NOTICE`): the structure and
//! these names follow it (`JMAX`, `u1ij`/`v1ij`/`u2ij`/`v2ij`, `k20`, `to2an`/`to4an`). Orekit's
//! is not a solution of the equation in five places, and this module departs from it there:
//! * **U₁'s third sum** carries `+Cᵏσⱼ₊ₖ`: `cos kL·sin(j + k)L` gives `+½ sin jL`. Orekit's `−`.
//! * **D¹ and D₆²** take the constant of `F − ⟨F⟩` in L, `𝒞⁰ = C⁰/2 − ⟨F⟩ = −Σⱼ(Cʲρⱼ + Sʲσⱼ)`.
//!   Orekit takes `C⁰ = (1/π)∫F dL`, and a constant rate `F ≡ c` then gets `η = −(2c/n)(L − λ)`
//!   where the solution is 0.
//! * **U₂'s last sum** has a second half that belongs to V₂. Orekit adds it to U₂.
//! * **The λ correction** carries `(3/an)·𝒞₀⁰·Σⱼ(ρⱼ cos jL + σⱼ sin jL)/j²`, the periodic part of
//!   `∫(L − λ) dλ` that `−(3/2a)∫η_a dλ` brings in. Orekit has only its `(L − λ)²` part.
//! * **The quadrature** runs at `SHORT_PERIOD_ORDER` nodes and is the periodic trapezoid on a full
//!   turn; SRP's pieces take the rule of its mean rates. Orekit runs Gauss–Legendre at the order its
//!   mean rates select, which is 12 for drag in LEO.
//!
//! `tests/dsst_short_period_identity.rs` holds the equation on drag and SRP to the round-off of the
//! test and to the two cuts of the series, which it measures.
//! `derivations/dsst_gaussian_short_period.mac` proves `ρⱼ + iσⱼ` and the product to sum and solves
//! `F = cos L` in closed form; `derivations/dsst_gaussian_short_period.gp` checks all three by other
//! routes in exact arithmetic. The tests below check the quadrature against `2Iⱼ(z)` and the
//! assembly against that closed form.
//!
//! Orekit's `ShortPeriodicsInterpolatedCoefficient` (Hermite interpolation of the coefficients
//! between integration steps) is a propagator detail: this module computes the coefficients at the
//! output mean state, where the interpolation is exact.

use std::vec::Vec;

use libm::{cos, sin};

use super::auxiliary::DsstAuxiliary;
use super::gaussian::{self, NodeState};
use super::srp::mean_to_true;

/// Maximum value of the Fourier index `j` (`AbstractGaussianContribution.JMAX`).
const JMAX: usize = 12;

/// Nodes of the Fourier quadrature, whatever order the mean rates settle on: the most
/// [`gaussian::GAUSS_ORDER`] offers. The integrand carries `cos(JMAX·L)` on top of the force's own
/// harmonics; `fourier_coefficients_of_the_exponential_are_bessel_functions` measures the rules.
const SHORT_PERIOD_ORDER: usize = gaussian::MAX_GAUSS_ORDER;

/// The six equinoctial short-period corrections `[Δa, Δk, Δh, Δq, Δp, Δλ]` (Orekit ordering) of a
/// Gaussian contribution at the given **mean** state, for the true-longitude integration window `ll`
/// (`getLLimits`) and the per-node perturbing acceleration `accel`.
///
/// Returns zeros when `ll[0] ≥ ll[1]`: for drag, the whole orbit is above the atmosphere, so `F`
/// vanishes.
#[must_use]
pub(crate) fn short_period_correction<F>(aux: &DsstAuxiliary, ll: [f64; 2], accel: F) -> [f64; 6]
where
    F: Fn(&NodeState) -> [f64; 3],
{
    short_periods(aux, ll, |l| gaussian::element_derivatives(aux, l, &accel).0)
}

/// [`short_period_correction`] with the Fourier coefficients taken on a given rule, true longitudes
/// `nodes` and their `weights`, instead of on a window. `F` is zero away from the nodes' support:
/// for SRP, the umbra between the lit pieces of the orbit. An empty rule returns zeros.
#[must_use]
pub(crate) fn short_period_correction_on<F>(
    aux: &DsstAuxiliary,
    nodes: &[f64],
    weights: &[f64],
    accel: F,
) -> [f64; 6]
where
    F: Fn(&NodeState) -> [f64; 3],
{
    short_periods_on(aux, nodes, weights, |l| {
        gaussian::element_derivatives(aux, l, &accel).0
    })
}

/// [`short_period_correction`] of the six rates `rates(L)` at true longitude `L`, zero outside `ll`.
fn short_periods(aux: &DsstAuxiliary, ll: [f64; 2], rates: impl Fn(f64) -> [f64; 6]) -> [f64; 6] {
    if ll[0] < ll[1] {
        let (nodes, weights) = quadrature(ll);
        short_periods_on(aux, &nodes, &weights, rates)
    } else {
        [0.0; 6]
    }
}

/// [`short_periods`] on the rule `nodes`, `weights`.
fn short_periods_on(
    aux: &DsstAuxiliary,
    nodes: &[f64],
    weights: &[f64],
    rates: impl Fn(f64) -> [f64; 6],
) -> [f64; 6] {
    if nodes.is_empty() {
        return [0.0; 6];
    }
    let (rho, sigma) = compute_rho_sigma(aux);
    let (ccoef, scoef) = fourier_coefficients(nodes, weights, rates);
    let uv = UijVij::new(&ccoef, &scoef, &rho, &sigma);
    StoredCoefficients::new(aux, &ccoef, &scoef, &uv, &rho, &sigma).value(aux)
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

/// The quadrature rule of the Fourier coefficients on the window `ll`, as true longitudes and weights.
///
/// A window of a full turn (no eclipse, or an orbit inside the atmosphere) takes the periodic
/// trapezoid: `F` is then periodic and analytic, so the rule is exact on every harmonic of `F` below
/// `SHORT_PERIOD_ORDER − JMAX` and aliases only the higher ones. Gauss–Legendre with as many nodes is
/// not built for a period: it lands 4.6e-10 off at an exponential atmosphere's `z = ae/H = 5` and
/// 6.7e-7 at `z = 10`. An arc is not periodic and takes Gauss–Legendre. Both windows of a full turn
/// come from `[−π + L, π + L]`, whose width is 2π to a rounding.
fn quadrature(ll: [f64; 2]) -> (Vec<f64>, Vec<f64>) {
    let width = ll[1] - ll[0];
    if width >= core::f64::consts::TAU * (1.0 - 4.0 * f64::EPSILON) {
        let step = width / SHORT_PERIOD_ORDER as f64;
        (0..SHORT_PERIOD_ORDER)
            .map(|i| (ll[0] + step * i as f64, step))
            .unzip()
    } else {
        gauss_legendre(ll, SHORT_PERIOD_ORDER)
    }
}

/// Gauss–Legendre nodes and weights of `order` points mapped to `ll`.
fn gauss_legendre(ll: [f64; 2], order: usize) -> (Vec<f64>, Vec<f64>) {
    let (points, weights) = gaussian::nodes_weights(order);
    let scale = (ll[1] - ll[0]) / 2.0;
    let shift = ll[0] + scale;
    points
        .iter()
        .zip(weights)
        .map(|(x, w)| (x * scale + shift, w * scale))
        .unzip()
}

/// The Fourier Cᵢʲ / Sᵢʲ coefficients (Orekit `FourierCjSjCoefficients`) as `ccoef[j][i]` / `scoef[j][i]`
/// with `i` the equinoctial element (0:a, 1:k, 2:h, 3:q, 4:p, 5:λ) and `j = 1 … JMAX`. Index 0 is
/// unused: the constant of `F − ⟨F⟩` follows from the others (`StoredCoefficients::new`).
///
/// For each `j` the twelve-component integrand `[cos(jL)·Fᵢ, sin(jL)·Fᵢ]` is summed over the
/// quadrature `xs`, `ws` and divided by π. The six `Fᵢ = rates(L)` are evaluated once per node and
/// reused across all `j`. The compensated (Kahan) accumulation and the `w·(cos(jL)·Fᵢ)`
/// multiplication order are those of Orekit's `GaussQuadrature.integrate`.
pub(super) fn fourier_coefficients(
    xs: &[f64],
    ws: &[f64],
    rates: impl Fn(f64) -> [f64; 6],
) -> (Vec<[f64; 6]>, Vec<[f64; 6]>) {
    let m = xs.len();
    let derivs: Vec<[f64; 6]> = xs.iter().map(|&x| rates(x)).collect();

    let oo_pi = 1.0 / core::f64::consts::PI;
    let mut ccoef = std::vec![[0.0f64; 6]; JMAX + 1];
    let mut scoef = std::vec![[0.0f64; 6]; JMAX + 1];
    for j in 1..=JMAX {
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
                // Orekit: `-cij(i, k) * sigma[j + k]`.
                for k in 1..=JMAX {
                    u += cij(i, k) * sigma[j + k] - sij(i, k) * rho[j + k];
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
                u += cij(0, k) * sigma[j + k] - sij(0, k) * rho[j + k];
                v += cij(0, k) * rho[j + k] + sij(0, k) * sigma[j + k];
            }
            u1[j][0] = -ooj * u;
            v1[j][0] = ooj * v;
        }

        // computeU2V2Coefficients (only Fourier index i = 0): the same product to sum, with U₁ʲ, V₁ʲ
        // in place of Cʲ, Sʲ.
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
                // Orekit adds this half to `u` as well.
                v += u1[j + l][0] * rho[l]
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
/// the free term Cᵢ⁰, the per-frequency Cᵢʲ / Sᵢʲ (j = 1…JMAX) and the polynomial Dᵢ¹ / Dᵢ² (Dᵢ² only
/// for the mean longitude i = 5). Danielson 2.5.3-(26).
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
        scoef: &[[f64; 6]],
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

        // 𝒞ᵢ⁰ = Cᵢ⁰/2 − ⟨Fᵢ⟩, the constant of F − ⟨F⟩ in L: ⟨cos jL⟩ = ρⱼ and ⟨sin jL⟩ = σⱼ over λ.
        let mut c0 = [0.0f64; 6];
        for (i, c0i) in c0.iter_mut().enumerate() {
            for j in 1..=JMAX {
                *c0i -= ccoef[j][i] * rho[j] + scoef[j][i] * sigma[j];
            }
        }

        let mut cij0 = [0.0f64; 6];
        let mut cij = std::vec![[0.0f64; 6]; JMAX + 1];
        let mut sij = std::vec![[0.0f64; 6]; JMAX + 1];
        let mut di1 = [0.0f64; 6];
        let mut di2 = [0.0f64; 6];

        for i in 0..6 {
            // Dᵢ¹, Dᵢ² (all j-orders above 2 are 0; Dᵢ² only for the mean longitude i = 5).
            di1[i] = -oon * c0[i];
            if i == 5 {
                di1[i] += to2an * uv.u1[0][0]; // getU1(0, 0)
            }
            di2[i] = 0.0;
            if i == 5 {
                di2[i] += -to4an * c0[0];
            }

            // Cᵢ⁰ accumulates from all Cᵢʲ / Sᵢʲ (Danielson 2.5.2-(15a)).
            cij0[i] = -di2[i] * k20;

            for j in 1..=JMAX {
                let mut c = oon * uv.u1[j][i]; // getU1(j, i)
                let mut s = oon * uv.v1[j][i]; // getV1(j, i)
                if i == 5 {
                    // −(3/2a)∫η_a dλ: U₂ʲ, V₂ʲ, and the periodic part of 𝒞₀⁰∫(L − λ) dλ,
                    // 2Σⱼ(ρⱼ cos jL + σⱼ sin jL)/j², which Orekit leaves out.
                    let jj = (j * j) as f64;
                    c += -to2an * uv.u2[j] + 2.0 * to2an * c0[0] * rho[j] / jj;
                    s += -to2an * uv.v2[j] + 2.0 * to2an * c0[0] * sigma[j] / jj;
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

    use core::f64::consts::PI;

    use super::super::drag::{Atmosphere, DsstAtmosphericDrag};
    use super::super::srp::DsstSolarRadiationPressure;
    use super::{
        DsstAuxiliary, JMAX, SHORT_PERIOD_ORDER, compute_rho_sigma, cos, fourier_coefficients,
        gauss_legendre, gaussian, quadrature, short_periods, sin,
    };

    // Orekit 13's OSCULATING `DSSTPropagator` (`tools/orekit-oracle`, orekit_dsst_srp_sp.csv /
    // orekit_dsst_drag_sp.csv), seeded from a MEAN equinoctial LEO with one Gaussian force each and
    // sampled at t > 0: a DSST OSCULATING propagator returns the raw mean state at its epoch. Fed
    // Orekit's own mean state, `osc − mean` is the short-period assembly alone. The two tests that read
    // these are comparators: they report a distance and decide nothing.
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

    /// `(max |Δa|, max |Δ|)` over the samples, `Δ` the other five elements.
    fn distance(pairs: impl Iterator<Item = ([f64; 6], [f64; 6])>) -> (f64, f64) {
        pairs.fold((0.0_f64, 0.0_f64), |(da, rest), (got, want)| {
            let others = (1..6).fold(0.0_f64, |m, i| nan_max(m, (got[i] - want[i]).abs()));
            (nan_max(da, (got[0] - want[0]).abs()), nan_max(rest, others))
        })
    }

    /// The distance to Orekit's SRP short periods on its fixture: 2.7e-5 m in `a`, of a ~62 mm swing,
    /// and 2.5e-10 in the other elements, from the departures the module doc lists and from Orekit's
    /// shadow cylinder (the `srp` module doc).
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn srp_short_period_distance_to_orekit() {
        let srp = DsstSolarRadiationPressure::new(CR, SRP_AOM, MU, AE);
        let (da, rest) =
            distance((0..4).map(|k| (srp.osculating(SRP_MEAN[k], SRP_SUN[k]), SRP_OSC[k])));
        std::println!("SRP short periods vs Orekit 13: |Δa| {da:.2e} m, others {rest:.2e}");
    }

    /// The distance to Orekit's drag short periods on its fixture, a full turn inside a constant
    /// co-rotating atmosphere: 4.4e-2 m in `a`, four times Orekit's own 1.1 cm swing, and 8.1e-9 in the
    /// other elements. The `a` distance is Orekit's `−(C⁰/n)(L − λ)`: drag's secular rate makes
    /// `C⁰ ≈ 2⟨ȧ⟩`, and `(2/n)·1.19e-3 m/s·2e` is 4.4 cm. Before the departures it was 6e-7 m and
    /// below 5e-11.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn drag_short_period_distance_to_orekit() {
        let drag = DsstAtmosphericDrag::with_rbar(CD, DRAG_AOM, MU, RBAR);
        let atmosphere = CoRotatingConstant {
            rho: RHO,
            omega: OMEGA,
        };
        let (da, rest) =
            distance((0..4).map(|k| (drag.osculating(DRAG_MEAN[k], &atmosphere), DRAG_OSC[k])));
        std::println!("drag short periods vs Orekit 13: |Δa| {da:.2e} m, others {rest:.2e}");
    }

    /// `I₀(z) … I₁₂(z)` at z = 1, 5, 10: Arb balls (FLINT 3.4 `arb_hypgeom_bessel_i`, 256 bits, radius
    /// below 1e-70) printed to 21 digits. Maxima's 60-digit `bessel_i` lies inside every ball.
    #[rustfmt::skip]
    const BESSEL_I: [(f64, [f64; JMAX + 1]); 3] = [
        (1.0, [
            1.26606587775200833560e+0, 5.65159103992485027208e-1, 1.35747669767038281183e-1,
            2.21684249243319024763e-2, 2.73712022104686632514e-3, 2.71463155956971875181e-4,
            2.24886614771475733273e-5, 1.59921823120099525293e-6, 9.96062403336397862981e-8,
            5.51838586275867216308e-9, 2.75294803983687362524e-10, 1.24897830849249126136e-11,
            5.19576115339285025250e-13,
        ]),
        (5.0, [
            2.72398718236044468945e+1, 2.43356421424505271991e+1, 1.75056149666242360149e+1,
            1.03311501691511383872e+1, 5.10823476364286995021e+0, 2.15797454732254646690e+0,
            7.92285668997777016402e-1, 2.56488941727881627538e-1, 7.41166321597084592968e-2,
            1.93157188168145577877e-2, 4.58004441917605126119e-3, 9.95541140110352742935e-4,
            1.99663402690499192271e-4,
        ]),
        (10.0, [
            2.81571662846625447147e+3, 2.67098830370125465434e+3, 2.28151896772600354060e+3,
            1.75838071661085323810e+3, 1.22649053775949159774e+3, 7.77188286403259959907e+2,
            4.49302251356231637834e+2, 2.38025584775781994506e+2, 1.16066432670136845525e+2,
            5.23192925035630416662e+1, 2.18917061637233705259e+1, 8.53588017611630061436e+0,
            3.11276977626750917434e+0,
        ]),
    ];

    /// `(1/π)∫e^{z cos(L − ϖ)}·(cos jL, sin jL) dL = 2Iⱼ(z)·(cos jϖ, sin jϖ)` over a turn: the
    /// density profile of an exponential atmosphere along an orbit, `z = ae/H`, each element at its
    /// own `ϖ`. The shipped trapezoid is within 1.7e-15 of `2I₀(z)` up to z = 10. Gauss–Legendre at
    /// the same 48 nodes is 4.6e-10 off at z = 5 and 6.7e-7 at z = 10, and at the 12 nodes Orekit's
    /// drag selects in LEO the coefficients are gone: 0.80 off at z = 1. (G3)
    #[test]
    fn fourier_coefficients_of_the_exponential_are_bessel_functions() {
        let worst = |(xs, ws): (Vec<f64>, Vec<f64>), z: f64, bessel: &[f64; JMAX + 1]| -> f64 {
            let varpi = |e: usize| 0.3 + 0.9 * e as f64;
            let (c, s) = fourier_coefficients(&xs, &ws, |l| {
                core::array::from_fn(|e| libm::exp(z * cos(l - varpi(e))))
            });
            let mut worst = 0.0_f64;
            for j in 1..=JMAX {
                for e in 0..6 {
                    let jw = j as f64 * varpi(e);
                    let dc = (c[j][e] - 2.0 * bessel[j] * cos(jw)).abs();
                    let ds = (s[j][e] - 2.0 * bessel[j] * sin(jw)).abs();
                    worst = nan_max(worst, nan_max(dc, ds));
                }
            }
            worst / (2.0 * bessel[0])
        };
        let turn = [-PI + 0.4, PI + 0.4];
        for (z, bessel) in &BESSEL_I {
            let shipped = worst(quadrature(turn), *z, bessel);
            let gauss = worst(gauss_legendre(turn, SHORT_PERIOD_ORDER), *z, bessel);
            let twelve = worst(gauss_legendre(turn, 12), *z, bessel);
            std::println!(
                "z = {z}: trapezoid {shipped:.2e}, Gauss-Legendre {gauss:.2e} at {SHORT_PERIOD_ORDER} nodes and {twelve:.2e} at 12"
            );
            assert!(shipped < 4e-15, "z = {z}: {shipped:.3e} of 2I₀(z)");
        }
        let gauss = worst(
            gauss_legendre(turn, SHORT_PERIOD_ORDER),
            5.0,
            &BESSEL_I[1].1,
        );
        assert!(gauss > 1e-11, "Gauss-Legendre at z = 5: {gauss:.3e}");
    }

    /// The eccentric longitude `F` of the mean longitude `λ = F − k sin F + h cos F`.
    fn eccentric_longitude(k: f64, h: f64, lam: f64) -> f64 {
        let mut f = lam;
        for _ in 0..50 {
            f -= (f - k * sin(f) + h * cos(f) - lam) / (1.0 - k * cos(f) - h * sin(f));
        }
        f
    }

    /// A constant rate has no short period: `F ≡ c` is its own mean, at every element. Orekit's `D¹`
    /// takes `C⁰ = 2c` and gives `η = −(2c/n)(L − λ)`, the equation of the centre times `2c/n`. (G3)
    #[test]
    fn a_constant_rate_has_no_short_period() {
        let rates = [3.0e-3, -2.0e-9, 1.5e-9, 4.0e-10, -7.0e-10, 2.5e-9];
        for (k, h) in [(0.008, -0.006), (0.08, -0.06)] {
            for lam in [0.3, 1.9, 4.4] {
                let aux = DsstAuxiliary::new(7.2e6, k, h, 0.3, 0.35, lam, MU);
                let eta = short_periods(&aux, [-PI, PI], |_| rates);
                for (i, (&got, &c)) in eta.iter().zip(&rates).enumerate() {
                    assert!(
                        got.abs() < 1e-14 * c.abs() / aux.n,
                        "e = {}, λ = {lam}, element {i}: η = {got:.3e}, c/n = {:.3e}",
                        libm::hypot(k, h),
                        c / aux.n
                    );
                }
            }
        }
    }

    /// `F = cos L` in one element has the closed-form short period
    /// `η = (P sin F + Q cos F + hB/2)/n`, `P = 1 − k² − bh²`, `Q = bBhk`, `F` the eccentric longitude
    /// (`derivations/dsst_gaussian_short_period.mac` solves it). In the semi-major axis it also drives
    /// the mean longitude, `η_λ = −(3/2an)·(G − (kP − 3hQ)/4)` with `G` the antiderivative of
    /// `n·η_a·dλ/dF` the same file derives. The assembly is the series in L cut at `JMAX`, which drops
    /// `ρ₁₃` and above, below 1e-15 at e = 0.1: measured 1.2e-15 of the amplitude at e = 0.01 and
    /// 1.9e-15 at e = 0.1, both round-off.
    #[test]
    fn a_cosine_rate_has_the_closed_form_short_period() {
        for (k, h) in [(0.008, -0.006), (0.08, -0.06)] {
            let mut worst = 0.0_f64;
            for lam in (0..16).map(|i| 0.1 + PI * f64::from(i) / 8.0) {
                let aux = DsstAuxiliary::new(7.2e6, k, h, 0.3, 0.35, lam, MU);
                let (n, a, bb) = (aux.n, aux.sma, aux.big_b);
                let b = 1.0 / (1.0 + bb);
                let (p, q, r) = (1.0 - k * k - b * h * h, b * bb * h * k, 0.5 * h * bb);
                let (sf, cf) = libm::sincos(eccentric_longitude(k, h, lam));
                let eta = (p * sf + q * cf + r) / n;
                let g = -p * cf + q * sf - 0.5 * (k * p + h * q) * sf * sf
                    + 0.5 * (h * p - k * q) * sf * cf
                    - k * r * sf
                    + h * r * cf;
                let eta_lambda = -1.5 / (a * n) * (g - 0.25 * (k * p - 3.0 * h * q));
                for (driven, want) in [
                    (1, [0.0, eta, 0.0, 0.0, 0.0, 0.0]),
                    (0, [eta, 0.0, 0.0, 0.0, 0.0, eta_lambda]),
                ] {
                    let got = short_periods(&aux, [-PI, PI], |l| {
                        let mut f = [0.0; 6];
                        f[driven] = cos(l);
                        f
                    });
                    let scale = [1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n, 1.5 / (a * n)];
                    for i in 0..6 {
                        worst = nan_max(worst, (got[i] - want[i]).abs() / scale[i]);
                    }
                }
            }
            std::println!("F = cos L, e = {:.3}: worst {worst:.2e}", libm::hypot(k, h));
            assert!(worst < 1e-14, "e = {}: {worst:.3e}", libm::hypot(k, h));
        }
    }

    /// `ρⱼ + iσⱼ = (1 + jB)(−b)ʲ(k + ih)ʲ`, `b = 1/(1 + B)` — Danielson 2.5.3-(4). (G3)
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
