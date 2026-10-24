//! Shared **Gauss-quadrature** numerical averaging for the DSST non-conservative (Gaussian)
//! contributions — solar radiation pressure and atmospheric drag.
//!
//! Direct port of Orekit's `AbstractGaussianContribution` mean-element-rate path: the equinoctial
//! Gauss variational equations `dyᵢ/dt = (δaᵢ/δv)·q` are averaged over one orbit by Gauss–Legendre
//! quadrature in the true longitude `L`, where `q` is the perturbing acceleration. Restricted to the
//! analytic **mean-element rates**; the short-period corrections are out of scope.
//!
//! The pieces mirror Orekit's inner classes:
//! * [`NodeState`] — the satellite position/velocity at a quadrature node, fed to the caller's
//!   acceleration closure (Orekit's `IntegrableFunction` shifts the orbit to the node's true
//!   longitude; the equinoctial position is `X·f + Y·g`, the velocity `Ẋ·f + Ẏ·g`);
//! * [`value`] — the `IntegrableFunction.value(L)` mean-mode integrand `roa²·(δaᵢ/δv)·q`, the six
//!   equinoctial partial-derivative vectors dotted with the acceleration;
//! * [`integrate`] — the `GaussQuadrature` change-of-variable + compensated (Kahan) summation;
//! * [`mean_element_rates`] — one `getMeanElementRate` call: quadrature at the maximum order (48),
//!   scaled by `1/(2πB)`. Orekit's first `getMeanElementRate` returns the order-48 result, then
//!   caches a smaller order for later steps (see [`select_order`]).
//!
//! Naming (`GAUSS_ORDER`, `POINTS_*`/`WEIGHTS_*`, `ton2a`, `ooA`, `ooAB`, `co2AB`, `ooBpo`, `ooMu`,
//! `naob`, `roa`, `getAoV`/`getKoV`/`getHoV`/`getQoV`/`getPoV`/`getLoV`) follows Orekit for
//! auditability.
//!
//! # Checked against the average that defines the rates
//!
//! `tests/dsst_short_period_identity.rs` checks [`variational_matrix`] against central differences
//! of a Cartesian → equinoctial conversion written in the test: 1.1e-9 at worst, the (δv/v)² error of
//! the differences themselves. It checks [`mean_element_rates`] against `(1/2π)∫F dλ`, taken by
//! Simpson's rule in the mean longitude, with F from a drag-like and an SRP-like force written in
//! the test: 2.2e-10 at worst, over full turns and over an arc of an orbit with e = 0.07.

use libm::{cos, sin, sqrt};

use super::auxiliary::{DsstAuxiliary, I};

/// Available orders for Gauss quadrature (`AbstractGaussianContribution.GAUSS_ORDER`).
pub const GAUSS_ORDER: [usize; 7] = [12, 16, 20, 24, 32, 40, 48];

/// Maximum quadrature order — Orekit's `GAUSS_ORDER[MAX_ORDER_RANK]`, used for the returned rate of
/// the first `getMeanElementRate` call.
pub const MAX_GAUSS_ORDER: usize = 48;

/// Satellite kinematics at a quadrature node, in the orbit (inertial) frame — the argument to the
/// perturbing-acceleration closure. `pos = X·f + Y·g`, `vel = Ẋ·f + Ẏ·g` (Orekit recomposes the
/// orbit at the node's true longitude with the epoch held fixed; that position/velocity is exactly
/// these equinoctial combinations).
#[derive(Debug, Clone, Copy)]
pub struct NodeState {
    /// Satellite position, m.
    pub pos: [f64; 3],
    /// Satellite velocity, m/s.
    pub vel: [f64; 3],
}

/// The mean-element rates `[da, dk, dh, dq, dp, dλ]` (Orekit equinoctial ordering) of a Gaussian
/// contribution whose perturbing acceleration is `accel(node)`, averaged over `[low, high]` in the
/// true longitude `L`. Returns zeros when `low ≥ high` (Orekit skips the integral).
///
/// Quadrature is done at the maximum order (48), matching the value Orekit's first
/// `getMeanElementRate` returns (it caches a smaller order for subsequent steps — see
/// [`select_order`]). The `1/(2πB)` scaling is Orekit's post-integration `coef`.
#[must_use]
pub fn mean_element_rates<F>(aux: &DsstAuxiliary, low: f64, high: f64, accel: F) -> [f64; 6]
where
    F: Fn(&NodeState) -> [f64; 3],
{
    if low < high {
        let mut rate = integrate(MAX_GAUSS_ORDER, aux, low, high, &accel);
        let coef = 1.0 / (2.0 * core::f64::consts::PI * aux.big_b);
        for r in &mut rate {
            *r *= coef;
        }
        rate
    } else {
        [0.0; 6]
    }
}

/// Reproduce Orekit's Gauss-order selection (`getMeanElementRate`'s `isDirty` pass): from the lowest
/// order up, the first whose rates differ from the order-48 rates by less than `threshold` (weighted
/// like `getRatesDiff`: `|Δda|/a` for the semi-major axis, `|Δ|` for the rest). Returns 48 if none
/// qualifies. In a propagation Orekit fixes this order once and reuses it every step.
#[must_use]
pub fn select_order<F>(aux: &DsstAuxiliary, low: f64, high: f64, accel: F, threshold: f64) -> usize
where
    F: Fn(&NodeState) -> [f64; 3],
{
    if low < high {
        let coef = 1.0 / (2.0 * core::f64::consts::PI * aux.big_b);
        let mut reference = integrate(MAX_GAUSS_ORDER, aux, low, high, &accel);
        for r in &mut reference {
            *r *= coef;
        }
        // GAUSS_ORDER[0 .. MAX_ORDER_RANK] — the max order itself is excluded from the search.
        for &order in &GAUSS_ORDER[..GAUSS_ORDER.len() - 1] {
            let mut current = integrate(order, aux, low, high, &accel);
            for r in &mut current {
                *r *= coef;
            }
            let mut max_diff = (reference[0] - current[0]).abs() / aux.sma;
            for i in 1..6 {
                max_diff = max_diff.max((reference[i] - current[i]).abs());
            }
            if max_diff < threshold {
                return order;
            }
        }
    }
    MAX_GAUSS_ORDER
}

/// Gauss–Legendre quadrature of the mean-mode integrand over `[low, high]` at the given `order`,
/// **without** the `1/(2πB)` scaling (Orekit's `GaussQuadrature.integrate` result). The `[-1, 1]`
/// nodes/weights are affinely mapped to `[low, high]` and summed with the same compensated (Kahan)
/// accumulation as Orekit's `basicIntegrate`, so the result matches term-for-term.
///
/// # Panics
///
/// If `order` is not one of [`GAUSS_ORDER`].
#[must_use]
pub fn integrate<F>(order: usize, aux: &DsstAuxiliary, low: f64, high: f64, accel: &F) -> [f64; 6]
where
    F: Fn(&NodeState) -> [f64; 3],
{
    let (points, weights) = nodes_weights(order);

    // GaussQuadrature.transform: map [-1, 1] → [low, high].
    let scale = (high - low) / 2.0;
    let shift = low + scale;

    // basicIntegrate: compensated (Kahan) summation, first node seeds the running sum.
    let x0 = points[0] * scale + shift;
    let w0 = weights[0] * scale;
    let v0 = value(aux, x0, accel);
    let mut s = [0.0_f64; 6];
    for j in 0..6 {
        s[j] = w0 * v0[j];
    }
    let mut c = [0.0_f64; 6];
    for i in 1..points.len() {
        let x = points[i] * scale + shift;
        let w = weights[i] * scale;
        let v = value(aux, x, accel);
        for j in 0..6 {
            let y = w * v[j] - c[j];
            let t = s[j] + y;
            c[j] = (t - s[j]) - y;
            s[j] = t;
        }
    }
    s
}

/// Orekit's `IntegrableFunction.value(L)` in mean mode: the six equinoctial rate integrands
/// `roa²·(δaᵢ/δv)·q` at true longitude `L`, where `q = accel(node)` is the perturbing acceleration.
///
/// The satellite position/velocity at `L` are the equinoctial combinations `X·f + Y·g` and
/// `Ẋ·f + Ẏ·g`; the six partial-derivative vectors `δaᵢ/δv` (`getAoV`/`getKoV`/`getHoV`/`getQoV`/
/// `getPoV`/`getLoV`) are assembled in the equinoctial f/g/w basis and dotted with `q`.
#[must_use]
pub fn value<F>(aux: &DsstAuxiliary, x: f64, accel: &F) -> [f64; 6]
where
    F: Fn(&NodeState) -> [f64; 3],
{
    let (deriv, roa2) = element_derivatives(aux, x, accel);
    [
        roa2 * deriv[0],
        roa2 * deriv[1],
        roa2 * deriv[2],
        roa2 * deriv[3],
        roa2 * deriv[4],
        roa2 * deriv[5],
    ]
}

/// The Gauss variational matrix `B(œ)` at true longitude `L`: the `6 × 3` Jacobian of the
/// equinoctial element rates with respect to an applied acceleration.
///
/// The acceleration is in the **orbit (inertial) frame**, the same one [`NodeState`] is expressed in
/// and the one every DSST force returns its acceleration in — *not* the equinoctial `(f, g, w)`
/// triad, which is a basis expressed in that frame rather than a frame to resolve into. Units follow
/// [`NodeState`]: metres, so `B` maps m/s² to element rates.
///
/// `œ̇ = B(œ) u` in the DSST's element order `[a, k, h, q, p, λ]`. This is the *osculating* variation
/// of parameters, not an orbit average — the same `δaᵢ/δv` vectors [`value`] integrates, evaluated
/// at one point instead of quadratured over a revolution. A feedback law that acts on elements needs
/// exactly this, and it is recovered here rather than written a second time.
///
/// Built by evaluating `element_derivatives` against the three unit accelerations. That looks
/// wasteful and is exact: the extended-precision `dot_ext` of a vector with a unit vector is a
/// single exact product plus two exact zeros, so the columns come back bit-identical to the
/// `δaᵢ/δv` components the shared kernel computed. Refactoring the kernel to hand back the six
/// vectors instead would change the Dekker two-product sums that are matched against Hipparchus, so
/// it is not done.
#[must_use]
pub fn variational_matrix(aux: &DsstAuxiliary, true_longitude: f64) -> [[f64; 3]; 6] {
    let mut b = [[0.0; 3]; 6];
    for axis in 0..3 {
        let mut unit = [0.0; 3];
        unit[axis] = 1.0;
        let (column, _) = element_derivatives(aux, true_longitude, &|_: &NodeState| unit);
        for (element, value) in column.into_iter().enumerate() {
            b[element][axis] = value;
        }
    }
    b
}

/// The six raw equinoctial rate integrands `deriv[i] = (δaᵢ/δv)·q` at true longitude `L`, together
/// with the mean-mode weight `roa² = (r/a)²`. The mean-element-rate integrand ([`value`]) is
/// `roa²·deriv[i]`; the Gaussian **short-period** Fourier coefficients instead weight `deriv[i]` by
/// `cos(jL)`/`sin(jL)` — Orekit's `IntegrableFunction.value` in short-periodic mode. This shared core
/// returns the unweighted `deriv` (and `roa²`) so both callers reuse the identical per-node kinematics
/// and the six extended-precision δaᵢ/δv dot products.
#[must_use]
pub(crate) fn element_derivatives<F>(aux: &DsstAuxiliary, x: f64, accel: &F) -> ([f64; 6], f64)
where
    F: Fn(&NodeState) -> [f64; 3],
{
    let cos_l = cos(x);
    let sin_l = sin(x);

    // roa = r/a from the equinoctial conic equation; roa2 is the mean-rate weight.
    let b2 = aux.big_b * aux.big_b;
    let roa = b2 / (1.0 + aux.h * sin_l + aux.k * cos_l);
    let roa2 = roa * roa;
    let r = aux.sma * roa;

    // Position/velocity components along the equinoctial f and g vectors.
    let x_pos = r * cos_l;
    let y_pos = r * sin_l;
    let naob = aux.n * aux.sma / aux.big_b;
    let x_dot = -naob * (aux.h + sin_l);
    let y_dot = naob * (aux.k + cos_l);

    // Hipparchus builds every f/g/w combination and dot product with extended-precision
    // `linearCombination` (Dekker two-product); reproduced so the heavily-cancelled `da` matches.
    let f = aux.f_vec;
    let g = aux.g_vec;
    let w = aux.w_vec;
    let pos = [
        linear_combination2(x_pos, f[0], y_pos, g[0]),
        linear_combination2(x_pos, f[1], y_pos, g[1]),
        linear_combination2(x_pos, f[2], y_pos, g[2]),
    ];
    let vel = [
        linear_combination2(x_dot, f[0], y_dot, g[0]),
        linear_combination2(x_dot, f[1], y_dot, g[1]),
        linear_combination2(x_dot, f[2], y_dot, g[2]),
    ];

    let acc = accel(&NodeState { pos, vel });

    // Gaussian-context factors (AbstractGaussianContributionContext).
    let ton2a = 2.0 / (aux.n * aux.n * aux.sma);
    let oo_a = 1.0 / aux.big_a;
    let oo_ab = aux.ooab;
    let co2ab = aux.co2ab;
    let oo_bpo = 1.0 / (1.0 + aux.big_b);
    let oo_mu = 1.0 / aux.mu;

    let iqy_m_px = I * aux.q * y_pos - aux.p * x_pos;

    // δa/δv = ton2a · v  [getAoV]; the scalar-vector is naive, the dot is extended.
    let aov = [ton2a * vel[0], ton2a * vel[1], ton2a * vel[2]];
    let deriv0 = dot_ext(aov, acc);

    // δk/δv (dex/dv) = -kf·f + kg·g - kw·w  [getKoV]; kw uses h.
    let kf_k = y_pos * y_dot * oo_mu;
    let kg_k = (2.0 * x_pos * y_dot - x_dot * y_pos) * oo_mu;
    let kw_k = aux.h * iqy_m_px * oo_ab;
    let kov = [
        linear_combination3(-kf_k, f[0], kg_k, g[0], -kw_k, w[0]),
        linear_combination3(-kf_k, f[1], kg_k, g[1], -kw_k, w[1]),
        linear_combination3(-kf_k, f[2], kg_k, g[2], -kw_k, w[2]),
    ];
    let deriv1 = dot_ext(kov, acc);

    // δh/δv (dey/dv) = kf·f - kg·g + kw·w  [getHoV]; kw uses k.
    let kf_h = (2.0 * x_dot * y_pos - x_pos * y_dot) * oo_mu;
    let kg_h = x_pos * x_dot * oo_mu;
    let kw_h = aux.k * iqy_m_px * oo_ab;
    let hov = [
        linear_combination3(kf_h, f[0], -kg_h, g[0], kw_h, w[0]),
        linear_combination3(kf_h, f[1], -kg_h, g[1], kw_h, w[1]),
        linear_combination3(kf_h, f[2], -kg_h, g[2], kw_h, w[2]),
    ];
    let deriv2 = dot_ext(hov, acc);

    // δq/δv (dhx/dv) = I·co2AB·X · w  [getQoV].
    let q_coef = I * co2ab * x_pos;
    let qov = [q_coef * w[0], q_coef * w[1], q_coef * w[2]];
    let deriv3 = dot_ext(qov, acc);

    // δp/δv (dhy/dv) = co2AB·Y · w  [getPoV].
    let p_coef = co2ab * y_pos;
    let pov = [p_coef * w[0], p_coef * w[1], p_coef * w[2]];
    let deriv4 = dot_ext(pov, acc);

    // δλ/δv = -2·ooA·pos + ooBpo·(k·hov - h·kov) + (I·q·Y - p·X)·ooA·w  [getLoV].
    let v2 = [
        linear_combination2(aux.k, hov[0], -aux.h, kov[0]),
        linear_combination2(aux.k, hov[1], -aux.h, kov[1]),
        linear_combination2(aux.k, hov[2], -aux.h, kov[2]),
    ];
    let lw = iqy_m_px * oo_a;
    let m2oa = -2.0 * oo_a;
    let lov = [
        linear_combination3(m2oa, pos[0], oo_bpo, v2[0], lw, w[0]),
        linear_combination3(m2oa, pos[1], oo_bpo, v2[1], lw, w[1]),
        linear_combination3(m2oa, pos[2], oo_bpo, v2[2], lw, w[2]),
    ];
    let deriv5 = dot_ext(lov, acc);

    ([deriv0, deriv1, deriv2, deriv3, deriv4, deriv5], roa2)
}

/// Node points and weights for a supported Gauss order.
///
/// # Panics
///
/// If `order` is not one of [`GAUSS_ORDER`].
pub(crate) fn nodes_weights(order: usize) -> (&'static [f64], &'static [f64]) {
    match order {
        12 => (&POINTS_12, &WEIGHTS_12),
        16 => (&POINTS_16, &WEIGHTS_16),
        20 => (&POINTS_20, &WEIGHTS_20),
        24 => (&POINTS_24, &WEIGHTS_24),
        32 => (&POINTS_32, &WEIGHTS_32),
        40 => (&POINTS_40, &WEIGHTS_40),
        48 => (&POINTS_48, &WEIGHTS_48),
        _ => panic!("unsupported Gauss order {order}"),
    }
}

/// Euclidean norm of a 3-vector.
#[must_use]
pub fn norm(v: [f64; 3]) -> f64 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// Dot product of two 3-vectors.
#[must_use]
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors, each component an extended-precision `linearCombination2`
/// (Hipparchus `Vector3D.crossProduct`).
#[must_use]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        linear_combination2(a[1], b[2], -a[2], b[1]),
        linear_combination2(a[2], b[0], -a[0], b[2]),
        linear_combination2(a[0], b[1], -a[1], b[0]),
    ]
}

/// Extended-precision dot product of two 3-vectors — Hipparchus `Vector3D.dotProduct`
/// (`linearCombination3`).
#[must_use]
pub fn dot_ext(a: [f64; 3], b: [f64; 3]) -> f64 {
    linear_combination3(a[0], b[0], a[1], b[1], a[2], b[2])
}

/// Dekker split of a double into a 27-bit high part (Hipparchus masks the low 27 mantissa bits).
#[inline]
fn split_high(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0xFFFF_FFFF_F800_0000)
}

/// Dekker two-product: `(high, low)` with `high = fl(a·b)` and `a·b = high + low` exactly.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let a_high = split_high(a);
    let a_low = a - a_high;
    let b_high = split_high(b);
    let b_low = b - b_high;
    let high = a * b;
    let low = a_low * b_low - (((high - a_high * b_high) - a_low * b_high) - a_high * b_low);
    (high, low)
}

/// Extended-precision `a1·b1 + a2·b2` — Hipparchus `MathArrays.linearCombination(a1,b1,a2,b2)`
/// (two-products summed with a compensated two-sum, falling back to the naive form on NaN/zero).
#[must_use]
pub fn linear_combination2(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    let (prod1_high, prod1_low) = two_product(a1, b1);
    let (prod2_high, prod2_low) = two_product(a2, b2);

    let sum = prod1_high + prod2_high;
    let z = sum - prod2_high;
    let sum_low = (prod2_high - (sum - z)) + (prod1_high - z);

    let result = sum + ((prod1_low + prod2_low) + sum_low);
    if result.is_nan() || result == 0.0 {
        a1 * b1 + a2 * b2
    } else {
        result
    }
}

/// Extended-precision `a1·b1 + a2·b2 + a3·b3` — Hipparchus
/// `MathArrays.linearCombination(a1,b1,a2,b2,a3,b3)`.
#[must_use]
pub fn linear_combination3(a1: f64, b1: f64, a2: f64, b2: f64, a3: f64, b3: f64) -> f64 {
    let (prod1_high, prod1_low) = two_product(a1, b1);
    let (prod2_high, prod2_low) = two_product(a2, b2);
    let (prod3_high, prod3_low) = two_product(a3, b3);

    let sum12 = prod1_high + prod2_high;
    let z12 = sum12 - prod2_high;
    let low12 = (prod2_high - (sum12 - z12)) + (prod1_high - z12);

    let sum123 = sum12 + prod3_high;
    let z123 = sum123 - prod3_high;
    let low123 = (prod3_high - (sum123 - z123)) + (sum12 - z123);

    let result = sum123 + ((((prod1_low + prod2_low) + prod3_low) + low12) + low123);
    if result.is_nan() || result == 0.0 {
        a1 * b1 + a2 * b2 + a3 * b3
    } else {
        result
    }
}

// Gauss–Legendre nodes/weights for the supported orders, transcribed verbatim from Orekit's
// `AbstractGaussianContribution` (`P_n`/`W_n`), machine-generated from the Java source to avoid
// truncation.
#[rustfmt::skip]
const POINTS_12: [f64; 12] = [-0.98156063424671910000, -0.90411725637047490000, -0.76990267419430470000, -0.58731795428661740000, -0.36783149899818024000, -0.12523340851146890000, 0.12523340851146890000, 0.36783149899818024000, 0.58731795428661740000, 0.76990267419430470000, 0.90411725637047490000, 0.98156063424671910000];
#[rustfmt::skip]
const POINTS_16: [f64; 16] = [-0.98940093499164990000, -0.94457502307323260000, -0.86563120238783160000, -0.75540440835500310000, -0.61787624440264380000, -0.45801677765722737000, -0.28160355077925890000, -0.09501250983763745000, 0.09501250983763745000, 0.28160355077925890000, 0.45801677765722737000, 0.61787624440264380000, 0.75540440835500310000, 0.86563120238783160000, 0.94457502307323260000, 0.98940093499164990000];
#[rustfmt::skip]
const POINTS_20: [f64; 20] = [-0.99312859918509490000, -0.96397192727791390000, -0.91223442825132600000, -0.83911697182221890000, -0.74633190646015080000, -0.63605368072651510000, -0.51086700195082700000, -0.37370608871541955000, -0.22778585114164507000, -0.07652652113349734000, 0.07652652113349734000, 0.22778585114164507000, 0.37370608871541955000, 0.51086700195082700000, 0.63605368072651510000, 0.74633190646015080000, 0.83911697182221890000, 0.91223442825132600000, 0.96397192727791390000, 0.99312859918509490000];
#[rustfmt::skip]
const POINTS_24: [f64; 24] = [-0.99518721999702130000, -0.97472855597130950000, -0.93827455200273270000, -0.88641552700440100000, -0.82000198597390300000, -0.74012419157855440000, -0.64809365193697550000, -0.54542147138883950000, -0.43379350762604520000, -0.31504267969616340000, -0.19111886747361634000, -0.06405689286260563000, 0.06405689286260563000, 0.19111886747361634000, 0.31504267969616340000, 0.43379350762604520000, 0.54542147138883950000, 0.64809365193697550000, 0.74012419157855440000, 0.82000198597390300000, 0.88641552700440100000, 0.93827455200273270000, 0.97472855597130950000, 0.99518721999702130000];
#[rustfmt::skip]
const POINTS_32: [f64; 32] = [-0.99726386184948160000, -0.98561151154526840000, -0.96476225558750640000, -0.93490607593773970000, -0.89632115576605220000, -0.84936761373256990000, -0.79448379596794250000, -0.73218211874028970000, -0.66304426693021520000, -0.58771575724076230000, -0.50689990893222950000, -0.42135127613063540000, -0.33186860228212767000, -0.23928736225213710000, -0.14447196158279646000, -0.04830766568773831000, 0.04830766568773831000, 0.14447196158279646000, 0.23928736225213710000, 0.33186860228212767000, 0.42135127613063540000, 0.50689990893222950000, 0.58771575724076230000, 0.66304426693021520000, 0.73218211874028970000, 0.79448379596794250000, 0.84936761373256990000, 0.89632115576605220000, 0.93490607593773970000, 0.96476225558750640000, 0.98561151154526840000, 0.99726386184948160000];
#[rustfmt::skip]
const POINTS_40: [f64; 40] = [-0.99823770971055930000, -0.99072623869945710000, -0.97725994998377420000, -0.95791681921379170000, -0.93281280827867660000, -0.90209880696887420000, -0.86595950321225960000, -0.82461223083331170000, -0.77830565142651940000, -0.72731825518992710000, -0.67195668461417960000, -0.61255388966798030000, -0.54946712509512820000, -0.48307580168617870000, -0.41377920437160500000, -0.34199409082575850000, -0.26815218500725370000, -0.19269758070137110000, -0.11608407067525522000, -0.03877241750605081600, 0.03877241750605081600, 0.11608407067525522000, 0.19269758070137110000, 0.26815218500725370000, 0.34199409082575850000, 0.41377920437160500000, 0.48307580168617870000, 0.54946712509512820000, 0.61255388966798030000, 0.67195668461417960000, 0.72731825518992710000, 0.77830565142651940000, 0.82461223083331170000, 0.86595950321225960000, 0.90209880696887420000, 0.93281280827867660000, 0.95791681921379170000, 0.97725994998377420000, 0.99072623869945710000, 0.99823770971055930000];
#[rustfmt::skip]
const POINTS_48: [f64; 48] = [-0.99877100725242610000, -0.99353017226635080000, -0.98412458372282700000, -0.97059159254624720000, -0.95298770316043080000, -0.93138669070655440000, -0.90587913671556960000, -0.87657202027424800000, -0.84358826162439350000, -0.80706620402944250000, -0.76715903251574020000, -0.72403413092381470000, -0.67787237963266400000, -0.62886739677651370000, -0.57722472608397270000, -0.52316097472223300000, -0.46690290475095840000, -0.40868648199071680000, -0.34875588629216070000, -0.28736248735545555000, -0.22476379039468908000, -0.16122235606889174000, -0.09700469920946270000, -0.03238017096286937000, 0.03238017096286937000, 0.09700469920946270000, 0.16122235606889174000, 0.22476379039468908000, 0.28736248735545555000, 0.34875588629216070000, 0.40868648199071680000, 0.46690290475095840000, 0.52316097472223300000, 0.57722472608397270000, 0.62886739677651370000, 0.67787237963266400000, 0.72403413092381470000, 0.76715903251574020000, 0.80706620402944250000, 0.84358826162439350000, 0.87657202027424800000, 0.90587913671556960000, 0.93138669070655440000, 0.95298770316043080000, 0.97059159254624720000, 0.98412458372282700000, 0.99353017226635080000, 0.99877100725242610000];
#[rustfmt::skip]
const WEIGHTS_12: [f64; 12] = [0.04717533638651220000, 0.10693932599531830000, 0.16007832854334633000, 0.20316742672306584000, 0.23349253653835478000, 0.24914704581340286000, 0.24914704581340286000, 0.23349253653835478000, 0.20316742672306584000, 0.16007832854334633000, 0.10693932599531830000, 0.04717533638651220000];
#[rustfmt::skip]
const WEIGHTS_16: [f64; 16] = [0.02715245941175405800, 0.06225352393864777000, 0.09515851168249283000, 0.12462897125553388000, 0.14959598881657685000, 0.16915651939500256000, 0.18260341504492360000, 0.18945061045506847000, 0.18945061045506847000, 0.18260341504492360000, 0.16915651939500256000, 0.14959598881657685000, 0.12462897125553388000, 0.09515851168249283000, 0.06225352393864777000, 0.02715245941175405800];
#[rustfmt::skip]
const WEIGHTS_20: [f64; 20] = [0.01761400713915226400, 0.04060142980038684000, 0.06267204833410904000, 0.08327674157670477000, 0.10193011981724048000, 0.11819453196151844000, 0.13168863844917678000, 0.14209610931838212000, 0.14917298647260380000, 0.15275338713072600000, 0.15275338713072600000, 0.14917298647260380000, 0.14209610931838212000, 0.13168863844917678000, 0.11819453196151844000, 0.10193011981724048000, 0.08327674157670477000, 0.06267204833410904000, 0.04060142980038684000, 0.01761400713915226400];
#[rustfmt::skip]
const WEIGHTS_24: [f64; 24] = [0.01234122979998733500, 0.02853138862893380600, 0.04427743881741981000, 0.05929858491543691500, 0.07334648141108027000, 0.08619016153195320000, 0.09761865210411391000, 0.10744427011596558000, 0.11550566805372553000, 0.12167047292780335000, 0.12583745634682825000, 0.12793819534675221000, 0.12793819534675221000, 0.12583745634682825000, 0.12167047292780335000, 0.11550566805372553000, 0.10744427011596558000, 0.09761865210411391000, 0.08619016153195320000, 0.07334648141108027000, 0.05929858491543691500, 0.04427743881741981000, 0.02853138862893380600, 0.01234122979998733500];
#[rustfmt::skip]
const WEIGHTS_32: [f64; 32] = [0.00701861000947013600, 0.01627439473090571200, 0.02539206530926214200, 0.03427386291302141000, 0.04283589802222658600, 0.05099805926237621600, 0.05868409347853559000, 0.06582222277636193000, 0.07234579410884862000, 0.07819389578707042000, 0.08331192422694673000, 0.08765209300440380000, 0.09117387869576390000, 0.09384439908080441000, 0.09563872007927487000, 0.09654008851472784000, 0.09654008851472784000, 0.09563872007927487000, 0.09384439908080441000, 0.09117387869576390000, 0.08765209300440380000, 0.08331192422694673000, 0.07819389578707042000, 0.07234579410884862000, 0.06582222277636193000, 0.05868409347853559000, 0.05099805926237621600, 0.04283589802222658600, 0.03427386291302141000, 0.02539206530926214200, 0.01627439473090571200, 0.00701861000947013600];
#[rustfmt::skip]
const WEIGHTS_40: [f64; 40] = [0.00452127709853309800, 0.01049828453115270400, 0.01642105838190797300, 0.02224584919416689000, 0.02793700698002338000, 0.03346019528254786500, 0.03878216797447199000, 0.04387090818567333000, 0.04869580763507221000, 0.05322784698393679000, 0.05743976909939157000, 0.06130624249292891000, 0.06480401345660108000, 0.06791204581523394000, 0.07061164739128681000, 0.07288658239580408000, 0.07472316905796833000, 0.07611036190062619000, 0.07703981816424793000, 0.07750594797842482000, 0.07750594797842482000, 0.07703981816424793000, 0.07611036190062619000, 0.07472316905796833000, 0.07288658239580408000, 0.07061164739128681000, 0.06791204581523394000, 0.06480401345660108000, 0.06130624249292891000, 0.05743976909939157000, 0.05322784698393679000, 0.04869580763507221000, 0.04387090818567333000, 0.03878216797447199000, 0.03346019528254786500, 0.02793700698002338000, 0.02224584919416689000, 0.01642105838190797300, 0.01049828453115270400, 0.00452127709853309800];
#[rustfmt::skip]
const WEIGHTS_48: [f64; 48] = [0.00315334605230596250, 0.00732755390127620800, 0.01147723457923446900, 0.01557931572294386600, 0.01961616045735556700, 0.02357076083932435600, 0.02742650970835688000, 0.03116722783279807000, 0.03477722256477045000, 0.03824135106583080600, 0.04154508294346483000, 0.04467456085669424000, 0.04761665849249054000, 0.05035903555385448000, 0.05289018948519365000, 0.05519950369998416500, 0.05727729210040315000, 0.05911483969839566000, 0.06070443916589384000, 0.06203942315989268000, 0.06311419228625403000, 0.06392423858464817000, 0.06446616443595010000, 0.06473769681268386000, 0.06473769681268386000, 0.06446616443595010000, 0.06392423858464817000, 0.06311419228625403000, 0.06203942315989268000, 0.06070443916589384000, 0.05911483969839566000, 0.05727729210040315000, 0.05519950369998416500, 0.05289018948519365000, 0.05035903555385448000, 0.04761665849249054000, 0.04467456085669424000, 0.04154508294346483000, 0.03824135106583080600, 0.03477722256477045000, 0.03116722783279807000, 0.02742650970835688000, 0.02357076083932435600, 0.01961616045735556700, 0.01557931572294386600, 0.01147723457923446900, 0.00732755390127620800, 0.00315334605230596250];

#[cfg(test)]
mod tests {
    use libm::exp2;
    use spacelab_autodiff::{DoubleDouble, Scalar};
    use spacelab_testkit::{nan_max, ulps_between};

    use super::*;

    /// `Σaᵢbᵢ` in double-double, then rounded once to `f64`: the products are exact (`two_prod` via
    /// `fma`) and the running sum keeps ~106 bits, so the high word returned here is the correctly
    /// rounded double of the exact value.
    fn dd_dot(terms: &[(f64, f64)]) -> f64 {
        let mut acc = DoubleDouble::new(0.0, 0.0);
        for &(a, b) in terms {
            acc = acc + DoubleDouble::from_f64(a) * DoubleDouble::from_f64(b);
        }
        acc.value()
    }

    /// `(Pₙ(x), Pₙ′(x))` from Bonnet's recurrence `(j+1)P_{j+1} = (2j+1)·x·P_j − j·P_{j−1}` and the
    /// derivative identity `(1−x²)·Pₙ′ = n·(P_{n−1} − x·Pₙ)`, carried in double-double.
    fn legendre_dd(n: usize, x: DoubleDouble) -> (DoubleDouble, DoubleDouble) {
        let one = DoubleDouble::from_f64(1.0);
        let mut p_prev = one;
        let mut p = x;
        for j in 1..n {
            let next = (DoubleDouble::from_f64(2.0 * j as f64 + 1.0) * x * p
                - DoubleDouble::from_f64(j as f64) * p_prev)
                / DoubleDouble::from_f64(j as f64 + 1.0);
            p_prev = p;
            p = next;
        }
        // (1 − x)(1 + x) rather than 1 − x², which cancels catastrophically as |x| → 1 and the
        // outermost nodes sit within 1.2e-3 of the endpoints.
        let d = DoubleDouble::from_f64(n as f64) * (p_prev - x * p) / ((one - x) * (one + x));
        (p, d)
    }

    /// The `n`-point Gauss–Legendre rule on `[−1, 1]`, rebuilt from its definition: Newton on `Pₙ`
    /// from the asymptotic seed `cos(π(i+¾)/(n+½))`, then `wᵢ = 2/((1−xᵢ²)·Pₙ′(xᵢ)²)`. Double-double
    /// throughout, so the reconstruction carries ~30 digits and can judge an `f64` table rather than
    /// merely agree with it. Only the first `n` entries of each array are written.
    fn reference_rule(
        n: usize,
    ) -> (
        [DoubleDouble; MAX_GAUSS_ORDER],
        [DoubleDouble; MAX_GAUSS_ORDER],
    ) {
        let zero = DoubleDouble::new(0.0, 0.0);
        let one = DoubleDouble::from_f64(1.0);
        let mut x = [zero; MAX_GAUSS_ORDER];
        let mut w = [zero; MAX_GAUSS_ORDER];
        for i in 0..n.div_ceil(2) {
            let mut z = DoubleDouble::from_f64(cos(
                core::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)
            ));
            // The seed is good to ~1e-3 and Newton doubles the digits each pass; eight passes reach
            // the double-double floor from there with room to spare.
            for _ in 0..8 {
                let (p, d) = legendre_dd(n, z);
                z = z - p / d;
            }
            let (_, d) = legendre_dd(n, z);
            let wi = DoubleDouble::from_f64(2.0) / ((one - z) * (one + z) * d * d);
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = wi;
            w[n - 1 - i] = wi;
        }
        (x, w)
    }

    /// `P_m(t)` from the same recurrence in plain `f64` — the exactness test needs the values, not
    /// the digits.
    fn legendre(m: usize, t: f64) -> f64 {
        if m == 0 {
            return 1.0;
        }
        let mut p_prev = 1.0;
        let mut p = t;
        for j in 1..m {
            let next = ((2.0 * j as f64 + 1.0) * t * p - j as f64 * p_prev) / (j as f64 + 1.0);
            p_prev = p;
            p = next;
        }
        p
    }

    /// Every tabulated node is a root of `Pₙ` and every tabulated weight is `2/((1−xᵢ²)Pₙ′(xᵢ)²)`,
    /// for **all seven** supported orders — checked against a from-scratch double-double
    /// reconstruction of the rule, not against anything this module computes.
    ///
    /// The old suite only ever evaluated two of the seven tables. `mean_element_rates` fixes the
    /// order at 48, and the only two orders any fixture pins are the ones [`select_order`] happens to
    /// pick there — 12 for drag, 48 for SRP. Orders 16/20/24/32/40 were transcribed and never read,
    /// so all 66 sign-flip mutations of their negative nodes survived. A flipped sign moves a node by
    /// `2|xᵢ| ≥ 0.077`, fifteen orders above the bound here.
    #[test]
    fn gauss_legendre_tables_reconstruct_from_the_legendre_polynomial() {
        let mut worst_node = 0.0_f64;
        let mut worst_weight = 0.0_f64;
        for &order in &GAUSS_ORDER {
            let (points, weights) = nodes_weights(order);
            let (ref_x, ref_w) = reference_rule(order);
            for i in 0..order {
                let dx = (DoubleDouble::from_f64(points[i]) - ref_x[i]).value().abs();
                let dw = ((DoubleDouble::from_f64(weights[i]) - ref_w[i]) / ref_w[i])
                    .value()
                    .abs();
                worst_node = nan_max(worst_node, dx);
                worst_weight = nan_max(worst_weight, dw);
            }
        }
        std::println!(
            "Gauss-Legendre tables vs a double-double reconstruction: worst node {worst_node:.3e} abs, worst weight {worst_weight:.3e} rel"
        );
        // Measured 1.290e-16 (1.16 ulp at |x| → 1) and 3.929e-14: Orekit's own constants carry ~16
        // digits in the nodes and ~13-14 in the weights, and the transcription is faithful to them.
        // The bounds say the tables are that rule to that accuracy, nothing tighter.
        assert!(
            worst_node < 2.0e-16,
            "worst node deviation {worst_node:.3e} abs"
        );
        assert!(
            worst_weight < 1.0e-13,
            "worst weight deviation {worst_weight:.3e} rel"
        );
    }

    /// Each rule integrates every polynomial of degree `≤ 2n−1` exactly and no more — the defining
    /// property of Gauss quadrature, tested on the Legendre basis where `∫₋₁¹ P_m = 2·δ_{m,0}`.
    ///
    /// A second, formula-free reading of the same tables: it needs no weight formula and no root
    /// finding, only the exact integral of a polynomial. The degree-`2n` line is what makes the claim
    /// tight rather than vacuous — a rule that were exact there would be impossible, and a table that
    /// passed the `≤ 2n−1` half by accident would have to fail it.
    #[test]
    fn gauss_legendre_rules_are_exact_through_degree_two_n_minus_one() {
        let mut worst = 0.0_f64;
        let mut weakest_beyond = f64::INFINITY;
        for &order in &GAUSS_ORDER {
            let (points, weights) = nodes_weights(order);
            let quadrature = |m: usize| -> f64 {
                (0..order)
                    .map(|i| weights[i] * legendre(m, points[i]))
                    .sum()
            };
            for m in 0..2 * order {
                let want = if m == 0 { 2.0 } else { 0.0 };
                worst = nan_max(worst, (quadrature(m) - want).abs());
            }
            weakest_beyond = weakest_beyond.min(quadrature(2 * order).abs());
        }
        std::println!(
            "Gauss-Legendre exactness: worst residual {worst:.3e} up to degree 2n-1, smallest |residual| {weakest_beyond:.3e} at degree 2n"
        );
        // Measured 1.332e-15 against 1.797e-1: the two regimes are fourteen orders apart, and a
        // single flipped node sign puts the degree-1 residual at ≥ 4.192e-3.
        assert!(worst < 1.0e-14, "worst exactness residual {worst:.3e}");
        assert!(
            weakest_beyond > 0.1,
            "degree-2n residual {weakest_beyond:.3e} is too small for the degree bound to be tight"
        );
    }

    /// [`linear_combination2`] and [`linear_combination3`] return the exact `Σaᵢbᵢ` — bit for bit —
    /// on inputs where that value is a double and the naive form is not.
    ///
    /// `s = 2²⁷ + 1` is the smallest split-boundary case: `s² = 2⁵⁴ + 2²⁸ + 1` needs 55 bits, so `fl`
    /// drops exactly the trailing one. Cancelling the `2⁵⁴` head against a second exact product
    /// leaves that dropped bit as the *entire* difference between the exact answer and the naive one
    /// — an oracle from integer arithmetic, with nothing floating-point about it.
    ///
    /// The old suite measured these functions only through the propagation fixtures, whose tolerances
    /// are `1e-9` relative; every mutation of the two-product/two-sum algebra moved those numbers in
    /// the sixteenth digit and none of them failed. Here the compensation is the whole answer: the
    /// naive form is 2²⁴ ulps off on `lc2` and returns `0` where the answer is `1` on `lc3`.
    #[test]
    fn compensated_linear_combinations_are_exact_where_the_naive_form_is_not() {
        // 2^27 + 1, the smallest value whose square needs 55 bits.
        let s = 134_217_729.0_f64;
        // [a1, b1, a2, b2] and the exact a1·b1 + a2·b2.
        let cases2: [([f64; 4], f64); 5] = [
            ([s, s, -134_217_728.0, 134_217_728.0], 268_435_457.0),
            ([134_217_728.0, -134_217_728.0, s, s], 268_435_457.0),
            ([-s, s, 134_217_728.0, 134_217_728.0], -268_435_457.0),
            ([s, s, -9_007_199_254_740_992.0, 2.0], 268_435_457.0),
            (
                [
                    s,
                    s * 2.0_f64.powi(-80),
                    -134_217_728.0,
                    134_217_728.0 * 2.0_f64.powi(-80),
                ],
                268_435_457.0 * 2.0_f64.powi(-80),
            ),
        ];
        // [a1, b1, a2, b2, a3, b3] and the exact a1·b1 + a2·b2 + a3·b3.
        let cases3: [([f64; 6], f64); 4] = [
            (
                [s, s, -134_217_728.0, 134_217_728.0, -16_384.0, 16_384.0],
                1.0,
            ),
            (
                [-16_384.0, 16_384.0, s, s, -134_217_728.0, 134_217_728.0],
                1.0,
            ),
            (
                [-134_217_728.0, 134_217_728.0, -16_384.0, 16_384.0, s, s],
                1.0,
            ),
            (
                [s, s, -134_217_728.0, 134_217_728.0, 16_384.0, 16_384.0],
                536_870_913.0,
            ),
        ];

        let mut naive2 = 0_i64;
        let mut naive3 = 0_i64;
        for (v, want) in cases2 {
            let got = linear_combination2(v[0], v[1], v[2], v[3]);
            naive2 = naive2.max(ulps_between(v[0] * v[1] + v[2] * v[3], want));
            assert_eq!(
                ulps_between(got, want),
                0,
                "linear_combination2{v:?} got {got:.17e} want {want:.17e}"
            );
        }
        for (v, want) in cases3 {
            let got = linear_combination3(v[0], v[1], v[2], v[3], v[4], v[5]);
            naive3 = naive3.max(ulps_between(v[0] * v[1] + v[2] * v[3] + v[4] * v[5], want));
            assert_eq!(
                ulps_between(got, want),
                0,
                "linear_combination3{v:?} got {got:.17e} want {want:.17e}"
            );
        }
        std::println!(
            "compensated linear combinations: 0 ulps from exact; the naive form is {naive2} ulps off on lc2 and {naive3} on lc3"
        );
        assert!(
            naive2 > 1_000_000 && naive3 > 1_000_000,
            "the cases stopped being adversarial: naive is only {naive2}/{naive3} ulps off"
        );
    }

    /// Both compensated combinations land on the **correctly rounded** double — the exact `Σaᵢbᵢ`
    /// rounded once — over a deterministic sweep of operands spanning `2⁻¹⁵ … 2¹⁴`.
    ///
    /// The reference is [`DoubleDouble`]: `fma`-exact products summed by the Hida–Li–Bailey accurate
    /// add, ~106 bits, so its high word is the correctly rounded `f64`. That is a different algorithm
    /// in a different crate — not this module's Dekker split reused as its own judge.
    ///
    /// This is the assertion the propagation fixtures could not make. Their tolerance is `1e-9`
    /// relative and every perturbation of the two-product/two-sum algebra costs about one ulp, so all
    /// fifteen of them passed while moving the printed state in its sixteenth digit. At one ulp the
    /// sweep resolves them: measured over this distribution, each mutated form disagrees with the
    /// correctly rounded value on 15-96% of draws, so 256 of them leave no room.
    #[test]
    fn compensated_linear_combinations_are_correctly_rounded() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        // Zero-mean uniform mantissa in [−1, 1) times a power of two in 2⁻¹⁵ … 2¹⁴, so the two
        // products routinely differ in exponent — which is what makes the two-sum round and the
        // compensation terms carry something.
        let mut operand = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mantissa = (f64::from((seed >> 32) as u32) / 4294967296.0) * 2.0 - 1.0;
            mantissa * exp2(((seed >> 8) & 0x1f) as f64 - 15.0)
        };
        let mut worst = 0_i64;
        for _ in 0..256 {
            let v = [
                operand(),
                operand(),
                operand(),
                operand(),
                operand(),
                operand(),
            ];
            let got2 = linear_combination2(v[0], v[1], v[2], v[3]);
            worst = worst.max(ulps_between(got2, dd_dot(&[(v[0], v[1]), (v[2], v[3])])));
            let got3 = linear_combination3(v[0], v[1], v[2], v[3], v[4], v[5]);
            worst = worst.max(ulps_between(
                got3,
                dd_dot(&[(v[0], v[1]), (v[2], v[3]), (v[4], v[5])]),
            ));
        }
        std::println!(
            "compensated linear combinations vs a double-double reference: worst {worst} ulps over 256 draws"
        );
        assert_eq!(
            worst, 0,
            "worst departure from correct rounding: {worst} ulps"
        );
    }

    /// When the compensated sum comes out exactly zero, the returned value is the **naive** form, not
    /// the compensated one — Hipparchus's `isNaN(result) || result == 0` guard, confirmed against the
    /// `hipparchus-core-4.0.3` bytecode. The two differ in the sign of the zero, and on a
    /// mutated fallback expression they differ by the whole result.
    ///
    /// The lc2 half of this branch is reached by the propagation fixtures (an equatorial `f`/`g`
    /// component multiplies out to zero); the lc3 half is not, which is why every mutation of its
    /// fallback survived. Bits, not values: `-0.0 == 0.0`, so only `to_bits` can see the difference
    /// that the `||` guard is there to preserve.
    #[test]
    fn the_naive_fallback_owns_the_result_when_the_compensated_sum_vanishes() {
        assert_eq!(
            linear_combination2(-0.0, 1.0, -0.0, 1.0).to_bits(),
            (-0.0_f64).to_bits(),
            "lc2 must return the naive -0.0, not the compensated +0.0"
        );
        assert_eq!(
            linear_combination3(-0.0, 1.0, -0.0, 1.0, -0.0, 1.0).to_bits(),
            (-0.0_f64).to_bits(),
            "lc3 must return the naive -0.0, not the compensated +0.0"
        );
        // Exact products summing to zero: the fallback runs, so its arithmetic is observable.
        assert_eq!(
            linear_combination3(2.0, 1.0, -1.0, 1.0, -1.0, 1.0).to_bits(),
            0.0_f64.to_bits(),
            "lc3 fallback must evaluate a1·b1 + a2·b2 + a3·b3"
        );
    }

    /// A GTO-like state and a drag-shaped acceleration, chosen so the quadrature error actually
    /// spreads across the order table: at `e = 0.7` the `ρ²` weight peaks sharply at perigee and the
    /// low orders miss it by a measurable amount.
    fn selection_case() -> (DsstAuxiliary, impl Fn(&NodeState) -> [f64; 3]) {
        let aux = DsstAuxiliary::new(26_600_000.0, 0.7, 0.0, 0.02, -0.01, 0.3, 3.986_004_415e14);
        let accel = |node: &NodeState| {
            let v = node.vel;
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let c = -1.0e-12;
            [c * speed * v[0], c * speed * v[1], c * speed * v[2]]
        };
        (aux, accel)
    }

    /// [`select_order`] is the one function in this module no fixture calls, and eleven arithmetic
    /// mutations of it survived: the `coef` both rate sets are scaled by, the `1/a` weight on the
    /// semi-major-axis component, the sign of both differences, and both comparisons.
    ///
    /// Its contract is a *minimum*, so that is what is asserted — the returned order meets the
    /// criterion and every lower order fails it. The diffs are recomputed here from [`integrate`],
    /// which `gauss_legendre_tables_reconstruct_from_the_legendre_polynomial` and
    /// `gauss_legendre_rules_are_exact_through_degree_two_n_minus_one` pin against the Legendre
    /// polynomial itself, so the expectation is the selection rule rather than the selection loop.
    ///
    /// Each measured diff is used as a threshold twice: once exactly (the criterion is strict, so
    /// that order must *not* be selected) and once one ulp above (it must be). A wrong `coef`
    /// rescales every diff and a wrong component weight rescales one of them, so either way some
    /// boundary lands on the other side of its threshold.
    #[test]
    fn the_selected_order_is_the_lowest_meeting_the_threshold() {
        let (aux, accel) = selection_case();
        let (low, high) = (0.0, core::f64::consts::TAU);
        let coef = 1.0 / (2.0 * core::f64::consts::PI * aux.big_b);

        let scaled = |order: usize| {
            let mut r = integrate(order, &aux, low, high, &accel);
            for x in &mut r {
                *x *= coef;
            }
            r
        };
        let reference = scaled(MAX_GAUSS_ORDER);
        let searchable = &GAUSS_ORDER[..GAUSS_ORDER.len() - 1];

        // getRatesDiff: |Δda|/a for the semi-major axis, |Δ| for the other five.
        let diffs: Vec<f64> = searchable
            .iter()
            .map(|&order| {
                let cur = scaled(order);
                let mut d = (reference[0] - cur[0]).abs() / aux.sma;
                for i in 1..6 {
                    d = nan_max(d, (reference[i] - cur[i]).abs());
                }
                d
            })
            .collect();
        println!("order-vs-48 weighted diffs {searchable:?} -> {diffs:?}");
        assert!(
            diffs[0] > diffs[diffs.len() - 1],
            "the case must resolve orders: diffs {diffs:?} carry no information"
        );

        let expected = |threshold: f64| {
            searchable
                .iter()
                .zip(&diffs)
                .find(|&(_, &d)| d < threshold)
                .map_or(MAX_GAUSS_ORDER, |(&o, _)| o)
        };
        for &d in &diffs {
            for threshold in [d, f64::from_bits(d.to_bits() + 1)] {
                assert_eq!(
                    select_order(&aux, low, high, &accel, threshold),
                    expected(threshold),
                    "threshold {threshold:e} over diffs {diffs:?}"
                );
            }
        }

        // An empty arc integrates to zero at every order, so every diff is zero and the lowest order
        // would trivially qualify. Orekit skips the integral instead and keeps the maximum.
        assert_eq!(
            select_order(&aux, high, high, &accel, 1e-30),
            MAX_GAUSS_ORDER,
            "an empty longitude arc must not select an order"
        );
        // No diff is below zero, so nothing qualifies and the fallback owns the answer.
        assert_eq!(
            select_order(&aux, low, high, &accel, 0.0),
            MAX_GAUSS_ORDER,
            "a zero threshold admits no order"
        );
        // Excluding the maximum order from the search is a no-op only because the loop's last
        // candidate and the fallback are the same number; that is what the `.rs:87:56` exclusion
        // in `.cargo/mutants.toml` rests on.
        assert_eq!(
            GAUSS_ORDER[GAUSS_ORDER.len() - 1],
            MAX_GAUSS_ORDER,
            "the order table must end at the maximum order"
        );
    }
}
