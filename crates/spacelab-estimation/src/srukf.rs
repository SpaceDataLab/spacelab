//! The square-root unscented Kalman filter (Van der Merwe & Wan 2001): a sigma-point filter that
//! carries the Cholesky factor `S` of the covariance, never `P`, and never re-forms it.
//!
//! # Why the square root, again
//!
//! The plain UKF forms the posterior covariance as `P⁺ = P⁻ − K P_y Kᵀ`, a *subtraction* that can and
//! does drift indefinite in finite precision — the same failure the sequential
//! [`crate::kalman::SquareRootKalman`] avoids on the linear side. The square-root UKF applies the same
//! cure to the nonlinear filter: keep `S` with `P = S Sᵀ`, and every update is an orthogonal
//! transformation or a rank-1 Cholesky modification, so `P` is positive-semidefinite *by construction*.
//! It also halves the conditioning: `S` carries the dynamic range of `P` under a square root.
//!
//! # The two square-root moves
//!
//! * **QR of the residual matrix.** The `2N` off-center sigma residuals `√Wᵢᶜ (Xᵢ − x̄)` and the
//!   process- (or measurement-) noise square root stack into a tall matrix whose triangular `QR`
//!   factor is exactly `chol(Σ Wᵢᶜ (Xᵢ − x̄)(·)ᵀ + Q)`. It is folded row by row with the same Givens
//!   rotations as [`crate::Srif`] — orthogonal, allocation-free, condition-number-preserving.
//! * **Rank-1 cholupdate of the center.** The center weight `W₀ᶜ` can be *negative* (it is, for the
//!   usual small `α`), so it cannot be a QR row — a QR only ever adds outer products. Instead the
//!   center residual is folded in with a rank-1 Cholesky update whose sign follows `W₀ᶜ`. The
//!   measurement update likewise *downdates* `S` by the columns of `K S_y`, giving `P⁻ − K P_y Kᵀ`
//!   without ever forming that difference as a matrix.
//!
//! Gate: `derivations/srukf_square_root.mac` restates `cholupdate` and `fold_row` step for step
//! on a general 3 × 3 factor and shows each adds (or subtracts) exactly one outer product and keeps
//! the factor triangular. The module tests `the_square_root_moves_add_and_subtract_exactly_one_outer_product`
//! (G1) and `the_srukf_is_the_exact_rational_kalman_step` (G3) — the linear-Gaussian step against
//! `derivations/kalman_square_root.mac`'s exact recursion — tie the shipped code to it.

// Matrix code indexes several arrays by one i/j/k, and the filter symbols (S, P, K, Q, R, χ) are one
// letter by convention; the paper's locals (xm, yb, ry, sy, w0c) read as its notation.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use crate::consistency::normalized_squared;
use crate::robust::{self, RobustOutcome, RobustPolicy};
use crate::unscented::{UnscentedParams, centered_residuals, cholesky, sigma_points};

/// A square-root unscented Kalman filter over an `N`-dimensional state with `L = 2N + 1` sigma points.
///
/// The estimate [`x`](Self::x) is public and mirrors [`crate::kalman::SquareRootKalman`]: the filter
/// owns the covariance factor `S`, while [`predict`](Self::predict) advances both the state and `S`
/// through the caller's nonlinear dynamics, and [`update`](Self::update) folds in a nonlinear
/// measurement. `L` is a const parameter because stable Rust cannot yet size the sigma array by
/// `2N + 1`; the pairing is checked at construction, as `spacelab_propagation::stt` pairs `N` with
/// its augmented dimension.
#[derive(Debug, Clone)]
pub struct SquareRootUnscentedKalman<const N: usize, const L: usize> {
    /// The state estimate. Advanced through the dynamics by [`predict`](Self::predict) and corrected
    /// by [`update`](Self::update).
    pub x: [f64; N],
    /// Covariance square root `S`, with `P = S Sᵀ` (lower-triangular). Never `P` itself.
    s: [[f64; N]; N],
    /// The scaled-transform parameters `α, β, κ`.
    params: UnscentedParams,
}

impl<const N: usize, const L: usize> SquareRootUnscentedKalman<N, L> {
    /// Seed with an estimate, the square root of its covariance (`P₀ = S₀ S₀ᵀ`, e.g. a Cholesky
    /// factor), and the transform parameters.
    ///
    /// # Panics
    ///
    /// If `L != 2N + 1` — the sigma-point storage would be mis-sized.
    #[must_use]
    pub const fn new(
        x0: [f64; N],
        covariance_sqrt: [[f64; N]; N],
        params: UnscentedParams,
    ) -> Self {
        assert!(L == 2 * N + 1, "sigma-point count L must equal 2N + 1");
        Self {
            x: x0,
            s: covariance_sqrt,
            params,
        }
    }

    /// Seed from an estimate and a full covariance matrix, taking its lower Cholesky factor.
    ///
    /// Returns `None` if `p0` is not positive-definite, exactly as
    /// [`crate::kalman::SquareRootKalman::from_covariance`] does — the filter refuses to start from a
    /// matrix that is not a covariance.
    ///
    /// # Panics
    ///
    /// If `L != 2N + 1`.
    #[must_use]
    pub fn from_covariance(
        x0: [f64; N],
        p0: &[[f64; N]; N],
        params: UnscentedParams,
    ) -> Option<Self> {
        assert!(L == 2 * N + 1, "sigma-point count L must equal 2N + 1");
        let s = cholesky::<N>(p0)?;
        Some(Self { x: x0, s, params })
    }

    /// Time update: propagate the sigma cloud through the nonlinear `dynamics` (the one-step flow) and
    /// recover the predicted mean and covariance factor, adding process noise `Q = S_Q S_Qᵀ`
    /// (`process_noise_sqrt` is any `S_Q`; the zero matrix for a noiseless step).
    ///
    /// Unlike the linear [`crate::kalman::SquareRootKalman::time_update`], which needs the
    /// state-transition matrix, this takes the dynamics themselves — the sigma points sample the
    /// curvature the linear map cannot see.
    pub fn predict(
        &mut self,
        dynamics: impl Fn(&[f64; N]) -> [f64; N],
        process_noise_sqrt: &[[f64; N]; N],
    ) {
        let (x, s) =
            unscented_predict::<N, L>(&self.params, &self.x, &self.s, dynamics, process_noise_sqrt);
        self.x = x;
        self.s = s;
    }

    /// Measurement update for a `D`-dimensional observation `z`, related to the state by the nonlinear
    /// `measurement` map, with noise covariance `R = S_R S_Rᵀ` (`noise_sqrt` is any `S_R`).
    ///
    /// Propagates the sigma cloud through `measurement`, builds the innovation-covariance factor `S_y`
    /// by QR + center cholupdate, forms the cross-covariance `P_xy`, and applies the gain
    /// `K = P_xy (S_y S_yᵀ)⁻¹` by triangular back-substitution. The state moves by `K (z − ẑ)` and `S`
    /// is *downdated* by the columns of `K S_y`, realising `P⁺ = P⁻ − K P_y Kᵀ` in factored form.
    pub fn update<const D: usize>(
        &mut self,
        measurement: impl Fn(&[f64; N]) -> [f64; D],
        z: &[f64; D],
        noise_sqrt: &[[f64; D]; D],
    ) {
        self.update_robust(measurement, z, noise_sqrt, RobustPolicy::Plain);
    }

    /// [`update`](Self::update) under a [`RobustPolicy`], returning what the policy did with the
    /// observation. Feed it to a [`RobustTally`](crate::RobustTally), one per measurement stream.
    ///
    /// The sigma cloud is propagated once. `δ² = νᵀ(S_y S_yᵀ)⁻¹ν` comes from a forward substitution on
    /// the nominal factor, and re-folding the noise columns at the policy's scale rebuilds `S_y`
    /// without touching `measurement` again — so robustness costs one `D×D` QR, not a second cloud.
    ///
    /// `noise_sqrt` is scaled by `√scale`, which is the isotropic-`R` assumption
    /// [`RobustPolicy::variance_scale`] documents: for a diagonal `R` with unequal entries the discount
    /// still lands on every axis in proportion, so a spike on one channel de-weights the others.
    pub fn update_robust<const D: usize>(
        &mut self,
        measurement: impl Fn(&[f64; N]) -> [f64; D],
        z: &[f64; D],
        noise_sqrt: &[[f64; D]; D],
        policy: RobustPolicy,
    ) -> RobustOutcome {
        self.fold(measurement, z, noise_sqrt, policy).0
    }

    /// [`update`](Self::update), also returning the observation's log-likelihood under this filter's
    /// own predicted innovation distribution, `ln 𝒩(ν; 0, S_y S_yᵀ)`.
    ///
    /// That number is what weighs one dynamics hypothesis against another — it is the whole of a model
    /// bank's evidence, and [`crate::imm`] consumes nothing else from the filters it runs. It comes out
    /// of the update that already formed `ν` and `S_y`, so asking for it costs `D` logarithms and no
    /// second pass over the sigma cloud. The log form is not a convenience: for a poorly-matched model
    /// the density itself underflows to zero long before its logarithm stops being informative.
    pub fn update_with_log_likelihood<const D: usize>(
        &mut self,
        measurement: impl Fn(&[f64; N]) -> [f64; D],
        z: &[f64; D],
        noise_sqrt: &[[f64; D]; D],
    ) -> f64 {
        self.fold(measurement, z, noise_sqrt, RobustPolicy::Plain).1
    }

    /// The one measurement update both public forms are made of: the outcome the policy produced, and
    /// the observation's log-likelihood under the *undiscounted* innovation covariance — the evidence
    /// for this model, which a robust discount must not flatter.
    fn fold<const D: usize>(
        &mut self,
        measurement: impl Fn(&[f64; N]) -> [f64; D],
        z: &[f64; D],
        noise_sqrt: &[[f64; D]; D],
        policy: RobustPolicy,
    ) -> (RobustOutcome, f64) {
        let w = self.params.weights::<N>();
        let pts = sigma_points::<N, L>(&self.x, &self.s, w.gamma);

        let mut y = [[0.0; D]; L];
        for i in 0..L {
            y[i] = measurement(&pts[i]);
        }

        let (offset, resid) = centered_residuals::<D, L>(&y, &w);
        let mut innov = [0.0; D];
        for j in 0..D {
            // `z − Y₀` is a subtraction of neighbours and exact by Sterbenz; `z − ŷ` written out
            // would not be, since `ŷ` is a sum of huge weighted terms.
            innov[j] = (z[j] - y[0][j]) - offset[j];
        }
        let nominal = innovation_factor::<D, L>(&resid, &w, noise_sqrt, 1.0);
        let normalized = normalized_squared(&nominal, &innov);
        let log_likelihood = gaussian_log_density::<D>(&nominal, normalized);
        let Some(scale) = policy.variance_scale(normalized, D) else {
            return (robust::rejected(normalized), log_likelihood);
        };
        // `1.0` is an exact sentinel every policy returns for an undiscounted observation, not a rounded
        // value, so this skips a QR that would reproduce `nominal` bit for bit rather than approximating
        // one. `Plain` takes this branch always, which is why robustness costs nothing when unused.
        #[allow(clippy::float_cmp)]
        let undiscounted = scale == 1.0;
        let sy = if undiscounted {
            nominal
        } else {
            innovation_factor::<D, L>(&resid, &w, noise_sqrt, scale)
        };

        // Cross-covariance P_xy = Σ Wᵢᶜ (χᵢ − x)(Yᵢ − ŷ)ᵀ (the center residual is zero, so i ≥ 1).
        let mut pxy = [[0.0; D]; N];
        for i in 1..L {
            for a in 0..N {
                let da = pts[i][a] - self.x[a];
                for b in 0..D {
                    pxy[a][b] += w.wi * da * resid[i][b];
                }
            }
        }

        // Gain K = P_xy (S_y S_yᵀ)⁻¹ by two triangular solves; state move by the innovation.
        let k = solve_gain::<N, D>(&pxy, &sy);
        for a in 0..N {
            let mut acc = 0.0;
            for b in 0..D {
                acc += k[a][b] * innov[b];
            }
            self.x[a] += acc;
        }

        // Downdate S by each column of U = K S_y: S Sᵀ ← P⁻ − K P_y Kᵀ.
        let mut u = [[0.0; D]; N];
        for a in 0..N {
            for j in 0..D {
                let mut acc = 0.0;
                for c in 0..D {
                    acc += k[a][c] * sy[c][j];
                }
                u[a][j] = acc;
            }
        }
        for j in 0..D {
            let mut col = [0.0; N];
            for a in 0..N {
                col[a] = u[a][j];
            }
            cholupdate(&mut self.s, &col, -1.0);
        }

        let mut trace = 0.0;
        for i in 0..D {
            for j in 0..=i {
                trace += sy[i][j] * sy[i][j];
            }
        }
        #[allow(clippy::cast_precision_loss)] // D is a measurement dimension, never above a handful
        let outcome = RobustOutcome {
            accepted: true,
            weight: 1.0 / scale,
            nominal_normalized_squared: normalized,
            innovation_variance: trace / D as f64,
        };
        (outcome, log_likelihood)
    }

    /// The current covariance `P = S Sᵀ`, formed only on request.
    #[must_use]
    pub fn covariance(&self) -> [[f64; N]; N] {
        let mut p = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += self.s[i][k] * self.s[j][k];
                }
                p[i][j] = acc;
            }
        }
        p
    }

    /// The covariance square root `S` (lower-triangular, `P = S Sᵀ`).
    #[must_use]
    pub fn covariance_sqrt(&self) -> [[f64; N]; N] {
        self.s
    }
}

/// The square-root unscented time update as a free function: push `N(mean, sqrt·sqrtᵀ)` through
/// `dynamics`, add process noise `S_Q S_Qᵀ`, and return the predicted `(mean, S)` in factored form.
///
/// Shared by [`SquareRootUnscentedKalman::predict`] and [`crate::mixture::GaussianMixture::propagate`]
/// so a mixture component predicts through exactly the same square-root recursion as a lone filter.
///
/// # Panics
///
/// If `L != 2N + 1`.
#[must_use]
pub(crate) fn unscented_predict<const N: usize, const L: usize>(
    params: &UnscentedParams,
    mean: &[f64; N],
    sqrt: &[[f64; N]; N],
    dynamics: impl Fn(&[f64; N]) -> [f64; N],
    process_noise_sqrt: &[[f64; N]; N],
) -> ([f64; N], [[f64; N]; N]) {
    let w = params.weights::<N>();
    let pts = sigma_points::<N, L>(mean, sqrt, w.gamma);

    let mut prop = [[0.0; N]; L];
    for i in 0..L {
        prop[i] = dynamics(&pts[i]);
    }

    let (offset, resid) = centered_residuals::<N, L>(&prop, &w);
    let mut xm = [0.0; N];
    for j in 0..N {
        xm[j] = prop[0][j] + offset[j];
    }

    let sqrt_wi = libm::sqrt(w.wi);
    let mut r = [[0.0; N]; N];
    for i in 1..L {
        let mut row = [0.0; N];
        for j in 0..N {
            row[j] = sqrt_wi * resid[i][j];
        }
        fold_row(&mut r, &row);
    }
    for k in 0..N {
        let mut col = [0.0; N];
        for i in 0..N {
            col[i] = process_noise_sqrt[i][k];
        }
        fold_row(&mut r, &col);
    }
    let mut s = transpose_to_lower(&r);

    let mut center = [0.0; N];
    let mag = libm::sqrt(libm::fabs(w.w0c));
    for j in 0..N {
        center[j] = mag * resid[0][j];
    }
    cholupdate(&mut s, &center, if w.w0c < 0.0 { -1.0 } else { 1.0 });
    (xm, s)
}

/// `ln 𝒩(ν; 0, S Sᵀ)` from the innovation's lower factor and `δ² = νᵀ(S Sᵀ)⁻¹ν`.
///
/// `ln|S Sᵀ| = 2 Σ ln|Sᵢᵢ|`, and the absolute value is load-bearing rather than defensive: the factor
/// comes from a Givens sweep followed by a rank-1 update, neither of which is asked to keep the
/// diagonal positive, and only the magnitude enters the determinant of `S Sᵀ`.
fn gaussian_log_density<const D: usize>(factor: &[[f64; D]; D], normalized_squared: f64) -> f64 {
    let mut log_det = 0.0;
    for i in 0..D {
        log_det += libm::log(libm::fabs(factor[i][i]));
    }
    #[allow(clippy::cast_precision_loss)] // D is a measurement dimension, never above a handful
    let dimension = D as f64;
    -0.5 * (normalized_squared + dimension * libm::log(core::f64::consts::TAU)) - log_det
}

/// `S_y = chol(Σ Wᵢᶜ (Yᵢ − ŷ)(·)ᵀ + scale·R)`: a QR of the off-center measurement residuals and the
/// noise columns, then the rank-1 (possibly negative) center update. `scale` multiplies `R`, so the
/// columns take its square root.
///
/// Free rather than a closure because the robust path builds it twice — once at the nominal `R` to
/// measure `δ²`, once at the policy's scale — and the two must be the same code, not two transcriptions
/// of it.
fn innovation_factor<const D: usize, const L: usize>(
    resid: &[[f64; D]; L],
    w: &crate::unscented::Weights,
    noise_sqrt: &[[f64; D]; D],
    scale: f64,
) -> [[f64; D]; D] {
    let sqrt_wi = libm::sqrt(w.wi);
    let root = libm::sqrt(scale);
    let mut ry = [[0.0; D]; D];
    for i in 1..L {
        let mut row = [0.0; D];
        for j in 0..D {
            row[j] = sqrt_wi * resid[i][j];
        }
        fold_row(&mut ry, &row);
    }
    for k in 0..D {
        let mut col = [0.0; D];
        for i in 0..D {
            col[i] = root * noise_sqrt[i][k];
        }
        fold_row(&mut ry, &col);
    }
    let mut sy = transpose_to_lower(&ry);
    let mut ycen = [0.0; D];
    let mag = libm::sqrt(libm::fabs(w.w0c));
    for j in 0..D {
        ycen[j] = mag * resid[0][j];
    }
    cholupdate(&mut sy, &ycen, if w.w0c < 0.0 { -1.0 } else { 1.0 });
    sy
}

/// Fold one row vector into the upper-triangular factor `r` by Givens rotations, so that after the
/// fold `rᵀr = Σ (rows so far)(·)ᵀ`. The measurement-free twin of [`crate::Srif::add_row`].
fn fold_row<const M: usize>(r: &mut [[f64; M]; M], row: &[f64; M]) {
    let mut a = *row;
    for k in 0..M {
        if a[k] == 0.0 {
            continue;
        }
        let rkk = r[k][k];
        let denom = libm::hypot(rkk, a[k]);
        if denom == 0.0 {
            continue;
        }
        let c = rkk / denom;
        let s = a[k] / denom;
        r[k][k] = denom;
        for j in (k + 1)..M {
            let temp = c * r[k][j] + s * a[j];
            a[j] = c * a[j] - s * r[k][j];
            r[k][j] = temp;
        }
    }
}

/// The transpose of an upper-triangular `r` — a lower-triangular factor `l` with `l lᵀ = rᵀr`.
fn transpose_to_lower<const M: usize>(r: &[[f64; M]; M]) -> [[f64; M]; M] {
    let mut l = [[0.0; M]; M];
    for i in 0..M {
        for j in 0..M {
            l[i][j] = r[j][i];
        }
    }
    l
}

/// Rank-1 Cholesky update/downdate of a lower-triangular factor `l` in place:
/// `l lᵀ ← l lᵀ + sign · v vᵀ`, with `sign = ±1`. A downdate whose radicand would go negative is
/// clamped at the positive-definite boundary rather than taking the square root of a negative — that
/// only bites when the update would push `P` indefinite, and the square-root form exists precisely to
/// keep that boundary reachable but never crossed.
fn cholupdate<const M: usize>(l: &mut [[f64; M]; M], v: &[f64; M], sign: f64) {
    let mut x = *v;
    for k in 0..M {
        let dk = l[k][k];
        if dk == 0.0 {
            continue;
        }
        let rr = dk * dk + sign * x[k] * x[k];
        let r = if rr > 0.0 { libm::sqrt(rr) } else { 0.0 };
        l[k][k] = r;
        if r == 0.0 {
            continue;
        }
        let c = r / dk;
        let s = x[k] / dk;
        for i in (k + 1)..M {
            l[i][k] = (l[i][k] + sign * s * x[i]) / c;
            x[i] = c * x[i] - s * l[i][k];
        }
    }
}

/// Solve the Kalman gain `K = P_xy (S_y S_yᵀ)⁻¹` for lower-triangular `S_y`, by a forward solve
/// `M S_yᵀ = P_xy` followed by a backward solve `K S_y = M` — never forming or inverting `P_y`.
#[must_use]
fn solve_gain<const N: usize, const D: usize>(
    pxy: &[[f64; D]; N],
    sy: &[[f64; D]; D],
) -> [[f64; D]; N] {
    let mut m = [[0.0; D]; N];
    for n in 0..N {
        for j in 0..D {
            let mut acc = pxy[n][j];
            for k in 0..j {
                acc -= m[n][k] * sy[j][k];
            }
            m[n][j] = if sy[j][j] == 0.0 { 0.0 } else { acc / sy[j][j] };
        }
    }
    let mut k = [[0.0; D]; N];
    for n in 0..N {
        for j in (0..D).rev() {
            let mut acc = m[n][j];
            for c in (j + 1)..D {
                acc -= k[n][c] * sy[c][j];
            }
            k[n][j] = if sy[j][j] == 0.0 { 0.0 } else { acc / sy[j][j] };
        }
    }
    k
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]
    use super::*;
    use crate::kalman::SquareRootKalman;
    use crate::unscented::unscented_transform;
    use spacelab_testkit::nan_max;

    fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
        (0..N).map(|i| a[i] * b[i]).sum()
    }

    fn matvec<const N: usize>(a: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
        let mut y = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                y[i] += a[i][j] * x[j];
            }
        }
        y
    }

    /// Oracle (a) — the primary correctness gate. On a linear-Gaussian problem the unscented transform
    /// is exact, so the square-root UKF must reproduce the existing square-root Kalman filter's
    /// posterior mean *and* covariance step for step, to rounding. Any drift means the sigma-point
    /// recursion is wrong, not merely different.
    #[test]
    fn the_srukf_reproduces_the_linear_kalman_filter() {
        let (worst_x, worst_p) = linear_kalman_departure(UnscentedParams::new(1.0, 2.0, 0.0));
        println!(
            "SR-UKF vs SR-KF (linear-Gaussian): max |Δx| {worst_x:.3e}, max |ΔP| {worst_p:.3e}"
        );
        assert!(worst_x < 1e-11, "state drifted from the KF: {worst_x:.3e}");
        assert!(
            worst_p < 1e-11,
            "covariance drifted from the KF: {worst_p:.3e}"
        );
    }

    /// Oracle (a), downdate arm — the same comparison at the Van der Merwe default `α = 1e-3`.
    ///
    /// Every other test in this module runs at `α ∈ {0.7, 0.9, 1.0}`, where the centre covariance
    /// weight comes out positive and `cholupdate` is only ever called with `+1`. At the crate's own
    /// [`UnscentedParams::gaussian`] the weight is about `−10⁶`, which is the case the square-root
    /// form exists for: the rank-one correction has to be *subtracted* from the factor, and a plain
    /// QR cannot express it. Reaching it costs precision — the weights are `±10⁶` and their sum is
    /// 1 — and the measured cost is uneven: 4.9e-10 in the state against 8.9e-16 on the well-scaled
    /// arm, but only 8.6e-14 in the covariance against 2.6e-15. The factored recursion absorbs the
    /// scaling in the covariance; what it cannot absorb is the innovation weighting in the mean.
    #[test]
    fn the_srukf_reproduces_the_linear_kalman_filter_through_the_downdate() {
        let params = UnscentedParams::gaussian();
        let w0c = params.weights::<3>().w0c;
        assert!(
            w0c < 0.0,
            "the fixture must reach the downdate, W₀ᶜ = {w0c}"
        );

        let (worst_x, worst_p) = linear_kalman_departure(params);
        println!("SR-UKF vs SR-KF (α = 1e-3): max |Δx| {worst_x:.3e}, max |ΔP| {worst_p:.3e}");
        assert!(worst_x < 5e-9, "state drifted from the KF: {worst_x:.3e}");
        assert!(
            worst_p < 1e-12,
            "covariance drifted from the KF: {worst_p:.3e}"
        );
    }

    /// The posterior is still the Kalman answer when the state dwarfs its own uncertainty.
    ///
    /// Every other fixture here sits at `|x̂| ≈ 1`, where nothing about the accumulation shows. An
    /// orbit-determination filter does not: its state is a position in metres and its converged
    /// covariance is metre-level or better, so the sigma cloud spans `γS ≈ 1e-6` on a number of order
    /// 1e6 and the mean weights are `±1e6`. Measured with the direct `Σ Wᵢᵐ Yᵢ`: **11%** relative
    /// error in the posterior correction and **2.0%** in the covariance, against **2.3e-7** and
    /// **7.7e-7** once the cloud is accumulated from its own centre.
    /// `examples/srukf_origin_sweep.rs` walks the decades either side of this one, and shows where
    /// even the centred form gives out — by `|x̂| = 1e7` the sigma offset itself is quantized and the
    /// posterior is 1e-3 relative, which is a floor on the state's magnitude, not on the arithmetic.
    #[test]
    fn the_posterior_survives_a_state_that_dwarfs_its_uncertainty() {
        let (p0, sigma) = ([[1e-6, 0.0], [0.0, 4e-6]], 1e-3);
        let x0 = [1e6, 1e6];
        let mut filter = SquareRootUnscentedKalman::<2, 5>::from_covariance(
            x0,
            &p0,
            UnscentedParams::gaussian(),
        )
        .unwrap();

        // Both differences are between neighbours and therefore exact, so the reference below is the
        // innovation the filter itself sees rather than the one this line asked for.
        let predicted = x0[0] + x0[1];
        let z = predicted + 1e-3;
        let innovation = z - predicted;
        filter.update(|x| [x[0] + x[1]], &[z], &[[sigma]]);

        // K = P Hᵀ (H P Hᵀ + R)⁻¹ with H = [1, 1]; P⁺ = P⁻ − K (H P Hᵀ + R) Kᵀ.
        let hph_r = p0[0][0] + p0[1][1] + sigma * sigma;
        let gain = [p0[0][0] / hph_r, p0[1][1] / hph_r];
        let posterior = filter.covariance();
        for a in 0..2 {
            let want = gain[a] * innovation;
            let error = ((filter.x[a] - x0[a]) - want).abs() / want.abs();
            assert!(error < 1e-5, "correction[{a}]: relative error {error:e}");
            for b in 0..2 {
                let want = if a == b { p0[a][a] } else { 0.0 } - gain[a] * hph_r * gain[b];
                let error = (posterior[a][b] - want).abs() / want.abs();
                assert!(error < 1e-5, "P⁺[{a}][{b}]: relative error {error:e}");
            }
        }
    }

    /// The square-root accessor is a square root of the covariance, and lower-triangular.
    ///
    /// It had no caller anywhere in the workspace, so nothing said what it returns.
    #[test]
    fn the_reported_factor_multiplies_out_to_the_reported_covariance() {
        let s0 = [[2.0, 0.0, 0.0], [0.4, 1.7, 0.0], [-0.2, 0.3, 1.3]];
        let mut ukf = SquareRootUnscentedKalman::<3, 7>::new(
            [0.5, -0.3, 0.1],
            s0,
            UnscentedParams::gaussian(),
        );
        ukf.update::<1>(|x| [x[0] + 0.5 * x[2]], &[1.25], &[[0.5]]);

        let (s, p) = (ukf.covariance_sqrt(), ukf.covariance());
        for i in 0..3 {
            for j in 0..3 {
                if j > i {
                    assert_eq!(s[i][j], 0.0, "S[{i}][{j}] is above the diagonal");
                }
                let sst: f64 = (0..3).map(|k| s[i][k] * s[j][k]).sum();
                assert!(
                    (sst - p[i][j]).abs() < 1e-12,
                    "S Sᵀ[{i}][{j}] = {sst}, covariance says {}",
                    p[i][j]
                );
            }
        }
        assert!(s[0][0] > 0.0, "a covariance factor cannot be all zeros");
    }

    /// The linear-Gaussian comparison, as the worst state and covariance departure over twelve steps.
    fn linear_kalman_departure(params: UnscentedParams) -> (f64, f64) {
        let phi = [[1.0, 0.05, 0.0], [0.0, 1.0, 0.05], [0.02, 0.0, 1.0]];
        let sq = [[0.10, 0.0, 0.0], [0.02, 0.12, 0.0], [0.0, 0.03, 0.15]];
        let s0 = [[2.0, 0.0, 0.0], [0.4, 1.7, 0.0], [-0.2, 0.3, 1.3]];
        let x0 = [0.5, -0.3, 0.1];

        let mut ukf = SquareRootUnscentedKalman::<3, 7>::new(x0, s0, params);
        let mut kf = SquareRootKalman::<3>::new(x0, s0);

        // Rotating measurement rows so every direction becomes observable; a synthetic truth supplies
        // identical measured values to both filters.
        let rows = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 0.2],
        ];
        let r = 0.25_f64;
        let mut truth = [1.0, 2.0, -1.0];

        let mut worst_x = 0.0_f64;
        let mut worst_p = 0.0_f64;
        for step in 0..12 {
            truth = matvec(&phi, &truth);
            ukf.predict(|x| matvec(&phi, x), &sq);
            kf.time_update(&phi, &sq);
            // kalman.rs::time_update advances only the covariance; the state is the caller's to
            // propagate (its doc), whereas the UKF predict advances both — so mirror it here.
            kf.x = matvec(&phi, &kf.x);

            let h = rows[step % 4];
            let measured = dot(&h, &truth);
            ukf.update::<1>(|x| [dot(&h, x)], &[measured], &[[libm::sqrt(r)]]);
            kf.measurement_update(&h, measured - dot(&h, &kf.x), r);

            let pu = ukf.covariance();
            let pk = kf.covariance();
            for i in 0..3 {
                worst_x = nan_max(worst_x, (ukf.x[i] - kf.x[i]).abs());
                for j in 0..3 {
                    worst_p = nan_max(worst_p, (pu[i][j] - pk[i][j]).abs());
                }
            }
        }
        (worst_x, worst_p)
    }

    /// Oracle (a), vector arm — exercise the `D ≥ 2` measurement path (multi-column `S_y` QR, the
    /// two-triangular-solve gain, and the multi-column downdate) that the scalar arm cannot reach. A
    /// batch two-component linear update with diagonal `R` must equal the existing filter fed the two
    /// rows *sequentially*, since a linear-Gaussian batch and its scalar decomposition share a
    /// posterior. Match to rounding, or the vector algebra is wrong.
    #[test]
    fn the_srukf_vector_update_matches_sequential_scalar() {
        let h = [[1.0, 0.5, 0.2], [-0.3, 1.0, 0.4]];
        let rvar = [0.20_f64, 0.35_f64];
        let s0 = [[1.8, 0.0, 0.0], [0.5, 1.4, 0.0], [-0.2, 0.3, 1.1]];
        let x0 = [0.4, -0.6, 0.2];
        let params = UnscentedParams::new(1.0, 2.0, 0.0);

        let mut ukf = SquareRootUnscentedKalman::<3, 7>::new(x0, s0, params);
        let mut kf = SquareRootKalman::<3>::new(x0, s0);
        let z = [1.1, -0.4];

        ukf.update::<2>(
            |x| [dot(&h[0], x), dot(&h[1], x)],
            &z,
            &[[libm::sqrt(rvar[0]), 0.0], [0.0, libm::sqrt(rvar[1])]],
        );
        for r in 0..2 {
            kf.measurement_update(&h[r], z[r] - dot(&h[r], &kf.x), rvar[r]);
        }

        let pu = ukf.covariance();
        let pk = kf.covariance();
        let mut worst = 0.0_f64;
        for i in 0..3 {
            worst = nan_max(worst, (ukf.x[i] - kf.x[i]).abs());
            for j in 0..3 {
                worst = nan_max(worst, (pu[i][j] - pk[i][j]).abs());
            }
        }
        println!("SR-UKF vector (D=2) vs sequential scalar KF: max |Δx,ΔP| {worst:.3e}");
        assert!(
            worst < 1e-11,
            "vector update disagrees with sequential scalar: {worst:.3e}"
        );
    }

    /// A minimal covariance-form UKF measurement update (scalar), forming `P`, `P_y`, `K`, and
    /// `P⁻ − K P_y Kᵀ` explicitly — the independent reference for oracle (c). It reuses only the
    /// validated pure transform for the moments; the square-root recursions under test are not shared.
    fn reference_update<const N: usize, const L: usize>(
        params: &UnscentedParams,
        x: &mut [f64; N],
        p: &mut [[f64; N]; N],
        measurement: impl Fn(&[f64; N]) -> [f64; 1],
        z: f64,
        r: f64,
    ) {
        let s = cholesky::<N>(p).unwrap();
        let (yb, pyy0) = unscented_transform::<N, 1, L>(params, x, &s, &measurement);
        let pyy = pyy0[0][0] + r;

        let w = params.weights::<N>();
        let pts = sigma_points::<N, L>(x, &s, w.gamma);
        let mut pxy = [0.0; N];
        for i in 1..L {
            let dy = measurement(&pts[i])[0] - yb[0];
            for a in 0..N {
                pxy[a] += w.wi * (pts[i][a] - x[a]) * dy;
            }
        }
        let mut k = [0.0; N];
        for a in 0..N {
            k[a] = pxy[a] / pyy;
        }
        let innov = z - yb[0];
        for a in 0..N {
            x[a] += k[a] * innov;
        }
        for a in 0..N {
            for b in 0..N {
                p[a][b] -= k[a] * k[b] * pyy;
            }
        }
    }

    /// Oracle (c) — the square-root form equals the covariance form. Run both through the same
    /// nonlinear predict (a damped pendulum step) and nonlinear scalar update (range from the origin);
    /// the reconstructed `P = S Sᵀ` must match the reference `P` to `~1e-10`, confirming the QR +
    /// cholupdate arithmetic realises the same posterior as `P⁻ − K P_y Kᵀ`.
    #[test]
    fn the_square_root_form_equals_the_plain_ukf() {
        let worst = square_root_vs_covariance(UnscentedParams::new(0.7, 2.0, 0.0));
        assert!(
            worst < 1e-10,
            "square-root and covariance forms disagree: {worst:.3e}"
        );
    }

    /// The same comparison with `W₀ᶜ < 0`, which is the only way to see the sign of the centre
    /// correction at all.
    ///
    /// The two properties have to arrive together. `..._through_the_downdate` reaches `W₀ᶜ < 0` but
    /// with a *linear* measurement, and there `ŷ = Σ Wᵢᵐ h(χᵢ) = h(x̄) = Y₀` exactly, so the centre
    /// residual is the zero vector and `cholupdate` is a no-op whatever sign it is handed. The
    /// nonlinear test above has a non-zero centre residual but `W₀ᶜ = +1.47`, so it only ever takes
    /// the `+1.0` arm. Neither could see the branch, on either side of it. `α = 0.9, β = 0, κ = 0`
    /// gives `W₀ᶜ = −0.045` at `N = 2` while keeping every weight of order one, so the comparison
    /// stays as sharp as the well-scaled arm instead of paying the `±10⁶` price of
    /// [`UnscentedParams::gaussian`].
    #[test]
    fn the_negative_centre_weight_subtracts_rather_than_adds() {
        let params = UnscentedParams::new(0.9, 0.0, 0.0);
        let w0c = params.weights::<2>().w0c;
        assert!(
            w0c < 0.0,
            "the fixture must reach the downdate, W₀ᶜ = {w0c}"
        );
        let worst = square_root_vs_covariance(params);
        assert!(
            worst < 1e-10,
            "square-root and covariance forms disagree at W₀ᶜ = {w0c}: {worst:.3e}"
        );
    }

    /// The worst disagreement between the square-root filter and the covariance-form reference over
    /// 15 nonlinear predict/update steps.
    fn square_root_vs_covariance(params: UnscentedParams) -> f64 {
        let dt = 0.1;
        let dynamics = |x: &[f64; 2]| {
            [
                x[0] + dt * x[1],
                x[1] - dt * (libm::sin(x[0]) + 0.05 * x[1]),
            ]
        };
        let meas = |x: &[f64; 2]| [libm::sqrt(x[0] * x[0] + x[1] * x[1])];

        let p0 = [[0.20, 0.02], [0.02, 0.10]];
        let sq = [[0.03, 0.0], [0.0, 0.03]];
        let x0 = [0.8, -0.2];
        let r = 0.04_f64;

        let mut ukf = SquareRootUnscentedKalman::<2, 5>::from_covariance(x0, &p0, params).unwrap();
        let mut xref = x0;
        let mut pref = p0;

        let mut worst = 0.0_f64;
        let mut truth = [1.0, 0.3];
        for _ in 0..15 {
            truth = dynamics(&truth);
            ukf.predict(dynamics, &sq);

            // Reference predict: transform mean/cov, add Q.
            let sref = cholesky::<2>(&pref).unwrap();
            let (xm, mut cov) = unscented_transform::<2, 2, 5>(&params, &xref, &sref, dynamics);
            for i in 0..2 {
                for j in 0..2 {
                    cov[i][j] += sq[i][0] * sq[j][0] + sq[i][1] * sq[j][1];
                }
            }
            xref = xm;
            pref = cov;

            let measured = meas(&truth)[0];
            ukf.update::<1>(meas, &[measured], &[[libm::sqrt(r)]]);
            reference_update::<2, 5>(&params, &mut xref, &mut pref, meas, measured, r);

            let pu = ukf.covariance();
            for i in 0..2 {
                worst = nan_max(worst, (ukf.x[i] - xref[i]).abs());
                for j in 0..2 {
                    worst = nan_max(worst, (pu[i][j] - pref[i][j]).abs());
                }
            }
        }
        println!("SR-UKF vs covariance-form UKF (nonlinear): max |ΔP,Δx| {worst:.3e}");
        worst
    }

    /// The covariance stays symmetric and positive-definite across a run of nonlinear updates — the
    /// invariant the square-root factor exists to guarantee.
    #[test]
    fn the_covariance_stays_positive_definite() {
        let params = UnscentedParams::new(0.9, 2.0, 0.0);
        let dt = 0.05;
        let dynamics = |x: &[f64; 2]| [x[0] + dt * x[1], x[1] - dt * libm::sin(x[0])];
        let meas = |x: &[f64; 2]| [x[0]];
        let sq = [[0.02, 0.0], [0.0, 0.02]];

        let mut ukf = SquareRootUnscentedKalman::<2, 5>::from_covariance(
            [0.5, 0.0],
            &[[0.3, 0.0], [0.0, 0.2]],
            params,
        )
        .unwrap();
        for _ in 0..40 {
            ukf.predict(dynamics, &sq);
            ukf.update::<1>(meas, &[0.4], &[[0.1]]);
            let p = ukf.covariance();
            for i in 0..2 {
                for j in 0..2 {
                    assert!((p[i][j] - p[j][i]).abs() < 1e-12, "asymmetry");
                }
            }
            assert!(cholesky::<2>(&p).is_some(), "lost positive-definiteness");
        }
    }

    /// The unscented filter's report check. `innovation_variance` is `tr(S_y S_yᵀ)/D` — the one place
    /// the refolded factor is observable without reaching inside — so re-running the same observation
    /// under an inflating policy must raise it by exactly the extra noise variance admitted, and
    /// `weight` must be the reciprocal of the scale that did it.
    #[test]
    fn the_outcome_reports_the_variance_the_discount_actually_admitted() {
        let params = UnscentedParams::gaussian();
        let x0 = [0.4, -0.2];
        let p0 = [[4e-2, 1e-2], [1e-2, 2.5e-2]];
        let sigma = 0.3;
        let noise_sqrt = [[sigma, 0.0], [0.0, sigma]];
        let z = [1.1, -0.9];
        let measurement = |x: &[f64; 2]| [x[0], x[0] + x[1]];

        let mut plain =
            SquareRootUnscentedKalman::<2, 5>::from_covariance(x0, &p0, params).unwrap();
        let base = plain.update_robust(measurement, &z, &noise_sqrt, RobustPolicy::Plain);
        assert!(base.accepted);
        assert_eq!(base.weight, 1.0);

        let policy = RobustPolicy::ChiSquareInflation {
            threshold: base.nominal_normalized_squared / 2.0,
        };
        let scale = policy
            .variance_scale(base.nominal_normalized_squared, 2)
            .unwrap();
        assert!(scale > 1.5, "the control must actually discount: {scale}");

        let mut inflated =
            SquareRootUnscentedKalman::<2, 5>::from_covariance(x0, &p0, params).unwrap();
        let out = inflated.update_robust(measurement, &z, &noise_sqrt, policy);

        assert_eq!(
            out.nominal_normalized_squared,
            base.nominal_normalized_squared
        );
        assert_eq!(out.weight, 1.0 / scale);
        let admitted = out.innovation_variance - base.innovation_variance;
        let expected = sigma * sigma * (scale - 1.0);
        assert!(
            (admitted - expected).abs() < 1e-9 * expected,
            "admitted {admitted:.17e}, expected {expected:.17e}"
        );
    }

    /// The log-density's normalizing constant, against the density itself.
    ///
    /// A bank of models compares likelihoods at one epoch, and every constant that does not depend on
    /// the model cancels in that comparison — so nothing in [`crate::imm`] can see whether
    /// `(2π)^{D/2}|S|` is right. The check that can is the definition: exponentiate, and compare
    /// against `exp(−δ²/2) / sqrt((2π)^D |S Sᵀ|)` formed from the determinant rather than from a sum of
    /// logarithms. Absolute log-likelihoods are reported by
    /// [`SquareRootUnscentedKalman::update_with_log_likelihood`] and used for model-consistency
    /// judgements across epochs, where the constant does not cancel.
    #[test]
    fn the_log_density_carries_the_normalizing_constant_of_a_gaussian() {
        let factor = [[2.0, 0.0], [1.0, 3.0]];
        let normalized_squared = 1.5;

        let log_density = gaussian_log_density(&factor, normalized_squared);

        let determinant = (factor[0][0] * factor[1][1]).powi(2);
        let expected = (-0.5 * normalized_squared).exp()
            / (core::f64::consts::TAU.powi(2) * determinant).sqrt();
        let density = log_density.exp();
        assert!(
            (density - expected).abs() < 1e-15 * expected,
            "density {density:.17e}, expected {expected:.17e}"
        );
    }

    /// (G1) The two square-root moves keep the identities `derivations/srukf_square_root.mac`
    /// proves for them on a general 3 × 3 factor: `cholupdate(L, v, ±1)` is a lower factor of
    /// `L Lᵀ ± v vᵀ`, and `fold_row(R, a)` an upper factor `R'` with `R'ᵀR' = RᵀR + a aᵀ`. On dyadic
    /// inputs both Gram sums are exact in f64, so the bound is 8 ulps of the largest entry, 42.
    #[test]
    fn the_square_root_moves_add_and_subtract_exactly_one_outer_product() {
        let l = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 4.0, 5.0]];
        let v = [1.0, 0.5, -0.25];
        let tol = 8.0 * f64::EPSILON * 42.0;
        for sign in [1.0, -1.0] {
            let mut factor = l;
            cholupdate(&mut factor, &v, sign);
            for i in 0..3 {
                for j in 0..3 {
                    let want = (0..3).map(|k| l[i][k] * l[j][k]).sum::<f64>() + sign * v[i] * v[j];
                    let got = (0..3).map(|k| factor[i][k] * factor[j][k]).sum::<f64>();
                    assert!(
                        (got - want).abs() <= tol,
                        "sign {sign}: [{i}][{j}] {got} vs {want}"
                    );
                    if j > i {
                        assert_eq!(
                            factor[i][j], 0.0,
                            "sign {sign}: [{i}][{j}] above the diagonal"
                        );
                    }
                }
            }
        }

        let r = [[2.0, 1.0, -1.0], [0.0, 3.0, 4.0], [0.0, 0.0, 5.0]];
        let mut folded = r;
        fold_row(&mut folded, &v);
        for i in 0..3 {
            for j in 0..3 {
                let want = (0..3).map(|k| r[k][i] * r[k][j]).sum::<f64>() + v[i] * v[j];
                let got = (0..3).map(|k| folded[k][i] * folded[k][j]).sum::<f64>();
                assert!(
                    (got - want).abs() <= tol,
                    "fold_row [{i}][{j}] {got} vs {want}"
                );
                if i > j {
                    assert_eq!(folded[i][j], 0.0, "fold_row [{i}][{j}] below the diagonal");
                }
            }
        }
    }

    /// (G3) On a linear-Gaussian step the unscented transform is exact
    /// (`derivations/unscented_transform.mac`), so one predict and one update must land on the exact
    /// rational Kalman recursion of `derivations/kalman_square_root.mac` — the same fixture as
    /// `tests/kalman.rs::the_filter_reproduces_the_exact_rational_kalman_recursion`, whose first
    /// update these literals are. Run at a positive centre weight `(1, 2, 0)` and a negative one
    /// `(0.9, 0, 0)`, where the measurement downdate meets a negative `W₀ᶜ`. The bound is 64 ulps of
    /// the largest entry, 42: the weights are of order one but the cloud is summed twice.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn the_srukf_is_the_exact_rational_kalman_step() {
        let x1 = [
            1.379_046_159_739_513_503_160_314,
            -8.522_074_315_265_274_851_561_004e-1,
            1.293_909_212_794_483_815_361_042,
        ];
        let p1 = [
            [
                2.544_292_281_172_189_235_778_587,
                3.397_433_441_869_373_683_202_452,
                -7.259_145_757_517_716_912_468_876e-2,
            ],
            [
                3.397_433_441_869_373_683_202_452,
                1.502_392_980_271_978_548_170_849e1,
                1.891_282_800_229_841_026_623_252e1,
            ],
            [
                -7.259_145_757_517_716_912_468_876e-2,
                1.891_282_800_229_841_026_623_252e1,
                4.002_140_394_560_429_036_583_03e1,
            ],
        ];
        let phi = [[1.0, 0.5, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]];
        let sq = [[0.5, 0.0, 0.0], [0.25, 0.5, 0.0], [0.0, 0.125, 0.25]];
        let h = [1.0, -0.5, 0.25];
        let tol = 64.0 * f64::EPSILON * 42.0;
        for params in [
            UnscentedParams::new(1.0, 2.0, 0.0),
            UnscentedParams::new(0.9, 0.0, 0.0),
        ] {
            let l0 = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 4.0, 5.0]];
            let mut ukf = SquareRootUnscentedKalman::<3, 7>::new([1.0, -2.0, 0.5], l0, params);
            ukf.predict(|x| matvec(&phi, x), &sq);
            // The predicted measurement is h·(Φ x₀) = 1.0625; the innovation is 5/4.
            let outcome = ukf.update_robust(
                |x| [dot(&h, x)],
                &[2.3125],
                &[[libm::sqrt(0.75)]],
                RobustPolicy::Plain,
            );
            assert!(
                (outcome.innovation_variance - 5.098_632_812_5).abs() <= tol,
                "{params:?}: f = {}",
                outcome.innovation_variance
            );
            let p = ukf.covariance();
            for i in 0..3 {
                assert!(
                    (ukf.x[i] - x1[i]).abs() <= tol,
                    "{params:?}: x[{i}] = {}",
                    ukf.x[i]
                );
                for j in 0..3 {
                    assert!(
                        (p[i][j] - p1[i][j]).abs() <= tol,
                        "{params:?}: P[{i}][{j}] = {}",
                        p[i][j]
                    );
                }
            }
        }
    }
}
