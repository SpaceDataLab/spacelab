//! The sequential square-root Kalman filter: covariance carried as its square root, updated by
//! orthogonal array transformations, never formed and never symmetrised.
//!
//! # Why the square root
//!
//! A conventional Kalman filter stores `P` and updates it as `(I − KH)P̄` or the Joseph form. Both
//! can, over many steps in finite precision, lose symmetry or positive-definiteness — a covariance
//! with a negative eigenvalue is a filter that has diverged and does not know it. Carrying the factor
//! `S` with `P = S Sᵀ` makes `P` positive-semidefinite *by construction*: there is no arithmetic that
//! can produce an indefinite covariance, so the stabilising `Symmetrize(P) = (P + Pᵀ)/2` that GMAT
//! calls after every update (`ExtendedKalmanFilter.cpp`) is a no-op here and is absent. This is the
//! same argument, and the same orthogonal machinery, as the batch [`Srif`](crate::Srif); the two
//! are the information- and covariance-side duals of one method.
//!
//! # Both updates are one LQ
//!
//! * **Time update** `P⁻ = Φ P⁺ Φᵀ + Q`: lower-triangularise the pre-array `[Φ S⁺ | S_Q]` (`N × 2N`)
//!   to `[S⁻ | 0]`. Then `S⁻ S⁻ᵀ = Φ S⁺ S⁺ᵀ Φᵀ + S_Q S_Qᵀ`.
//! * **Measurement update** (scalar): lower-triangularise the pre-array
//!   `[[√R, h S⁻], [0, S⁻]]` (`(N+1) × (N+1)`) to `[[√f, 0], [√f K, S⁺]]`. Reading `Post Postᵀ =
//!   Pre Preᵀ` off the blocks gives the innovation variance `f = h P⁻ hᵀ + R`, the gain
//!   `K = P⁻ hᵀ / f`, and `S⁺ S⁺ᵀ = P⁻ − K f Kᵀ = P⁺` in one transformation.
//!
//! The array form is taken over Potter / Carlson / U-D deliberately (notes/14 §2): it is the modern
//! equivalent and it keeps `S` triangular.

// Matrix code indexes several arrays by one i/j/k, and the Kalman symbols (P, S, K, H, Q, f) are one
// letter by universal convention.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

/// A sequential square-root Kalman filter over an `N`-dimensional state.
///
/// The estimate [`x`](Self::x) is public: the caller advances it by its own dynamics between steps
/// (nonlinearly, for an extended filter), while [`time_update`](Self::time_update) advances the
/// covariance through the corresponding state-transition matrix.
#[derive(Debug, Clone)]
pub struct SquareRootKalman<const N: usize> {
    /// The state estimate. Propagated by the caller's dynamics; corrected by
    /// [`measurement_update`](Self::measurement_update).
    pub x: [f64; N],
    /// Covariance square root `S`, with `P = S Sᵀ`. Never `P` itself, never symmetrised.
    s: [[f64; N]; N],
}

impl<const N: usize> SquareRootKalman<N> {
    /// Seed with an estimate and the square root of its covariance (`P₀ = S₀ S₀ᵀ`; any square root,
    /// e.g. a Cholesky factor from [`from_covariance`](Self::from_covariance)).
    #[must_use]
    pub const fn new(x0: [f64; N], covariance_sqrt: [[f64; N]; N]) -> Self {
        Self {
            x: x0,
            s: covariance_sqrt,
        }
    }

    /// Seed from an estimate and a full covariance matrix, taking its lower Cholesky factor.
    ///
    /// Returns `None` if `p0` is not positive-definite — a covariance that is not is not a covariance,
    /// and the filter refuses to start from one rather than papering over it.
    #[must_use]
    pub fn from_covariance(x0: [f64; N], p0: &[[f64; N]; N]) -> Option<Self> {
        let mut l = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..=i {
                let mut sum = p0[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return None;
                    }
                    l[i][j] = libm::sqrt(sum);
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        Some(Self { x: x0, s: l })
    }

    /// Time update: carry the covariance through the state-transition matrix and add process noise,
    /// `P⁻ = Φ P⁺ Φᵀ + Q`, in square-root form. `process_noise_sqrt` is any `S_Q` with `Q = S_Q S_Qᵀ`
    /// (the zero matrix for a noiseless dynamic). The state estimate is the caller's to propagate.
    pub fn time_update(&mut self, stm: &[[f64; N]; N], process_noise_sqrt: &[[f64; N]; N]) {
        // Pre-array [Φ S | S_Q], as two N×N blocks: left = Φ S, right = S_Q.
        let mut left = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += stm[i][k] * self.s[k][j];
                }
                left[i][j] = acc;
            }
        }
        let mut right = *process_noise_sqrt;

        // Lower-triangularise `left`, rotating away everything to the right of its diagonal — the
        // super-diagonal of `left` and the whole of `right` — into it. Column `j ≥ N` lives in `right`.
        for i in 0..N {
            for j in (i + 1)..(2 * N) {
                let p = left[i][i];
                let q = if j < N { left[i][j] } else { right[i][j - N] };
                let denom = libm::hypot(p, q);
                if denom == 0.0 {
                    continue;
                }
                let (c, s) = (p / denom, q / denom);
                for r in 0..N {
                    let a_i = left[r][i];
                    let a_j = if j < N { left[r][j] } else { right[r][j - N] };
                    left[r][i] = c * a_i + s * a_j;
                    let updated = -s * a_i + c * a_j;
                    if j < N {
                        left[r][j] = updated;
                    } else {
                        right[r][j - N] = updated;
                    }
                }
            }
        }
        self.s = left;
    }

    /// Measurement update for a scalar observation with partial `h` (`∂z/∂x`), innovation
    /// `ν = z − ẑ` (the caller forms `ẑ`, so a nonlinear measurement is handled by passing its
    /// predicted value), and noise variance `R`. Returns the innovation variance `f = h P⁻ hᵀ + R`,
    /// so the caller can gate on the normalised residual `ν²/f`.
    pub fn measurement_update(
        &mut self,
        h: &[f64; N],
        innovation: f64,
        noise_variance: f64,
    ) -> f64 {
        // Border form of the (N+1)×(N+1) pre-array:
        //   alpha = √R (top-left), top = h S (top-right row), col = 0 (below alpha), block = S.
        let mut alpha = libm::sqrt(noise_variance);
        let mut top = [0.0; N];
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += h[k] * self.s[k][j];
            }
            top[j] = acc;
        }
        let mut col = [0.0; N];
        let mut block = self.s;

        // Row 0: rotate each top[j] into column 0. Afterwards alpha = √f and col = √f·K.
        for j in 0..N {
            let denom = libm::hypot(alpha, top[j]);
            if denom == 0.0 {
                continue;
            }
            let (c, s) = (alpha / denom, top[j] / denom);
            alpha = denom;
            top[j] = 0.0;
            for r in 0..N {
                let a0 = col[r];
                let aj = block[r][j];
                col[r] = c * a0 + s * aj;
                block[r][j] = -s * a0 + c * aj;
            }
        }

        // Lower-triangularise the remaining block into S⁺ (column 0 is untouched by these).
        for i in 0..N {
            for j in (i + 1)..N {
                let denom = libm::hypot(block[i][i], block[i][j]);
                if denom == 0.0 {
                    continue;
                }
                let (c, s) = (block[i][i] / denom, block[i][j] / denom);
                for r in 0..N {
                    let b_i = block[r][i];
                    let b_j = block[r][j];
                    block[r][i] = c * b_i + s * b_j;
                    block[r][j] = -s * b_i + c * b_j;
                }
            }
        }

        self.s = block;
        if alpha != 0.0 {
            for i in 0..N {
                self.x[i] += (col[i] / alpha) * innovation;
            }
        }
        alpha * alpha
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

    /// The covariance square root `S`.
    #[must_use]
    pub fn covariance_sqrt(&self) -> [[f64; N]; N] {
        self.s
    }
}
