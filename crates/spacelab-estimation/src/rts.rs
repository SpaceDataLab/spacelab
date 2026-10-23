//! The Rauch–Tung–Striebel smoother: the backward pass the filter does not have.
//!
//! # What the backward pass buys
//!
//! A filter's estimate at epoch `k` conditions on measurements up to `k`. A smoother's conditions on
//! all of them, so its covariance is never larger and is much smaller in the middle of an arc, where
//! the filter is still forgetting its initialisation and the data that would pin it has not arrived
//! yet. Over a fitted arc the smoothed estimate is the one worth having; the filtered one is merely
//! the one available in real time.
//!
//! # Why not the textbook recursion
//!
//! RTS is normally written
//!
//! ```text
//! Cₖ = P⁺ₖ Φₖᵀ (P⁻ₖ₊₁)⁻¹
//! x̂ₖ = x⁺ₖ + Cₖ (x̂ₖ₊₁ − x⁻ₖ₊₁)
//! P̂ₖ = P⁺ₖ + Cₖ (P̂ₖ₊₁ − P⁻ₖ₊₁) Cₖᵀ
//! ```
//!
//! and the last line subtracts one covariance from another. `P̂ₖ₊₁ − P⁻ₖ₊₁` is negative-semidefinite
//! in exact arithmetic and only approximately so in `f64`, so the recursion can return a `P̂` with a
//! negative eigenvalue — precisely the failure that [`SquareRootKalman`](crate::SquareRootKalman)
//! is built to make unrepresentable. Bolting it onto a square-root forward pass would surrender
//! that at the last step, and on the arc's interior, where the smoother gain is largest, is exactly
//! where it would surrender it.
//!
//! # The array form
//!
//! Lower-triangularise the first `N` rows of the `2N × 2N` pre-array, applying the rotations to all
//! `2N`:
//!
//! ```text
//! [ Φₖ S⁺ₖ   S_Qₖ ]        [ S⁻ₖ₊₁   0  ]
//! [ S⁺ₖ      0    ]   ⟶    [ Xₖ      Yₖ ]
//! ```
//!
//! Reading `Post Postᵀ = Pre Preᵀ` block by block gives all three quantities the recursion needs:
//!
//! * top·top — `S⁻ S⁻ᵀ = Φ P⁺ Φᵀ + Q`, which is the filter's own time update, so the smoother and
//!   the forward pass cannot disagree about the predicted covariance;
//! * bottom·top — `X S⁻ᵀ = P⁺ Φᵀ`, hence `Cₖ = P⁺ Φᵀ (P⁻)⁻¹ = Xₖ (S⁻ₖ₊₁)⁻¹`, one triangular solve
//!   rather than an inverse;
//! * bottom·bottom — `X Xᵀ + Y Yᵀ = P⁺`, hence `Yₖ Yₖᵀ = P⁺ₖ − Cₖ P⁻ₖ₊₁ Cₖᵀ`.
//!
//! That third line is the half of `P̂` that does not depend on the future, and it arrives already
//! factored. What is left is a sum of two Gram products,
//!
//! ```text
//! P̂ₖ = Yₖ Yₖᵀ + (Cₖ Ŝₖ₊₁)(Cₖ Ŝₖ₊₁)ᵀ
//! ```
//!
//! so `Ŝₖ` is the lower-triangular factor of the `N × 2N` array `[ Yₖ | Cₖ Ŝₖ₊₁ ]`. Nothing is
//! subtracted anywhere: `P̂` is a Gram product at every step and there is no arithmetic in the
//! recursion that can make it indefinite.
//!
//! The alternative considered was the Mayne–Fraser two-filter smoother, which runs a backward
//! information filter and combines the two estimates at each epoch. It was rejected because the
//! join needs an information-to-covariance inversion at every step and it carries a second full
//! filter's state; the array form costs one extra `N × N` triangular solve per step and no second
//! filter.

// Matrix code indexes several arrays by one i/j/k, and the Kalman symbols (P, S, C, X, Y, Q) are one
// letter by universal convention.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

/// What the forward pass has to keep at epoch `k` so the backward pass can step over `k → k+1`.
///
/// Everything here is a quantity the filter already handled; the one that has to be *recorded*
/// rather than recomputed is [`predicted`](Self::predicted), because the caller propagates the state
/// and the filter never sees it.
#[derive(Clone, Copy, Debug)]
pub struct SmoothingStep<const N: usize> {
    /// The filtered estimate at epoch `k`, `x⁺ₖ` — after that epoch's measurement update.
    pub filtered: [f64; N],
    /// Its covariance square root `S⁺ₖ`, with `P⁺ₖ = S⁺ₖ S⁺ₖᵀ`.
    pub filtered_sqrt: [[f64; N]; N],
    /// The state-transition matrix `Φₖ` over `k → k+1`, as handed to
    /// [`SquareRootKalman::time_update`](crate::SquareRootKalman::time_update).
    pub stm: [[f64; N]; N],
    /// The process-noise square root `S_Qₖ` over the same interval, likewise.
    pub process_noise_sqrt: [[f64; N]; N],
    /// The state the caller's dynamics predicted at `k+1`, `x⁻ₖ₊₁`, before that epoch's measurement
    /// update.
    pub predicted: [f64; N],
}

/// The backward half of a Rauch–Tung–Striebel pass, in square-root form.
///
/// Seeded at the last epoch, where the smoothed estimate *is* the filtered one because there is no
/// future left to condition on, then stepped backwards one recorded interval at a time.
#[derive(Clone, Debug)]
pub struct RtsSmoother<const N: usize> {
    x: [f64; N],
    s: [[f64; N]; N],
}

/// Why a backward step could not be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtsError {
    /// The predicted covariance `P⁻ₖ₊₁` is singular, so the smoother gain `P⁺ Φᵀ (P⁻)⁻¹` does not
    /// exist: the prediction is deterministic along this direction, and no amount of future data
    /// says anything about where the state was in it. The offending index is reported rather than
    /// a fabricated estimate returned.
    SingularPrediction {
        /// The index of the zero diagonal in the predicted covariance square root.
        direction: usize,
    },
}

impl core::fmt::Display for RtsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SingularPrediction { direction } => write!(
                f,
                "predicted covariance is singular in direction {direction}; the smoother gain does not exist"
            ),
        }
    }
}

impl core::error::Error for RtsError {}

impl<const N: usize> RtsSmoother<N> {
    /// Seed at the last epoch from the filtered estimate there, `x⁺_M` and `S⁺_M`.
    #[must_use]
    pub const fn new(x: [f64; N], covariance_sqrt: [[f64; N]; N]) -> Self {
        Self {
            x,
            s: covariance_sqrt,
        }
    }

    /// Step back over one recorded interval, leaving the smoothed estimate at its earlier epoch.
    ///
    /// # Errors
    ///
    /// [`RtsError::SingularPrediction`] if the interval's predicted covariance is singular.
    pub fn step(&mut self, step: &SmoothingStep<N>) -> Result<(), RtsError> {
        // Pre-array, as four N×N blocks: [ Φ S⁺ | S_Q ] over [ S⁺ | 0 ].
        let mut top_left = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += step.stm[i][k] * step.filtered_sqrt[k][j];
                }
                top_left[i][j] = acc;
            }
        }
        let mut top_right = step.process_noise_sqrt;
        let mut bottom_left = step.filtered_sqrt;
        let mut bottom_right = [[0.0; N]; N];

        // Triangularise the top row block; the same rotations carry the bottom one to [X | Y].
        for i in 0..N {
            for j in (i + 1)..(2 * N) {
                let p = top_left[i][i];
                let q = if j < N {
                    top_left[i][j]
                } else {
                    top_right[i][j - N]
                };
                let denom = libm::hypot(p, q);
                if denom == 0.0 {
                    continue;
                }
                let (c, s) = (p / denom, q / denom);
                rotate_columns(&mut top_left, &mut top_right, i, j, c, s);
                rotate_columns(&mut bottom_left, &mut bottom_right, i, j, c, s);
            }
        }

        // C = X (S⁻)⁻¹, by back-substitution against the lower-triangular S⁻ = top_left.
        let mut gain = [[0.0; N]; N];
        for r in 0..N {
            for i in (0..N).rev() {
                if top_left[i][i] == 0.0 {
                    return Err(RtsError::SingularPrediction { direction: i });
                }
                let mut acc = bottom_left[r][i];
                for j in (i + 1)..N {
                    acc -= gain[r][j] * top_left[j][i];
                }
                gain[r][i] = acc / top_left[i][i];
            }
        }

        // x̂ₖ = x⁺ₖ + C (x̂ₖ₊₁ − x⁻ₖ₊₁), read off the smoothed state this step still holds.
        let mut lookahead = [0.0; N];
        for i in 0..N {
            lookahead[i] = self.x[i] - step.predicted[i];
        }
        let mut x = step.filtered;
        for i in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += gain[i][k] * lookahead[k];
            }
            x[i] += acc;
        }

        // Ŝₖ = lower-triangular factor of [ Y | C Ŝₖ₊₁ ].
        let mut left = bottom_right;
        let mut right = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += gain[i][k] * self.s[k][j];
                }
                right[i][j] = acc;
            }
        }
        for i in 0..N {
            for j in (i + 1)..(2 * N) {
                let p = left[i][i];
                let q = if j < N { left[i][j] } else { right[i][j - N] };
                let denom = libm::hypot(p, q);
                if denom == 0.0 {
                    continue;
                }
                let (c, s) = (p / denom, q / denom);
                rotate_columns(&mut left, &mut right, i, j, c, s);
            }
        }

        self.x = x;
        self.s = left;
        Ok(())
    }

    /// The smoothed estimate at the epoch the backward pass has reached.
    #[must_use]
    pub const fn state(&self) -> &[f64; N] {
        &self.x
    }

    /// Its covariance square root `Ŝ`, with `P̂ = Ŝ Ŝᵀ`.
    #[must_use]
    pub const fn covariance_sqrt(&self) -> [[f64; N]; N] {
        self.s
    }

    /// The smoothed covariance `P̂ = Ŝ Ŝᵀ`, formed only on request.
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
}

/// Apply one Givens rotation to columns `i` and `j` of a row block held as the pair
/// `[left | right]`, where column `j >= N` lives in `right`. `i` is always a left-hand column.
fn rotate_columns<const N: usize>(
    left: &mut [[f64; N]; N],
    right: &mut [[f64; N]; N],
    i: usize,
    j: usize,
    c: f64,
    s: f64,
) {
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
