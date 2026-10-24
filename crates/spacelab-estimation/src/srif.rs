//! The square-root information filter: least squares by orthogonal triangularisation, never by the
//! normal equations.
//!
//! # Why not the normal equations
//!
//! The obvious batch solver forms the information matrix `Λ = HᵀWH` and inverts it. It is wrong, and
//! measurably so: `cond(HᵀWH) = cond(H)²`, and orbit-determination Jacobians routinely reach
//! `cond(H) ≈ 10⁶–10⁸` (a short arc barely separates semi-major axis from mean motion). Squaring the
//! condition number reaches the `f64` limit — `derivations/normal_equations_vs_qr.m`, which builds
//! `H` from prescribed singular values, shows that at `cond(H) = 10⁸` the normal equations retain
//! about **one** correct digit while QR retains **nine**. `tests/srif.rs` measures the same effect
//! one regime lower, where a Vandermonde can actually be taken: at `cond(H) = 3.4·10⁵` the two
//! errors are 1.6e-5 and 6.1e-12. GMAT forms the normal equations and inverts them
//! (`BatchEstimator.cpp`, after the GTDS mathematical specification, eq. 8-57); Orekit does not;
//! nor do we.
//!
//! # What this maintains
//!
//! An upper-triangular **information square root** `R` and a transformed right-hand side `d`, such
//! that the least-squares problem `min ‖ W^½(Hx − z) ‖` has solution `x = R⁻¹ d`, and the estimate
//! covariance is `P = (HᵀWH)⁻¹ = R⁻¹ R⁻ᵀ`. Each measurement row is folded into `R` by Givens
//! rotations — orthogonal, so the condition number is never squared, and allocation-free. Rows can
//! be added one at a time (sequential) or all at once (batch); the two give the same `R`.
//!
//! `R` is the Cholesky factor of the information matrix, so an *a priori* covariance `P₀` enters as
//! `R = chol(P₀⁻¹)` — folded in as if it were another measurement block, never as the matrix
//! addition `information += Pdx0_inv` that GMAT's `BatchEstimator.cpp` uses.

#![allow(clippy::needless_range_loop, clippy::many_single_char_names)] // matrix code indexes several arrays by one i/j/k

/// A square-root information filter over an `N`-dimensional state.
///
/// Accumulates whitened measurement rows into the upper-triangular factor `r` and the transformed
/// right-hand side `d`. Solve at any point; keep adding rows and solve again.
#[derive(Debug, Clone)]
pub struct Srif<const N: usize> {
    /// Upper-triangular information square root; `r[i][j]` for `j >= i`, zero below.
    r: [[f64; N]; N],
    /// Transformed right-hand side.
    d: [f64; N],
    /// Accumulated sum of squared post-fit residuals (the part rotated past row `N`).
    residual_ss: f64,
    /// Number of scalar measurements folded in.
    rows: u32,
}

impl<const N: usize> Default for Srif<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Srif<N> {
    /// An empty filter: no information, no rows. Solving before it is full-rank fails.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            r: [[0.0; N]; N],
            d: [0.0; N],
            residual_ss: 0.0,
            rows: 0,
        }
    }

    /// Seed with an a-priori estimate `x0` and its information square root `r0` (`= chol(P₀⁻¹)`,
    /// upper triangular). The prior contributes `d = R x0`.
    #[must_use]
    pub fn with_prior(x0: &[f64; N], r0: &[[f64; N]; N]) -> Self {
        let mut d = [0.0; N];
        for i in 0..N {
            let mut acc = 0.0;
            for j in i..N {
                acc += r0[i][j] * x0[j];
            }
            d[i] = acc;
        }
        Self {
            r: *r0,
            d,
            residual_ss: 0.0,
            rows: 0,
        }
    }

    /// Fold one scalar measurement `zᵢ = hᵢᵀ x + noise`, with `1/σ²` weight, into `R`.
    ///
    /// The row and value are pre-whitened by `√weight`, then rotated into the triangle by Givens
    /// rotations — orthogonal, so nothing squares the condition number.
    pub fn add_row(&mut self, h: &[f64; N], z: f64, weight: f64) {
        let w = libm::sqrt(weight);
        let mut a = *h;
        for v in &mut a {
            *v *= w;
        }
        let mut zz = z * w;

        for k in 0..N {
            if a[k] == 0.0 {
                continue;
            }
            let rkk = self.r[k][k];
            let denom = libm::hypot(rkk, a[k]);
            if denom == 0.0 {
                continue;
            }
            let c = rkk / denom;
            let s = a[k] / denom;
            self.r[k][k] = denom;
            for j in (k + 1)..N {
                let temp = c * self.r[k][j] + s * a[j];
                a[j] = c * a[j] - s * self.r[k][j];
                self.r[k][j] = temp;
            }
            let temp = c * self.d[k] + s * zz;
            zz = c * zz - s * self.d[k];
            self.d[k] = temp;
        }
        // whatever remains in `zz` is a post-fit residual contribution
        self.residual_ss += zz * zz;
        self.rows += 1;
    }

    /// Solve for the state estimate `x = R⁻¹ d` by back-substitution.
    ///
    /// # Errors
    ///
    /// [`SrifError::RankDeficient`] if any diagonal of `R` is zero — the problem is not yet
    /// observable in every direction, and the refiner refuses to invent a solution.
    pub fn solve(&self) -> Result<[f64; N], SrifError> {
        let mut x = [0.0; N];
        for i in (0..N).rev() {
            if self.r[i][i] == 0.0 {
                return Err(SrifError::RankDeficient { direction: i });
            }
            let mut acc = self.d[i];
            for j in (i + 1)..N {
                acc -= self.r[i][j] * x[j];
            }
            x[i] = acc / self.r[i][i];
        }
        Ok(x)
    }

    /// The estimate covariance `P = R⁻¹ R⁻ᵀ`.
    ///
    /// # Errors
    ///
    /// [`SrifError::RankDeficient`] if `R` is singular.
    pub fn covariance(&self) -> Result<[[f64; N]; N], SrifError> {
        // R⁻¹ (upper triangular) by back-substitution, column by column.
        let mut rinv = [[0.0; N]; N];
        for col in 0..N {
            for i in (0..N).rev() {
                if self.r[i][i] == 0.0 {
                    return Err(SrifError::RankDeficient { direction: i });
                }
                let mut acc = f64::from(u8::from(i == col));
                for j in (i + 1)..N {
                    acc -= self.r[i][j] * rinv[j][col];
                }
                rinv[i][col] = acc / self.r[i][i];
            }
        }
        // P = R⁻¹ (R⁻¹)ᵀ
        let mut p = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += rinv[i][k] * rinv[j][k];
                }
                p[i][j] = acc;
            }
        }
        Ok(p)
    }

    /// The upper-triangular information square root `R`.
    #[must_use]
    pub fn information_sqrt(&self) -> [[f64; N]; N] {
        self.r
    }

    /// The accumulated post-fit residual sum of squares and the number of measurements. Their ratio
    /// against the degrees of freedom is the reduced chi-square — the honest goodness-of-fit.
    #[must_use]
    pub fn residual_stats(&self) -> (f64, u32) {
        (self.residual_ss, self.rows)
    }
}

/// Why a solve failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrifError {
    /// The information matrix is singular: a state direction is unobserved. The offending diagonal
    /// index is reported rather than a fabricated estimate returned.
    RankDeficient {
        /// The state index whose information diagonal was zero.
        direction: usize,
    },
}

impl core::fmt::Display for SrifError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RankDeficient { direction } => {
                write!(
                    f,
                    "state direction {direction} is unobserved (singular information matrix)"
                )
            }
        }
    }
}

impl core::error::Error for SrifError {}
