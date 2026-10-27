//! The scaled unscented (sigma-point) transform: a small, deterministic set of samples that carries a
//! Gaussian through an arbitrary nonlinear map and recovers the pushed-forward mean and covariance.
//!
//! # Why sigma points
//!
//! The extended Kalman filter linearises the map at the mean and pushes the covariance through the
//! Jacobian, `P' = A P Aᵀ`. That is the first-order `crate::variational`-style prediction, and it is
//! blind to curvature — the same limitation the second-order state-transition tensor
//! (`spacelab_propagation::stt`) exists to repair. The unscented transform takes the other route: it
//! places `2N+1` deterministic *sigma points* so their sample mean is the prior mean and their sample
//! covariance is the prior covariance, propagates each through the *exact* nonlinear map, and reads the
//! posterior moments off the transformed cloud. No Jacobian, no Hessian, and the mean is captured to
//! third order for any smooth map (Julier & Uhlmann 1997; Van der Merwe 2004).
//!
//! # The scaled weights (Van der Merwe)
//!
//! With state dimension `N` and parameters `α, β, κ`, the scaling is `λ = α²(N + κ) − N` and the
//! spread `γ = √(N + λ)`. The center point carries weights
//!
//! ```text
//! W₀ᵐ = λ / (N + λ),   W₀ᶜ = λ / (N + λ) + (1 − α² + β),
//! ```
//!
//! and the `2N` off-center points share `Wᵢᵐ = Wᵢᶜ = 1 / (2(N + λ))`. `α` sets how tightly the points
//! cluster (small `α` for a strongly nonlinear map), `β` folds prior knowledge of the distribution
//! into the covariance weight (`β = 2` is optimal for a Gaussian), and `κ` is a secondary freedom
//! (`κ = 0`, or `κ = 3 − N` to match the fourth moment of a Gaussian along each axis). For an *affine*
//! map the transform is exact for **any** valid choice: the sigma cloud reproduces `P` by construction,
//! so `Σ Wᵢᶜ (A χᵢ − A x̄)(·)ᵀ = A P Aᵀ` identically — the property the linear-Gaussian oracle checks.
//!
//! Gate: `derivations/unscented_transform.mac` shows, for every `α, β, κ`, that the sigma points
//! reproduce the mean and `S Sᵀ` exactly, that the mean of any quadratic is exact, and that a scalar
//! quadratic's variance is `(b + 2aμ)²σ² + (α²κ + β)a²σ⁴` — exact precisely when `α²κ + β = 2`;
//! `derivations/linear_filters.gp` recomputes the fixture. The module test
//! `the_transform_is_exact_for_a_quadratic_where_its_algebra_says` (G3) holds the transform to it.

// Matrix code indexes several arrays by one i/j/k, and the sigma-point symbols (S, P, χ, γ) are one
// letter by universal convention; the scaled-weight locals (w0m, w0c, wi) read as the paper's symbols.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names
)]

/// The three scaling parameters of the scaled unscented transform (Van der Merwe 2004).
///
/// `alpha ∈ (0, 1]` controls the spread of the sigma points about the mean, `beta ≥ 0` injects prior
/// distribution knowledge into the covariance weight (`2.0` is optimal for a Gaussian), and `kappa`
/// is a secondary degree of freedom (`0.0`, or `3 − N` to match a Gaussian's per-axis fourth moment).
#[derive(Debug, Clone, Copy)]
pub struct UnscentedParams {
    /// Sigma-point spread. Smaller clusters the points nearer the mean for a strongly nonlinear map.
    pub alpha: f64,
    /// Distribution prior for the covariance weight; `2.0` is optimal for a Gaussian.
    pub beta: f64,
    /// Secondary scaling freedom; `0.0`, or `3 − N` to match the Gaussian fourth moment on each axis.
    pub kappa: f64,
}

impl UnscentedParams {
    /// Parameters with the given `alpha`, `beta`, and `kappa`.
    #[must_use]
    pub const fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self { alpha, beta, kappa }
    }

    /// The canonical Van der Merwe Gaussian defaults: `α = 1e-3`, `β = 2`, `κ = 0`.
    #[must_use]
    pub const fn gaussian() -> Self {
        Self {
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    /// The resolved sigma-point spread and weights for a state of dimension `N`.
    ///
    /// # Panics
    ///
    /// If the scaling `N + λ` is not positive — a spread `γ = √(N + λ)` would be imaginary, so the
    /// parameters are rejected rather than silently producing `NaN` sigma points.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // N is a tiny state dimension, exactly representable in f64
    pub(crate) fn weights<const N: usize>(&self) -> Weights {
        let n = N as f64;
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let c = n + lambda;
        assert!(
            c > 0.0,
            "unscented scaling N + λ must be positive (check alpha, kappa)"
        );
        let w0m = lambda / c;
        let wi = 1.0 / (2.0 * c);
        Weights {
            gamma: libm::sqrt(c),
            w0m,
            w0c: w0m + (1.0 - self.alpha * self.alpha + self.beta),
            wi,
        }
    }
}

/// The resolved spread `γ` and the mean/covariance weights of a sigma set (center vs off-center).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Weights {
    /// Sigma-point spread `γ = √(N + λ)`; each off-center point sits `γ` factor-columns from the mean.
    pub gamma: f64,
    /// Center-point mean weight `W₀ᵐ`.
    pub w0m: f64,
    /// Center-point covariance weight `W₀ᶜ` (may be negative — the reason the square-root form needs a
    /// rank-1 downdate rather than a plain QR).
    pub w0c: f64,
    /// Shared off-center weight `Wᵢᵐ = Wᵢᶜ = 1 / (2(N + λ))`.
    pub wi: f64,
}

/// The `2N+1` scaled sigma points of a Gaussian `N(mean, S Sᵀ)`: the mean, then `mean ± γ·(column k of
/// S)` for each column `k`. `sqrt` is any square root of the covariance (a lower-triangular Cholesky
/// factor here), so the cloud has exactly the prior covariance for an affine downstream map.
///
/// # Panics
///
/// If `L != 2N + 1` — the storage length must match the sigma-point count, checked at runtime because
/// stable Rust cannot yet size the array by `2N + 1` directly.
#[must_use]
pub(crate) fn sigma_points<const N: usize, const L: usize>(
    mean: &[f64; N],
    sqrt: &[[f64; N]; N],
    gamma: f64,
) -> [[f64; N]; L] {
    assert!(L == 2 * N + 1, "sigma-point count L must equal 2N + 1");
    let mut pts = [[0.0; N]; L];
    pts[0] = *mean;
    for k in 0..N {
        let mut plus = *mean;
        let mut minus = *mean;
        for i in 0..N {
            let d = gamma * sqrt[i][k];
            plus[i] += d;
            minus[i] -= d;
        }
        pts[1 + k] = plus;
        pts[1 + N + k] = minus;
    }
    pts
}

/// A sigma cloud's residuals about its own weighted mean, accumulated from the *center point* rather
/// than from the origin: returns `(m̄ − M₀, [Mᵢ − m̄])` for a cloud `M` under the weights `w`.
///
/// `m̄ = M₀ + Σ_{i≥1} Wᵢᵐ (Mᵢ − M₀)` is the same number as `Σ Wᵢᵐ Mᵢ` in exact arithmetic — the mean
/// weights sum to one, which [`the_mean_weights_sum_to_one`](self::tests) pins — and a very different
/// one in f64. At the Van der Merwe default `α = 1e-3` the weights are of order `1/α² = 1e6`, so the
/// direct sum builds intermediates a million times the answer and hands it back carrying
/// `ulp(1e6·M)`, while the sigma spread it is about to be subtracted from is `γS`, a *thousandth* of
/// a standard deviation. On a state of 1e7 m with a millimetre covariance that noise buried the
/// residuals outright: 99% error in the posterior mean, against 1e-9 for the same filter at
/// `|x̂| = 1` (`examples/srukf_origin_sweep.rs`). Every difference formed here is between neighbours.
///
/// The center residual is `−(m̄ − M₀)` exactly, so it is returned rather than recomputed.
pub(crate) fn centered_residuals<const M: usize, const L: usize>(
    cloud: &[[f64; M]; L],
    w: &Weights,
) -> ([f64; M], [[f64; M]; L]) {
    // The premise, checked where it is used rather than assumed: the center weight is whatever the
    // off-center ones leave over. Exact in ℝ for any `λ` with `c = N + λ`. The bound is relative
    // because at `α = 1e-3` the two terms are ±1e6 and the cancellation is in this check.
    #[allow(clippy::cast_precision_loss)] // L is 2N + 1 for a tiny state dimension
    let off_center = (L - 1) as f64;
    debug_assert!(
        libm::fabs(w.w0m + off_center * w.wi - 1.0)
            <= 8.0 * f64::EPSILON * libm::fmax(libm::fabs(w.w0m), 1.0),
        "the mean weights must sum to one"
    );
    let wi = w.wi;
    let mut resid = [[0.0; M]; L];
    for i in 1..L {
        for j in 0..M {
            resid[i][j] = cloud[i][j] - cloud[0][j];
        }
    }
    let mut offset = [0.0; M];
    for i in 1..L {
        for j in 0..M {
            offset[j] += wi * resid[i][j];
        }
    }
    for i in 1..L {
        for j in 0..M {
            resid[i][j] -= offset[j];
        }
    }
    for j in 0..M {
        resid[0][j] = -offset[j];
    }
    (offset, resid)
}

/// Push a Gaussian `N(mean, sqrt·sqrtᵀ)` through an arbitrary nonlinear `map`, returning the
/// unscented posterior mean and covariance `(ŷ, P_y)` of the `D`-dimensional output.
///
/// This is the transform in isolation — no filter state — so it is the piece the nonlinear-covariance
/// oracle exercises directly: propagate a Gaussian through the second-order flow and compare `P_y`
/// against the state-transition-tensor / Gauss–Hermite reference. The covariance uses the residual
/// form `Σ Wᵢᶜ (yᵢ − ŷ)(yᵢ − ŷ)ᵀ`, which never squares the mean and stays well-conditioned when the
/// spread is small.
///
/// # Panics
///
/// If `L != 2N + 1`.
#[must_use]
pub fn unscented_transform<const N: usize, const D: usize, const L: usize>(
    params: &UnscentedParams,
    mean: &[f64; N],
    sqrt: &[[f64; N]; N],
    map: impl Fn(&[f64; N]) -> [f64; D],
) -> ([f64; D], [[f64; D]; D]) {
    let w = params.weights::<N>();
    let pts = sigma_points::<N, L>(mean, sqrt, w.gamma);

    let mut y = [[0.0; D]; L];
    for i in 0..L {
        y[i] = map(&pts[i]);
    }

    let (offset, resid) = centered_residuals::<D, L>(&y, &w);
    let mut ybar = [0.0; D];
    for j in 0..D {
        ybar[j] = y[0][j] + offset[j];
    }

    let mut cov = [[0.0; D]; D];
    for i in 0..L {
        let weight = if i == 0 { w.w0c } else { w.wi };
        for a in 0..D {
            for b in 0..D {
                cov[a][b] += weight * resid[i][a] * resid[i][b];
            }
        }
    }
    (ybar, cov)
}

/// The lower-triangular Cholesky factor `L` of a symmetric positive-definite `a` (`a = L Lᵀ`), or
/// `None` if `a` is not positive-definite.
///
/// Shared by the sigma-point machinery wherever a covariance must be factored for sampling; the same
/// algorithm seeds [`crate::kalman::SquareRootKalman::from_covariance`]. A non-positive-definite input
/// is reported rather than papered over — a covariance that admits no factor is not a covariance.
#[must_use]
pub(crate) fn cholesky<const N: usize>(a: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut l = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = a[i][j];
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
    Some(l)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use spacelab_testkit::nan_max;

    /// Parameter sets that keep `N + λ` positive at both dimensions used below.
    const SETS: [(f64, f64, f64); 5] = [
        (1e-3, 2.0, 0.0),
        (0.9, 2.0, 1.0),
        (1.0, 0.0, -1.0),
        (0.5, 1.0, 3.0),
        (0.25, 2.0, 2.0),
    ];

    /// The mean weights reproduce a constant: `W₀ᵐ + 2N·Wᵢ = 1`.
    ///
    /// Every sigma point coincides with the mean when the deviations vanish, so weights that did not
    /// sum to one would bias the transform of a constant map. The identity holds for *any* `λ` so
    /// long as `c = N + λ`, which is why `the_transform_is_exact_for_an_affine_map` cannot see a
    /// wrong `λ`: affine exactness needs `Wᵢ γ² = ½` and `ΣWᵐ = 1`, both of which survive any
    /// rescaling of `λ` that `c` follows. What it does see is a `c` that stops tracking `λ`.
    ///
    /// The bound is 1e-9 rather than an ulp because at `α = 1e-3` the two terms are ±1e6 and their
    /// sum is 1 — the cancellation is in the assertion, not in the weights.
    #[test]
    fn the_mean_weights_sum_to_one() {
        for (alpha, beta, kappa) in SETS {
            let p = UnscentedParams::new(alpha, beta, kappa);
            let (w3, w6) = (p.weights::<3>(), p.weights::<6>());
            let (s3, s6) = (w3.w0m + 6.0 * w3.wi - 1.0, w6.w0m + 12.0 * w6.wi - 1.0);
            assert!(s3.abs() < 1e-9, "N=3 α={alpha} κ={kappa}: ΣWᵐ − 1 = {s3:e}");
            assert!(s6.abs() < 1e-9, "N=6 α={alpha} κ={kappa}: ΣWᵐ − 1 = {s6:e}");
        }
    }

    /// The spread is `α` times the spread of the unscaled transform: `γ = α √(N + κ)`.
    ///
    /// That is what `α` means — the factor by which the scaled transform pulls the sigma points in
    /// toward the mean — and it is a different expression from the `√(N + λ)` the code evaluates.
    ///
    /// Going through `λ = α²(N + κ) − N` subtracts `N` and adds it straight back, so the small-`α`
    /// regime the crate's own `gaussian()` default recommends loses digits to cancellation: measured
    /// 4.1e-12 relative at `α = 1e-3` and 3.0e-9 at `α = 1e-4`, against exact rational arithmetic.
    /// Both are far below the transform's own second-order truncation, and the weights' *sum* is
    /// unaffected because every one of them carries the same `c`; the bound here is 1e-10 to cover
    /// the worst of the sets below rather than because anything is wrong.
    #[test]
    fn the_spread_is_alpha_times_the_unscaled_spread() {
        for (alpha, beta, kappa) in SETS {
            let p = UnscentedParams::new(alpha, beta, kappa);
            for (n, w) in [(3.0, p.weights::<3>()), (6.0, p.weights::<6>())] {
                let want = alpha * libm::sqrt(n + kappa);
                assert!(
                    (w.gamma - want).abs() <= 1e-10 * want,
                    "N={n} α={alpha} κ={kappa}: γ = {}, want {want}",
                    w.gamma
                );
            }
        }
    }

    /// The covariance weight sits above the mean weight by the scaling correction `1 − α² + β`.
    ///
    /// Two statements about that correction, neither of them a restatement of the expression: it
    /// vanishes exactly when `β = α² − 1`, collapsing `W₀ᶜ` onto `W₀ᵐ` — the degenerate case where
    /// the scaled transform's covariance weight is the unscaled one — and `β` enters it with unit
    /// slope, so shifting `β` shifts `W₀ᶜ` by the same amount and nothing else.
    #[test]
    fn the_covariance_weight_differs_from_the_mean_weight_by_the_scaling_correction() {
        let (alpha, kappa) = (0.9, 1.0);
        let collapsed = UnscentedParams::new(alpha, alpha * alpha - 1.0, kappa).weights::<3>();
        assert_eq!(
            collapsed.w0c, collapsed.w0m,
            "β = α² − 1 leaves no correction, so the two weights are the same number"
        );

        let base = UnscentedParams::new(alpha, 2.0, kappa).weights::<3>();
        let shifted = UnscentedParams::new(alpha, 2.75, kappa).weights::<3>();
        let slope = shifted.w0c - base.w0c;
        assert!(
            (slope - 0.75).abs() < 1e-14,
            "β moved by 0.75 and W₀ᶜ by {slope}"
        );
        assert_eq!(shifted.w0m, base.w0m, "β must not reach the mean weight");
        assert_eq!(shifted.gamma, base.gamma, "β must not reach the spread");
    }

    /// A linear map is reproduced exactly: pushing `N(mean, P)` through `x ↦ A x + b` must give
    /// `(A mean + b, A P Aᵀ)` to rounding, for any parameter choice — the affine-exactness that
    /// underpins the linear-Gaussian filter equivalence.
    #[test]
    fn the_transform_is_exact_for_an_affine_map() {
        let mean = [1.0, -2.0, 0.5];
        // A lower-triangular square root S, so P = S Sᵀ.
        let s = [[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.3, 0.4, 1.0]];
        let a = [[1.0, 0.2, -0.1], [0.0, 1.3, 0.4], [0.5, -0.2, 0.9]];
        let b = [0.7, -0.3, 1.1];
        let params = UnscentedParams::new(1.0, 2.0, 0.0);

        let (ybar, cov) = unscented_transform::<3, 3, 7>(&params, &mean, &s, |x| {
            let mut y = b;
            for i in 0..3 {
                for j in 0..3 {
                    y[i] += a[i][j] * x[j];
                }
            }
            y
        });

        // Reference: A mean + b, and A (S Sᵀ) Aᵀ.
        let mut mean_ref = b;
        for i in 0..3 {
            for j in 0..3 {
                mean_ref[i] += a[i][j] * mean[j];
            }
        }
        let mut p = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    p[i][j] += s[i][k] * s[j][k];
                }
            }
        }
        let mut ap = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    ap[i][j] += a[i][k] * p[k][j];
                }
            }
        }
        let mut cov_ref = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    cov_ref[i][j] += ap[i][k] * a[j][k];
                }
            }
        }

        for i in 0..3 {
            assert!((ybar[i] - mean_ref[i]).abs() < 1e-12, "mean[{i}]");
            for j in 0..3 {
                assert!(
                    (cov[i][j] - cov_ref[i][j]).abs() < 1e-11,
                    "cov[{i}][{j}]: {} vs {}",
                    cov[i][j],
                    cov_ref[i][j]
                );
            }
        }
    }

    /// The residuals are taken about the transformed *mean*, not about the centre point.
    ///
    /// For an affine map the two are the same set of vectors — the offset `ŷ − Y₀` is zero, because
    /// the cloud is symmetric — so `the_transform_is_exact_for_an_affine_map` cannot see the
    /// difference and neither can any linear-Gaussian oracle. It takes curvature to separate them,
    /// and then the offset is second order in the spread: at `α = 1` on the quadratic below it is
    /// `(1.21, −0.168)` against residuals of at most 2.48, which is why this fixture uses the
    /// full-spread parameters rather than the near-linear `gaussian()` default.
    ///
    /// The reference is the definition itself, recomputed here from the same sigma points at a mean
    /// of order one where the naive sums are accurate — not a second implementation of the shortcut.
    #[test]
    fn the_residuals_are_taken_about_the_transformed_mean_not_the_center() {
        let mean = [0.3, -0.7];
        let s = [[1.1, 0.0], [-0.4, 0.9]];
        let params = UnscentedParams::new(1.0, 2.0, 0.0);
        let map = |x: &[f64; 2]| [x[0] * x[0] + x[1], libm::sin(x[0]) * x[1]];

        let (ybar, cov) = unscented_transform::<2, 2, 5>(&params, &mean, &s, map);

        let w = params.weights::<2>();
        let pts = sigma_points::<2, 5>(&mean, &s, w.gamma);
        let y: [[f64; 2]; 5] = core::array::from_fn(|i| map(&pts[i]));

        let mut want_mean = [0.0; 2];
        for j in 0..2 {
            want_mean[j] = w.w0m * y[0][j];
            for i in 1..5 {
                want_mean[j] += w.wi * y[i][j];
            }
        }
        let mut want_cov = [[0.0; 2]; 2];
        for i in 0..5 {
            let weight = if i == 0 { w.w0c } else { w.wi };
            for a in 0..2 {
                for b in 0..2 {
                    want_cov[a][b] += weight * (y[i][a] - want_mean[a]) * (y[i][b] - want_mean[b]);
                }
            }
        }

        for a in 0..2 {
            assert!((ybar[a] - want_mean[a]).abs() < 1e-13, "mean[{a}]");
            for b in 0..2 {
                assert!(
                    (cov[a][b] - want_cov[a][b]).abs() < 1e-13,
                    "cov[{a}][{b}]: {} vs {}",
                    cov[a][b],
                    want_cov[a][b]
                );
            }
        }
    }

    /// Affine exactness has to survive a mean that dwarfs its own covariance — a position in metres
    /// against a millimetre covariance, which is where a converged orbit-determination filter lives.
    ///
    /// The test above runs at `α = 1`, which spreads the sigma points a full standard deviation and
    /// keeps every weight of order one. The default `α = 1e-3` does neither: the cloud spans 1e-9 of
    /// the mean and the weights are 1e6, so `Σ Wᵢᵐ Yᵢ` is a sum of terms a million times its own
    /// answer. Measured with that accumulation, **4.9e-2** worst covariance departure; **2.6e-3**
    /// with [`centered_residuals`].
    ///
    /// The bound is 1e-2 rather than an ulp because 2.6e-3 is what this transform can do here, and
    /// the residue is not the accumulation's: `sigma_points` can only offset a component of 1e6 by
    /// `γS = 3.5e-6`, which is quantized at `ulp(1e6) = 1.2e-10`, and the center covariance weight
    /// `W₀ᶜ ≈ −1e6` then multiplies the *square* of a mean offset that is zero in exact arithmetic.
    /// The square-root form avoids that amplification — same decade, 7.7e-7 — so the covariance-form
    /// transform is the one to keep away from a large state. `docs/limitations.md` carries the sweep.
    #[test]
    fn the_transform_is_exact_for_an_affine_map_far_from_the_origin() {
        let mean = [1e6, -2e6, 5e5];
        let s = [[2e-3, 0.0, 0.0], [5e-4, 1.5e-3, 0.0], [-3e-4, 4e-4, 1e-3]];
        let a = [[1.0, 0.2, -0.1], [0.0, 1.3, 0.4], [0.5, -0.2, 0.9]];
        let params = UnscentedParams::gaussian();

        let (_, cov) = unscented_transform::<3, 3, 7>(&params, &mean, &s, |x| {
            let mut y = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    y[i] += a[i][j] * x[j];
                }
            }
            y
        });

        // A (S Sᵀ) Aᵀ, formed as (A S)(A S)ᵀ so the reference itself never squares the mean.
        let mut asx = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    asx[i][j] += a[i][k] * s[k][j];
                }
            }
        }
        let mut want = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    want[i][j] += asx[i][k] * asx[j][k];
                }
            }
        }
        // Scaled by `√(Pᵢᵢ Pⱼⱼ)`, not by the entry itself: an off-diagonal may be near zero without
        // the transform being wrong there, and dividing by it would report the covariance's own
        // conditioning as an error.
        let mut worst: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let scale = (want[i][i] * want[j][j]).sqrt();
                worst = worst.max((cov[i][j] - want[i][j]).abs() / scale);
            }
        }
        assert!(worst < 1e-2, "worst covariance departure {worst:e}");
    }

    /// (G3) On a quadratic the transform is exact where `derivations/unscented_transform.mac` says
    /// it is and off by exactly what it says otherwise. For `f = a x² + b x` of `x ~ N(μ, σ²)` the
    /// mean is `aμ² + bμ + aσ²` for every parameter choice and the variance is
    /// `(b + 2aμ)²σ² + (α²κ + β) a²σ⁴`, the Gaussian's own `2a²σ⁴` precisely when `α²κ + β = 2`:
    /// at `(1, 0, 2)` that is 1.28125, at `(½, 2, 2)` it is 1.3515625. A two-state quadratic
    /// `xᵀAx + bᵀx + d` has the mean `f(x̄) + tr(A P)`, here 1. Every value is dyadic; the bound is
    /// 8 ulps of 2.
    #[test]
    fn the_transform_is_exact_for_a_quadratic_where_its_algebra_says() {
        let tol = 8.0 * f64::EPSILON * 2.0;
        let scalar = |x: &[f64; 1]| [1.5 * x[0] * x[0] - 0.25 * x[0]];
        for ((alpha, beta, kappa), variance) in
            [((1.0, 0.0, 2.0), 1.281_25), ((0.5, 2.0, 2.0), 1.351_562_5)]
        {
            let params = UnscentedParams::new(alpha, beta, kappa);
            let (mean, cov) = unscented_transform::<1, 1, 3>(&params, &[0.75], &[[0.5]], scalar);
            assert!(
                (mean[0] - 1.031_25).abs() <= tol,
                "α = {alpha}: mean {}",
                mean[0]
            );
            assert!(
                (cov[0][0] - variance).abs() <= tol,
                "α = {alpha}: variance {}",
                cov[0][0]
            );
        }

        let params = UnscentedParams::new(0.5, 2.0, 1.0);
        let s = [[0.5, 0.0], [0.25, 1.0]];
        let (mean, _) = unscented_transform::<2, 1, 5>(&params, &[1.0, -0.5], &s, |x| {
            [x[0] * x[0] + x[0] * x[1] - 2.0 * x[1] * x[1] + 0.25 * x[0] + x[1] + 3.0]
        });
        assert!((mean[0] - 1.0).abs() <= tol, "two-state mean {}", mean[0]);
    }

    /// The Cholesky helper round-trips: `L Lᵀ` reconstructs the input, and an indefinite matrix is
    /// rejected rather than returning a bogus factor.
    #[test]
    fn cholesky_factors_and_rejects() {
        let p = [[4.0, 2.0, 0.6], [2.0, 3.0, 0.4], [0.6, 0.4, 1.0]];
        let l = cholesky::<3>(&p).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += l[i][k] * l[j][k];
                }
                assert!((acc - p[i][j]).abs() < 1e-12, "L Lᵀ [{i}][{j}]");
            }
        }
        // Not positive-definite (negative eigenvalue): rejected.
        let bad = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(cholesky::<3>(&bad).is_none());
    }

    /// Oracle (b) — nonlinear covariance against the in-repo state-transition-tensor / Gauss–Hermite
    /// reference. Push a Gaussian through the *validated* second-order two-body + J₂ flow
    /// (`map_deviation`, the same quadratic map [`spacelab_propagation::stt`] checks against
    /// Gauss–Hermite) and compare the unscented posterior to two independent references on that flow:
    /// the analytic Park–Scheeres moments (`map_covariance`) and a 3-point-per-axis Gauss–Hermite
    /// quadrature (exact for the ≤4th moments of a quadratic map).
    ///
    /// The unscented mean must match to rounding — the `2N+1` axial rule integrates a quadratic's mean
    /// exactly. The covariance does not: the axial sigma points carry no *cross*-fourth-moment
    /// (`E[δxᵢ²δxⱼ²]`, `i ≠ j`), which the full Gauss–Hermite grid does, so the two disagree by exactly
    /// that kurtosis term of the (already small) second-order inflation. We measure and report the gap;
    /// this is the documented unscented-vs-Gauss–Hermite tolerance, not a defect.
    #[test]
    #[allow(clippy::too_many_lines)] // one self-contained oracle: STT integration, UT, and two references
    fn the_nonlinear_covariance_matches_the_stt_and_gauss_hermite() {
        use spacelab_propagation::driver::{Controls, integrate};
        use spacelab_propagation::forces::{J2, Newtonian, PointMass};
        use spacelab_propagation::stt::{
            VariationalStt, augment_stt, augmented_len, map_covariance, map_deviation, split_stt,
        };
        use spacelab_propagation::{Dop853, Tolerance};

        const MU: f64 = 3.986_004_418e14;
        const RE: f64 = 6_378_137.0;
        const J2_COEFF: f64 = 1.082_626_68e-3;
        const M: usize = augmented_len(6);

        // Propagate the two-body + J₂ state, STM Φ, and STT Ψ to ~⅓ orbit (mirrors tests/stt.rs).
        let y0 = [6.9e6, 1.0e6, 0.8e6, -1.5e3, 6.8e3, 2.5e3];
        let t1 = 1800.0;
        let force = Newtonian((
            PointMass { mu: MU },
            J2 {
                mu: MU,
                re: RE,
                j2: J2_COEFF,
            },
        ));
        let sys = VariationalStt::<6, _>::new(&force);
        let aug0: [f64; M] = augment_stt::<6, M>(&y0);
        let traj = integrate(
            &Dop853,
            &sys,
            0.0,
            &aug0,
            t1,
            10.0,
            Controls::new(Tolerance::new(1e-12, 1e-12), 1e-6),
        )
        .unwrap();
        let (_, phi, psi) = split_stt::<6, M>(&traj.final_state());

        // The validated quadratic flow δx₀ ↦ Φ δx₀ + ½ Ψ δx₀ δx₀, and a diagonal prior.
        let g = |dx0: &[f64; 6]| map_deviation(&phi, &psi, dx0);
        let sigma = [500.0, 500.0, 500.0, 0.5, 0.5, 0.5];
        let mut p0 = [[0.0; 6]; 6];
        for i in 0..6 {
            p0[i][i] = sigma[i] * sigma[i];
        }
        let s0 = cholesky::<6>(&p0).unwrap();

        // Unscented transform with κ = 3 − N so the axial nodes sit at ±√3, aligned with the
        // Gauss–Hermite abscissae.
        let params = UnscentedParams::new(1.0, 2.0, 3.0 - 6.0);
        let (ut_mean, ut_cov) = unscented_transform::<6, 6, 13>(&params, &[0.0; 6], &s0, g);

        // Reference 1: analytic Park–Scheeres second-order moments.
        let (an_mean, an_cov) = map_covariance(&phi, &psi, &p0);

        // Reference 2: 3-point-per-axis Gauss–Hermite (nodes ±√3, 0; weights 1/6, 2/3, 1/6), exact for
        // the ≤4th moments of the quadratic map.
        let nodes = [-(3.0_f64.sqrt()), 0.0, 3.0_f64.sqrt()];
        let wts = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        let mut mean_gh = [0.0; 6];
        let mut second = [[0.0; 6]; 6];
        for gpt in 0..3usize.pow(6) {
            let mut idx = gpt;
            let mut w = 1.0;
            let mut x = [0.0; 6];
            for dim in 0..6 {
                let t = idx % 3;
                idx /= 3;
                w *= wts[t];
                x[dim] = sigma[dim] * nodes[t];
            }
            let y = g(&x);
            for i in 0..6 {
                mean_gh[i] += w * y[i];
                for j in 0..6 {
                    second[i][j] += w * y[i] * y[j];
                }
            }
        }
        let mut cov_gh = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                cov_gh[i][j] = second[i][j] - mean_gh[i] * mean_gh[j];
            }
        }

        // Scales and gaps.
        let mut cov_scale = 0.0_f64;
        let mut mean_scale = 0.0_f64;
        for i in 0..6 {
            cov_scale = nan_max(cov_scale, cov_gh[i][i].abs());
            mean_scale = nan_max(mean_scale, mean_gh[i].abs());
        }
        let mut mean_err_gh = 0.0_f64;
        let mut mean_err_an = 0.0_f64;
        let mut cov_err_gh = 0.0_f64;
        let mut cov_err_an = 0.0_f64;
        for i in 0..6 {
            mean_err_gh = nan_max(mean_err_gh, (ut_mean[i] - mean_gh[i]).abs());
            mean_err_an = mean_err_an.max((ut_mean[i] - an_mean[i]).abs());
            for j in 0..6 {
                cov_err_gh = nan_max(cov_err_gh, (ut_cov[i][j] - cov_gh[i][j]).abs());
                cov_err_an = cov_err_an.max((ut_cov[i][j] - an_cov[i][j]).abs());
            }
        }

        println!(
            "UT vs Gauss-Hermite: mean rel err {:.3e}, cov rel err {:.3e} (mean scale {mean_scale:.3e} m, cov scale {cov_scale:.3e})",
            mean_err_gh / mean_scale,
            cov_err_gh / cov_scale
        );
        println!(
            "UT vs analytic STT (Park-Scheeres): mean rel err {:.3e}, cov rel err {:.3e}",
            mean_err_an / mean_scale,
            cov_err_an / cov_scale
        );

        assert!(
            mean_scale > 0.0,
            "the curvature must bias the mean off zero"
        );
        // Mean: the axial rule integrates the quadratic mean exactly.
        assert!(
            mean_err_gh / mean_scale < 1e-9,
            "UT mean must match Gauss-Hermite to rounding, got {:.3e}",
            mean_err_gh / mean_scale
        );
        // Covariance: the missing cross-kurtosis of the axial rule is a small fraction of the
        // (linear-dominated) covariance. Measured at ~2.1e-8 relative for this LEO J₂ arc — the same
        // order as the analytic-STT-vs-Gauss–Hermite floor, so the axial unscented rule loses almost
        // nothing here. The bound is set with headroom above the measured value, not assumed.
        assert!(
            cov_err_gh / cov_scale < 1e-7,
            "UT covariance gap to Gauss-Hermite exceeds the expected kurtosis bound, got {:.3e}",
            cov_err_gh / cov_scale
        );
    }
}
