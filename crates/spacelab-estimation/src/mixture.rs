//! A Gaussian-mixture state estimate: a weighted set of Gaussian components, each carried in
//! square-root form and each unscented-propagated, for uncertainty that a single Gaussian cannot hold.
//!
//! # Why a mixture
//!
//! A single Gaussian is the wrong shape for a banana-shaped orbit-uncertainty tube, a bimodal
//! post-manoeuvre ambiguity, or any distribution the flow has bent away from an ellipsoid. Splitting
//! the prior into several narrow Gaussians makes each component nearly linear over its own support, so
//! the unscented (or STT) prediction stays accurate where a lone wide Gaussian would smear across the
//! curvature (`Terejanu 2008`; the Gaussian-sum filter, `Alspach & Sorenson 1972`). This is the
//! non-Gaussian companion to [`crate::SquareRootUnscentedKalman`].
//!
//! # What this holds
//!
//! `C` components, each a weight and a Gaussian in square-root form (mean and Cholesky factor `S`,
//! `P = S Sᵀ`, sharing the representation of the filters). [`propagate`](GaussianMixture::propagate)
//! advances every component through the same square-root unscented time update; `collapse` moment-
//! matches the whole mixture back to a single mean and covariance by the law of total covariance.
//! Adaptive splitting is deliberately out of scope — hold, propagate, collapse is the core.
//!
//! Gate: `derivations/gaussian_mixture_moments.mac` integrates each component against the standard
//! normal density, shows the collapse is the mixture density's mean and covariance (so merging
//! components preserves the first two moments) and that a linear flow commutes with it; the module
//! test `the_collapse_has_the_moments_of_the_mixture_density` (G4) checks the shipped collapse
//! against its exact fixture.

// Matrix code indexes several arrays by one i/j/k, and the mixture symbols (S, P, μ, w) follow the
// domain's notation.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::srukf::unscented_predict;
use crate::unscented::UnscentedParams;

/// One weighted Gaussian component of a mixture, in square-root form.
#[derive(Debug, Clone, Copy)]
pub struct GaussianComponent<const N: usize> {
    /// Mixture weight. Weights across the mixture are expected to sum to one; see
    /// [`GaussianMixture::normalize`].
    pub weight: f64,
    /// Component mean.
    pub mean: [f64; N],
    /// Component covariance square root `S` (lower-triangular, `P = S Sᵀ`).
    pub sqrt_cov: [[f64; N]; N],
}

/// A Gaussian mixture over an `N`-dimensional state with `C` components and `L = 2N + 1` sigma points
/// per component.
#[derive(Debug, Clone)]
pub struct GaussianMixture<const N: usize, const C: usize, const L: usize> {
    /// The mixture components.
    pub components: [GaussianComponent<N>; C],
    /// The scaled-transform parameters shared by every component's unscented propagation.
    params: UnscentedParams,
}

impl<const N: usize, const C: usize, const L: usize> GaussianMixture<N, C, L> {
    /// Build a mixture from its components and the transform parameters.
    ///
    /// # Panics
    ///
    /// If `L != 2N + 1` — the per-component sigma storage would be mis-sized.
    #[must_use]
    pub const fn new(components: [GaussianComponent<N>; C], params: UnscentedParams) -> Self {
        assert!(L == 2 * N + 1, "sigma-point count L must equal 2N + 1");
        Self { components, params }
    }

    /// Rescale the component weights to sum to one. A no-op mathematically when they already do; a
    /// guard against drift after resampling or pruning. Does nothing if the weights sum to zero — an
    /// all-zero mixture has no normalisation, and inventing one would hide the caller's error.
    pub fn normalize(&mut self) {
        let total: f64 = self.components.iter().map(|c| c.weight).sum();
        if total == 0.0 {
            return;
        }
        for c in &mut self.components {
            c.weight /= total;
        }
    }

    /// Propagate every component through the nonlinear `dynamics` with process noise `Q = S_Q S_Qᵀ`,
    /// using the same square-root unscented time update as [`crate::SquareRootUnscentedKalman`].
    /// Weights are invariant under a deterministic flow, so only the means and factors move.
    pub fn propagate(
        &mut self,
        dynamics: impl Fn(&[f64; N]) -> [f64; N],
        process_noise_sqrt: &[[f64; N]; N],
    ) {
        for c in &mut self.components {
            let (mean, s) = unscented_predict::<N, L>(
                &self.params,
                &c.mean,
                &c.sqrt_cov,
                &dynamics,
                process_noise_sqrt,
            );
            c.mean = mean;
            c.sqrt_cov = s;
        }
    }

    /// Collapse the mixture to a single mean and covariance by moment matching (law of total
    /// covariance): `μ = Σ wⱼ μⱼ`, `P = Σ wⱼ (Pⱼ + (μⱼ − μ)(μⱼ − μ)ᵀ)`. The spread-of-means term is
    /// what a single-Gaussian filter throws away — it is the whole reason to carry a mixture.
    ///
    /// Returns the full covariance rather than a factor: a collapse is a moment summary, not a filter
    /// step, and the caller factors it only if the next stage needs a square root.
    #[must_use]
    pub fn collapse(&self) -> ([f64; N], [[f64; N]; N]) {
        let mut mean = [0.0; N];
        for c in &self.components {
            for i in 0..N {
                mean[i] += c.weight * c.mean[i];
            }
        }

        let mut cov = [[0.0; N]; N];
        for c in &self.components {
            for i in 0..N {
                for j in 0..N {
                    // Within-component covariance Pⱼ = Sⱼ Sⱼᵀ.
                    let mut p_ij = 0.0;
                    for k in 0..N {
                        p_ij += c.sqrt_cov[i][k] * c.sqrt_cov[j][k];
                    }
                    let spread = (c.mean[i] - mean[i]) * (c.mean[j] - mean[j]);
                    cov[i][j] += c.weight * (p_ij + spread);
                }
            }
        }
        (mean, cov)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use crate::unscented::cholesky;

    fn component<const N: usize>(
        weight: f64,
        mean: [f64; N],
        p: &[[f64; N]; N],
    ) -> GaussianComponent<N> {
        GaussianComponent {
            weight,
            mean,
            sqrt_cov: cholesky::<N>(p).unwrap(),
        }
    }

    /// Normalisation divides by the total, and declines to invent one for an all-zero mixture.
    ///
    /// Weights 3, 1, 4 sum to 8, so the normalised weights 0.375, 0.125, 0.5 are dyadic and the
    /// division is exact — the comparison can be equality. The mixture's own tests all start from
    /// weights that already sum to one, which is precisely why nothing here was observable: a
    /// no-op, a remainder and a multiplication all leave such a mixture where it was.
    #[test]
    fn normalizing_divides_by_the_total_and_leaves_an_all_zero_mixture_alone() {
        let p = [[1.0, 0.0], [0.0, 1.0]];
        let mut mix = GaussianMixture::<2, 3, 5>::new(
            [
                component(3.0, [1.0, 0.0], &p),
                component(1.0, [0.0, 1.0], &p),
                component(4.0, [-1.0, 2.0], &p),
            ],
            UnscentedParams::gaussian(),
        );
        mix.normalize();
        let got: [f64; 3] = core::array::from_fn(|c| mix.components[c].weight);
        assert_eq!(got, [0.375, 0.125, 0.5], "weights 3,1,4 over a total of 8");
        assert_eq!(
            got.iter().sum::<f64>(),
            1.0,
            "the invariant collapse relies on"
        );

        // The zero mixture has no normalisation. Dividing anyway would put NaN in every weight,
        // which is why this asserts the value and not merely that nothing panicked.
        let mut zeroed = GaussianMixture::<2, 2, 5>::new(
            [
                component(0.0, [1.0, 0.0], &p),
                component(0.0, [0.0, 1.0], &p),
            ],
            UnscentedParams::gaussian(),
        );
        zeroed.normalize();
        for c in 0..2 {
            assert_eq!(zeroed.components[c].weight, 0.0, "component {c}");
        }
    }

    /// Collapsing a mixture recovers the total-covariance moments. Against a hand-computed reference
    /// the mean is the weighted mean and the covariance carries both the within-component spread and
    /// the between-means spread.
    #[test]
    fn collapse_matches_the_total_covariance() {
        let p0 = [[0.4, 0.05], [0.05, 0.3]];
        let p1 = [[0.2, -0.02], [-0.02, 0.5]];
        let mix = GaussianMixture::<2, 2, 5>::new(
            [
                component(0.6, [1.0, -1.0], &p0),
                component(0.4, [-0.5, 2.0], &p1),
            ],
            UnscentedParams::gaussian(),
        );
        let (mean, cov) = mix.collapse();

        // Reference: μ = Σ wⱼ μⱼ; P = Σ wⱼ (Pⱼ + (μⱼ−μ)(μⱼ−μ)ᵀ).
        let means = [[1.0, -1.0], [-0.5, 2.0]];
        let ps = [p0, p1];
        let ws = [0.6, 0.4];
        let mut mean_ref = [0.0; 2];
        for c in 0..2 {
            for i in 0..2 {
                mean_ref[i] += ws[c] * means[c][i];
            }
        }
        let mut cov_ref = [[0.0; 2]; 2];
        for c in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let spread = (means[c][i] - mean_ref[i]) * (means[c][j] - mean_ref[j]);
                    cov_ref[i][j] += ws[c] * (ps[c][i][j] + spread);
                }
            }
        }

        for i in 0..2 {
            assert!((mean[i] - mean_ref[i]).abs() < 1e-12, "mean[{i}]");
            for j in 0..2 {
                assert!((cov[i][j] - cov_ref[i][j]).abs() < 1e-12, "cov[{i}][{j}]");
            }
        }
    }

    /// Propagating every component through a *linear* flow is exact, so the collapsed mixture equals
    /// the linear pushforward of the collapsed prior — `μ ↦ A μ`, `P ↦ A P Aᵀ + Q` — since a linear
    /// map commutes with moment matching. A clean equivalence oracle for the propagation path.
    #[test]
    fn linear_propagation_commutes_with_collapse() {
        let a = [[1.0, 0.1], [-0.05, 0.98]];
        let sq = [[0.05, 0.0], [0.01, 0.04]];
        let p0 = [[0.4, 0.05], [0.05, 0.3]];
        let p1 = [[0.2, -0.02], [-0.02, 0.5]];
        let params = UnscentedParams::new(1.0, 2.0, 0.0);
        let mut mix = GaussianMixture::<2, 2, 5>::new(
            [
                component(0.6, [1.0, -1.0], &p0),
                component(0.4, [-0.5, 2.0], &p1),
            ],
            params,
        );

        let (mean_before, cov_before) = mix.collapse();
        mix.propagate(
            |x| {
                [
                    a[0][0] * x[0] + a[0][1] * x[1],
                    a[1][0] * x[0] + a[1][1] * x[1],
                ]
            },
            &sq,
        );
        let (mean_after, cov_after) = mix.collapse();

        // Reference: A applied to the collapsed prior, plus Q.
        let mut mean_ref = [0.0; 2];
        for i in 0..2 {
            for j in 0..2 {
                mean_ref[i] += a[i][j] * mean_before[j];
            }
        }
        let mut ap = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    ap[i][j] += a[i][k] * cov_before[k][j];
                }
            }
        }
        let mut cov_ref = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    cov_ref[i][j] += ap[i][k] * a[j][k];
                }
                cov_ref[i][j] += sq[i][0] * sq[j][0] + sq[i][1] * sq[j][1];
            }
        }

        for i in 0..2 {
            assert!((mean_after[i] - mean_ref[i]).abs() < 1e-10, "mean[{i}]");
            for j in 0..2 {
                assert!(
                    (cov_after[i][j] - cov_ref[i][j]).abs() < 1e-10,
                    "cov[{i}][{j}]"
                );
            }
        }
    }

    /// The collapse is the mean and covariance of the mixture density (G4).
    ///
    /// Reference: `derivations/gaussian_mixture_moments.mac` integrates `x` and `x xᵀ` against each
    /// component's density, so the moments below are the density's, not a second copy of the
    /// collapse formula; for this fixture they are the exact rationals `(2/5, 1/5)` and
    /// `[[43/50, −529/500], [−529/500, 127/50]]`. A collapse that dropped the spread of the means
    /// would return `[[0.32, 0.022], [0.022, 0.38]]`.
    #[test]
    fn the_collapse_has_the_moments_of_the_mixture_density() {
        let mix = GaussianMixture::<2, 2, 5>::new(
            [
                component(0.6, [1.0, -1.0], &[[0.4, 0.05], [0.05, 0.3]]),
                component(0.4, [-0.5, 2.0], &[[0.2, -0.02], [-0.02, 0.5]]),
            ],
            UnscentedParams::gaussian(),
        );
        let (mean, cov) = mix.collapse();
        let mean_want = [2.0 / 5.0, 1.0 / 5.0];
        let cov_want = [
            [43.0 / 50.0, -529.0 / 500.0],
            [-529.0 / 500.0, 127.0 / 50.0],
        ];
        for i in 0..2 {
            assert!(
                (mean[i] - mean_want[i]).abs() < 1e-15,
                "mean[{i}] = {:.17e}",
                mean[i]
            );
            for j in 0..2 {
                assert!(
                    (cov[i][j] - cov_want[i][j]).abs() < 1e-15,
                    "cov[{i}][{j}] = {:.17e}, derivation says {:.17e}",
                    cov[i][j],
                    cov_want[i][j]
                );
            }
        }
    }
}
