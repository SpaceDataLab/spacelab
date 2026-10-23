//! The ensemble Kalman filter: the covariance is carried by a sample, and never formed.
//!
//! # Why this is not [`crate::SquareRootUnscentedKalman`]
//!
//! A sigma-point filter spends `2N+1` propagations to carry an `N×N` covariance *exactly* through a
//! nonlinear map to second order. That is the right trade when `N` is a state vector. It is the wrong
//! trade when `N` is a *field*: the calibration layer of #135–#140 estimates a density correction over
//! a grid, where `N` runs to thousands, `2N+1` propagations are unaffordable and an `N×N` covariance
//! does not fit in cache, let alone in a `no_std` array.
//!
//! The ensemble filter replaces the covariance with `M` samples, `M ≪ N`. Nothing `N×N` is ever
//! formed: the gain is applied through the ensemble anomalies, so the cost is `M` propagations and
//! `O(M(N + D²))` arithmetic. The price is a *sampling* error that falls as `1/√M` — real, measured
//! in `tests/ensemble_particle.rs` against the exact Kalman solution, and the reason the ensemble
//! size is a modelling decision rather than a tuning knob.
//!
//! # The recursion (Evensen 1994; Burgers, van Leeuwen & Evensen 1998)
//!
//! Forecast every member through the dynamics, each with its own process-noise draw:
//!
//! ```text
//! xᵢ⁻ = f(xᵢ⁺) + S_Q ξᵢ,   ξᵢ ~ N(0, I)
//! ```
//!
//! Then, with `yᵢ = h(xᵢ⁻)` and anomalies about the ensemble means,
//!
//! ```text
//! A = [xᵢ⁻ − x̄⁻],  B = [yᵢ − ȳ]
//! P_xy = A Bᵀ / (M−1),   P_yy = B Bᵀ / (M−1) + R
//! xᵢ⁺ = xᵢ⁻ + P_xy P_yy⁻¹ (z + εᵢ − yᵢ),   εᵢ ~ N(0, R)
//! ```
//!
//! **The perturbation `εᵢ` is not optional.** Updating every member with the *same* innovation
//! shrinks the posterior ensemble spread to `(I − KH) P⁻ (I − KH)ᵀ`, which is smaller than the
//! Kalman posterior by `K R Kᵀ` — the filter would report a covariance it has not earned and
//! diverge with growing confidence. Perturbing the observation per member restores the missing term
//! in expectation, and is what makes the sample covariance an estimator of the right quantity. It is
//! also the one line most re-implementations get wrong.
//!
//! `P_yy` is factored once per update and each member's innovation is back-substituted through that
//! factor, so the `N×D` gain is never assembled either.

// The recursion's symbols (A, B, P, K, R, z, h) are Evensen's, one letter each.
// `M as f64` is the ensemble size, an array length bounded by the stack — exact in f64.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use crate::sampling::Rng;
use crate::unscented::cholesky;

/// Why an ensemble update could not be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleError {
    /// The innovation covariance `P_yy = B Bᵀ/(M−1) + R` admits no Cholesky factor, so the gain is
    /// undefined. With a positive-definite `R` this cannot happen from the sample term, which is
    /// positive-semi-definite by construction — it means `R` itself is singular or indefinite, or
    /// the forecast measurements have gone non-finite.
    InnovationCovarianceNotPositiveDefinite,
}

/// An ensemble of `M` members of an `N`-dimensional state.
///
/// `M ≥ 2` is checked at compile time: the sample covariance divides by `M−1`, and a one-member
/// "ensemble" is a point estimate with a division by zero attached. Rejecting it in the type system
/// rather than at runtime keeps the error enum to failures the caller can actually act on.
#[derive(Debug, Clone)]
pub struct EnsembleKalman<const N: usize, const M: usize> {
    members: [[f64; N]; M],
}

impl<const N: usize, const M: usize> EnsembleKalman<N, M> {
    /// Take an ensemble the caller has drawn — from a previous filter, a Monte-Carlo run, or a
    /// deliberately non-Gaussian prior. Nothing here assumes the sample is Gaussian; only the *update*
    /// does, and only to second moments.
    #[must_use]
    pub const fn from_members(members: [[f64; N]; M]) -> Self {
        const { assert!(M >= 2, "an ensemble needs at least two members") };
        Self { members }
    }

    /// Draw an ensemble from `N(mean, A Aᵀ)`, given any square root `A` of the initial covariance.
    #[must_use]
    pub fn from_mean_and_covariance_sqrt(
        mean: &[f64; N],
        covariance_sqrt: &[[f64; N]; N],
        rng: &mut Rng,
    ) -> Self {
        const { assert!(M >= 2, "an ensemble needs at least two members") };
        let mut members = [[0.0; N]; M];
        for member in &mut members {
            let draw = rng.correlated_normal(covariance_sqrt);
            for i in 0..N {
                member[i] = mean[i] + draw[i];
            }
        }
        Self { members }
    }

    /// The ensemble itself. The state estimate is its mean, but the *ensemble* is the estimate: a
    /// caller propagating a non-Gaussian prior wants the members, not their first two moments.
    #[must_use]
    pub const fn members(&self) -> &[[f64; N]; M] {
        &self.members
    }

    /// The ensemble mean.
    #[must_use]
    pub fn mean(&self) -> [f64; N] {
        let mut mean = [0.0; N];
        for member in &self.members {
            for i in 0..N {
                mean[i] += member[i];
            }
        }
        for value in &mut mean {
            *value /= M as f64;
        }
        mean
    }

    /// The sample covariance, `A Aᵀ/(M−1)`, formed only on request.
    ///
    /// The filter never needs this — it is a diagnostic and a hand-off to a Gaussian consumer. In the
    /// regime the ensemble filter exists for, `N` is large enough that forming it is the expensive
    /// operation in the step.
    #[must_use]
    pub fn covariance(&self) -> [[f64; N]; N] {
        let mean = self.mean();
        let mut cov = [[0.0; N]; N];
        for member in &self.members {
            for i in 0..N {
                let di = member[i] - mean[i];
                for j in 0..N {
                    cov[i][j] += di * (member[j] - mean[j]);
                }
            }
        }
        let scale = 1.0 / (M as f64 - 1.0);
        for row in &mut cov {
            for value in row {
                *value *= scale;
            }
        }
        cov
    }

    /// Forecast: push every member through `f` and add an independent process-noise draw.
    ///
    /// `process_noise_sqrt` is any `S_Q` with `Q = S_Q S_Qᵀ`; pass the zero matrix for a noiseless
    /// dynamic. The draw is *per member* — a shared draw would move the whole ensemble together and
    /// add no spread, which is the failure mode that makes an ensemble filter collapse onto its mean
    /// over a long arc.
    pub fn predict(
        &mut self,
        dynamics: impl Fn(&[f64; N]) -> [f64; N],
        process_noise_sqrt: &[[f64; N]; N],
        rng: &mut Rng,
    ) {
        for member in &mut self.members {
            let propagated = dynamics(member);
            let noise = rng.correlated_normal(process_noise_sqrt);
            for i in 0..N {
                member[i] = propagated[i] + noise[i];
            }
        }
    }

    /// Analysis: correct every member with its own perturbed observation.
    ///
    /// `measurement_noise_sqrt` is any `S_R` with `R = S_R S_Rᵀ`. `h` may be nonlinear; the filter
    /// linearises nothing, it only takes second moments of the forecast measurement sample.
    ///
    /// # Errors
    ///
    /// [`EnsembleError::InnovationCovarianceNotPositiveDefinite`] if `P_yy` admits no Cholesky
    /// factor. The ensemble is left untouched in that case — a half-updated ensemble is worse than
    /// none, because the members would no longer share a common analysis time.
    pub fn update<const D: usize>(
        &mut self,
        observation: &[f64; D],
        measurement: impl Fn(&[f64; N]) -> [f64; D],
        measurement_noise_sqrt: &[[f64; D]; D],
        rng: &mut Rng,
    ) -> Result<(), EnsembleError> {
        let mut forecast = [[0.0; D]; M];
        let mut y_mean = [0.0; D];
        for (slot, member) in forecast.iter_mut().zip(&self.members) {
            *slot = measurement(member);
            for k in 0..D {
                y_mean[k] += slot[k];
            }
        }
        for value in &mut y_mean {
            *value /= M as f64;
        }
        let x_mean = self.mean();

        let scale = 1.0 / (M as f64 - 1.0);
        let mut pxy = [[0.0; D]; N];
        let mut pyy = [[0.0; D]; D];
        for (member, y) in self.members.iter().zip(&forecast) {
            for k in 0..D {
                let dk = y[k] - y_mean[k];
                for i in 0..N {
                    pxy[i][k] += (member[i] - x_mean[i]) * dk;
                }
                for l in 0..D {
                    pyy[k][l] += dk * (y[l] - y_mean[l]);
                }
            }
        }
        for i in 0..N {
            for k in 0..D {
                pxy[i][k] *= scale;
            }
        }
        for k in 0..D {
            for l in 0..D {
                pyy[k][l] *= scale;
                // R = S_R S_Rᵀ, accumulated in place rather than formed separately.
                let mut r = 0.0;
                for m in 0..D {
                    r += measurement_noise_sqrt[k][m] * measurement_noise_sqrt[l][m];
                }
                pyy[k][l] += r;
            }
        }

        let factor =
            cholesky(&pyy).ok_or(EnsembleError::InnovationCovarianceNotPositiveDefinite)?;

        for (member, y) in self.members.iter_mut().zip(&forecast) {
            let perturbation = rng.correlated_normal(measurement_noise_sqrt);
            let mut innovation = [0.0; D];
            for k in 0..D {
                innovation[k] = observation[k] + perturbation[k] - y[k];
            }
            let weighted = solve_spd(&factor, &innovation);
            for i in 0..N {
                let mut delta = 0.0;
                for k in 0..D {
                    delta += pxy[i][k] * weighted[k];
                }
                member[i] += delta;
            }
        }
        Ok(())
    }
}

/// Solve `L Lᵀ u = b` for `u`, given the lower-triangular Cholesky factor `L`.
///
/// Forward substitution then back substitution — never an explicit inverse, which would cost `D³`
/// per update and lose digits the factorisation already paid for.
fn solve_spd<const D: usize>(factor: &[[f64; D]; D], b: &[f64; D]) -> [f64; D] {
    let mut y = [0.0; D];
    for i in 0..D {
        let mut sum = b[i];
        for k in 0..i {
            sum -= factor[i][k] * y[k];
        }
        y[i] = sum / factor[i][i];
    }
    let mut u = [0.0; D];
    for i in (0..D).rev() {
        let mut sum = y[i];
        for k in (i + 1)..D {
            sum -= factor[k][i] * u[k];
        }
        u[i] = sum / factor[i][i];
    }
    u
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::expect_used)]
    use super::*;

    /// `solve_spd` inverts what `cholesky` factors, on a matrix with a known solution.
    #[test]
    fn the_triangular_solve_inverts_the_factorisation() {
        let a = [[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]];
        let l = cholesky(&a).expect("positive definite");
        let want = [1.5, -2.0, 0.25];
        let mut b = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i] += a[i][j] * want[j];
            }
        }
        let got = solve_spd(&l, &b);
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-12, "component {i}: {got:?}");
        }
    }

    /// A singular measurement covariance is reported, and the ensemble is left as it was.
    ///
    /// The zero `R` here is the honest way to reach it: with a one-dimensional measurement whose
    /// forecast sample has zero spread, `P_yy` is exactly zero and no factor exists.
    #[test]
    fn a_singular_innovation_covariance_is_reported_and_changes_nothing() {
        let members = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let mut filter = EnsembleKalman::<2, 4>::from_members(members);
        let mut rng = Rng::new(5);
        // The measurement reads component 0, which is identical across the ensemble.
        let outcome = filter.update(&[1.0], |x| [x[0]], &[[0.0]], &mut rng);
        assert_eq!(
            outcome,
            Err(EnsembleError::InnovationCovarianceNotPositiveDefinite)
        );
        assert_eq!(
            *filter.members(),
            members,
            "a failed update must not mutate"
        );
    }

    /// The analysis gain is the one the sample moments imply — measured, not re-derived.
    ///
    /// Every member is corrected by `K(z + pᵢ − yᵢ)` with the same `K`, so two updates from the same
    /// prior with the same seed and two different observations leave
    /// `Δx̄(z₁) − Δx̄(z₂) = K(z₁ − z₂)`: the perturbation draws replay identically and cancel, and
    /// what survives is the gain alone. That is the only way to see `K` without a Monte-Carlo, and
    /// it is what pins the `M − 1` in the sample covariance — the divisor cancels out of the gain
    /// when `R = 0`, so a test with a noiseless measurement cannot see it at all.
    #[test]
    fn the_analysis_gain_is_the_one_the_sample_moments_imply() {
        // x̄ = (3, 1); the measurement reads component 0, so ȳ = 3.
        let members = [[1.0, 0.5], [2.0, -0.5], [3.0, 1.5], [6.0, 2.5]];
        // Σ(x−x̄)(y−ȳ)/3 = (14/3, 7/3); P_yy = 14/3 + R with R = 0.25, so K = (56/59, 28/59).
        let want = [56.0 / 59.0, 28.0 / 59.0];
        let r_sqrt = [[0.5]];

        let shifted = |observation: f64| {
            let mut filter = EnsembleKalman::<2, 4>::from_members(members);
            let mut rng = Rng::new(2024);
            filter
                .update(&[observation], |x| [x[0]], &r_sqrt, &mut rng)
                .expect("R is positive definite");
            filter.mean()
        };
        let high = shifted(10.0);
        let low = shifted(4.0);

        for i in 0..2 {
            let response = (high[i] - low[i]) / 6.0;
            assert!(
                (response - want[i]).abs() < 1e-12,
                "component {i}: gain {response}, want {}",
                want[i]
            );
        }
    }

    /// The mean and covariance are the sample moments of the members, on a hand-checkable ensemble.
    #[test]
    fn the_moments_are_the_sample_moments() {
        let filter = EnsembleKalman::<1, 4>::from_members([[1.0], [2.0], [3.0], [6.0]]);
        assert_eq!(filter.mean()[0], 3.0);
        // Σ(x−3)² = 4 + 1 + 0 + 9 = 14, over M−1 = 3.
        assert!((filter.covariance()[0][0] - 14.0 / 3.0).abs() < 1e-12);
    }
}
