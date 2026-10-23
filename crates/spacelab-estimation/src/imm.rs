//! The interacting-multiple-model filter: several dynamics hypotheses run in parallel, mixed by a
//! Markov transition matrix, with the posterior model probability read out as the answer.
//!
//! # Why this is not [`crate::GaussianMixture`]
//!
//! A Gaussian mixture is a representation of a non-Gaussian *state*: several components, one truth,
//! one dynamics. A model bank is a statement about the *dynamics*: several mutually exclusive laws of
//! motion, one of which the object is obeying right now, and the filter's job is to say which. The
//! mixture cannot answer that question, because nothing in it distinguishes a component from a
//! hypothesis — its weights are shape, not evidence.
//!
//! For space surveillance the distinction is the product. A catalogued object that stops following the
//! force model it was fitted to has manoeuvred, and the posterior probability of the manoeuvring model
//! *is* the manoeuvre detector. It arrives with a number attached rather than a threshold on a
//! residual, and it degrades gracefully: a weak, sustained low-thrust burn moves the probability slowly
//! where a residual test would stay under its threshold until the orbit had already changed.
//!
//! # The recursion (Bar-Shalom, *Estimation with Applications to Tracking and Navigation*, §11.6)
//!
//! With `M` models, model probabilities `μᵢ` and a Markov transition matrix `πᵢⱼ = P(model j now |
//! model i last epoch)`:
//!
//! ```text
//! c̄ⱼ    = Σᵢ πᵢⱼ μᵢ                                   predicted model probability
//! μᵢ|ⱼ  = πᵢⱼ μᵢ / c̄ⱼ                                 mixing weights
//! x⁰ⱼ   = Σᵢ μᵢ|ⱼ x̂ᵢ                                   mixed prior for model j
//! P⁰ⱼ   = Σᵢ μᵢ|ⱼ [Pᵢ + (x̂ᵢ − x⁰ⱼ)(x̂ᵢ − x⁰ⱼ)ᵀ]
//! Λⱼ    = 𝒩(νⱼ; 0, Sⱼ)                                model j's evidence for this observation
//! μⱼ    = c̄ⱼ Λⱼ / Σₖ c̄ₖ Λₖ
//! ```
//!
//! The mixing step is what separates IMM from running `M` independent filters and comparing their
//! likelihoods (which is MMAE, and which cannot track a *switch*): before each epoch every model is
//! restarted from a weighted blend of all the others, so a model that has been unlikely for an hour
//! still holds a usable state when the object manoeuvres into it.
//!
//! Each model is filtered by a [`SquareRootUnscentedKalman`], so the bank inherits the square-root
//! covariance and the sigma-point handling of nonlinear dynamics — and the mixed `P⁰ⱼ`, which is a sum
//! of outer products and therefore *not* available in factored form, is re-factored once per model per
//! epoch. That Cholesky is the only place the covariance is formed in full, and a failure there is
//! reported rather than absorbed.
//!
//! # Two numerical decisions, both load-bearing
//!
//! * **Likelihoods are combined in the log domain.** A model that a manoeuvre has invalidated produces
//!   normalised squares in the hundreds, whose density underflows to exactly zero. Subtracting the
//!   largest log-likelihood before exponentiating keeps the *ratios* — which is all the recursion
//!   needs — at full precision.
//! * **The likelihood ratio is floored at [`f64::MIN_POSITIVE`].** A model whose probability reaches
//!   exactly zero can never come back: `c̄ⱼ` is a positive combination of the previous probabilities, so
//!   zero propagates. Since the object may manoeuvre back into the model that was ruled out an hour
//!   ago, that is a bug and not a saving. The floor is `2.2e-308`, so a model held down by it needs a
//!   log-likelihood deficit of ~708 nats before the floor binds at all — a depth no measurement noise
//!   reaches by accident, and one a real manoeuvre passes in a single epoch.

// Matrix code indexes several arrays by one i/j/k, and the bank's symbols (μ, π, c̄, P, S) follow
// Bar-Shalom's notation.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::srukf::SquareRootUnscentedKalman;
use crate::unscented::UnscentedParams;

/// Why a model bank could not be built or stepped. Each variant names the model or row responsible,
/// because with `M` hypotheses running "the filter failed" is not a diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmError {
    /// A row of the transition matrix is not a probability distribution: it has a negative or
    /// non-finite entry, or it does not sum to one. Row `i` is `P(· | model i)`, so it must.
    TransitionRowNotStochastic {
        /// Index of the offending row.
        row: usize,
    },
    /// A column of the transition matrix is all zeros: no model, including itself, can transition into
    /// model `column`. Such a model is unreachable, its predicted probability `c̄` is identically zero,
    /// and its mixing weights are `0/0` — it does not belong in the bank.
    UnreachableModel {
        /// Index of the unreachable model.
        column: usize,
    },
    /// The initial model probabilities are negative, non-finite, or do not sum to one.
    ProbabilitiesNotNormalized,
    /// A model's initial covariance is not positive-definite, so no filter can start from it.
    SingularPriorCovariance {
        /// Index of the model whose prior failed to factor.
        model: usize,
    },
    /// The mixed prior `P⁰ⱼ` failed to factor. It is positive-semidefinite by construction — a sum of
    /// covariances and outer products with non-negative weights — so this is a *singular* blend rather
    /// than an invalid one: two models agreeing exactly with covariances that share a null direction.
    SingularMixedCovariance {
        /// Index of the model whose mixed prior failed to factor.
        model: usize,
    },
}

impl core::fmt::Display for ImmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TransitionRowNotStochastic { row } => {
                write!(f, "transition matrix row {row} is not a distribution")
            }
            Self::UnreachableModel { column } => {
                write!(f, "no transition enters model {column}")
            }
            Self::ProbabilitiesNotNormalized => {
                write!(f, "model probabilities do not sum to one")
            }
            Self::SingularPriorCovariance { model } => {
                write!(
                    f,
                    "prior covariance of model {model} is not positive-definite"
                )
            }
            Self::SingularMixedCovariance { model } => {
                write!(f, "mixed prior covariance of model {model} is singular")
            }
        }
    }
}

impl core::error::Error for ImmError {}

/// A bank of `M` dynamics hypotheses over an `N`-dimensional state, each filtered by a square-root
/// unscented filter with `L = 2N + 1` sigma points.
///
/// The models differ only in what the caller passes to [`step`](Self::step): the dynamics closure is
/// given the model index, and the process-noise factor is per model. A ballistic / low-thrust /
/// impulsive bank is three dynamics and three `Q`s, not three types.
#[derive(Debug, Clone)]
pub struct InteractingMultipleModel<const N: usize, const M: usize, const L: usize> {
    estimate: [[f64; N]; M],
    sqrt_cov: [[[f64; N]; N]; M],
    probability: [f64; M],
    transition: [[f64; M]; M],
    params: UnscentedParams,
}

impl<const N: usize, const M: usize, const L: usize> InteractingMultipleModel<N, M, L> {
    /// Seed every model with the same estimate and covariance, the prior model probabilities, and the
    /// Markov transition matrix.
    ///
    /// `transition[i][j]` is `P(model j at this epoch | model i at the last)`. Its diagonal encodes how
    /// long a mode is expected to persist: for a mean sojourn time `τ` at step `Δt`, `πᵢᵢ ≈ 1 − Δt/τ`.
    /// A manoeuvre that lasts one epoch and a manoeuvre that lasts an hour are different filters, and
    /// this matrix is where the difference is stated.
    ///
    /// # Errors
    ///
    /// [`ImmError::TransitionRowNotStochastic`], [`ImmError::UnreachableModel`],
    /// [`ImmError::ProbabilitiesNotNormalized`] or [`ImmError::SingularPriorCovariance`].
    ///
    /// # Panics
    ///
    /// If `L != 2N + 1` — the sigma-point storage of the underlying filters would be mis-sized.
    pub fn new(
        estimate: [f64; N],
        covariance: &[[f64; N]; N],
        probability: [f64; M],
        transition: [[f64; M]; M],
        params: UnscentedParams,
    ) -> Result<Self, ImmError> {
        assert!(L == 2 * N + 1, "sigma-point count L must equal 2N + 1");
        validate_distribution(&probability).map_err(|()| ImmError::ProbabilitiesNotNormalized)?;
        for i in 0..M {
            validate_distribution(&transition[i])
                .map_err(|()| ImmError::TransitionRowNotStochastic { row: i })?;
        }
        for j in 0..M {
            if (0..M).all(|i| transition[i][j] == 0.0) {
                return Err(ImmError::UnreachableModel { column: j });
            }
        }
        let seed = SquareRootUnscentedKalman::<N, L>::from_covariance(estimate, covariance, params)
            .ok_or(ImmError::SingularPriorCovariance { model: 0 })?;
        Ok(Self {
            estimate: [estimate; M],
            sqrt_cov: [seed.covariance_sqrt(); M],
            probability,
            transition,
            params,
        })
    }

    /// One epoch: mix, filter every model through its own dynamics, reweigh by evidence, and return the
    /// posterior model probabilities.
    ///
    /// `dynamics(model, state)` is the one-step flow under hypothesis `model`; `process_noise_sqrt`
    /// carries each model's `S_Q` with `Q = S_Q S_Qᵀ`. The measurement map and its noise are shared —
    /// the bank disagrees about the motion, not about the sensor.
    ///
    /// # Errors
    ///
    /// [`ImmError::SingularMixedCovariance`] if a model's mixed prior cannot be factored.
    pub fn step<const D: usize>(
        &mut self,
        dynamics: impl Fn(usize, &[f64; N]) -> [f64; N],
        process_noise_sqrt: &[[[f64; N]; N]; M],
        measurement: impl Fn(&[f64; N]) -> [f64; D],
        z: &[f64; D],
        noise_sqrt: &[[f64; D]; D],
    ) -> Result<[f64; M], ImmError> {
        let predicted = self.predicted_probabilities();
        let mut log_likelihood = [0.0; M];

        for j in 0..M {
            let (mixed_estimate, mixed_covariance) = self.mix_into(j, &predicted);
            let mut filter = SquareRootUnscentedKalman::<N, L>::from_covariance(
                mixed_estimate,
                &mixed_covariance,
                self.params,
            )
            .ok_or(ImmError::SingularMixedCovariance { model: j })?;

            filter.predict(|x| dynamics(j, x), &process_noise_sqrt[j]);
            log_likelihood[j] = filter.update_with_log_likelihood(&measurement, z, noise_sqrt);
            self.estimate[j] = filter.x;
            self.sqrt_cov[j] = filter.covariance_sqrt();
        }

        self.reweigh(&predicted, &log_likelihood);
        Ok(self.probability)
    }

    /// The current model probabilities. For a manoeuvre detector this is the output: sum the entries of
    /// the manoeuvring models and compare against a decision threshold.
    #[must_use]
    pub const fn probabilities(&self) -> [f64; M] {
        self.probability
    }

    /// The index of the most probable model, lowest index winning a tie.
    #[must_use]
    pub fn most_likely(&self) -> usize {
        let mut best = 0;
        for j in 1..M {
            if self.probability[j] > self.probability[best] {
                best = j;
            }
        }
        best
    }

    /// One model's own estimate and covariance square root, before combination.
    #[must_use]
    pub const fn model(&self, index: usize) -> ([f64; N], [[f64; N]; N]) {
        (self.estimate[index], self.sqrt_cov[index])
    }

    /// The bank's combined estimate: the probability-weighted mean and the moment-matched covariance
    /// `Σ μⱼ [Pⱼ + (x̂ⱼ − x̂)(x̂ⱼ − x̂)ᵀ]`.
    ///
    /// The spread-of-means term is the honest part. While the bank is undecided, the models disagree
    /// about where the object is, and that disagreement belongs in the covariance — a combined estimate
    /// that reported only `Σ μⱼ Pⱼ` would be confident precisely when the filter is not.
    #[must_use]
    pub fn combined(&self) -> ([f64; N], [[f64; N]; N]) {
        let mut mean = [0.0; N];
        for j in 0..M {
            for i in 0..N {
                mean[i] += self.probability[j] * self.estimate[j][i];
            }
        }
        let mut covariance = [[0.0; N]; N];
        for j in 0..M {
            for a in 0..N {
                for b in 0..N {
                    let mut p_ab = 0.0;
                    for k in 0..N {
                        p_ab += self.sqrt_cov[j][a][k] * self.sqrt_cov[j][b][k];
                    }
                    let spread = (self.estimate[j][a] - mean[a]) * (self.estimate[j][b] - mean[b]);
                    covariance[a][b] += self.probability[j] * (p_ab + spread);
                }
            }
        }
        (mean, covariance)
    }

    /// `c̄ⱼ = Σᵢ πᵢⱼ μᵢ`, the model probabilities the transition matrix alone predicts.
    fn predicted_probabilities(&self) -> [f64; M] {
        let mut predicted = [0.0; M];
        for j in 0..M {
            for i in 0..M {
                predicted[j] += self.transition[i][j] * self.probability[i];
            }
        }
        predicted
    }

    /// Model `j`'s mixed prior: the blend of every model's posterior weighted by `μᵢ|ⱼ`.
    fn mix_into(&self, j: usize, predicted: &[f64; M]) -> ([f64; N], [[f64; N]; N]) {
        let mut weight = [0.0; M];
        for i in 0..M {
            weight[i] = self.transition[i][j] * self.probability[i] / predicted[j];
        }

        let mut mean = [0.0; N];
        for i in 0..M {
            for a in 0..N {
                mean[a] += weight[i] * self.estimate[i][a];
            }
        }

        let mut covariance = [[0.0; N]; N];
        for i in 0..M {
            for a in 0..N {
                for b in 0..N {
                    let mut p_ab = 0.0;
                    for k in 0..N {
                        p_ab += self.sqrt_cov[i][a][k] * self.sqrt_cov[i][b][k];
                    }
                    let spread = (self.estimate[i][a] - mean[a]) * (self.estimate[i][b] - mean[b]);
                    covariance[a][b] += weight[i] * (p_ab + spread);
                }
            }
        }
        (mean, covariance)
    }

    /// `μⱼ ∝ c̄ⱼ Λⱼ`, computed from the log-likelihoods with the largest subtracted out.
    fn reweigh(&mut self, predicted: &[f64; M], log_likelihood: &[f64; M]) {
        let mut best = log_likelihood[0];
        for j in 1..M {
            if log_likelihood[j] > best {
                best = log_likelihood[j];
            }
        }

        let mut unnormalized = [0.0; M];
        let mut total = 0.0;
        for j in 0..M {
            // The floor keeps a ruled-out model revivable; see the module documentation.
            let ratio = libm::exp(log_likelihood[j] - best).max(f64::MIN_POSITIVE);
            unnormalized[j] = predicted[j] * ratio;
            total += unnormalized[j];
        }
        for j in 0..M {
            self.probability[j] = unnormalized[j] / total;
        }
    }
}

/// A vector is a probability distribution: every entry finite and non-negative, and the sum one to
/// within the rounding a sum of `M` terms can carry.
fn validate_distribution<const M: usize>(p: &[f64; M]) -> Result<(), ()> {
    let mut total = 0.0;
    for value in p {
        if !value.is_finite() || *value < 0.0 {
            return Err(());
        }
        total += value;
    }
    #[allow(clippy::cast_precision_loss)] // M is a model count, never above a handful
    let tolerance = 8.0 * f64::EPSILON * M as f64;
    if libm::fabs(total - 1.0) > tolerance {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    /// A constant-velocity pair and a position measurement, as in the linear filters' own tests.
    fn dynamics(_model: usize, x: &[f64; 2]) -> [f64; 2] {
        [x[0] + x[1], x[1]]
    }
    fn measure(x: &[f64; 2]) -> [f64; 1] {
        [x[0]]
    }
    const Q: [[f64; 2]; 2] = [[0.02, 0.0], [0.0, 0.05]];
    const P0: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 0.5]];
    const R: [[f64; 1]; 1] = [[0.3]];

    fn observations(count: usize) -> Vec<f64> {
        let (mut truth, mut seed) = (0.0_f64, 12_345_u64);
        (0..count)
            .map(|_| {
                truth += 1.0;
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                #[allow(clippy::cast_precision_loss)]
                let jitter = ((seed >> 33) as f64 / f64::from(1_u32 << 31) - 0.5) * 0.6;
                truth + jitter
            })
            .collect()
    }

    /// Oracle — a bank of one model is the filter it wraps. Every piece of the IMM machinery still
    /// runs (mixing with itself, the likelihood, the reweighting), and none of it may move the answer.
    #[test]
    fn a_bank_of_one_model_is_the_filter_it_wraps() {
        let params = UnscentedParams::gaussian();
        let mut bank =
            InteractingMultipleModel::<2, 1, 5>::new([0.0, 0.0], &P0, [1.0], [[1.0]], params)
                .unwrap();
        let mut filter =
            SquareRootUnscentedKalman::<2, 5>::from_covariance([0.0, 0.0], &P0, params).unwrap();

        let mut worst = 0.0_f64;
        for z in observations(15) {
            let probabilities = bank.step(dynamics, &[Q], measure, &[z], &R).unwrap();
            assert_eq!(
                probabilities,
                [1.0],
                "a single model owns all the probability"
            );
            filter.predict(|x| dynamics(0, x), &Q);
            filter.update(measure, &[z], &R);
            let (mean, _) = bank.combined();
            for i in 0..2 {
                worst = worst.max((mean[i] - filter.x[i]).abs());
            }
        }
        println!("IMM(M=1) vs SR-UKF: worst |dx| {worst:.3e}");
        assert!(
            worst < 1e-12,
            "the bank departed from the filter: {worst:.3e}"
        );
    }

    /// The evidence has to move the probabilities the right way: two models that differ only in how
    /// much process noise they admit, on data that the quiet one explains.
    #[test]
    fn the_model_that_explains_the_data_takes_the_probability() {
        let params = UnscentedParams::gaussian();
        let quiet = Q;
        let loud = [[2.0, 0.0], [0.0, 2.0]];
        let mut bank = InteractingMultipleModel::<2, 2, 5>::new(
            [0.0, 0.0],
            &P0,
            [0.5, 0.5],
            [[0.95, 0.05], [0.05, 0.95]],
            params,
        )
        .unwrap();

        let mut probabilities = [0.5, 0.5];
        for z in observations(30) {
            probabilities = bank
                .step(dynamics, &[quiet, loud], measure, &[z], &R)
                .unwrap();
            let total: f64 = probabilities.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "probabilities must stay a distribution"
            );
        }
        println!("quiet/loud posterior: {probabilities:?}");
        assert!(
            probabilities[0] > 0.5,
            "the model matching the data must win: {probabilities:?}"
        );
        assert_eq!(bank.most_likely(), 0);
    }

    /// The floor on the likelihood ratio, stated as behaviour.
    ///
    /// One observation a model cannot explain at all — here a 1e6 jump against a metre-level noise,
    /// which is `δ² ≈ 3e12` and an exponent of `−1.7e12` — sends its likelihood ratio through the
    /// bottom of `f64`. Without the floor its probability becomes exactly zero, and `c̄ⱼ` is a positive
    /// combination of probabilities, so zero is absorbing: the model is dead for the rest of the arc.
    /// Measured: the shock puts it at the floor, `2.2e-308`, and four quiet epochs bring it back
    /// through 0.5.
    #[test]
    fn an_impossible_observation_does_not_zero_a_model_forever() {
        let params = UnscentedParams::gaussian();
        let loud = [[1e6, 0.0], [0.0, 1e6]];
        let mut bank = InteractingMultipleModel::<2, 2, 5>::new(
            [0.0, 1.0],
            &P0,
            [0.5, 0.5],
            [[0.99, 0.01], [0.01, 0.99]],
            params,
        )
        .unwrap();

        // The shock: only the model that admits enormous process noise can account for it.
        bank.step(dynamics, &[Q, loud], measure, &[1e6], &R)
            .unwrap();
        let after_shock = bank.probabilities();
        println!("after the shock: {after_shock:?}");
        assert!(
            after_shock[0] > 0.0 && after_shock[0].is_finite(),
            "the quiet model must survive the shock as a positive probability: {after_shock:?}"
        );
        assert!(
            after_shock[1] > 0.99,
            "the loud model must take the shock: {after_shock:?}"
        );

        // Quiet data again: the model that was ruled out has to be able to win it back.
        let mut epochs = 0;
        let mut truth = 1e6;
        while bank.probabilities()[0] <= 0.5 {
            truth += 1.0;
            bank.step(dynamics, &[Q, loud], measure, &[truth], &R)
                .unwrap();
            epochs += 1;
            assert!(
                epochs < 50,
                "the quiet model never recovered: {:?}",
                bank.probabilities()
            );
        }
        println!("quiet model recovered after {epochs} epochs");
    }

    /// While the bank is undecided the models disagree about where the object is, and the combined
    /// covariance has to carry that disagreement — not just the average of the within-model ones.
    ///
    /// Keeping the bank undecided is the whole fixture: the two models are pushed 20 apart per epoch,
    /// and the observation sits between them under a noise wide enough (`σ² = 100`) that neither is
    /// refuted. Measured after three epochs: the models sit 44 apart, the within-model variance is
    /// 18.2 and the combined one 493.8 — 27 times larger. A combined covariance that dropped the
    /// spread would be claiming a precision the bank does not have.
    #[test]
    fn the_combined_covariance_carries_the_spread_of_the_means() {
        let params = UnscentedParams::gaussian();
        let ambiguous = [[10.0; 1]; 1];
        let split = |model: usize, x: &[f64; 2]| {
            let push = if model == 0 { -10.0 } else { 10.0 };
            [x[0] + x[1] + push, x[1]]
        };
        let mut bank = InteractingMultipleModel::<2, 2, 5>::new(
            [0.0, 1.0],
            &P0,
            [0.5, 0.5],
            [[0.99, 0.01], [0.01, 0.99]],
            params,
        )
        .unwrap();
        // Each observation sits exactly between the two hypotheses, so neither gains on the other.
        for z in [1.0, 2.0, 3.0] {
            bank.step(split, &[Q, Q], measure, &[z], &ambiguous)
                .unwrap();
        }

        let probabilities = bank.probabilities();
        let (_, combined) = bank.combined();
        let (mean0, s0) = bank.model(0);
        let (mean1, _) = bank.model(1);
        let within: f64 = (0..2).map(|k| s0[0][k] * s0[0][k]).sum();
        let separation = (mean0[0] - mean1[0]).abs();
        println!(
            "probabilities {probabilities:?}, combined P00 {:.4}, within-model P00 {within:.4}, separation {separation:.4}",
            combined[0][0]
        );
        assert!(
            (0.3..0.7).contains(&probabilities[0]),
            "the bank must stay undecided for this to measure anything: {probabilities:?}"
        );
        assert!(
            separation > 10.0,
            "the fixture must actually split the models"
        );
        assert!(
            combined[0][0] > 10.0 * within,
            "the spread of means is missing from the combined covariance: {:.4} vs {within:.4}",
            combined[0][0]
        );
    }

    /// Every refusal is checked, and each names what it refused.
    #[test]
    fn a_bank_that_cannot_be_a_bank_is_refused() {
        let params = UnscentedParams::gaussian();
        let build = |probability: [f64; 2], transition: [[f64; 2]; 2]| {
            InteractingMultipleModel::<2, 2, 5>::new(
                [0.0, 0.0],
                &P0,
                probability,
                transition,
                params,
            )
        };
        let good = [[0.9, 0.1], [0.1, 0.9]];

        assert_eq!(
            build([0.5, 0.6], good).err(),
            Some(ImmError::ProbabilitiesNotNormalized)
        );
        assert_eq!(
            build([0.5, 0.5], [[0.9, 0.1], [0.2, 0.9]]).err(),
            Some(ImmError::TransitionRowNotStochastic { row: 1 })
        );
        assert_eq!(
            build([0.5, 0.5], [[0.9, 0.1], [f64::NAN, 0.1]]).err(),
            Some(ImmError::TransitionRowNotStochastic { row: 1 })
        );
        assert_eq!(
            build([0.5, 0.5], [[1.0, 0.0], [1.0, 0.0]]).err(),
            Some(ImmError::UnreachableModel { column: 1 })
        );
        assert_eq!(
            InteractingMultipleModel::<2, 2, 5>::new(
                [0.0, 0.0],
                &[[1.0, 0.0], [0.0, 0.0]],
                [0.5, 0.5],
                good,
                params
            )
            .err(),
            Some(ImmError::SingularPriorCovariance { model: 0 })
        );
    }

    /// Each refusal says which row, column or model it refused. With `M` hypotheses the index is the
    /// diagnosis — a bank that reports "not a distribution" and nothing else leaves the caller to find
    /// the offending row themselves.
    #[test]
    fn every_refusal_names_the_row_column_or_model() {
        let messages = [
            (
                ImmError::TransitionRowNotStochastic { row: 2 },
                "transition matrix row 2 is not a distribution",
            ),
            (
                ImmError::UnreachableModel { column: 1 },
                "no transition enters model 1",
            ),
            (
                ImmError::ProbabilitiesNotNormalized,
                "model probabilities do not sum to one",
            ),
            (
                ImmError::SingularPriorCovariance { model: 3 },
                "prior covariance of model 3 is not positive-definite",
            ),
            (
                ImmError::SingularMixedCovariance { model: 0 },
                "mixed prior covariance of model 0 is singular",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(format!("{error}"), expected);
            let dynamic: &dyn core::error::Error = &error;
            assert_eq!(format!("{dynamic}"), expected);
        }
    }

    /// The argmax, and what it does with a tie. A detector reads [`InteractingMultipleModel::most_likely`]
    /// to name the mode, so "which index wins when two models are equally probable" is behaviour, not an
    /// implementation detail: the lowest index wins, which is the ballistic hypothesis in a bank ordered
    /// the way [`crate::imm`]'s own tests order it.
    #[test]
    fn the_most_likely_model_is_the_argmax_and_a_tie_goes_to_the_lower_index() {
        let params = UnscentedParams::gaussian();
        let mut bank = InteractingMultipleModel::<2, 3, 5>::new(
            [0.0, 0.0],
            &P0,
            [0.2, 0.5, 0.3],
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            params,
        )
        .unwrap();
        assert_eq!(bank.most_likely(), 1);

        bank.probability = [0.3, 0.2, 0.5];
        assert_eq!(bank.most_likely(), 2);

        bank.probability = [0.5, 0.5, 0.0];
        assert_eq!(bank.most_likely(), 0, "a tie must go to the lower index");
    }

    /// The two moment-matching sums, against hand-computed matrices.
    ///
    /// [`InteractingMultipleModel::combined`] and [`InteractingMultipleModel::mix_into`] are the same
    /// formula under different weights — `Σ wⱼ [Pⱼ + (x̂ⱼ − x̄)(x̂ⱼ − x̄)ᵀ]` — and the end-to-end tests
    /// only ever see them through a filtered estimate, where an error in the spread term is a change of
    /// covariance rather than a wrong answer. Here the state is set directly and both are compared
    /// against the arithmetic done by hand, which every entry is exact in binary:
    ///
    /// `μ = [¼, ¾]`, `x̂ = ([1,2], [5,6])`, `P = ([[4,1],[1,1.25]], [[1,0],[0,9]])` gives the mean
    /// `[4,5]`, spreads `[-3,-3]` and `[1,1]`, and the combination `[[4.75, 3.25], [3.25, 10.0625]]`.
    /// The transition matrix is chosen so that model 0's mixing weights are the model probabilities
    /// themselves, which makes the mixed prior the same matrix — the mixing *is* a combination with
    /// `μᵢ|ⱼ` in place of `μᵢ`.
    #[test]
    fn the_combination_and_the_mixing_are_the_same_moment_match() {
        let params = UnscentedParams::gaussian();
        let mut bank = InteractingMultipleModel::<2, 2, 5>::new(
            [0.0, 0.0],
            &P0,
            [0.5, 0.5],
            [[0.5, 0.5], [0.5, 0.5]],
            params,
        )
        .unwrap();
        bank.probability = [0.25, 0.75];
        bank.estimate = [[1.0, 2.0], [5.0, 6.0]];
        bank.sqrt_cov = [[[2.0, 0.0], [0.5, 1.0]], [[1.0, 0.0], [0.0, 3.0]]];

        let expected_mean = [4.0, 5.0];
        let expected_covariance = [[4.75, 3.25], [3.25, 10.0625]];

        let (mean, covariance) = bank.combined();
        assert_eq!(mean, expected_mean);
        assert_eq!(covariance, expected_covariance);

        let predicted = bank.predicted_probabilities();
        assert_eq!(predicted, [0.5, 0.5]);
        let (mixed_mean, mixed_covariance) = bank.mix_into(0, &predicted);
        assert_eq!(mixed_mean, expected_mean);
        assert_eq!(mixed_covariance, expected_covariance);
    }

    /// The tolerance on "sums to one" is `8 ε M`, and both halves of that are load-bearing.
    ///
    /// It scales with `M` because the caller's own sum of `M` terms carries that much rounding before
    /// it ever reaches this crate, and refusing a distribution the caller computed correctly would be a
    /// false refusal. The comparison is strict, so a vector landing exactly on the bound is accepted —
    /// `[½, ½ + 2⁻⁴⁸]` sums to exactly `1 + 2⁻⁴⁸`, and `8 ε M = 2⁻⁴⁸` for `M = 2`.
    #[test]
    fn the_tolerance_on_a_distribution_scales_with_its_length_and_includes_its_bound() {
        assert!(validate_distribution(&[0.5, 0.5 + 16.0 * f64::EPSILON]).is_ok());
        // 2e-15 is inside 8 ε M = 5.3e-15 for M = 3, and outside it for any M that divides instead.
        assert!(validate_distribution(&[0.5, 0.25, 0.25 + 2e-15]).is_ok());
        assert!(validate_distribution(&[0.5, 0.25, 0.25 + 1e-13]).is_err());
    }
}
