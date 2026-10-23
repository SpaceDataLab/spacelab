//! The bootstrap particle filter: a weighted sample, with no assumption that the posterior is
//! Gaussian in any coordinate system.
//!
//! # Why this is not [`crate::GaussianMixture`] or a sigma-point filter
//!
//! Every other estimator in this crate answers with a mean and a covariance, which is a complete
//! answer only when the posterior is Gaussian. Two problems in this workspace are not:
//!
//! * **Long-propagated debris uncertainty.** An along-track error grows into a banana that no
//!   ellipsoid contains — the reason the state-transition-tensor work of #90 exists. The tensors fix
//!   the *propagation*; they do not make the posterior Gaussian.
//! * **Angles-only tracking.** A bearings-only measurement of a single object is genuinely ambiguous
//!   in range, and the posterior can be bimodal. An extended Kalman filter does not report the
//!   ambiguity: it picks whichever mode its linearisation happened to sit on and reports a tight
//!   covariance about it. That failure is *confident*, which makes it worse than a wide one.
//!
//! A Gaussian mixture can represent several modes, but its components are chosen in advance and its
//! update is still Gaussian per component. The particle filter's representation is the sample
//! itself, so a mode it did not anticipate costs nothing to carry.
//!
//! # The recursion (Gordon, Salmond & Smith 1993)
//!
//! ```text
//! xᵢ⁻  = f(xᵢ⁺) + S_Q ξᵢ                     propagate, each particle with its own noise draw
//! wᵢ  ∝ wᵢ · p(z | xᵢ⁻)                      re-weight by the measurement likelihood
//! ```
//!
//! with the weights renormalised to sum to one. The proposal is the prior — that is what "bootstrap"
//! means, and it is why the likelihood is the whole update.
//!
//! # Two things that decide whether it works
//!
//! * **Weights are accumulated in the log domain.** A particle three sigma from a six-dimensional
//!   observation has a likelihood around `1e-8`; twenty of those in a row underflow to exactly zero,
//!   and a zero weight can never recover. [`ParticleFilter::update`] takes a *log* likelihood,
//!   subtracts the maximum before exponentiating, and so keeps the ratios — which is all the
//!   recursion needs — at full precision.
//! * **Resampling is triggered by the effective sample size, not by the clock.** Resampling every
//!   step throws away information (it is a random operation that only adds variance) and accelerates
//!   sample impoverishment; never resampling lets one particle take all the weight. The standard
//!   trigger is `N_eff = 1/Σwᵢ² < αM`, with `α` around a half.
//!
//! # Choosing a resampling scheme
//!
//! All four schemes in [`Resampling`] are unbiased — `E[nᵢ] = M wᵢ` — and they differ in variance.
//! [`Resampling::Systematic`] is the default in the tracking literature because it is `O(M)` with one
//! random draw and the lowest variance of the four; the comparison is measured, not asserted, in
//! `tests/ensemble_particle.rs`.

// The recursion's symbols (w, x, z, M) are Gordon–Salmond–Smith's.
// `M as f64` is the particle count, an array length bounded by the stack — exact in f64.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use crate::sampling::Rng;

/// Why a particle update could not be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleError {
    /// Every particle's posterior weight is zero or non-finite: the observation is inconsistent with
    /// the entire sample. Renormalising is impossible, and continuing would either divide by zero or
    /// silently restore a uniform weighting that no measurement supports.
    ///
    /// This is a *diagnosis*, not a nuisance: it means the prior has lost the truth — through an
    /// unmodelled manoeuvre, a wrong association, or too few particles for the state dimension — and
    /// the caller must re-initialise rather than continue.
    WeightCollapse,
}

/// How to draw the resampled indices.
///
/// Every scheme returns `M` indices into the current particle set, with particle `i` appearing
/// `nᵢ` times where `E[nᵢ] = M wᵢ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    /// `M` independent uniforms located in the cumulative weights. The reference scheme, and the
    /// highest-variance one: nothing stops it from drawing the same particle `M` times.
    Multinomial,
    /// Deterministic `⌊M wᵢ⌋` copies, with the remainder drawn multinomially from the fractional
    /// parts. Removes the variance of the integer part.
    Residual,
    /// One uniform per stratum `[i/M, (i+1)/M)`. A particle with weight `≥ 1/M` is guaranteed to
    /// survive, which multinomial resampling cannot promise.
    Stratified,
    /// One uniform for the whole set, then a regular comb of spacing `1/M`. Lowest variance of the
    /// four, and `O(M)` with a single random draw.
    Systematic,
}

impl Resampling {
    /// The resampled indices under this scheme.
    #[must_use]
    pub fn indices<const M: usize>(self, weights: &[f64; M], rng: &mut Rng) -> [usize; M] {
        match self {
            Self::Multinomial => multinomial(weights, rng),
            Self::Residual => residual(weights, rng),
            Self::Stratified => comb(weights, &stratified_points(rng)),
            Self::Systematic => comb(weights, &systematic_points(rng)),
        }
    }
}

/// `M` particles of an `N`-dimensional state, with their weights.
#[derive(Debug, Clone)]
pub struct ParticleFilter<const N: usize, const M: usize> {
    particles: [[f64; N]; M],
    weights: [f64; M],
}

impl<const N: usize, const M: usize> ParticleFilter<N, M> {
    /// Take a particle set the caller has drawn, with uniform weights.
    ///
    /// This is the constructor for a non-Gaussian prior — a range ambiguity sampled over two
    /// intervals, an admissible region, a previous filter's posterior — which is the case the filter
    /// exists for and which no mean-and-covariance constructor can express.
    #[must_use]
    pub const fn from_particles(particles: [[f64; N]; M]) -> Self {
        const { assert!(M >= 2, "a particle filter needs at least two particles") };
        Self {
            particles,
            weights: [1.0 / M as f64; M],
        }
    }

    /// Draw a particle set from `N(mean, A Aᵀ)`, given any square root `A` of the covariance.
    #[must_use]
    pub fn from_mean_and_covariance_sqrt(
        mean: &[f64; N],
        covariance_sqrt: &[[f64; N]; N],
        rng: &mut Rng,
    ) -> Self {
        const { assert!(M >= 2, "a particle filter needs at least two particles") };
        let mut particles = [[0.0; N]; M];
        for particle in &mut particles {
            let draw = rng.correlated_normal(covariance_sqrt);
            for i in 0..N {
                particle[i] = mean[i] + draw[i];
            }
        }
        Self {
            particles,
            weights: [1.0 / M as f64; M],
        }
    }

    /// The particles.
    #[must_use]
    pub const fn particles(&self) -> &[[f64; N]; M] {
        &self.particles
    }

    /// The normalised weights.
    #[must_use]
    pub const fn weights(&self) -> &[f64; M] {
        &self.weights
    }

    /// Propagate every particle through `f`, each with its own process-noise draw.
    ///
    /// The per-particle draw is what keeps the sample diverse. With a shared draw the particles move
    /// rigidly, the filter stops exploring, and resampling then collapses it onto one point — the
    /// classical sample impoverishment, arrived at in one step instead of many.
    pub fn predict(
        &mut self,
        dynamics: impl Fn(&[f64; N]) -> [f64; N],
        process_noise_sqrt: &[[f64; N]; N],
        rng: &mut Rng,
    ) {
        for particle in &mut self.particles {
            let propagated = dynamics(particle);
            let noise = rng.correlated_normal(process_noise_sqrt);
            for i in 0..N {
                particle[i] = propagated[i] + noise[i];
            }
        }
    }

    /// Re-weight by a measurement, given each particle's **log** likelihood `ln p(z | xᵢ)`.
    ///
    /// A log likelihood rather than a likelihood because the difference is whether the filter works
    /// at all after a few tens of steps: see the module docs. `f64::NEG_INFINITY` is a legitimate
    /// return for a particle a hard constraint rules out.
    ///
    /// # Errors
    ///
    /// [`ParticleError::WeightCollapse`] if no particle retains positive finite weight. The particle
    /// set is left untouched, so the caller can inspect what the observation rejected.
    pub fn update(
        &mut self,
        log_likelihood: impl Fn(&[f64; N]) -> f64,
    ) -> Result<(), ParticleError> {
        let mut logs = [f64::NEG_INFINITY; M];
        let mut peak = f64::NEG_INFINITY;
        for (slot, (particle, weight)) in logs
            .iter_mut()
            .zip(self.particles.iter().zip(&self.weights))
        {
            if *weight <= 0.0 {
                continue;
            }
            let value = log_likelihood(particle) + libm::log(*weight);
            if value.is_nan() {
                continue;
            }
            *slot = value;
            if value > peak {
                peak = value;
            }
        }
        if !peak.is_finite() {
            return Err(ParticleError::WeightCollapse);
        }

        let mut total = 0.0;
        let mut posterior = [0.0; M];
        for (slot, &value) in posterior.iter_mut().zip(&logs) {
            // Subtracting the peak before exponentiating keeps the ratios and cannot overflow: the
            // largest term is exactly 1.
            *slot = libm::exp(value - peak);
            total += *slot;
        }
        // `peak` is finite and its own particle contributes `exp(0) = 1` exactly, so the total is at
        // least one and the division below cannot fail. Guarding it again would be dead code that
        // reads like a live failure mode.
        debug_assert!(total >= 1.0, "the peak particle contributes exactly 1");
        for (weight, value) in self.weights.iter_mut().zip(&posterior) {
            *weight = value / total;
        }
        Ok(())
    }

    /// Kish's effective sample size, `1/Σwᵢ²`.
    ///
    /// `M` when the weights are uniform, `1` when one particle holds everything. It is the honest
    /// count of how many particles are still doing work, and it is the resampling trigger.
    #[must_use]
    pub fn effective_sample_size(&self) -> f64 {
        let mut sum_squares = 0.0;
        for &w in &self.weights {
            sum_squares += w * w;
        }
        // Both constructors and every update leave at least one strictly positive weight, so the
        // sum of squares is positive and needs no guard.
        debug_assert!(sum_squares > 0.0, "the weights are never all zero");
        1.0 / sum_squares
    }

    /// Resample unconditionally, resetting the weights to uniform.
    pub fn resample(&mut self, scheme: Resampling, rng: &mut Rng) {
        let indices = scheme.indices(&self.weights, rng);
        let source = self.particles;
        for (slot, &index) in self.particles.iter_mut().zip(&indices) {
            *slot = source[index];
        }
        self.weights = [1.0 / M as f64; M];
    }

    /// Resample only if the effective sample size has fallen below `fraction · M`; returns whether
    /// it did.
    ///
    /// Resampling is not free — it is a random operation, so it *adds* variance to every estimate
    /// downstream and duplicates particles rather than creating information. Doing it on a schedule
    /// rather than on a diagnosis is the most common way to make a working filter worse.
    pub fn resample_if_degenerate(
        &mut self,
        fraction: f64,
        scheme: Resampling,
        rng: &mut Rng,
    ) -> bool {
        if self.effective_sample_size() >= fraction * M as f64 {
            return false;
        }
        self.resample(scheme, rng);
        true
    }

    /// The weighted mean, `Σ wᵢ xᵢ`.
    ///
    /// For a bimodal posterior this sits *between* the modes, where no particle is — which is why
    /// the mean is a summary of the sample and not a substitute for it.
    #[must_use]
    pub fn mean(&self) -> [f64; N] {
        let mut mean = [0.0; N];
        for (particle, &w) in self.particles.iter().zip(&self.weights) {
            for i in 0..N {
                mean[i] += w * particle[i];
            }
        }
        mean
    }

    /// The weighted covariance about the weighted mean.
    #[must_use]
    pub fn covariance(&self) -> [[f64; N]; N] {
        let mean = self.mean();
        let mut cov = [[0.0; N]; N];
        for (particle, &w) in self.particles.iter().zip(&self.weights) {
            for i in 0..N {
                let di = particle[i] - mean[i];
                for j in 0..N {
                    cov[i][j] += w * di * (particle[j] - mean[j]);
                }
            }
        }
        cov
    }
}

/// The cumulative weights, with the last entry pinned to 1 so that a uniform draw arbitrarily close
/// to 1 cannot fall off the end through accumulated rounding.
fn cumulative<const M: usize>(weights: &[f64; M]) -> [f64; M] {
    let mut cdf = [0.0; M];
    let mut running = 0.0;
    for (slot, &w) in cdf.iter_mut().zip(weights) {
        running += w;
        *slot = running;
    }
    cdf[M - 1] = 1.0;
    cdf
}

/// Locate `M` sorted points in the cumulative weights, in one pass.
///
/// The advance is a bounded `take_while` rather than a `while` over a mutable cursor: the two
/// compute the same index, and only one of them can be made to spin forever by getting the step
/// wrong.
fn comb<const M: usize>(weights: &[f64; M], points: &[f64; M]) -> [usize; M] {
    let cdf = cumulative(weights);
    let mut indices = [0usize; M];
    let mut cursor = 0;
    for (slot, &point) in indices.iter_mut().zip(points) {
        cursor += cdf[cursor..M - 1]
            .iter()
            .take_while(|&&edge| point >= edge)
            .count();
        *slot = cursor;
    }
    indices
}

/// One uniform per stratum: `uᵢ ∈ [i/M, (i+1)/M)`.
fn stratified_points<const M: usize>(rng: &mut Rng) -> [f64; M] {
    let mut points = [0.0; M];
    for (i, slot) in points.iter_mut().enumerate() {
        *slot = (i as f64 + rng.unit()) / M as f64;
    }
    points
}

/// One uniform for the whole comb: `uᵢ = (i + U)/M`.
fn systematic_points<const M: usize>(rng: &mut Rng) -> [f64; M] {
    let offset = rng.unit();
    let mut points = [0.0; M];
    for (i, slot) in points.iter_mut().enumerate() {
        *slot = (i as f64 + offset) / M as f64;
    }
    points
}

/// `M` independent uniforms, each located by binary search — the draws are not sorted, so the
/// one-pass comb does not apply.
///
/// `partition_point` is the binary search: it returns the first index whose cumulative weight the
/// point has not reached, which is the particle that owns it. `cumulative` pins `cdf[M−1]` to 1 and
/// [`Rng::unit`] draws below 1, so that index is never `M`.
fn multinomial<const M: usize>(weights: &[f64; M], rng: &mut Rng) -> [usize; M] {
    let cdf = cumulative(weights);
    let mut indices = [0usize; M];
    for slot in &mut indices {
        let point = rng.unit();
        *slot = cdf.partition_point(|&edge| point >= edge);
    }
    indices
}

/// `⌊M wᵢ⌋` deterministic copies, then multinomial resampling of the fractional remainder.
fn residual<const M: usize>(weights: &[f64; M], rng: &mut Rng) -> [usize; M] {
    let mut indices = [0usize; M];
    let mut filled = 0;
    let mut remainder = [0.0; M];
    let mut remainder_total = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        let copies = libm::floor(w * M as f64);
        let mut taken = 0.0;
        while taken < copies && filled < M {
            indices[filled] = i;
            filled += 1;
            taken += 1.0;
        }
        remainder[i] = w * M as f64 - copies;
        remainder_total += remainder[i];
    }
    if filled == M {
        return indices;
    }
    // The fractional parts are a distribution once normalised; if rounding has left them summing to
    // zero, the deterministic part already carried every unit of weight, so fall back to the largest
    // remainder rather than dividing by zero.
    if remainder_total <= 0.0 {
        for slot in &mut indices[filled..] {
            *slot = M - 1;
        }
        return indices;
    }
    for value in &mut remainder {
        *value /= remainder_total;
    }
    let drawn = multinomial(&remainder, rng);
    for (slot, &index) in indices[filled..].iter_mut().zip(&drawn) {
        *slot = index;
    }
    indices
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::expect_used)]
    use super::*;

    const WEIGHTS: [f64; 5] = [0.1, 0.2, 0.4, 0.25, 0.05];

    /// Every scheme returns `M` indices inside the particle set.
    #[test]
    fn every_scheme_returns_valid_indices() {
        let mut rng = Rng::new(3);
        for scheme in [
            Resampling::Multinomial,
            Resampling::Residual,
            Resampling::Stratified,
            Resampling::Systematic,
        ] {
            for _ in 0..200 {
                let indices = scheme.indices(&WEIGHTS, &mut rng);
                assert_eq!(indices.len(), 5);
                assert!(
                    indices.iter().all(|&i| i < 5),
                    "{scheme:?} produced {indices:?}"
                );
            }
        }
    }

    /// A particle whose weight exceeds `1/M` always survives stratified and systematic resampling.
    ///
    /// This is the property that separates them from multinomial, which can drop it by chance, and
    /// it holds for *every* draw rather than on average — so a single counterexample is a bug.
    #[test]
    fn a_heavy_particle_survives_the_low_variance_schemes() {
        let mut rng = Rng::new(11);
        for scheme in [Resampling::Stratified, Resampling::Systematic] {
            for _ in 0..2000 {
                let indices = scheme.indices(&WEIGHTS, &mut rng);
                assert!(
                    indices.contains(&2),
                    "{scheme:?} dropped the weight-0.4 particle: {indices:?}"
                );
            }
        }
    }

    /// Residual resampling's deterministic half, where nothing is random at all.
    ///
    /// `⌊M wᵢ⌋` copies of each particle is the whole scheme when every `M wᵢ` is an integer, so the
    /// indices are fixed and can be written down. Which is the point of the scheme: the variance it
    /// removes is exactly the part it does not draw for.
    #[test]
    fn the_deterministic_half_of_residual_resampling_is_deterministic() {
        let mut rng = Rng::new(101);
        // M·w = (2, 1, 1, 0) — every unit accounted for, so no remainder is drawn.
        let indices = residual(&[0.5, 0.25, 0.25, 0.0], &mut rng);
        assert_eq!(indices, [0, 0, 1, 2]);
    }

    /// Weights that do not sum to one leave the deterministic half short, and the fallback fills the
    /// rest rather than normalising a zero remainder.
    ///
    /// `Resampling::indices` takes the weights from the caller, so this is reachable from the public
    /// API even though the filter's own weights always sum to one — and the alternative to the
    /// fallback is a division by zero that would return `NaN` indices.
    #[test]
    fn residual_resampling_fills_the_tail_when_no_remainder_is_left() {
        let mut rng = Rng::new(102);
        // M·w = (1, 1, 1, 0): three units placed, one slot short, and every fractional part zero.
        let indices = residual(&[0.25, 0.25, 0.25, 0.0], &mut rng);
        assert_eq!(indices, [0, 1, 2, 3]);
    }

    /// Weights that sum to more than one cannot write past the end of the index array.
    ///
    /// `Resampling::indices` takes the weights from the caller, and `⌊M wᵢ⌋` summed over an
    /// over-unity vector exceeds `M` — the deterministic half would keep placing copies after the
    /// array is full. The `filled < M` bound is what stops it, and this is the only input that asks.
    #[test]
    fn residual_resampling_cannot_overrun_on_over_unity_weights() {
        let mut rng = Rng::new(103);
        // M·w = 2.4 each: four particles claiming two copies apiece, in four slots.
        let indices = residual(&[0.6, 0.6, 0.6, 0.6], &mut rng);
        assert_eq!(indices, [0, 0, 1, 1]);
    }

    /// The effective sample size is `M` for uniform weights and 1 for a collapsed one.
    #[test]
    fn the_effective_sample_size_brackets_its_range() {
        let mut filter = ParticleFilter::<1, 8>::from_particles([[0.0]; 8]);
        assert!((filter.effective_sample_size() - 8.0).abs() < 1e-12);
        filter.weights = [0.0; 8];
        filter.weights[3] = 1.0;
        assert!((filter.effective_sample_size() - 1.0).abs() < 1e-12);
    }

    /// An observation no particle can explain is reported rather than absorbed.
    #[test]
    fn a_total_weight_collapse_is_an_error() {
        let mut filter = ParticleFilter::<1, 4>::from_particles([[0.0], [1.0], [2.0], [3.0]]);
        let outcome = filter.update(|_| f64::NEG_INFINITY);
        assert_eq!(outcome, Err(ParticleError::WeightCollapse));
        assert_eq!(
            *filter.weights(),
            [0.25; 4],
            "a failed update must not mutate"
        );
    }

    /// The log-domain update survives likelihoods whose linear form underflows to zero.
    ///
    /// `exp(-800)` is 0 in `f64`. A filter that multiplied likelihoods would have every weight zero
    /// here and no way back; the ratios are what the recursion needs, and they are intact.
    #[test]
    fn the_update_survives_underflowing_likelihoods() {
        let mut filter = ParticleFilter::<1, 3>::from_particles([[0.0], [1.0], [2.0]]);
        filter
            .update(|x| -800.0 - x[0] * libm::log(2.0))
            .expect("the log domain carries this");
        // Ratios 1 : 1/2 : 1/4, normalised.
        let want = [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0];
        for i in 0..3 {
            assert!(
                (filter.weights()[i] - want[i]).abs() < 1e-12,
                "weight {i} = {}",
                filter.weights()[i]
            );
        }
    }

    /// A drawn prior starts uniformly weighted, and centred where it was asked to be.
    #[test]
    fn the_drawn_prior_starts_uniformly_weighted() {
        let mut rng = Rng::new(21);
        let filter = ParticleFilter::<2, 512>::from_mean_and_covariance_sqrt(
            &[1.0, -2.0],
            &[[0.5, 0.0], [0.0, 0.25]],
            &mut rng,
        );
        assert_eq!(*filter.weights(), [1.0 / 512.0; 512]);
        assert!((filter.effective_sample_size() - 512.0).abs() < 1e-12);
        let mean = filter.mean();
        assert!(
            (mean[0] - 1.0).abs() < 0.05 && (mean[1] + 2.0).abs() < 0.03,
            "the draw is centred on {mean:?}, not on (1, −2)"
        );
    }

    /// `predict` is the dynamics, and the noise it adds is drawn per particle rather than shared.
    ///
    /// The noiseless leg pins the dynamics exactly; the noisy one pins the *independence*, which is
    /// what keeps the sample from moving rigidly and collapsing on the next resample.
    #[test]
    fn the_prediction_is_the_dynamics_plus_an_independent_draw() {
        let mut rng = Rng::new(8);
        let mut noiseless = ParticleFilter::<1, 3>::from_particles([[0.0], [1.0], [2.0]]);
        noiseless.predict(|x| [2.0 * x[0] + 1.0], &[[0.0]], &mut rng);
        assert_eq!(*noiseless.particles(), [[1.0], [3.0], [5.0]]);

        let mut noisy = ParticleFilter::<1, 3>::from_particles([[7.0]; 3]);
        noisy.predict(|x| [x[0]], &[[1.0]], &mut rng);
        let moved = noisy.particles();
        assert!(
            moved[0][0] != moved[1][0] && moved[1][0] != moved[2][0],
            "one draw was shared across the sample: {moved:?}"
        );
    }

    /// Two updates without a resample are one update with the log-likelihoods added.
    ///
    /// That identity *is* the recursion `wᵢ ∝ wᵢ p(z|xᵢ)`, and it is the only thing that makes the
    /// prior weights matter: an update that ignored them would restart from uniform every time, and
    /// would look right for as long as the test only ever performed one.
    #[test]
    fn sequential_updates_multiply_the_likelihoods() {
        let particles = [[0.0], [1.0], [2.0], [3.0]];
        let mut twice = ParticleFilter::<1, 4>::from_particles(particles);
        twice.update(|x| -x[0]).expect("finite likelihoods");
        twice.update(|x| -2.0 * x[0]).expect("finite likelihoods");

        let mut once = ParticleFilter::<1, 4>::from_particles(particles);
        once.update(|x| -3.0 * x[0]).expect("finite likelihoods");

        for i in 0..4 {
            assert!(
                (twice.weights()[i] - once.weights()[i]).abs() < 1e-15,
                "particle {i}: {} after two updates, {} after one",
                twice.weights()[i],
                once.weights()[i]
            );
        }
    }

    /// The resampling trigger is `N_eff < fraction·M`, and the boundary belongs to the no-resample
    /// side.
    ///
    /// Both halves are load-bearing: a trigger that fires on a healthy sample throws information
    /// away every step, and one that never fires lets a single particle take all the weight.
    #[test]
    fn the_resampling_trigger_is_the_effective_sample_size() {
        let mut filter = ParticleFilter::<1, 8>::from_particles([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
        ]);
        let mut rng = Rng::new(13);
        assert!(
            !filter.resample_if_degenerate(0.5, Resampling::Systematic, &mut rng),
            "a uniformly weighted sample is not degenerate"
        );

        // Four particles at ¼ and four at zero: N_eff = 1/(4·1/16) = 4 exactly.
        filter.weights = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(filter.effective_sample_size(), 4.0);
        assert!(
            !filter.resample_if_degenerate(0.5, Resampling::Systematic, &mut rng),
            "N_eff = 4 is not below 0.5·8"
        );
        assert!(
            filter.resample_if_degenerate(0.51, Resampling::Systematic, &mut rng),
            "N_eff = 4 is below 0.51·8"
        );
        assert_eq!(*filter.weights(), [0.125; 8]);
    }

    /// The weighted mean and covariance, against hand-computed values exact in binary.
    #[test]
    fn the_weighted_moments_are_the_moments_of_the_weights() {
        let mut filter = ParticleFilter::<2, 4>::from_particles([
            [0.0, 1.0],
            [2.0, 1.0],
            [4.0, 3.0],
            [6.0, 7.0],
        ]);
        filter.weights = [0.5, 0.25, 0.125, 0.125];
        // x̄ = 0·½ + 2·¼ + 4·⅛ + 6·⅛ = 1.75; ȳ = 1·½ + 1·¼ + 3·⅛ + 7·⅛ = 2.
        assert_eq!(filter.mean(), [1.75, 2.0]);
        // Σ wᵢ (xᵢ − x̄)(xⱼ − x̄) with dx = (−1.75, 0.25, 2.25, 4.25), dy = (−1, −1, 1, 5).
        let cov = filter.covariance();
        assert_eq!(cov[0][0], 4.4375);
        assert_eq!(cov[0][1], 3.75);
        assert_eq!(cov[1][0], 3.75);
        assert_eq!(cov[1][1], 4.0);
    }

    /// Resampling restores uniform weights and keeps only particles that had weight.
    #[test]
    fn resampling_drops_the_zero_weight_particles() {
        let mut filter = ParticleFilter::<1, 4>::from_particles([[0.0], [1.0], [2.0], [3.0]]);
        filter.weights = [0.0, 0.5, 0.5, 0.0];
        let mut rng = Rng::new(17);
        filter.resample(Resampling::Systematic, &mut rng);
        assert_eq!(*filter.weights(), [0.25; 4]);
        for particle in filter.particles() {
            assert!(
                particle[0] == 1.0 || particle[0] == 2.0,
                "a zero-weight particle survived: {particle:?}"
            );
        }
    }
}
