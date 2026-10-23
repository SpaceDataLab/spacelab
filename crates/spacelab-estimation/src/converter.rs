//! Fitting a propagator's own parameter vector to a sampled reference trajectory.
//!
//! This is the direction opposite to propagation. A propagator turns `N` numbers — six mean
//! elements, an epoch state, a drag coefficient — into a trajectory; a *converter* takes a
//! trajectory somebody else produced and recovers the `N` numbers that reproduce it best in the
//! weighted least-squares sense. It is what makes a precise ephemeris expressible as an element
//! set, and what makes two propagators comparable on the same orbit.
//!
//! # Shape
//!
//! [`PropagatorBuilder`] is the parameter vector's owner: it turns `N` numbers into a
//! [`SampledPropagator`], which answers with a state at a time. The split is not decoration — for
//! every propagator worth converting, going from parameters to a propagator is the expensive half
//! (SGP4's per-object initialisation, an integrator's whole arc) and evaluating it at one more
//! epoch is the cheap half. A converter that could only ask for `(parameters, time) -> state` would
//! pay the expensive half once per *sample* instead of once per *parameter vector*, which for a
//! finite-difference Jacobian over 96 samples and 7 parameters is 1440 initialisations against 15.
//!
//! # Two sources of partials, one converter
//!
//! Orekit splits this across `FiniteDifferencePropagatorConverter` and
//! `JacobianPropagatorConverter`. The only thing that differs between them is where
//! `∂state(t)/∂parameters` comes from, so here it is two methods on one [`PropagatorConverter`]
//! rather than two types with a shared abstract base: the stopping rules, the weights and the
//! reported residuals are the same object either way, and a caller choosing between them is
//! choosing a Jacobian, not a converter.
//!
//! Use [`PropagatorConverter::fit_by_jacobian`] whenever the propagator can produce its own
//! partials — a numerically integrated arc carries them in the variational equations, and they are
//! exact. Use [`PropagatorConverter::fit_by_finite_difference`] when it cannot: SGP4 is a fixed
//! listing with no derivative in it, and differencing the listing is the only Jacobian there is.
//!
//! # Frames and units are the caller's
//!
//! Nothing here knows what frame the states are in or what units the parameters carry. The one
//! invariant is that [`TrajectorySample::state`] and [`SampledPropagator::state_at`] are expressed
//! in the *same* frame — SGP4 answers in TEME, so a reference arc in ITRF has to be rotated before
//! it gets here, not after. Position and velocity weights are given in that frame's units, and the
//! finite-difference steps in the parameters' own units, because only the caller knows them.

#![allow(clippy::needless_range_loop)] // matrix code indexes several arrays by one i/j/k

use core::fmt;

use crate::gauss_newton::{GaussNewton, GaussNewtonReport, ResidualModel};
use crate::srif::SrifError;

/// Residual substituted for a sample the propagator has no state for.
///
/// A parameter vector the propagator cannot evaluate has no residual at all, and the Gauss-Newton
/// loop compares costs rather than catching failures. A large *finite* number makes such a vector
/// lose every comparison it enters — the damped step is rejected and the damping tightens, which is
/// exactly the response wanted — where a `NaN` would make the comparison itself meaningless and an
/// infinity would make every failing vector equally bad as the cost stopped being ordered.
///
/// `1e15` m is 6700 astronomical units: no state a propagator returns is that far from a reference
/// sample, so the substitution can never beat a real residual. Squared and summed over a million
/// samples it is `1e36`, which leaves 270 decades of headroom under the `f64` ceiling.
const UNEVALUABLE_RESIDUAL: f64 = 1e15;

/// One state of the reference trajectory being fitted.
///
/// `t` is seconds from whatever epoch the parameter vector is referred to, and `state` is position
/// (first three) and velocity (last three) in the propagator's own output frame.
///
/// A position-only fit ([`PropagatorConverter::velocity_sigma`] left `None`) never reads the last
/// three, so a caller whose reference product carries no velocity may leave them at zero — but
/// [`residual_statistics`] measures them regardless, and its `velocity_rms` is then the orbital
/// speed rather than a residual. Read `position_rms` alone in that case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
    /// Seconds from the parameter set's epoch. Negative propagates backwards.
    pub t: f64,
    /// Position and velocity, `[x, y, z, vx, vy, vz]`.
    pub state: [f64; 6],
}

/// A propagator already built from one parameter vector: cheap to evaluate, at any epoch.
pub trait SampledPropagator {
    /// The state at `t` seconds from the epoch, or `None` where the propagator has no answer.
    ///
    /// `None` is a statement about the model, not an error: SGP4 outside its domain, an object that
    /// has re-entered, an integrator that will not converge. The converter reads it as "this trial
    /// parameter vector is worse than any evaluable one" and steps away from it.
    fn state_at(&self, t: f64) -> Option<[f64; 6]>;
}

/// A propagator that carries its own partials with respect to the fitted parameters.
///
/// The natural implementer is a numerically integrated arc whose variational equations were
/// integrated alongside the state: the state-transition matrix *is* `∂state(t)/∂parameters` when
/// the parameters are the epoch state, and its extension by the parameter columns is when they are
/// not.
pub trait DifferentiablePropagator<const N: usize>: SampledPropagator {
    /// The state at `t` and `∂state(t)/∂parameters`, indexed `[component][parameter]`.
    fn state_and_partials(&self, t: f64) -> Option<([f64; 6], [[f64; N]; 6])>;
}

/// The owner of a parameter vector: `N` numbers in, a propagator out.
pub trait PropagatorBuilder<const N: usize> {
    /// What one parameter vector builds into.
    type Propagator: SampledPropagator;

    /// Build from `parameters`, or `None` if they are outside the model's domain.
    ///
    /// Returning `None` rather than clamping matters: a silently clamped eccentricity produces a
    /// propagator whose partials point the wrong way, and the fit then converges onto the clamp.
    fn build(&self, parameters: &[f64; N]) -> Option<Self::Propagator>;
}

/// Weighted least squares of a propagator's parameters against a reference trajectory.
///
/// The sigmas are weights, not claims: with one uniform position sigma the solution is plain least
/// squares and the sigma only rescales the reported covariance.
#[derive(Debug, Clone, Copy)]
pub struct PropagatorConverter {
    /// The nonlinear solver — iteration cap, tolerances and Levenberg-Marquardt damping.
    pub solver: GaussNewton,
    /// Per-component position uncertainty, in the reference trajectory's length unit.
    pub position_sigma: f64,
    /// Per-component velocity uncertainty. `None` fits position only.
    ///
    /// Fitting velocity is not free information: an SP3 product's velocity records are usually the
    /// analysis centre's own differentiation of its positions, so weighting them alongside the
    /// positions counts the same measurement twice and tightens the reported covariance without
    /// tightening the fit.
    pub velocity_sigma: Option<f64>,
}

impl Default for PropagatorConverter {
    fn default() -> Self {
        Self {
            solver: GaussNewton::default(),
            position_sigma: 1.0,
            velocity_sigma: None,
        }
    }
}

/// How well a propagator reproduces a set of reference samples.
///
/// Reported over the fit span by [`PropagatorConverter`], and over any other span by
/// [`residual_statistics`] — the same measurement either way, which is what makes a fit residual
/// and a prediction residual comparable numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryResiduals {
    /// Root mean square of the 3-D position difference.
    pub position_rms: f64,
    /// Root mean square of the 3-D velocity difference.
    pub velocity_rms: f64,
    /// Largest single 3-D position difference.
    pub worst_position: f64,
    /// Samples the statistics were taken over.
    pub samples: usize,
}

/// A converged fit: the parameters, their covariance, the solver's report and the residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConverterFit<const N: usize> {
    /// The fitted parameter vector.
    pub parameters: [f64; N],
    /// Formal covariance of the parameters, scaled by the sigmas the fit was weighted with.
    pub covariance: [[f64; N]; N],
    /// Iterations taken, final weighted cost, and whether a convergence criterion was met.
    pub report: GaussNewtonReport,
    /// Post-fit residuals over the samples that were fitted.
    pub residuals: TrajectoryResiduals,
}

/// Why a conversion could not be produced.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ConverterError {
    /// No reference samples were given: there is nothing to fit to.
    NoSamples,
    /// A sigma was not strictly positive, so its weight `1/σ²` is not a weight.
    NonPositiveSigma,
    /// A finite-difference step was not strictly positive.
    NonPositiveStep {
        /// Index into the parameter vector.
        parameter: usize,
    },
    /// The propagator refused the parameter vector — the initial guess, or the converged answer.
    Unbuildable,
    /// The propagator was built but has no state at one of the samples.
    ///
    /// Reported against the *converged* parameters: a trial vector that fails mid-iteration is
    /// simply stepped away from, but an answer that cannot be evaluated over its own fit span is
    /// not an answer.
    Unevaluable {
        /// Index into the sample slice.
        sample: usize,
    },
    /// The normal direction is unobserved: some parameter leaves the trajectory unchanged.
    Solver(SrifError),
}

impl fmt::Display for ConverterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSamples => write!(f, "no reference samples to fit to"),
            Self::NonPositiveSigma => write!(f, "a measurement sigma is not strictly positive"),
            Self::NonPositiveStep { parameter } => write!(
                f,
                "the finite-difference step for parameter {parameter} is not strictly positive"
            ),
            Self::Unbuildable => write!(f, "the propagator refused the parameter vector"),
            Self::Unevaluable { sample } => {
                write!(f, "the propagator has no state at sample {sample}")
            }
            Self::Solver(e) => write!(f, "the least-squares solve failed: {e}"),
        }
    }
}

impl core::error::Error for ConverterError {}

impl From<SrifError> for ConverterError {
    fn from(e: SrifError) -> Self {
        Self::Solver(e)
    }
}

impl PropagatorConverter {
    /// Fit `N` parameters with a central-difference Jacobian, one step per parameter.
    ///
    /// The steps are absolute and in the parameters' own units, because a relative step is
    /// meaningless for a parameter whose value passes through zero — an eccentricity vector
    /// component, an empirical acceleration, `B*` on a drag-free orbit.
    ///
    /// Cost per solver evaluation is `2N + 1` calls to [`PropagatorBuilder::build`] and
    /// `(2N + 1) · samples` calls to [`SampledPropagator::state_at`].
    ///
    /// # Errors
    ///
    /// [`ConverterError::NoSamples`] for an empty sample slice, [`ConverterError::NonPositiveSigma`]
    /// or [`ConverterError::NonPositiveStep`] for a weight or step that is not one,
    /// [`ConverterError::Unbuildable`] if the initial guess is outside the propagator's domain,
    /// [`ConverterError::Unevaluable`] if the converged answer is, and [`ConverterError::Solver`]
    /// if the samples never make some parameter observable.
    pub fn fit_by_finite_difference<const N: usize, B: PropagatorBuilder<N>>(
        &self,
        builder: &B,
        samples: &[TrajectorySample],
        initial: &[f64; N],
        steps: &[f64; N],
    ) -> Result<ConverterFit<N>, ConverterError> {
        let weights = self.weights(samples)?;
        for (parameter, &step) in steps.iter().enumerate() {
            if step <= 0.0 || !step.is_finite() {
                return Err(ConverterError::NonPositiveStep { parameter });
            }
        }
        if builder.build(initial).is_none() {
            return Err(ConverterError::Unbuildable);
        }
        let model = FiniteDifferenceModel {
            builder,
            samples,
            steps,
            weights,
        };
        self.finish(&model, builder, samples, initial)
    }

    /// Fit `N` parameters with the propagator's own partials.
    ///
    /// Exact where finite differences are approximate, and `2N` builds cheaper per evaluation. The
    /// price is that the propagator has to carry the partials, which for an integrated arc means
    /// integrating the variational equations alongside the state.
    ///
    /// # Errors
    ///
    /// As [`PropagatorConverter::fit_by_finite_difference`], less the step checks.
    pub fn fit_by_jacobian<const N: usize, B: PropagatorBuilder<N>>(
        &self,
        builder: &B,
        samples: &[TrajectorySample],
        initial: &[f64; N],
    ) -> Result<ConverterFit<N>, ConverterError>
    where
        B::Propagator: DifferentiablePropagator<N>,
    {
        let weights = self.weights(samples)?;
        if builder.build(initial).is_none() {
            return Err(ConverterError::Unbuildable);
        }
        let model = JacobianModel {
            builder,
            samples,
            weights,
        };
        self.finish(&model, builder, samples, initial)
    }

    /// Validate the weights and decide how many components each sample contributes.
    fn weights(&self, samples: &[TrajectorySample]) -> Result<Weights, ConverterError> {
        if samples.is_empty() {
            return Err(ConverterError::NoSamples);
        }
        if self.position_sigma <= 0.0 || !self.position_sigma.is_finite() {
            return Err(ConverterError::NonPositiveSigma);
        }
        match self.velocity_sigma {
            None => Ok(Weights {
                position: 1.0 / (self.position_sigma * self.position_sigma),
                velocity: 0.0,
                components: 3,
            }),
            Some(sigma) if sigma > 0.0 && sigma.is_finite() => Ok(Weights {
                position: 1.0 / (self.position_sigma * self.position_sigma),
                velocity: 1.0 / (sigma * sigma),
                components: 6,
            }),
            Some(_) => Err(ConverterError::NonPositiveSigma),
        }
    }

    /// Run the solver and measure the answer it converged to.
    fn finish<const N: usize, B: PropagatorBuilder<N>, M: ResidualModel<N>>(
        &self,
        model: &M,
        builder: &B,
        samples: &[TrajectorySample],
        initial: &[f64; N],
    ) -> Result<ConverterFit<N>, ConverterError> {
        let (parameters, covariance, report) = self.solver.solve(model, initial)?;
        let propagator = builder
            .build(&parameters)
            .ok_or(ConverterError::Unbuildable)?;
        let residuals = residual_statistics(&propagator, samples)?;
        Ok(ConverterFit {
            parameters,
            covariance,
            report,
            residuals,
        })
    }
}

/// Position and velocity residuals of a propagator against reference samples.
///
/// Separate from the fit so that a prediction span can be measured with the same code the fit span
/// was measured with — the comparison between "how well it fits" and "how well it predicts" is only
/// meaningful if both numbers came out of the same expression.
///
/// # Errors
///
/// [`ConverterError::NoSamples`] for an empty slice, [`ConverterError::Unevaluable`] if the
/// propagator has no state at one of them.
pub fn residual_statistics<P: SampledPropagator>(
    propagator: &P,
    samples: &[TrajectorySample],
) -> Result<TrajectoryResiduals, ConverterError> {
    if samples.is_empty() {
        return Err(ConverterError::NoSamples);
    }
    let mut position_sum = 0.0;
    let mut velocity_sum = 0.0;
    let mut worst_position = 0.0_f64;
    for (sample, reference) in samples.iter().enumerate() {
        let state = propagator
            .state_at(reference.t)
            .ok_or(ConverterError::Unevaluable { sample })?;
        let mut position = 0.0;
        let mut velocity = 0.0;
        for k in 0..3 {
            let dr = state[k] - reference.state[k];
            let dv = state[k + 3] - reference.state[k + 3];
            position += dr * dr;
            velocity += dv * dv;
        }
        position_sum += position;
        velocity_sum += velocity;
        worst_position = worst_position.max(libm::sqrt(position));
    }
    #[allow(clippy::cast_precision_loss)] // sample counts are far below 2⁵³
    let count = samples.len() as f64;
    Ok(TrajectoryResiduals {
        position_rms: libm::sqrt(position_sum / count),
        velocity_rms: libm::sqrt(velocity_sum / count),
        worst_position,
        samples: samples.len(),
    })
}

/// Measurement weights and how many components of each sample are fitted.
#[derive(Debug, Clone, Copy)]
struct Weights {
    position: f64,
    velocity: f64,
    components: usize,
}

impl Weights {
    const fn of(self, component: usize) -> f64 {
        if component < 3 {
            self.position
        } else {
            self.velocity
        }
    }
}

/// Emit one sample's rows: the residual `predicted − reference` against its partials.
fn emit_sample<const N: usize>(
    state: &[f64; 6],
    reference: &TrajectorySample,
    partials: &[[f64; N]; 6],
    weights: Weights,
    emit: &mut dyn FnMut(&[f64; N], f64, f64),
) {
    for k in 0..weights.components {
        emit(&partials[k], state[k] - reference.state[k], weights.of(k));
    }
}

/// Emit one sample's rows when there is no state: a zero row, so the information matrix is
/// untouched, against [`UNEVALUABLE_RESIDUAL`], so the cost rejects the step.
fn emit_unevaluable<const N: usize>(weights: Weights, emit: &mut dyn FnMut(&[f64; N], f64, f64)) {
    let zero = [0.0; N];
    for k in 0..weights.components {
        emit(&zero, UNEVALUABLE_RESIDUAL, weights.of(k));
    }
}

/// The residual model whose partials come from central differences of the propagator.
struct FiniteDifferenceModel<'a, const N: usize, B: PropagatorBuilder<N>> {
    builder: &'a B,
    samples: &'a [TrajectorySample],
    steps: &'a [f64; N],
    weights: Weights,
}

impl<const N: usize, B: PropagatorBuilder<N>> ResidualModel<N> for FiniteDifferenceModel<'_, N, B> {
    fn residuals(&self, x: &[f64; N], emit: &mut dyn FnMut(&[f64; N], f64, f64)) {
        let Some(nominal) = self.builder.build(x) else {
            for _ in self.samples {
                emit_unevaluable(self.weights, emit);
            }
            return;
        };
        // One pair of propagators per parameter, built once for the whole sample set rather than
        // once per sample: this is the reason `PropagatorBuilder` and `SampledPropagator` are two
        // traits.
        let perturbed: [Option<(B::Propagator, B::Propagator)>; N] = core::array::from_fn(|j| {
            let mut forward = *x;
            forward[j] += self.steps[j];
            let mut backward = *x;
            backward[j] -= self.steps[j];
            Some((
                self.builder.build(&forward)?,
                self.builder.build(&backward)?,
            ))
        });

        for reference in self.samples {
            let Some(state) = nominal.state_at(reference.t) else {
                emit_unevaluable(self.weights, emit);
                continue;
            };
            let mut partials = [[0.0; N]; 6];
            let mut complete = true;
            for j in 0..N {
                let Some((forward, backward)) = &perturbed[j] else {
                    complete = false;
                    break;
                };
                let (Some(plus), Some(minus)) = (
                    forward.state_at(reference.t),
                    backward.state_at(reference.t),
                ) else {
                    complete = false;
                    break;
                };
                let scale = 0.5 / self.steps[j];
                for k in 0..6 {
                    partials[k][j] = (plus[k] - minus[k]) * scale;
                }
            }
            if complete {
                emit_sample(&state, reference, &partials, self.weights, emit);
            } else {
                // A column that cannot be differenced makes the whole row a guess. Reject the
                // parameter vector instead of fitting against a partially zero Jacobian, which
                // would move the solution in a direction nothing measured.
                emit_unevaluable(self.weights, emit);
            }
        }
    }
}

/// The residual model whose partials come from the propagator itself.
struct JacobianModel<'a, const N: usize, B: PropagatorBuilder<N>> {
    builder: &'a B,
    samples: &'a [TrajectorySample],
    weights: Weights,
}

impl<const N: usize, B: PropagatorBuilder<N>> ResidualModel<N> for JacobianModel<'_, N, B>
where
    B::Propagator: DifferentiablePropagator<N>,
{
    fn residuals(&self, x: &[f64; N], emit: &mut dyn FnMut(&[f64; N], f64, f64)) {
        let Some(propagator) = self.builder.build(x) else {
            for _ in self.samples {
                emit_unevaluable(self.weights, emit);
            }
            return;
        };
        for reference in self.samples {
            match propagator.state_and_partials(reference.t) {
                Some((state, partials)) => {
                    emit_sample(&state, reference, &partials, self.weights, emit);
                }
                None => emit_unevaluable(self.weights, emit),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    /// A straight line in six dimensions: `state(t) = p[0..6] + t · [1, 2, 3, 0, 0, 0]`.
    ///
    /// Linear in its parameters, so both Jacobians are the identity to the bit and the two
    /// converters have to agree exactly. It exists to test the plumbing — weights, component
    /// counts, the domain refusal, the unevaluable path — without any orbital mechanics in the way.
    struct Line {
        /// Times past which the propagator refuses to answer.
        horizon: f64,
    }

    struct LineArc {
        parameters: [f64; 6],
        horizon: f64,
    }

    const DRIFT: [f64; 6] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];

    impl SampledPropagator for LineArc {
        fn state_at(&self, t: f64) -> Option<[f64; 6]> {
            if t > self.horizon {
                return None;
            }
            Some(core::array::from_fn(|k| self.parameters[k] + t * DRIFT[k]))
        }
    }

    impl DifferentiablePropagator<6> for LineArc {
        fn state_and_partials(&self, t: f64) -> Option<([f64; 6], [[f64; 6]; 6])> {
            let state = self.state_at(t)?;
            let mut partials = [[0.0; 6]; 6];
            for k in 0..6 {
                partials[k][k] = 1.0;
            }
            Some((state, partials))
        }
    }

    impl PropagatorBuilder<6> for Line {
        type Propagator = LineArc;
        fn build(&self, parameters: &[f64; 6]) -> Option<LineArc> {
            // A domain: the first component must stay positive.
            if parameters[0] <= 0.0 {
                return None;
            }
            Some(LineArc {
                parameters: *parameters,
                horizon: self.horizon,
            })
        }
    }

    const TRUTH: [f64; 6] = [10.0, 20.0, 30.0, 0.5, -0.5, 0.25];

    fn samples(horizon: f64) -> Vec<TrajectorySample> {
        (0..20)
            .map(|i| {
                let t = f64::from(i) * horizon / 19.0;
                TrajectorySample {
                    t,
                    state: core::array::from_fn(|k| TRUTH[k] + t * DRIFT[k]),
                }
            })
            .collect()
    }

    #[test]
    fn both_jacobian_sources_recover_the_same_parameters() {
        let line = Line { horizon: 100.0 };
        let samples = samples(100.0);
        let converter = PropagatorConverter {
            velocity_sigma: Some(1.0),
            ..PropagatorConverter::default()
        };
        let guess = [9.0, 19.0, 29.0, 0.0, 0.0, 0.0];
        let steps = [1e-4; 6];

        let difference = converter
            .fit_by_finite_difference(&line, &samples, &guess, &steps)
            .unwrap();
        let analytic = converter.fit_by_jacobian(&line, &samples, &guess).unwrap();

        // Neither reaches the exact answer, and the reason is the stopping rule rather than either
        // Jacobian: `GaussNewton::step_tolerance` is relative at 1e-12 and the state norm is ~40, so
        // the loop stops with ~4e-11 of correction still on the table. Measured 2.7e-11 either way.
        for k in 0..6 {
            assert!(
                (difference.parameters[k] - TRUTH[k]).abs() < 1e-9,
                "finite difference recovered {} for {}",
                difference.parameters[k],
                TRUTH[k]
            );
            assert!(
                (analytic.parameters[k] - TRUTH[k]).abs() < 1e-9,
                "analytic recovered {} for {}",
                analytic.parameters[k],
                TRUTH[k]
            );
            // The two Jacobians are the same matrix here — the model is linear, so the central
            // difference is exact — and the fits must therefore agree far more tightly than either
            // agrees with the truth.
            assert!(
                (difference.parameters[k] - analytic.parameters[k]).abs() < 1e-10,
                "the two Jacobian sources disagree at parameter {k}: {} against {}",
                difference.parameters[k],
                analytic.parameters[k]
            );
        }
        assert!(difference.residuals.position_rms < 1e-9);
        assert!(analytic.residuals.position_rms < 1e-9);
        assert_eq!(analytic.residuals.samples, 20);
    }

    #[test]
    fn a_guess_outside_the_domain_is_refused_rather_than_clamped() {
        let line = Line { horizon: 100.0 };
        let samples = samples(100.0);
        assert_eq!(
            PropagatorConverter::default()
                .fit_by_jacobian(&line, &samples, &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap_err(),
            ConverterError::Unbuildable
        );
    }

    /// Samples the converged propagator cannot reach are an error, not a shorter fit.
    #[test]
    fn a_sample_past_the_propagator_horizon_is_reported() {
        let line = Line { horizon: 50.0 };
        let samples = samples(100.0);
        // Velocity is weighted so that all six parameters are observable: with position only, the
        // solve would fail as rank-deficient before it ever reached the unevaluable samples.
        assert!(matches!(
            PropagatorConverter {
                velocity_sigma: Some(1.0),
                ..PropagatorConverter::default()
            }
            .fit_by_jacobian(&line, &samples, &TRUTH)
            .unwrap_err(),
            ConverterError::Unevaluable { .. }
        ));
    }

    #[test]
    fn empty_samples_and_bad_weights_are_refused_before_any_propagation() {
        let line = Line { horizon: 100.0 };
        assert_eq!(
            PropagatorConverter::default()
                .fit_by_jacobian(&line, &[], &TRUTH)
                .unwrap_err(),
            ConverterError::NoSamples
        );
        let samples = samples(100.0);
        assert_eq!(
            PropagatorConverter {
                position_sigma: 0.0,
                ..PropagatorConverter::default()
            }
            .fit_by_jacobian(&line, &samples, &TRUTH)
            .unwrap_err(),
            ConverterError::NonPositiveSigma
        );
        assert_eq!(
            PropagatorConverter {
                velocity_sigma: Some(-1.0),
                ..PropagatorConverter::default()
            }
            .fit_by_jacobian(&line, &samples, &TRUTH)
            .unwrap_err(),
            ConverterError::NonPositiveSigma
        );
        // Zero is refused too, and it is the case that matters: a sigma of zero passes any check
        // written as "not negative" and turns into an infinite weight rather than an error.
        assert_eq!(
            PropagatorConverter {
                velocity_sigma: Some(0.0),
                ..PropagatorConverter::default()
            }
            .fit_by_jacobian(&line, &samples, &TRUTH)
            .unwrap_err(),
            ConverterError::NonPositiveSigma
        );
        assert_eq!(
            PropagatorConverter::default()
                .fit_by_finite_difference(&line, &samples, &TRUTH, &[1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
                .unwrap_err(),
            ConverterError::NonPositiveStep { parameter: 1 }
        );
    }

    /// Position-only fitting leaves the velocity parameters unobserved, and the solver says so
    /// rather than returning whatever the guess held.
    #[test]
    fn an_unobserved_parameter_is_reported_rather_than_invented() {
        let line = Line { horizon: 100.0 };
        let samples = samples(100.0);
        assert!(matches!(
            PropagatorConverter::default()
                .fit_by_jacobian(&line, &samples, &[9.0, 19.0, 29.0, 0.0, 0.0, 0.0])
                .unwrap_err(),
            ConverterError::Solver(SrifError::RankDeficient { .. })
        ));
    }

    /// Two samples the propagator can reach, read row by row rather than through a fit.
    ///
    /// A converged parameter vector is insensitive to the weights — a weight of `1/σ` instead of
    /// `1/σ²` reaches the same minimum on a consistent problem — so the only place the sigmas are
    /// visible is the row the model emits. Both sigmas are chosen away from 1 and away from each
    /// other so that `1/σ²`, `σ²`, `1/(σ+σ)` and `1/(σ/σ)` are four different numbers, and the
    /// position and velocity weights cannot be confused for one another.
    #[test]
    fn a_weighted_row_is_the_residual_and_the_weight_the_sigmas_imply() {
        let line = Line { horizon: 100.0 };
        let converter = PropagatorConverter {
            position_sigma: 4.0,
            velocity_sigma: Some(0.5),
            ..PropagatorConverter::default()
        };
        // `state(t) = TRUTH + t · DRIFT`, so the propagated states are [10, 20, 30, 0.5, -0.5, 0.25]
        // at t = 0 and [12, 24, 36, 0.5, -0.5, 0.25] at t = 2. Each reference below is one of those
        // minus the residual the row must carry; every value is exact in binary.
        let samples = [
            TrajectorySample {
                t: 0.0,
                state: [9.0, 22.0, 30.5, 0.25, -0.5, 0.75],
            },
            TrajectorySample {
                t: 2.0,
                state: [12.5, 20.0, 36.0, 1.0, -1.5, 0.25],
            },
        ];
        let weights = converter.weights(&samples).unwrap();
        let model = JacobianModel {
            builder: &line,
            samples: &samples,
            weights,
        };

        let mut rows = Vec::new();
        model.residuals(&TRUTH, &mut |_h, r, w| rows.push((r, w)));

        // 1/4² = 0.0625 on a position component, 1/0.5² = 4 on a velocity one.
        let expected = [
            (1.0, 0.0625),
            (-2.0, 0.0625),
            (-0.5, 0.0625),
            (0.25, 4.0),
            (0.0, 4.0),
            (-0.5, 4.0),
            (-0.5, 0.0625),
            (4.0, 0.0625),
            (0.0, 0.0625),
            (-0.5, 4.0),
            (1.0, 4.0),
            (0.0, 4.0),
        ];
        assert_eq!(rows.len(), expected.len(), "one row per fitted component");
        for (k, (&(residual, weight), &(want_residual, want_weight))) in
            rows.iter().zip(&expected).enumerate()
        {
            assert_eq!(residual, want_residual, "the residual of row {k}");
            assert_eq!(weight, want_weight, "the weight of row {k}");
        }
    }

    /// A sample the propagator cannot reach still emits its rows — a zero Jacobian against the
    /// substitute residual — and that is what makes the trial vector lose the cost comparison.
    ///
    /// Asserted on the rows rather than through a fit: a fit survives the sample either way, because
    /// emitting nothing for it merely fits the remaining samples, which on a consistent problem is
    /// the same answer.
    #[test]
    fn an_unreachable_sample_emits_a_rejecting_row_rather_than_no_row() {
        let line = Line { horizon: 1.0 };
        let converter = PropagatorConverter {
            position_sigma: 4.0,
            velocity_sigma: Some(0.5),
            ..PropagatorConverter::default()
        };
        let samples = [
            TrajectorySample {
                t: 0.0,
                state: TRUTH,
            },
            TrajectorySample {
                t: 5.0,
                state: TRUTH,
            },
        ];
        let weights = converter.weights(&samples).unwrap();
        let model = JacobianModel {
            builder: &line,
            samples: &samples,
            weights,
        };

        let mut rows = Vec::new();
        model.residuals(&TRUTH, &mut |h: &[f64; 6], r, w| rows.push((*h, r, w)));

        assert_eq!(rows.len(), 12, "both samples emit their six components");
        for (k, &(jacobian, residual, _)) in rows.iter().take(6).enumerate() {
            assert_eq!(jacobian[k], 1.0, "the unit partial of component {k}");
            assert_eq!(residual, 0.0, "the residual at t = 0, where TRUTH is exact");
        }
        for (k, &(jacobian, residual, weight)) in rows.iter().skip(6).enumerate() {
            assert_eq!(
                jacobian, [0.0; 6],
                "an unevaluable row must leave the information matrix untouched"
            );
            assert_eq!(residual, UNEVALUABLE_RESIDUAL, "the substitute residual");
            assert_eq!(
                weight,
                if k < 3 { 0.0625 } else { 4.0 },
                "an unevaluable row carries the component's own weight"
            );
        }
    }

    /// A propagator that answers with the same state at every epoch: the residuals are then the
    /// differences the samples were built from, with no propagation arithmetic in between.
    struct Frozen([f64; 6]);

    impl SampledPropagator for Frozen {
        fn state_at(&self, _t: f64) -> Option<[f64; 6]> {
            Some(self.0)
        }
    }

    /// The reported statistics against differences small enough to sum in the head.
    ///
    /// Position differences (3, 4, 12), (2, 6, 9) and (1, 3, 0) have squared norms 169, 121 and 10,
    /// summing to 300; velocity differences (1, 2, 2), (2, 3, 6) and (3, 4, 5) have 9, 49 and 50,
    /// summing to 108. Over three samples that is `√(300/3) = 10` of position RMS, `√(108/3) = 6` of
    /// velocity RMS and a worst position difference of `√169 = 13` — all exact in `f64`, so the
    /// assertions are equalities rather than tolerances.
    #[test]
    fn the_reported_residuals_are_the_root_mean_square_of_the_differences() {
        let state = [10.0, 20.0, 30.0, 5.0, 7.0, 11.0];
        let differences = [
            [3.0, 4.0, 12.0, 1.0, 2.0, 2.0],
            [2.0, 6.0, 9.0, 2.0, 3.0, 6.0],
            [1.0, 3.0, 0.0, 3.0, 4.0, 5.0],
        ];
        let times = [0.0, 10.0, 20.0];
        let samples: Vec<TrajectorySample> = times
            .iter()
            .zip(&differences)
            .map(|(&t, d)| TrajectorySample {
                t,
                state: core::array::from_fn(|k| state[k] - d[k]),
            })
            .collect();

        let residuals = residual_statistics(&Frozen(state), &samples).unwrap();
        assert_eq!(residuals.position_rms, 10.0);
        assert_eq!(residuals.velocity_rms, 6.0);
        assert_eq!(residuals.worst_position, 13.0);
        assert_eq!(residuals.samples, 3);
    }

    /// Each refusal says which one it is, and the solver's own diagnosis is carried through rather
    /// than flattened into "the solve failed".
    #[test]
    fn every_converter_error_says_which_one_it_is() {
        let messages = [
            (ConverterError::NoSamples, "no reference samples to fit to"),
            (
                ConverterError::NonPositiveSigma,
                "a measurement sigma is not strictly positive",
            ),
            (
                ConverterError::NonPositiveStep { parameter: 4 },
                "the finite-difference step for parameter 4 is not strictly positive",
            ),
            (
                ConverterError::Unbuildable,
                "the propagator refused the parameter vector",
            ),
            (
                ConverterError::Unevaluable { sample: 7 },
                "the propagator has no state at sample 7",
            ),
            (
                ConverterError::Solver(SrifError::RankDeficient { direction: 2 }),
                "the least-squares solve failed: state direction 2 is unobserved (singular information matrix)",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(format!("{error}"), expected);
            let dynamic: &dyn core::error::Error = &error;
            assert_eq!(format!("{dynamic}"), expected);
        }
    }
}
