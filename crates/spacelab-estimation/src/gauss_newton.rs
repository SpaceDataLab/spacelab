//! Nonlinear least squares by Gauss–Newton with Levenberg–Marquardt damping, on top of the SRIF.
//!
//! Orbit determination is nonlinear: the predicted observation is a nonlinear function of the epoch
//! state. Gauss–Newton linearises at the current estimate and solves for a correction; the linear
//! solve is the [`Srif`], so the condition number is never squared. Levenberg–Marquardt damping —
//! which GMAT's batch estimator lacks ([notes/14-estimation.md](../notes/14-estimation.md) §6) —
//! degrades the step gracefully toward steepest descent when the Gauss–Newton step would overshoot,
//! so a poor initial guess converges instead of diverging.
//!
//! The damping is Marquardt's `λ·diag(HᵀWH)`, not Levenberg's `λI`. The difference is not cosmetic:
//! with `λI` the damping parameter is compared against the information matrix's own entries, so it
//! carries the units of the state and the weights and has to be retuned per problem, and a `λ`
//! several decades too small is spent climbing the schedule rather than taking steps.

#![allow(clippy::needless_range_loop, clippy::many_single_char_names)] // matrix code indexes several arrays by one i/j/k

use crate::srif::{Srif, SrifError};

/// A nonlinear residual model over an `N`-dimensional state.
///
/// At a given `x`, emit each observation's linearisation: the Jacobian row `∂rₖ/∂x`, the residual
/// `rₖ = predicted − observed`, and the weight `1/σₖ²`. Emitting one at a time keeps the observation
/// count off the type and out of memory.
pub trait ResidualModel<const N: usize> {
    /// Report each observation at `x` through `emit(jacobian_row, residual, weight)`.
    fn residuals(&self, x: &[f64; N], emit: &mut dyn FnMut(&[f64; N], f64, f64));
}

/// Stopping criteria and damping schedule for [`GaussNewton`].
#[derive(Debug, Clone, Copy)]
pub struct GaussNewton {
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Converged when the weighted cost falls by less than this fraction in a step.
    pub cost_tolerance: f64,
    /// Converged when the correction norm (relative to the state norm) is below this.
    pub step_tolerance: f64,
    /// Initial Levenberg–Marquardt damping. `0.0` is pure Gauss–Newton.
    ///
    /// Dimensionless: the damping is Marquardt's `λ·diag(HᵀWH)`, so `λ` is the fraction by which
    /// each direction's own information is inflated, and its useful range does not depend on the
    /// units the state or the weights are written in.
    pub initial_lambda: f64,
    /// Converged once the weighted cost falls to or below this absolute value — a noiseless problem
    /// drives it to machine zero, where the relative-reduction test stops being informative.
    pub absolute_cost_floor: f64,
}

impl Default for GaussNewton {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            cost_tolerance: 1e-12,
            step_tolerance: 1e-12,
            initial_lambda: 1e-3,
            absolute_cost_floor: 1e-18,
        }
    }
}

/// A solved estimate: the state, its covariance, and the run report.
pub type Estimate<const N: usize> = ([f64; N], [[f64; N]; N], GaussNewtonReport);

/// The outcome of a solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussNewtonReport {
    /// Iterations taken.
    pub iterations: u32,
    /// Final weighted sum of squared residuals.
    pub cost: f64,
    /// Whether a convergence criterion was met (vs hitting the iteration cap).
    pub converged: bool,
}

impl GaussNewton {
    /// Solve `min_x Σ wₖ rₖ(x)²` from `x0`, returning the estimate, its covariance, and a report.
    ///
    /// # Errors
    ///
    /// [`SrifError::RankDeficient`] if the observations never make some state direction observable.
    pub fn solve<const N: usize, M: ResidualModel<N>>(
        &self,
        model: &M,
        x0: &[f64; N],
    ) -> Result<Estimate<N>, SrifError> {
        let mut x = *x0;
        let mut lambda = self.initial_lambda;
        let mut cost = weighted_cost(model, &x);
        let mut converged = false;
        let mut iterations = 0;

        while iterations < self.max_iterations {
            iterations += 1;

            // Linearise at x: fold every observation row into a fresh SRIF, solving Jᵀ..Δx = −r.
            // The information matrix's own diagonal, Σₖ wₖ hₖᵢ², is accumulated in the same pass —
            // it is what makes the damping below scale-free.
            let mut srif = Srif::<N>::new();
            let mut information_diagonal = [0.0; N];
            model.residuals(&x, &mut |h, r, w| {
                srif.add_row(h, -r, w);
                for i in 0..N {
                    information_diagonal[i] += w * h[i] * h[i];
                }
            });

            // Levenberg–Marquardt in Marquardt's scaling: damp direction i by λ times its own
            // information, i.e. add λ·diag(HᵀWH) rather than λI. As a damping "observation" with a
            // zero right-hand side that is one row per direction, of weight λ·Λᵢᵢ.
            if lambda > 0.0 {
                for i in 0..N {
                    let mut e = [0.0; N];
                    e[i] = 1.0;
                    srif.add_row(&e, 0.0, lambda * information_diagonal[i]);
                }
            }

            let delta = srif.solve()?;
            let mut trial = x;
            for i in 0..N {
                trial[i] += delta[i];
            }
            let trial_cost = weighted_cost(model, &trial);

            if trial_cost < cost {
                // accept, relax the damping toward Gauss–Newton
                let step_norm = norm(&delta) / norm(&x).max(1.0);
                let cost_drop = (cost - trial_cost) / cost.max(f64::MIN_POSITIVE);
                x = trial;
                cost = trial_cost;
                lambda = (lambda * 0.3).max(1e-12);
                if step_norm < self.step_tolerance
                    || cost_drop < self.cost_tolerance
                    || cost <= self.absolute_cost_floor
                {
                    converged = true;
                    break;
                }
            } else {
                // No damped step improves the cost: a local minimum is reached, so this is
                // convergence — whether the fit itself is good or bad is reported separately as
                // `cost`. Tighten the damping and retry; give up (still converged) if it saturates.
                lambda *= 10.0;
                if lambda >= 1e12 {
                    converged = true;
                    break;
                }
            }
        }

        // Final covariance from the undamped information at the solution.
        let mut srif = Srif::<N>::new();
        model.residuals(&x, &mut |h, _r, w| srif.add_row(h, 0.0, w));
        let cov = srif.covariance()?;

        Ok((
            x,
            cov,
            GaussNewtonReport {
                iterations,
                cost,
                converged,
            },
        ))
    }
}

fn weighted_cost<const N: usize, M: ResidualModel<N>>(model: &M, x: &[f64; N]) -> f64 {
    let mut c = 0.0;
    model.residuals(x, &mut |_h, r, w| c += w * r * r);
    c
}

fn norm<const N: usize>(v: &[f64; N]) -> f64 {
    let mut s = 0.0;
    for &x in v {
        s += x * x;
    }
    libm::sqrt(s)
}
