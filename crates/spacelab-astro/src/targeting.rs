//! The differential corrector a mission sequence is built on: *vary* these quantities until those
//! goals are *achieved*.
//!
//! Every mission-design tool has one, and it is the piece this workspace had the parts for and never
//! assembled. Halo orbits get their own corrector in [`crate::periodic`], the low-thrust shooter has
//! one welded into [`crate::optimal`], and `spacelab_estimation::GaussNewton` solves the
//! least-squares problem — but all three want the Jacobian handed to them, and the question a
//! mission designer asks does not come with one. "How large must this burn be for the apoapsis to
//! reach 42 164 km" has no analytic derivative available at the call site: the trajectory is
//! whatever a propagator, an atmosphere and a manoeuvre model do to it. The Jacobian has to be
//! *measured*, by running the sequence again with one control nudged.
//!
//! So the sequence is a closure. [`target`] takes `run: FnMut(&[f64; C]) -> Option<[f64; G]>` —
//! anything at all that turns `C` control values into `G` achieved quantities — and drives it to the
//! goals by damped Newton on a finite-difference Jacobian. What that closure contains is the
//! caller's: a Kepler coast and an impulsive burn, a full numerical propagation with a manoeuvre
//! sequence, or a call into another targeter.
//!
//! # Scope, stated rather than implied
//!
//! This is the *numerics* of `Target`/`Vary`/`Achieve`, not the script. A named mission sequence
//! with typed segments, a scripting surface and a UI is a product shape rather than a library one,
//! and it belongs to `OrbitScope`; the library's job is that the corrector underneath it is correct,
//! scale-free and honest about failing. Nothing here knows what a burn is.
//!
//! # Why the residuals are divided by their tolerances
//!
//! A targeter's goals are in unrelated units — a radius in metres beside an inclination in radians —
//! and its controls likewise, a `Δv` in m/s beside a coast duration in seconds. A Newton step formed
//! from the raw Jacobian is then a step in a space with no metric, and a Levenberg–Marquardt `λI`
//! damping term added to it is being compared against numbers whose size is an accident of the units
//! the caller chose. That is the defect #251 measured in the least-squares path, and the fix is the
//! same one: **every residual is divided by its goal's tolerance and every control step is measured
//! in units of that control's perturbation**, so the linear system this module solves is
//! dimensionless. Convergence is then exactly `‖r‖∞ ≤ 1`, the damping is Marquardt's
//! `λ·diag(JᵀJ)`, and neither depends on whether the caller writes kilometres or metres.
//!
//! # Failure modes, deliberately distinguished
//!
//! * The sequence cannot be evaluated at all — a hyperbolic escape, an impact, a solver that gave
//!   up: [`TargeterError::SequenceFailed`], carrying which iteration and which control column was
//!   being perturbed, because "the nominal failed" and "the +δ perturbation failed" are different
//!   problems with different fixes.
//! * A control turns out to have no effect on any goal at this point, so the Jacobian has no rank
//!   to give: [`TargeterError::SingularJacobian`]. GMAT's corrector reports this as a failure to
//!   converge, which sends the user looking for a better initial guess when the actual problem is
//!   that they varied the wrong quantity.
//! * The iteration cap is reached with the goals still missed. That is **not** an error: the
//!   returned [`TargeterReport`] carries `converged: false` along with the worst goal and by how
//!   many tolerances it missed, and the controls come back so the run can be continued or plotted.
//!   A targeter that throws away twenty propagations because the twenty-first would have made it is
//!   not being careful, it is being wasteful.

// Matrix code indexes several arrays by one i/j/k, and the Jacobian is `J` by universal convention.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

/// A quantity the corrector is allowed to vary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Control {
    /// The starting value.
    pub initial: f64,
    /// The finite-difference step, in this control's own units.
    ///
    /// It sets the accuracy of the Jacobian and nothing else, and both directions of getting it
    /// wrong are real: too small and the difference is rounding noise, too large and the derivative
    /// is of a chord rather than a tangent. A tenth of the correction expected on the first
    /// iteration is a serviceable default.
    pub perturbation: f64,
    /// The largest correction a single iteration may apply, in this control's own units.
    ///
    /// This is the guard that keeps a Newton step from throwing the trajectory into the planet on
    /// the way to a solution that exists. [`f64::INFINITY`] disables it.
    pub max_step: f64,
    /// Hard lower bound. The control is clamped into `[lower, upper]` after every step.
    pub lower: f64,
    /// Hard upper bound.
    pub upper: f64,
}

impl Control {
    /// An unbounded control with the given start and finite-difference step, and **no** step cap.
    ///
    /// A default cap would have to be some multiple of the perturbation, and the perturbation is
    /// chosen for the accuracy of a derivative rather than the size of a correction — the two are
    /// unrelated, and a Hohmann burn is routinely five hundred perturbations away from its initial
    /// guess. A cap set that way turns "converges in four iterations" into "did not converge in
    /// twenty-five", which is a false statement about the problem rather than a safe default. The
    /// cap is opt-in through [`with_max_step`](Self::with_max_step), for the sequences that have a
    /// domain edge to stay inside of.
    #[must_use]
    pub const fn new(initial: f64, perturbation: f64) -> Self {
        Self {
            initial,
            perturbation,
            max_step: f64::INFINITY,
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    /// Restrict the control to `[lower, upper]`.
    #[must_use]
    pub const fn bounded(mut self, lower: f64, upper: f64) -> Self {
        self.lower = lower;
        self.upper = upper;
        self
    }

    /// Replace the per-iteration step cap.
    #[must_use]
    pub const fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = max_step;
        self
    }
}

/// A quantity the corrector must drive to a value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Goal {
    /// The value to achieve.
    pub target: f64,
    /// How close counts as achieved, in the goal's own units. Also the scale the residual is
    /// divided by, so it sets the corrector's metric as well as its stopping rule.
    pub tolerance: f64,
}

impl Goal {
    /// A goal with a target and a tolerance.
    #[must_use]
    pub const fn new(target: f64, tolerance: f64) -> Self {
        Self { target, tolerance }
    }
}

/// Stopping rules and damping for [`target`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentialCorrector {
    /// Maximum Newton iterations. Each costs one nominal run plus one (or two) per control.
    pub max_iterations: u32,
    /// Take central rather than forward differences: twice the runs, one order better.
    pub central_differences: bool,
    /// Initial Marquardt damping, dimensionless because the system is scaled. `0.0` is pure Newton.
    pub initial_lambda: f64,
}

impl Default for DifferentialCorrector {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            central_differences: false,
            initial_lambda: 1e-6,
        }
    }
}

/// What a completed run did, converged or not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargeterReport {
    /// Newton iterations taken.
    pub iterations: u32,
    /// How many times the sequence was evaluated — the cost that actually matters, since one
    /// evaluation may be a numerical propagation.
    pub evaluations: u32,
    /// Whether every goal came inside its tolerance.
    pub converged: bool,
    /// Index of the goal furthest from its target, measured in tolerances.
    pub worst_goal: usize,
    /// That goal's miss, in units of its own tolerance. At or below 1 when converged.
    pub worst_miss: f64,
}

/// Why a run could not be completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargeterError {
    /// The sequence returned `None`.
    SequenceFailed {
        /// The iteration it failed on.
        iteration: u32,
        /// The control being perturbed, or `None` for the nominal run.
        control: Option<usize>,
    },
    /// The Jacobian lost rank: at least one control moves no goal at this point.
    SingularJacobian {
        /// The iteration it went singular on.
        iteration: u32,
    },
    /// A perturbation or a tolerance was not positive, or a control started outside its bounds.
    InvalidSetup,
}

/// Drive `run`'s outputs to `goals` by varying `controls`.
///
/// Returns the final control values and a report. Reaching the iteration cap without achieving the
/// goals is reported through [`TargeterReport::converged`] rather than as an error.
///
/// # Errors
///
/// [`TargeterError::InvalidSetup`] for a non-positive perturbation or tolerance, or a control whose
/// initial value is outside its own bounds; [`TargeterError::SequenceFailed`] when `run` returns
/// `None`; [`TargeterError::SingularJacobian`] when no step can be formed.
pub fn target<const C: usize, const G: usize, F>(
    corrector: &DifferentialCorrector,
    controls: &[Control; C],
    goals: &[Goal; G],
    mut run: F,
) -> Result<([f64; C], TargeterReport), TargeterError>
where
    F: FnMut(&[f64; C]) -> Option<[f64; G]>,
{
    for control in controls {
        let usable = control.perturbation > 0.0
            && control.initial >= control.lower
            && control.initial <= control.upper;
        if !usable {
            return Err(TargeterError::InvalidSetup);
        }
    }
    // `> 0.0` written positively so a NaN tolerance falls out as unusable rather than through a
    // negated comparison that reads as if NaN were fine.
    if !goals.iter().all(|goal| goal.tolerance > 0.0) {
        return Err(TargeterError::InvalidSetup);
    }

    let mut x: [f64; C] = core::array::from_fn(|j| controls[j].initial);
    let mut lambda = corrector.initial_lambda;
    let mut evaluations = 0_u32;
    let mut iterations = 0_u32;

    // Scaled residuals: rᵢ = (achieved − target)/tolerance, so ‖r‖∞ ≤ 1 is convergence.
    let mut residual = {
        let achieved =
            evaluate(&mut run, &x, &mut evaluations).ok_or(TargeterError::SequenceFailed {
                iteration: 0,
                control: None,
            })?;
        scale(&achieved, goals)
    };
    let (mut worst_goal, mut worst_miss) = worst(&residual);

    while worst_miss > 1.0 && iterations < corrector.max_iterations {
        iterations += 1;

        let jacobian = differences(
            corrector,
            controls,
            goals,
            &mut run,
            &x,
            &residual,
            iterations,
            &mut evaluations,
        )?;

        // Marquardt: solve (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr, retrying at higher damping while the step
        // fails to reduce ‖r‖. Rejected steps cost an evaluation each, which is why they are
        // counted.
        let mut accepted = false;
        for _ in 0..MAX_DAMPING_RETRIES {
            let Some(delta) = damped_step(&jacobian, &residual, lambda) else {
                return Err(TargeterError::SingularJacobian {
                    iteration: iterations,
                });
            };
            let mut trial = x;
            for j in 0..C {
                let raw = delta[j] * controls[j].perturbation;
                let capped = raw.clamp(-controls[j].max_step, controls[j].max_step);
                trial[j] = (trial[j] + capped).clamp(controls[j].lower, controls[j].upper);
            }
            let Some(achieved) = evaluate(&mut run, &trial, &mut evaluations) else {
                // A step into a region the sequence cannot evaluate is a step that was too long,
                // which is the case damping exists for.
                lambda = increase_damping(lambda);
                continue;
            };
            let candidate = scale(&achieved, goals);
            let (goal, miss) = worst(&candidate);
            if miss < worst_miss {
                x = trial;
                residual = candidate;
                worst_goal = goal;
                worst_miss = miss;
                lambda *= 0.1;
                accepted = true;
                break;
            }
            lambda = increase_damping(lambda);
        }
        if !accepted {
            break;
        }
    }

    Ok((
        x,
        TargeterReport {
            iterations,
            evaluations,
            converged: worst_miss <= 1.0,
            worst_goal,
            worst_miss,
        },
    ))
}

/// How many times the damping may be raised within one iteration before the run is abandoned.
///
/// Ten decades of `λ` is the point past which the step is a rounding-error multiple of the state and
/// another decade cannot help; stopping there and reporting `converged: false` is more useful than
/// spending propagations to reach the same conclusion.
const MAX_DAMPING_RETRIES: u32 = 10;

/// Raise the Marquardt parameter after a rejected step.
///
/// The `else` branch is the reason this is a function rather than a multiplication. A corrector
/// configured with `initial_lambda = 0.0` is a pure Gauss-Newton one, and multiplying zero by ten
/// leaves it Gauss-Newton for ever: the retries would repeat the identical rejected step until the
/// cap and the corrector would report failure on a problem damping would have solved. The seed
/// value is small enough to perturb the step rather than replace it.
const fn increase_damping(lambda: f64) -> f64 {
    if lambda > 0.0 { lambda * 10.0 } else { 1e-3 }
}

/// The scaled Jacobian: column `j` is `∂r/∂(xⱼ/perturbationⱼ)`, by forward or central differences.
///
/// `residual` is the already-evaluated nominal, which the forward-difference branch reuses rather
/// than paying for again — the saving is one whole sequence run per iteration.
#[allow(clippy::too_many_arguments)]
fn differences<const C: usize, const G: usize, F>(
    corrector: &DifferentialCorrector,
    controls: &[Control; C],
    goals: &[Goal; G],
    run: &mut F,
    x: &[f64; C],
    residual: &[f64; G],
    iteration: u32,
    evaluations: &mut u32,
) -> Result<[[f64; C]; G], TargeterError>
where
    F: FnMut(&[f64; C]) -> Option<[f64; G]>,
{
    let mut jacobian = [[0.0; C]; G];
    for j in 0..C {
        let step = controls[j].perturbation;
        let mut plus = *x;
        plus[j] += step;
        let forward = evaluate(run, &plus, evaluations).ok_or(TargeterError::SequenceFailed {
            iteration,
            control: Some(j),
        })?;
        let forward = scale(&forward, goals);
        if corrector.central_differences {
            let mut minus = *x;
            minus[j] -= step;
            let backward =
                evaluate(run, &minus, evaluations).ok_or(TargeterError::SequenceFailed {
                    iteration,
                    control: Some(j),
                })?;
            let backward = scale(&backward, goals);
            for i in 0..G {
                jacobian[i][j] = (forward[i] - backward[i]) / 2.0;
            }
        } else {
            for i in 0..G {
                jacobian[i][j] = forward[i] - residual[i];
            }
        }
    }
    Ok(jacobian)
}

fn evaluate<const C: usize, const G: usize, F>(
    run: &mut F,
    x: &[f64; C],
    evaluations: &mut u32,
) -> Option<[f64; G]>
where
    F: FnMut(&[f64; C]) -> Option<[f64; G]>,
{
    *evaluations += 1;
    run(x)
}

fn scale<const G: usize>(achieved: &[f64; G], goals: &[Goal; G]) -> [f64; G] {
    core::array::from_fn(|i| (achieved[i] - goals[i].target) / goals[i].tolerance)
}

fn worst<const G: usize>(residual: &[f64; G]) -> (usize, f64) {
    let mut index = 0;
    let mut value = 0.0_f64;
    for (i, r) in residual.iter().enumerate() {
        if libm::fabs(*r) > value {
            value = libm::fabs(*r);
            index = i;
        }
    }
    (index, value)
}

/// Solve `(JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr` by Cholesky, returning `None` if the matrix is not
/// positive-definite — which for a Gram matrix means the Jacobian is rank-deficient.
///
/// The normal equations are formed here rather than avoided, and that is a deliberate departure
/// from the rule the estimation crate is built on. The reason is that this Jacobian is *already*
/// only as accurate as a finite difference: its entries carry a relative error near `√ε` from the
/// differencing itself, so the `ε → √ε` that squaring the condition number costs is below the noise
/// the columns already have. `C` here is a handful of controls, not a state vector, and the
/// alternative — carrying an orthogonal factorisation for a 3×3 — buys nothing.
fn damped_step<const C: usize, const G: usize>(
    jacobian: &[[f64; C]; G],
    residual: &[f64; G],
    lambda: f64,
) -> Option<[f64; C]> {
    let mut normal = [[0.0; C]; C];
    let mut gradient = [0.0; C];
    for a in 0..C {
        for b in 0..C {
            let mut acc = 0.0;
            for i in 0..G {
                acc += jacobian[i][a] * jacobian[i][b];
            }
            normal[a][b] = acc;
        }
        let mut acc = 0.0;
        for i in 0..G {
            acc -= jacobian[i][a] * residual[i];
        }
        gradient[a] = acc;
    }
    for a in 0..C {
        normal[a][a] += lambda * normal[a][a];
    }

    // Lower Cholesky, then two triangular solves.
    let mut l = [[0.0; C]; C];
    for i in 0..C {
        for j in 0..=i {
            let mut sum = normal[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                // A non-positive (or NaN) pivot is a Gram matrix that is not positive-definite,
                // which for `JᵀJ` means the Jacobian has lost rank.
                if sum <= 0.0 || sum.is_nan() {
                    return None;
                }
                l[i][j] = libm::sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    let mut y = [0.0; C];
    for i in 0..C {
        let mut sum = gradient[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut delta = [0.0; C];
    for i in (0..C).rev() {
        let mut sum = y[i];
        for k in (i + 1)..C {
            sum -= l[k][i] * delta[k];
        }
        delta[i] = sum / l[i][i];
    }
    if delta.iter().any(|d| !d.is_finite()) {
        return None;
    }
    Some(delta)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use spacelab_core::Vec3;

    use super::*;
    use crate::kepler::propagate;

    const MU_EARTH: f64 = 3.986_004_418e14;

    #[test]
    fn a_scalar_burn_reaches_the_apoapsis_hohmann_predicts() {
        // The canonical mission-design question, and the one case where the answer is on paper:
        // from a circular orbit at r1, how large a prograde impulse puts apoapsis at r2? Hohmann
        // says Δv = √(μ/r1)·(√(2r2/(r1+r2)) − 1), and the targeter is told none of that — it sees a
        // black box that returns an apoapsis radius.
        let r1 = 6_678e3;
        let r2 = 42_164e3;
        let v1 = (MU_EARTH / r1).sqrt();
        let closed_form = v1 * ((2.0 * r2 / (r1 + r2)).sqrt() - 1.0);

        let (solution, report) = target(
            &DifferentialCorrector::default(),
            &[Control::new(1_000.0, 1.0)],
            &[Goal::new(r2, 1.0)],
            |x| {
                let state = [r1, 0.0, 0.0, 0.0, v1 + x[0], 0.0];
                Some([apoapsis_radius(&state)])
            },
        )
        .expect("the sequence is evaluable everywhere on this line");

        assert!(report.converged, "{report:?}");
        assert!(
            (solution[0] - closed_form).abs() < 1e-6,
            "targeted {} against Hohmann {closed_form}",
            solution[0]
        );
        // A one-control Newton on a smooth scalar should not need many passes. Seven measured,
        // from an initial guess 478 m/s short of the answer.
        assert!(report.iterations <= 10, "{report:?}");
    }

    #[test]
    fn three_controls_reproduce_the_lambert_arc_they_were_never_told_about() {
        // The strongest check available without an oracle file: vary all three components of an
        // impulse so that the position after a fixed flight time is a specified one. That problem
        // has a closed-form answer — it is Lambert's — and `lambert()` solves it by Izzo's
        // algorithm, which shares no line with a finite-difference Newton shooter.
        let r1 = Vec3::new(7_000e3, 0.0, 0.0);
        let v1 = Vec3::new(0.0, (MU_EARTH / 7_000e3).sqrt(), 0.0);
        let tof = 3_000.0;
        // Somewhere the transfer actually reaches: propagate the departure orbit and take the
        // point, then displace it so the arc is not the trivial one.
        let (coasted, _) = propagate(r1, v1, MU_EARTH, tof);
        let goal_position = Vec3::new(coasted.x + 300e3, coasted.y - 200e3, coasted.z + 150e3);

        let (departure, _) = crate::lambert(r1, goal_position, tof, MU_EARTH, true)
            .expect("Lambert solves this geometry");
        let expected = [departure.x - v1.x, departure.y - v1.y, departure.z - v1.z];

        let (solution, report) = target(
            &DifferentialCorrector {
                central_differences: true,
                ..DifferentialCorrector::default()
            },
            &[
                Control::new(0.0, 0.5),
                Control::new(0.0, 0.5),
                Control::new(0.0, 0.5),
            ],
            &[
                Goal::new(goal_position.x, 1e-2),
                Goal::new(goal_position.y, 1e-2),
                Goal::new(goal_position.z, 1e-2),
            ],
            |x| {
                let v = Vec3::new(v1.x + x[0], v1.y + x[1], v1.z + x[2]);
                let (r, _) = propagate(r1, v, MU_EARTH, tof);
                Some([r.x, r.y, r.z])
            },
        )
        .expect("the shooter stays on elliptic arcs here");

        assert!(report.converged, "{report:?}");
        for i in 0..3 {
            let miss = (solution[i] - expected[i]).abs();
            println!(
                "MEASURED component {i} shooter {} Lambert {} miss {miss:.3e}",
                solution[i], expected[i]
            );
            // A centimetre of position tolerance over a 3 000 s arc is worth a few micrometres per
            // second of departure velocity; 1e-4 leaves two decades of room above that floor and is
            // still four decades below the 187 m/s the components carry.
            assert!(
                miss < 1e-4,
                "component {i}: shooter {} against Lambert {}",
                solution[i],
                expected[i]
            );
        }
    }

    #[test]
    fn a_control_that_moves_nothing_is_reported_as_singular_rather_than_unconverged() {
        // The failure GMAT reports as "did not converge", which sends the user hunting for a better
        // guess when the actual fault is that the wrong quantity was varied. Here the second
        // control is not read at all.
        let outcome = target(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 1e-3), Control::new(0.0, 1e-3)],
            &[Goal::new(5.0, 1e-9), Goal::new(2.0, 1e-9)],
            |x| Some([x[0], x[0] * 2.0]),
        );
        assert_eq!(
            outcome,
            Err(TargeterError::SingularJacobian { iteration: 1 }),
            "{outcome:?}"
        );
    }

    #[test]
    fn a_sequence_that_cannot_run_names_the_column_it_failed_on() {
        // The nominal is fine and the perturbed run is not, which is the signature of a
        // perturbation stepping over a domain edge — a distinct fault from a bad initial guess, and
        // the error says which.
        let outcome = target::<1, 1, _>(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 1.0)],
            &[Goal::new(10.0, 1e-6)],
            |x| (x[0] <= 0.0).then_some([0.0]),
        );
        assert_eq!(
            outcome,
            Err(TargeterError::SequenceFailed {
                iteration: 1,
                control: Some(0)
            }),
            "{outcome:?}"
        );
    }

    #[test]
    fn a_bound_is_honoured_even_when_the_goal_lies_past_it() {
        // A control clamped short of its solution must stop at the bound and report the miss,
        // rather than walk through it or spin to the iteration cap pretending progress.
        let (solution, report) = target(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 0.1).bounded(-1.0, 1.0)],
            &[Goal::new(5.0, 1e-6)],
            |x| Some([x[0]]),
        )
        .expect("evaluable everywhere");
        assert!(!report.converged, "{report:?}");
        assert!(
            (solution[0] - 1.0).abs() < 1e-12,
            "clamped to {}",
            solution[0]
        );
        // Missed by four units against a 1e-6 tolerance.
        assert!((report.worst_miss - 4.0 / 1e-6).abs() < 1.0, "{report:?}");
    }

    #[test]
    fn the_step_cap_bounds_the_first_iteration_rather_than_the_answer() {
        // A max_step smaller than the required correction must not prevent convergence, only slow
        // it: the corrector should walk there in ceil(5/0.5) = 10 capped steps.
        let (solution, report) = target(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 0.1).with_max_step(0.5)],
            &[Goal::new(5.0, 1e-9)],
            |x| Some([x[0]]),
        )
        .expect("evaluable everywhere");
        assert!(report.converged, "{report:?}");
        assert!((solution[0] - 5.0).abs() < 1e-9, "{}", solution[0]);
        assert!(report.iterations >= 10, "{report:?}");
    }

    #[test]
    fn scaling_the_goal_units_does_not_change_the_iteration_count() {
        // The claim the tolerance-scaling exists to support. The same problem posed in metres and
        // in kilometres — every goal, tolerance and control scaled by a thousand — must take the
        // same path, because the linear system the corrector forms is identical.
        let solve = |scale: f64| {
            target(
                &DifferentialCorrector::default(),
                &[Control::new(0.0, 1.0 * scale)],
                &[Goal::new(1_234.0 * scale, 1e-3 * scale)],
                |x| Some([x[0] * x[0] / (1_000.0 * scale) + x[0]]),
            )
            .expect("evaluable")
        };
        let (metres, metres_report) = solve(1.0);
        let (kilometres, kilometres_report) = solve(1e-3);
        assert_eq!(metres_report.iterations, kilometres_report.iterations);
        assert_eq!(metres_report.evaluations, kilometres_report.evaluations);
        assert!(
            (metres[0] * 1e-3 / kilometres[0] - 1.0).abs() < 1e-9,
            "{} vs {}",
            metres[0],
            kilometres[0]
        );
    }

    #[test]
    fn an_overdetermined_system_settles_at_the_least_squares_point() {
        // Two goals, one control, and no value of the control satisfies both. The corrector must
        // not diverge or go singular; it must land on the least-squares compromise of the scaled
        // residuals — here x = 1.5 — and report the miss honestly.
        let (solution, report) = target(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 1e-2)],
            &[Goal::new(1.0, 1.0e-6), Goal::new(2.0, 1.0e-6)],
            |x| Some([x[0], x[0]]),
        )
        .expect("evaluable everywhere");
        assert!(!report.converged, "{report:?}");
        assert!((solution[0] - 1.5).abs() < 1e-6, "{}", solution[0]);
    }

    #[test]
    fn a_non_positive_perturbation_or_tolerance_is_refused_before_anything_runs() {
        let mut ran = false;
        let outcome = target::<1, 1, _>(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 0.0)],
            &[Goal::new(1.0, 1.0)],
            |_| {
                ran = true;
                Some([0.0])
            },
        );
        assert_eq!(outcome, Err(TargeterError::InvalidSetup));
        assert!(
            !ran,
            "the sequence must not be evaluated on an invalid setup"
        );

        let outcome = target::<1, 1, _>(
            &DifferentialCorrector::default(),
            &[Control::new(0.0, 1.0)],
            &[Goal::new(1.0, -1.0)],
            |_| Some([0.0]),
        );
        assert_eq!(outcome, Err(TargeterError::InvalidSetup));

        // And a start outside its own bounds, which would otherwise be silently clamped on the
        // first accepted step and make the reported solution unrelated to the requested one.
        let outcome = target::<1, 1, _>(
            &DifferentialCorrector::default(),
            &[Control::new(5.0, 1.0).bounded(-1.0, 1.0)],
            &[Goal::new(1.0, 1.0)],
            |_| Some([0.0]),
        );
        assert_eq!(outcome, Err(TargeterError::InvalidSetup));
    }

    #[test]
    fn a_setup_already_at_its_goals_costs_one_evaluation_and_no_iterations() {
        let (solution, report) = target(
            &DifferentialCorrector::default(),
            &[Control::new(3.0, 0.1)],
            &[Goal::new(3.0, 1e-6)],
            |x| Some([x[0]]),
        )
        .expect("evaluable");
        assert!(report.converged && report.iterations == 0, "{report:?}");
        assert_eq!(report.evaluations, 1);
        assert!((solution[0] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn a_goal_with_a_zero_or_negative_tolerance_is_refused_rather_than_divided_by() {
        // The tolerance is the residual's denominator, so admitting zero would not merely accept a
        // stricter goal -- it would make every residual infinite and the corrector's own metric
        // meaningless. NaN goes the same way, through the positively-written predicate.
        for tolerance in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                target(
                    &DifferentialCorrector::default(),
                    &[Control::new(0.0, 0.1)],
                    &[Goal::new(1.0, tolerance)],
                    |x| Some([x[0]]),
                ),
                Err(TargeterError::InvalidSetup),
                "tolerance {tolerance} was accepted"
            );
        }
    }

    #[test]
    fn a_miss_of_exactly_one_tolerance_is_converged_and_costs_no_iteration() {
        // The stopping rule is `worst_miss > 1.0`, and the boundary is the whole content of it: a
        // state sitting exactly on its tolerance has achieved the goal by the definition the
        // tolerance gives, and iterating on it would be work with nothing to show.
        let (_, report) = target(
            &DifferentialCorrector::default(),
            &[Control::new(1.0, 0.1)],
            &[Goal::new(0.0, 1.0)],
            |x| Some([x[0]]),
        )
        .expect("evaluable");
        assert!(report.converged, "{report:?}");
        assert_eq!(report.iterations, 0, "{report:?}");
        assert!((report.worst_miss - 1.0).abs() < 1e-15, "{report:?}");
    }

    #[test]
    fn the_iteration_cap_is_exact_and_the_report_says_it_was_not_reached() {
        // A goal the sequence cannot reach at any control value. The corrector must spend exactly
        // the iterations it was allowed -- one more would be a cap that does not bind, one fewer a
        // corrector that gives up early -- and must say it did not converge.
        for cap in [1_u32, 2, 5] {
            let corrector = DifferentialCorrector {
                max_iterations: cap,
                ..DifferentialCorrector::default()
            };
            let (_, report) = target(
                &corrector,
                // The step cap makes the goal unreachable in the iterations allowed while keeping
                // every step an improvement, so the loop stops on the cap and not on a rejection.
                &[Control::new(0.0, 1e-3).with_max_step(1.0)],
                &[Goal::new(1.0e6, 1e-6)],
                |x| Some([x[0]]),
            )
            .expect("evaluable");
            assert!(!report.converged, "cap {cap}: {report:?}");
            assert_eq!(report.iterations, cap, "cap {cap}: {report:?}");
        }
    }

    #[test]
    fn a_step_that_does_not_improve_the_worst_miss_is_not_a_step() {
        // The acceptance test is strict, and it has to be: a sequence that keeps returning the same
        // miss would otherwise be walked for ever at shrinking damping, each move accepted because
        // it was not worse. Here the sequence answers honestly while the Jacobian is measured and
        // then returns a mirrored value -- same miss, opposite sign -- for every trial step, so a
        // corrector that accepted equal misses would run to its cap instead of stopping at one
        // iteration.
        let mut probing = 0_u32;
        let (_, report) = target(
            &DifferentialCorrector {
                max_iterations: 20,
                ..DifferentialCorrector::default()
            },
            &[Control::new(2.0, 1e-3)],
            &[Goal::new(0.0, 1.0)],
            |x| {
                probing += 1;
                // Calls 1 and 2 are the nominal and the perturbed evaluation the Jacobian needs;
                // every trial after them mirrors the start, so the miss is identical, never smaller.
                if probing <= 2 {
                    Some([x[0]])
                } else {
                    Some([-2.0])
                }
            },
        )
        .expect("evaluable");
        assert!(!report.converged, "{report:?}");
        assert_eq!(report.iterations, 1, "{report:?}");
        assert!((report.worst_miss - 2.0).abs() < 1e-15, "{report:?}");
        // The nominal evaluation, one for the forward difference, and the ten trials the damping
        // ladder allows before it gives up.
        assert_eq!(report.evaluations, 1 + 1 + 10, "{report:?}");
    }

    #[test]
    fn the_worst_goal_reported_on_a_tie_is_the_first_one() {
        // Two goals missed by exactly the same amount. Which index the report names is arbitrary
        // mathematics and not arbitrary API: a caller that logs `worst_goal` needs it stable, and
        // the first is the only choice that does not depend on iteration order.
        let corrector = DifferentialCorrector {
            max_iterations: 0,
            ..DifferentialCorrector::default()
        };
        let (_, report) = target(
            &corrector,
            &[Control::new(0.0, 0.1)],
            &[Goal::new(-5.0, 1.0), Goal::new(5.0, 1.0)],
            |_| Some([0.0, 0.0]),
        )
        .expect("evaluable");
        assert_eq!(report.worst_goal, 0, "{report:?}");
        assert!((report.worst_miss - 5.0).abs() < 1e-15, "{report:?}");
    }

    #[test]
    fn raising_the_damping_from_zero_seeds_it_rather_than_multiplying_it() {
        // `increase_damping` exists for this branch alone. A pure Gauss-Newton corrector starts at
        // zero, and ten times zero is still zero: without the seed a rejected step would be retried
        // identically until the cap.
        assert!((increase_damping(0.0) - 1e-3).abs() < 1e-18);
        assert!((increase_damping(-1.0) - 1e-3).abs() < 1e-18);
        assert!((increase_damping(1e-6) - 1e-5).abs() < 1e-20);
        assert!((increase_damping(1e-3) - 1e-2).abs() < 1e-17);
        // Strictly increasing on the positive branch, which is what makes the retry ladder a ladder.
        assert!(increase_damping(1e-6) > 1e-6);
    }

    #[test]
    fn a_rank_deficient_jacobian_is_reported_rather_than_solved() {
        // Two controls the sequence responds to identically: `JᵀJ` is singular, the Marquardt
        // damping scales the diagonal it already has and cannot restore rank, and the Cholesky must
        // say so rather than return a step built from a division by a zero pivot.
        let outcome = target(
            &DifferentialCorrector {
                initial_lambda: 0.0,
                ..DifferentialCorrector::default()
            },
            &[Control::new(0.0, 1e-3), Control::new(0.0, 1e-3)],
            &[Goal::new(1.0, 1e-9)],
            |x| Some([x[0] + x[1]]),
        );
        assert_eq!(
            outcome,
            Err(TargeterError::SingularJacobian { iteration: 1 }),
            "{outcome:?}"
        );
    }

    /// Apoapsis radius of a two-body state, from the eccentricity vector and the semi-latus rectum.
    fn apoapsis_radius(state: &[f64; 6]) -> f64 {
        let r = [state[0], state[1], state[2]];
        let v = [state[3], state[4], state[5]];
        let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
        let e: [f64; 3] =
            core::array::from_fn(|i| ((v2 - MU_EARTH / rn) * r[i] - rv * v[i]) / MU_EARTH);
        let ecc = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        let energy = v2 / 2.0 - MU_EARTH / rn;
        let a = -MU_EARTH / (2.0 * energy);
        a * (1.0 + ecc)
    }
}
