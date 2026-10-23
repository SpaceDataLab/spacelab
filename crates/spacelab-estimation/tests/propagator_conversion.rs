//! The analytic-Jacobian converter, against the finite-difference one on the same problem.
//!
//! The unit tests in `src/converter.rs` drive both Jacobian sources through a propagator that is
//! linear in its parameters, which proves the plumbing and nothing about orbits. This drives them
//! through a real one: a numerically integrated arc under point-mass gravity and the Sun, whose
//! partials come out of the variational equations integrated alongside the state. The parameters
//! are the epoch state, so `∂state(t)/∂parameters` *is* the state-transition matrix, and the two
//! converters differ in exactly one thing — where those partials came from.
//!
//! ## Measured
//!
//! A 700 km LEO, 20 position samples on a 300 s grid over 5700 s, fitted from a guess displaced by
//! 5 km and 5 m/s in every component:
//!
//! | | iterations | post-fit position RMS | error against the truth state |
//! |---|---|---|---|
//! | analytic (variational equations) | 6 | 3.14e-8 m | 1.02e-8 m, 1.59e-11 m/s |
//! | central differences | 6 | 2.53e-8 m | 1.58e-8 m, 1.51e-11 m/s |
//!
//! The two land on the same parameters to **5.6e-9 m** and **2.7e-12 m/s** — twelve decades below
//! the 5 km they started apart. That is the point of running both: they share the residuals, the
//! damping and the stopping rule, so a disagreement between them could only be the Jacobian, and
//! there is none to find at the level the `rtol = 1e-9` integrator can resolve.
//!
//! The formal sigmas of the analytic fit, one metre of assumed position uncertainty: 0.22, 0.42 and
//! 0.45 m on the epoch position and 5.1e-4, 3.1e-4 and 2.9e-4 m/s on the epoch velocity.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // `r`, `v`, `g`, `s` are the accelerations of the force model, one letter each because that is
    // how the equations of motion are written everywhere else in this workspace.
    clippy::many_single_char_names
)]

use spacelab_autodiff::Scalar;
use spacelab_estimation::{
    DifferentiablePropagator, PropagatorBuilder, PropagatorConverter, SampledPropagator,
    TrajectorySample,
};
use spacelab_propagation::dop853::Dop853Dense;
use spacelab_propagation::driver::{Controls, Trajectory, integrate};
use spacelab_propagation::ephemeris::AnalyticSun;
use spacelab_propagation::forces::{Acceleration, PointMass, ThirdBody};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, Tolerance};

/// Gravitational parameter of the Earth, m³/s² (EGM2008 / WGS-84).
const MU_EARTH: f64 = 3.986_004_418e14;
/// Gravitational parameter of the Sun, m³/s² (IAU 2015 nominal / DE440).
const MU_SUN: f64 = 1.327_124_400_18e20;

/// Six states plus their 6×6 state-transition matrix.
const AUGMENTED: usize = 42;

/// Point-mass Earth plus the Sun: non-Keplerian, and needs no data file to evaluate.
struct Dynamics {
    earth: PointMass,
    sun: ThirdBody<AnalyticSun>,
}

impl ScalarSystem<6> for Dynamics {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 6]) -> [S; 6] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let g = self.earth.acceleration(t, &r, &v);
        let s = self.sun.acceleration(t, &r, &v);
        [y[3], y[4], y[5], g[0] + s[0], g[1] + s[1], g[2] + s[2]]
    }
}

/// One integrated arc, dense over its span, carrying its own state-transition matrix.
struct Arc {
    trajectory: Trajectory<AUGMENTED, Dop853Dense<AUGMENTED>>,
    span: f64,
}

impl SampledPropagator for Arc {
    fn state_at(&self, t: f64) -> Option<[f64; 6]> {
        // `Trajectory::eval` extrapolates the nearest segment outside its span rather than
        // refusing, so the refusal has to be here: an extrapolated dense output is not a state.
        if !(0.0..=self.span).contains(&t) {
            return None;
        }
        Some(split::<6, AUGMENTED>(&self.trajectory.eval(t)).0)
    }
}

impl DifferentiablePropagator<6> for Arc {
    fn state_and_partials(&self, t: f64) -> Option<([f64; 6], [[f64; 6]; 6])> {
        if !(0.0..=self.span).contains(&t) {
            return None;
        }
        Some(split::<6, AUGMENTED>(&self.trajectory.eval(t)))
    }
}

/// The parameter vector is the epoch state, so building is integrating.
struct ArcBuilder<'a> {
    dynamics: &'a Dynamics,
    span: f64,
}

impl PropagatorBuilder<6> for ArcBuilder<'_> {
    type Propagator = Arc;

    fn build(&self, parameters: &[f64; 6]) -> Option<Arc> {
        let variational = Variational::<6, _>::new(self.dynamics);
        let trajectory = integrate(
            &Dop853,
            &variational,
            0.0,
            &augment::<6, AUGMENTED>(parameters),
            self.span,
            10.0,
            Controls::new(Tolerance::new(1e-9, 1e-12), 1e-3),
        )
        .ok()?;
        Some(Arc {
            trajectory,
            span: self.span,
        })
    }
}

/// A 700 km circular LEO, inclined enough that no component of the state is degenerate.
const TRUTH: [f64; 6] = [7_078_137.0, 0.0, 0.0, 0.0, 4_712.983_5, 5_450.0];

/// Just under one revolution, sampled every 300 s.
const SPAN: f64 = 5_700.0;
const STEP: f64 = 300.0;

fn dynamics() -> Dynamics {
    Dynamics {
        earth: PointMass { mu: MU_EARTH },
        sun: ThirdBody {
            mu: MU_SUN,
            ephemeris: AnalyticSun {
                epoch_j2000_seconds: 0.0,
            },
        },
    }
}

#[test]
fn the_two_jacobian_sources_fit_the_same_arc_to_the_same_state() {
    let dynamics = dynamics();
    let builder = ArcBuilder {
        dynamics: &dynamics,
        span: SPAN,
    };
    let truth_arc = builder.build(&TRUTH).expect("the truth arc integrates");
    let samples: Vec<TrajectorySample> = (0..=(SPAN / STEP) as usize)
        .map(|i| {
            let t = i as f64 * STEP;
            TrajectorySample {
                t,
                state: truth_arc.state_at(t).expect("inside the span"),
            }
        })
        .collect();
    assert_eq!(samples.len(), 20);

    // Five kilometres and five metres per second out in every component: a guess, not a nudge.
    let guess: [f64; 6] = core::array::from_fn(|k| TRUTH[k] + if k < 3 { 5_000.0 } else { 5.0 });

    let converter = PropagatorConverter {
        position_sigma: 1.0,
        ..PropagatorConverter::default()
    };
    let analytic = converter
        .fit_by_jacobian(&builder, &samples, &guess)
        .expect("the arc is observable in all six state components");
    let steps = [1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3];
    let difference = converter
        .fit_by_finite_difference(&builder, &samples, &guess, &steps)
        .expect("the arc is observable in all six state components");

    let error = |fit: &[f64; 6]| {
        let position = (0..3)
            .map(|k| (fit[k] - TRUTH[k]).abs())
            .fold(0.0_f64, f64::max);
        let velocity = (3..6)
            .map(|k| (fit[k] - TRUTH[k]).abs())
            .fold(0.0_f64, f64::max);
        (position, velocity)
    };
    let (analytic_position, analytic_velocity) = error(&analytic.parameters);
    let (difference_position, difference_velocity) = error(&difference.parameters);
    let spread_position = (0..3)
        .map(|k| (analytic.parameters[k] - difference.parameters[k]).abs())
        .fold(0.0_f64, f64::max);
    let spread_velocity = (3..6)
        .map(|k| (analytic.parameters[k] - difference.parameters[k]).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "{} samples over {SPAN} s\n\
         analytic     {} iterations, post-fit RMS {:.4e} m, worst {:.4e} m, \
         epoch error {analytic_position:.3e} m / {analytic_velocity:.3e} m/s\n\
         differences  {} iterations, post-fit RMS {:.4e} m, worst {:.4e} m, \
         epoch error {difference_position:.3e} m / {difference_velocity:.3e} m/s\n\
         the two agree to {spread_position:.3e} m and {spread_velocity:.3e} m/s",
        samples.len(),
        analytic.report.iterations,
        analytic.residuals.position_rms,
        analytic.residuals.worst_position,
        difference.report.iterations,
        difference.residuals.position_rms,
        difference.residuals.worst_position,
    );

    assert!(analytic.report.converged && difference.report.converged);
    // Measured: 1.02e-8 m (analytic) and 1.58e-8 m (differences) from the truth state, at post-fit
    // RMS of 3.14e-8 and 2.53e-8 m. The bound is the integrator's own tolerance, which is what
    // actually limits this: with `rtol = 1e-9` on a 7 Mm radius the arc carries millimetres.
    assert!(analytic_position < 1e-3, "analytic {analytic_position:e} m");
    assert!(
        difference_position < 1e-3,
        "differences {difference_position:e} m"
    );
    assert!(
        spread_position < 1e-3 && spread_velocity < 1e-6,
        "the two Jacobian sources disagree by {spread_position:e} m and {spread_velocity:e} m/s"
    );
    assert!(analytic.residuals.position_rms < 1e-3);
    assert!(difference.residuals.position_rms < 1e-3);
}

/// The converter reports the covariance the arc actually supports, not a number it made up.
///
/// A one-revolution position-only arc determines the epoch state well in every direction, and the
/// formal sigmas say by how much. They are worth asserting because a covariance is the one output
/// nothing else in the fit cross-checks: the parameters can be verified against a known truth, the
/// residuals against the samples, and the covariance against neither.
#[test]
fn the_reported_covariance_is_positive_definite_and_scaled_by_the_sigma() {
    let dynamics = dynamics();
    let builder = ArcBuilder {
        dynamics: &dynamics,
        span: SPAN,
    };
    let truth_arc = builder.build(&TRUTH).expect("the truth arc integrates");
    let samples: Vec<TrajectorySample> = (0..=(SPAN / STEP) as usize)
        .map(|i| {
            let t = i as f64 * STEP;
            TrajectorySample {
                t,
                state: truth_arc.state_at(t).expect("inside the span"),
            }
        })
        .collect();

    let one_metre = PropagatorConverter {
        position_sigma: 1.0,
        ..PropagatorConverter::default()
    }
    .fit_by_jacobian(&builder, &samples, &TRUTH)
    .expect("observable");
    let ten_metres = PropagatorConverter {
        position_sigma: 10.0,
        ..PropagatorConverter::default()
    }
    .fit_by_jacobian(&builder, &samples, &TRUTH)
    .expect("observable");

    for k in 0..6 {
        let sigma = one_metre.covariance[k][k].sqrt();
        println!("  parameter {k}: sigma {sigma:.6e}");
        assert!(sigma > 0.0 && sigma.is_finite(), "sigma {sigma}");
        // Weights enter as 1/σ², so the covariance scales as σ² exactly.
        let scaled = ten_metres.covariance[k][k] / one_metre.covariance[k][k];
        assert!(
            (scaled - 100.0).abs() < 1e-6,
            "covariance scaled by {scaled} for a tenfold sigma"
        );
    }
    // Symmetry is a property of `R⁻¹R⁻ᵀ` and would be lost by an indexing error in the back
    // substitution, which no residual test would notice.
    for i in 0..6 {
        for j in 0..6 {
            let (a, b) = (one_metre.covariance[i][j], one_metre.covariance[j][i]);
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(f64::MIN_POSITIVE),
                "covariance is not symmetric at ({i}, {j}): {a} against {b}"
            );
        }
    }
}
