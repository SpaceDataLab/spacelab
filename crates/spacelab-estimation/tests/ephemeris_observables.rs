//! Fitting an orbit to a published ephemeris through the [`StateComponent`] observable.
//!
//! An OEM record is the state itself, so the observable is a read and its partial is a unit row.
//! What has to be shown is not the formula but the wiring: that a state sample entering the *same*
//! weighted least squares as a range or a TDOA recovers the epoch state, that the velocity half
//! carries information the position half does not, and that the covariance the fit reports for it
//! is realistic.
//!
//! # Units
//!
//! The dynamics are canonical (`μ = 1`, length unit 7000 km) so that one integrator tolerance
//! serves the state and its transition matrix alike; the samples are SI, as a CCSDS product is.
//! The boundary is one diagonal scaling in the chain rule, and every error below is in metres.
//!
//! Reading the OEM itself is `spacelab_ndm::parse_oem` and rotating it to the propagation frame is
//! `spacelab_frames`; neither is exercised here, because a fit to real published states is a
//! statement about the *force model*, and this file is a statement about the observable.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_autodiff::Scalar;
use spacelab_estimation::ephemeris::{Component, StateComponent};
use spacelab_estimation::modifier::partials;
use spacelab_estimation::{
    GaussNewton, ResidualModel, Rng, SquareRootKalman, chi_square_mean_interval, is_consistent,
    normalized_squared,
};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};

/// IERS 2010 conventional GM, m³/s².
const EARTH_GM: f64 = 3.986_004_418e14;
/// The canonical length unit, metres.
const LENGTH_UNIT: f64 = 7.0e6;
/// One canonical time unit less than a full revolution of the truth orbit.
const SPAN: f64 = 8.0;
/// Per-component uncertainty of a published state: metres and m/s.
const POSITION_SIGMA: f64 = 1.0;
const VELOCITY_SIGMA: f64 = 1.0e-3;

/// The truth orbit, canonical: perigee at one length unit, out of the equatorial plane.
const TRUTH: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.4];

fn time_unit() -> f64 {
    libm::sqrt(LENGTH_UNIT * LENGTH_UNIT * LENGTH_UNIT / EARTH_GM)
}

/// `∂x_SI/∂x_canonical` — diagonal, so a vector.
fn si_scale() -> [f64; 6] {
    let (lu, tu) = (LENGTH_UNIT, time_unit());
    [lu, lu, lu, lu / tu, lu / tu, lu / tu]
}

fn to_si(x: &[f64; 6]) -> [f64; 6] {
    let s = si_scale();
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = x[i] * s[i];
    }
    out
}

/// Two-body in canonical units (`μ = 1`), as in `orbit_determination.rs`.
struct TwoBody;
impl ScalarSystem<6> for TwoBody {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 6]) -> [S; 6] {
        let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        let inv_r3 = r2.powf(-1.5);
        [
            y[3],
            y[4],
            y[5],
            -(y[0] * inv_r3),
            -(y[1] * inv_r3),
            -(y[2] * inv_r3),
        ]
    }
}

struct TwoBodyState;
impl DynamicalSystem<6> for TwoBodyState {
    fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        *dydt = TwoBody.eval::<f64>(t, y);
    }
}

fn controls() -> Controls {
    Controls::new(Tolerance::new(1e-13, 1e-13), 1e-10)
}

#[derive(Clone, Copy)]
struct Sample {
    time: f64,
    component: Component,
    value: f64,
    sigma: f64,
}

fn sigma_of(component: Component) -> f64 {
    if component.is_velocity() {
        VELOCITY_SIGMA
    } else {
        POSITION_SIGMA
    }
}

fn sample_times() -> Vec<f64> {
    (1..=10).map(|k| SPAN * f64::from(k) / 11.0).collect()
}

/// Sample the truth orbit at every epoch, publishing `components` of each state in SI.
fn publish(components: &[Component], noise: Option<&mut Rng>) -> Vec<Sample> {
    let traj = integrate(&Dop853, &TwoBodyState, 0.0, &TRUTH, SPAN, 0.01, controls())
        .expect("the truth orbit integrates");
    let mut rng = noise;
    let mut out = Vec::new();
    for t in sample_times() {
        let state = to_si(&traj.eval(t));
        for &component in components {
            let sigma = sigma_of(component);
            let clean = state[component.index()];
            let value = match rng {
                Some(ref mut r) => clean + sigma * r.standard_normal(),
                None => clean,
            };
            out.push(Sample {
                time: t,
                component,
                value,
                sigma,
            });
        }
    }
    out
}

struct EphemerisFit<'a> {
    samples: &'a [Sample],
}

impl ResidualModel<6> for EphemerisFit<'_> {
    fn residuals(&self, x: &[f64; 6], emit: &mut dyn FnMut(&[f64; 6], f64, f64)) {
        let var = Variational::<6, _>::new(&TwoBody);
        let traj = integrate(
            &Dop853,
            &var,
            0.0,
            &augment::<6, 42>(x),
            SPAN,
            0.01,
            controls(),
        )
        .expect("propagated augmented system");
        let scale = si_scale();
        let none: [f64; 0] = [];

        for s in self.samples {
            let (state, phi) = split::<6, 42>(&traj.eval(s.time));
            let p = partials(&StateComponent::new(s.component), &to_si(&state), &none);
            let mut h = [0.0; 6];
            for j in 0..6 {
                for n in 0..6 {
                    h[j] += p.state[n] * scale[n] * phi[n][j];
                }
            }
            emit(&h, p.value - s.value, 1.0 / (s.sigma * s.sigma));
        }
    }
}

fn position_error(x: &[f64; 6]) -> f64 {
    libm::sqrt(
        (0..3)
            .map(|i| (x[i] - TRUTH[i]) * (x[i] - TRUTH[i]))
            .sum::<f64>(),
    ) * LENGTH_UNIT
}

/// Position samples alone recover the epoch state, from a start 1 % off in every component.
///
/// Ten published positions over one revolution, no velocity, no other observable. The measured
/// epoch position error is printed.
#[test]
fn published_positions_recover_the_epoch_state() {
    let samples = publish(&Component::POSITION, None);
    let mut x0 = TRUTH;
    for i in 0..6 {
        x0[i] += 0.01 * (if i % 2 == 0 { 1.0 } else { -1.0 }) * TRUTH[i].abs().max(0.1);
    }

    let (x, cov, report) = GaussNewton {
        max_iterations: 60,
        ..GaussNewton::default()
    }
    .solve(&EphemerisFit { samples: &samples }, &x0)
    .expect("position samples observe the whole state");

    assert!(report.converged, "the fit did not converge: {report:?}");
    let error = position_error(&x);
    println!(
        "position-only: recovered the epoch position to {error:.3e} m in {} iterations, cost {:.3e}",
        report.iterations, report.cost
    );
    assert!(error < 1.0e-3, "{error} m of epoch position error");
    for i in 0..6 {
        assert!(
            cov[i][i].is_finite() && cov[i][i] > 0.0,
            "unphysical variance {} for state {i}",
            cov[i][i]
        );
    }
}

/// A PV fit is a position fit with three more rows per epoch, so its covariance is no larger in any
/// component — information only adds. The measured shrinkage is printed.
#[test]
fn the_velocity_components_only_shrink_the_covariance() {
    let solve = |components: &[Component]| {
        let samples = publish(components, None);
        let (_, cov, _) = GaussNewton::default()
            .solve(&EphemerisFit { samples: &samples }, &TRUTH)
            .expect("solved from truth");
        cov
    };
    let position_only = solve(&Component::POSITION);
    let full = solve(&Component::ALL);

    for i in 0..6 {
        assert!(
            full[i][i] <= position_only[i][i] * (1.0 + 1e-9),
            "variance of state {i} grew when the velocity rows were added: {} -> {}",
            position_only[i][i],
            full[i][i]
        );
    }
    let ratio = libm::sqrt(full[0][0] / position_only[0][0]);
    println!("PV against position-only: σ of the first epoch state shrank by ×{ratio:.4}");
}

/// The PV fit's covariance is realistic, by NEES over 40 independent noise realisations.
///
/// The acceptance band from `chi_square_mean_interval(6, 40, 1.96)` is printed with the measured
/// mean.
#[test]
fn the_pv_fit_covariance_passes_the_nees_consistency_test() {
    let runs = 40;
    let mut rng = Rng::new(0x5eed_0e01);
    let mut total = 0.0;

    for _ in 0..runs {
        let samples = publish(&Component::ALL, Some(&mut rng));
        let (x, cov, report) = GaussNewton::default()
            .solve(&EphemerisFit { samples: &samples }, &TRUTH)
            .expect("solved");
        assert!(report.converged, "a run did not converge: {report:?}");

        let mut error = [0.0; 6];
        for i in 0..6 {
            error[i] = TRUTH[i] - x[i];
        }
        let root = SquareRootKalman::<6>::from_covariance([0.0; 6], &cov)
            .expect("the reported covariance is positive definite")
            .covariance_sqrt();
        total += normalized_squared(&root, &error);
    }

    let mean = total / runs as f64;
    let (lo, hi) = chi_square_mean_interval(6, runs, 1.96);
    println!("PV fit: mean NEES {mean:.3} in ({lo:.3}, {hi:.3})");
    assert!(
        is_consistent(mean, 6, runs, 1.96),
        "mean NEES {mean} is outside the 95 % band ({lo}, {hi})"
    );
}
