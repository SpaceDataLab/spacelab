//! Closing the loop: recover an orbit's epoch state from synthetic tracking data.
//!
//! This is the capstone that ties the last several pieces together — the AD Jacobian of the force
//! model (STM), the STM propagation (`spacelab_propagation::variational`), the AD partials of the
//! measurement model, and the square-root information filter. Every observation's partial with
//! respect to the *epoch* state is `∂measurement/∂state(t) · Φ(t, t₀)`: the first factor from AD, the
//! second from the propagated STM. Given noiseless observations of a true orbit, the estimator must
//! recover the true epoch state to near the integration tolerance — the honest end-to-end check.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_autodiff::{Scalar, jacobian};
use spacelab_estimation::measurement::{Station, range, range_rate};
use spacelab_estimation::{GaussNewton, ResidualModel};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};

/// Two-body in canonical units (`μ = 1`), generic for the STM's AD Jacobian.
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

/// Plain 6-state view for generating the truth trajectory.
struct TwoBodyState;
impl DynamicalSystem<6> for TwoBodyState {
    fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        *dydt = TwoBody.eval::<f64>(t, y);
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Range,
    RangeRate,
}

#[derive(Clone, Copy)]
struct Obs {
    time: f64,
    kind: Kind,
    station: Station,
    value: f64,
    sigma: f64,
}

fn measure<S: Scalar>(kind: Kind, sat: &[S; 6], station: &Station) -> S {
    match kind {
        Kind::Range => range(sat, station),
        Kind::RangeRate => range_rate(sat, station),
    }
}

/// The orbit-determination residual model: propagate the current epoch estimate with its STM, and
/// for each observation emit `(∂measurement/∂epoch, residual, weight)`.
struct OrbitDetermination<'a> {
    obs: &'a [Obs],
    span: f64,
}

impl ResidualModel<6> for OrbitDetermination<'_> {
    fn residuals(&self, x: &[f64; 6], emit: &mut dyn FnMut(&[f64; 6], f64, f64)) {
        let var = Variational::<6, _>::new(&TwoBody);
        let aug0: [f64; 42] = augment::<6, 42>(x);
        let traj = integrate(
            &Dop853,
            &var,
            0.0,
            &aug0,
            self.span,
            0.01,
            Controls::new(Tolerance::new(1e-13, 1e-13), 1e-10),
        )
        .expect("propagated augmented system");

        for o in self.obs {
            let aug = traj.eval(o.time);
            let (state, phi) = split::<6, 42>(&aug);

            // predicted measurement and ∂m/∂state(t), from the same AD source.
            let (val, dm_ds) = jacobian::<6, 1, _>(|s| [measure(o.kind, s, &o.station)], &state);

            // chain to the epoch: h_j = Σ_n (∂m/∂state_n) · Φ_{n j}
            let mut h = [0.0; 6];
            for j in 0..6 {
                for n in 0..6 {
                    h[j] += dm_ds[n][0] * phi[n][j];
                }
            }
            let residual = val[0] - o.value;
            emit(&h, residual, 1.0 / (o.sigma * o.sigma));
        }
    }
}

/// Three well-separated fixed tracking points — enough geometry to observe all six states of the
/// orbit, so noiseless data pins the epoch state to near the integration tolerance. A single station
/// over a partial arc is only weakly observable in some directions; that is a property of the
/// geometry, not of the estimator.
fn stations() -> [Station; 3] {
    [
        Station::fixed([0.6, -0.4, 0.5]),
        Station::fixed([-0.7, 0.3, -0.2]),
        Station::fixed([0.2, 0.8, 0.4]),
    ]
}

/// Propagate the truth and synthesise noiseless observations: range and range-rate from every
/// station at every time.
fn synthesise(x_true: &[f64; 6], stations: &[Station], times: &[f64], span: f64) -> Vec<Obs> {
    let traj = integrate(
        &Dop853,
        &TwoBodyState,
        0.0,
        x_true,
        span,
        0.01,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-10),
    )
    .expect("truth");
    let mut obs = Vec::new();
    for &t in times {
        let s = traj.eval(t);
        for &st in stations {
            obs.push(Obs {
                time: t,
                kind: Kind::Range,
                station: st,
                value: range(&s, &st),
                sigma: 1e-3,
            });
            obs.push(Obs {
                time: t,
                kind: Kind::RangeRate,
                station: st,
                value: range_rate(&s, &st),
                sigma: 1e-4,
            });
        }
    }
    obs
}

#[test]
fn recovers_the_epoch_state_from_range_and_range_rate_tracking() {
    // A bound, out-of-plane orbit tracked by three stations over more than one period.
    let x_true = [1.0, 0.0, 0.0, 0.0, 1.05, 0.25];
    let sts = stations();
    let span = 12.0;
    let times: Vec<f64> = (1..=15).map(|k| span * f64::from(k) / 16.0).collect();

    let obs = synthesise(&x_true, &sts, &times, span);
    let model = OrbitDetermination { obs: &obs, span };

    // start 2% off in every component — a real initial-orbit error.
    let mut x0 = x_true;
    for i in 0..6 {
        x0[i] += 0.02 * (if i % 2 == 0 { 1.0 } else { -1.0 }) * x_true[i].abs().max(0.1);
    }

    let gn = GaussNewton {
        max_iterations: 40,
        ..GaussNewton::default()
    };
    let (x, cov, report) = gn
        .solve(&model, &x0)
        .expect("orbit determination converged");

    assert!(report.converged, "OD did not converge: {report:?}");
    for i in 0..6 {
        assert!(
            (x[i] - x_true[i]).abs() < 1e-7,
            "epoch state[{i}] = {} vs true {} (report {report:?})",
            x[i],
            x_true[i]
        );
    }
    assert!(
        report.cost < 1e-6,
        "noiseless OD residual cost should be tiny, got {}",
        report.cost
    );
    for i in 0..6 {
        assert!(
            cov[i][i].is_finite() && cov[i][i] >= 0.0,
            "covariance diagonal {i} not physical"
        );
    }
}

#[test]
fn the_covariance_shrinks_as_nested_observations_are_added() {
    // Information only adds, so a *superset* of observations gives a covariance no larger in any
    // component. The subsets must be nested for this to hold — different time samples are not
    // comparable — so both use the first `n` times of one fixed schedule.
    let x_true = [1.0, 0.0, 0.0, 0.0, 1.1, 0.1];
    let sts = stations();
    let span = 10.0;
    let full: Vec<f64> = (1..=30).map(|k| span * f64::from(k) / 31.0).collect();

    let solve_with = |n: usize| -> [[f64; 6]; 6] {
        let obs = synthesise(&x_true, &sts, &full[..n], span);
        let model = OrbitDetermination { obs: &obs, span };
        let (_, cov, _) = GaussNewton::default()
            .solve(&model, &x_true)
            .expect("solved from truth");
        cov
    };

    let few = solve_with(8);
    let many = solve_with(30);
    for i in 0..6 {
        assert!(
            many[i][i] <= few[i][i] * (1.0 + 1e-9),
            "variance of state {i} grew when observations were added: {} -> {}",
            few[i][i],
            many[i][i]
        );
    }
}
