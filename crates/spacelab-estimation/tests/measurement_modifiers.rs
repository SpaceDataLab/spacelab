//! Solving for a measurement bias: the modifier layer wired into an orbit determination.
//!
//! `orbit_determination.rs` recovers six states from clean geometry. This is the seventh unknown —
//! a range bias that no amount of orbit fitting can absorb — carried as an estimated measurement
//! parameter through [`spacelab_estimation::modifier`] rather than subtracted at the call site.
//! Two things have to be true for that to be worth anything: the solve must *find* an injected
//! bias, and the covariance it reports for it must be honest, which is the NEES test the crate
//! already carries in `consistency`.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_autodiff::Scalar;
use spacelab_estimation::measurement::Station;
use spacelab_estimation::modifier::{EstimatedBias, Modified, Observable, Range, partials};
use spacelab_estimation::{
    GaussNewton, ResidualModel, Rng, SquareRootKalman, chi_square_mean_interval, is_consistent,
    normalized_squared,
};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};

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

#[derive(Clone, Copy)]
struct Obs {
    time: f64,
    station: Station,
    value: f64,
}

/// Range, biased by the one estimated measurement parameter — the whole chain this test is about.
type BiasedRange = Modified<Range, EstimatedBias<1>>;

fn biased_range(station: Station) -> BiasedRange {
    Modified::new(
        Range { station },
        EstimatedBias::<1>::new(0).expect("the only parameter"),
    )
}

const SIGMA: f64 = 1e-4;
const SPAN: f64 = 8.0;

/// The seven-state residual model: six orbital elements and one range bias.
///
/// The state partials go through the STM to reach the epoch; the bias partial does not, because a
/// bias is not propagated — it is the same number at every observation, which is exactly what makes
/// it separable from the orbit and what a fixed subtraction throws away.
struct BiasedOrbitDetermination<'a> {
    obs: &'a [Obs],
}

impl ResidualModel<7> for BiasedOrbitDetermination<'_> {
    fn residuals(&self, x: &[f64; 7], emit: &mut dyn FnMut(&[f64; 7], f64, f64)) {
        let epoch: [f64; 6] = [x[0], x[1], x[2], x[3], x[4], x[5]];
        let var = Variational::<6, _>::new(&TwoBody);
        let aug0: [f64; 42] = augment::<6, 42>(&epoch);
        let traj = integrate(
            &Dop853,
            &var,
            0.0,
            &aug0,
            SPAN,
            0.01,
            Controls::new(Tolerance::new(1e-12, 1e-12), 1e-10),
        )
        .expect("propagated augmented system");

        for o in self.obs {
            let aug = traj.eval(o.time);
            let (state, phi) = split::<6, 42>(&aug);
            let p = partials(&biased_range(o.station), &state, &[x[6]]);

            let mut h = [0.0; 7];
            for j in 0..6 {
                for n in 0..6 {
                    h[j] += p.state[n] * phi[n][j];
                }
            }
            h[6] = p.parameters[0];
            emit(&h, p.value - o.value, 1.0 / (SIGMA * SIGMA));
        }
    }
}

/// The same model with the bias column removed: six states, the bias held at zero.
struct Unbiased<'a> {
    obs: &'a [Obs],
}

impl ResidualModel<6> for Unbiased<'_> {
    fn residuals(&self, x: &[f64; 6], emit: &mut dyn FnMut(&[f64; 6], f64, f64)) {
        let seven = [x[0], x[1], x[2], x[3], x[4], x[5], 0.0];
        BiasedOrbitDetermination { obs: self.obs }.residuals(&seven, &mut |h, r, w| {
            emit(&[h[0], h[1], h[2], h[3], h[4], h[5]], r, w);
        });
    }
}

fn stations() -> [Station; 3] {
    [
        Station::fixed([0.6, -0.4, 0.5]),
        Station::fixed([-0.7, 0.3, -0.2]),
        Station::fixed([0.2, 0.8, 0.4]),
    ]
}

fn times() -> Vec<f64> {
    (1..=10).map(|k| SPAN * f64::from(k) / 11.0).collect()
}

/// Synthesise ranges from a truth orbit, biased by `bias` and perturbed by `noise` (in σ).
fn synthesise(x_true: &[f64; 7], noise: Option<&mut Rng>) -> Vec<Obs> {
    let epoch: [f64; 6] = [
        x_true[0], x_true[1], x_true[2], x_true[3], x_true[4], x_true[5],
    ];
    let traj = integrate(
        &Dop853,
        &TwoBodyState,
        0.0,
        &epoch,
        SPAN,
        0.01,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-10),
    )
    .expect("truth");

    let mut rng = noise;
    let mut obs = Vec::new();
    for t in times() {
        let s = traj.eval(t);
        for station in stations() {
            let clean = biased_range(station).value(&s, &[x_true[6]]);
            let value = match rng {
                Some(ref mut r) => clean + SIGMA * r.standard_normal(),
                None => clean,
            };
            obs.push(Obs {
                time: t,
                station,
                value,
            });
        }
    }
    obs
}

/// A bias injected into the data comes back out, and the orbit does not absorb it.
///
/// The bias is 40 σ — an offset a residual plot would show and a fit without the parameter would
/// smear into the orbit. Measured: from a start with the bias at zero and the state 1 % off, eight
/// iterations recover the injected 4e-3 canonical units to **1.3e-14**, at a weighted cost of
/// 3.2e-18.
#[test]
fn an_injected_range_bias_is_recovered_as_a_solved_parameter() {
    let x_true = [1.0, 0.0, 0.0, 0.0, 1.05, 0.25, 40.0 * SIGMA];
    let obs = synthesise(&x_true, None);
    let model = BiasedOrbitDetermination { obs: &obs };

    // Start 1 % off in every state and with no bias at all — the estimator has to find it.
    let mut x0 = x_true;
    for i in 0..6 {
        x0[i] += 0.01 * (if i % 2 == 0 { 1.0 } else { -1.0 }) * x_true[i].abs().max(0.1);
    }
    x0[6] = 0.0;

    let (x, cov, report) = GaussNewton {
        max_iterations: 40,
        ..GaussNewton::default()
    }
    .solve(&model, &x0)
    .expect("the bias is observable alongside the orbit");

    assert!(report.converged, "did not converge: {report:?}");
    assert!(
        (x[6] - x_true[6]).abs() < 1e-9,
        "bias {} against the injected {} (report {report:?})",
        x[6],
        x_true[6]
    );
    for i in 0..6 {
        assert!(
            (x[i] - x_true[i]).abs() < 1e-7,
            "epoch state[{i}] = {} vs true {}",
            x[i],
            x_true[i]
        );
    }
    assert!(
        cov[6][6] > 0.0 && cov[6][6].is_finite(),
        "the bias has no variance: {}",
        cov[6][6]
    );
}

/// Without the parameter the same data cannot be fitted, which is what makes the parameter worth
/// carrying: the bias goes somewhere, and where it goes is the orbit and the residuals.
///
/// Measured: holding the bias at zero leaves a weighted cost of **2.06e4** against 3.6e-18 with it
/// solved for, and moves the epoch position by **7.1e-3** canonical units — 40 σ of range error
/// re-expressed as an orbit error, seventy times the σ it came from.
#[test]
fn a_bias_that_cannot_be_solved_for_lands_in_the_orbit() {
    let x_true = [1.0, 0.0, 0.0, 0.0, 1.05, 0.25, 40.0 * SIGMA];
    let obs = synthesise(&x_true, None);

    let with_bias = GaussNewton::default()
        .solve(&BiasedOrbitDetermination { obs: &obs }, &x_true)
        .expect("solved with the bias");

    let start: [f64; 6] = [
        x_true[0], x_true[1], x_true[2], x_true[3], x_true[4], x_true[5],
    ];
    let (x, _, report) = GaussNewton::default()
        .solve(&Unbiased { obs: &obs }, &start)
        .expect("solved without the bias");

    assert!(
        report.cost > 1e2 * with_bias.2.cost.max(1e-12),
        "leaving the bias out cost {} against {}",
        report.cost,
        with_bias.2.cost
    );
    let moved = libm::sqrt(
        (0..3)
            .map(|i| (x[i] - x_true[i]) * (x[i] - x_true[i]))
            .sum::<f64>(),
    );
    assert!(
        moved > 1e-4,
        "the unmodelled bias moved the epoch position by only {moved}"
    );
}

/// The covariance the solve reports for the seven states is realistic, by NEES over 60 independent
/// noise realisations.
///
/// Each run draws its own Gaussian range noise at σ, solves from the truth, and forms
/// `(x_true − x̂)ᵀ P⁻¹ (x_true − x̂)` through the covariance's square root. For a consistent filter
/// the mean over the runs is the state dimension, 7; the acceptance band from
/// `chi_square_mean_interval(7, 60, 1.96)` is **(6.085, 7.978)** and the measured mean is
/// **6.72**.
#[test]
fn the_solved_covariance_passes_the_nees_consistency_test() {
    let x_true = [1.0, 0.0, 0.0, 0.0, 1.05, 0.25, 40.0 * SIGMA];
    let runs = 60;
    let mut rng = Rng::new(0x5eed_0313);
    let mut total = 0.0;

    for _ in 0..runs {
        let obs = synthesise(&x_true, Some(&mut rng));
        let (x, cov, report) = GaussNewton::default()
            .solve(&BiasedOrbitDetermination { obs: &obs }, &x_true)
            .expect("solved");
        assert!(report.converged, "a run did not converge: {report:?}");

        let mut error = [0.0; 7];
        for i in 0..7 {
            error[i] = x_true[i] - x[i];
        }
        let root = SquareRootKalman::<7>::from_covariance([0.0; 7], &cov)
            .expect("the reported covariance is positive definite")
            .covariance_sqrt();
        total += normalized_squared(&root, &error);
    }

    let mean = total / runs as f64;
    let (lo, hi) = chi_square_mean_interval(7, runs, 1.96);
    assert!(
        is_consistent(mean, 7, runs, 1.96),
        "mean NEES {mean} is outside the 95 % band ({lo}, {hi}) for 7 degrees of freedom"
    );
}
