//! DOP853 accuracy: the measured convergence order, Kepler conservation, dense output, and a
//! differential cross-check against the independently-verified DOPRI5.
//!
//! The convergence-order test reads the *actual* order off the numbers, so a wrong coefficient — or
//! a wrong error/dense formula — shows immediately, whatever a single step looks like.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

use spacelab_propagation::dop853::Dop853;
use spacelab_propagation::driver::{Controls, Method, integrate_dop853, integrate_dopri5};
use spacelab_propagation::{DynamicalSystem, Tolerance};
use spacelab_testkit::nan_max;

struct Exponential;
impl DynamicalSystem<1> for Exponential {
    fn derivative(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = -y[0];
    }
}

struct Harmonic;
impl DynamicalSystem<2> for Harmonic {
    fn derivative(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

struct Kepler;
impl DynamicalSystem<4> for Kepler {
    fn derivative(&self, _t: f64, y: &[f64; 4], dydt: &mut [f64; 4]) {
        let r3 = (y[0] * y[0] + y[1] * y[1]).powf(1.5);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -y[0] / r3;
        dydt[3] = -y[1] / r3;
    }
}

fn energy(y: &[f64; 4]) -> f64 {
    0.5 * (y[2] * y[2] + y[3] * y[3]) - 1.0 / (y[0] * y[0] + y[1] * y[1]).sqrt()
}
fn angular_momentum(y: &[f64; 4]) -> f64 {
    y[0] * y[3] - y[1] * y[2]
}

fn dop853_fixed<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    steps: u32,
) -> [f64; N] {
    let h = (t1 - t0) / f64::from(steps);
    let mut t = t0;
    let mut y = *y0;
    for _ in 0..steps {
        y = Dop853.step_fixed(sys, t, &y, h);
        t += h;
    }
    y
}

#[test]
fn dop853_converges_at_order_eight() {
    // Order 8 error falls as h^8, so it reaches the machine floor fast — sample coarse grids where
    // the error is still well above it and read the order from the finest usable pair.
    let mut errs = Vec::new();
    let mut ns = Vec::new();
    for &n in &[2u32, 3, 4, 5, 6] {
        let y = dop853_fixed(&Exponential, 0.0, &[1.0], 5.0, n);
        let err = (y[0] - (-5.0f64).exp()).abs();
        if err > 1e-13 {
            errs.push(err);
            ns.push(n);
        }
    }
    assert!(errs.len() >= 2, "need at least two grids above the floor");
    let k = errs.len() - 1;
    let order = (errs[k - 1] / errs[k]).log2() / (f64::from(ns[k]) / f64::from(ns[k - 1])).log2();
    assert!(
        (7.0..9.5).contains(&order),
        "DOP853 measured order {order:.2} is not ~8 (a coefficient error would drop it): errs={errs:?}"
    );
}

#[test]
fn dop853_conserves_kepler_energy_and_momentum_over_many_orbits() {
    let y0 = [1.0, 0.0, 0.0, 1.3];
    let (e0, l0) = (energy(&y0), angular_momentum(&y0));
    let a = -1.0 / (2.0 * e0);
    let period = std::f64::consts::TAU * (a * a * a).sqrt();
    let traj = integrate_dop853(
        &Kepler,
        0.0,
        &y0,
        20.0 * period,
        0.05,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
    )
    .expect("integrated");
    let yf = traj.final_state();
    assert!(
        (energy(&yf) - e0).abs() < 1e-9,
        "energy drift {:.3e} over 20 orbits",
        (energy(&yf) - e0).abs()
    );
    assert!(
        (angular_momentum(&yf) - l0).abs() < 1e-10,
        "L drift over 20 orbits"
    );
}

#[test]
fn dop853_agrees_with_dopri5_on_the_same_ivp() {
    // Two independently-verified methods on the same Kepler IVP must agree to the shared tolerance.
    // A different tableau lineage reaching the same answer is the cross-check `plans/04` §1.2 asks
    // of every component.
    let y0 = [1.0, 0.0, 0.0, 1.1];
    let t1 = 15.0;
    let tol = Tolerance::new(1e-12, 1e-12);
    let a =
        integrate_dop853(&Kepler, 0.0, &y0, t1, 0.05, Controls::new(tol, 1e-9)).expect("dop853");
    let b =
        integrate_dopri5(&Kepler, 0.0, &y0, t1, 0.05, Controls::new(tol, 1e-9)).expect("dopri5");
    let (ya, yb) = (a.final_state(), b.final_state());
    let d = (0..4).map(|i| (ya[i] - yb[i]).abs()).fold(0.0, f64::max);
    assert!(
        d < 1e-8,
        "DOP853 and DOPRI5 disagree by {d:.3e} — one of them is wrong"
    );
}

#[test]
fn dop853_is_more_efficient_than_dopri5_at_a_tight_tolerance() {
    // The whole point of order 8: fewer steps for the same accuracy on a smooth problem.
    let y0 = [1.0, 0.0];
    let t1 = 100.0;
    let tol = Tolerance::new(1e-12, 1e-12);
    let hi =
        integrate_dop853(&Harmonic, 0.0, &y0, t1, 0.1, Controls::new(tol, 1e-9)).expect("dop853");
    let lo =
        integrate_dopri5(&Harmonic, 0.0, &y0, t1, 0.1, Controls::new(tol, 1e-9)).expect("dopri5");
    assert!(
        hi.step_counts().0 < lo.step_counts().0,
        "DOP853 took {} steps, DOPRI5 {} — order 8 should need fewer",
        hi.step_counts().0,
        lo.step_counts().0
    );
    // and both are actually accurate
    assert!((hi.final_state()[0] - t1.cos()).abs() < 1e-8);
}

#[test]
fn dop853_dense_output_matches_the_analytic_solution() {
    let t1 = 10.0;
    let traj = integrate_dop853(
        &Harmonic,
        0.0,
        &[1.0, 0.0],
        t1,
        0.1,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-9),
    )
    .expect("integrated");
    // endpoints exact
    assert!((traj.eval(0.0)[0] - 1.0).abs() < 1e-12, "θ=0 start");
    // interior: 1000 samples against cos
    let mut worst = 0.0_f64;
    for k in 0..=1000 {
        let t = t1 * k as f64 / 1000.0;
        worst = nan_max(worst, (traj.eval(t)[0] - t.cos()).abs());
    }
    assert!(worst < 1e-9, "DOP853 dense worst error {worst:.3e}");
}

/// `dy/dt = t`: a derivative that is a function of time alone. Every other system in this file is
/// autonomous and discards `t`, which leaves the stage times unobservable.
struct Ramp;
impl DynamicalSystem<1> for Ramp {
    fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = t;
    }
}

/// Depends on both arguments, so a stage evaluated at the wrong time *or* from the wrong state
/// moves the answer.
struct Coupled;
impl DynamicalSystem<1> for Coupled {
    fn derivative(&self, t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = t + y[0];
    }
}

struct Stationary;
impl DynamicalSystem<2> for Stationary {
    fn derivative(&self, _t: f64, _y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = 0.0;
        dydt[1] = 0.0;
    }
}

/// A nonlinear scalar decay, and the same thing run as two uncoupled copies.
struct Decay;
impl DynamicalSystem<1> for Decay {
    fn derivative(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = -y[0] * y[0];
    }
}
struct DecayPair;
impl DynamicalSystem<2> for DecayPair {
    fn derivative(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = -y[0] * y[0];
        dydt[1] = -y[1] * y[1];
    }
}

struct Counting {
    calls: std::cell::Cell<usize>,
}
impl DynamicalSystem<1> for Counting {
    fn derivative(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        self.calls.set(self.calls.get() + 1);
        dydt[0] = -y[0];
    }
}

#[test]
fn the_fixed_step_evaluates_its_stages_at_the_right_times() {
    // Quadrature: for dy/dt = t the whole step reduces to h²·ΣB[j]C[j], which the order conditions
    // make exactly 1/2. So one step from t = 0 with h = 2 must return ∫₀² t dt = 2 to round-off,
    // and any perturbation of the stage time C[stage]·h moves it by an O(1) amount -- taking the
    // offset away gives -2, multiplying by it gives 0, adding h gives 5, dividing by h gives 1/2.
    //
    // h is 2 rather than 1 on purpose: at h = 1 a stage time divided by h is the stage time.
    let y = Dop853.step_fixed(&Ramp, 0.0, &[0.0], 2.0);
    assert!(
        (y[0] - 2.0).abs() < 1e-14,
        "one fixed step of dy/dt = t over [0,2] gave {}, not 2",
        y[0]
    );
}

#[test]
#[allow(clippy::float_cmp)] // the two share a tableau; anything but bit equality is a divergence
fn the_fixed_step_reproduces_the_adaptive_stepper_exactly() {
    // `step_fixed` carries its own copy of the stage loop. It is the order-8 solution and nothing
    // else, so it has to agree with `try_step`'s `y_next` bit for bit -- the two read the same `A`,
    // `B` and `C` rows through separate code. The system depends on `t` and on `y`, so a copy that
    // drifts in either the stage weights or the stage times shows here.
    let (t, h, y0) = (0.5, 0.25, [1.5]);
    let fixed = Dop853.step_fixed(&Coupled, t, &y0, h);
    let adaptive = Dop853.try_step(&Coupled, t, &y0, h, Tolerance::new(1e-12, 1e-12));
    assert_eq!(fixed[0], adaptive.y_next[0]);
}

#[test]
#[allow(clippy::float_cmp)] // the claim is exactly zero, against the NaN the guard exists to avoid
fn a_stationary_system_reports_a_zero_error_norm_and_not_a_nan() {
    // Both error estimates vanish identically when every stage derivative is zero, and the norm
    // divides by their combination. Without the guard that is 0/0: the step would be rejected on a
    // NaN comparison and the driver would shrink `h` for ever on a problem with no error at all.
    let step = Dop853.try_step(
        &Stationary,
        0.0,
        &[7.0e6, -1.2e3],
        60.0,
        Tolerance::new(1e-12, 1e-12),
    );
    assert_eq!(step.error_norm, 0.0);
}

#[test]
#[allow(clippy::float_cmp)] // the scaling is exact in binary64; see the comment
fn the_error_norm_does_not_depend_on_how_many_copies_of_a_system_are_integrated() {
    // The norm is an RMS over the state, so running two uncoupled copies of the same scalar problem
    // must report the same error as running one. Both squared sums double, the denominator doubles,
    // and the dimension doubles -- the 2 and the √(2·2) cancel. Every one of those steps is exact
    // in binary64 (doubling and √4 are), so the two norms agree bit for bit.
    //
    // That cancellation is what pins `denom * N`: dividing by N instead makes the pair read twice
    // the single, and adding N makes it read √2 times.
    let (t, h, tol) = (0.0, 0.5, Tolerance::new(1e-10, 1e-10));
    let one = Dop853.try_step(&Decay, t, &[1.0], h, tol);
    let pair = Dop853.try_step(&DecayPair, t, &[1.0, 1.0], h, tol);
    assert!(one.error_norm > 0.0, "the fixture must have a live error");
    assert_eq!(one.error_norm, pair.error_norm);
}

#[test]
fn the_stage_counts_are_what_a_step_actually_costs() {
    // The four public constants are a cost model callers budget against, and nothing had ever
    // checked them against the loop they describe.
    let sys = Counting {
        calls: std::cell::Cell::new(0),
    };
    let _ = Dop853.try_step(&sys, 0.0, &[1.0], 0.1, Tolerance::new(1e-12, 1e-12));
    assert_eq!(sys.calls.get(), Dop853::EXTENDED_STAGES);

    sys.calls.set(0);
    let _ = Dop853.step_fixed(&sys, 0.0, &[1.0], 0.1);
    assert_eq!(sys.calls.get(), Dop853::STAGES);

    assert_eq!(
        Dop853::EXTENDED_STAGES,
        Dop853::STAGES + Dop853::ERROR_STAGES + Dop853::INTERPOLANT_STAGES
    );
}
