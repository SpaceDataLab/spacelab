//! Accuracy of RK4 and DOPRI5, read off analytic solutions and the measured convergence order.
//!
//! The convergence-order tests are the ones that catch a wrong tableau coefficient: a method
//! claiming order 5 that is secretly order 4 shows it in the slope of `log(error)` vs `log(h)`,
//! no matter how good it looks at one step size. The conservation tests catch it a second way —
//! a bad coefficient breaks energy/momentum conservation of a Kepler orbit long before it breaks
//! a single step.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::abm::{integrate_abm8, integrate_abm8_dense};
use spacelab_propagation::adams_nordsieck::integrate_adams_nordsieck;
use spacelab_propagation::ark::integrate_ark;
use spacelab_propagation::dopri5::Dopri5;
use spacelab_propagation::driver::IntegrationError;
use spacelab_propagation::driver::{Controls, Method, integrate_dopri5};
use spacelab_propagation::gj::{integrate_gj8, integrate_gj8_dense};
use spacelab_propagation::mri::integrate_mri;
use spacelab_propagation::radau::{integrate_radau5, integrate_radau5_dense};
use spacelab_propagation::rk4::{rk4_integrate, rk4_step};
use spacelab_propagation::rosenbrock::{integrate_rosenbrock4, integrate_rosenbrock4_dense};
use spacelab_propagation::variational::ScalarSystem;
use spacelab_propagation::{DenseOutput, DynamicalSystem, Tolerance};
use spacelab_testkit::nan_max;

/// dy/dt = -y, y(0)=1 ⇒ y(t)=exp(-t).
struct Exponential;
impl DynamicalSystem<1> for Exponential {
    fn derivative(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = -y[0];
    }
}

/// y=[x, v], x''=-x ⇒ x=cos t, v=-sin t.
struct Harmonic;
impl DynamicalSystem<2> for Harmonic {
    fn derivative(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

/// Planar two-body: y=[x, y, vx, vy], a = -mu r / |r|³, mu = 1.
struct Kepler {
    mu: f64,
}
impl DynamicalSystem<4> for Kepler {
    fn derivative(&self, _t: f64, y: &[f64; 4], dydt: &mut [f64; 4]) {
        let r2 = y[0] * y[0] + y[1] * y[1];
        let r3 = r2 * r2.sqrt();
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -self.mu * y[0] / r3;
        dydt[3] = -self.mu * y[1] / r3;
    }
}

/// dy/dt = y·cos t ⇒ y(t) = y(t₀)·exp(sin t − sin t₀). Non-autonomous *and* state-dependent, so it
/// reads both arguments of `derivative` — which every other fixture here declines to do.
struct Modulated;
impl DynamicalSystem<1> for Modulated {
    fn derivative(&self, t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = y[0] * t.cos();
    }
}

fn modulated_exact(t0: f64, y0: f64, t: f64) -> f64 {
    y0 * (t.sin() - t0.sin()).exp()
}

fn kepler_energy(y: &[f64; 4], mu: f64) -> f64 {
    let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
    let v2 = y[2] * y[2] + y[3] * y[3];
    0.5 * v2 - mu / r
}
fn kepler_angular_momentum(y: &[f64; 4]) -> f64 {
    y[0] * y[3] - y[1] * y[2]
}

/// Fixed-step DOPRI5 (ignore the error estimate), for reading the convergence order cleanly.
fn dopri5_fixed<const N: usize>(
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
        y = Dopri5.try_step(sys, t, &y, h).y_next;
        t += h;
    }
    y
}

// -------------------------------------------------------------------------------------------------
// Convergence order — the coefficient-error detector.
// -------------------------------------------------------------------------------------------------

#[test]
fn rk4_converges_at_order_four() {
    // exp decay over [0, 5]; halving the step must cut the error by ~2^4 = 16.
    let mut prev = f64::INFINITY;
    let mut ratios = Vec::new();
    for &n in &[10u32, 20, 40, 80, 160] {
        let y = rk4_integrate(&Exponential, 0.0, &[1.0], 5.0, n);
        let err = (y[0] - (-5.0f64).exp()).abs();
        if prev.is_finite() && err > 1e-14 {
            ratios.push(prev / err);
        }
        prev = err;
    }
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    assert!(
        (13.0..19.0).contains(&mean),
        "RK4 error-reduction ratio {mean:.2} is not ~16 (order 4): {ratios:?}"
    );
}

/// RK4's stage times and `rk4_integrate`'s own clock are observable only through a system that
/// reads `t`, started somewhere other than the origin. Every other fixture in this file ignores its
/// time argument and every other call starts at `t0 = 0`, which leaves nine separate wrong answers
/// invisible at once: the three stage times (`t + h/2` twice, then `t + h`), the half-step factor
/// `0.5 * h` inside two of them, the step size `(t1 - t0)/n` -- which is `(t1 + t0)/n` exactly when
/// `t0 = 0` -- and the accumulator `t += h`, which `t *= h` reproduces perfectly from zero.
///
/// `y' = y cos t` reads both arguments and has the closed form `y(t) = y(t0) exp(sin t - sin t0)`.
/// Measured on `[1, 3]`: 2.749e-7, 1.801e-8 and 1.150e-9 relative at n = 20, 40, 80, ratios 15.3
/// and 15.7 -- the fourth order the method claims -- and 7.839e-11 for a single step of h = 0.05
/// from `t0 = 1`. Displacing any stage time costs the method its order, so the ratio is the
/// structural half of the check and the absolute bounds are the numerical half.
#[test]
fn rk4_reads_the_time_it_is_given() {
    let (t0, t1, y0) = (1.0_f64, 3.0_f64, 1.0_f64);
    let exact = modulated_exact(t0, y0, t1);

    let rel = |n| ((rk4_integrate(&Modulated, t0, &[y0], t1, n)[0] - exact) / exact).abs();
    let (coarse, fine) = (rel(40), rel(80));
    assert!(
        fine < 5.0e-9,
        "RK4 on a non-autonomous system is off by {fine:.3e} relative at n = 80"
    );
    assert!(
        (13.0..19.0).contains(&(coarse / fine)),
        "RK4 error-reduction ratio {:.2} is not ~16 on a system that reads t: {coarse:.3e} -> \
         {fine:.3e}",
        coarse / fine
    );

    // One step, so the stage times are pinned without the accumulator in the way.
    let h = 0.05;
    let step = rk4_step(&Modulated, t0, &[y0], h);
    let step_exact = modulated_exact(t0, y0, t0 + h);
    assert!(
        ((step[0] - step_exact) / step_exact).abs() < 5.0e-10,
        "one RK4 step from t0 = {t0} is off by {:.3e} relative",
        (step[0] - step_exact) / step_exact
    );
}

#[test]
fn dopri5_converges_at_order_five() {
    // Same test on the DOPRI5 tableau; the ratio must be ~2^5 = 32. This is what a wrong coefficient
    // would break: the measured order would drop, whatever the single-step error looks like.
    // As n grows the ratio approaches 2^5 = 32 from above; the finest pair is the most asymptotic.
    // The order read from it must be ~5 — decisively above 4, which a dropped coefficient would give.
    let mut errs = Vec::new();
    for &n in &[8u32, 16, 32, 64] {
        let y = dopri5_fixed(&Exponential, 0.0, &[1.0], 5.0, n);
        errs.push((y[0] - (-5.0f64).exp()).abs());
    }
    let orders: Vec<f64> = errs.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
    let finest = *orders.last().unwrap();
    assert!(
        (4.7..6.0).contains(&finest),
        "DOPRI5 measured order {finest:.2} at the finest grid is not ~5 (order 4 would be ~4): {orders:?}"
    );
}

// -------------------------------------------------------------------------------------------------
// Non-autonomous forcing — the half of the tableau an autonomous system cannot see.
// -------------------------------------------------------------------------------------------------

/// dy/dt = 5t⁴ ⇒ y(t) = t⁵ + const. Every other system in this file takes `_t`, and a system that
/// ignores `t` leaves the nodes `c` and every stage argument `t + cᵢh` carrying no information: they
/// reach the answer only through `f`, and `f` never reads them.
struct QuarticForcing;
impl DynamicalSystem<1> for QuarticForcing {
    fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = 5.0 * t * t * t * t;
    }
}

#[test]
fn dopri5_integrates_a_quartic_forcing_exactly_in_one_step() {
    // Pure quadrature: a method whose weights satisfy Σ bᵢcᵢ^(k−1) = 1/k for k ≤ 5 reproduces
    // ∫ 5t⁴ dt exactly, and no wrong node survives that. The span starts at t = 1, not 0, so a
    // stage argument mutated from `t + cᵢh` to `t · cᵢ · h` is not silently zero.
    let y = Dopri5.try_step(&QuarticForcing, 1.0, &[1.0], 1.0).y_next;
    assert!(
        (y[0] - 32.0).abs() < 1e-13,
        "one step from t=1 to t=2 of y' = 5t⁴: {} vs 2⁵",
        y[0]
    );
}

/// dy/dt = −y + t, y(0) = 1 ⇒ y(t) = t − 1 + 2e^(−t). Both `t` and `y` are live, so the stage
/// arguments matter at every order, not only through the quadrature.
struct LinearRamp;
impl DynamicalSystem<1> for LinearRamp {
    fn derivative(&self, t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = -y[0] + t;
    }
}

#[test]
fn dopri5_converges_at_order_five_on_a_non_autonomous_system() {
    let exact = 2.0 - 1.0 + 2.0 * (-2.0f64).exp();
    let mut errs = Vec::new();
    for &n in &[8u32, 16, 32, 64] {
        errs.push((dopri5_fixed(&LinearRamp, 0.0, &[1.0], 2.0, n)[0] - exact).abs());
    }
    let orders: Vec<f64> = errs.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
    let finest = *orders.last().unwrap();
    assert!(
        (4.7..6.0).contains(&finest),
        "measured order {finest:.2} on a t-dependent forcing is not ~5: {orders:?}"
    );
}

// -------------------------------------------------------------------------------------------------
// Analytic-solution accuracy.
// -------------------------------------------------------------------------------------------------

#[test]
fn dopri5_tracks_the_harmonic_oscillator_over_many_periods() {
    // cos(t)/-sin(t) over 20 periods, tight tolerance. Phase and amplitude must hold.
    let t1 = 40.0 * std::f64::consts::PI;
    let traj = integrate_dopri5(
        &Harmonic,
        0.0,
        &[1.0, 0.0],
        t1,
        0.1,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
    )
    .expect("integrated");
    let y = traj.final_state();
    assert!(
        (y[0] - t1.cos()).abs() < 1e-8,
        "x drifted: {} vs {}",
        y[0],
        t1.cos()
    );
    assert!(
        (y[1] + t1.sin()).abs() < 1e-8,
        "v drifted: {} vs {}",
        y[1],
        -t1.sin()
    );
}

#[test]
fn dopri5_conserves_kepler_energy_and_momentum() {
    // A moderately eccentric orbit (e ≈ 0.3): perigee at x=1, circular-ish speed. Over 10 orbits,
    // energy and angular momentum must not drift — the invariant a wrong coefficient destroys.
    let sys = Kepler { mu: 1.0 };
    // r0 = (1, 0), v0 = (0, 1.3): energy = 0.5*1.69 - 1 = -0.155, bound; L = 1.3.
    let y0 = [1.0, 0.0, 0.0, 1.3];
    let e0 = kepler_energy(&y0, 1.0);
    let l0 = kepler_angular_momentum(&y0);
    // period from a = -mu/(2E)
    let a = -1.0 / (2.0 * e0);
    let period = std::f64::consts::TAU * (a * a * a).sqrt();

    let traj = integrate_dopri5(
        &sys,
        0.0,
        &y0,
        10.0 * period,
        0.01,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
    )
    .expect("integrated");
    let yf = traj.final_state();
    let de = (kepler_energy(&yf, 1.0) - e0).abs();
    let dl = (kepler_angular_momentum(&yf) - l0).abs();
    assert!(de < 1e-9, "energy drifted by {de:.3e} over 10 orbits");
    assert!(
        dl < 1e-10,
        "angular momentum drifted by {dl:.3e} over 10 orbits (a coefficient bug gives orders more)"
    );
}

// -------------------------------------------------------------------------------------------------
// Dense output.
// -------------------------------------------------------------------------------------------------

#[test]
fn dense_output_reproduces_endpoints_and_the_analytic_solution_mid_step() {
    // One DOPRI5 step of the harmonic oscillator; check the interpolant at both ends and inside.
    let h = 0.3;
    let y0 = [1.0, 0.0];
    let trial = Dopri5.try_step(&Harmonic, 0.0, &y0, h);
    let mut out = [0.0; 2];

    trial.dense.eval(0.0, &mut out);
    assert!(
        (out[0] - y0[0]).abs() < 1e-14 && (out[1] - y0[1]).abs() < 1e-14,
        "θ=0 must be y0"
    );
    trial.dense.eval(h, &mut out);
    assert!(
        (out[0] - trial.y_next[0]).abs() < 1e-14 && (out[1] - trial.y_next[1]).abs() < 1e-14,
        "θ=1 must be y_next"
    );

    // mid-step must match cos/-sin to the interpolant's 4th order (h^4 ≈ 8e-3 here, but the constant
    // is small; demand a comfortable 1e-6)
    for &frac in &[0.25, 0.5, 0.75] {
        let t = frac * h;
        trial.dense.eval(t, &mut out);
        assert!(
            (out[0] - t.cos()).abs() < 1e-6,
            "dense x at θ={frac}: {} vs {}",
            out[0],
            t.cos()
        );
    }
}

#[test]
fn dense_output_across_a_full_trajectory_matches_the_analytic_solution() {
    let t1 = 10.0;
    let traj = integrate_dopri5(
        &Harmonic,
        0.0,
        &[1.0, 0.0],
        t1,
        0.1,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-9),
    )
    .expect("integrated");
    // sample the dense output at 1000 arbitrary interior times
    let mut worst = 0.0_f64;
    for k in 0..=1000 {
        let t = t1 * k as f64 / 1000.0;
        let y = traj.eval(t);
        worst = nan_max(worst, (y[0] - t.cos()).abs());
    }
    assert!(
        worst < 1e-9,
        "dense trajectory worst error {worst:.3e} exceeds the step tolerance"
    );
}

// -------------------------------------------------------------------------------------------------
// The controller and its error paths.
// -------------------------------------------------------------------------------------------------

#[test]
fn tighter_tolerance_gives_a_smaller_error_and_more_steps() {
    let solve = |rtol: f64| {
        let traj = integrate_dopri5(
            &Harmonic,
            0.0,
            &[1.0, 0.0],
            20.0,
            0.1,
            Controls::new(Tolerance::new(rtol, rtol), 1e-9),
        )
        .expect("integrated");
        let err = (traj.final_state()[0] - 20.0f64.cos()).abs();
        (err, traj.step_counts().0)
    };
    let (err_loose, steps_loose) = solve(1e-6);
    let (err_tight, steps_tight) = solve(1e-11);
    assert!(
        err_tight < err_loose,
        "tighter tol must reduce error: {err_tight:.2e} vs {err_loose:.2e}"
    );
    assert!(
        steps_tight > steps_loose,
        "tighter tol must cost more steps"
    );
}

#[test]
fn a_step_that_cannot_meet_tolerance_is_reported_not_crawled() {
    // rk4_step exists to be the baseline; here just confirm the driver rejects bad arguments.
    use spacelab_propagation::driver::IntegrationError;
    let r = integrate_dopri5(
        &Harmonic,
        5.0,
        &[1.0, 0.0],
        0.0,
        0.1,
        Controls::new(Tolerance::default(), 1e-9),
    );
    assert_eq!(
        r.err(),
        Some(IntegrationError::BadArguments),
        "t1 < t0 must be rejected"
    );

    // and rk4_step is callable as the documented baseline
    let y = rk4_step(&Harmonic, 0.0, &[1.0, 0.0], 0.01);
    assert!((y[0] - 0.01f64.cos()).abs() < 1e-6);
}

/// The van der Pol oscillator `y'' = mu (1 - y^2) y' - y`. Its limit cycle alternates slow arcs
/// with fast switches, so the step size the tolerance permits swings by two orders of magnitude
/// inside one period — which is what makes it a controller test rather than a method test.
struct VanDerPol {
    mu: f64,
}
impl DynamicalSystem<2> for VanDerPol {
    fn derivative(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
}

/// Every other test here asserts the answer, and the answer cannot see the step-size controller: a
/// controller that thrashes still converges, it just pays for it. Measured on the harmonic
/// oscillator, `error x steps^5` — the quantity a fifth-order method holds fixed along its own
/// work-accuracy curve — is 9310 for this controller and 9313 with the PI memory term removed. The
/// only observable that separates them is the cost, so this test asserts the cost.
///
/// A step is rejected when the error estimate exceeds the tolerance, i.e. when the controller
/// asked for a step the problem would not take. On a smooth problem that should be rare even
/// where the admissible step swings hard, and the measured worst cell here is 4.6%. Inverting the
/// error model with the wrong exponent (`h_new` from `err^0.97` rather than `err^0.17`) overshoots
/// on every step and drives it to 115%.
#[test]
fn dopri5_controller_keeps_step_rejections_rare_across_stiffness() {
    let (mut accepted, mut rejected) = (0u32, 0u32);
    for &mu in &[1.0f64, 5.0, 10.0, 20.0] {
        for &tol in &[1e-6f64, 1e-9] {
            let traj = integrate_dopri5(
                &VanDerPol { mu },
                0.0,
                &[2.0, 0.0],
                2.0 * mu,
                1e-3,
                Controls::new(Tolerance::new(tol, tol), 1e-12),
            )
            .expect("van der Pol is smooth and non-stiff at these mu");
            let (acc, rej) = traj.step_counts();
            accepted += acc;
            rejected += rej;
        }
    }
    assert!(
        10 * rejected <= accepted,
        "{rejected} rejected against {accepted} accepted is above the 10% a tuned PI controller \
         should need; measured 1.7% here and 115% with a wrong step exponent"
    );
}

/// The two controller constants are tuning, not derivation, so nothing downstream of them fails
/// loudly enough to pin them: dropping `beta` to zero still integrates every system in this file to
/// tolerance, and only shows up as 2-3x more rejected steps at every `mu` and every tolerance
/// measured above (van der Pol, 12 cells, monotone). Their provenance is the assertion.
#[test]
#[allow(clippy::float_cmp)] // pinning a literal to its source is exactly what this asserts
fn dopri5_controller_constants_match_their_derivation() {
    // `Method` is generic in the state dimension and neither constant depends on it.
    // 1/(p_hat + 1) for the embedded estimator's order p_hat = 4.
    assert_eq!(<Dopri5 as Method<1>>::step_exponent(&Dopri5), 0.2);
    // Hairer's dopri5.f, PI-controller memory. DOP853 ships 0, so it is not merely the default.
    assert_eq!(<Dopri5 as Method<1>>::beta(&Dopri5), 0.04);
}

/// The FSAL stage `k7 = f(t + h, y_next)` feeds only the embedded error estimate, never `y_next`,
/// so a wrong stage time there leaves every accuracy assertion in this file intact: the answer stays
/// fifth-order correct and the run merely pays more for it. Every *adaptive* test above integrates
/// an autonomous system, where that stage time is not just unobserved but genuinely equivalent.
///
/// So the observable is the cost. A fifth-order method whose estimator is order four needs
/// `h ~ tol^(1/5)`, i.e. about `2 / (1e-6)^0.2 = 32` steps over this span; it takes 10. Corrupting
/// the k7 stage time leaves an O(h^2) term in the estimate, and the controller chases it: measured
/// 493 steps for `t - h` and 58809 for `t * h`, for an answer no more accurate than 1e-6 asked for.
#[test]
fn dopri5_error_estimate_costs_only_what_its_order_implies() {
    let exact = 2.0 - 1.0 + 2.0 * (-2.0f64).exp();
    let traj = integrate_dopri5(
        &LinearRamp,
        0.0,
        &[1.0],
        2.0,
        0.1,
        Controls::new(Tolerance::new(1e-6, 1e-6), 1e-14),
    )
    .expect("integrated");
    let (accepted, _) = traj.step_counts();
    let err = (traj.final_state()[0] - exact).abs();
    assert!(err < 1e-5, "1e-6 was requested and {err:.3e} delivered");
    assert!(
        accepted <= 100,
        "{accepted} steps for a 1e-6 answer an order-4 estimator reaches in ~32; \
         an estimator of the wrong order is the only thing that costs this much"
    );
}

/// `y' = −y` again, in the autodiff-generic form the implicit methods take.
struct Decay;
impl ScalarSystem<1> for Decay {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
        [y[0] * S::from_f64(-1.0)]
    }
}

/// A step too small to walk the span is refused by every fixed-step entry point, not entered.
///
/// All ten computed their step count as `((t1 - t0) / h).round() as usize`, and a float-to-integer
/// `as` cast in Rust *saturates*: `1e300 as usize` is `usize::MAX`, silently. The loop that
/// followed was 1.8e19 iterations of a cheap body — nothing a caller could interrupt, and nothing a
/// test could tell from a hang. Two mutants of `integrate_whfast` were caught by that
/// non-termination and by nothing else, which is what pointed here: a suite that reports a defect
/// only by running forever is reporting it in the last way left to it.
///
/// `h = 1e-300` asks for 1e300 steps, which is finite and 293 orders of magnitude past the cap. An
/// `h` small enough to make the ratio *infinite* is a different answer — `BadArguments`, since an
/// infinite ratio is not a step count at all — and `driver::tests` holds that boundary. What this
/// holds is the wiring: every entry point must come straight back rather than start walking.
#[test]
fn a_step_that_cannot_walk_the_span_is_refused_before_the_loop() {
    fn is_limit<T>(r: &Result<T, IntegrationError>) -> bool {
        matches!(r, Err(IntegrationError::StepLimit { .. }))
    }
    let h = 1e-300;

    assert!(is_limit(&integrate_abm8(&Exponential, 0.0, &[1.0], 1.0, h)));
    assert!(is_limit(&integrate_abm8_dense(
        &Exponential,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_gj8(
        &Harmonic,
        0.0,
        &[1.0, 0.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_gj8_dense(
        &Harmonic,
        0.0,
        &[1.0, 0.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_radau5::<1, 3>(
        &Decay,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_radau5_dense::<1, 3>(
        &Decay,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_rosenbrock4(
        &Decay,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_rosenbrock4_dense(
        &Decay,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_adams_nordsieck(
        &Decay,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    assert!(is_limit(&integrate_ark(
        &Decay,
        &Decay,
        0.0,
        &[1.0],
        1.0,
        h
    )));
    let zero = |_t: f64, _y: &[f64; 1]| [0.0];
    assert!(is_limit(&integrate_mri(
        &zero,
        &zero,
        0.0,
        &[1.0],
        1.0,
        h,
        4
    )));
}

/// `y' = 0` — the constant solution. Every DOPRI5 stage derivative is exactly zero, so the embedded
/// estimate is exactly zero too, and the run walks the one branch of the step controller no other
/// fixture here reaches: `err > 0.0` is false and the growth factor comes out of `FACC2` rather than
/// out of `err.powf(expo1)`.
struct Constant;
impl DynamicalSystem<1> for Constant {
    fn derivative(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
        dydt[0] = 0.0;
    }
}

/// `arguments_ok` is five conjuncts and one test covered one of them. Each of the others has a
/// fall-through that ends somewhere else, so relaxing it does not hang or crash — it answers a
/// different question:
///
/// * an empty span leaves `while t < t1` false on the first look, so the driver returns an `Ok`
///   trajectory with no segments at all, and `Trajectory::eval` on that indexes
///   `segments.len() - 1` on an empty vector;
/// * a zero initial step walks the loop without advancing `t` and comes out as `StepUnderflow`;
/// * a zero floor makes `h < min_step` unreachable for any positive `h`, which disables the
///   underflow report and leaves only the step ceiling.
///
/// All three are `BadArguments`, and only an exact-variant assertion says so.
#[test]
fn the_driver_refuses_every_argument_it_documents() {
    let tol = Tolerance::new(1e-9, 1e-9);
    let cases = [
        ("a backwards span", 5.0, 0.0, 0.1, Controls::new(tol, 1e-9)),
        ("an empty span", 0.0, 0.0, 0.1, Controls::new(tol, 1e-9)),
        (
            "a zero initial step",
            0.0,
            1.0,
            0.0,
            Controls::new(tol, 1e-9),
        ),
        (
            "a negative initial step",
            0.0,
            1.0,
            -0.1,
            Controls::new(tol, 1e-9),
        ),
        ("a zero step floor", 0.0, 1.0, 0.1, Controls::new(tol, 0.0)),
        (
            "a negative step floor",
            0.0,
            1.0,
            0.1,
            Controls::new(tol, -1e-9),
        ),
        (
            "a non-finite tolerance",
            0.0,
            1.0,
            0.1,
            Controls::new(Tolerance::new(f64::NAN, 1e-9), 1e-9),
        ),
        (
            "no step ceiling at all",
            0.0,
            1.0,
            0.1,
            Controls::new(tol, 1e-9).with_max_steps(0),
        ),
    ];
    for (what, t0, t1, h0, controls) in cases {
        assert_eq!(
            integrate_dopri5(&Exponential, t0, &[1.0], t1, h0, controls).err(),
            Some(IntegrationError::BadArguments),
            "{what} was accepted"
        );
    }
}

/// The controller's zero-error branch, and the exact width of its own step floor.
///
/// On a solution the method reproduces exactly the error estimate is `0.0`, the `err > 0.0` test
/// takes its `else`, and the step grows by a fixed factor every accepted step —
/// `SAFE·facold^β / FACC2 = 0.9·(1e-4)^0.04 / 0.1 = 6.2265`, walking `[0, 1e6]` from `h = 1e-3` in
/// thirteen steps. Every mutation of that branch changes the factor and with it the count:
/// admitting `err == 0.0` into `err.powf(expo1)` gives `0^0.17 = 0`, which clamps to `FACC2` and
/// grows by the full cap of 10 (ten steps); dividing by the PI memory rather than multiplying gives
/// 10 as well, and taking its remainder gives 9 (eleven steps).
///
/// The floor is then pinned at its own boundary. `min_step` set to exactly the step the controller
/// arrives at must *not* trip `h < min_step` — a `<=` there would stop a run that is doing nothing
/// wrong, and the counts say it did not.
#[test]
fn a_constant_solution_grows_the_step_by_the_controller_and_not_by_its_cap() {
    let tol = Tolerance::new(1e-9, 1e-9);
    let (t1, h0) = (1.0e6, 1.0e-3);
    let traj = integrate_dopri5(&Constant, 0.0, &[1.0], t1, h0, Controls::new(tol, 1e-12))
        .expect("a constant solution is exactly integrable");
    assert_eq!(
        traj.step_counts(),
        (13, 0),
        "the zero-error growth factor moved"
    );
    let ends = traj.step_ends();
    let growth = (ends[1] - ends[0]) / (ends[0] - 0.0);
    assert!(
        (growth - 6.226_478_738_270_429).abs() < 1e-12,
        "zero-error growth factor {growth:.15} is not SAFE·facold^beta / FACC2"
    );

    let at_floor = integrate_dopri5(
        &Constant,
        0.0,
        &[1.0],
        t1,
        h0,
        Controls::new(tol, ends[1] - ends[0]),
    )
    .expect("a step exactly at the floor is not below it");
    assert_eq!(
        at_floor.step_counts(),
        (13, 0),
        "a floor equal to the step the controller chose changed the run"
    );
}

/// The underflow report is conditioned on `t < t1`, and that conjunct is the whole difference
/// between "the step collapsed" and "the run finished". A floor above whatever the controller
/// settles on has to stop a run that still has span left, and has to leave alone a run whose last
/// accepted step landed on `t1` — where `h` is below the floor too, and means nothing.
#[test]
fn a_step_floor_above_the_controller_stops_mid_span_and_not_at_the_end() {
    let floored = Controls::new(Tolerance::new(1e-9, 1e-9), 1.0);
    assert_eq!(
        integrate_dopri5(&Exponential, 0.0, &[1.0], 10.0, 0.1, floored).err(),
        Some(IntegrationError::StepUnderflow { t: 0.0 }),
        "a step floor above the controller's own step must stop the run"
    );

    let done = integrate_dopri5(&Exponential, 0.0, &[1.0], 0.05, 0.1, floored)
        .expect("a span the first step covers is finished, not underflowed");
    assert_eq!(done.step_counts(), (1, 0));
    assert_eq!(done.span(), (0.0, 0.05));
}

/// The PI controller's cost, pinned tightly enough to see its own arithmetic.
///
/// `dopri5_controller_keeps_step_rejections_rare_across_stiffness` bounds the rejection rate at 10%,
/// which every wrong controller measured here also meets. On one cell the counts separate cleanly:
/// this controller takes 411 steps and is refused 2, and dropping the PI memory term the wrong way
/// gives 390/5, taking the remainder by it gives 395/13, and multiplying by it rather than dividing
/// gives 386/9. Counting rejections as `rejected *= 1` gives 0. So the assertion is the rejection
/// count in its own right, not as a fraction, plus a band on the accepted count wide enough for a
/// `powf` that differs by an ulp between platforms.
#[test]
fn the_controller_pays_what_its_constants_say_it_should() {
    let traj = integrate_dopri5(
        &VanDerPol { mu: 5.0 },
        0.0,
        &[2.0, 0.0],
        10.0,
        1e-3,
        Controls::new(Tolerance::new(1e-9, 1e-9), 1e-12),
    )
    .expect("van der Pol at mu = 5 is smooth");
    let (accepted, rejected) = traj.step_counts();
    println!("van der Pol mu=5, tol=1e-9: {accepted} accepted, {rejected} rejected");
    assert!(
        (405..=417).contains(&accepted),
        "{accepted} accepted steps; this controller takes 411 and every wrong one measured takes \
         386-395"
    );
    assert!(
        (1..=2).contains(&rejected),
        "{rejected} rejected steps; this controller is refused 2, a wrong one 5-13, and a controller \
         that does not count them 0"
    );
}

/// The first step is capped by the *span*, not by a ratio of the two endpoints. Every adaptive
/// fixture in this file starts at `t = 0`, where `t1 / t0` is `+inf` and `h0.min(inf)` is `h0` — the
/// same number the span cap gives whenever `h0` fits. From a negative epoch the ratio is negative,
/// the driver would open with a backwards step, and `t` would walk away from `t1` until the step
/// ceiling stopped it.
#[test]
fn the_first_step_is_capped_by_the_span_and_not_by_a_ratio_of_its_ends() {
    let (t0, t1, h0) = (-2.0, 1.0, 0.5);
    let traj = integrate_dopri5(
        &Exponential,
        t0,
        &[1.0],
        t1,
        h0,
        Controls::new(Tolerance::new(1e-10, 1e-10), 1e-12),
    )
    .expect("a span that starts before the origin is still a forwards span");
    assert_eq!(traj.span(), (t0, t1));
    let err = (traj.final_state()[0] - (t0 - t1).exp()).abs();
    assert!(err < 1e-9, "y(1) off exp(-3) by {err:.3e}");
}

/// The fixed-step dense driver grids `t1 - t0`, not `t1 + t0`. Every other dense fixture starts at
/// `t = 0`, where the two are the same number and a grid built from the wrong one still lands on the
/// right answer; from `t0 = -3` the sum is `-2` and the grid runs backwards out of the span.
#[test]
fn the_fixed_step_dense_grid_spans_the_interval_and_not_the_endpoints() {
    let (t0, t1, h) = (-3.0, 1.0, 0.05);
    let traj = integrate_radau5_dense::<1, 3>(&Decay, t0, &[1.0], t1, h).expect("radau5 dense");
    assert_eq!(traj.span(), (t0, t1));
    let ends = traj.step_ends();
    assert_eq!(ends.len(), 80, "the span is 4.0 at a step of 0.05");
    assert!(
        (ends[0] - (t0 + h)).abs() < 1e-12,
        "first segment ends at {}, not {}",
        ends[0],
        t0 + h
    );
    assert!(
        (ends[ends.len() - 1] - t1).abs() < 1e-12,
        "last segment ends at {}, not {t1}",
        ends[ends.len() - 1]
    );
    let err = (traj.final_state()[0] - (t0 - t1).exp()).abs();
    assert!(err < 1e-9, "y(1) off exp(-4) by {err:.3e}");
}

/// `y' = y·cos t` in the autodiff-generic form the implicit methods take — [`Modulated`], for
/// [`ScalarSystem`].
struct ModulatedScalar;
impl ScalarSystem<1> for ModulatedScalar {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 1]) -> [S; 1] {
        [y[0] * t.cos()]
    }
}

/// The implicit fixed-step methods carry a clock that nothing read.
///
/// `radau_step` evaluates its stage residual at `t + cᵢ·h` and `rosenbrock_step` its second and
/// third stages at `t + c₂·h` and `t + c₃·h`; both loops size their step as `(t1 − t0)/n` and
/// advance it with `t += step`. Every fixture that reached them — `Decay` here,
/// `Newtonian(PointMass)` in `tests/radau.rs` and `tests/rosenbrock.rs`, the stiff scalar — drops
/// its time argument, and every call started at `t0 = 0`. That made fifteen separate wrong answers
/// indistinguishable from the right one: six stage displacements, the factor `cᵢ * h` inside them,
/// `(t1 − t0)` against `(t1 + t0)` — the same number from the origin — and `t += step` against
/// `t -= step` and `t *= step`, which reproduces it exactly from zero.
///
/// Same fixture and same reason as `rk4_reads_the_time_it_is_given`, two methods further in.
///
/// Radau IIA-5 keeps its order here — collocation evaluates the force at the stage times, so
/// non-autonomy is native to it. Measured relative error on `[1, 3]`: 2.1615e-9, 6.8407e-11,
/// 2.1468e-12 and 6.6648e-14 at n = 20, 40, 80, 160, ratios 31.6, 31.9 and 32.2 — the fifth order
/// it claims.
///
/// GRK4T did **not**, and that was a production defect this fixture found (#235): 2.2815e-3,
/// 1.1280e-3, 5.6075e-4, 2.7955e-4 and 1.3957e-4 at n = 20 … 320, every ratio 2.00–2.02 —
/// **order one, not four**, because `rosenbrock_step` omitted the `h·dᵢ·∂f/∂t` stage terms of the
/// reference `ros4.f`, and every force in `spacelab::forces` reads `t`. With the terms in place the
/// same ladder reads 4.3781e-7, 2.5937e-8, 1.5795e-9 and 9.8775e-11 at n = 20 … 160, ratios 16.88,
/// 16.42 and 15.99.
///
/// Both halves are pinned as a band rather than a ceiling: a displaced stage time moves the error
/// *constant* without moving the slope, so an upper bound alone would let any mutant through that
/// happens to make the answer smaller.
#[test]
fn the_implicit_fixed_step_methods_read_the_time_they_are_given() {
    let (t0, t1, y0) = (1.0_f64, 3.0_f64, 1.0_f64);
    let exact = modulated_exact(t0, y0, t1);
    let rel = |y: f64| ((y - exact) / exact).abs();

    let radau = |n: usize| {
        let h = (t1 - t0) / n as f64;
        rel(integrate_radau5::<1, 3>(&ModulatedScalar, t0, &[y0], t1, h).expect("radau5")[0])
    };
    let rosen = |n: usize| {
        let h = (t1 - t0) / n as f64;
        rel(integrate_rosenbrock4::<1>(&ModulatedScalar, t0, &[y0], t1, h).expect("grk4t")[0])
    };

    let (r80, r160) = (radau(80), radau(160));
    println!(
        "radau5 {r80:.4e} -> {r160:.4e}, order {:.2}",
        (r80 / r160).log2()
    );
    assert!(
        (4.6..5.4).contains(&(r80 / r160).log2()),
        "Radau IIA-5 on a system that reads t: order {:.2} from {r80:.3e} -> {r160:.3e}",
        (r80 / r160).log2()
    );
    assert!(
        (1.9e-12..2.4e-12).contains(&r80),
        "Radau IIA-5 relative error {r80:.4e} at n = 80 is not the measured 2.147e-12"
    );

    let (g20, g40, g80, g160) = (rosen(20), rosen(40), rosen(80), rosen(160));
    println!("grk4t {g20:.4e} {g40:.4e} {g80:.4e} {g160:.4e} (n = 20 … 160)");
    for (coarse, fine, n) in [(g20, g40, 40), (g40, g80, 80), (g80, g160, 160)] {
        let order = (coarse / fine).log2();
        assert!(
            (3.7..4.3).contains(&order),
            "GRK4T on a system that reads t: order {order:.2} at n = {n} from {coarse:.3e} -> \
             {fine:.3e} — with the h·dᵢ·∂f/∂t stage terms the expectation is four"
        );
    }
    assert!(
        (8.5e-11..1.15e-10).contains(&g160),
        "GRK4T relative error {g160:.4e} at n = 160 is not the measured 9.878e-11"
    );

    // The dense forms take the same steps through the same stage times, so they must land on the
    // same state; the interpolant is checked mid-span, where no step end can supply the answer.
    let (h, mid) = (0.05, 2.0);
    let dense_radau =
        integrate_radau5_dense::<1, 3>(&ModulatedScalar, t0, &[y0], t1, h).expect("radau5 dense");
    let dense_rosen =
        integrate_rosenbrock4_dense::<1>(&ModulatedScalar, t0, &[y0], t1, h).expect("grk4t dense");
    assert_eq!(
        dense_radau.final_state()[0].to_bits(),
        radau5_at(t0, y0, t1, h).to_bits()
    );
    assert_eq!(
        dense_rosen.final_state()[0].to_bits(),
        grk4t_at(t0, y0, t1, h).to_bits()
    );
    let mid_exact = modulated_exact(t0, y0, mid);
    let mid_rel = ((dense_radau.eval(mid)[0] - mid_exact) / mid_exact).abs();
    println!("radau5 dense at t = {mid}: {mid_rel:.3e} relative");
    assert!(
        mid_rel < 1.0e-9,
        "the Radau dense interpolant is off by {mid_rel:.3e} relative at t = {mid}"
    );
}

/// An empty span is a caller error, not a zero-length integration.
///
/// `step_count` rounds `(t1 − t0)/h` to zero and then clamps it to its floor of one, so a `t1 == t0`
/// that reaches it comes back as *one step of size zero*: the loop runs, the state is returned
/// unchanged, and the caller is told the propagation succeeded. The entry guard is the only thing
/// between those two readings, and every one of them is written `t1 > t0` — a single character from
/// `t1 >= t0`, and one `&&` from a disjunction that lets an empty span through whenever the step is
/// positive. Nothing here asserted the boundary, so both were free.
///
/// Radau carries a second guard nothing tested: `SN` is the flattened stage-array length and must be
/// `3N`. Given `SN ≠ 3N` the arrays are the wrong shape and the `SN×SN` Newton system is singular,
/// which surfaces as `StageSolveFailed` — a stage that did not converge, blamed on the problem
/// rather than on the caller's type arguments.
#[test]
fn the_fixed_step_entry_points_refuse_an_empty_span() {
    fn bad<T>(r: &Result<T, IntegrationError>) -> bool {
        matches!(r, Err(IntegrationError::BadArguments))
    }
    let (t, h) = (1.0, 0.1);

    assert!(bad(&integrate_abm8(&Exponential, t, &[1.0], t, h)));
    assert!(bad(&integrate_abm8_dense(&Exponential, t, &[1.0], t, h)));
    assert!(bad(&integrate_gj8(&Harmonic, t, &[1.0, 0.0], t, h)));
    assert!(bad(&integrate_gj8_dense(&Harmonic, t, &[1.0, 0.0], t, h)));
    assert!(bad(&integrate_radau5::<1, 3>(&Decay, t, &[1.0], t, h)));
    assert!(bad(&integrate_radau5_dense::<1, 3>(
        &Decay,
        t,
        &[1.0],
        t,
        h
    )));
    assert!(bad(&integrate_rosenbrock4(&Decay, t, &[1.0], t, h)));
    assert!(bad(&integrate_rosenbrock4_dense(&Decay, t, &[1.0], t, h)));
    assert!(bad(&integrate_adams_nordsieck(&Decay, t, &[1.0], t, h)));
    assert!(bad(&integrate_ark(&Decay, &Decay, t, &[1.0], t, h)));
    let zero = |_t: f64, _y: &[f64; 1]| [0.0];
    assert!(bad(&integrate_mri(&zero, &zero, t, &[1.0], t, h, 4)));

    // Not vacuous: the same calls over a span of one step return a state.
    assert!(integrate_radau5::<1, 3>(&Decay, t, &[1.0], t + h, h).is_ok());
    assert!(integrate_rosenbrock4(&Decay, t, &[1.0], t + h, h).is_ok());

    // `SN` must be `3N`, and saying so is the caller's error, not the stage solver's.
    assert!(bad(&integrate_radau5::<1, 4>(
        &Decay,
        t,
        &[1.0],
        t + 1.0,
        h
    )));
    assert!(bad(&integrate_radau5_dense::<1, 4>(
        &Decay,
        t,
        &[1.0],
        t + 1.0,
        h
    )));
}

/// GRK4T's `∂f/∂t` increment is measured from the epoch, and the epoch here is seconds from J2000.
///
/// `rosenbrock_step` takes `∂f/∂t` by a one-sided difference at `δ = √(u·max(1e-5,|t|))`, the
/// increment `ros4.f` picks for `IDFX = 0`. That heuristic sizes `δ` by the *independent variable*,
/// which is right when the problem is scaled so `|t| ≈ 1` and wrong here, where `t` is an absolute
/// epoch: at `t = 7.5e8` it asks for `δ = 4.1e-4 s` on a force whose only time-scale is the step.
///
/// The same ladder on `y' = y·cos t`, span 2 s, `n = 20 … 160`, measured:
///
/// | `t₀` | n = 20 | 40 | 80 | 160 | orders |
/// |---|---|---|---|---|---|
/// | 1 | 4.3781e-7 | 2.5937e-8 | 1.5795e-9 | 9.8775e-11 | 4.08, 4.04, 4.00 |
/// | 1e6 | 5.2521e-7 | 4.1066e-8 | 6.8916e-9 | 3.7164e-9 | 3.68, 2.58, 0.89 |
/// | 7.5e8 | 9.2998e-7 | 7.2808e-8 | 2.1303e-7 | 1.4644e-6 | 3.68, −1.55, −2.78 |
///
/// So order four survives only near the origin; by a J2000-scale epoch refinement makes the answer
/// *worse*, because the `δ`-truncation error in `∂f/∂t` outgrows the step error it is correcting.
/// The honest fix is not a better `δ` — every scale-free choice trades truncation for cancellation
/// somewhere. Nor is it AD: `ScalarSystem::eval` now takes `t` as the generic scalar, but a force
/// that reads the clock through an ephemeris or a density table takes `t.value()` and reports
/// `∂f/∂t = 0`, so differentiating would trade a truncation error for a silent zero on exactly the
/// forces the term exists for.
///
/// So this pins the behaviour, and pinning it is what kills `98:31`, `√(u·max(1e-5,|t|))`
/// against `√(u/max(1e-5,|t|))`. That mutant is invisible at `t₀ = 1` — 4.3781e-7 … 9.7644e-11, the
/// same four digits and the same order 4 — because at `|t| ≈ 1` the two increments differ by a
/// factor of one. It is only away from the origin that dividing collapses `δ` below `ulp(t)`, so
/// `t + δ == t`, `∂f/∂t` is identically zero, and the method drops to the order one of #235:
/// 1.3695e-3 … 1.8214e-4 at `t₀ = 1e6`, orders 0.95, 0.98, 0.99 — 49 000× the error this test
/// admits at `n = 160`.
#[test]
fn grk4t_takes_its_time_derivative_at_a_representable_increment() {
    let ladder = |t0: f64| {
        let (t1, y0) = (t0 + 2.0, 1.0);
        let exact = modulated_exact(t0, y0, t1);
        [20, 40, 80, 160].map(|n| {
            let y = grk4t_at(t0, y0, t1, 2.0 / f64::from(n));
            ((y - exact) / exact).abs()
        })
    };

    let mid = ladder(1.0e6);
    let order = (mid[0] / mid[1]).log2();
    println!(
        "grk4t at t0 = 1e6: {:.4e} {:.4e} {:.4e} {:.4e}, first order {order:.2}",
        mid[0], mid[1], mid[2], mid[3]
    );
    assert!(
        (3.4..4.0).contains(&order),
        "GRK4T at t0 = 1e6 starts at order {order:.2}, not the measured 3.68 — a δ that has \
         collapsed below ulp(t) leaves ∂f/∂t identically zero and the whole ladder at order one"
    );
    assert!(
        (2.0e-9..8.0e-9).contains(&mid[3]),
        "GRK4T at t0 = 1e6, n = 160: {:.4e} is not the measured 3.7164e-9",
        mid[3]
    );

    let far = ladder(7.5e8);
    println!(
        "grk4t at t0 = 7.5e8: {:.4e} {:.4e} {:.4e} {:.4e}",
        far[0], far[1], far[2], far[3]
    );
    assert!(
        (5.0e-7..5.0e-6).contains(&far[3]),
        "GRK4T at a J2000-scale epoch, n = 160: {:.4e} is not the measured 1.4644e-6",
        far[3]
    );
}

fn radau5_at(t0: f64, y0: f64, t1: f64, h: f64) -> f64 {
    integrate_radau5::<1, 3>(&ModulatedScalar, t0, &[y0], t1, h).expect("radau5")[0]
}
fn grk4t_at(t0: f64, y0: f64, t1: f64, h: f64) -> f64 {
    integrate_rosenbrock4::<1>(&ModulatedScalar, t0, &[y0], t1, h).expect("grk4t")[0]
}
