//! Adaptive Taylor integrator: exercised on a two-body orbit (nonlinear jet recurrences: √, 1/u,
//! uᵖ), a harmonic oscillator with an exact trigonometric solution, and a pendulum whose right-hand
//! side calls the jet `sin` recurrence, cross-checked against DOP853.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_autodiff::Scalar;
use spacelab_coverage::find_events;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::{DEFAULT_MAX_STEPS, IntegrationError, integrate};
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::taylor::{integrate_taylor, integrate_taylor_dense};
use spacelab_propagation::variational::ScalarSystem;
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;
const SEMI_MAJOR: f64 = 8_000e3;
const ECC: f64 = 0.1;

fn kepler(span: f64) -> [f64; 3] {
    let n = (MU / SEMI_MAJOR.powi(3)).sqrt();
    let m = n * span;
    let mut ea = m;
    for _ in 0..100 {
        ea -= (ea - ECC * ea.sin() - m) / (1.0 - ECC * ea.cos());
    }
    [
        SEMI_MAJOR * (ea.cos() - ECC),
        SEMI_MAJOR * (1.0 - ECC * ECC).sqrt() * ea.sin(),
        0.0,
    ]
}

#[test]
fn taylor_matches_the_exact_orbit() {
    // Order-20 Taylor on gravity, whose right-hand side puts the jet through √ (r), uᵖ (r⁻³) and the
    // Cauchy product. Adaptive step at tol 1e-13 over 5 orbits should hold to a few millimetres.
    let r_p = SEMI_MAJOR * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (SEMI_MAJOR * (1.0 - ECC))).sqrt();
    let period = std::f64::consts::TAU * (SEMI_MAJOR.powi(3) / MU).sqrt();
    let y0 = [r_p, 0.0, 0.0, 0.0, v_p, 0.0];
    let field = Newtonian(PointMass { mu: MU });
    let span = 5.0 * period;
    let yf = integrate_taylor::<6, 21>(&field, 0.0, &y0, span, 1e-13).expect("taylor");
    let exact = kepler(span);
    let err =
        ((yf[0] - exact[0]).powi(2) + (yf[1] - exact[1]).powi(2) + (yf[2] - exact[2]).powi(2))
            .sqrt();
    println!("Taylor(20) vs exact Kepler over 5 orbits: {err:.3e} m");
    assert!(err < 1e-2, "Taylor integrator not accurate: {err:.3e} m");
}

/// Harmonic oscillator `y'' = −ω² y` as a first-order system; exact solution is trigonometric.
struct Harmonic {
    omega2: f64,
}

impl ScalarSystem<2> for Harmonic {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 2]) -> [S; 2] {
        [y[1], y[0] * S::from_f64(-self.omega2)]
    }
}

#[test]
fn taylor_is_exact_on_a_harmonic_oscillator() {
    // x(0)=1, x'(0)=0 ⇒ x(t) = cos(ωt). A high-order Taylor method reproduces this to rounding.
    let omega = 3.0;
    let sys = Harmonic {
        omega2: omega * omega,
    };
    let t = 2.7;
    let yf = integrate_taylor::<2, 21>(&sys, 0.0, &[1.0, 0.0], t, 1e-14).expect("taylor");
    let exact = (omega * t).cos();
    println!(
        "Taylor SHM: x = {:.15}, cos(ωt) = {exact:.15}, err {:.2e}",
        yf[0],
        (yf[0] - exact).abs()
    );
    assert!(
        (yf[0] - exact).abs() < 1e-11,
        "SHM not exact: {}",
        (yf[0] - exact).abs()
    );
}

/// Pendulum `θ'' = −sin θ`: the right-hand side calls the jet `sin` recurrence.
struct Pendulum;

impl ScalarSystem<2> for Pendulum {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 2]) -> [S; 2] {
        [y[1], -y[0].sin()]
    }
}

#[test]
fn taylor_pendulum_agrees_with_dop853() {
    // Large-amplitude pendulum has no elementary solution; check the jet-sin path against DOP853.
    let sys = Pendulum;
    let y0 = [2.0, 0.0]; // released from 2 rad
    let span = 20.0;
    let taylor = integrate_taylor::<2, 21>(&sys, 0.0, &y0, span, 1e-13).expect("taylor");
    let dop = integrate(
        &Dop853,
        &Plain(Newtonian(PendulumAccel)),
        0.0,
        &pad(&y0),
        span,
        1.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-3),
    )
    .expect("reference")
    .final_state();
    let err = (taylor[0] - dop[0]).abs();
    println!(
        "Taylor pendulum θ = {:.9}, DOP853 θ = {:.9}, err {err:.2e}",
        taylor[0], dop[0]
    );
    assert!(
        err < 1e-8,
        "pendulum jet-sin path disagrees with DOP853: {err:.2e}"
    );
}

/// The pendulum as a 3-D "acceleration" so it can reuse the `Newtonian`/DOP853 path (only x used).
struct PendulumAccel;
impl spacelab_propagation::forces::Acceleration for PendulumAccel {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        [-r[0].sin(), S::from_f64(0.0), S::from_f64(0.0)]
    }
}
fn pad(y: &[f64; 2]) -> [f64; 6] {
    [y[0], 0.0, 0.0, y[1], 0.0, 0.0]
}

#[test]
fn taylor_dense_composes_with_the_event_detector() {
    // The dense-output payoff: propagate once, then let the coverage event detector find crossings
    // by sampling the trajectory at arbitrary t. g(t) = x-coordinate crosses zero twice per orbit
    // (perigee at +x, apogee at −x), so two orbits give four events, alternating open/close.
    let r_p = SEMI_MAJOR * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (SEMI_MAJOR * (1.0 - ECC))).sqrt();
    let period = std::f64::consts::TAU * (SEMI_MAJOR.powi(3) / MU).sqrt();
    let y0 = [r_p, 0.0, 0.0, 0.0, v_p, 0.0];
    let field = Newtonian(PointMass { mu: MU });
    let span = 2.0 * period;
    let traj =
        integrate_taylor_dense::<6, 21>(&field, 0.0, &y0, span, 1e-12).expect("taylor dense");

    let mut events = Vec::new();
    let n = find_events(
        |t| traj.eval(t)[0],
        0.0,
        span,
        period / 20.0,
        1e-4,
        |e| events.push(e),
    );
    println!("Taylor-dense → find_events: {n} node crossings over 2 orbits");
    assert_eq!(n, 4, "expected 4 x-axis crossings over 2 orbits, got {n}");
    // Each located crossing must actually be a zero of the interpolated x-coordinate.
    for e in &events {
        let x = traj.eval(e.at.estimate())[0];
        assert!(
            x.abs() < 100.0,
            "event not at a zero crossing: x = {x:.3e} m"
        );
    }
    // The dense interpolant must agree with the final state at t1.
    let end = traj.eval(span);
    let f = traj.final_state();
    assert!(
        (end[0] - f[0]).abs() < 1e-6,
        "dense eval disagrees with final state"
    );
}

#[test]
fn taylor_rejects_bad_arguments() {
    let field = Newtonian(PointMass { mu: MU });
    let y0 = [7e6, 0.0, 0.0, 0.0, 7e3, 0.0];
    assert!(integrate_taylor::<6, 21>(&field, 0.0, &y0, -1.0, 1e-12).is_err());
    assert!(integrate_taylor::<6, 21>(&field, 0.0, &y0, 100.0, 0.0).is_err());
}

/// `y' = y`, whose jets are `c[k] = y/k!` — the shortest fixture that makes the step estimate small
/// without making it *zero*.
struct Exponential;
impl ScalarSystem<1> for Exponential {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
        [y[0]]
    }
}

/// A tolerance below the epoch's resolution is caught by `advances`; a tolerance *above* it is not.
///
/// At `M = 3` the tail estimate is `h = 0.9·min((tol/|c₂|)^(1/2), tol/|c₁|)`, and for `y' = y` at
/// `y = 1` that is `0.9·min(1.41e-15, 1e-30) = 9e-31`. From `t0 = 0` the ulp is `4.9e-324`, so every
/// one of those steps genuinely advances the clock and `StepUnderflow` never fires — the loop simply
/// needs `1.1e30` of them to cross a span of `1`. `advances` bounds how small a step may be, not how
/// many the span holds, and the two are not the same bound: the step floor is `ulp(t)/2`, which
/// depends on the epoch and is `2.2e-308 s` near zero. Without a step count this call does not
/// return in any human time, and the caller cannot interrupt it.
#[test]
fn taylor_stops_at_the_step_limit_rather_than_running_forever() {
    let err = integrate_taylor::<1, 3>(&Exponential, 0.0, &[1.0], 1.0, 1e-30).unwrap_err();
    assert!(
        matches!(err, IntegrationError::StepLimit { steps }
            if (steps - f64::from(DEFAULT_MAX_STEPS)).abs() < 0.5),
        "expected StepLimit at {DEFAULT_MAX_STEPS} steps, got {err:?}"
    );

    let err = integrate_taylor_dense::<1, 3>(&Exponential, 0.0, &[1.0], 1.0, 1e-30).unwrap_err();
    assert!(
        matches!(err, IntegrationError::StepLimit { steps }
            if (steps - f64::from(DEFAULT_MAX_STEPS)).abs() < 0.5),
        "expected StepLimit at {DEFAULT_MAX_STEPS} steps, got {err:?}"
    );

    // Non-vacuity: the same system at a workable tolerance crosses the span well inside the cap.
    let y = integrate_taylor::<1, 12>(&Exponential, 0.0, &[1.0], 1.0, 1e-14).unwrap();
    assert!(
        (y[0] - 1.0_f64.exp()).abs() < 1e-12,
        "exp(1) came out {}",
        y[0]
    );
}

/// The step floor is the epoch's, not a constant, and `advances` is the only thing that reads it.
///
/// The same system and tolerance as the cap test above, moved from `t = 0` to `t = 1e9` — a J2000
/// epoch. The tail estimate does not depend on `t` and still asks for `h = 9e-31`, but `ulp(1e9)/2`
/// is `1.2e-7`, so `t + h` is now exactly `t` and the step buys nothing. Nothing else in the crate
/// distinguishes `t + h > t` from `t + h >= t` or from `true`: at `t = 0` the ulp is `4.9e-324` and
/// every step the estimate returns advances the clock, which is why the cap test cannot see this
/// and this test cannot see the cap.
#[test]
fn taylor_reports_underflow_when_the_step_is_below_the_epochs_own_resolution() {
    let t0 = 1.0e9;
    let expect = |err: IntegrationError| {
        assert!(
            matches!(err, IntegrationError::StepUnderflow { t } if (t - t0).abs() < 1.0),
            "expected StepUnderflow at t = {t0:e}, got {err:?}"
        );
    };
    expect(integrate_taylor::<1, 3>(&Exponential, t0, &[1.0], t0 + 1.0, 1e-30).unwrap_err());
    expect(integrate_taylor_dense::<1, 3>(&Exponential, t0, &[1.0], t0 + 1.0, 1e-30).unwrap_err());

    // Non-vacuity: at the same epoch a workable tolerance still crosses the span. The residual is
    // 9.88e-8, five orders above the 1e-12 the same call returns from `t0 = 0`, and it is the
    // epoch's rather than the method's: `t += h` rounds to `ulp(1e9) = 1.2e-7` each step, so the
    // step the jets are evaluated on and the step the clock actually took differ by that much.
    let y = integrate_taylor::<1, 12>(&Exponential, t0, &[1.0], t0 + 1.0, 1e-14).unwrap();
    assert!(
        (y[0] - 1.0_f64.exp()).abs() < 2.0e-7,
        "exp(1) came out {}",
        y[0]
    );
}
