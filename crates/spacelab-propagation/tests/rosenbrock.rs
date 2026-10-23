//! Rosenbrock GRK4T: order 4 against an exact Kepler reference, agreement with DOP853, and stable
//! integration of a stiff decay where an explicit method of the same order would blow up.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_autodiff::Scalar;
use spacelab_coverage::find_events;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::rosenbrock::{integrate_rosenbrock4, integrate_rosenbrock4_dense};
use spacelab_propagation::variational::ScalarSystem;
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;
const SEMI_MAJOR: f64 = 8_000e3;
const ECC: f64 = 0.1;

fn orbit() -> ([f64; 6], f64) {
    let r_p = SEMI_MAJOR * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (SEMI_MAJOR * (1.0 - ECC))).sqrt();
    let period = std::f64::consts::TAU * (SEMI_MAJOR.powi(3) / MU).sqrt();
    ([r_p, 0.0, 0.0, 0.0, v_p, 0.0], period)
}

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

fn err_at(span: f64, steps: usize) -> f64 {
    let (y0, _) = orbit();
    let field = Newtonian(PointMass { mu: MU });
    let yf = integrate_rosenbrock4::<6>(&field, 0.0, &y0, span, span / steps as f64)
        .expect("rosenbrock");
    let r = kepler(span);
    ((yf[0] - r[0]).powi(2) + (yf[1] - r[1]).powi(2) + (yf[2] - r[2]).powi(2)).sqrt()
}

#[test]
fn grk4t_converges_at_order_four() {
    let (_, period) = orbit();
    let span = period;
    // Order approaches 4 from above (4.6 at 40→80, 4.09 at 640→1280); measured on a fine grid.
    let coarse = err_at(span, 640);
    let fine = err_at(span, 1280);
    let order = (coarse / fine).log2();
    println!("GRK4T: err {coarse:.3e} → {fine:.3e} (640→1280 steps), order {order:.2}");
    assert!(
        (order - 4.0).abs() < 0.3,
        "measured order {order:.2} is not 4"
    );
}

#[test]
fn grk4t_agrees_with_dop853_over_several_orbits() {
    let (y0, period) = orbit();
    let span = 3.0 * period;
    let field = Newtonian(PointMass { mu: MU });
    let yf =
        integrate_rosenbrock4::<6>(&field, 0.0, &y0, span, period / 2000.0).expect("rosenbrock");
    let dop = integrate(
        &Dop853,
        &Plain(field),
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-14, 1e-14), 1e-4),
    )
    .expect("reference")
    .final_state();
    let err =
        ((yf[0] - dop[0]).powi(2) + (yf[1] - dop[1]).powi(2) + (yf[2] - dop[2]).powi(2)).sqrt();
    println!("GRK4T vs DOP853 over 3 orbits: {err:.3e} m");
    assert!(err < 1e-2, "GRK4T diverged from DOP853: {err:.3e} m");
}

#[test]
fn grk4t_dense_matches_plain_and_finds_events() {
    let (y0, period) = orbit();
    let field = Newtonian(PointMass { mu: MU });
    let span = 2.0 * period;
    let step = period / 800.0;
    let plain = integrate_rosenbrock4::<6>(&field, 0.0, &y0, span, step).expect("rosenbrock");
    let traj = integrate_rosenbrock4_dense::<6>(&field, 0.0, &y0, span, step).expect("dense");

    let f = traj.final_state();
    let dfinal =
        ((f[0] - plain[0]).powi(2) + (f[1] - plain[1]).powi(2) + (f[2] - plain[2]).powi(2)).sqrt();
    assert!(
        dfinal < 1e-9,
        "dense final state differs from plain: {dfinal:.3e} m"
    );

    let mid = traj.eval(0.37 * span);
    let exact = kepler(0.37 * span);
    let dmid =
        ((mid[0] - exact[0]).powi(2) + (mid[1] - exact[1]).powi(2) + (mid[2] - exact[2]).powi(2))
            .sqrt();
    println!("GRK4T Hermite-dense interior error: {dmid:.3e} m");
    assert!(
        dmid < 100.0,
        "Hermite interior interpolation off: {dmid:.3e} m"
    );

    let n = find_events(|t| traj.eval(t)[0], 0.0, span, period / 20.0, 1e-3, |_| {});
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
}

/// The scalar stiff decay `y' = −λ y`, exact solution `e^{−λt}`.
struct StiffDecay {
    lambda: f64,
}

impl ScalarSystem<1> for StiffDecay {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
        [y[0] * S::from_f64(-self.lambda)]
    }
}

#[test]
fn grk4t_is_stable_on_a_stiff_decay() {
    // λh = 5 on the negative real axis: an explicit 4th-order method (RK4 stability limit ≈ 2.79)
    // would amplify and blow up. GRK4T is A(89.3°)-stable — the whole negative real axis is inside
    // its stability region — so the solution decays and stays finite without any step reduction.
    let lambda = 1.0e4;
    let span = 1.0e-2; // λ·span = 100 ⇒ exact e^{−100} ≈ 0
    let steps = 20; // λ·h = 5
    let y = integrate_rosenbrock4::<1>(
        &StiffDecay { lambda },
        0.0,
        &[1.0],
        span,
        span / f64::from(steps),
    )
    .expect("rosenbrock");
    println!("GRK4T stiff decay (λh=5): y = {:.3e} (exact ≈ 0)", y[0]);
    assert!(y[0].is_finite(), "blew up: {}", y[0]);
    assert!(
        y[0].abs() < 1e-2,
        "not stable/decaying on the stiff mode: y = {:.3e}",
        y[0]
    );

    // Moderate step where the exact value is representable: λh = 0.1, exact e^{−3} = 0.0498.
    let y2 = integrate_rosenbrock4::<1>(&StiffDecay { lambda: 100.0 }, 0.0, &[1.0], 0.03, 0.001)
        .expect("rosenbrock");
    let exact = (-3.0_f64).exp();
    let rel = (y2[0] - exact).abs() / exact;
    println!(
        "GRK4T stiff decay (λh=0.1): y = {:.6}, exact {exact:.6}, rel err {rel:.2e}",
        y2[0]
    );
    assert!(
        rel < 1e-4,
        "order-4 accuracy lost on the stiff decay: rel err {rel:.2e}"
    );
}
