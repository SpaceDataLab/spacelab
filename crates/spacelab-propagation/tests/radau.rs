//! Radau IIA-5: order 5 against an exact Kepler reference, agreement with DOP853, and the property
//! that motivates an implicit method — L-stability on a stiff decay where an explicit method of the
//! same order would need a far smaller step to avoid blowing up.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_autodiff::Scalar;
use spacelab_coverage::find_events;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::radau::{integrate_radau5, integrate_radau5_dense};
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
    let yf = integrate_radau5::<6, 18>(&field, 0.0, &y0, span, span / steps as f64).expect("radau");
    let r = kepler(span);
    ((yf[0] - r[0]).powi(2) + (yf[1] - r[1]).powi(2) + (yf[2] - r[2]).powi(2)).sqrt()
}

#[test]
fn radau5_converges_at_order_five() {
    let (_, period) = orbit();
    let span = period;
    let coarse = err_at(span, 40);
    let fine = err_at(span, 80);
    let order = (coarse / fine).log2();
    println!("Radau5: err {coarse:.3e} → {fine:.3e} (40→80 steps), order {order:.2}");
    assert!(
        (order - 5.0).abs() < 0.3,
        "measured order {order:.2} is not 5"
    );
}

#[test]
fn radau5_agrees_with_dop853_over_several_orbits() {
    let (y0, period) = orbit();
    let span = 3.0 * period;
    let field = Newtonian(PointMass { mu: MU });
    let yf = integrate_radau5::<6, 18>(&field, 0.0, &y0, span, period / 500.0).expect("radau");
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
    println!("Radau5 vs DOP853 over 3 orbits: {err:.3e} m");
    assert!(err < 1e-2, "Radau5 diverged from DOP853: {err:.3e} m");
}

#[test]
fn radau5_dense_matches_plain_and_finds_events() {
    let (_, period) = orbit();
    let (y0, _) = orbit();
    let field = Newtonian(PointMass { mu: MU });
    let span = 2.0 * period;
    let step = period / 400.0;
    let plain = integrate_radau5::<6, 18>(&field, 0.0, &y0, span, step).expect("radau");
    let traj = integrate_radau5_dense::<6, 18>(&field, 0.0, &y0, span, step).expect("dense");

    let f = traj.final_state();
    let dfinal =
        ((f[0] - plain[0]).powi(2) + (f[1] - plain[1]).powi(2) + (f[2] - plain[2]).powi(2)).sqrt();
    assert!(
        dfinal < 1e-9,
        "dense final state differs from plain: {dfinal:.3e} m"
    );

    // Cubic-Hermite interior interpolation over these fine steps stays close to the exact orbit.
    let mid = traj.eval(0.37 * span);
    let exact = kepler(0.37 * span);
    let dmid =
        ((mid[0] - exact[0]).powi(2) + (mid[1] - exact[1]).powi(2) + (mid[2] - exact[2]).powi(2))
            .sqrt();
    println!("Radau5 Hermite-dense interior error: {dmid:.3e} m");
    assert!(
        dmid < 100.0,
        "Hermite interior interpolation off: {dmid:.3e} m"
    );

    let n = find_events(|t| traj.eval(t)[0], 0.0, span, period / 20.0, 1e-3, |_| {});
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
}

/// The scalar stiff decay `y' = −λ y`, whose exact solution is `e^{−λt}`.
struct StiffDecay {
    lambda: f64,
}

impl ScalarSystem<1> for StiffDecay {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
        [y[0] * S::from_f64(-self.lambda)]
    }
}

#[test]
fn radau5_is_l_stable_on_a_stiff_decay() {
    // λh = 5 per step: the amplification factor of any explicit 5th-order method at z = −5 has
    // magnitude ≫ 1, so it would blow up. Radau IIA is L-stable — |R(z)| < 1 on the whole negative
    // axis and R(−∞) = 0 — so the numerical solution decays monotonically toward the exact e^{−λt}.
    let lambda = 1.0e4;
    let span = 1.0e-2; // λ·span = 100 ⇒ exact e^{−100} ≈ 3.7e−44 ≈ 0
    let steps = 20; // λ·h = 5
    let y = integrate_radau5::<1, 3>(
        &StiffDecay { lambda },
        0.0,
        &[1.0],
        span,
        span / f64::from(steps),
    )
    .expect("radau");
    println!("Radau5 stiff decay (λh=5): y = {:.3e} (exact ≈ 0)", y[0]);
    assert!(y[0].is_finite(), "blew up: {}", y[0]);
    assert!(
        y[0] >= 0.0 && y[0] < 1e-6,
        "not L-stable decay: y = {:.3e}",
        y[0]
    );

    // A moderate step where the exact value is representable: λh = 0.1, exact e^{−3} = 0.0498.
    let y2 = integrate_radau5::<1, 3>(&StiffDecay { lambda: 100.0 }, 0.0, &[1.0], 0.03, 0.001)
        .expect("radau");
    let exact = (-3.0_f64).exp();
    let rel = (y2[0] - exact).abs() / exact;
    println!(
        "Radau5 stiff decay (λh=0.1): y = {:.6}, exact {exact:.6}, rel err {rel:.2e}",
        y2[0]
    );
    assert!(
        rel < 1e-6,
        "order-5 accuracy lost on the stiff decay: rel err {rel:.2e}"
    );
}
