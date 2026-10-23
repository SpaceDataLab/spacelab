//! Adams–Bashforth–Moulton-8, checked for order 8 and against DOP853 on a Kepler orbit.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_coverage::find_events;
use spacelab_propagation::abm::{integrate_abm8, integrate_abm8_dense};
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;

fn orbit() -> ([f64; 6], f64) {
    let a = 8_000e3;
    let e = 0.1;
    let r_p = a * (1.0 - e);
    let v_p = (MU * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    ([r_p, 0.0, 0.0, 0.0, v_p, 0.0], period)
}

fn reference(span: f64) -> [f64; 6] {
    let (y0, _) = orbit();
    integrate(
        &Dop853,
        &Plain(Newtonian(PointMass { mu: MU })),
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-14, 1e-14), 1e-4),
    )
    .expect("reference")
    .final_state()
}

fn err_at(span: f64, steps: usize) -> f64 {
    let (y0, _) = orbit();
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let yf = integrate_abm8(&field, 0.0, &y0, span, span / steps as f64).expect("abm");
    let r = reference(span);
    ((yf[0] - r[0]).powi(2) + (yf[1] - r[1]).powi(2) + (yf[2] - r[2]).powi(2)).sqrt()
}

#[test]
fn abm8_converges_at_order_eight() {
    let (_, period) = orbit();
    let span = period;
    let coarse = err_at(span, 40);
    let fine = err_at(span, 80);
    let order = (coarse / fine).log2();
    println!("ABM8: err {coarse:.3e} → {fine:.3e} (40→80 steps), order {order:.2}");
    assert!(fine > 1e-12, "already at roundoff; use coarser grids");
    assert!(
        (order - 8.0).abs() < 1.0,
        "measured order {order:.2} is not 8"
    );
}

#[test]
fn abm8_agrees_with_dop853_over_several_orbits() {
    let (y0, period) = orbit();
    let span = 3.0 * period;
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let yf = integrate_abm8(&field, 0.0, &y0, span, period / 400.0).expect("abm");
    let r = reference(span);
    let err = ((yf[0] - r[0]).powi(2) + (yf[1] - r[1]).powi(2) + (yf[2] - r[2]).powi(2)).sqrt();
    println!("ABM8 vs DOP853 over 3 orbits: {err:.3e} m");
    assert!(err < 1e-3, "ABM8 diverged from DOP853: {err:.3e} m");
}

#[test]
fn abm8_dense_matches_plain_and_finds_events() {
    let (y0, period) = orbit();
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let span = 2.0 * period;
    let h = period / 400.0;
    let plain = integrate_abm8(&field, 0.0, &y0, span, h).expect("abm");
    let traj = integrate_abm8_dense(&field, 0.0, &y0, span, h).expect("dense");

    let f = traj.final_state();
    let dfinal =
        ((f[0] - plain[0]).powi(2) + (f[1] - plain[1]).powi(2) + (f[2] - plain[2]).powi(2)).sqrt();
    assert!(
        dfinal < 1e-6,
        "dense final state differs from plain: {dfinal:.3e} m"
    );

    let mid = traj.eval(0.37 * span);
    let r = reference(0.37 * span);
    let dmid = ((mid[0] - r[0]).powi(2) + (mid[1] - r[1]).powi(2) + (mid[2] - r[2]).powi(2)).sqrt();
    println!("ABM8 Hermite-dense interior error: {dmid:.3e} m");
    assert!(
        dmid < 100.0,
        "Hermite interior interpolation off: {dmid:.3e} m"
    );

    let n = find_events(|t| traj.eval(t)[0], 0.0, span, period / 20.0, 1e-3, |_| {});
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
}
