//! Gauss–Jackson-8, checked for order 8 against an exact two-body reference, and cross-checked
//! against DOP853 on a Kepler orbit.
//!
//! The reference is the analytic Kepler solution, not another integrator: for a pure `PointMass`
//! orbit the position is exact, so the order measurement is not floored by a reference integrator's
//! own error. The method's own floor is DOP853's ~1e-7 m global error at the startup tolerance —
//! well below any grid used here.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_coverage::find_events;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::gj::{integrate_gj8, integrate_gj8_dense};
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

/// Exact two-body position after `span`, starting from perigee on +x (so `M0 = 0` and the perifocal
/// frame coincides with the integration frame). Solves Kepler's equation by Newton iteration.
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

fn dist(y: &[f64; 6], r: &[f64; 3]) -> f64 {
    ((y[0] - r[0]).powi(2) + (y[1] - r[1]).powi(2) + (y[2] - r[2]).powi(2)).sqrt()
}

fn err_at(span: f64, steps: usize) -> f64 {
    let (y0, _) = orbit();
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let yf = integrate_gj8(&field, 0.0, &y0, span, span / steps as f64).expect("gj");
    dist(&yf, &kepler(span))
}

#[test]
fn gj8_converges_at_order_eight() {
    let (_, period) = orbit();
    let span = period;
    // Asymptotic window: both errors are far above the ~1e-7 m startup floor, so the ratio measures
    // truncation. An order-8 method reads slightly under 8 here (the h⁹ term has not yet vanished),
    // exactly as yoshida6 reads 5.96 — the derivation in gauss_jackson_coeffs.mac proves order 8.
    let coarse = err_at(span, 120);
    let fine = err_at(span, 240);
    let order = (coarse / fine).log2();
    println!("GJ8: err {coarse:.3e} → {fine:.3e} (120→240 steps), order {order:.2}");
    assert!(
        fine > 1e-6,
        "fine grid at the startup floor; use coarser grids"
    );
    assert!(
        (order - 8.0).abs() < 0.6,
        "measured order {order:.2} is not 8"
    );
}

#[test]
fn gj8_agrees_with_dop853_over_several_orbits() {
    let (y0, period) = orbit();
    let span = 3.0 * period;
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let yf = integrate_gj8(&field, 0.0, &y0, span, period / 400.0).expect("gj");
    let dop = integrate(
        &Dop853,
        &field,
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-14, 1e-14), 1e-4),
    )
    .expect("reference")
    .final_state();
    let err = dist(&yf, &[dop[0], dop[1], dop[2]]);
    println!("GJ8 vs DOP853 over 3 orbits: {err:.3e} m");
    assert!(err < 1e-3, "GJ8 diverged from DOP853: {err:.3e} m");
}

#[test]
fn gj8_dense_matches_plain_and_finds_events() {
    let (y0, period) = orbit();
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let span = 2.0 * period;
    let h = period / 300.0;
    let plain = integrate_gj8(&field, 0.0, &y0, span, h).expect("gj");
    let traj = integrate_gj8_dense(&field, 0.0, &y0, span, h).expect("dense");

    let f = traj.final_state();
    let dfinal =
        ((f[0] - plain[0]).powi(2) + (f[1] - plain[1]).powi(2) + (f[2] - plain[2]).powi(2)).sqrt();
    assert!(
        dfinal < 1e-6,
        "dense final state differs from plain: {dfinal:.3e} m"
    );

    let mid = traj.eval(0.37 * span);
    let exact = kepler(0.37 * span);
    let dmid =
        ((mid[0] - exact[0]).powi(2) + (mid[1] - exact[1]).powi(2) + (mid[2] - exact[2]).powi(2))
            .sqrt();
    println!("GJ8 Hermite-dense interior error: {dmid:.3e} m");
    assert!(
        dmid < 100.0,
        "Hermite interior interpolation off: {dmid:.3e} m"
    );

    let n = find_events(|t| traj.eval(t)[0], 0.0, span, period / 20.0, 1e-3, |_| {});
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
}

#[test]
fn gj8_holds_a_tight_track_over_many_orbits() {
    // The second-sum form's payoff is a low round-off floor: 25 orbits at a fine step stay within a
    // metre of the exact orbit, error growing far slower than a naive Störmer–Cowell would.
    let (y0, period) = orbit();
    let span = 25.0 * period;
    let field = Plain(Newtonian(PointMass { mu: MU }));
    let yf = integrate_gj8(&field, 0.0, &y0, span, period / 600.0).expect("gj");
    let err = dist(&yf, &kepler(span));
    println!("GJ8 vs exact Kepler over 25 orbits: {err:.3e} m");
    assert!(err < 1.0, "GJ8 drifted over 25 orbits: {err:.3e} m");
}
