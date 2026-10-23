//! RKN12(10)17, checked by its order and against DOP853 — the coefficient transcription's proof.
//!
//! A wrong tableau entry drops the measured convergence order below twelve, so measuring order 12 is
//! the transcription check. The adaptive driver is then held to a tight DOP853 reference.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_propagation::driver::integrate;
use spacelab_propagation::driver::{Controls, DEFAULT_MAX_STEPS};
use spacelab_propagation::forces::{Acceleration, Newtonian, Plain, PointMass};
use spacelab_propagation::rkn1210;
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;

fn orbit() -> ([f64; 3], [f64; 3], f64) {
    let a = 8_000e3;
    let e = 0.1;
    let r_p = a * (1.0 - e);
    let v_p = (MU * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    ([r_p, 0.0, 0.0], [0.0, v_p, 0.0], period)
}

fn gravity(_t: f64, r: &[f64; 3]) -> [f64; 3] {
    PointMass { mu: MU }.acceleration::<f64>(0.0, r, &[0.0; 3])
}

fn reference(span: f64) -> [f64; 3] {
    let (r0, v0, _) = orbit();
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let yf = integrate(
        &Dop853,
        &Plain(Newtonian(PointMass { mu: MU })),
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-14, 1e-14), 1e-4),
    )
    .expect("reference")
    .final_state();
    [yf[0], yf[1], yf[2]]
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Final position after `steps` fixed RKN12(10) steps over `span`.
fn rkn_fixed(steps: usize, span: f64) -> [f64; 3] {
    let (r0, v0, _) = orbit();
    let (mut r, mut v) = (r0, v0);
    let h = span / steps as f64;
    for k in 0..steps {
        let (rn, vn, _, _) = rkn1210::step(&gravity, k as f64 * h, &r, &v, h);
        r = rn;
        v = vn;
    }
    r
}

#[test]
fn rkn1210_is_twelfth_order() {
    let (_, _, period) = orbit();
    let span = period;
    let reference = reference(span);
    // The order-12 window is narrow — error falls 4096× per step doubling and hits the roundoff
    // floor fast. Read the order at the coarsest usable pair, where both errors are well above it.
    let e_coarse = dist(&rkn_fixed(5, span), &reference);
    let e_fine = dist(&rkn_fixed(10, span), &reference);
    let order = (e_coarse / e_fine).log2();
    println!(
        "RKN12(10): err {e_coarse:.3e} → {e_fine:.3e} (5→10 steps/orbit), measured order {order:.2}"
    );
    assert!(
        e_fine > 1e-12,
        "fine step already at roundoff; pick coarser grids"
    );
    assert!(
        (order - 12.0).abs() < 1.0,
        "measured order {order:.2} is not 12"
    );
}

#[test]
fn the_adaptive_driver_matches_dop853() {
    let (r0, v0, period) = orbit();
    let span = 3.0 * period;
    let (r, _v) = rkn1210::integrate(
        &gravity,
        0.0,
        &r0,
        &v0,
        span,
        period / 20.0,
        Tolerance::new(1e-11, 1e-11),
        DEFAULT_MAX_STEPS,
    )
    .expect("three orbits fit the default step budget");
    let err = dist(&r, &reference(span));
    println!("RKN12(10) adaptive vs DOP853 over 3 orbits: {err:.3e} m");
    assert!(err < 1e-4, "adaptive RKN diverged from DOP853: {err:.3e} m");
}

#[test]
fn the_embedded_error_estimate_tracks_the_true_error() {
    // The (10) half exists to estimate the step error; its estimate must fall with h at roughly its
    // own order and never collapse to zero on a real step.
    let (r0, v0, period) = orbit();
    let (_, _, err_coarse, _) = rkn1210::step(&gravity, 0.0, &r0, &v0, period / 8.0);
    let (_, _, err_fine, _) = rkn1210::step(&gravity, 0.0, &r0, &v0, period / 16.0);
    let order = (err_coarse / err_fine).log2();
    println!("embedded error estimate: {err_coarse:.3e} → {err_fine:.3e}, order {order:.2}");
    assert!(
        err_coarse > 0.0 && err_fine > 0.0,
        "error estimate collapsed to zero"
    );
    assert!(
        order > 9.0,
        "embedded estimator order {order:.2} below its tenth-order design"
    );
}
