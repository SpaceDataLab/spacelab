//! Adams–Nordsieck (order 5): convergence at order 5 on a Kepler orbit, agreement with DOP853, and
//! the variable-step property — rescaling the Nordsieck vector to change the step is consistent.
//! Those three all drive an autonomous force from `t = 0`, so a pure quadrature `y' = g(t)` carries
//! the rest: the closed form of a single predict–evaluate–correct step, the RMS that step returns,
//! and the epoch and direction of the clock the integrator keeps.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::adams_nordsieck::{Nordsieck, integrate_adams_nordsieck};
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::{IntegrationError, integrate};
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
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
    let yf = integrate_adams_nordsieck::<6>(&field, 0.0, &y0, span, span / steps as f64)
        .expect("nordsieck");
    let e = kepler(span);
    ((yf[0] - e[0]).powi(2) + (yf[1] - e[1]).powi(2) + (yf[2] - e[2]).powi(2)).sqrt()
}

#[test]
fn adams_nordsieck_converges_at_order_five() {
    let (_, period) = orbit();
    let span = period;
    let coarse = err_at(span, 80);
    let fine = err_at(span, 160);
    let order = (coarse / fine).log2();
    println!("Adams-Nordsieck: err {coarse:.3e} → {fine:.3e} (80→160 steps), order {order:.2}");
    assert!(fine > 1e-10, "already at roundoff; use coarser grids");
    assert!(
        (order - 5.0).abs() < 0.6,
        "measured order {order:.2} is not 5"
    );
}

#[test]
fn adams_nordsieck_agrees_with_dop853() {
    let (y0, period) = orbit();
    let field = Newtonian(PointMass { mu: MU });
    let span = 2.0 * period;
    let yf =
        integrate_adams_nordsieck::<6>(&field, 0.0, &y0, span, period / 400.0).expect("nordsieck");
    let dop = integrate(
        &Dop853,
        &Plain(field),
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-4),
    )
    .expect("reference")
    .final_state();
    let err =
        ((yf[0] - dop[0]).powi(2) + (yf[1] - dop[1]).powi(2) + (yf[2] - dop[2]).powi(2)).sqrt();
    println!("Adams-Nordsieck vs DOP853 over 2 orbits: {err:.3e} m");
    assert!(err < 2.0, "disagrees with DOP853: {err:.3e} m");
}

#[test]
fn rescaling_the_nordsieck_vector_changes_the_step_consistently() {
    // The variable-step property: advancing one step of h, then rescaling to h/2 and advancing two
    // steps of h/2, reaches almost the same state as three steps of a matched fixed grid — the
    // rescale correctly re-expresses the scaled derivatives for the new step.
    let (y0, period) = orbit();
    let field = Newtonian(PointMass { mu: MU });
    let h = period / 400.0;

    let mut ns = Nordsieck::<6>::start(&field, 0.0, &y0, h);
    ns.step(&field, 0.0); // one step of h → t = h
    ns.rescale(h / 2.0); // switch to h/2 with no re-interpolation
    ns.step(&field, h);
    ns.step(&field, h + h / 2.0); // two steps of h/2 → t = 2h
    let mixed = ns.value();

    // Reference: the same span [0, 2h] at a uniform fine step.
    let refr = integrate_adams_nordsieck::<6>(&field, 0.0, &y0, 2.0 * h, h / 8.0).expect("ref");
    let d = ((mixed[0] - refr[0]).powi(2)
        + (mixed[1] - refr[1]).powi(2)
        + (mixed[2] - refr[2]).powi(2))
    .sqrt();
    println!("Nordsieck rescale consistency over 2h: {d:.3e} m");
    assert!(d < 1.0, "rescaling the step was not consistent: {d:.3e} m");
}

#[test]
fn adams_nordsieck_rejects_bad_arguments() {
    let (y0, _) = orbit();
    let field = Newtonian(PointMass { mu: MU });
    for (t0, t1, h) in [
        (0.0, -1.0, 1.0),   // backwards span
        (0.0, 100.0, 0.0),  // zero step
        (0.0, 100.0, -1.0), // backwards step over a forwards span
        (7.5, 7.5, 1.0),    // empty span
        (0.0, 0.0, 1.0),    // empty span at the origin
    ] {
        assert_eq!(
            integrate_adams_nordsieck::<6>(&field, t0, &y0, t1, h),
            Err(IntegrationError::BadArguments),
            "accepted t0={t0} t1={t1} h={h}"
        );
    }
}

/// `y' = g(t)` — a pure quadrature whose right-hand side never reads `y`, with three components of
/// deliberately different magnitude. It is the only fixture in this file that makes the *time*
/// argument observable at all: `Newtonian(PointMass)` is autonomous, so nothing above can tell
/// which instant the corrector samples the force at, nor which way the clock runs.
struct Quadrature;

const G_AMP: [f64; 3] = [0.75, 2.5, 11.0];
const G_RATE: [f64; 3] = [1.3, 0.41, 0.17];
const G_BIAS: [f64; 3] = [0.2, -1.1, 3.0];

/// The Nordsieck l-vector's leading entry, the Adams–Moulton `β₀`.
const L0: f64 = 251.0 / 720.0;

fn g<S: Scalar>(t: S) -> [S; 3] {
    let mut out = [S::from_f64(0.0); 3];
    for c in 0..3 {
        out[c] =
            S::from_f64(G_AMP[c]) * (S::from_f64(G_RATE[c]) * t).sin() + S::from_f64(G_BIAS[c]);
    }
    out
}

/// `dᵏg/dtᵏ`, component by component: `A sin(Rt) + B` differentiates to `A Rᵏ sin(Rt + kπ/2)`, and
/// only `k = 0` keeps the bias.
fn g_deriv(k: i32, t: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = G_AMP[c]
            * G_RATE[c].powi(k)
            * (G_RATE[c] * t + f64::from(k) * std::f64::consts::FRAC_PI_2).sin();
        if k == 0 {
            out[c] += G_BIAS[c];
        }
    }
    out
}

/// `k!` as an `f64`.
fn factorial(k: i32) -> f64 {
    (1..=k).map(f64::from).product()
}

/// `∫ₐᵇ g` in closed form, component by component.
fn g_integral(a: f64, b: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = G_AMP[c] / G_RATE[c] * ((G_RATE[c] * a).cos() - (G_RATE[c] * b).cos())
            + G_BIAS[c] * (b - a);
    }
    out
}

impl ScalarSystem<3> for Quadrature {
    fn eval<S: Scalar>(&self, t: S, _y: &[S; 3]) -> [S; 3] {
        g(t)
    }
}

#[test]
fn the_corrector_evaluates_the_force_at_the_end_of_the_step() {
    // On a right-hand side that ignores `y`, the startup jets are the exact Taylor coefficients of
    // the quadrature — `zᵢ = hⁱ g⁽ⁱ⁻¹⁾(t₀)/i!` for `i ≥ 1` — so one predict–evaluate–correct step
    // collapses to a closed form written from `g` and its derivatives alone. The Pascal predictor
    // sums those series: `z_pred[0]` is the degree-5 Taylor polynomial of `y` at `t₀+h`, and
    // `z_pred[1]` is `h·T₄g(t₀+h)` with `T₄g` the degree-4 Taylor polynomial of `g` about `t₀`.
    //
    //     yₚ = y₀ + Σ_{i=1..5} hⁱ g⁽ⁱ⁻¹⁾(t₀)/i!,   e = h·(g(t₀+h) − T₄g(t₀+h)),   y ← yₚ + ℓ₀·e,
    //
    // and the returned RMS is `√(Σ (eᶜ/(1+|yₚᶜ|))² / N)`. So `e` is that polynomial's remainder,
    // which pins the instant `g` is sampled at eight orders more sharply than the value does:
    // sampling the corrector at `t₀` instead of `t₀+h` moves `e` from 2e-11 to 1e-3.
    let t0 = 137.25;
    let h = 0.031_25;
    let y0 = [0.5, 3.0, 40.0];

    let mut ns = Nordsieck::<3>::start(&Quadrature, t0, &y0, h);
    let rms = ns.step(&Quadrature, t0);
    let got = ns.value();

    let mut yp = y0;
    let mut t4g = [0.0; 3];
    for i in 1..=5_i32 {
        let d = g_deriv(i - 1, t0);
        let (into_y, into_g) = (h.powi(i) / factorial(i), h.powi(i - 1) / factorial(i - 1));
        for c in 0..3 {
            yp[c] += into_y * d[c];
            t4g[c] += into_g * d[c];
        }
    }

    let g1 = g(t0 + h);
    let mut acc = 0.0;
    for c in 0..3 {
        let e = h * (g1[c] - t4g[c]);
        let want = yp[c] + L0 * e;
        assert!(
            (got[c] - want).abs() <= 1e-15 * want.abs(),
            "component {c}: corrected state {:.17e}, closed form {want:.17e}",
            got[c]
        );
        acc += (e / (1.0 + yp[c].abs())).powi(2);
    }
    let want_rms = (acc / 3.0).sqrt();
    // `e` is a difference of two numbers that agree to eleven figures — on both sides of this
    // comparison — so the RMS agrees to the ~1e-7 that leaves, not to the last bit. Eight orders of
    // margin still separate it from a corrector that samples the wrong instant.
    println!("Nordsieck step RMS {rms:.17e} vs closed form {want_rms:.17e}");
    assert!(
        (rms - want_rms).abs() <= 1e-6 * want_rms,
        "step returned RMS {rms:.17e}, closed form is {want_rms:.17e}"
    );
}

#[test]
fn the_integrator_advances_the_clock_from_a_non_zero_epoch() {
    // Every fixture above starts at `t0 = 0`, where `t1 − t0` and `t1 + t0` are the same number, and
    // drives an autonomous force, where the clock could run backwards unnoticed. On a quadrature the
    // answer is an integral with a closed form, so both are off by whole orders of magnitude.
    //
    // The tolerance is measured, not aspirational. Here the point is only that the epoch and the
    // direction of the clock are pinned; the order the startup vector reaches is
    // [`the_startup_vector_is_order_five_on_a_non_autonomous_force`].
    let t0 = 137.25;
    let span = 4.0;
    let y0 = [0.5, 3.0, 40.0];
    let exact = g_integral(t0, t0 + span);
    let got = integrate_adams_nordsieck::<3>(&Quadrature, t0, &y0, t0 + span, span / 256.0)
        .expect("quadrature");
    for c in 0..3 {
        let err = (got[c] - y0[c] - exact[c]).abs();
        println!("quadrature component {c}: err {err:.3e}");
        assert!(
            err < 5e-11,
            "component {c} off the exact integral: {err:.3e}"
        );
    }
}

#[test]
fn the_startup_vector_is_order_five_on_a_non_autonomous_force() {
    // The method is order 5 only if its *startup* vector is. `Nordsieck::start` seeds the scaled
    // derivatives from `taylor::solution_jets`, whose recurrence has to carry `∂f/∂t` — hold the
    // clock fixed inside it and every coefficient above the first comes out zero, which is a
    // one-off O(h²) defect no amount of stepping recovers from. Kepler cannot see it (autonomous);
    // a quadrature `y' = g(t)` is nothing but `∂f/∂t`.
    let t0 = 137.25;
    let span = 4.0;
    let y0 = [0.5, 3.0, 40.0];
    let exact = g_integral(t0, t0 + span);
    let ladder = [64, 128, 256, 512].map(|n| {
        let got =
            integrate_adams_nordsieck::<3>(&Quadrature, t0, &y0, t0 + span, span / f64::from(n))
                .expect("quadrature");
        (got[0] - y0[0] - exact[0]).abs()
    });
    let order = (ladder[0] / ladder[1]).log2();
    println!(
        "quadrature component 0: {:.4e} {:.4e} {:.4e} {:.4e}, first order {order:.2}",
        ladder[0], ladder[1], ladder[2], ladder[3]
    );
    assert!(
        (4.7..5.3).contains(&order),
        "startup vector is order {order:.2}, not 5"
    );
    assert!(
        ladder[3] < 1e-12,
        "n = 512 error {:.3e} is above the measured 4.53e-13",
        ladder[3]
    );
}
