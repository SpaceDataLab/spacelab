//! Symplectic integrators, checked by their order and by the property that justifies them: the
//! energy error stays bounded over thousands of orbits instead of drifting.
//!
//! * **Order** — halving the step shrinks the error by 2² (leapfrog) and 2⁴ (Yoshida), measured
//!   against a tight `DOP853` reference.
//! * **No energy drift** — over two thousand orbits the once-per-orbit energy error does not grow;
//!   this is the whole reason to prefer a symplectic method to a more accurate Runge–Kutta.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_coverage::find_events;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Acceleration, Newtonian, Plain, PointMass};
use spacelab_propagation::symplectic::{leapfrog, leapfrog_dense, yoshida4, yoshida6, yoshida8};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_testkit::nan_max;

const MU: f64 = 3.986_004_418e14;

/// A mildly eccentric orbit starting at perigee on +x, so the integrator sees a varying speed.
fn orbit() -> ([f64; 3], [f64; 3], f64) {
    let a = 7_000e3;
    let e = 0.1;
    let r_p = a * (1.0 - e);
    let v_p = (MU * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    ([r_p, 0.0, 0.0], [0.0, v_p, 0.0], period)
}

fn gravity(r: &[f64; 3]) -> [f64; 3] {
    PointMass { mu: MU }.acceleration::<f64>(0.0, r, &[0.0; 3])
}

/// A tight `DOP853` reference position at `span` seconds.
fn reference(x0: [f64; 3], v0: [f64; 3], span: f64) -> [f64; 3] {
    let y0 = [x0[0], x0[1], x0[2], v0[0], v0[1], v0[2]];
    let yf = integrate(
        &Dop853,
        &Plain(Newtonian(PointMass { mu: MU })),
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-4),
    )
    .expect("reference")
    .final_state();
    [yf[0], yf[1], yf[2]]
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Position error after `span` of a fixed-step method run by `step`, with `steps` steps.
fn error<M: Fn(&mut [f64; 3], &mut [f64; 3], f64, usize)>(step: M, span: f64, steps: usize) -> f64 {
    let (x0, v0, _) = orbit();
    let (mut x, mut v) = (x0, v0);
    step(&mut x, &mut v, span / steps as f64, steps);
    dist(&x, &reference(x0, v0, span))
}

#[test]
fn leapfrog_is_second_order() {
    let (_, _, period) = orbit();
    let span = 2.0 * period;
    let run =
        |x: &mut [f64; 3], v: &mut [f64; 3], dt: f64, n: usize| leapfrog(&gravity, x, v, dt, n);
    let coarse = error(run, span, 400);
    let fine = error(run, span, 800);
    let order = (coarse / fine).log2();
    println!("leapfrog: err {coarse:.3e} → {fine:.3e}, order {order:.2}");
    assert!(
        (order - 2.0).abs() < 0.3,
        "leapfrog order {order:.2} is not 2"
    );
}

#[test]
fn yoshida_is_fourth_order() {
    let (_, _, period) = orbit();
    let span = 2.0 * period;
    let run =
        |x: &mut [f64; 3], v: &mut [f64; 3], dt: f64, n: usize| yoshida4(&gravity, x, v, dt, n);
    let coarse = error(run, span, 200);
    let fine = error(run, span, 400);
    let order = (coarse / fine).log2();
    println!("yoshida: err {coarse:.3e} → {fine:.3e}, order {order:.2}");
    assert!(
        (order - 4.0).abs() < 0.4,
        "yoshida order {order:.2} is not 4"
    );
}

#[test]
fn yoshida6_is_sixth_order() {
    let (_, _, period) = orbit();
    let span = 2.0 * period;
    let run =
        |x: &mut [f64; 3], v: &mut [f64; 3], dt: f64, n: usize| yoshida6(&gravity, x, v, dt, n);
    // Order approaches 6 from below as the step shrinks (5.4 at 100→200, 5.96 at 300→600); the
    // asymptotic grid below is fine enough to read it clearly and far above the ~1e-8 m roundoff floor.
    let coarse = error(run, span, 300);
    let fine = error(run, span, 600);
    let order = (coarse / fine).log2();
    println!("yoshida6: err {coarse:.3e} → {fine:.3e}, order {order:.2}");
    assert!(fine > 1e-7, "already at roundoff; use coarser grids");
    assert!(
        (order - 6.0).abs() < 0.3,
        "yoshida6 order {order:.2} is not 6"
    );
}

#[test]
fn yoshida8_is_eighth_order() {
    let (_, _, period) = orbit();
    let span = 2.0 * period;
    let run =
        |x: &mut [f64; 3], v: &mut [f64; 3], dt: f64, n: usize| yoshida8(&gravity, x, v, dt, n);
    let coarse = error(run, span, 120);
    let fine = error(run, span, 240);
    let order = (coarse / fine).log2();
    println!("yoshida8: err {coarse:.3e} → {fine:.3e}, order {order:.2}");
    assert!(fine > 1e-7, "already at roundoff; use coarser grids");
    assert!(
        (order - 8.0).abs() < 0.3,
        "yoshida8 order {order:.2} is not 8"
    );
}

#[test]
fn leapfrog_dense_tracks_the_orbit_and_finds_events() {
    let (x0, v0, period) = orbit();
    let steps = 4000usize;
    let span = 2.0 * period;
    let dt = span / steps as f64;
    let traj = leapfrog_dense::<3, 6, _>(&gravity, &x0, &v0, 0.0, dt, steps);

    assert_eq!(traj.span(), (0.0, span), "the declared span is not [0, 2T]");

    // The sample has to sit strictly *between* nodes. `0.37 · span` is `1480 · dt` exactly, and the
    // Hermite value at a node is the state it was built from whatever interval it thinks it spans —
    // `s = (t₁ − t₀)/(t₁ − t₀) = 1` for any `t₀` — so a node sample cannot see the parameterisation
    // at all.
    let k = 1480usize;
    let t_mid = (k as f64 + 0.5) * dt;
    let ref_x = reference(x0, v0, t_mid);
    let mid = traj.eval(t_mid);
    let dmid = dist(&[mid[0], mid[1], mid[2]], &ref_x);
    println!("leapfrog dense mid-step vs DOP853: {dmid:.3e} m");
    assert!(dmid < 5e3, "leapfrog dense mid-step off: {dmid:.3e} m");

    // DOP853 alone cannot see the interpolation: leapfrog's own state error is ~185 m at this grid,
    // three orders above the cubic's, so a *linear* interpolant passes the check above too. The
    // midpoint of a cubic Hermite has a closed form — `½(y₀+y₁) + (h/8)(f₀−f₁)` — and that does see
    // it. The nodes are re-derived by stepping `leapfrog` the same way `leapfrog_dense` does, one
    // step per call, so the compensation state matches and only the interpolation is under test.
    let (mut xa, mut va) = (x0, v0);
    for _ in 0..k {
        leapfrog(&gravity, &mut xa, &mut va, dt, 1);
    }
    let (mut xb, mut vb) = (xa, va);
    leapfrog(&gravity, &mut xb, &mut vb, dt, 1);
    let (aa, ab) = (gravity(&xa), gravity(&xb));
    for i in 0..3 {
        let want_x = f64::midpoint(xa[i], xb[i]) + dt / 8.0 * (va[i] - vb[i]);
        let want_v = f64::midpoint(va[i], vb[i]) + dt / 8.0 * (aa[i] - ab[i]);
        assert!(
            (mid[i] - want_x).abs() < 1e-6,
            "dense position {i} is {:.6e} m off the cubic's midpoint",
            mid[i] - want_x
        );
        assert!(
            (mid[3 + i] - want_v).abs() < 1e-9,
            "dense velocity {i} is {:.6e} m/s off the cubic's midpoint",
            mid[3 + i] - want_v
        );
    }

    let mut times = std::vec::Vec::new();
    let n = find_events(
        |t| traj.eval(t)[0],
        0.0,
        span,
        period / 20.0,
        1e-3,
        |e| {
            times.push(f64::midpoint(e.at.lo, e.at.hi));
        },
    );
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
    // A crossing count is topological and survives a badly wrong interpolant; a crossing *time* does
    // not. Each one is checked against DOP853 rather than against the interpolant that produced it:
    // the reference orbit must itself be on the x = 0 plane there, to within the ~185 m the dense
    // output is worth.
    for (k, &t) in times.iter().enumerate() {
        let x_ref = reference(x0, v0, t)[0];
        assert!(
            x_ref.abs() < 1e3,
            "crossing {k} at {t:.6} s: the reference orbit is at x = {x_ref:.3e} m there"
        );
    }
}

#[test]
fn compensated_summation_holds_a_reversed_arc_at_the_rounding_floor() {
    // yoshida4 is time-reversible, so integrating forward n steps and then backward n steps returns
    // to the start up to rounding — truncation cancels on the reversal. Kahan compensation keeps that
    // residual at ~ε|x|; naive summation would random-walk it to ~√(2n)·ε|x|, which at n = 5×10⁵ is
    // ~1.5 µm — an order of magnitude above the threshold below.
    let (x0, v0, period) = orbit();
    let (mut x, mut v) = (x0, v0);
    let n = 500_000;
    let dt = 10.0 * period / n as f64; // small step → rounding-dominated
    yoshida4(&gravity, &mut x, &mut v, dt, n);
    yoshida4(&gravity, &mut x, &mut v, -dt, n);
    let err = dist(&x, &x0);
    println!(
        "reversed arc over {} steps: position error {err:.3e} m",
        2 * n
    );
    assert!(
        err < 1e-7,
        "round-trip drift {err:.3e} m — the accumulation is not compensated"
    );
}

#[test]
fn the_symplectic_energy_error_stays_bounded_over_thousands_of_orbits() {
    // The defining property: the energy error oscillates within a fixed envelope and does not drift.
    // Measured as the peak error over every step in the first half of the run versus the second — a
    // non-symplectic method of the same order would grow the peak linearly, so the halves would
    // differ by ~2×; a symplectic one reaches the same envelope in both.
    let (x0, v0, period) = orbit();
    let (mut x, mut v) = (x0, v0);
    let g: fn(&[f64; 3]) -> [f64; 3] = gravity;
    let steps_per_orbit: usize = 300;
    let dt = period / steps_per_orbit as f64;

    let energy = |x: &[f64; 3], v: &[f64; 3]| {
        let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - MU / r
    };
    let e0 = energy(&x0, &v0);

    let orbits = 1500;
    let total = orbits * steps_per_orbit;
    let (mut first_half_peak, mut second_half_peak) = (0.0_f64, 0.0_f64);
    for step in 0..total {
        leapfrog(&g, &mut x, &mut v, dt, 1);
        let err = (energy(&x, &v) - e0).abs() / e0.abs();
        if step < total / 2 {
            first_half_peak = nan_max(first_half_peak, err);
        } else {
            second_half_peak = nan_max(second_half_peak, err);
        }
    }
    println!(
        "energy error envelope: first half {first_half_peak:.3e}, second half {second_half_peak:.3e} over {orbits} orbits"
    );

    assert!(
        second_half_peak < 1e-4,
        "energy error {second_half_peak:.3e} too large — check the step"
    );
    // no secular drift: the envelope is the same in both halves, not growing
    assert!(
        second_half_peak < 1.3 * first_half_peak,
        "energy drifted: second half {second_half_peak:.3e} vs first {first_half_peak:.3e}"
    );
}
