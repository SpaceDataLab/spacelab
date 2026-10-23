//! KS regularization: a very eccentric orbit integrated with uniform steps in fictitious time closes
//! to high accuracy and far outperforms a uniform-time-step Cartesian method that struggles at
//! perihelion.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use spacelab_propagation::forces::{Acceleration, J2, PointMass};
use spacelab_propagation::ks::integrate_ks;
use spacelab_propagation::symplectic::leapfrog;

const MU: f64 = 3.986_004_418e14;
const A: f64 = 10_000e3;
const ECC: f64 = 0.9; // very eccentric — perihelion speed ~4x perihelion-to-aphelion mean

fn perigee_state() -> ([f64; 3], [f64; 3], f64) {
    let r_p = A * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (A * (1.0 - ECC))).sqrt();
    let period = std::f64::consts::TAU * (A * A * A / MU).sqrt();
    ([r_p, 0.0, 0.0], [0.0, v_p, 0.0], period)
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn ks_closes_a_high_eccentricity_orbit() {
    // One revolution is s = π/√(−h/2); the orbit must return to its start and the elapsed real time
    // must equal the Keplerian period.
    let (r0, v0, period) = perigee_state();
    let h = 0.5 * (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]) - MU / (r0[0]);
    let omega = (-h / 2.0).sqrt();
    let s_span = std::f64::consts::PI / omega; // one revolution: u and -u give the same r
    let no_pert = J2 {
        mu: MU,
        re: 6_378_137.0,
        j2: 0.0,
    };

    let (rf, vf, t) = integrate_ks(MU, &no_pert, &r0, &v0, s_span, 4000).expect("ks");
    let dr = dist(&rf, &r0);
    let dv = dist(&vf, &v0);
    println!(
        "KS e=0.9 one rev: Δr = {dr:.3e} m, Δv = {dv:.3e} m/s, t = {t:.3} vs period {period:.3}"
    );
    assert!(dr < 1.0, "orbit did not close in position: {dr:.3e} m");
    assert!(dv < 1e-3, "orbit did not close in velocity: {dv:.3e} m/s");
    assert!(
        (t - period).abs() / period < 1e-6,
        "elapsed time is not one period: {t} vs {period}"
    );
}

#[test]
fn ks_beats_a_uniform_time_step_cartesian_method() {
    // Same orbit, same number of force evaluations. KS steps uniformly in eccentric anomaly (dense in
    // real time near perihelion, exactly where it is needed); a uniform-time leapfrog under-resolves
    // the fast perihelion passage and closes far worse.
    let (r0, v0, period) = perigee_state();
    let h = 0.5 * (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]) - MU / (r0[0]);
    let omega = (-h / 2.0).sqrt();
    let s_span = std::f64::consts::PI / omega; // one revolution: u and -u give the same r
    let no_pert = J2 {
        mu: MU,
        re: 6_378_137.0,
        j2: 0.0,
    };
    let steps = 3000;

    let (rf_ks, _, _) = integrate_ks(MU, &no_pert, &r0, &v0, s_span, steps).expect("ks");
    let ks_close = dist(&rf_ks, &r0);

    let gravity = |r: &[f64; 3]| PointMass { mu: MU }.acceleration::<f64>(0.0, r, &[0.0; 3]);
    let (mut x, mut v) = (r0, v0);
    leapfrog(&gravity, &mut x, &mut v, period / steps as f64, steps);
    let lf_close = dist(&x, &r0);

    println!("one rev, {steps} steps: KS closes to {ks_close:.3e} m, leapfrog to {lf_close:.3e} m");
    assert!(
        ks_close < lf_close,
        "KS ({ks_close:.3e}) not better than leapfrog ({lf_close:.3e})"
    );
    assert!(
        ks_close < lf_close / 100.0,
        "KS not dramatically better on the eccentric orbit"
    );
}

#[test]
fn ks_rejects_bad_arguments() {
    let (r0, v0, _) = perigee_state();
    let no_pert = J2 {
        mu: MU,
        re: 6_378_137.0,
        j2: 0.0,
    };
    assert!(integrate_ks(MU, &no_pert, &r0, &v0, -1.0, 100).is_err());
    assert!(integrate_ks(MU, &no_pert, &r0, &v0, 10.0, 0).is_err());
    // Unbound (hyperbolic) state: energy ≥ 0 is rejected.
    let fast = [0.0, 2.0 * v0[1], 0.0];
    assert!(integrate_ks(MU, &no_pert, &r0, &fast, 10.0, 100).is_err());
    // The guard is `!(s_span > 0.0 && mu > 0.0)`, and each `>` has a `>=` mutant that only an
    // argument sitting exactly on the boundary can see -- `-1.0` above passes either way. A zero arc
    // is the plain case: mutated, the guard falls through and the loop hands the initial state back
    // as a success.
    assert!(integrate_ks(MU, &no_pert, &r0, &v0, 0.0, 100).is_err());
    // A zero `mu` is subtler, because `h >= 0.0` below catches it for every ordinary state: with no
    // central body `h = v²/2 - mu/r` cannot be negative. It stops catching it at the origin, where
    // `mu / r` is `0 / 0` -- `h` is NaN, NaN fails every comparison, and the mutated driver
    // integrates NaN all the way to a successful NaN.
    assert!(integrate_ks(0.0, &no_pert, &[0.0; 3], &v0, 10.0, 100).is_err());
}

use spacelab_propagation::driver::DEFAULT_MAX_STEPS;
use spacelab_propagation::ias15::integrate_ias15;

/// The same eccentric orbit, but with a real inclination, node and argument of perigee, so no
/// component of the position, the velocity or the spinor is zero. Every existing fixture in this file
/// starts at `[r_p, 0, 0]` with velocity along `+y`, which sends `from_cartesian` down one branch and
/// leaves it holding the spinor `[√r_p, 0, 0, 0]` -- three quarters of the KS matrix multiplied by a
/// zero the fixture supplied, and the reason 60 of this module's mutants sat in `lu` and `ltu`.
fn inclined_perigee_state() -> ([f64; 3], [f64; 3], f64) {
    oriented_perigee_state(1.1)
}

/// The same, at an argument of perigee the caller chooses, because `from_cartesian` branches on the
/// sign of `x`: at `x < 0` it builds the spinor around `u₁`, at `x ≥ 0` around `u₀`, and the two
/// branches share no line. `inclined_perigee_state` lands at `x = −0.0100·r_p`, so it only ever
/// exercised the first; `perigee_state` takes the second but hands it `y = z = 0`, which multiplies
/// away the only three expressions that branch contains. Eight mutants sat in those three
/// expressions between the two fixtures.
fn oriented_perigee_state(argp: f64) -> ([f64; 3], [f64; 3], f64) {
    let r_p = A * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (A * (1.0 - ECC))).sqrt();
    let (inc, raan) = (0.9_f64, 0.7_f64);
    let rotate = |p: [f64; 3]| {
        let (c, s) = (argp.cos(), argp.sin());
        let w = [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];
        let (c, s) = (inc.cos(), inc.sin());
        let i = [w[0], c * w[1] - s * w[2], s * w[1] + c * w[2]];
        let (c, s) = (raan.cos(), raan.sin());
        [c * i[0] - s * i[1], s * i[0] + c * i[1], i[2]]
    };
    let h = 0.5 * v_p * v_p - MU / r_p;
    (
        rotate([r_p, 0.0, 0.0]),
        rotate([0.0, v_p, 0.0]),
        std::f64::consts::PI / (-h / 2.0).sqrt(),
    )
}

fn oblate() -> J2 {
    J2 {
        mu: MU,
        re: 6_378_137.0,
        j2: 1.082_626_68e-3,
    }
}

fn full_gravity(r: &[f64; 3]) -> [f64; 3] {
    let point = PointMass { mu: MU }.acceleration::<f64>(0.0, r, &[0.0; 3]);
    let zonal = oblate().acceleration::<f64>(0.0, r, &[0.0; 3]);
    [
        point[0] + zonal[0],
        point[1] + zonal[1],
        point[2] + zonal[2],
    ]
}

#[test]
fn ks_matches_a_cartesian_integrator_under_a_real_perturbation() {
    // Every other test here runs with `j2 = 0.0`, so `ltp` is the zero vector and both the forcing
    // term `(r/2)·L(u)ᵀP` and the whole energy equation `h' = 2u'ᵀL(u)ᵀP` are multiplied away. With
    // J₂ switched on they carry the answer, and IAS15 -- a different method, pinned to the analytic
    // two-body orbit at machine precision in its own file -- says what the answer is.
    let (r0, v0, s_span) = inclined_perigee_state();
    assert!(
        r0.iter().all(|c| c.abs() > 1.0) && v0.iter().all(|c| c.abs() > 1.0),
        "fixture is degenerate: r0 = {r0:?}, v0 = {v0:?}"
    );

    let (rf, vf, t) = integrate_ks(MU, &oblate(), &r0, &v0, s_span, 8000).expect("ks");
    let (ri, vi) = integrate_ias15(
        &full_gravity,
        &r0,
        &v0,
        0.0,
        t,
        t / 200.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("ias15");
    let (dr, dv) = (dist(&rf, &ri), dist(&vf, &vi));
    println!("KS vs IAS15 over one J2-perturbed revolution: Δr = {dr:.3e} m, Δv = {dv:.3e} m/s");
    // Measured 2.49e-4 m and 5.7e-7 m/s at 8000 steps -- RK4 truncation in `ds`, not disagreement
    // about the physics, and the next test is the one that pins the order.
    assert!(dr < 1e-3, "KS and IAS15 disagree by {dr:.3e} m");
    assert!(dv < 5e-6, "KS and IAS15 disagree by {dv:.3e} m/s");
}

#[test]
fn ks_matches_a_cartesian_integrator_from_the_positive_x_spinor_branch() {
    // `argp = 0.3` puts perigee at `x = +0.612·r_p` with `y` and `z` both a good fraction of `r_p`,
    // so `from_cartesian` takes its `x ≥ 0` branch with nothing multiplied away. Same oracle as
    // above: the branch has to produce a spinor that propagates to IAS15's answer, not merely a
    // finite one.
    let (r0, v0, s_span) = oriented_perigee_state(0.3);
    assert!(r0[0] > 0.0, "fixture is on the wrong branch: r0 = {r0:?}");
    assert!(
        r0.iter().all(|c| c.abs() > 1.0) && v0.iter().all(|c| c.abs() > 1.0),
        "fixture is degenerate: r0 = {r0:?}, v0 = {v0:?}"
    );

    let (rf, vf, t) = integrate_ks(MU, &oblate(), &r0, &v0, s_span, 8000).expect("ks");
    let (ri, vi) = integrate_ias15(
        &full_gravity,
        &r0,
        &v0,
        0.0,
        t,
        t / 200.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("ias15");
    let (dr, dv) = (dist(&rf, &ri), dist(&vf, &vi));
    println!("KS vs IAS15 from the x ≥ 0 branch: Δr = {dr:.3e} m, Δv = {dv:.3e} m/s");
    assert!(dr < 1e-3, "KS and IAS15 disagree by {dr:.3e} m");
    assert!(dv < 5e-6, "KS and IAS15 disagree by {dv:.3e} m/s");
}

#[test]
fn ks_converges_at_the_fourth_order_of_its_stepper() {
    // The step is RK4 in fictitious time, so halving `ds` must cut the error by about sixteen. An
    // order that collapses is how a wrong right-hand side shows up when the answer it converges to is
    // still roughly right -- the accuracy assertions above would pass, this will not.
    let (r0, v0, s_span) = inclined_perigee_state();
    let mut errors = Vec::new();
    for n in [2000usize, 4000, 8000] {
        let (rf, _, t) = integrate_ks(MU, &oblate(), &r0, &v0, s_span, n).expect("ks");
        let (ri, _) = integrate_ias15(
            &full_gravity,
            &r0,
            &v0,
            0.0,
            t,
            t / 200.0,
            DEFAULT_MAX_STEPS,
        )
        .expect("ias15");
        errors.push(dist(&rf, &ri));
    }
    println!("KS error at 2000/4000/8000 steps: {errors:?}");
    for pair in errors.windows(2) {
        let ratio = pair[0] / pair[1];
        assert!(
            (10.0..=25.0).contains(&ratio),
            "convergence ratio {ratio:.2} is not fourth order: {errors:?}"
        );
    }
}
