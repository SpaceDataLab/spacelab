//! IAS15: checked against the two properties that define it — machine-precision accuracy on a
//! Kepler orbit and a non-drifting energy error over thousands of orbits.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    // The step ceiling is an integer carried in an `f64`; a tolerance would accept an off-by-one.
    clippy::float_cmp
)]

use spacelab_coverage::find_events;
use spacelab_propagation::driver::{DEFAULT_MAX_STEPS, IntegrationError};
use spacelab_propagation::ias15::{integrate_ias15, integrate_ias15_dense};
use spacelab_testkit::nan_max;

const MU: f64 = 3.986_004_418e14;
const SEMI_MAJOR: f64 = 8_000e3;
const ECC: f64 = 0.1;

fn state0() -> ([f64; 3], [f64; 3], f64) {
    let r_p = SEMI_MAJOR * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (SEMI_MAJOR * (1.0 - ECC))).sqrt();
    let period = std::f64::consts::TAU * (SEMI_MAJOR.powi(3) / MU).sqrt();
    ([r_p, 0.0, 0.0], [0.0, v_p, 0.0], period)
}

fn gravity(r: &[f64; 3]) -> [f64; 3] {
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let inv_r3 = (r2 * r2.sqrt()).recip();
    [
        -MU * r[0] * inv_r3,
        -MU * r[1] * inv_r3,
        -MU * r[2] * inv_r3,
    ]
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

#[test]
fn ias15_matches_the_exact_orbit_to_near_machine_precision() {
    let (r0, v0, period) = state0();
    // Ten orbits, adaptively stepped. The reference is the analytic Kepler position; IAS15 should
    // agree to a level set by the ~1e-9 tolerance and rounding, far below any other fixed-step method.
    let span = 10.0 * period;
    let (r, _) = integrate_ias15(
        &gravity,
        &r0,
        &v0,
        0.0,
        span,
        period / 50.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("ias15");
    let exact = kepler(span);
    let err =
        ((r[0] - exact[0]).powi(2) + (r[1] - exact[1]).powi(2) + (r[2] - exact[2]).powi(2)).sqrt();
    println!("IAS15 vs exact Kepler over 10 orbits: {err:.3e} m");
    assert!(err < 1e-3, "IAS15 not accurate: {err:.3e} m");
}

#[test]
fn ias15_energy_error_does_not_drift_over_thousands_of_orbits() {
    // The defining property: over a long integration the energy error stays bounded (Brouwer's law),
    // it does not grow linearly the way a non-symplectic fixed-step method's would. Measured as the
    // peak relative energy error over the first half of the run versus the second.
    let (r0, v0, period) = state0();
    let energy = |r: &[f64; 3], v: &[f64; 3]| {
        let rm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - MU / rm
    };
    let e0 = energy(&r0, &v0);

    let orbits = 2000;
    let (mut r, mut v) = (r0, v0);
    let mut first_half_peak = 0.0_f64;
    let mut second_half_peak = 0.0_f64;
    for orbit in 0..orbits {
        let out = integrate_ias15(
            &gravity,
            &r,
            &v,
            0.0,
            period,
            period / 50.0,
            DEFAULT_MAX_STEPS,
        )
        .expect("ias15");
        r = out.0;
        v = out.1;
        let rel = (energy(&r, &v) - e0).abs() / e0.abs();
        if orbit < orbits / 2 {
            first_half_peak = nan_max(first_half_peak, rel);
        } else {
            second_half_peak = nan_max(second_half_peak, rel);
        }
    }
    println!(
        "IAS15 energy error over {orbits} orbits: first half {first_half_peak:.3e}, second half {second_half_peak:.3e}"
    );
    assert!(
        second_half_peak < 1e-12,
        "energy error too large: {second_half_peak:.3e}"
    );
    // No secular drift: the second half is not dramatically worse than the first.
    assert!(
        second_half_peak < 10.0 * first_half_peak.max(1e-15),
        "energy drifted: {second_half_peak:.3e} vs {first_half_peak:.3e}"
    );
}

#[test]
fn ias15_dense_matches_the_orbit_and_finds_events() {
    // The b-polynomial dense output must agree with the plain integration's final state, interpolate
    // the orbit faithfully at interior points, and compose with the coverage event detector.
    let (r0, v0, period) = state0();
    let span = 2.0 * period;
    let (rf, vf) = integrate_ias15(
        &gravity,
        &r0,
        &v0,
        0.0,
        span,
        period / 50.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("ias15");
    let traj = integrate_ias15_dense::<3, 6, _>(
        &gravity,
        &r0,
        &v0,
        0.0,
        span,
        period / 50.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("dense");

    // Same final state as the plain driver.
    let f = traj.final_state();
    let dfinal = ((f[0] - rf[0]).powi(2) + (f[1] - rf[1]).powi(2) + (f[2] - rf[2]).powi(2)).sqrt();
    assert!(
        dfinal < 1e-6,
        "dense final state differs from plain: {dfinal:.3e} m"
    );
    assert!(
        (f[3] - vf[0]).abs() < 1e-9 && (f[5] - vf[2]).abs() < 1e-9,
        "dense velocity differs"
    );

    // Interior interpolation agrees with the exact orbit at a mid-span sample.
    let mid = traj.eval(0.37 * span);
    let exact = kepler(0.37 * span);
    let dmid =
        ((mid[0] - exact[0]).powi(2) + (mid[1] - exact[1]).powi(2) + (mid[2] - exact[2]).powi(2))
            .sqrt();
    println!("IAS15 dense interior error at 0.37·span: {dmid:.3e} m");
    assert!(
        dmid < 1e-3,
        "dense interior interpolation off: {dmid:.3e} m"
    );

    // Node crossings via the event detector: x crosses zero twice per orbit → four over two orbits.
    let n = find_events(|t| traj.eval(t)[0], 0.0, span, period / 20.0, 1e-4, |_| {});
    println!("IAS15 dense → find_events: {n} crossings");
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
}

#[test]
fn ias15_adapts_the_step_to_eccentricity() {
    // A very eccentric orbit forces the controller to shrink the step near perigee; the method must
    // still return to the start after one period (closed orbit) to high precision.
    let a = 10_000e3;
    let e = 0.7;
    let r_p = a * (1.0 - e);
    let v_p = (MU * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    let r0 = [r_p, 0.0, 0.0];
    let v0 = [0.0, v_p, 0.0];
    let (r, v) = integrate_ias15(
        &gravity,
        &r0,
        &v0,
        0.0,
        period,
        period / 100.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("ias15");
    let dr = ((r[0] - r0[0]).powi(2) + (r[1] - r0[1]).powi(2) + (r[2] - r0[2]).powi(2)).sqrt();
    let dv = ((v[0] - v0[0]).powi(2) + (v[1] - v0[1]).powi(2) + (v[2] - v0[2]).powi(2)).sqrt();
    println!("IAS15 e=0.7 closed-orbit return: Δr = {dr:.3e} m, Δv = {dv:.3e} m/s");
    assert!(
        dr < 1.0,
        "eccentric orbit did not close in position: {dr:.3e} m"
    );
    assert!(
        dv < 1e-3,
        "eccentric orbit did not close in velocity: {dv:.3e} m/s"
    );
}

/// The exact two-body state, position and velocity, from the eccentric anomaly.
fn kepler_rv(span: f64) -> ([f64; 3], [f64; 3]) {
    let n = (MU / SEMI_MAJOR.powi(3)).sqrt();
    let m = n * span;
    let mut ea = m;
    for _ in 0..100 {
        ea -= (ea - ECC * ea.sin() - m) / (1.0 - ECC * ea.cos());
    }
    let (se, ce) = (ea.sin(), ea.cos());
    let beta = (1.0 - ECC * ECC).sqrt();
    let speed = (MU * SEMI_MAJOR).sqrt() / (SEMI_MAJOR * (1.0 - ECC * ce));
    (
        [SEMI_MAJOR * (ce - ECC), SEMI_MAJOR * beta * se, 0.0],
        [-speed * se, speed * beta * ce, 0.0],
    )
}

#[test]
fn ias15_dense_reproduces_the_orbit_between_its_nodes() {
    // `ias15_dense_matches_the_orbit_and_finds_events` samples the interpolant at one interior point
    // and its velocity only at the final node, where the polynomial argument is exactly 1. That
    // leaves the degree-9 position and degree-8 velocity expansions free between the nodes: 74 of the
    // module's mutants live in `Ias15Dense::eval` and none of them was caught. Here the whole span is
    // swept and both halves of the state are compared against the analytic two-body solution, which
    // is exact for this force model rather than an integration of it.
    let (r0, v0, period) = state0();
    let span = 2.0 * period;
    let traj = integrate_ias15_dense::<3, 6, _>(
        &gravity,
        &r0,
        &v0,
        0.0,
        span,
        period / 50.0,
        DEFAULT_MAX_STEPS,
    )
    .expect("dense");

    let (mut worst_r, mut worst_v) = (0.0_f64, 0.0_f64);
    for k in 0..=400 {
        let t = span * f64::from(k) / 400.0;
        let got = traj.eval(t);
        let (r, v) = kepler_rv(t);
        worst_r = nan_max(
            worst_r,
            ((got[0] - r[0]).powi(2) + (got[1] - r[1]).powi(2) + (got[2] - r[2]).powi(2)).sqrt(),
        );
        worst_v = nan_max(
            worst_v,
            ((got[3] - v[0]).powi(2) + (got[4] - v[1]).powi(2) + (got[5] - v[2]).powi(2)).sqrt(),
        );
    }
    println!("IAS15 dense over 401 samples: Δr = {worst_r:.3e} m, Δv = {worst_v:.3e} m/s");
    // Measured 1.07e-7 m and 8.25e-11 m/s; the bounds leave an order of magnitude.
    assert!(worst_r < 1e-6, "dense position off: {worst_r:.3e} m");
    assert!(worst_v < 1e-9, "dense velocity off: {worst_v:.3e} m/s");
}

#[test]
fn ias15_costs_what_its_warm_start_earns() {
    // `predict_next_step` rescales the b-polynomial onto the next step so the corrector starts from
    // a fit rather than from zero. Nothing it does can change the answer -- the corrector converges
    // to the same state either way -- so 88 of its mutants survived every accuracy assertion in this
    // file. What it changes is the price: force evaluations, and whether the controller has to reject
    // a step at all. Both are observable, so both are asserted.
    let expected = [
        // (a, e, force evaluations, accepted steps, rejected steps)
        (8_000e3, 0.0, 1662, 73, 0),
        (8_000e3, 0.1, 2718, 121, 0),
        (10_000e3, 0.7, 6132, 321, 1),
        (10_000e3, 0.9, 8537, 472, 1),
    ];
    let mut previous_calls = 0;
    for (a, e, calls_ref, steps_ref, rejected_ref) in expected {
        let r_p = a * (1.0 - e);
        let v_p = (MU * (1.0 + e) / (a * (1.0 - e))).sqrt();
        let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
        let calls = std::cell::Cell::new(0u32);
        let counted = |r: &[f64; 3]| {
            calls.set(calls.get() + 1);
            gravity(r)
        };
        let span = 2.0 * period;
        let traj = integrate_ias15_dense::<3, 6, _>(
            &counted,
            &[r_p, 0.0, 0.0],
            &[0.0, v_p, 0.0],
            0.0,
            span,
            period / 50.0,
            DEFAULT_MAX_STEPS,
        )
        .expect("dense");
        let (accepted, rejected) = traj.step_counts();
        let calls = calls.get();
        println!("IAS15 e={e}: {calls} evaluations, {accepted} steps, {rejected} rejected");

        // Exact equality, not a band. A tenth either way was the first attempt and it left 52
        // mutants of the two controllers alive: the convergence test and its oscillation break, the
        // `1/7` step exponent, the rejection arm's ratio and its `SAFETY` comparison, the growth cap
        // and the warm-start rescale all move the trajectory through this loop by less than a tenth
        // of its total work. The counts are a deterministic function of the arithmetic -- reproduced
        // here to the evaluation -- so a band buys nothing but blindness. If a toolchain bump moves
        // `powf` by an ulp and shifts one of these by a step or two, re-measure and say so in the
        // commit; do not widen the band back.
        assert_eq!(calls, calls_ref, "e={e}: force evaluations");
        assert_eq!(accepted, steps_ref, "e={e}: accepted steps");
        // Exact, not a band: rejections are single digits, and a warm start that starts from the
        // wrong fit shows up here before it shows up in the evaluation count.
        assert_eq!(rejected, rejected_ref, "e={e}: {rejected} rejected steps");
        // Physics rather than a fitted number: a more eccentric orbit needs more work.
        assert!(
            calls > previous_calls,
            "e={e}: {calls} not above {previous_calls}"
        );
        previous_calls = calls;
    }
}

#[test]
fn ias15_refuses_a_span_or_step_it_cannot_integrate() {
    // The guard is `!(t1 > t0 && dt0 > 0.0)` and each of its three pieces has a mutant only an
    // argument sitting exactly on the boundary can see. An empty span reaches the first `>`. A
    // reversed span with a good step reaches `&&`, which mutated to `||` lets the driver fall
    // through and hand the initial state back as a success. A zero step reaches the second `>`, and
    // that one the suite detects as a hang rather than a failure -- a zero step advances `t` by
    // nothing, so the mutated driver never leaves its `while`.
    let (r0, v0, _) = state0();
    for (t0, t1, dt0) in [(0.0, 0.0, 60.0), (3600.0, 0.0, 60.0), (0.0, 3600.0, 0.0)] {
        assert!(
            matches!(
                integrate_ias15(&gravity, &r0, &v0, t0, t1, dt0, DEFAULT_MAX_STEPS),
                Err(IntegrationError::BadArguments)
            ),
            "accepted t0={t0} t1={t1} dt0={dt0}"
        );
        assert!(
            matches!(
                integrate_ias15_dense::<3, 6, _>(
                    &gravity,
                    &r0,
                    &v0,
                    t0,
                    t1,
                    dt0,
                    DEFAULT_MAX_STEPS
                ),
                Err(IntegrationError::BadArguments)
            ),
            "dense accepted t0={t0} t1={t1} dt0={dt0}"
        );
    }
}

#[test]
fn ias15_throws_away_a_first_step_far_too_large_for_the_perigee() {
    // Every other fixture here is warm-started well enough to reject nothing, which leaves the
    // rejection arm unexecuted in both drivers: the ratio `dt_new / dt`, its comparison against
    // `SAFETY`, and the rescale it hands `predict_next_step`. Handing an e = 0.95 perigee a whole
    // period as its first step forces the controller to throw that step away before it can take one.
    // The answer still has to be right afterwards -- a rejection that loses the warm start would
    // pass a count-only check.
    let a = 10_000e3;
    let e = 0.95;
    let r_p = a * (1.0 - e);
    let v_p = (MU * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    let span = 2.0 * period;
    let traj = integrate_ias15_dense::<3, 6, _>(
        &gravity,
        &[r_p, 0.0, 0.0],
        &[0.0, v_p, 0.0],
        0.0,
        span,
        period,
        DEFAULT_MAX_STEPS,
    )
    .expect("dense");
    let (accepted, rejected) = traj.step_counts();
    let end = traj.eval(span);
    println!(
        "IAS15 e={e} dt0=period: {accepted} accepted, {rejected} rejected, \
         Δr = {:.3e} m, Δv = {:.3e} m/s",
        ((end[0] - r_p).powi(2) + end[1] * end[1] + end[2] * end[2]).sqrt(),
        (end[3] * end[3] + (end[4] - v_p).powi(2) + end[5] * end[5]).sqrt()
    );
    // Exact, for the reason given in `ias15_costs_what_its_warm_start_earns`: this is the only
    // fixture that executes the rejection arm, so it is the only place the ratio `dt_new / dt`, its
    // `SAFETY` comparison, the `dt < t1 - t` guard beside it and the rescale handed to
    // `predict_next_step` are observable at all. `rejected > 0` sees none of them.
    assert_eq!(
        (accepted, rejected),
        (563, 2),
        "the oversized first step changed the controller's trajectory"
    );
    // Two whole periods return to the perigee they started from.
    assert!(
        ((end[0] - r_p).powi(2) + end[1] * end[1] + end[2] * end[2]).sqrt() < 1.0,
        "position after two periods: {:?}",
        &end[..3]
    );
    assert!(
        (end[3] * end[3] + (end[4] - v_p).powi(2) + end[5] * end[5]).sqrt() < 1e-3,
        "velocity after two periods: {:?}",
        &end[3..]
    );
}

#[test]
fn ias15_drifts_in_a_straight_line_when_no_force_acts() {
    // Both controllers divide by a maximum acceleration -- `db6_max / a7_max` for the corrector's
    // convergence test, `b6_max / a0_max` for the step-size estimate -- and both guard the divide
    // with `> 0.0`. Under gravity that maximum is never zero, so the guard is always true and its
    // false arm has never executed: `>` widened to `>=` divides by zero there instead, `/` swapped
    // for `*` or `%` changes the ratio, and the `dt / SAFETY` fallback the false arm reaches is dead
    // code. A body with no force on it makes every one of those the taken path.
    //
    // The oracle is closed-form rather than a tolerance in disguise: with zero acceleration the
    // Gauss-Radau polynomial fit is exact and the answer is the straight line `r0 + v0·t`.
    let calls = std::cell::Cell::new(0u32);
    let free = |_r: &[f64; 3]| {
        calls.set(calls.get() + 1);
        [0.0; 3]
    };
    let r0 = [7_000e3, -1_000e3, 500e3];
    let v0 = [1_200.0, 7_400.0, -300.0];
    let span = 86_400.0;

    let (r, v) =
        integrate_ias15(&free, &r0, &v0, 0.0, span, 60.0, DEFAULT_MAX_STEPS).expect("free drift");
    let plain_calls = calls.get();
    let mut worst_r = 0.0_f64;
    let mut worst_v = 0.0_f64;
    for i in 0..3 {
        worst_r = worst_r.max((r[i] - (r0[i] + v0[i] * span)).abs());
        worst_v = worst_v.max((v[i] - v0[i]).abs());
    }
    println!(
        "IAS15 free drift: {plain_calls} evaluations, Δr = {worst_r:.3e} m, Δv = {worst_v:.3e} m/s"
    );

    calls.set(0);
    let traj =
        integrate_ias15_dense::<3, 6, _>(&free, &r0, &v0, 0.0, span, 60.0, DEFAULT_MAX_STEPS)
            .expect("dense");
    let dense_calls = calls.get();
    let (accepted, rejected) = traj.step_counts();
    let mid = traj.eval(0.37 * span);
    let mut worst_mid = 0.0_f64;
    for i in 0..3 {
        worst_mid = worst_mid.max((mid[i] - (r0[i] + v0[i] * 0.37 * span)).abs());
    }
    println!(
        "IAS15 free drift dense: {dense_calls} evaluations, {accepted} accepted, \
         {rejected} rejected, interior Δr = {worst_mid:.3e} m"
    );

    assert!(
        worst_r < 1e-7,
        "drift is not a straight line: {worst_r:.3e} m"
    );
    assert!(
        worst_v < 1e-12,
        "velocity changed under no force: {worst_v:.3e} m/s"
    );
    assert!(
        worst_mid < 1e-7,
        "dense interior is not the line: {worst_mid:.3e} m"
    );
    // Measured: both residuals are exactly 0.0, and both drivers spend 56 evaluations over 7 steps.
    // Seven is the whole story of the false arm -- `dt / SAFETY` quadruples the step every time, and
    // `60 s x 4^7` overshoots a day, so the last step is the one clamped to the endpoint. Exact, for
    // the reason given in `ias15_costs_what_its_warm_start_earns`.
    assert_eq!(rejected, 0, "a force-free drift needs no rejection");
    assert_eq!(plain_calls, 56, "force evaluations, plain driver");
    assert_eq!(dense_calls, 56, "force evaluations, dense driver");
    assert_eq!(accepted, 7, "accepted steps");
}

#[test]
fn ias15_measures_its_span_from_whatever_epoch_it_is_given() {
    // `gravity` is a function of position alone, so the state after a given elapsed span cannot
    // depend on where that span sits on the clock. Every other fixture in this file starts at
    // `t0 = 0`, and that is the one epoch at which the driver's span arithmetic is unobservable:
    // `t1 - t0` and `t1 + t0` are both `t1`, and `t1 / t0` is `+inf`, which `dt0.min(..)` swallows.
    //
    // How far off zero matters as much as the sign. `dt0.min(t1 - t0)` only does work when its
    // second argument is the smaller one, and `t1 + t0 = 2·t0 + period` beats `dt0 = period/50`
    // only once `|t0|` is large against the period -- at `t0 = -1800` the mutated clamp still
    // returns `dt0` and nothing is seen. A megasecond is comfortably past that, in both directions.
    //
    // Two independent checks, because the epoch invariance alone would be a symmetry the code could
    // satisfy while being wrong in the same way at every epoch: each run is also held to the
    // closed-form Kepler position, and the evaluation counts are held exactly.
    let (r0, v0, period) = state0();
    let exact = kepler(period);
    let mut reference: Option<([f64; 3], [f64; 3])> = None;

    for t0 in [0.0, -1.0e6, 1.0e6] {
        let calls = std::cell::Cell::new(0u32);
        let counted = |r: &[f64; 3]| {
            calls.set(calls.get() + 1);
            gravity(r)
        };
        let (r, v) = integrate_ias15(
            &counted,
            &r0,
            &v0,
            t0,
            t0 + period,
            period / 50.0,
            DEFAULT_MAX_STEPS,
        )
        .expect("shifted epoch");
        let err =
            ((r[0] - exact[0]).powi(2) + (r[1] - exact[1]).powi(2) + (r[2] - exact[2]).powi(2))
                .sqrt();
        println!(
            "IAS15 from t0 = {t0:e}: {} evaluations, |Δr| vs Kepler = {err:.3e} m",
            calls.get()
        );
        assert!(err < 1e-3, "epoch {t0:e}: {err:.3e} m from the exact orbit");
        assert_eq!(calls.get(), 1391, "epoch {t0:e}: force evaluations");

        // `integrate_ias15_dense` carries its own copy of the same span arithmetic, so the plain
        // driver's fixture leaves the duplicate untested.
        calls.set(0);
        let traj = integrate_ias15_dense::<3, 6, _>(
            &counted,
            &r0,
            &v0,
            t0,
            t0 + period,
            period / 50.0,
            DEFAULT_MAX_STEPS,
        )
        .expect("shifted epoch, dense");
        let f = traj.final_state();
        let dense_err =
            ((f[0] - exact[0]).powi(2) + (f[1] - exact[1]).powi(2) + (f[2] - exact[2]).powi(2))
                .sqrt();
        assert!(
            dense_err < 1e-3,
            "epoch {t0:e}, dense: {dense_err:.3e} m from the exact orbit"
        );
        assert_eq!(
            calls.get(),
            1391,
            "epoch {t0:e}: force evaluations, dense driver"
        );

        match reference {
            None => reference = Some((r, v)),
            Some((r_ref, v_ref)) => {
                let mut dr = 0.0_f64;
                let mut dv = 0.0_f64;
                for i in 0..3 {
                    dr = nan_max(dr, (r[i] - r_ref[i]).abs());
                    dv = nan_max(dv, (v[i] - v_ref[i]).abs());
                }
                // Not bit-identical, and it should not be: `t` is accumulated at magnitude
                // `|t0|`, so its representation error is `|t0|·2⁻⁵²` and the state inherits
                // `|t0|·2⁻⁵²·|v| = 1e6 · 2.22e-16 · 7.5e3 = 1.7e-6 m` and the same times `|a|` in
                // velocity. Measured: 3.7e-6 m, the predicted order. The bound is that estimate
                // with room, and the sharp assertion is the evaluation count above -- which is
                // identical across all three epochs, to the unit.
                assert!(
                    dr < 1e-4 && dv < 1e-7,
                    "epoch {t0:e} moved the answer by {dr:.3e} m / {dv:.3e} m/s"
                );
            }
        }
    }
}

#[test]
fn ias15_gives_up_on_a_fall_into_the_singularity_instead_of_grinding_to_a_halt() {
    // Dropped from rest, the orbit is degenerate: it falls straight down the x-axis into `r = 0`,
    // where `gravity` is unbounded. The controller answers correctly -- it shrinks the step without
    // limit -- and that is exactly the shape that has no end. The step reaches the point where
    // `t + dt` rounds back to `t`, and from there every further pass repeats the same instant, so
    // the loop condition can never be met again however long it runs.
    //
    // Free-fall time from 7000 km is `(pi/2) * sqrt(r0^3 / (2*mu))` = 1030.4 s. The span is set past
    // it on purpose: an integration that stops short of the singularity is not the case under test.
    let r0 = [7_000e3, 0.0, 0.0];
    let v0 = [0.0, 0.0, 0.0];

    assert!(
        matches!(
            integrate_ias15(&gravity, &r0, &v0, 0.0, 2_000.0, 60.0, DEFAULT_MAX_STEPS),
            Err(IntegrationError::StepUnderflow { .. })
        ),
        "the plain driver did not stop at the singularity"
    );
    assert!(
        matches!(
            integrate_ias15_dense::<3, 6, _>(
                &gravity,
                &r0,
                &v0,
                0.0,
                2_000.0,
                60.0,
                DEFAULT_MAX_STEPS
            ),
            Err(IntegrationError::StepUnderflow { .. })
        ),
        "the dense driver did not stop at the singularity"
    );
}

#[test]
fn ias15_stops_at_the_step_ceiling_it_is_given() {
    // The half of the guard no step floor can cover: every step of this run is healthy and there
    // are merely too many of them. A floor is blind to that by construction, because nothing ever
    // approaches the floor.
    //
    // The cost is measured, not assumed — the dense driver reports its own counts — and the cap is
    // then set below it. That is also what makes the assertion non-vacuous: the run stopped *at*
    // the ceiling and not a pass past it, and the count is an integer well inside f64's exact
    // range, so equality is the honest comparison.
    let (r0, v0, period) = state0();
    let step = period / 50.0;
    let healthy =
        integrate_ias15_dense::<3, 6, _>(&gravity, &r0, &v0, 0.0, period, step, DEFAULT_MAX_STEPS)
            .expect("one orbit is integrable")
            .step_counts();
    let needed = healthy.0 + healthy.1;
    assert!(needed > 4, "fixture too cheap to cap: {needed} passes");
    assert!(
        needed < DEFAULT_MAX_STEPS,
        "one orbit already exceeds the default: {needed}"
    );

    let cap = needed / 2;
    match integrate_ias15(&gravity, &r0, &v0, 0.0, period, step, cap) {
        Err(IntegrationError::StepLimit { steps }) => assert_eq!(steps, f64::from(cap), "plain"),
        other => panic!("expected StepLimit at {cap} passes from the plain driver, got {other:?}"),
    }
    match integrate_ias15_dense::<3, 6, _>(&gravity, &r0, &v0, 0.0, period, step, cap) {
        Err(IntegrationError::StepLimit { steps }) => assert_eq!(steps, f64::from(cap), "dense"),
        other => panic!("expected StepLimit at {cap} passes from the dense driver, got {other:?}"),
    }

    // A ceiling of zero admits no run at all, so it is an argument error rather than a limit
    // reached — the same reading `Controls`'s own zero-ceiling case gets in the driver.
    assert!(matches!(
        integrate_ias15(&gravity, &r0, &v0, 0.0, period, step, 0),
        Err(IntegrationError::BadArguments)
    ));
    assert!(matches!(
        integrate_ias15_dense::<3, 6, _>(&gravity, &r0, &v0, 0.0, period, step, 0),
        Err(IntegrationError::BadArguments)
    ));
}

/// A diverged state must end the run, not come back as an answer.
///
/// `f64::max` returns the *other* operand when one is NaN, so an acceleration that had left the reals
/// vanished from `a0_max`, which stayed at zero — and zero is how the controller spells "the forces
/// cancelled to rounding, grow the step freely". The run then marched to `t1` growing the step by
/// `1/SAFETY` each pass and returned `Ok` on a state of NaNs. Measured before the fix on the fixture
/// below: 68 evaluations and `Ok(([NaN], [NaN]))`; after it, 44 and `StepUnderflow` at the epoch.
///
/// The fixture is a harmonic oscillator of period 6.3e-6 s asked for a first step of 0.5 s — some
/// 80 000 periods — which overflows the b-fit within two sweeps.
#[test]
fn ias15_refuses_a_state_that_has_left_the_reals() {
    let stiff = |x: &[f64; 1]| [-1.0e12 * x[0]];
    for label in ["plain", "dense"] {
        let out = if label == "plain" {
            integrate_ias15::<1, _>(&stiff, &[1.0], &[0.0], 0.0, 1.0, 0.5, DEFAULT_MAX_STEPS)
        } else {
            integrate_ias15_dense::<1, 2, _>(
                &stiff,
                &[1.0],
                &[0.0],
                0.0,
                1.0,
                0.5,
                DEFAULT_MAX_STEPS,
            )
            .map(|traj| {
                let end = traj.final_state();
                ([end[0]], [end[1]])
            })
        };
        match out {
            Err(IntegrationError::StepUnderflow { t }) => assert_eq!(t, 0.0, "{label}"),
            Ok((r, v)) => panic!("{label}: returned r = {r:?}, v = {v:?}"),
            other => panic!("{label}: {other:?}"),
        }
    }
}

/// The step controller is dimensionless, so the same orbit must cost the same in metres and in
/// kilometres.
///
/// Both of the controller's error measures are ratios of accelerations — `db6/a7` inside the
/// predictor–corrector sweep and `b6/a0` for the step size — and `EPSILON` is a fractional
/// tolerance. Rescale lengths by 1000 and every acceleration scales with them, so both ratios and
/// therefore every decision the controller takes are unchanged. A mutant that turns either ratio
/// into a product breaks exactly that: the criterion picks up units of acceleration squared and the
/// two runs stop agreeing, which nothing else in this file can see because every other fixture is in
/// metres.
///
/// The states are compared after rescaling rather than bit for bit: `1e-3` is not a power of two, so
/// the two runs are the same arithmetic on differently rounded inputs.
#[test]
fn ias15_costs_the_same_whatever_length_unit_the_orbit_is_given_in() {
    fn run(mu: f64, a: f64) -> (u32, [f64; 3], [f64; 3]) {
        let r_p = a * (1.0 - ECC);
        let v_p = (mu * (1.0 + ECC) / r_p).sqrt();
        let period = std::f64::consts::TAU * (a.powi(3) / mu).sqrt();
        let calls = std::cell::Cell::new(0u32);
        let counted = |r: &[f64; 3]| {
            calls.set(calls.get() + 1);
            let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            let k = -mu / (d * d * d);
            [k * r[0], k * r[1], k * r[2]]
        };
        let (r, v) = integrate_ias15::<3, _>(
            &counted,
            &[r_p, 0.0, 0.0],
            &[0.0, v_p, 0.0],
            0.0,
            2.0 * period,
            period / 50.0,
            DEFAULT_MAX_STEPS,
        )
        .expect("two orbits");
        (calls.get(), r, v)
    }

    let (calls_metre, in_metres, speed_metres) = run(MU, SEMI_MAJOR);
    let (calls_km, in_km, speed_km) = run(MU * 1e-9, SEMI_MAJOR * 1e-3);
    println!("IAS15 units: {calls_metre} evaluations in metres, {calls_km} in kilometres");
    // A band and not equality, because 1e-3 is not a power of two: the two runs are the same
    // decisions taken on differently rounded numbers, and a step boundary can fall either side.
    // Measured 2711 against 2718 as written -- 0.26% -- and 2711 against 1871, 31% apart, with the
    // sweep's `db6/a7` mutated to `db6*a7`. The band has two orders of margin either way.
    let spread = i64::from(calls_km) - i64::from(calls_metre);
    assert!(
        spread.abs() * 100 <= i64::from(calls_metre),
        "the controller is not scale-free: {calls_metre} against {calls_km}"
    );
    for i in 0..3 {
        let (want_r, want_v) = (in_metres[i] * 1e-3, speed_metres[i] * 1e-3);
        let scale = SEMI_MAJOR * 1e-3;
        assert!(
            (in_km[i] - want_r).abs() < 1e-9 * scale,
            "r[{i}]: {} vs {want_r}",
            in_km[i]
        );
        assert!(
            (speed_km[i] - want_v).abs() < 1e-9 * scale,
            "v[{i}]: {} vs {want_v}",
            speed_km[i]
        );
    }
}

/// What a step hopelessly larger than the dynamics costs, and where the corrector gives up.
///
/// A linear force is the blind spot of a *fractional* error criterion: `a = -k x` makes the fitted
/// acceleration polynomial self-consistent at any step size, so `b6/a0` stays moderate however
/// meaningless the answer is, and an early step is accepted that should not be. REBOUND's controller
/// has the same blind spot; it is recorded here rather than worked around, because the alternative is
/// an absolute tolerance that the method deliberately does not have. `ω = 1e4 rad/s` puts 1592 full
/// oscillations inside the unit span, and the run pays for every one of them and still comes back
/// with a number 1e110 times the true bound.
///
/// What that leaves observable is the price: the corrector's oscillation break (`sweeps > 2 &&
/// err >= predictor_error`) and the rejection arm's `dt_new / dt` ratio both decide this run, and
/// neither decides any Kepler fixture above, where the corrector improves monotonically to
/// `CONVERGENCE`.
#[test]
fn ias15_pays_a_measured_price_for_a_step_far_larger_than_the_dynamics() {
    let stiff = |x: &[f64; 1]| [-1.0e8 * x[0]];
    let calls = std::cell::Cell::new(0u32);
    let counted = |x: &[f64; 1]| {
        calls.set(calls.get() + 1);
        stiff(x)
    };
    let (r, _v) =
        integrate_ias15::<1, _>(&counted, &[1.0], &[0.0], 0.0, 1.0, 1.0, DEFAULT_MAX_STEPS)
            .expect("a linear force never trips the fractional criterion");
    let calls = calls.get();
    println!(
        "IAS15 oversized step: {calls} evaluations, x(1) = {:.3e}",
        r[0]
    );
    assert_eq!(calls, 1_233_516, "{calls} evaluations");
    // Not an accuracy assertion but its opposite: |x| must be nowhere near the true |x| <= 1, so the
    // count above is not being read off a run that accidentally worked.
    assert!(r[0].abs() > 1e100, "x(1) = {:.3e} is not the runaway", r[0]);
}

/// A first step covering the whole span is adapted away, not accepted because it is also the last.
///
/// The driver clamps every step to `t1 - t`, so when `dt0` covers the span the first step is *equal*
/// to the remainder. An exemption that spared the clamped final step from rejection therefore spared
/// the one step that most needed rejecting, and the integrator returned `Ok` on a single Gauss–Radau
/// step across an entire orbit: measured 4.067e3 m of closure error at `e = 0`, 5.081e6 m at
/// `e = 0.5` and 1.999e9 m at `e = 0.95` — three metres short of the orbit itself.
///
/// The oracle is the orbit's own periodicity: after exactly one period the state is the state it
/// started in, which owes nothing to this integrator. `dt0 = period` is not a pathological input —
/// it is what a caller who does not know the dynamics passes.
#[test]
fn ias15_adapts_a_first_step_that_spans_the_whole_orbit() {
    for (e, calls_ref) in [(0.0, 947u32), (0.5, 2621), (0.95, 5065)] {
        let period = std::f64::consts::TAU * (SEMI_MAJOR.powi(3) / MU).sqrt();
        let rp = SEMI_MAJOR * (1.0 - e);
        let vp = (MU * (1.0 + e) / rp).sqrt();
        let calls = std::cell::Cell::new(0u32);
        let counted = |s: &[f64; 2]| {
            calls.set(calls.get() + 1);
            let r3 = (s[0] * s[0] + s[1] * s[1]).powf(1.5);
            [-MU * s[0] / r3, -MU * s[1] / r3]
        };
        let (r, v) = integrate_ias15::<2, _>(
            &counted,
            &[rp, 0.0],
            &[0.0, vp],
            0.0,
            period,
            period,
            DEFAULT_MAX_STEPS,
        )
        .expect("one period of a bound orbit");
        let dr = ((r[0] - rp).powi(2) + r[1] * r[1]).sqrt();
        let dv = (v[0] * v[0] + (v[1] - vp).powi(2)).sqrt();
        let calls = calls.get();
        println!(
            "IAS15 dt0=span e={e}: {calls} evaluations, |Δr| = {dr:.3e} m, |Δv| = {dv:.3e} m/s"
        );
        assert!(dr < 1e-5, "e={e}: closure error {dr:.3e} m");
        assert!(dv < 1e-6, "e={e}: closure error {dv:.3e} m/s");
        assert_eq!(calls, calls_ref, "e={e}: force evaluations");
    }
}

/// A weak central force, where the rejection arm's step ratio is the only thing deciding the cost.
///
/// 433 Eros has `μ = 4.463e5 m³/s²` — nine orders below Earth's — so a 35 km orbit is held by
/// 3.6e-4 m/s². Nothing about the controller is supposed to care: its error measures are ratios of
/// accelerations. What does change is which arm of the step control the run spends its time in. A
/// LEO fixture accepts almost every step, so `predict_next_step`'s argument `dt_new / dt` is barely
/// exercised; here the run rejects often enough that corrupting that ratio moves the total.
///
/// μ is the value from NEAR-Shoemaker's radio tracking (Yeomans et al. 2000), not a round number.
#[test]
fn ias15_costs_what_a_weak_central_force_earns() {
    const MU_EROS: f64 = 4.463e5;
    let a = 35e3;
    let r_p = a * (1.0 - ECC);
    let v_p = (MU_EROS * (1.0 + ECC) / r_p).sqrt();
    let calls = std::cell::Cell::new(0u32);
    let counted = |r: &[f64; 3]| {
        calls.set(calls.get() + 1);
        let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let k = -MU_EROS / (d * d * d);
        [k * r[0], k * r[1], k * r[2]]
    };
    let (r, v) = integrate_ias15::<3, _>(
        &counted,
        &[r_p, 0.0, 0.0],
        &[0.0, v_p, 0.0],
        0.0,
        4.0e5,
        4.0e3,
        DEFAULT_MAX_STEPS,
    )
    .expect("eros orbit");
    let calls = calls.get();
    let energy = |r: &[f64; 3], v: &[f64; 3]| {
        let rm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - MU_EROS / rm
    };
    let e0 = energy(&[r_p, 0.0, 0.0], &[0.0, v_p, 0.0]);
    let rel = ((energy(&r, &v) - e0) / e0).abs();
    println!("IAS15 Eros orbit: {calls} evaluations, relative energy error {rel:.3e}");
    assert!(rel < 1e-13, "energy error {rel:.3e}");
    assert_eq!(calls, 8803, "{calls} evaluations");
}
