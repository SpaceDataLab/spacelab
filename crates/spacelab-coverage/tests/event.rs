//! The sign-change scanner: correct roots, correct directions, and the Nyquist limit made explicit.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]

use core::f64::consts::PI;
use spacelab_coverage::{Event, EventKind, ScanError, find_events, find_events_rate_bounded};

fn collect<F: FnMut(f64) -> f64>(g: F, t0: f64, t1: f64, step: f64, tol: f64) -> Vec<Event> {
    let mut v = Vec::new();
    find_events(g, t0, t1, step, tol, |e| v.push(e));
    v
}

/// The rate-bounded scan's events, with its returned count held to the number it emitted.
fn collect_bounded<F: FnMut(f64) -> f64>(
    g: F,
    max_rate: f64,
    t0: f64,
    t1: f64,
    tol: f64,
) -> Vec<Event> {
    let mut v = Vec::new();
    let returned = find_events_rate_bounded(g, max_rate, t0, t1, tol, |e| v.push(e))
        .expect("the arguments are valid");
    assert_eq!(returned as usize, v.len(), "returned count against emitted");
    v
}

/// The zeros of `sin` on `[0.5, 7]` are π and 2π exactly. (G3)
#[test]
fn finds_the_zeros_of_a_sine_with_the_right_direction() {
    // sin(t) on [0.5, 7]: a falling zero at π, a rising zero at 2π.
    let ev = collect(|t: f64| t.sin(), 0.5, 7.0, 0.05, 1e-12);
    assert_eq!(
        ev.len(),
        2,
        "expected exactly two zeros of sin on [0.5, 7]: {ev:?}"
    );

    assert_eq!(ev[0].kind, EventKind::Falling);
    assert!(
        (ev[0].at.estimate() - PI).abs() < 1e-11,
        "first zero at {}",
        ev[0].at.estimate()
    );

    assert_eq!(ev[1].kind, EventKind::Rising);
    assert!(
        (ev[1].at.estimate() - 2.0 * PI).abs() < 1e-11,
        "second zero at {}",
        ev[1].at.estimate()
    );

    // events are in increasing time order and each bracket is valid
    assert!(ev[0].at.hi <= ev[1].at.lo);
    for e in &ev {
        assert!(e.at.f_lo.signum() != e.at.f_hi.signum() || e.at.f_lo == 0.0 || e.at.f_hi == 0.0);
    }
}

/// The roots of `A − (t − t_c)²` are `t_c ± √A` exactly. (G3)
#[test]
fn a_parabolic_pass_opens_then_closes_at_the_exact_roots() {
    // g(t) = A - (t - tc)²: a single access, opening at tc - √A, closing at tc + √A.
    let (a, tc) = (0.25_f64, 5.0_f64);
    let g = move |t: f64| a - (t - tc) * (t - tc);
    let ev = collect(g, 0.0, 10.0, 0.1, 1e-12);
    assert_eq!(ev.len(), 2, "one pass = one rising + one falling: {ev:?}");
    assert_eq!(ev[0].kind, EventKind::Rising);
    assert_eq!(ev[1].kind, EventKind::Falling);
    assert!((ev[0].at.estimate() - (tc - a.sqrt())).abs() < 1e-11);
    assert!((ev[1].at.estimate() - (tc + a.sqrt())).abs() < 1e-11);
}

#[test]
fn a_tangency_is_not_an_event() {
    // g(t) = (t - 5)² touches zero without crossing. It is not an access boundary, and a scanner
    // that reported it would invent a zero-duration pass. Orekit's findRoot skips these "fake
    // roots"; ours never brackets one because there is no sign change.
    let ev = collect(|t: f64| (t - 5.0) * (t - 5.0), 0.0, 10.0, 0.1, 1e-12);
    assert!(ev.is_empty(), "a tangency is not a crossing: {ev:?}");

    // even sampled so a grid point lands exactly on the touch, still no event
    let ev = collect(|t: f64| (t - 5.0) * (t - 5.0), 0.0, 10.0, 0.5, 1e-12);
    assert!(
        ev.is_empty(),
        "a tangency sampled at its vertex is still not a crossing: {ev:?}"
    );
}

#[test]
fn the_nyquist_limit_is_real_and_the_fix_is_a_smaller_step() {
    // A narrow pass of half-width 0.05 (full width 0.1). With a coarse 0.5 s step the whole pass
    // can fall between two samples, both negative, and be missed. With a step below the pass width
    // it is caught. This is the guarantee `find_events` documents, demonstrated.
    // Pass [4.72, 4.82] centred at 4.77 — the midpoint between the coarse samples at 4.52 and 5.02
    // (grid 0.02 + 0.5·k), so no coarse sample lands inside it and it is invisible to that scan.
    let (a, tc) = (0.0025_f64, 4.77_f64); // roots at 4.77 ± 0.05
    let g = move |t: f64| a - (t - tc) * (t - tc);
    let coarse = collect(g, 0.02, 10.0, 0.5, 1e-12);
    let fine = collect(g, 0.02, 10.0, 0.01, 1e-12);

    assert_eq!(
        fine.len(),
        2,
        "a step below the pass width catches both edges: {fine:?}"
    );
    // the coarse scan misses the pass entirely (0 events) — the documented Nyquist limit
    assert!(
        coarse.len() < 2,
        "coarse step should miss the sub-step pass, got {coarse:?}"
    );
}

#[test]
fn several_passes_are_all_found_in_order() {
    // cos over five periods: ten crossings, alternating falling/rising, all in time order.
    let ev = collect(|t: f64| t.cos(), 0.0, 5.0 * PI, 0.02, 1e-12);
    assert_eq!(ev.len(), 5, "cos has five zeros on [0, 5π): {}", ev.len());
    for w in ev.windows(2) {
        assert!(w[0].at.hi <= w[1].at.lo, "events out of order");
        assert_ne!(w[0].kind, w[1].kind, "crossings must alternate direction");
    }
    // cos(0)=+1, so the first crossing is falling
    assert_eq!(ev[0].kind, EventKind::Falling);
}

#[test]
fn a_crossing_exactly_on_a_sample_is_found_once() {
    // g has an exact zero at t = 3.0, which is also a grid point. It must be found, and found once.
    let ev = collect(|t: f64| t - 3.0, 0.0, 6.0, 1.0, 1e-12);
    assert_eq!(ev.len(), 1, "one rising crossing at t=3: {ev:?}");
    assert_eq!(ev[0].kind, EventKind::Rising);
    assert!((ev[0].at.estimate() - 3.0).abs() < 1e-11);
}

#[test]
fn degenerate_arguments_yield_no_events() {
    assert!(
        collect(|t: f64| t.sin(), 0.0, 0.0, 0.1, 1e-9).is_empty(),
        "empty interval"
    );
    assert!(
        collect(|t: f64| t.sin(), 5.0, 0.0, 0.1, 1e-9).is_empty(),
        "reversed interval"
    );
    assert_eq!(
        find_events(|t: f64| t.sin(), 0.0, 7.0, 0.0, 1e-9, |_| {}),
        0,
        "zero step"
    );
    assert_eq!(
        find_events(|t: f64| t.sin(), 0.0, 7.0, -0.1, 1e-9, |_| {}),
        0,
        "negative step"
    );
}

#[test]
fn the_returned_count_is_the_number_of_events_emitted() {
    // Every case above reads the events through the callback and throws the return value away, so
    // the counter that `find_events` maintains is free to be anything at all.
    let mut emitted = 0;
    let returned = find_events(|t: f64| t.sin(), 0.5, 7.0, 0.05, 1e-12, |_| emitted += 1);
    assert_eq!(emitted, 2, "two zeros of sin on [0.5, 7]");
    assert_eq!(
        returned, emitted,
        "the return value disagrees with the callback"
    );
}

#[test]
fn a_sample_landing_exactly_on_a_tangency_is_not_a_crossing() {
    // `g = −(t−5)²` is negative either side and exactly zero at t = 5, which the step of 1 s lands
    // on precisely. `g` does not change sign there, so no boundary exists; reading the zero as
    // positive instead would invent two — an entry and an exit that never happened.
    let mut emitted = 0;
    let returned = find_events(
        |t: f64| -(t - 5.0) * (t - 5.0),
        0.0,
        10.0,
        1.0,
        1e-12,
        |_| {
            emitted += 1;
        },
    );
    assert_eq!(emitted, 0, "a touch at zero is not an access boundary");
    assert_eq!(returned, 0);
}

#[test]
fn an_empty_interval_is_rejected_before_g_is_evaluated() {
    // `t1 > t0` is a precondition, not a hint: with no interval there is nothing to scan, and the
    // scanner must not sample the switching function to discover that. Counting the calls is the
    // only way to see it — the event count is zero either way.
    let mut calls = 0;
    let returned = find_events(
        |t: f64| {
            calls += 1;
            t.sin()
        },
        5.0,
        5.0,
        0.1,
        1e-12,
        |_| {},
    );
    assert_eq!(returned, 0, "no events on an empty interval");
    assert_eq!(
        calls, 0,
        "g was evaluated {calls} times on an empty interval"
    );
}

/// The planted pair (rule 16): a pass whose two crossings lie inside one fixed step. `find_events`
/// misses both, as its doc states; the rate-bounded scan finds both at `t_c ∓ √A` exactly. (G3)
#[test]
fn a_pair_inside_one_fixed_step_is_missed_by_the_fixed_scan_and_found_by_the_rate_bound() {
    // Crossings at 4.72 and 4.82. The fixed samples 0.02 + 0.5·k put 4.52 and 5.02 around them, and
    // g = A − 0.0625 < 0 at both: no sign change, no event.
    let (a, tc) = (0.0025_f64, 4.77_f64);
    let g = move |t: f64| a - (t - tc) * (t - tc);
    let (t0, t1) = (0.02, 10.0);
    // |dg/dt| = 2·|t − tc| ≤ 2·max(tc − t0, t1 − tc) = 10.46 on [t0, t1].
    let max_rate = 11.0;

    let fixed = collect(g, t0, t1, 0.5, 1e-12);
    assert!(
        fixed.is_empty(),
        "the fixed scan reports neither crossing: {fixed:?}"
    );

    let bounded = collect_bounded(g, max_rate, t0, t1, 1e-12);
    assert_eq!(bounded.len(), 2, "both crossings: {bounded:?}");
    assert_eq!(bounded[0].kind, EventKind::Rising);
    assert_eq!(bounded[1].kind, EventKind::Falling);
    assert!(
        (bounded[0].at.estimate() - (tc - a.sqrt())).abs() < 1e-11,
        "rise at {}",
        bounded[0].at.estimate()
    );
    assert!(
        (bounded[1].at.estimate() - (tc + a.sqrt())).abs() < 1e-11,
        "set at {}",
        bounded[1].at.estimate()
    );
}

/// The guarantee rests on the bound. The same pass with a bound a hundred times too small: the first
/// step, `|g(t0)| / max_rate ≈ 205`, jumps the whole span.
#[test]
fn an_understated_rate_bound_loses_the_pair() {
    let (a, tc) = (0.0025_f64, 4.77_f64);
    let g = move |t: f64| a - (t - tc) * (t - tc);
    let ev = collect_bounded(g, 0.11, 0.02, 10.0, 1e-12);
    assert!(ev.is_empty(), "a false bound is no guarantee: {ev:?}");
}

/// `|d sin/dt| ≤ 1`, a tight bound: the zeros on `[0.5, 7]` are π and 2π exactly. (G3)
#[test]
fn the_rate_bounded_scan_lands_on_the_zeros_of_a_sine() {
    let ev = collect_bounded(f64::sin, 1.0, 0.5, 7.0, 1e-12);
    assert_eq!(ev.len(), 2, "two zeros of sin on [0.5, 7]: {ev:?}");
    assert_eq!(ev[0].kind, EventKind::Falling);
    assert_eq!(ev[1].kind, EventKind::Rising);
    assert!((ev[0].at.estimate() - PI).abs() < 1e-11);
    assert!((ev[1].at.estimate() - 2.0 * PI).abs() < 1e-11);
}

/// Rule 23: each refusal is typed, and `g` is never evaluated to reach it. The scan bounds sit at
/// `2⁵²`, where four ulps make the step floor about 4 s, so a guard that let a case through would end
/// in two steps with `Ok(0)` rather than hang.
#[test]
fn bad_arguments_are_refused_by_type_before_g_is_evaluated() {
    let (t0, t1) = (4_503_599_627_370_496.0_f64, 4_503_599_627_370_504.0_f64);
    let cases = [
        (0.0, t0, t1, 1e-9, ScanError::BadRate),
        (-1.0, t0, t1, 1e-9, ScanError::BadRate),
        (f64::NAN, t0, t1, 1e-9, ScanError::BadRate),
        (f64::INFINITY, t0, t1, 1e-9, ScanError::BadRate),
        (1.0, t0, t0, 1e-9, ScanError::BadInterval),
        (1.0, t1, t0, 1e-9, ScanError::BadInterval),
        (1.0, f64::NEG_INFINITY, t1, 1e-9, ScanError::BadInterval),
        (1.0, t0, f64::INFINITY, 1e-9, ScanError::BadInterval),
        (1.0, t0, f64::NAN, 1e-9, ScanError::BadInterval),
        (1.0, f64::NAN, t1, 1e-9, ScanError::BadInterval),
        // The floor underflows to zero: (1e-316)·1e-9 and 4ε·1e-316 both round to 0.
        (1.0, 0.0, 1e-316, 1e-9, ScanError::BadInterval),
        (1.0, t0, t1, 0.0, ScanError::BadTolerance),
        (1.0, t0, t1, -1e-9, ScanError::BadTolerance),
        (1.0, t0, t1, f64::NAN, ScanError::BadTolerance),
        (1.0, t0, t1, f64::INFINITY, ScanError::BadTolerance),
    ];
    for (max_rate, a, b, tol, want) in cases {
        let mut calls = 0;
        let got = find_events_rate_bounded(
            |t: f64| {
                calls += 1;
                1.0 + t.abs()
            },
            max_rate,
            a,
            b,
            tol,
            |_| {},
        );
        assert_eq!(got, Err(want), "max_rate {max_rate}, [{a}, {b}], tol {tol}");
        assert_eq!(calls, 0, "g evaluated on a refused call: {want:?}");
    }

    // Just above the underflow the floor is the subnormal 1e-319, and the scan runs.
    assert_eq!(
        find_events_rate_bounded(f64::sin, 1.0, 0.0, 1e-310, 1e-320, |_| {}),
        Ok(0)
    );
}
