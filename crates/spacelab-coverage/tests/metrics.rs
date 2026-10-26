//! Coverage metrics, checked against hand-computed gaps and, for a periodic pattern, against the exact
//! rationals of `derivations/metrics_periodic_gaps.mac`.

// `float_cmp`: on integer-second windows every sum is exact in f64, so each metric is one correctly
// rounded quotient and equals the f64 nearest the exact rational bit for bit.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use spacelab_coverage::{Window, metrics};

fn w(rise: f64, set: f64) -> Window {
    Window { rise, set }
}

#[test]
fn metrics_match_the_hand_computed_gaps() {
    // Two passes on [0, 100]: covered 10+10, gaps 10 (lead), 30 (between), 40 (trail).
    let windows = [w(10.0, 20.0), w(50.0, 60.0)];
    let m = metrics(&windows, 0.0, 100.0);
    assert_eq!(m.passes, 2);
    assert!((m.covered_seconds - 20.0).abs() < 1e-12);
    assert!((m.coverage_fraction - 0.2).abs() < 1e-12);
    assert!((m.max_gap - 40.0).abs() < 1e-12, "max gap {}", m.max_gap);
    assert!(
        (m.mean_gap - (10.0 + 30.0 + 40.0) / 3.0).abs() < 1e-12,
        "mean gap {}",
        m.mean_gap
    );
    assert!((m.longest_pass - 10.0).abs() < 1e-12);
    assert!((m.shortest_pass - 10.0).abs() < 1e-12);
}

#[test]
fn no_passes_means_the_whole_span_is_one_gap() {
    let m = metrics(&[], 0.0, 100.0);
    assert_eq!(m.passes, 0);
    assert!((m.covered_seconds).abs() < 1e-12);
    assert!((m.coverage_fraction).abs() < 1e-12);
    assert!(
        (m.max_gap - 100.0).abs() < 1e-12,
        "an empty schedule is a 100 s gap"
    );
}

#[test]
fn continuous_coverage_has_no_gap() {
    // A single window spanning the whole interval: fraction 1, max gap 0.
    let m = metrics(&[w(0.0, 100.0)], 0.0, 100.0);
    assert_eq!(m.passes, 1);
    assert!((m.coverage_fraction - 1.0).abs() < 1e-12);
    assert!(
        m.max_gap.abs() < 1e-12,
        "no uncovered interval, got {}",
        m.max_gap
    );
}

#[test]
fn every_metric_matches_its_closed_form_on_an_epoch_that_is_not_the_origin() {
    // The cases above all start at t0 = 0, where `t1 − t0` and `t1 + t0` are the same number and half
    // the arithmetic of `metrics` cannot be observed at all; and none of them reads
    // `mean_response_time`. Here the origin is 1000 s and every field is computed by hand.
    //
    //   span     = 1700 − 1000 = 700
    //   passes   = 30, 10, 40  → covered 80, fraction 80/700
    //   gaps     = 1120−1000 = 120 │ 1300−1150 = 150 │ 1600−1310 = 290 │ 1700−1640 = 60
    //   max 290, sum 620 over 4 → mean 155
    //   Σ gap²   = 14400 + 22500 + 84100 + 3600 = 124600
    //   response = 124600 / (2·700) = 89
    let windows = [w(1120.0, 1150.0), w(1300.0, 1310.0), w(1600.0, 1640.0)];
    let m = metrics(&windows, 1000.0, 1700.0);
    assert_eq!(m.passes, 3);
    assert!(
        (m.covered_seconds - 80.0).abs() < 1e-12,
        "{}",
        m.covered_seconds
    );
    assert!(
        (m.coverage_fraction - 80.0 / 700.0).abs() < 1e-12,
        "fraction {} — a span read off the wrong origin",
        m.coverage_fraction
    );
    assert!((m.max_gap - 290.0).abs() < 1e-12, "max gap {}", m.max_gap);
    assert!(
        (m.mean_gap - 155.0).abs() < 1e-12,
        "mean gap {}",
        m.mean_gap
    );
    assert!(
        (m.mean_response_time - 89.0).abs() < 1e-12,
        "mean response time {} — the Wertz Σgap²/(2·span)",
        m.mean_response_time
    );
    assert!((m.longest_pass - 40.0).abs() < 1e-12, "{}", m.longest_pass);
    assert!(
        (m.shortest_pass - 10.0).abs() < 1e-12,
        "{}",
        m.shortest_pass
    );
}

#[test]
fn an_empty_schedule_waits_half_the_span_on_average() {
    // With no passes the wait is uniform on [0, span], so its time-average is span/2 — the same
    // Σgap²/(2·span) with one gap of the whole span. Nothing above reads this field.
    let m = metrics(&[], 1000.0, 1700.0);
    assert!(
        (m.mean_response_time - 350.0).abs() < 1e-12,
        "mean response time {} over an empty 700 s span",
        m.mean_response_time
    );
    assert!((m.max_gap - 700.0).abs() < 1e-12, "max gap {}", m.max_gap);
    assert!(
        (m.mean_gap - 700.0).abs() < 1e-12,
        "mean gap {}",
        m.mean_gap
    );
}

#[test]
fn a_zero_length_span_reports_no_response_time_rather_than_dividing_by_it() {
    // Both response-time branches guard on `span > 0`, and the guard is the whole of what separates
    // a reported zero from a 0/0. An analysis window of no length is degenerate but reachable — a
    // caller slicing a schedule at a single instant — and it must not produce NaN.
    for (windows, what) in [
        (&[][..], "no passes"),
        (&[w(5.0, 5.0)][..], "one instantaneous pass"),
    ] {
        let m = metrics(windows, 5.0, 5.0);
        assert!(
            m.mean_response_time.abs() < 1e-12,
            "{what} over a zero-length span gave {}",
            m.mean_response_time
        );
    }
}

#[test]
fn uneven_passes_report_the_extremes() {
    let windows = [w(5.0, 6.0), w(20.0, 40.0), w(90.0, 95.0)];
    let m = metrics(&windows, 0.0, 100.0);
    assert_eq!(m.passes, 3);
    assert!(
        (m.longest_pass - 20.0).abs() < 1e-12,
        "longest {}",
        m.longest_pass
    );
    assert!(
        (m.shortest_pass - 1.0).abs() < 1e-12,
        "shortest {}",
        m.shortest_pass
    );
    // gaps: 5 (lead), 14, 50, 5 (trail) → max 50
    assert!((m.max_gap - 50.0).abs() < 1e-12, "max gap {}", m.max_gap);
}

/// (G3) A periodic access pattern — 16 ten-minute passes every 90 minutes, the first 1000 s in, over
/// one day — against the exact rationals derived from the metrics' definitions (the mean response
/// as the time-average of the wait to the next pass) in `derivations/metrics_periodic_gaps.mac` and
/// enumerated in `derivations/metrics_periodic_gaps.g`: coverage 1/9, max gap 4800, mean gap
/// 76800/17, mean response 112825/54.
///
/// On an absolute epoch (t0 = 8.64e8 s) every window edge and every sum is an integer below 2^53,
/// so each metric is a single correctly rounded quotient: the bound is 0 ulps.
#[test]
fn a_periodic_pattern_has_the_exact_rational_metrics() {
    let t0 = 8.64e8_f64;
    let windows: Vec<Window> = (0..16)
        .map(|i| {
            let rise = t0 + 1000.0 + 5400.0 * f64::from(i);
            w(rise, rise + 600.0)
        })
        .collect();
    let m = metrics(&windows, t0, t0 + 86_400.0);
    assert_eq!(m.passes, 16);
    assert_eq!(m.covered_seconds, 9600.0);
    assert_eq!(m.coverage_fraction, 1.0 / 9.0);
    assert_eq!(m.max_gap, 4800.0);
    assert_eq!(m.mean_gap, 76_800.0 / 17.0);
    assert_eq!(m.mean_response_time, 112_825.0 / 54.0);
    assert_eq!(m.longest_pass, 600.0);
    assert_eq!(m.shortest_pass, 600.0);
}
