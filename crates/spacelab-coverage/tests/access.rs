//! Access-window scanning, checked against closed-form pass geometry.
//!
//! * A bare `sin` switching function yields windows at its analytic zeros.
//! * A circular pass over a ground cell opens and closes at `±acos(Rₑ/r)/ω` — the horizon crossings.
//! * A higher elevation mask shortens the window; its edges are genuine zeros of `g`.
//! * A window far narrower than the natural scan step is still found — the adaptive step shrinks to
//!   it, the whole point of stepping by `|g|/max_rate`.
//! * The same narrow window is still found when the caller's time base is an absolute epoch, where
//!   one ulp is larger than the step floor.
//! * The scan's sign test treats an exact zero as covered, at either end of a probe interval.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::many_single_char_names,
    // The exact comparisons below are exact by construction: a clamp to a scan bound reproduces that
    // bound bit for bit, and 160−100 and (100+160)/2 are integers in f64.
    clippy::float_cmp
)]

use std::f64::consts::{PI, TAU};

use spacelab_core::Vec3;
use spacelab_coverage::{Window, access_windows, windows};
use spacelab_sensors::{ElevationMask, OrbitBound};

const RE: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;

fn collect_windows_of<G: FnMut(f64) -> f64>(
    g: G,
    max_rate: f64,
    t0: f64,
    t1: f64,
    tol: f64,
) -> Vec<Window> {
    let mut out = Vec::new();
    windows(g, max_rate, t0, t1, tol, |w| out.push(w));
    out
}

#[test]
fn windows_of_sine_land_on_its_zeros() {
    // sin ≥ 0 on (0, π) and (2π, 3π); |d sin/dt| ≤ 1 is a tight rate bound.
    let ws = collect_windows_of(f64::sin, 1.0, 0.5, 10.0, 1e-10);
    assert_eq!(ws.len(), 2, "expected two windows, got {}", ws.len());
    assert!(
        (ws[0].rise - 0.5).abs() < 1e-9,
        "already covered at t0 → clamp to t0"
    );
    assert!(
        (ws[0].set - PI).abs() < 1e-8,
        "first close at π, got {}",
        ws[0].set
    );
    assert!(
        (ws[1].rise - TAU).abs() < 1e-8,
        "second open at 2π, got {}",
        ws[1].rise
    );
    assert!(
        (ws[1].set - 3.0 * PI).abs() < 1e-8,
        "second close at 3π, got {}",
        ws[1].set
    );
}

/// A circular orbit of radius `r` in the x–y plane, at angular rate `ω`, centred on the cell at
/// `t_c` (satellite over the cell = `+x`).
fn circular_pass(r: f64, omega: f64, t_c: f64) -> impl Fn(f64) -> Vec3 {
    move |t: f64| {
        let theta = omega * (t - t_c);
        Vec3::new(r * theta.cos(), r * theta.sin(), 0.0)
    }
}

#[test]
fn a_pass_opens_and_closes_at_the_horizon_crossings() {
    let r = 7_000e3;
    let omega = (MU / (r * r * r)).sqrt();
    let t_c = 3_000.0;
    let cell = Vec3::new(RE, 0.0, 0.0);
    let zenith = Vec3::new(1.0, 0.0, 0.0);
    let orbit = OrbitBound {
        semi_major_axis: r,
        eccentricity: 0.0,
        mu: MU,
    };

    // horizon (elevation 0): g = (r cosθ − Rₑ), zero at cosθ = Rₑ/r, θ = ±acos(Rₑ/r).
    let half = (RE / r).acos() / omega;

    let mut ws = Vec::new();
    access_windows(
        &ElevationMask::HORIZON,
        &orbit,
        0.0,
        cell,
        zenith,
        circular_pass(r, omega, t_c),
        0.0,
        6_000.0,
        1e-3,
        |w| ws.push(w),
    );

    assert_eq!(ws.len(), 1, "one pass expected, got {}", ws.len());
    assert!(
        (ws[0].rise - (t_c - half)).abs() < 5e-3,
        "rise {} vs {}",
        ws[0].rise,
        t_c - half
    );
    assert!(
        (ws[0].set - (t_c + half)).abs() < 5e-3,
        "set {} vs {}",
        ws[0].set,
        t_c + half
    );
    assert!(
        (ws[0].midpoint() - t_c).abs() < 1e-2,
        "closest approach at the culmination"
    );
}

#[test]
fn a_higher_elevation_mask_shortens_the_window() {
    let r = 7_000e3;
    let omega = (MU / (r * r * r)).sqrt();
    let t_c = 3_000.0;
    let cell = Vec3::new(RE, 0.0, 0.0);
    let zenith = Vec3::new(1.0, 0.0, 0.0);
    let orbit = OrbitBound {
        semi_major_axis: r,
        eccentricity: 0.0,
        mu: MU,
    };
    let pass = circular_pass(r, omega, t_c);

    let mut horizon = Vec::new();
    access_windows(
        &ElevationMask::HORIZON,
        &orbit,
        0.0,
        cell,
        zenith,
        &pass,
        0.0,
        6_000.0,
        1e-4,
        |w| horizon.push(w),
    );
    let mut raised = Vec::new();
    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    access_windows(
        &mask,
        &orbit,
        0.0,
        cell,
        zenith,
        &pass,
        0.0,
        6_000.0,
        1e-6,
        |w| raised.push(w),
    );

    assert_eq!(horizon.len(), 1);
    assert_eq!(raised.len(), 1);
    assert!(
        raised[0].duration() < horizon[0].duration(),
        "10° mask must give a shorter pass"
    );
    // the window edges are genuine zeros of the 10°-mask switching function. g is in metres, so at a
    // 1 µs abscissa tolerance and ~7 km/s closing speed it pins to a few centimetres.
    let g = |t: f64| {
        let s = pass(t);
        let los = Vec3::new(s.x - cell.x, s.y - cell.y, s.z - cell.z);
        los.dot(zenith) - 10.0_f64.to_radians().sin() * los.norm()
    };
    assert!(
        g(raised[0].rise).abs() < 0.05,
        "rise is not a zero of g: {} m",
        g(raised[0].rise)
    );
    assert!(
        g(raised[0].set).abs() < 0.05,
        "set is not a zero of g: {} m",
        g(raised[0].set)
    );
}

#[test]
fn a_window_narrower_than_the_scan_step_is_not_missed() {
    // g = 0.01 − (t−5)² is positive only on (4.9, 5.1) — a 0.2 s window. |dg/dt| = |2(t−5)| ≤ 10 on
    // [0,10], so max_rate = 10 gives a natural step of ~0.1 away from the window; it is caught anyway
    // because the step shrinks to zero as g → 0 near the crossing.
    let g = |t: f64| 0.01 - (t - 5.0) * (t - 5.0);
    let ws = collect_windows_of(g, 10.0, 0.0, 10.0, 1e-9);
    assert_eq!(
        ws.len(),
        1,
        "the narrow window was stepped over: {} found",
        ws.len()
    );
    assert!((ws[0].rise - 4.9).abs() < 1e-6, "rise {}", ws[0].rise);
    assert!((ws[0].set - 5.1).abs() < 1e-6, "set {}", ws[0].set);
}

#[test]
fn the_same_narrow_window_is_found_on_an_absolute_epoch() {
    // The window above, moved to a caller time base of seconds since J2000. Nothing about the
    // geometry changed — only the magnitude of `t`, where one ulp is 0.12 µs, twelve times the
    // nanosecond step floor. A floor expressed only as a fraction of the span leaves `t + h == t`
    // and the scan never reaches `t1`: an infinite loop, not a wrong answer.
    let t0 = 8.64e8_f64; // 10 000 days
    let t_c = t0 + 5.0;
    assert!(
        t0.next_up() - t0 > 10.0 * 1e-9,
        "the epoch must be large enough that one ulp exceeds the span's nanosecond floor"
    );
    let g = |t: f64| 0.01 - (t - t_c) * (t - t_c);
    let ws = collect_windows_of(g, 10.0, t0, t0 + 10.0, 1e-6);
    assert_eq!(ws.len(), 1, "{} windows on an absolute epoch", ws.len());
    // The edges cannot be resolved below one ulp of the epoch; that is the f64 API's floor, not the
    // scanner's, and it is three orders below the tolerances coverage actually asks for.
    let ulp = t0.next_up() - t0;
    assert!(
        (ws[0].rise - (t_c - 0.1)).abs() < 4.0 * ulp,
        "rise {}",
        ws[0].rise
    );
    assert!(
        (ws[0].set - (t_c + 0.1)).abs() < 4.0 * ulp,
        "set {}",
        ws[0].set
    );
}

#[test]
fn coverage_open_at_t0_and_immediately_falling_closes_at_once() {
    // g(t0) = 0 exactly, decreasing: the satellite sits on the horizon at the scan start and sets.
    // The scan counts an exact zero as covered (`g >= 0` opens), so the window opens at t0 and must
    // close on the same instant. Reading the zero as *un*covered instead loses the falling crossing
    // altogether and reports the whole span as one long pass.
    let ws = collect_windows_of(|t| -t, 1.0, 0.0, 10.0, 1e-12);
    assert_eq!(ws.len(), 1, "expected one degenerate window, got {ws:?}");
    assert_eq!(ws[0].rise, 0.0);
    assert!(
        ws[0].set < 1e-6,
        "an immediately-falling zero must close at once, not run to t1: set = {}",
        ws[0].set
    );
}

#[test]
fn a_probe_landing_exactly_on_a_zero_still_opens_the_window() {
    // g = t − 5 with max_rate = 1 makes the first step |g(0)|/1 = 5, so the probe lands on t = 5
    // where g is exactly 0.0 — not near it, on it. That zero is the rising edge, and reading it as
    // negative would leave the window closed for the rest of the scan.
    let ws = collect_windows_of(|t| t - 5.0, 1.0, 0.0, 10.0, 1e-12);
    assert_eq!(ws.len(), 1, "expected one window, got {ws:?}");
    assert!((ws[0].rise - 5.0).abs() < 1e-11, "rise {}", ws[0].rise);
    assert_eq!(ws[0].set, 10.0, "still covered at t1 → clamp to t1");
}

#[test]
fn the_window_predicate_is_the_closed_interval() {
    let w = Window {
        rise: 100.0,
        set: 160.0,
    };
    assert_eq!(w.duration(), 60.0);
    assert_eq!(w.midpoint(), 130.0);
    // Closed at both ends: an event exactly on an edge is inside the pass.
    assert!(w.contains(w.rise));
    assert!(w.contains(w.set));
    assert!(w.contains(w.midpoint()));
    // and each bound is tested independently — a point failing only the lower one is outside, as is
    // a point failing only the upper one.
    assert!(!w.contains(w.rise - 1e-9));
    assert!(!w.contains(w.set + 1e-9));
    assert!(!w.contains(f64::NAN));
}

#[test]
fn a_scan_that_starts_mid_pass_clamps_the_window_to_t0() {
    let r = 7_000e3;
    let omega = (MU / (r * r * r)).sqrt();
    let t_c = 3_000.0;
    let cell = Vec3::new(RE, 0.0, 0.0);
    let zenith = Vec3::new(1.0, 0.0, 0.0);
    let orbit = OrbitBound {
        semi_major_axis: r,
        eccentricity: 0.0,
        mu: MU,
    };
    let half = (RE / r).acos() / omega;

    // start the scan at the culmination, where the cell is already covered
    let mut ws = Vec::new();
    access_windows(
        &ElevationMask::HORIZON,
        &orbit,
        0.0,
        cell,
        zenith,
        circular_pass(r, omega, t_c),
        t_c,
        6_000.0,
        1e-3,
        |w| ws.push(w),
    );

    assert_eq!(ws.len(), 1);
    assert!(
        (ws[0].rise - t_c).abs() < 1e-12,
        "an open window at t0 must clamp to t0, got {}",
        ws[0].rise
    );
    assert!((ws[0].set - (t_c + half)).abs() < 5e-3, "set {}", ws[0].set);
}
