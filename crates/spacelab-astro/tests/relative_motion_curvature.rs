//! Where the straight-line relative frame stops being good enough, as a number.
//!
//! Clohessy–Wiltshire treats the along-track axis of the chief's local frame as a straight line. The
//! chief travels an arc, so a rectilinear offset carries a curvature error before any dynamics run
//! at all. [`cw_propagate_curvilinear`] removes that term by changing coordinates (Vallado & Alfano
//! 2014, CMDA 118(3):253–271, with the 2016 errata in 125:263–264). This file measures what the
//! change buys and — as importantly — what it does not.
//!
//! The truth here is exact rather than integrated: the deputy is the chief's *own* orbit, phase-
//! shifted by an along-track arc, so both states come from the same two-body flow and the residual
//! is the linear model's alone. The relative state is read in the chief's LVLH frame at `t = 0` and
//! both predictions are compared against the same frame at `t`.
//!
//! ## Co-orbital: the case the coordinates are for
//!
//! Circular chief at 7000 km, deputy on the same orbit ahead by `s`. Position error against the
//! nonlinear truth, metres:
//!
//! | `s` | rectilinear, `T/4` | rectilinear, `T` | curvilinear |
//! |---|---|---|---|
//! | 100 m | 0.0033 | 0.0269 | round-off |
//! | 1 km | 0.325 | 2.693 | round-off |
//! | 10 km | 32.52 | 269.3 | round-off |
//! | 100 km | 3 252 | 26 927 | round-off |
//! | 500 km | 81 268 | 672 912 | round-off |
//! | 1000 km | 324 656 | 2 688 217 | round-off |
//!
//! The curvilinear error is not "small", it is **zero to round-off** (worst 9.1e-9 m over the whole
//! sweep) — a phase offset on a circular orbit is a *constant* curvilinear state, and the model
//! propagates it unchanged. The rectilinear error is exactly quadratic in `s`, and stating it as a
//! fraction of the separation gives the number a caller can act on:
//!
//! **the rectilinear error reaches 1 % of the separation at 30.75 km and 10 % at 307.5 km over a
//! quarter period; over a full period at 3.71 km and 37.14 km.**
//!
//! `cw.rs`'s own accuracy test bounds the model at "well under 1 %" and measures it at a 2 km
//! separation, which the first of those numbers confirms and bounds: that 1 % holds to 30 km at a
//! quarter period, and only to 3.7 km if the prediction has to survive a whole orbit.
//!
//! ## Cross-track: where the change of coordinates is not free
//!
//! Add an inclination difference giving a cross-track offset `w`, and the curvilinear residual is
//! set by `w` alone — it does not grow with `s` (at `w` = 5 km it is 3.79 m at 1 km of along-track
//! separation and 3.50 m at 1000 km), but it is no longer zero, and it scales as `w²`: 0.152 m at
//! `w` = 1 km, 3.79 m at 5 km, 60.65 m at 20 km. Below `s ≈ 0.44 w` the **rectilinear** form is the
//! better of the two, by up to 1.7×. That is the honest crossover, and it is why this is a choice
//! rather than a strict upgrade.
//!
//! ## What it does not fix at all
//!
//! Curvature is a *coordinate* error. Clohessy–Wiltshire also assumes a circular chief, and an
//! eccentric one breaks the dynamics, which no change of variables reaches: at 100 m of separation
//! over a quarter period, where the circular curvilinear residual is round-off, `e` = 0.001 costs
//! 0.72 m and `e` = 0.01 costs 7.43 m — both forms alike.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_astro::cw::{cw_propagate, cw_propagate_curvilinear, relative_lvlh_state};
use spacelab_astro::kepler::propagate;
use spacelab_core::Vec3;

const MU: f64 = 3.986_004_418e14;
const TAU: f64 = core::f64::consts::TAU;
/// A chief radius in the middle of LEO — the regime where a formation or a rendezvous happens.
const CHIEF_RADIUS: f64 = 7_000e3;

/// Rectilinear error, curvilinear error and the initial separation, metres.
///
/// The deputy is the chief's own orbit advanced by the time it takes to cover an along-track arc
/// `along_track`, optionally tilted by `cross_track` about the along-track axis to stand in for an
/// inclination difference. Both states then come from the same two-body flow, so nothing in the
/// residual is integration error.
fn errors(along_track: f64, frac: f64, eccentricity: f64, cross_track: f64) -> (f64, f64, f64) {
    // Perigee state: |v| = √(μ(1+e)/(r(1−e))) puts the chief at perigee with the requested e, and
    // reduces to the circular speed at e = 0.
    let speed = (MU * (1.0 + eccentricity) / (CHIEF_RADIUS * (1.0 - eccentricity))).sqrt();
    let chief_r = Vec3::new(CHIEF_RADIUS, 0.0, 0.0);
    let chief_v = Vec3::new(0.0, speed, 0.0);
    // The mean motion the linear model is driven at, from the chief's actual semi-major axis.
    let a = 1.0 / (2.0 / CHIEF_RADIUS - speed * speed / MU);
    let n = (MU / (a * a * a)).sqrt();

    let (lead_r, lead_v) = propagate(chief_r, chief_v, MU, along_track / speed);
    let tilt = cross_track / CHIEF_RADIUS;
    let (sin, cos) = (tilt.sin(), tilt.cos());
    let rotate = |v: Vec3| Vec3::new(v.x, v.y * cos - v.z * sin, v.y * sin + v.z * cos);
    let (deputy_r, deputy_v) = (rotate(lead_r), rotate(lead_v));

    let s0 = relative_lvlh_state(chief_r, chief_v, deputy_r - chief_r, deputy_v - chief_v, n);

    let dt = frac * TAU / n;
    let (cr, cv) = propagate(chief_r, chief_v, MU, dt);
    let (dr, dv) = propagate(deputy_r, deputy_v, MU, dt);
    let truth = relative_lvlh_state(cr, cv, dr - cr, dv - cv, n);

    let gap = |p: [f64; 6]| {
        ((truth[0] - p[0]).powi(2) + (truth[1] - p[1]).powi(2) + (truth[2] - p[2]).powi(2)).sqrt()
    };
    (
        gap(cw_propagate(s0, n, dt)),
        gap(cw_propagate_curvilinear(s0, CHIEF_RADIUS, n, dt)),
        (s0[0] * s0[0] + s0[1] * s0[1] + s0[2] * s0[2]).sqrt(),
    )
}

/// The curvilinear coordinates are a change of variables, so the round trip has to be the identity —
/// including at separations far outside where the *dynamics* mean anything, because a transformation
/// that is only invertible in its comfortable range is not a transformation.
#[test]
fn the_curvilinear_round_trip_is_the_identity_at_every_separation() {
    use spacelab_astro::cw::{curvilinear_from_lvlh, lvlh_from_curvilinear};
    let mut worst: f64 = 0.0;
    for factor in [1.0, 10.0, 100.0, 1_000.0, 3_000.0] {
        let s = [
            1_000.0 * factor,
            -2_000.0 * factor,
            500.0 * factor,
            1.0,
            -0.5,
            0.3,
        ];
        let back = lvlh_from_curvilinear(curvilinear_from_lvlh(s, CHIEF_RADIUS), CHIEF_RADIUS);
        for (i, (&got, &want)) in back.iter().zip(&s).enumerate() {
            // Scaled by the component's own size: the position block reaches 6e6 m here, where an
            // absolute nanometre bound would be a statement about f64 and not about the map.
            let rel = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel < 1e-12,
                "factor {factor}, component {i}: {got} vs {want} ({rel:.3e} relative)"
            );
            worst = worst.max(rel);
        }
    }
    assert!(
        worst > 0.0,
        "an exactly-zero round-trip error would mean the forward map is the identity"
    );
}

/// At a separation where nothing is curved enough to matter, the curvilinear coordinates *are* the
/// rectilinear ones — which is what makes the change of variables safe to adopt as a default.
#[test]
fn the_two_coordinate_systems_agree_when_the_arc_is_short() {
    use spacelab_astro::cw::curvilinear_from_lvlh;
    let s = [100.0, -200.0, 50.0, 1.0, -0.5, 0.3];
    let c = curvilinear_from_lvlh(s, CHIEF_RADIUS);
    // The leading difference is the sagitta y²/(2 r_c) = 2.9 mm at 200 m of along-track separation.
    for (i, (&got, &want)) in c.iter().zip(&s).enumerate() {
        assert!(
            (got - want).abs() < 0.01,
            "component {i}: curvilinear {got} vs rectilinear {want}"
        );
    }
}

/// The co-orbital table in the module docs, asserted rather than printed. The claim is not that the
/// curvilinear error is small but that it is *absent*: a constant curvilinear state stays constant
/// under the model, so only round-off is left, at every separation out to 1000 km.
#[test]
fn a_phase_offset_is_exact_in_curvilinear_coordinates_and_quadratic_in_rectilinear_ones() {
    let mut worst_curvilinear: f64 = 0.0;
    for along_track in [100.0, 1e3, 1e4, 1e5, 5e5, 1e6] {
        for frac in [0.25, 1.0] {
            let (_, curvilinear, _) = errors(along_track, frac, 0.0, 0.0);
            worst_curvilinear = worst_curvilinear.max(curvilinear);
        }
    }
    assert!(
        worst_curvilinear < 1e-6,
        "co-orbital curvilinear residual {worst_curvilinear:.3e} m is above round-off"
    );

    // Ten times the separation, a hundred times the error: the rectilinear residual is the curvature
    // term and nothing else, which is what makes the 1 %/10 % thresholds below extrapolate.
    let decade = |small: f64, large: f64, frac: f64| {
        let (a, _, _) = errors(small, frac, 0.0, 0.0);
        let (b, _, _) = errors(large, frac, 0.0, 0.0);
        b / a
    };
    for (small, large, frac) in [(1e3, 1e4, 0.25), (1e4, 1e5, 0.25), (1e4, 1e5, 1.0)] {
        let growth = decade(small, large, frac);
        assert!(
            (99.0..101.0).contains(&growth),
            "{small:.0} m to {large:.0} m at {frac} period: error grew {growth:.2}×, not ~100×"
        );
    }
}

/// The deliverable of the ticket: the separation at which the rectilinear model's error is 1 % of
/// the separation, and where it is 10 %. Asserted at the separations the module docs name, so the
/// documented numbers cannot drift away from the code.
#[test]
fn the_rectilinear_error_crosses_one_percent_and_ten_percent_at_the_documented_separations() {
    // (along-track separation, fraction of a period, the percentage claimed there)
    let cases = [
        (30_750.0, 0.25, 1.0),
        (307_500.0, 0.25, 10.0),
        (3_714.0, 1.0, 1.0),
        (37_140.0, 1.0, 10.0),
    ];
    for (along_track, frac, claimed) in cases {
        let (rectilinear, _, separation) = errors(along_track, frac, 0.0, 0.0);
        let percent = 100.0 * rectilinear / separation;
        assert!(
            (percent - claimed).abs() < 0.01 * claimed,
            "{:.2} km at {frac} period: rectilinear error is {percent:.4} % of the separation, \
             documented as {claimed} %",
            along_track / 1e3
        );
    }
}

/// Cross-track motion is where the change of coordinates stops being free: the curvilinear residual
/// is set by the cross-track offset rather than the along-track separation, so it is bounded where
/// the rectilinear one is quadratic — but it is not zero, and at short along-track separation it is
/// the *worse* of the two.
#[test]
fn a_cross_track_offset_bounds_the_curvilinear_error_and_can_invert_the_ordering() {
    // Bounded in the along-track separation: 1 km against 1000 km, same cross-track offset.
    let (_, near, _) = errors(1e3, 0.25, 0.0, 5_000.0);
    let (rect_far, far, _) = errors(1e6, 0.25, 0.0, 5_000.0);
    assert!(
        (far - near).abs() < 0.1 * near,
        "curvilinear residual moved from {near:.3} m to {far:.3} m over three decades of \
         along-track separation; it should be set by the cross-track offset alone"
    );
    assert!(
        rect_far > 1_000.0 * far,
        "at 1000 km along-track the rectilinear form should be orders worse: \
         {rect_far:.1} m against {far:.3} m"
    );

    // Quadratic in the cross-track offset, which is what makes the crossover predictable.
    let (_, at_5km, _) = errors(1e3, 0.25, 0.0, 5_000.0);
    let (_, at_20km, _) = errors(1e3, 0.25, 0.0, 20_000.0);
    let growth = at_20km / at_5km;
    assert!(
        (15.0..17.0).contains(&growth),
        "quadrupling the cross-track offset grew the residual {growth:.2}×, not ~16×"
    );

    // And the inversion, asserted rather than hedged: with 20 km of cross-track and 1 km of
    // along-track, the rectilinear form wins. A caller below s ≈ 0.44 w should not switch.
    let (rect_near, curv_near, _) = errors(1e3, 0.25, 0.0, 20_000.0);
    assert!(
        rect_near < curv_near,
        "the documented inversion is gone: rectilinear {rect_near:.2} m, \
         curvilinear {curv_near:.2} m"
    );
}

/// An eccentric chief breaks the *dynamics*, and a change of coordinates does not repair dynamics.
/// Without this, the tables above read as a claim about relative motion in general.
#[test]
fn curvilinear_coordinates_do_not_repair_an_eccentric_chief() {
    let (_, circular, _) = errors(100.0, 0.25, 0.0, 0.0);
    assert!(circular < 1e-6, "circular baseline {circular:.3e} m");

    for (eccentricity, floor) in [(0.001, 0.7), (0.01, 7.0)] {
        let (rectilinear, curvilinear, _) = errors(100.0, 0.25, eccentricity, 0.0);
        assert!(
            curvilinear > floor,
            "e = {eccentricity} at 100 m of separation: curvilinear {curvilinear:.3} m, \
             expected the eccentricity floor above {floor} m"
        );
        // Both forms alike: the eccentricity error is common to them, not something one absorbs.
        assert!(
            (rectilinear - curvilinear).abs() < 0.01 * curvilinear,
            "e = {eccentricity}: rectilinear {rectilinear:.3} m and curvilinear \
             {curvilinear:.3} m should carry the same eccentricity error"
        );
    }
}
