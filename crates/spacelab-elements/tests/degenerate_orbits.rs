//! The degenerate-orbit branches of `from_state` and the conic radius accessors.
//!
//! The main keplerian round-trip test uses generic inclined-elliptical orbits, leaving the
//! circular / equatorial special cases (where `ω` or `Ω` is undefined) and the
//! semi-latus-rectum / apsis accessors unexercised.
#![allow(clippy::excessive_precision)] // Maxima references, kept at the width they were computed

use spacelab_core::{Angle, Vec3};
use spacelab_elements::KeplerianElements;

const MU: f64 = 3.986_004_418e14;

fn elements(
    a: f64,
    e: f64,
    i_deg: f64,
    raan_deg: f64,
    argp_deg: f64,
    nu_deg: f64,
) -> KeplerianElements {
    KeplerianElements {
        semi_major_axis: a,
        eccentricity: e,
        inclination: Angle::from_degrees(i_deg),
        raan: Angle::from_degrees(raan_deg),
        arg_periapsis: Angle::from_degrees(argp_deg),
        true_anomaly: Angle::from_degrees(nu_deg),
        mu: MU,
    }
}

#[test]
fn degenerate_orbits_round_trip_through_the_cartesian_state() {
    // Circular-inclined, elliptical-equatorial and circular-equatorial each take a different
    // from_state branch; the angle it substitutes for the undefined element must still reproduce the
    // same (r, v).
    for (label, a, e, i, raan, argp, nu) in [
        ("circular inclined", 7_000e3, 0.0, 45.0, 30.0, 0.0, 60.0),
        ("elliptical equatorial", 7_000e3, 0.1, 0.0, 0.0, 40.0, 60.0),
        ("circular equatorial", 7_000e3, 0.0, 0.0, 0.0, 0.0, 120.0),
    ] {
        let els = elements(a, e, i, raan, argp, nu);
        let (r, v) = els.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        let (r2, v2) = back.to_state();
        assert!((r2 - r).norm() < 1e-6, "position round-trip: {label}");
        assert!((v2 - v).norm() < 1e-9, "velocity round-trip: {label}");
        // a and e are well defined even where the angles are not.
        assert!((back.semi_major_axis - a).abs() < 1e-3, "a: {label}");
        assert!((back.eccentricity - e).abs() < 1e-9, "e: {label}");
    }
}

#[test]
fn the_radius_accessors_are_the_conic_formulas() {
    let (a, e) = (7_000e3, 0.1);
    let els = elements(a, e, 45.0, 0.0, 0.0, 0.0);
    assert!((els.semi_latus_rectum() - a * (1.0 - e * e)).abs() < 1e-6);
    assert!((els.periapsis_radius() - a * (1.0 - e)).abs() < 1e-6);
    assert!((els.apoapsis_radius() - a * (1.0 + e)).abs() < 1e-6);
    // At periapsis (ν = 0) the state radius equals the periapsis radius.
    let (r, _) = elements(a, e, 45.0, 0.0, 0.0, 0.0).to_state();
    assert!((r.norm() - a * (1.0 - e)).abs() < 1e-3);
}

#[test]
fn a_state_with_no_orbital_plane_has_no_angles() {
    // `r ∥ v` is a line, not a conic in a plane, so the three cases above do not cover it: there is
    // no node to measure Ω from and no plane to measure i in. The line lies in every plane through
    // it, and the first case says what that costs — a fall straight down the pole came back as an
    // equatorial orbit, i = Ω = 0, at three angles whose round trip is NaN because p = a(1 − e²) is
    // zero. `a` and `e` are the two that do survive: energy fixes a, and the eccentricity vector
    // reduces to −r̂, so e = 1 — the rectilinear ellipse, apoapsis at r and periapsis at the centre.
    // The three semi-major axes are `μ / (2(μ/r − v²/2))` evaluated in Maxima at 40 digits: the two
    // at rest reduce to r/2, which is the statement that a body dropped from r has its apoapsis
    // there, and the third is the vis-viva value for the same line climbed at 1e-4 r per second.
    for (label, r, v, a) in [
        (
            "free fall down the pole",
            Vec3::new(0.0, 0.0, 7_000e3),
            Vec3::new(0.0, 0.0, 0.0),
            3.5e6,
        ),
        (
            "free fall in the equator",
            Vec3::new(7_000e3, 1_000e3, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            3.535_533_905_932_737_6e6,
        ),
        (
            "climbing radially",
            Vec3::new(7_000e3, 1_000e3, 2_000e3),
            Vec3::new(700.0, 100.0, 200.0),
            3.692_615_096_739_608_8e6,
        ),
    ] {
        let k = KeplerianElements::from_state(r, v, MU);
        assert!(k.inclination.radians().is_nan(), "i: {label}");
        assert!(k.raan.radians().is_nan(), "raan: {label}");
        assert!(k.arg_periapsis.radians().is_nan(), "argp: {label}");
        assert!(k.true_anomaly.radians().is_nan(), "nu: {label}");
        // All three land on exactly 1.0; the bound is for the cancellation in the eccentricity
        // vector, (v² − μ/r) r − (r·v) v, which is exact only because these v are exact multiples
        // of these r.
        assert!((k.eccentricity - 1.0).abs() < 1e-15, "e: {label}");
        assert!((k.semi_major_axis - a).abs() < 1e-15 * a, "a: {label}");
    }

    // The other way the plane goes: h overflows, and `n_mag ≤ EQUATORIAL·h_mag` reads ∞ ≤ ∞ as an
    // equatorial orbit. r and v are each finite here — it is only their cross product that is not.
    let k =
        KeplerianElements::from_state(Vec3::new(0.0, 1e200, 0.0), Vec3::new(0.0, 0.0, 1e150), MU);
    assert!(k.inclination.radians().is_nan(), "i: overflowing h");
    assert!(k.raan.radians().is_nan(), "raan: overflowing h");
    assert!(k.arg_periapsis.radians().is_nan(), "argp: overflowing h");
    assert!(k.true_anomaly.radians().is_nan(), "nu: overflowing h");
}
