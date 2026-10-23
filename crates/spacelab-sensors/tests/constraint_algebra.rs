//! The constraint combinators, the default sign test, the rate bound and the error surface.
//!
//! `predicate.rs` verifies the concrete constraints (`Cone`, `ElevationMask`) and the `And`
//! combinator against closed-form geometry. This file covers the rest of `lib.rs`, which had none:
//! the `Or`/`Not` combinators (composition of access windows is where a wrong sign silently drops
//! passes), the trait's default `covered`, `OrbitBound::max_angular_rate`, and every `SensorError`
//! message. The combinator laws are algebraic identities — `min`/`max`/negation — so they must hold
//! at every geometry, not merely at ones we hand-pick.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]

use spacelab_core::Vec3;
use spacelab_sensors::{
    AccessGeometry, And, Cone, Constraint, ElevationMask, Not, Or, OrbitBound, SensorError,
};

const MU_EARTH: f64 = 3.986_004_418e14;
const EARTH_ROTATION_RATE: f64 = 7.292_115e-5;
const RE: f64 = 6_371_000.0;
const R_ORBIT: f64 = 7_000_000.0;

fn orbit() -> OrbitBound {
    OrbitBound {
        semi_major_axis: R_ORBIT,
        eccentricity: 0.1,
        mu: MU_EARTH,
    }
}

/// Nadir: the satellite is straight above the cell, so any sane sensor covers it.
fn nadir() -> AccessGeometry {
    AccessGeometry {
        satellite: Vec3::new(0.0, 0.0, R_ORBIT),
        cell: Vec3::new(0.0, 0.0, RE),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    }
}

/// A cell 90° around the limb: the satellite is below its local horizon, so nothing covers it.
fn limb() -> AccessGeometry {
    AccessGeometry {
        satellite: Vec3::new(0.0, 0.0, R_ORBIT),
        cell: Vec3::new(RE, 0.0, 0.0),
        zenith: Vec3::new(1.0, 0.0, 0.0),
    }
}

#[test]
fn the_combinators_are_min_max_and_negation_at_every_geometry() {
    let cone = Cone::new(30.0_f64.to_radians()).unwrap();
    let mask = ElevationMask::new(5.0_f64.to_radians()).unwrap();
    let orbit = orbit();

    for geom in [nadir(), limb()] {
        let (g1, g2) = (cone.g(&geom), mask.g(&geom));

        // g: And = min, Or = max, Not = negation — exactly.
        assert_eq!(And(cone, mask).g(&geom), g1.min(g2));
        assert_eq!(Or(cone, mask).g(&geom), g1.max(g2));
        assert_eq!(Not(cone).g(&geom), -g1);

        // covered: And = both, Or = either, Not keeps the boundary on the `>= 0` side.
        assert_eq!(
            And(cone, mask).covered(&geom),
            cone.covered(&geom) && mask.covered(&geom),
        );
        assert_eq!(
            Or(cone, mask).covered(&geom),
            cone.covered(&geom) || mask.covered(&geom),
        );
        assert_eq!(Not(cone).covered(&geom), -cone.g(&geom) >= 0.0);

        // max_rate: And/Or take the faster component; Not keeps its inner bound.
        let (r1, r2) = (
            cone.max_rate(&orbit, EARTH_ROTATION_RATE),
            mask.max_rate(&orbit, EARTH_ROTATION_RATE),
        );
        assert_eq!(
            And(cone, mask).max_rate(&orbit, EARTH_ROTATION_RATE),
            r1.max(r2)
        );
        assert_eq!(
            Or(cone, mask).max_rate(&orbit, EARTH_ROTATION_RATE),
            r1.max(r2)
        );
        assert_eq!(Not(cone).max_rate(&orbit, EARTH_ROTATION_RATE), r1);
    }

    // The two geometries really do straddle the boundary, or the tests above prove nothing.
    assert!(cone.covered(&nadir()) && mask.covered(&nadir()));
    assert!(!cone.covered(&limb()) && !mask.covered(&limb()));
}

/// A constraint that does *not* override `covered`, to exercise the trait default `g(..) >= 0`.
#[derive(Clone, Copy)]
struct UpperHalfSpace;

impl Constraint for UpperHalfSpace {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        geom.satellite.z
    }
    fn max_rate(&self, _orbit: &OrbitBound, _frame_rate: f64) -> f64 {
        0.0
    }
}

#[test]
fn the_default_covered_is_the_sign_of_g() {
    let above = AccessGeometry {
        satellite: Vec3::new(0.0, 0.0, 1.0),
        cell: Vec3::new(0.0, 0.0, 0.0),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    };
    let below = AccessGeometry {
        satellite: Vec3::new(0.0, 0.0, -1.0),
        ..above
    };
    assert!(UpperHalfSpace.covered(&above));
    assert!(!UpperHalfSpace.covered(&below));
    // And it agrees with `g >= 0` exactly, including the boundary.
    let on = AccessGeometry {
        satellite: Vec3::new(0.0, 0.0, 0.0),
        ..above
    };
    assert_eq!(UpperHalfSpace.covered(&on), UpperHalfSpace.g(&on) >= 0.0);
}

#[test]
fn orbit_bound_rates_match_the_closed_form() {
    let o = orbit();
    let rp = o.semi_major_axis * (1.0 - o.eccentricity);
    let ra = o.semi_major_axis * (1.0 + o.eccentricity);
    let vp = (o.mu * (1.0 + o.eccentricity) / rp).sqrt();

    assert!((o.perigee_radius() - rp).abs() <= 1e-6);
    assert!((o.apogee_radius() - ra).abs() <= 1e-6);
    assert!((o.perigee_speed() - vp).abs() <= 1e-9 * vp);
    assert!((o.max_angular_rate() - vp / rp).abs() <= 1e-12 * (vp / rp));
    assert!(
        (o.max_relative_speed(EARTH_ROTATION_RATE) - (vp + EARTH_ROTATION_RATE * ra)).abs()
            <= 1e-9 * vp
    );
    // The rotating-frame bound must exceed the inertial one — it adds the frame drag.
    assert!(o.max_relative_speed(EARTH_ROTATION_RATE) > o.perigee_speed());
}

#[test]
fn every_sensor_error_prints_a_distinct_non_empty_message() {
    let all = [
        SensorError::ElevationOutOfRange { eps_min: 2.0 },
        SensorError::HalfAngleOutOfRange { half_angle: 3.0 },
        SensorError::IncidenceBandOutOfRange {
            near: 1.0,
            far: 0.5,
        },
        SensorError::LinkNeverCloses,
        SensorError::RaggedGrid {
            shortest: 3,
            longest: 5,
        },
        SensorError::OutputLengthMismatch {
            cells: 10,
            buffer: 7,
        },
    ];
    let messages: Vec<String> = all.iter().map(ToString::to_string).collect();
    assert!(messages.iter().all(|m| !m.is_empty()));
    let distinct: std::collections::HashSet<&String> = messages.iter().collect();
    assert_eq!(
        distinct.len(),
        all.len(),
        "duplicate error message: {messages:?}"
    );
}
