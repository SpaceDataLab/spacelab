//! What the scan catalogue and the direct sensor model do at the edges of their domain.
//!
//! The pyorbital comparison in `pyorbital_geoloc.rs` says the two agree where both are defined. It
//! cannot say anything about where only one of them is, and that is exactly where this port departs
//! from its source on purpose: pyorbital's ray-ellipsoid intersection takes an unguarded square
//! root, its `geodetic_lat` loops `while True`, and neither refuses an index. Each of those is a
//! `NaN`, a hang or a silent wrong answer waiting for an input a caller supplies.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::Vec3;
use spacelab_sensors::geoloc::{
    Attitude, WGS84, geodetic_latitude, geolocate, intersect_ellipsoid, line_of_sight,
    nadir_direction, rotate_about, subpoint,
};
use spacelab_sensors::scan::{Instrument, ScanSample};

const EVERY_INSTRUMENT: [Instrument; 11] = [
    Instrument::Avhrr,
    Instrument::AvhrrGac,
    Instrument::Viirs,
    Instrument::AmsuA,
    Instrument::Mhs,
    Instrument::Hirs4,
    Instrument::Atms,
    Instrument::Mwhs2,
    Instrument::Olci,
    Instrument::Ascat,
    Instrument::SlstrNadir,
];

/// A NOAA-18 state from the oracle fixture — a real LEO position, not a contrived one.
fn leo_state() -> (Vec3, Vec3) {
    (
        Vec3::new(
            -1_800.984_915_857_122,
            -752.959_121_483_218_8,
            6_950.028_708_742_64,
        ),
        Vec3::new(
            3.722_077_566_333_073,
            6.223_832_500_444_195,
            1.626_285_480_324_818,
        ),
    )
}

#[test]
fn an_index_off_the_detector_is_refused_rather_than_wrapped() {
    for instrument in EVERY_INSTRUMENT {
        let (pixels, lines) = (instrument.pixels_per_line(), instrument.detector_lines());
        assert!(
            instrument.sample(0, lines - 1, pixels - 1).is_some(),
            "{instrument:?} refused its own last pixel"
        );
        assert!(
            instrument.sample(0, lines - 1, pixels).is_none(),
            "{instrument:?} answered for pixel {pixels}, one past its {pixels}-sample line"
        );
        assert!(
            instrument.sample(0, lines, 0).is_none(),
            "{instrument:?} answered for detector row {lines}, one past its {lines} rows"
        );
    }
}

#[test]
fn every_cross_track_swath_is_symmetric_and_the_push_brooms_are_not() {
    for instrument in EVERY_INSTRUMENT {
        let last = instrument.pixels_per_line() - 1;
        let first_roll = instrument.sample(0, 0, 0).unwrap().roll;
        let last_roll = instrument.sample(0, 0, last).unwrap().roll;
        match instrument {
            // The two Sentinel-3 push-brooms are tilted to keep sun glint out of the swath, so
            // theirs is the one asymmetry in the catalogue and it is a design fact, not a typo.
            Instrument::Olci | Instrument::SlstrNadir => assert!(
                (first_roll + last_roll).abs() > 0.2,
                "{instrument:?} runs {first_roll:.4} to {last_roll:.4} rad, which is symmetric — \
                 the ESA coverage page gives +46.5° to −22.1°, and a symmetric swath means the \
                 asymmetric ramp was lost"
            ),
            _ => assert!(
                (first_roll + last_roll).abs() < 1e-15,
                "{instrument:?} runs {first_roll:.17e} to {last_roll:.17e} rad, which is not \
                 symmetric about the boresight"
            ),
        }
    }
}

#[test]
fn ascat_leaves_the_ground_track_uncovered_because_a_scatterometer_has_two_fans() {
    // The two 21-sample fans span 25°–53° either side, so the innermost samples of each are the
    // 25° ones and nothing between them is observed. A single-fan port would close this gap and
    // look like better coverage.
    let inner_left = Instrument::Ascat.sample(0, 0, 20).unwrap().roll;
    let inner_right = Instrument::Ascat.sample(0, 0, 21).unwrap().roll;
    assert!(
        inner_left > 0.0 && inner_right < 0.0,
        "the fans should straddle nadir: {inner_left:.4} and {inner_right:.4} rad"
    );
    assert!(
        (inner_left.to_degrees() - 25.0).abs() < 1e-12
            && (inner_right.to_degrees() + 25.0).abs() < 1e-12,
        "the inner edges should be ±25°, not {:.4}° and {:.4}°",
        inner_left.to_degrees(),
        inner_right.to_degrees()
    );
}

#[test]
fn a_look_past_the_limb_is_refused_rather_than_returning_a_nan_position() {
    let (position, velocity) = leo_state();
    // 80° off the boresight, well past the ~63° horizon at this altitude.
    let over_the_horizon = ScanSample {
        roll: 80.0_f64.to_radians(),
        pitch: 0.0,
        time_offset: 0.0,
    };
    assert!(
        geolocate(
            position,
            velocity,
            over_the_horizon,
            Attitude::default(),
            WGS84
        )
        .is_none(),
        "a ray that misses the ellipsoid must be refused; the unguarded square root returns NaN, \
         which averages into a swath centre without complaint"
    );
    // And the widest real instrument still hits, so the guard is not simply refusing everything.
    let swath_edge = Instrument::Viirs.sample(0, 0, 0).unwrap();
    assert!(
        geolocate(position, velocity, swath_edge, Attitude::default(), WGS84).is_some(),
        "VIIRS's 56.28° swath edge should still reach the ground"
    );
}

#[test]
fn the_degenerate_arguments_return_a_defined_value_rather_than_a_nan() {
    let centre = Vec3::new(0.0, 0.0, 0.0);
    assert!(
        geodetic_latitude(centre, WGS84).is_finite(),
        "the latitude at the geocentre is undefined; returning NaN propagates it into every \
         line of sight built from it"
    );
    assert!(subpoint(centre, WGS84).norm().is_finite());
    assert!(nadir_direction(centre, WGS84).norm().is_finite());

    let v = Vec3::new(1.0, 2.0, 3.0);
    let unmoved = rotate_about(v, Vec3::new(0.0, 0.0, 0.0), 1.0);
    assert!(
        (unmoved - v).norm() == 0.0,
        "a rotation about a zero axis has no defined plane, so the vector is the only answer"
    );

    let (position, _) = leo_state();
    let stationary = line_of_sight(
        position,
        Vec3::new(0.0, 0.0, 0.0),
        ScanSample {
            roll: 0.3,
            pitch: 0.1,
            time_offset: 0.0,
        },
        Attitude::default(),
        WGS84,
    );
    assert!(
        (stationary.norm() - 1.0).abs() < 1e-12,
        "with no velocity there is no along-track axis and the scan angles cannot be applied, but \
         the boresight still exists and is still a unit vector: got norm {}",
        stationary.norm()
    );

    assert!(
        intersect_ellipsoid(position, Vec3::new(0.0, 0.0, 0.0), WGS84).is_none(),
        "a ray with no direction meets nothing"
    );
    assert!(
        intersect_ellipsoid(position, position, WGS84).is_none(),
        "a ray pointing straight up never comes back down, and the near root is behind it"
    );
}

#[test]
fn the_boresight_and_the_swath_edge_bracket_the_slant_range() {
    // A weak claim, deliberately: it holds for every cross-track instrument regardless of its
    // angles, so it catches a sign flip in the roll rotation that a per-instrument number would
    // have to be tuned to notice.
    let (position, velocity) = leo_state();
    let nadir = ScanSample {
        roll: 0.0,
        pitch: 0.0,
        time_offset: 0.0,
    };
    let straight_down = geolocate(position, velocity, nadir, Attitude::default(), WGS84)
        .expect("the boresight reaches the ground");
    let nadir_range = (straight_down - position).norm();

    for instrument in EVERY_INSTRUMENT {
        let edge = instrument.sample(0, 0, 0).unwrap();
        let hit = geolocate(position, velocity, edge, Attitude::default(), WGS84)
            .unwrap_or_else(|| panic!("{instrument:?}'s first sample missed the ellipsoid"));
        let range = (hit - position).norm();
        assert!(
            range > nadir_range,
            "{instrument:?} looks {:.2}° off nadir, so its slant range {range:.1} km cannot be \
             shorter than the nadir range {nadir_range:.1} km — a roll applied about the wrong \
             axis, or with the wrong sign convention, is what makes it so",
            edge.roll.to_degrees().abs(),
        );
    }
}
