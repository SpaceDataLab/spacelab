//! The analytic Sun, checked against textbook astronomy — the J2000 solar position is a published
//! fact (right ascension ≈ 18ʰ45ᵐ, declination ≈ −23.0°, range ≈ 0.983 AU near perihelion), not a
//! number copied from another program.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]

use spacelab_propagation::ephemeris::{ASTRONOMICAL_UNIT, AnalyticMoon, AnalyticSun, Ephemeris};
use spacelab_testkit::nan_max;

const DAY: f64 = 86_400.0;

/// Right ascension (deg, 0..360), declination (deg), and range (AU) of a geocentric position.
fn ra_dec_range(p: [f64; 3]) -> (f64, f64, f64) {
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let ra = p[1].atan2(p[0]).to_degrees().rem_euclid(360.0);
    let dec = (p[2] / r).asin().to_degrees();
    (ra, dec, r / ASTRONOMICAL_UNIT)
}

#[test]
fn the_sun_at_j2000_matches_the_published_position() {
    // 2000-01-01 12:00 TT: the Sun sits near RA 18ʰ45ᵐ (281.2°), Dec −23.0°, a few days before
    // perihelion so the range is ≈ 0.983 AU.
    let sun = AnalyticSun {
        epoch_j2000_seconds: 0.0,
    };
    let (ra, dec, range) = ra_dec_range(sun.position(0.0));
    assert!((ra - 281.2).abs() < 0.3, "RA {ra:.3}° not near 281.2°");
    assert!((dec + 23.04).abs() < 0.1, "Dec {dec:.3}° not near −23.04°");
    assert!(
        (range - 0.9833).abs() < 0.001,
        "range {range:.5} AU not near 0.9833"
    );
}

#[test]
fn the_solar_range_stays_near_one_au_all_year() {
    // Earth's orbit: perihelion ≈ 0.983 AU, aphelion ≈ 1.017 AU, and never outside.
    let sun = AnalyticSun {
        epoch_j2000_seconds: 0.0,
    };
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for day in 0..366 {
        let (_, _, range) = ra_dec_range(sun.position(f64::from(day) * DAY));
        min = min.min(range);
        max = nan_max(max, range);
    }
    assert!(
        (0.982..0.985).contains(&min),
        "perihelion range {min:.5} AU off"
    );
    assert!(
        (1.015..1.018).contains(&max),
        "aphelion range {max:.5} AU off"
    );
}

#[test]
fn the_solar_declination_reaches_the_obliquity_at_the_solstices() {
    // Over a year the declination swings between ±ε (≈ 23.44°), reached at the solstices.
    let sun = AnalyticSun {
        epoch_j2000_seconds: 0.0,
    };
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for day in 0..366 {
        let (_, dec, _) = ra_dec_range(sun.position(f64::from(day) * DAY));
        min = min.min(dec);
        max = nan_max(max, dec);
    }
    assert!(
        (max - 23.44).abs() < 0.1,
        "peak declination {max:.3}° not the obliquity"
    );
    assert!(
        (min + 23.44).abs() < 0.1,
        "trough declination {min:.3}° not −obliquity"
    );
}

#[test]
fn the_moon_at_j2000_matches_the_jpl_horizons_vector() {
    // JPL Horizons, geocentric ICRF frame, epoch 2451545.0 TDB (= J2000): the Moon was near apogee.
    let reference: [f64; 3] = [-291_608.38e3, -266_716.83e3, -76_102.49e3];
    let ref_dist =
        (reference[0] * reference[0] + reference[1] * reference[1] + reference[2] * reference[2])
            .sqrt();

    let p = AnalyticMoon {
        epoch_j2000_seconds: 0.0,
    }
    .position(0.0);
    let err = ((p[0] - reference[0]).powi(2)
        + (p[1] - reference[1]).powi(2)
        + (p[2] - reference[2]).powi(2))
    .sqrt();
    let p_dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let dot = p[0] * reference[0] + p[1] * reference[1] + p[2] * reference[2];
    let angular = (dot / (p_dist * ref_dist))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    println!(
        "Moon J2000: |error| {:.0} km ({:.3}% of range), angular sep {:.4}°, range {:.0} vs {:.0} km",
        err / 1e3,
        100.0 * err / ref_dist,
        angular,
        p_dist / 1e3,
        ref_dist / 1e3
    );
    // Montenbruck–Gill low precision: a few arcminutes in angle, a few hundred km in range.
    assert!(
        angular < 0.15,
        "Moon J2000 angular error {angular:.4}° exceeds the ~arcminute series accuracy"
    );
    assert!(
        (p_dist - ref_dist).abs() < 3_000e3,
        "Moon J2000 range off by {:.0} km",
        (p_dist - ref_dist) / 1e3
    );
}

#[test]
fn the_lunar_distance_stays_within_the_perigee_apogee_band() {
    // The Moon's geocentric distance never leaves 356 500 – 406 700 km.
    let moon = AnalyticMoon {
        epoch_j2000_seconds: 0.0,
    };
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for day in 0..40 {
        let p = moon.position(f64::from(day) * DAY);
        let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() / 1e3;
        min = min.min(d);
        max = max.max(d);
    }
    assert!(
        (350_000.0..362_000.0).contains(&min),
        "perigee distance {min:.0} km out of band"
    );
    assert!(
        (400_000.0..412_000.0).contains(&max),
        "apogee distance {max:.0} km out of band"
    );
}
