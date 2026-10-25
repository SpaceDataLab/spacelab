//! Vermeille's closed form against ERFA's Fukushima iteration.
//!
//! Two *different algorithms*, not two transcriptions of one. Agreement here is
//! evidence that both solve the quartic they claim to; agreement between `pyerfa`
//! and `sofars` would only have been evidence that both descend from SOFA.

// x, y, z and lat, lon, h are the names of the things themselves.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]

use spacelab_core::{Angle, Length, Vec3};
use spacelab_earth::{
    EQUATORIAL_RADIUS, Geodetic, GeodeticError, POLAR_RADIUS, geodetic_to_itrf, itrf_to_geodetic,
    itrf_to_geodetic_into,
};
use spacelab_frames::{Itrf, Position};

const FIXTURES: &str = include_str!("fixtures/gc2gd_pyerfa.csv");

/// Latitude agreement, radians.
///
/// The two algorithms differ only in how they round. Measured worst case over
/// 4000 random points from LEO to GEO is `2.6e-11` rad, which is 0.17 mm on the
/// surface. The bound below is four times that, and still far below any physical
/// meaning: the geoid departs from this ellipsoid by a hundred metres.
const TOL_LAT_RAD: f64 = 1e-10;

/// Height agreement, metres. Measured worst case `1.5e-8` m.
const TOL_H_M: f64 = 1e-6;

fn parse(line: &str) -> [f64; 6] {
    let v: Vec<f64> = line
        .split(',')
        .map(|t| t.trim().parse().expect("a float"))
        .collect();
    [v[0], v[1], v[2], v[3], v[4], v[5]]
}

#[test]
fn vermeille_agrees_with_fukushima() {
    let (mut worst_lat, mut worst_lon, mut worst_h) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut checked = 0;

    for line in FIXTURES.lines().filter(|l| !l.trim().is_empty()) {
        let [x, y, z, lat, lon, h] = parse(line);
        let got = itrf_to_geodetic(Position::new(Vec3::new(x, y, z))).expect("outside the lens");

        let dlat = (got.latitude.radians() - lat).abs();
        // Longitude wraps; compare on the circle.
        let dlon = {
            let d = (got.longitude.radians() - lon).rem_euclid(std::f64::consts::TAU);
            d.min(std::f64::consts::TAU - d)
        };
        let dh = (got.height.metres() - h).abs();

        worst_lat = worst_lat.max(dlat);
        worst_lon = worst_lon.max(dlon);
        worst_h = worst_h.max(dh);

        assert!(
            dlat < TOL_LAT_RAD,
            "lat differs by {dlat:.3e} rad at ({x}, {y}, {z})"
        );
        assert!(dlon < TOL_LAT_RAD, "lon differs by {dlon:.3e} rad");
        assert!(dh < TOL_H_M, "height differs by {dh:.3e} m");
        checked += 1;
    }

    assert!(checked >= 500);
    println!(
        "gc2gd: {checked} points; worst dlat {worst_lat:.2e} rad, dlon {worst_lon:.2e} rad, dh {worst_h:.2e} m"
    );
}

/// The round trip is the property the closed form must satisfy exactly, whatever
/// any oracle says. It is checked over the whole ellipsoid, poles included.
#[test]
fn geodetic_round_trips_through_cartesian() {
    let mut worst = 0.0_f64;
    for lat_deg in [-90.0, -89.999, -45.0, -0.001, 0.0, 23.5, 66.6, 89.999, 90.0] {
        for lon_deg in [-180.0, -179.9, -90.0, 0.0, 0.1, 90.0, 179.9] {
            for height in [-11_000.0, 0.0, 8_849.0, 550_000.0, 35_786_000.0] {
                let original = Geodetic {
                    latitude: Angle::from_degrees(lat_deg),
                    longitude: Angle::from_degrees(lon_deg),
                    height: Length::from_metres(height),
                };
                let back = itrf_to_geodetic(geodetic_to_itrf(original)).expect("well outside");

                let dlat = (back.latitude.radians() - original.latitude.radians()).abs();
                let dh = (back.height.metres() - original.height.metres()).abs();
                worst = worst.max(dh);

                assert!(dlat < 1e-12, "lat drifted {dlat:.3e} rad at {lat_deg}°");
                assert!(
                    dh < 1e-7,
                    "height drifted {dh:.3e} m at {lat_deg}°, {height} m"
                );
                // Longitude is undefined on the axis; skip it there.
                if lat_deg.abs() < 89.99 {
                    let dlon = (back.longitude.radians() - original.longitude.radians()).abs();
                    assert!(dlon < 1e-12 || (dlon - std::f64::consts::TAU).abs() < 1e-12);
                }
            }
        }
    }
    println!("round trip: worst height drift {worst:.2e} m");
}

#[test]
fn the_defining_constants_are_the_defining_constants() {
    // a and f are exact by decree; b is derived. WGS-84 quotes b = 6356752.314245.
    assert!((POLAR_RADIUS - 6_356_752.314_245_179).abs() < 1e-6);
    // Exact by decree, so compare exactly.
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(EQUATORIAL_RADIUS, 6_378_137.0);
    }
}

#[test]
fn the_poles_and_the_equator_land_where_they_should() {
    let north = itrf_to_geodetic(Position::new(Vec3::new(0.0, 0.0, POLAR_RADIUS))).unwrap();
    assert!((north.latitude.degrees() - 90.0).abs() < 1e-9);
    assert!(north.height.metres().abs() < 1e-6);

    let equator = itrf_to_geodetic(Position::new(Vec3::new(EQUATORIAL_RADIUS, 0.0, 0.0))).unwrap();
    assert!(equator.latitude.degrees().abs() < 1e-12);
    assert!(equator.longitude.degrees().abs() < 1e-12);
    assert!(equator.height.metres().abs() < 1e-9);
}

/// Geodetic latitude is not geocentric latitude. At 45° the two differ by 11.5
/// arcminutes — 21 km along the surface. Confusing them is a silent, plausible
/// twenty-kilometre error in every ground track.
#[test]
fn geodetic_latitude_is_not_geocentric_latitude() {
    let p = geodetic_to_itrf(Geodetic {
        latitude: Angle::from_degrees(45.0),
        longitude: Angle::from_degrees(0.0),
        height: Length::from_metres(0.0),
    });
    let geocentric = (p.vector.z / p.vector.norm()).asin().to_degrees();
    let difference_arcmin = (45.0 - geocentric) * 60.0;
    assert!(
        (difference_arcmin - 11.5).abs() < 0.2,
        "expected ~11.5', saw {difference_arcmin:.2}'"
    );
}

#[test]
fn the_geocentric_singularity_is_refused_not_returned_as_nonsense() {
    let centre = Position::new(Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(itrf_to_geodetic(centre), Err(GeodeticError::NearGeocentre));
}

/// The batch path gives every fixture position exactly what the scalar path gives it (S2).
#[test]
fn the_batch_path_agrees_with_the_scalar_path() {
    let positions: Vec<Position<Itrf>> = FIXTURES
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let [x, y, z, ..] = parse(l);
            Position::new(Vec3::new(x, y, z))
        })
        .collect();

    let mut out = vec![Err(GeodeticError::NearGeocentre); positions.len()];
    itrf_to_geodetic_into(&positions, &mut out);

    for (p, batched) in positions.iter().zip(&out) {
        assert_eq!(*batched, itrf_to_geodetic(*p));
    }
}
