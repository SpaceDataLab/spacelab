//! WGS-84 geodesy: the forward/inverse round-trip, the geocentric singularity, and the batch entry.
//!
//! `itrf_to_geodetic` is exercised end-to-end by the sgp4 ground-track test, but `spacelab-earth`
//! had no test of its own: the geodetic→ITRF→geodetic round-trip (the real correctness statement),
//! the `NearGeocentre` error and its message, and the batch converter's misuse guard.

#![allow(clippy::unwrap_used)]

use spacelab_core::{Angle, Length, Vec3};
use spacelab_earth::{
    Geodetic, GeodeticError, geodetic_to_itrf, itrf_to_geodetic, itrf_to_geodetic_into,
};
use spacelab_frames::{Itrf, Position};

fn geodetic(lat_deg: f64, lon_deg: f64, height_m: f64) -> Geodetic {
    Geodetic {
        latitude: Angle::from_degrees(lat_deg),
        longitude: Angle::from_degrees(lon_deg),
        height: Length::from_metres(height_m),
    }
}

#[test]
fn geodetic_and_itrf_round_trip_at_realistic_points() {
    // Vermeille's inverse is a closed form, so the round-trip must return to the ellipsoid point it
    // started from — at cities, near the pole, and near the equator/antimeridian.
    for (lat, lon, h) in [
        (48.8566, 2.3522, 35.0),    // Paris
        (-33.8688, 151.2093, 58.0), // Sydney
        (89.9, 12.0, 100.0),        // near the North Pole
        (0.0, 179.0, 0.0),          // equator, near the antimeridian
        (0.0, 0.0, 20_200_000.0),   // a MEO-altitude sub-point
    ] {
        let back = itrf_to_geodetic(geodetic_to_itrf(geodetic(lat, lon, h))).unwrap();
        assert!(
            (back.latitude.degrees() - lat).abs() < 1e-9,
            "lat at ({lat}, {lon})"
        );
        assert!(
            (back.longitude.degrees() - lon).abs() < 1e-9,
            "lon at ({lat}, {lon})"
        );
        assert!(
            (back.height.metres() - h).abs() < 1e-6,
            "height at ({lat}, {lon})"
        );
    }
}

#[test]
fn a_point_in_the_geocentric_singularity_has_no_geodetic_coordinates() {
    let centre = Position::<Itrf>::new(Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(itrf_to_geodetic(centre), Err(GeodeticError::NearGeocentre));
    assert_ne!(GeodeticError::NearGeocentre.to_string(), "");
}

#[test]
fn the_batch_converter_maps_each_slot_independently() {
    let paris = geodetic_to_itrf(geodetic(48.8566, 2.3522, 35.0));
    let centre = Position::<Itrf>::new(Vec3::new(0.0, 0.0, 0.0));
    let positions = [paris, centre];
    let mut out = [Err(GeodeticError::NearGeocentre); 2];
    itrf_to_geodetic_into(&positions, &mut out);
    assert!(out[0].is_ok());
    assert_eq!(out[1], Err(GeodeticError::NearGeocentre)); // a decayed slot does not spoil the batch
}

#[test]
#[should_panic(expected = "output slice holds")]
fn the_batch_converter_refuses_a_short_output() {
    let p = geodetic_to_itrf(geodetic(0.0, 0.0, 0.0));
    let positions = [p, p];
    let mut out: [Result<Geodetic, GeodeticError>; 1] = [Err(GeodeticError::NearGeocentre)];
    itrf_to_geodetic_into(&positions, &mut out);
}
