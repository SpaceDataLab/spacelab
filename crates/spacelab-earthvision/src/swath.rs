//! Sensor swath and ground-footprint geometry — where on the Earth a satellite looks.
//!
//! For a spherical Earth of radius `re` and a satellite at `altitude`, an off-nadir look angle `η`
//! (the angle at the satellite between nadir and the line of sight) reaches a ground point at
//! Earth-central angle `λ` from the sub-satellite point, seen there at elevation `ε` above the local
//! horizon. The satellite–centre–target triangle gives `λ + η + (90° + ε) = 180°` and, by the law of
//! sines, `sin η = (re/(re+h))·cos ε`; a line of sight steeper than the limb (`sin η = re/(re+h)`)
//! misses the Earth entirely. [`swath_geometry`] solves that triangle; [`ground_point_at`] walks a
//! great-circle arc to trace footprint edges and ground tracks.

// The single letters (h, λ, η, ε) are the standard swath-geometry notation.
#![allow(clippy::many_single_char_names)]

use core::f64::consts::FRAC_PI_2;

/// The ground geometry a sensor line of sight reaches at a given off-nadir look angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwathGeometry {
    /// Earth-central angle `λ` from the sub-satellite point to the ground point, radians.
    pub earth_central_angle: f64,
    /// Elevation `ε` of the satellite above the ground point's local horizon, radians.
    pub elevation: f64,
    /// Ground range (surface arc `re·λ`) from the sub-satellite point, metres.
    pub ground_range: f64,
    /// Slant range from the satellite to the ground point, metres.
    pub slant_range: f64,
}

/// The largest off-nadir look angle whose line of sight still meets the Earth — the limb angle
/// `asin(re/(re+altitude))`. Beyond it the sensor sees space.
#[must_use]
pub fn horizon_look_angle(re: f64, altitude: f64) -> f64 {
    libm::asin(re / (re + altitude))
}

/// The ground geometry for an off-nadir `look_angle` (radians) from a satellite at `altitude` over a
/// spherical Earth of radius `re`. `None` when the look angle exceeds the limb ([`horizon_look_angle`]).
#[must_use]
pub fn swath_geometry(re: f64, altitude: f64, look_angle: f64) -> Option<SwathGeometry> {
    let rs = re + altitude;
    let arg = (rs / re) * libm::sin(look_angle);
    if arg > 1.0 {
        return None; // the line of sight misses the Earth
    }
    let lam_plus_eta = libm::asin(arg); // λ + η
    let lambda = lam_plus_eta - look_angle;
    let elevation = FRAC_PI_2 - lam_plus_eta; // ε = 90° − (λ + η)
    let slant_range = libm::sqrt(re * re + rs * rs - 2.0 * re * rs * libm::cos(lambda));
    Some(SwathGeometry {
        earth_central_angle: lambda,
        elevation,
        ground_range: re * lambda,
        slant_range,
    })
}

/// The geodetic point reached by walking a great-circle arc of `central_angle` (radians) from
/// `(lat, lon)` (radians) on the initial `azimuth` (radians, clockwise from north). Returns
/// `(lat, lon)` in radians. Sweep the azimuth to trace a footprint circle; step along a heading to
/// trace a ground track.
#[must_use]
pub fn ground_point_at(lat: f64, lon: f64, central_angle: f64, azimuth: f64) -> (f64, f64) {
    let (sin_d, cos_d) = libm::sincos(central_angle);
    let (sin_lat, cos_lat) = libm::sincos(lat);
    let lat2 =
        libm::asin((sin_lat * cos_d + cos_lat * sin_d * libm::cos(azimuth)).clamp(-1.0, 1.0));
    let lon2 = lon
        + libm::atan2(
            libm::sin(azimuth) * sin_d * cos_lat,
            cos_d - sin_lat * libm::sin(lat2),
        );
    (lat2, lon2)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    const RE: f64 = 6_378_137.0;
    const PI: f64 = core::f64::consts::PI;

    #[test]
    fn nadir_look_is_directly_below_at_full_elevation() {
        let h = 700_000.0;
        let g = swath_geometry(RE, h, 0.0).unwrap();
        assert!(g.earth_central_angle.abs() < 1e-12);
        assert!((g.elevation - FRAC_PI_2).abs() < 1e-12);
        assert!(g.ground_range.abs() < 1e-6);
        assert!((g.slant_range - h).abs() < 1e-6); // straight down = altitude
    }

    #[test]
    fn the_limb_look_grazes_the_horizon() {
        let h = 700_000.0;
        let eta_max = horizon_look_angle(RE, h);
        let g = swath_geometry(RE, h, eta_max).unwrap();
        assert!(g.elevation.abs() < 1e-9, "elevation {}", g.elevation);
        // Just past the limb there is no ground intersection.
        assert!(swath_geometry(RE, h, eta_max + 1e-6).is_none());
    }

    #[test]
    fn swath_widens_monotonically_with_look_angle() {
        let h = 700_000.0;
        let a = swath_geometry(RE, h, 10.0_f64.to_radians()).unwrap();
        let b = swath_geometry(RE, h, 30.0_f64.to_radians()).unwrap();
        assert!(b.earth_central_angle > a.earth_central_angle);
        assert!(b.ground_range > a.ground_range);
        assert!(b.slant_range > a.slant_range);
        assert!(b.elevation < a.elevation); // steeper obliquity, lower elevation
    }

    #[test]
    fn the_swath_triangle_closes() {
        // λ + η + (90° + ε) = 180°.
        let (h, eta) = (550_000.0, 25.0_f64.to_radians());
        let g = swath_geometry(RE, h, eta).unwrap();
        let sum = g.earth_central_angle + eta + FRAC_PI_2 + g.elevation;
        assert!((sum - PI).abs() < 1e-12, "triangle sum {sum}");
    }

    #[test]
    fn ground_point_walks_the_expected_cardinal_directions() {
        let d = 0.02; // ~127 km central angle
        // Due north from the equator raises latitude by d, keeps longitude.
        let (lat_n, lon_n) = ground_point_at(0.0, 0.0, d, 0.0);
        assert!((lat_n - d).abs() < 1e-12 && lon_n.abs() < 1e-12);
        // Due east from the equator raises longitude by d, keeps latitude.
        let (lat_e, lon_e) = ground_point_at(0.0, 0.0, d, FRAC_PI_2);
        assert!(lat_e.abs() < 1e-12 && (lon_e - d).abs() < 1e-12);
    }

    #[test]
    fn the_walked_arc_inverts_to_the_distance_and_bearing_it_was_given() {
        // Off the equator and off the cardinal bearings. At lat = 0 every `sin_lat` product in the
        // direct formula is annihilated, which is exactly why the cardinal walk above cannot see a
        // wrong operator in any of them.
        for (lat, lon) in [(0.6, -1.2), (-0.35, 2.9), (1.4, 0.1)] {
            for d in [0.02, 0.4, 1.3] {
                for az in [0.7, 2.2, -1.9, 3.0] {
                    let (lat2, lon2) = ground_point_at(lat, lon, d, az);
                    // The inverse spherical geodesic — a different pair of formulas, not a
                    // rearrangement of the direct one.
                    let dlon = lon2 - lon;
                    let central = libm::acos(
                        (libm::sin(lat) * libm::sin(lat2)
                            + libm::cos(lat) * libm::cos(lat2) * libm::cos(dlon))
                        .clamp(-1.0, 1.0),
                    );
                    let bearing = libm::atan2(
                        libm::sin(dlon) * libm::cos(lat2),
                        libm::cos(lat) * libm::sin(lat2)
                            - libm::sin(lat) * libm::cos(lat2) * libm::cos(dlon),
                    );
                    assert!(
                        (central - d).abs() < 1e-12,
                        "({lat},{lon}) d={d} az={az}: central angle {central}"
                    );
                    assert!(
                        (bearing - az).abs() < 1e-12,
                        "({lat},{lon}) d={d} az={az}: bearing {bearing}"
                    );
                }
            }
        }
    }
}
