//! Sensor swath and ground-footprint geometry — where on the Earth a satellite looks.
//!
//! For a spherical Earth of radius `re` and a satellite at `altitude`, an off-nadir look angle `η`
//! (the angle at the satellite between nadir and the line of sight) reaches a ground point at
//! Earth-central angle `λ` from the sub-satellite point, seen there at elevation `ε` above the local
//! horizon. The satellite–centre–target triangle gives `λ + η + (90° + ε) = 180°` and, by the law of
//! sines, `sin η = (re/(re+h))·cos ε`; a line of sight steeper than the limb (`sin η = re/(re+h)`)
//! misses the Earth entirely. [`swath_geometry`] solves that triangle; [`ground_point_at`] walks a
//! great-circle arc to trace footprint edges and ground tracks.
//!
//! Gate: `derivations/swath_sphere.mac` shows that the coded slant range is the near root of the
//! line-of-sight/sphere quadratic `ρ² − 2 r_s cos η ρ + r_s² − r_e² = 0`, that the central angle and
//! elevation are those of the point it reaches, and that the great-circle walk is the rotated unit
//! vector; `derivations/swath_sphere.gp` computes both from the geometry at 60 digits. The module
//! tests `the_swath_is_the_root_of_the_line_of_sight_quadratic` (G2) and
//! `the_walk_arrives_where_the_rotated_unit_vector_points` (G1) hold the shipped code to those values.

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
    #![allow(
        clippy::unwrap_used,
        clippy::excessive_precision,
        clippy::unreadable_literal
    )]
    use super::*;

    const RE: f64 = 6_378_137.0;
    const PI: f64 = core::f64::consts::PI;

    /// `(altitude, look angle, central angle, elevation, slant range)` from
    /// `derivations/swath_sphere.gp`: the near root of the line-of-sight quadratic and the point it
    /// reaches, at 60 digits, for dyadic look angles.
    const SWATH_ROOTS: [(f64, f64, f64, f64, f64); 5] = [
        (
            550000.0,
            0.0625,
            5.397471159274181866665030596879e-3,
            1.502898855635622437364656661043e0,
            5.511690576215663590457907804894e5,
        ),
        (
            550000.0,
            0.4375,
            4.073255548317759323776038359836e-2,
            1.092563771311719025993561308041e0,
            6.130293604360249118946488549019e5,
        ),
        (
            700000.0,
            0.25,
            2.812847151475837662945910238311e-2,
            1.292667855280138242601862589257e0,
            7.250635286378530726355676955841e5,
        ),
        (
            700000.0,
            1.09375,
            3.086325190448010536417081209006e-1,
            1.684138077500955655896135707391e-1,
            2.180882341222800635418683068366e6,
        ),
        (
            35786000.0,
            0.125,
            8.437720112789820960804671737920e-1,
            6.020243155159145231508545178477e-1,
            3.822314250037663553443508276015e7,
        ),
    ];

    /// (G2) The swath geometry is the near root of `|S + ρu|² = r_e²` and the ground point it reaches,
    /// solved from the geometry at 60 digits by `derivations/swath_sphere.gp`.
    ///
    /// The bounds are each formula's own conditioning. `asin k` amplifies an ulp of `k` by
    /// `1/√(1 − k²)`, which is the elevation's floor and the central angle's, the latter also carrying
    /// the cancellation of `asin k − η`. The slant range is a law-of-cosines difference, so its floor is
    /// an ulp of `(r_e + r_s)²` divided by `2ρ`.
    #[test]
    fn the_swath_is_the_root_of_the_line_of_sight_quadratic() {
        let mut worst = 0.0_f64;
        for (h, eta, lambda, elevation, slant) in SWATH_ROOTS {
            let g = swath_geometry(RE, h, eta).unwrap();
            let rs = RE + h;
            let k = rs / RE * libm::sin(eta);
            let amp = 1.0 / libm::sqrt(1.0 - k * k);
            let tol_angle = 8.0 * f64::EPSILON * (1.0 + amp) * (1.0 + eta);
            let tol_slant = 8.0 * f64::EPSILON * (RE + rs) * (RE + rs) / (2.0 * slant);
            for (got, want, tol, what) in [
                (g.earth_central_angle, lambda, tol_angle, "central angle"),
                (g.elevation, elevation, tol_angle, "elevation"),
                (g.ground_range, RE * lambda, RE * tol_angle, "ground range"),
                (g.slant_range, slant, tol_slant, "slant range"),
            ] {
                let d = (got - want).abs();
                assert!(
                    d <= tol,
                    "h {h} η {eta}: {what} off by {d:.3e} (allowed {tol:.3e})"
                );
                worst = worst.max(d / tol);
            }
        }
        println!("swath vs 60 digits: worst {worst:.3} of the conditioning bound");
    }

    /// `(lat, lon, central angle, azimuth, lat2, lon2)` from `derivations/swath_sphere.gp`: the start
    /// rotated along the great circle, read back as latitude and longitude, at 60 digits.
    const WALKS: [(f64, f64, f64, f64, f64, f64); 3] = [
        (
            0.59375,
            -1.1875,
            0.40625,
            0.6875,
            8.742565506941572952042285405535e-1,
            -7.859122756041119413331303405993e-1,
        ),
        (
            -0.34375,
            2.90625,
            1.3125,
            -1.90625,
            -3.960133043806902617308392593109e-1,
            1.480360468182736245533905732415e0,
        ),
        (
            1.40625,
            0.125,
            0.015625,
            3.0,
            1.390767985419831560351296635965e0,
            1.373142938104364756326684924791e-1,
        ),
    ];

    /// (G1) `ground_point_at` is the rotation of the start point through the central angle along the
    /// initial bearing, `X = cos d P + sin d (cos az N + sin az E)` — the identity
    /// `derivations/swath_sphere.mac` proves for the code's two formulas, held here at 60-digit values.
    /// The bound is a few ulps of `π`, scaled by `1/cos lat2` where `asin` reads a sine near the pole.
    #[test]
    fn the_walk_arrives_where_the_rotated_unit_vector_points() {
        for (lat, lon, d, az, lat2, lon2) in WALKS {
            let (got_lat, got_lon) = ground_point_at(lat, lon, d, az);
            let tol = 16.0 * f64::EPSILON * PI / libm::cos(lat2);
            assert!((got_lat - lat2).abs() <= tol, "lat2 {got_lat} vs {lat2}");
            assert!((got_lon - lon2).abs() <= tol, "lon2 {got_lon} vs {lon2}");
        }
    }

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
