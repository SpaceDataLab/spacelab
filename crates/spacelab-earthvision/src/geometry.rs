//! Earth-observation viewing geometry — the sun/sensor angles atmospheric correction and BRDF need.
//!
//! Inputs are unit direction vectors in one consistent local frame: `to_sun` and `to_sensor` point
//! from the ground point toward the sun and the satellite, `zenith` is the local outward normal (up)
//! and `north` a horizontal reference. The caller builds these from an ephemeris; keeping the
//! geometry frame-agnostic keeps it dependency-light and exactly testable.

use spacelab_core::{Angle, Vec3};

const PI: f64 = core::f64::consts::PI;
const TAU: f64 = core::f64::consts::TAU;

/// The sun/sensor geometry at a ground point. All angles radians.
#[derive(Clone, Copy, Debug)]
pub struct ObservationGeometry {
    /// Solar zenith angle `θs` — angle of the sun from the local vertical.
    pub solar_zenith: Angle,
    /// View (sensor) zenith angle `θv`.
    pub view_zenith: Angle,
    /// Relative azimuth `φv − φs`, wrapped to `[−π, π)`.
    pub relative_azimuth: Angle,
    /// Scattering angle between the incident (sun → ground) and scattered (ground → sensor) rays:
    /// `0` is forward scatter, `π` is exact backscatter.
    pub scattering_angle: Angle,
    /// Relative optical air mass at the solar zenith (Kasten–Young).
    pub air_mass: f64,
}

/// Wrap an angle to `[−π, π)` — the half-open interval `fmod` produces, so the map is a bijection of
/// `ℝ/2πℤ` and `wrap_pi(π) = −π`.
fn wrap_pi(a: f64) -> f64 {
    let x = libm::fmod(a + PI, TAU);
    let x = if x < 0.0 { x + TAU } else { x };
    x - PI
}

/// Relative optical air mass at zenith `angle` (Kasten & Young 1989): `1 / (cos θ + 0.50572
/// (96.07995° − θ)^−1.6364)`. Finite even at the horizon, unlike the plane-parallel `1/cos θ`, and
/// accurate to a fraction of a percent through `θ = 90°`.
#[must_use]
pub fn air_mass(angle: Angle) -> f64 {
    let z = angle.radians();
    let z_deg = z * (180.0 / PI);
    1.0 / (libm::cos(z) + 0.50572 * libm::pow(96.079_95 - z_deg, -1.6364))
}

/// Compute the [`ObservationGeometry`] from the sun and sensor directions at a ground point.
#[must_use]
pub fn observation_geometry(
    to_sun: Vec3,
    to_sensor: Vec3,
    zenith: Vec3,
    north: Vec3,
) -> ObservationGeometry {
    let east = north.cross(zenith); // right-handed ENU: E = N × U
    let solar_zenith = libm::acos(to_sun.dot(zenith).clamp(-1.0, 1.0));
    let view_zenith = libm::acos(to_sensor.dot(zenith).clamp(-1.0, 1.0));
    // Azimuths measured from north toward east; the vertical component cancels in the ratio.
    let sun_az = libm::atan2(to_sun.dot(east), to_sun.dot(north));
    let view_az = libm::atan2(to_sensor.dot(east), to_sensor.dot(north));
    // The scattering angle is between the photon's incident propagation (−to_sun) and its scattered
    // propagation (to_sensor).
    let scattering = libm::acos((-to_sun.dot(to_sensor)).clamp(-1.0, 1.0));
    ObservationGeometry {
        solar_zenith: Angle::from_radians(solar_zenith),
        view_zenith: Angle::from_radians(view_zenith),
        relative_azimuth: Angle::from_radians(wrap_pi(view_az - sun_az)),
        scattering_angle: Angle::from_radians(scattering),
        air_mass: air_mass(Angle::from_radians(solar_zenith)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    const DEG: f64 = PI / 180.0;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn sun_and_sensor_at_zenith_give_zero_angles() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        let g = observation_geometry(up, up, up, north);
        assert!(approx(g.solar_zenith.radians(), 0.0, 1e-12));
        assert!(approx(g.view_zenith.radians(), 0.0, 1e-12));
        assert!(approx(g.air_mass, 1.0, 1e-3)); // straight up ⇒ one air mass
        // Sun and sensor coincident ⇒ exact backscatter.
        assert!(approx(g.scattering_angle.radians(), PI, 1e-9));
    }

    #[test]
    fn scattering_angle_spans_forward_to_back() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        // Sun at 45° in the north, sensor at 45° in the south ⇒ forward scatter (rays aligned).
        let s2 = 2.0_f64.sqrt() / 2.0;
        let to_sun = Vec3::new(s2, 0.0, s2);
        let to_sensor = Vec3::new(-s2, 0.0, s2);
        let g = observation_geometry(to_sun, to_sensor, up, north);
        // Incident (−to_sun) and scattered (to_sensor): (−s2,0,−s2)·(−s2,0,s2)=s2²−s2²=0 ⇒ 90°.
        assert!(approx(g.scattering_angle.radians(), PI / 2.0, 1e-9));
        // Relative azimuth is 180° (sun in the north, sensor in the south).
        assert!(approx(g.relative_azimuth.radians().abs(), PI, 1e-9));
        assert!(approx(g.solar_zenith.radians(), 45.0 * DEG, 1e-9));
    }

    #[test]
    fn wrap_pi_lands_in_its_half_open_range_and_moves_only_by_whole_turns() {
        // `fmod(π + π, 2π) = 0`, so +π is the excluded end and maps to −π.
        assert_eq!(wrap_pi(PI), -PI);
        assert_eq!(wrap_pi(-PI), -PI);
        assert_eq!(wrap_pi(0.0), 0.0);
        // −4.5 is below −π, the one regime where `fmod` returns a negative value and the branch that
        // lifts it back into `[0, 2π)` actually runs.
        for a in [-4.5_f64, -3.0, -1.7, -0.4, 0.0, 0.9, 2.6, 3.0] {
            for k in -3..=3 {
                let x = a + f64::from(k) * TAU;
                let w = wrap_pi(x);
                assert!((-PI..PI).contains(&w), "wrap_pi({x}) = {w} outside [−π, π)");
                let turns = (x - w) / TAU;
                assert!(
                    (turns - libm::round(turns)).abs() < 1e-12,
                    "wrap_pi({x}) = {w} moved by {turns} turns"
                );
            }
        }
    }

    #[test]
    fn the_geometry_reports_every_angle_its_directions_encode() {
        // Sun 35° from the vertical at azimuth 40°, sensor 20° at 250°: every angle distinct and
        // non-zero, so no part of the answer can hide behind a coordinate that happens to vanish.
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        // E = N × U = x̂ × ẑ = −ŷ.
        let dir = |zen: f64, az: f64| {
            let (sz, cz) = (libm::sin(zen), libm::cos(zen));
            Vec3::new(sz * libm::cos(az), -sz * libm::sin(az), cz)
        };
        let (zs, azs) = (35.0 * DEG, 40.0 * DEG);
        let (zv, azv) = (20.0 * DEG, 250.0 * DEG);
        let g = observation_geometry(dir(zs, azs), dir(zv, azv), up, north);

        assert!(approx(g.solar_zenith.radians(), zs, 1e-12));
        assert!(
            approx(g.view_zenith.radians(), zv, 1e-12),
            "view zenith {}",
            g.view_zenith.radians()
        );
        // 250° − 40° = 210°, one turn above the +π end, so −150°.
        assert!(
            approx(g.relative_azimuth.radians(), -150.0 * DEG, 1e-12),
            "relative azimuth {}",
            g.relative_azimuth.radians()
        );
        // Spherical law of cosines for the angle between the incident and the scattered ray.
        let cos_scat =
            -(libm::cos(zs) * libm::cos(zv) + libm::sin(zs) * libm::sin(zv) * libm::cos(azv - azs));
        assert!(approx(
            g.scattering_angle.radians(),
            libm::acos(cos_scat),
            1e-12
        ));
        // Kasten–Young at the solar zenith, written out rather than read back from `air_mass`.
        let ky = 1.0 / (libm::cos(zs) + 0.50572 * libm::pow(96.079_95 - 35.0, -1.6364));
        assert!(approx(g.air_mass, ky, 1e-15), "air mass {}", g.air_mass);
    }

    #[test]
    fn air_mass_matches_known_values_and_is_finite_at_the_horizon() {
        assert!(approx(air_mass(Angle::from_radians(0.0)), 1.0, 2e-3));
        // Kasten–Young sits just below the plane-parallel 1/cos θ (= 2.0 at 60°): ~1.994.
        let m60 = air_mass(Angle::from_radians(60.0 * DEG));
        assert!((1.98..2.0).contains(&m60), "air mass at 60° = {m60}");
        // Finite at the horizon (~38), where 1/cos θ diverges.
        let horizon = air_mass(Angle::from_radians(90.0 * DEG));
        assert!(
            (30.0..45.0).contains(&horizon),
            "horizon air mass {horizon}"
        );
    }
}
