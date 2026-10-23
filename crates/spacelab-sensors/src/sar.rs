//! Side-looking SAR: access is the ground incidence-angle band.

use spacelab_core::Vec3;

use crate::elevation::dots;
use crate::{AccessGeometry, Constraint, OrbitBound, SensorError};

/// A side-looking synthetic-aperture radar images a ground cell only when the local **incidence
/// angle** — the angle at the cell between the vertical and the line of sight to the satellite —
/// falls inside the instrument's band `[near, far]`. That band is the binding SAR access
/// constraint: near range is limited by layover and foreshortening, far range by the antenna
/// footprint and SNR. Representative bands: Sentinel-1 IW ≈ 29–46°, TerraSAR-X ≈ 20–45°, agile
/// X-band (Capella, ICEYE) ≈ 15–50°.
///
/// # Geometry — exact on the ellipsoid
///
/// With `d = r_sat − r_cell`, `n̂` the geodetic zenith the [`AccessGeometry`] already carries,
/// `t = d·n̂` and `q = d·d`: `cos θ_inc = n̂ · d/|d| = t/√q`. Incidence is the complement of
/// elevation (`θ_inc = 90° − ε`), so the band is an annulus between two elevation cones —
/// no spherical or small-Earth approximation, the geodetic normal carries the ellipsoid shape.
/// This differs from the satellite off-nadir *look* angle η by the Earth-central angle
/// (`θ_inc > η`); the incidence, not the look angle, is what the ground and the constraint see.
///
/// # Look side
///
/// Left or right of the ground track is a per-pass choice that depends on the satellite
/// *velocity*, which a [`Constraint`] (position only) does not see. Filter with [`look_side`]
/// before or after; the incidence band this models is shared by both sides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SideLookingSAR {
    /// `cos(θ_near)` — the larger cosine, since `θ_near < θ_far`.
    cos_near: f64,
    /// `cos(θ_far)`.
    cos_far: f64,
}

impl SideLookingSAR {
    /// Build from the near- and far-range incidence angles, radians, with
    /// `0 ≤ near < far < π/2`.
    ///
    /// # Errors
    ///
    /// [`SensorError::IncidenceBandOutOfRange`] if the band is empty or leaves `[0, π/2)`.
    pub fn new(near_incidence: f64, far_incidence: f64) -> Result<Self, SensorError> {
        if !(near_incidence >= 0.0
            && near_incidence < far_incidence
            && far_incidence < core::f64::consts::FRAC_PI_2)
        {
            return Err(SensorError::IncidenceBandOutOfRange {
                near: near_incidence,
                far: far_incidence,
            });
        }
        Ok(Self {
            cos_near: libm::cos(near_incidence),
            cos_far: libm::cos(far_incidence),
        })
    }
}

impl Constraint for SideLookingSAR {
    /// `min(θ_inc ≤ far, θ_inc ≥ near)` as a continuous scalar: `min(t − cosθ_far·√q,
    /// cosθ_near·√q − t)`. Zero on either band edge, positive inside the annulus.
    fn g(&self, geom: &AccessGeometry) -> f64 {
        let (t, q) = dots(geom);
        let root = libm::sqrt(q);
        let below_far = t - self.cos_far * root; // θ_inc ≤ far  ⇔ cosθ ≥ cosθ_far
        let above_near = self.cos_near * root - t; // θ_inc ≥ near ⇔ cosθ ≤ cosθ_near
        below_far.min(above_near)
    }

    /// Square-root-free: `t ≥ 0` (above the horizon) and `cosθ_far²·q ≤ t² ≤ cosθ_near²·q`.
    fn covered(&self, geom: &AccessGeometry) -> bool {
        let (t, q) = dots(geom);
        let (lhs, near, far) = (
            t * t,
            self.cos_near * self.cos_near,
            self.cos_far * self.cos_far,
        );
        t >= 0.0 && lhs >= far * q && lhs <= near * q
    }

    /// Each edge is an elevation-type constraint with `|dg/dt| ≤ (1 + cosθ)·v_rel`; the band's
    /// bound is the larger, `(1 + cosθ_near)·v_rel`. See [`crate::ElevationMask::max_rate`].
    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        (1.0 + self.cos_near.max(self.cos_far)) * orbit.max_relative_speed(frame_rate)
    }
}

/// Signed look side of `cell` relative to the ground track: **positive to the satellite's right
/// (starboard), negative to the left, zero in the orbit plane**. A right-looking SAR (Sentinel-1)
/// accepts `> 0`, a left-looking one `< 0`, an agile SAR (both) either.
///
/// `ĉ = d̂ × v̂` with down `d̂ = −r_sat/|r_sat|` and forward `v̂ = v/|v|` points starboard; the
/// returned value is the ground-pointing direction `(r_cell − r_sat) · ĉ`, whose sign is the side
/// (its magnitude is a projected length, not an angle).
#[must_use]
pub fn look_side(satellite: Vec3, velocity: Vec3, cell: Vec3) -> f64 {
    let down = satellite.scale(-1.0 / satellite.norm());
    let forward = velocity.scale(1.0 / velocity.norm());
    let starboard = down.cross(forward);
    let to_cell = Vec3::new(
        cell.x - satellite.x,
        cell.y - satellite.y,
        cell.z - satellite.z,
    );
    to_cell.dot(starboard)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    // A cell at the subsatellite point (nadir) has zero incidence; a cell near the horizon has
    // incidence approaching 90°. Build a simple spherical geometry and check the band.
    fn geom_at_incidence(theta: f64) -> AccessGeometry {
        // Cell at the north pole of a unit sphere; zenith = cell direction. Place the satellite so
        // the line of sight makes angle `theta` with the zenith.
        let re = 6_378_137.0;
        let alt = 700e3;
        let cell = Vec3::new(0.0, 0.0, re);
        let zenith = Vec3::new(0.0, 0.0, 1.0);
        // Satellite at radius re+alt, offset in x so the LOS zenith angle is `theta`.
        // LOS = sat - cell; choose sat = cell + (re+alt-ish)·(sinθ, 0, cosθ) direction from cell.
        let r = alt / libm::cos(theta).max(0.1); // rough slant so it stays above the surface
        let sat = Vec3::new(r * libm::sin(theta), 0.0, re + r * libm::cos(theta));
        AccessGeometry {
            satellite: sat,
            cell,
            zenith,
        }
    }

    #[test]
    fn incidence_inside_the_band_is_covered_outside_is_not() {
        let sar = SideLookingSAR::new(20f64.to_radians(), 46f64.to_radians()).unwrap();
        assert!(
            sar.covered(&geom_at_incidence(33f64.to_radians())),
            "mid-band"
        );
        assert!(
            !sar.covered(&geom_at_incidence(10f64.to_radians())),
            "inside near range"
        );
        assert!(
            !sar.covered(&geom_at_incidence(60f64.to_radians())),
            "past far range"
        );
    }

    #[test]
    fn g_agrees_with_covered_and_changes_sign_across_each_edge() {
        let sar = SideLookingSAR::new(20f64.to_radians(), 46f64.to_radians()).unwrap();
        // Clear of the exact edges, `g >= 0` and `covered` agree (at the edge the √q and squared
        // forms can differ by an ulp, as for ElevationMask).
        for deg in [5.0_f64, 15.0, 25.0, 33.0, 44.0, 55.0, 70.0] {
            let geom = geom_at_incidence(deg.to_radians());
            assert_eq!(
                sar.g(&geom) >= 0.0,
                sar.covered(&geom),
                "g/covered disagree at {deg}°"
            );
        }
        // The switching function brackets each band edge: it changes sign there.
        let g = |d: f64| sar.g(&geom_at_incidence(d.to_radians()));
        assert!(
            g(19.0) < 0.0 && g(21.0) > 0.0,
            "no sign change at the near edge"
        );
        assert!(
            g(45.0) > 0.0 && g(47.0) < 0.0,
            "no sign change at the far edge"
        );
    }

    #[test]
    fn look_side_is_positive_to_starboard() {
        // Satellite over the equator moving north (+y). Down is −z at (R,0,0)... use a clean setup:
        // sat at (R, 0, 0), velocity +z (moving "north" along the x-face). Down = (−1,0,0).
        // forward = (0,0,1). starboard = down × forward = (−1,0,0)×(0,0,1) = (0·1−0·0, 0·0−(−1)·1, 0) = (0, 1, 0).
        let sat = Vec3::new(7_000e3, 0.0, 0.0);
        let vel = Vec3::new(0.0, 0.0, 7_500.0);
        // A cell displaced toward +y is to starboard → positive.
        let right = look_side(sat, vel, Vec3::new(6_378e3, 500e3, 0.0));
        let left = look_side(sat, vel, Vec3::new(6_378e3, -500e3, 0.0));
        assert!(
            right > 0.0,
            "starboard cell should be positive, got {right}"
        );
        assert!(left < 0.0, "port cell should be negative, got {left}");
    }
}
