//! Orbital event detectors — the `g`-functions of Orekit `EclipseDetector`, `ElevationDetector` and
//! `FieldOfViewDetector`, ready to feed [`crate::find_events`].
//!
//! Each detector is a small parameter bundle with a `g(state) -> f64` method whose sign changes mark
//! the event; wrap it in a closure over a dense trajectory, `find_events(|t| det.g(pos(t), …), …)`,
//! and the bracketing root finder locates the crossing. Positions are `spacelab_core::Vec3` in one
//! consistent frame (inertial for eclipse, the same body-fixed frame for a ground station, either for
//! a field of view). The sign conventions match Orekit exactly so a port behaves identically:
//! * eclipse — `g > 0` in sunlight, `g < 0` inside the shadow;
//! * elevation — `g > 0` above the mask (visible), `g < 0` below;
//! * field of view — `g < 0` inside the cone (target visible), `g > 0` outside (Orekit's
//!   offset-from-boundary convention).

// The single letters (a, b, d, g) are the notation of the detector geometry.
#![allow(clippy::many_single_char_names)]

use spacelab_core::Vec3;

use crate::observability::{ApparentAltitude, Refraction};

/// IAU 2015 (Resolution B3) nominal solar radius `R_⊙ᴺ`, metres — a convenient occulted-body radius
/// for [`EclipseDetector`]. The detector stores whatever radius it is given; pass Orekit's
/// `Constants.SUN_RADIUS` (`6.955e8`) instead to match Orekit bit-for-bit.
pub const SUN_RADIUS: f64 = 6.957e8;

/// WGS-84 Earth equatorial radius, metres — the usual occulting-body radius for [`EclipseDetector`]
/// (matches Orekit's `Constants.WGS84_EARTH_EQUATORIAL_RADIUS`).
pub const EARTH_EQUATORIAL_RADIUS: f64 = 6_378_137.0;

/// The angle between two vectors, `atan2(|a×b|, a·b)` — accurate for small and near-π angles alike.
fn angle_between(a: Vec3, b: Vec3) -> f64 {
    libm::atan2(a.cross(b).norm(), a.dot(b))
}

/// `asin` of a ratio clamped to `[-1, 1]`, so an apparent radius stays finite even if the geometry is
/// degenerate (e.g. a point below the body's surface radius).
fn asin_clamped(x: f64) -> f64 {
    libm::asin(x.clamp(-1.0, 1.0))
}

/// Which shadow boundary an [`EclipseDetector`] tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EclipseKind {
    /// Total shadow — the occulting body fully hides the occulted one.
    Umbra,
    /// Partial shadow — the occulting body hides part of the occulted one; its outer edge is where the
    /// two apparent disks first touch.
    Penumbra,
}

/// Occultation of a light source (the Sun) by a body centred at the inertial origin (the Earth) —
/// Orekit `EclipseDetector`. `g > 0` in sunlight, `g < 0` inside the chosen shadow.
///
/// With `to_sun` the satellite→Sun vector and `to_body` the satellite→origin vector, `sep` their
/// apparent angular separation, and `α_occulted`, `α_occulting` the apparent angular radii, the
/// boundary is `g = sep − (α_occulting ∓ α_occulted)` (minus for the umbra, plus for the penumbra).
#[derive(Debug, Clone, Copy)]
pub struct EclipseDetector {
    /// Radius of the occulting body (Earth), metres.
    pub occulting_radius: f64,
    /// Radius of the occulted body (Sun), metres.
    pub occulted_radius: f64,
    /// Which shadow boundary this detector tracks.
    pub kind: EclipseKind,
}

impl EclipseDetector {
    /// An umbra (total-eclipse) detector.
    #[must_use]
    pub const fn umbra(occulting_radius: f64, occulted_radius: f64) -> Self {
        Self {
            occulting_radius,
            occulted_radius,
            kind: EclipseKind::Umbra,
        }
    }

    /// A penumbra (partial-eclipse) detector.
    #[must_use]
    pub const fn penumbra(occulting_radius: f64, occulted_radius: f64) -> Self {
        Self {
            occulting_radius,
            occulted_radius,
            kind: EclipseKind::Penumbra,
        }
    }

    /// The eclipse `g` for a satellite at inertial `sat` with the occulted body (Sun) at inertial
    /// `occulted`; the occulting body (Earth) is at the origin. `g > 0` in sunlight.
    #[must_use]
    pub fn g(&self, sat: Vec3, occulted: Vec3) -> f64 {
        let to_occulted = occulted - sat;
        let to_occulting = sat.scale(-1.0); // Earth at the origin
        let sep = angle_between(to_occulted, to_occulting);
        let alpha_occulted = asin_clamped(self.occulted_radius / to_occulted.norm());
        let alpha_occulting = asin_clamped(self.occulting_radius / to_occulting.norm());
        match self.kind {
            EclipseKind::Umbra => sep - (alpha_occulting - alpha_occulted),
            EclipseKind::Penumbra => sep - (alpha_occulting + alpha_occulted),
        }
    }
}

/// Line-of-sight of a satellite above a ground station's elevation mask — Orekit `ElevationDetector`.
/// `g = elevation − min_elevation`, so `g > 0` when the satellite is visible above the mask. The
/// station and satellite are in the same body-fixed frame; `up` is the station's local vertical.
///
/// The elevation is **geometric** — the straight line of sight, which is the angle Orekit's detector
/// reports before an `AtmosphericRefractionModel` is attached to it. Refraction is
/// [`crate::observability::Refraction`], and it enters here as a shifted `min_elevation` rather than
/// as a transform of `g`: `h ↦ h + R(h)` is monotone, so an apparent mask is exactly one geometric
/// mask, and `Refraction::geometric` is the conversion. What that is worth in pass-edge seconds —
/// 0.81 s to 20.3 s depending on the mask and how steeply the pass crosses it — is measured in
/// `spacelab/tests/refraction_in_access.rs`.
#[derive(Debug, Clone, Copy)]
pub struct ElevationDetector {
    /// Station position, body-fixed, metres.
    pub station: Vec3,
    /// Station local vertical (unit).
    pub up: Vec3,
    /// Minimum elevation to clear, radians.
    pub min_elevation: f64,
}

impl ElevationDetector {
    /// A detector with an explicit local vertical `up` (e.g. the geodetic normal); `up` is normalised.
    #[must_use]
    pub fn new(station: Vec3, up: Vec3, min_elevation: f64) -> Self {
        Self {
            station,
            up: up.scale(1.0 / up.norm()),
            min_elevation,
        }
    }

    /// A detector whose local vertical is the geocentric radial `station/|station|` (spherical Earth).
    #[must_use]
    pub fn geocentric(station: Vec3, min_elevation: f64) -> Self {
        Self::new(station, station, min_elevation)
    }

    /// A detector for a mask stated as an **apparent** elevation — what a station's "visible above
    /// 10°" actually means — with the atmosphere converted into the threshold once.
    ///
    /// [`Self::new`] takes the geometric mask, which is the angle this detector's `g` measures. The
    /// two differ by 5.3′ at 10° and 34′ at the horizon, worth 0.81 s to 20.3 s of pass edge with the
    /// same sign every time (`spacelab/tests/refraction_in_access.rs`), so the conversion is not
    /// optional for a station whose spec is written the usual way. It is exact and not an
    /// approximation, for the reason given on [`Refraction::elevation_mask`].
    ///
    /// `None` when the apparent mask is outside [`Refraction::geometric`]'s domain.
    #[must_use]
    pub fn apparent(
        station: Vec3,
        up: Vec3,
        mask: ApparentAltitude,
        refraction: Refraction,
    ) -> Option<Self> {
        Some(Self::new(station, up, refraction.geometric(mask)?.0))
    }

    /// The elevation `g` for a satellite at body-fixed `sat`. `g > 0` above the mask.
    ///
    /// Elevation is `atan2(vertical, |horizontal|)` — the well-conditioned topocentric form (Orekit
    /// `TopocentricFrame.getElevation`), not `asin(vertical/range)`. `asin` is singular at the zenith
    /// (`asin'(x) = 1/√(1−x²) → ∞`), so a `1e-16` rounding in a not-quite-unit `up` would blow up to
    /// `~1e-8` rad there; `atan2` against the horizontal residual vector stays exact at 90°.
    #[must_use]
    pub fn g(&self, sat: Vec3) -> f64 {
        let d = sat - self.station;
        let vertical = d.dot(self.up);
        let horizontal = (d - self.up.scale(vertical)).norm();
        libm::atan2(vertical, horizontal) - self.min_elevation
    }
}

/// A target's transit of a circular (conical) field of view — Orekit `FieldOfViewDetector` with a
/// `CircularFieldOfView`. Following Orekit, `g` is the target's angular offset from the cone boundary:
/// `g < 0` inside the field of view (visible), `g = 0` on the edge, `g > 0` outside.
#[derive(Debug, Clone, Copy)]
pub struct CircularFovDetector {
    /// Sensor position, metres.
    pub observer: Vec3,
    /// Boresight direction (unit).
    pub boresight: Vec3,
    /// Half-aperture of the cone, radians.
    pub half_aperture: f64,
}

impl CircularFovDetector {
    /// A detector looking along `boresight` (normalised) with the given half-aperture.
    #[must_use]
    pub fn new(observer: Vec3, boresight: Vec3, half_aperture: f64) -> Self {
        Self {
            observer,
            boresight: boresight.scale(1.0 / boresight.norm()),
            half_aperture,
        }
    }

    /// The field-of-view `g` for a target at `target`. `g < 0` when the target is inside the cone.
    #[must_use]
    pub fn g(&self, target: Vec3) -> f64 {
        angle_between(target - self.observer, self.boresight) - self.half_aperture
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EventKind, find_events};

    const PI: f64 = core::f64::consts::PI;
    const AU: f64 = 1.495_978_707e11;

    fn sun_far() -> Vec3 {
        Vec3::new(AU, 0.0, 0.0)
    }

    #[test]
    fn eclipse_is_negative_behind_the_earth_and_positive_facing_the_sun() {
        let umbra = EclipseDetector::umbra(EARTH_EQUATORIAL_RADIUS, SUN_RADIUS);
        let pen = EclipseDetector::penumbra(EARTH_EQUATORIAL_RADIUS, SUN_RADIUS);
        let h = 700_000.0;
        let behind = Vec3::new(-(EARTH_EQUATORIAL_RADIUS + h), 0.0, 0.0); // anti-Sun
        let facing = Vec3::new(EARTH_EQUATORIAL_RADIUS + h, 0.0, 0.0); // sub-solar

        assert!(
            umbra.g(behind, sun_far()) < 0.0,
            "deep umbra should be negative"
        );
        assert!(
            pen.g(behind, sun_far()) < 0.0,
            "deep penumbra should be negative"
        );
        assert!(
            umbra.g(facing, sun_far()) > 0.0,
            "sub-solar should be sunlit"
        );
        // The penumbra swallows the umbra: its boundary is always the more negative of the two.
        assert!(pen.g(behind, sun_far()) <= umbra.g(behind, sun_far()));
    }

    #[test]
    fn eclipse_boundary_is_bracketed_along_an_orbit_arc() {
        // A satellite sweeping from the sub-solar point round to the anti-Sun point must cross the
        // umbra boundary exactly once (sunlight → shadow), and find_events must bracket it.
        let umbra = EclipseDetector::umbra(EARTH_EQUATORIAL_RADIUS, SUN_RADIUS);
        let r = EARTH_EQUATORIAL_RADIUS + 700_000.0;
        let sat = |t: f64| Vec3::new(r * libm::cos(t), 0.0, r * libm::sin(t)); // t: 0 (sub-solar) → π
        let mut events = 0;
        let mut kind = None;
        find_events(
            |t| umbra.g(sat(t), sun_far()),
            0.1,
            PI - 0.1,
            0.02,
            1e-9,
            |e| {
                events += 1;
                kind = Some(e.kind);
            },
        );
        assert_eq!(events, 1, "exactly one umbra entry over the arc");
        assert_eq!(
            kind,
            Some(EventKind::Falling),
            "sunlight → shadow is a falling edge"
        );
    }

    #[test]
    fn the_eclipse_cone_angles_are_the_separation_minus_the_two_apparent_radii() {
        // The cases above read only the sign of `g`, which leaves every term of it free: the two
        // apparent radii could be swapped, added instead of subtracted, or scaled by the range
        // instead of divided by it, and the satellite would still be in shadow behind the Earth.
        //
        // Give the two bodies the same radius and put the satellite the same distance from each.
        // Then `asin(ρ/d)` is the same angle for both, so the umbra cone is the separation exactly
        // and the penumbra cone is the separation less twice that angle — no term left implicit.
        let d = 1.0e7;
        let rho = 5.0e6; // asin(ρ/d) = asin(0.5) = π/6
        let sep = 1.0_f64;
        let sat = Vec3::new(0.0, 0.0, d); // Earth at the origin, so |to_occulting| = d
        // `to_occulted` of length d, at `sep` from the satellite→Earth direction (0, 0, −1).
        let occulted = sat + Vec3::new(d * libm::sin(sep), 0.0, -d * libm::cos(sep));

        let umbra = EclipseDetector::umbra(rho, rho);
        let pen = EclipseDetector::penumbra(rho, rho);
        assert!(
            (umbra.g(sat, occulted) - sep).abs() < 1e-12,
            "umbra g = {} for equal apparent radii, expected the separation {sep}",
            umbra.g(sat, occulted)
        );
        assert!(
            (pen.g(sat, occulted) - (sep - PI / 3.0)).abs() < 1e-12,
            "penumbra g = {}, expected {} — the separation less 2·asin(1/2)",
            pen.g(sat, occulted),
            sep - PI / 3.0
        );
    }

    #[test]
    fn the_field_of_view_is_measured_from_the_observer_wherever_it_is() {
        // The case below puts the observer at the origin, where `target − observer` and
        // `target + observer` are the same vector and the subtraction is untested.
        let observer = Vec3::new(1.0e6, -2.0e6, 5.0e5);
        let half = 10.0_f64.to_radians();
        let det = CircularFovDetector::new(observer, Vec3::new(0.0, 0.0, 1.0), half);
        let d = 4.0e5;
        // On the boresight, and on the cone, both offset from the observer rather than the origin.
        let on_axis = observer + Vec3::new(0.0, 0.0, d);
        assert!((det.g(on_axis) + half).abs() < 1e-12, "{}", det.g(on_axis));
        let on_cone = observer + Vec3::new(d * libm::sin(half), 0.0, d * libm::cos(half));
        assert!(det.g(on_cone).abs() < 1e-9, "{}", det.g(on_cone));
    }

    #[test]
    fn elevation_is_ninety_degrees_overhead_and_zero_on_the_horizon() {
        let station = Vec3::new(EARTH_EQUATORIAL_RADIUS, 0.0, 0.0);
        let det = ElevationDetector::geocentric(station, 0.0);
        // Directly overhead: elevation 90°.
        let overhead = Vec3::new(EARTH_EQUATORIAL_RADIUS + 500_000.0, 0.0, 0.0);
        assert!((det.g(overhead) - PI / 2.0).abs() < 1e-9);
        // A 45° geometry: displacement (L, L, 0) from the station gives asin(1/√2).
        let l = 500_000.0;
        let at45 = station + Vec3::new(l, l, 0.0);
        assert!((det.g(at45) - PI / 4.0).abs() < 1e-9);
        // On the local horizon (displacement ⟂ up): elevation 0.
        let horizon = station + Vec3::new(0.0, l, 0.0);
        assert!(det.g(horizon).abs() < 1e-9);
    }

    #[test]
    fn elevation_mask_shifts_the_zero_crossing() {
        let station = Vec3::new(EARTH_EQUATORIAL_RADIUS, 0.0, 0.0);
        let mask = 5.0_f64.to_radians();
        let det = ElevationDetector::new(station, Vec3::new(1.0, 0.0, 0.0), mask);
        // A target exactly at the mask elevation gives g ≈ 0.
        let l = 500_000.0;
        let up = l * libm::tan(mask);
        let at_mask = station + Vec3::new(up, l, 0.0);
        assert!(
            det.g(at_mask).abs() < 1e-9,
            "g at mask elevation: {}",
            det.g(at_mask)
        );
    }

    #[test]
    fn circular_fov_is_negative_on_axis_and_positive_outside() {
        let det = CircularFovDetector::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            10.0_f64.to_radians(),
        );
        // On boresight: offset 0, so g = −half_aperture.
        assert!((det.g(Vec3::new(0.0, 0.0, 1.0)) + 10.0_f64.to_radians()).abs() < 1e-12);
        // A target 10° off-axis sits on the boundary.
        let edge = Vec3::new(
            libm::sin(10.0_f64.to_radians()),
            0.0,
            libm::cos(10.0_f64.to_radians()),
        );
        assert!(det.g(edge).abs() < 1e-9);
        // 20° off-axis is outside.
        let outside = Vec3::new(
            libm::sin(20.0_f64.to_radians()),
            0.0,
            libm::cos(20.0_f64.to_radians()),
        );
        assert!((det.g(outside) - 10.0_f64.to_radians()).abs() < 1e-9);
    }
}
