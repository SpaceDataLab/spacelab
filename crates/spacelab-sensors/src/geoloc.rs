//! Direct geolocation: from a scan sample and a state vector to a point on the ellipsoid.
//!
//! This is the whiskbroom/push-broom counterpart of `spacelab-earthvision`'s rational-polynomial
//! camera. An RPC is a *fitted* sensor model — a vendor's polynomial that reproduces the rigorous
//! model over one scene, and says nothing outside it. This is the rigorous model itself: build the
//! line of sight from the platform's own axes, and intersect it with the ellipsoid. It needs no
//! coefficients from anybody, works for any instrument in the [`scan`](crate::scan) catalogue, and
//! extrapolates because there is nothing to extrapolate.
//!
//! # Provenance
//!
//! Ported from `pyorbital.geoloc` (pytroll/pyorbital, **Apache-2.0**). Two of its conventions are
//! surprising enough to be worth stating before they surprise somebody here:
//!
//! * **Its `qrotate` turns the other way.** pyorbital builds its rotation matrix as the transpose of
//!   the usual quaternion form, so `qrotate(v, axis, θ)` is a rotation by **−θ** in the right-hand
//!   sense — `x̂` about `ẑ` by +90° comes back as `−ŷ`, measured, not assumed. [`rotate_about`] keeps
//!   that convention, because the scan angles in the catalogue are signed to match it.
//! * **Its nadir is neither of the two nadirs.** [`nadir_direction`] returns the geocentric direction
//!   of the *ellipsoid subpoint*, negated — not `−r/‖r‖` (the geocentric nadir) and not the inward
//!   geodetic vertical. The three differ by up to the geodetic-geocentric latitude difference, about
//!   0.19° at mid-latitude, which is kilometres on the ground. [`nadir_direction`]'s own docs give
//!   the measured number.
//!
//! # One ellipsoid, and what that costs
//!
//! pyorbital uses **two**: its `subpoint`/`geodetic_lat` default to GRS80's semi-minor axis
//! (6356.752 314 14 km) while its ray-ellipsoid intersection uses WGS-84's (6356.752 314 245 km),
//! in the same computation. This module uses [`WGS84`] throughout, and that was measured rather than
//! waved through: the 105 µm between the two axes moves the boresight by 1.04e-12 rad, which is
//! **0.9 µm** of ground displacement at an 830 km nadir range. What does show up — 0.24 mm at that
//! range, 1.1 mm at VIIRS's swath edge — is pyorbital stopping its geodetic-latitude iteration at
//! `np.allclose`'s default `rtol = 1e-5`, where [`geodetic_latitude`] here runs to the `f64` fixed
//! point. The whole difference between the two codes is that this one iterates further.
//! `tests/pyorbital_geoloc.rs` holds the measurement.
//!
//! # Gate
//!
//! `derivations/ray_ellipsoid_geolocation.mac` proves the hit is the near root of the ray's
//! ellipsoid quadratic, the geodetic point a fixed point of the latitude iteration, the subpoint on
//! the ellipsoid under the geodetic vertical and the rotation Rodrigues' by `−θ`;
//! `derivations/ray_ellipsoid_geolocation.gp` recomputes its 40-digit values by root-finding. The
//! gates are in `tests/geoloc_gates.rs`: `the_ray_hit_is_the_near_root_of_the_ellipsoid_quadratic`,
//! `the_geodetic_latitude_is_the_root_of_its_defining_equation`,
//! `the_subpoint_is_the_foot_of_the_geodetic_vertical` (G2) and
//! `rotate_about_is_the_rodrigues_rotation_by_minus_the_angle` (G1).

use crate::scan::ScanSample;
use libm::{atan2, cos, sin, sqrt};
use spacelab_core::Vec3;

/// A reference ellipsoid, by its two axes in kilometres.
///
/// Kilometres rather than metres because that is the unit the orbit side of this computation works
/// in, and a unit conversion in the middle of a geolocation is a class of bug worth not having.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipsoid {
    /// Semi-major (equatorial) axis, km.
    pub equatorial_radius: f64,
    /// Semi-minor (polar) axis, km.
    pub polar_radius: f64,
}

/// WGS-84, the ellipsoid this module geolocates on.
pub const WGS84: Ellipsoid = Ellipsoid {
    equatorial_radius: 6_378.137,
    polar_radius: 6_356.752_314_245,
};

impl Ellipsoid {
    /// First eccentricity squared, `(a² − b²)/a²`.
    fn eccentricity_squared(self) -> f64 {
        let (a, b) = (self.equatorial_radius, self.polar_radius);
        (a - b) * (a + b) / (a * a)
    }
}

/// How many fixed-point steps [`geodetic_latitude`] may take before giving up.
///
/// The iteration contracts by roughly `e² ≈ 0.0067` per step, so eight steps take a geocentric
/// starting guess below the last bit of an `f64`. Twelve is that with room to spare, and a *bound*
/// rather than pyorbital's `while True`: an unbounded loop over a value a caller supplies is a hang
/// waiting for a bad input, which is the shape of defect #221, #222 and #289 all were.
const GEODETIC_LATITUDE_STEPS: u32 = 12;

/// The geodetic latitude of a point in the body-fixed frame, radians.
///
/// Iterates `φ ← atan2(z + a·C·e²·sin φ, √(x²+y²))` from the geocentric latitude. At the geocentre
/// the latitude is undefined and this returns zero, which is the only value that does not propagate
/// a `NaN` into the caller's line of sight.
#[must_use]
pub fn geodetic_latitude(point: Vec3, ellipsoid: Ellipsoid) -> f64 {
    let horizontal = sqrt(point.x * point.x + point.y * point.y);
    if horizontal == 0.0 && point.z == 0.0 {
        return 0.0;
    }
    let e2 = ellipsoid.eccentricity_squared();
    let mut latitude = atan2(point.z, horizontal);
    for _ in 0..GEODETIC_LATITUDE_STEPS {
        let sin_lat = sin(latitude);
        let curvature = 1.0 / sqrt(1.0 - e2 * sin_lat * sin_lat);
        let next = atan2(
            point.z + ellipsoid.equatorial_radius * curvature * e2 * sin_lat,
            horizontal,
        );
        // Exact equality is the fixed point itself, not a tolerance: the iteration is a contraction
        // and lands on a repeating `f64` rather than merely near one, so this both terminates early
        // and says precisely what it converged to.
        #[allow(clippy::float_cmp)]
        let settled = next == latitude;
        if settled {
            return next;
        }
        latitude = next;
    }
    latitude
}

/// The point on the ellipsoid at the geodetic latitude and longitude of `point`.
///
/// Note what this is *not*: the foot of the perpendicular from `point`, nor the point where the line
/// to the geocentre crosses the surface. It is the surface point sharing `point`'s geodetic
/// coordinates, which is what makes [`nadir_direction`] behave as documented there.
#[must_use]
pub fn subpoint(point: Vec3, ellipsoid: Ellipsoid) -> Vec3 {
    let latitude = geodetic_latitude(point, ellipsoid);
    let longitude = atan2(point.y, point.x);
    let e2 = ellipsoid.eccentricity_squared();
    let sin_lat = sin(latitude);
    let prime_vertical = ellipsoid.equatorial_radius / sqrt(1.0 - e2 * sin_lat * sin_lat);
    Vec3::new(
        prime_vertical * cos(latitude) * cos(longitude),
        prime_vertical * cos(latitude) * sin(longitude),
        (1.0 - e2) * prime_vertical * sin_lat,
    )
}

/// The boresight direction pyorbital's scan angles are measured from.
///
/// This is `−subpoint(r)/‖subpoint(r)‖`: the geocentric direction of the ellipsoid subpoint, turned
/// around to point down. It is a *third* thing, distinct from both nadirs a reader is likely to
/// assume — it is not `−r/‖r‖`, and it is not the inward geodetic vertical `−n̂`. Measured on the
/// WGS-84 ellipsoid at 45° geodetic latitude and 800 km altitude, it lies **0.0215°** from the
/// geocentric nadir and **0.1924°** from the geodetic vertical, and the geocentric nadir sits
/// exactly between the two (0.0215 + 0.1709 = 0.1924, to the digit). So it is a small correction to
/// the geocentric nadir applied in the *opposite* direction to the geodetic one — 0.300 km of ground
/// displacement at that altitude, against 2.69 km for the geodetic vertical. Not a rounding
/// question, and not the direction a reader would guess.
///
/// It is kept because the catalogue's scan angles are calibrated against it, and changing it would
/// silently move every pixel. The place to fix a boresight convention is with a measurement against
/// a geolocated product, which is [`crate::scan`]'s open question rather than a detail to guess at.
#[must_use]
pub fn nadir_direction(position: Vec3, ellipsoid: Ellipsoid) -> Vec3 {
    let surface = subpoint(position, ellipsoid);
    let norm = surface.norm();
    if norm == 0.0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    surface.scale(-1.0 / norm)
}

/// Rotate `vector` about `axis` by `angle`, **in pyorbital's sense**.
///
/// That is a rotation by `−angle` under the usual right-hand rule; see this module's header for why.
/// `axis` need not be normalised. A zero-length axis leaves the vector alone, which is the only
/// non-`NaN` answer available.
#[must_use]
pub fn rotate_about(vector: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let norm = axis.norm();
    if norm == 0.0 {
        return vector;
    }
    let unit = axis.scale(1.0 / norm);
    let (c, s) = (cos(angle), sin(angle));
    let cross = unit.cross(vector);
    let parallel = unit.dot(vector) * (1.0 - c);
    Vec3::new(
        vector.x * c - cross.x * s + unit.x * parallel,
        vector.y * c - cross.y * s + unit.y * parallel,
        vector.z * c - cross.z * s + unit.z * parallel,
    )
}

/// The platform's own orientation, added to every pixel's scan angles.
///
/// Zero is the nominal attitude: boresight along [`nadir_direction`], roll about the velocity
/// vector, pitch about the cross-track axis, yaw about the boresight.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Attitude {
    /// Rotation about the along-track axis, radians.
    pub roll: f64,
    /// Rotation about the cross-track axis, radians.
    pub pitch: f64,
    /// Rotation about the boresight, radians.
    pub yaw: f64,
}

/// The unit line of sight of one pixel, in the frame `position` and `velocity` are given in.
///
/// The three rotations compose in a fixed order — roll about the along-track axis, then pitch about
/// the cross-track axis, then yaw about the boresight — and the cross-track axis is built from the
/// *unrotated* boresight, so the second rotation does not inherit the first. That is pyorbital's
/// order and pyorbital's axis, and both are observable in the answer.
#[must_use]
pub fn line_of_sight(
    position: Vec3,
    velocity: Vec3,
    sample: ScanSample,
    attitude: Attitude,
    ellipsoid: Ellipsoid,
) -> Vec3 {
    let boresight = nadir_direction(position, ellipsoid);
    let speed = velocity.norm();
    if speed == 0.0 {
        return boresight;
    }
    let along_track = velocity.scale(1.0 / speed);
    let cross_track = boresight.cross(velocity);
    let cross_norm = cross_track.norm();
    if cross_norm == 0.0 {
        return boresight;
    }
    let cross_track = cross_track.scale(1.0 / cross_norm);

    let rolled = rotate_about(boresight, along_track, sample.roll + attitude.roll);
    let pitched = rotate_about(rolled, cross_track, sample.pitch + attitude.pitch);
    rotate_about(pitched, boresight, attitude.yaw)
}

/// Where a ray from `origin` along `direction` first meets the ellipsoid, or `None` if it misses.
///
/// `None` is the honest answer for a look past the limb, and it is a case that happens: an
/// instrument with a 55° half-swath at 800 km is 10° from the horizon at the swath edge, and any
/// attitude error or any deliberate limb view puts the ray outside. Returning a `NaN` position
/// instead — which is what the unguarded square root does — turns a geometric fact into a value that
/// propagates silently through every downstream average.
#[must_use]
pub fn intersect_ellipsoid(origin: Vec3, direction: Vec3, ellipsoid: Ellipsoid) -> Option<Vec3> {
    // Scale to the unit sphere: the ellipsoid problem is the sphere problem in stretched axes.
    let (a, b) = (ellipsoid.equatorial_radius, ellipsoid.polar_radius);
    let stretch = |v: Vec3| Vec3::new(v.x / a, v.y / a, v.z / b);
    let ray = stretch(direction);
    let to_centre = stretch(origin.scale(-1.0));

    let ray_dot_centre = ray.dot(to_centre);
    let ray_squared = ray.dot(ray);
    let centre_squared = to_centre.dot(to_centre);
    if ray_squared == 0.0 {
        return None;
    }
    let discriminant = ray_dot_centre * ray_dot_centre - ray_squared * (centre_squared - 1.0);
    if discriminant < 0.0 {
        return None;
    }
    let distance = (ray_dot_centre - sqrt(discriminant)) / ray_squared;
    if distance < 0.0 {
        return None;
    }
    let hit = direction.scale(distance);
    Some(Vec3::new(
        hit.x + origin.x,
        hit.y + origin.y,
        hit.z + origin.z,
    ))
}

/// One pixel, from state vector to ground point.
///
/// `position` and `velocity` must be the state at *this pixel's* observation time — that is the
/// whole point of [`ScanSample::time_offset`], and evaluating the orbit once per scan line instead
/// of once per pixel displaces the swath edge by the several kilometres the platform travels during
/// a sweep. Returns `None` when the line of sight misses the ellipsoid.
#[must_use]
pub fn geolocate(
    position: Vec3,
    velocity: Vec3,
    sample: ScanSample,
    attitude: Attitude,
    ellipsoid: Ellipsoid,
) -> Option<Vec3> {
    let direction = line_of_sight(position, velocity, sample, attitude, ellipsoid);
    intersect_ellipsoid(position, direction, ellipsoid)
}
