//! WGS-84 geometry.

use core::fmt;

use spacelab_core::{Angle, Length, Vec3};
use spacelab_frames::{Itrf, Position};

/// Semi-major axis, metres. A *defining* constant of WGS-84: exact by decree.
pub const EQUATORIAL_RADIUS: f64 = 6_378_137.0;

/// Flattening. The other defining constant, `1/298.257223563`.
pub const FLATTENING: f64 = 1.0 / 298.257_223_563;

/// Semi-minor axis, metres. Derived: `a(1 − f)`.
pub const POLAR_RADIUS: f64 = EQUATORIAL_RADIUS * (1.0 - FLATTENING);

/// First eccentricity squared, `f(2 − f)`.
///
/// Computed this way rather than as `1 − b²/a²`, which subtracts two numbers close
/// to one and throws away four significant digits.
const E2: f64 = FLATTENING * (2.0 - FLATTENING);

/// `e⁴`.
const E4: f64 = E2 * E2;

/// A point, by latitude, longitude and height above the ellipsoid.
///
/// Height above the *ellipsoid*, not the geoid. The two differ by up to about a
/// hundred metres, and the difference is a model we do not yet carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geodetic {
    /// Geodetic latitude, positive north. Not geocentric latitude: the two differ
    /// by up to 11.5 arcminutes, which is 21 km along the surface.
    pub latitude: Angle,
    /// Longitude, positive east.
    pub longitude: Angle,
    /// Height above the WGS-84 ellipsoid.
    pub height: Length,
}

/// Why a Cartesian position has no geodetic coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeodeticError {
    /// The point lies inside a small region around the Earth's centre where
    /// Vermeille's solution loses its footing.
    ///
    /// The condition is `(x² + y²)/a² + (1 − e²)z²/a² ≤ e⁴`, which bounds a lens
    /// about 43 km across. No orbiting object has ever been there.
    NearGeocentre,
}

impl fmt::Display for GeodeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NearGeocentre => write!(f, "the point lies within the geocentric singularity"),
        }
    }
}

impl core::error::Error for GeodeticError {}

/// Geodetic coordinates to an Earth-fixed Cartesian position.
///
/// The easy direction: no equation to solve.
#[must_use]
pub fn geodetic_to_itrf(g: Geodetic) -> Position<Itrf> {
    let (sin_lat, cos_lat) = libm::sincos(g.latitude.radians());
    let (sin_lon, cos_lon) = libm::sincos(g.longitude.radians());
    let h = g.height.metres();

    // Radius of curvature in the prime vertical.
    let n = EQUATORIAL_RADIUS / libm::sqrt(1.0 - E2 * sin_lat * sin_lat);

    Position::new(Vec3::new(
        (n + h) * cos_lat * cos_lon,
        (n + h) * cos_lat * sin_lon,
        (n * (1.0 - E2) + h) * sin_lat,
    ))
}

/// An Earth-fixed Cartesian position to geodetic coordinates.
///
/// Vermeille (2002), closed form. Every point costs the same: two square roots, a
/// cube root, and an `atan2`. There is no iteration count to tune and no
/// convergence to fail.
///
/// # Errors
///
/// [`GeodeticError::NearGeocentre`] inside the singular lens around the Earth's
/// centre, which is 43 km across and contains nothing.
// The single-letter names below are Vermeille's own. Renaming them to something a
// linter prefers would sever the correspondence with the published derivation,
// which is the only thing making this sequence of square roots reviewable.
#[allow(clippy::many_single_char_names)]
pub fn itrf_to_geodetic(p: Position<Itrf>) -> Result<Geodetic, GeodeticError> {
    let (x, y, z) = (p.vector.x, p.vector.y, p.vector.z);
    let a2 = EQUATORIAL_RADIUS * EQUATORIAL_RADIUS;

    let pp = (x * x + y * y) / a2;
    let q = (1.0 - E2) / a2 * z * z;
    let r = (pp + q - E4) / 6.0;

    if r <= 0.0 {
        return Err(GeodeticError::NearGeocentre);
    }

    let s = E4 * pp * q / (4.0 * r * r * r);
    // `s ≥ 0`, so `t ≥ 1` and `1/t` is never a division by zero. On the polar axis
    // and on the equator `s` is exactly zero and `t` is exactly one.
    let t = libm::cbrt(1.0 + s + libm::sqrt(s * (2.0 + s)));
    let u = r * (1.0 + t + 1.0 / t);
    let v = libm::sqrt(u * u + E4 * q);
    let w = E2 * (u + v - q) / (2.0 * v);
    let k = libm::sqrt(u + v + w * w) - w;

    let d = k * libm::sqrt(x * x + y * y) / (k + E2);
    let hyp = libm::hypot(d, z);

    Ok(Geodetic {
        // `2·atan2(z, d + hypot(d, z))` rather than `atan2(z, d)`: the half-angle
        // form is well conditioned at the poles, where `d → 0`.
        latitude: Angle::from_radians(2.0 * libm::atan2(z, d + hyp)),
        longitude: Angle::from_radians(libm::atan2(y, x)),
        height: Length::from_metres((k + E2 - 1.0) / k * hyp),
    })
}

/// The inward geodetic normal at a position — the "down" a nadir-pointing attitude built on the
/// local vertical actually holds, as opposed to `−r/|r|`, which points at the Earth's centre.
///
/// The two are not the same direction and the difference is not small for a pointing instrument.
/// Derived in `derivations/nadir_deflection.mac`: on the ellipsoid surface the deflection peaks at
/// `atan(e²/(2√(1−e²)))` = **0.192 424 3°**, at geodetic latitude **45.096 2°** rather than at 45°.
/// It falls with height, because `(N(1−e²)+h)/(N+h) → 1`: **0.177 125°** at 550 km, **0.173 366°** at
/// 700 km, **0.029 066°** at GEO. At 550 km that is **1.70 km** of ground displacement straight down,
/// and more off-nadir.
///
/// Returns a unit vector. The geodetic normal is *not* `−p/|p|` normalised differently — it is the
/// ellipsoid's own normal, which is why this reuses [`itrf_to_geodetic`]'s Vermeille solution rather
/// than a separate derivation: `d` and `z` there are already `cos φ` and `sin φ` up to a common
/// factor, so the normal is `(x/ρ · d, y/ρ · d, z)` normalised, with no trigonometry at all.
///
/// # Errors
///
/// [`GeodeticError::NearGeocentre`] inside the same 43 km lens [`itrf_to_geodetic`] refuses, and on
/// the exact polar axis where `ρ = 0` leaves the longitude undefined — there the normal is `∓ẑ` and
/// is returned as such rather than refused, because a satellite directly over a pole is ordinary.
#[allow(clippy::many_single_char_names)]
pub fn geodetic_nadir(p: Position<Itrf>) -> Result<Vec3, GeodeticError> {
    let (x, y, z) = (p.vector.x, p.vector.y, p.vector.z);
    let a2 = EQUATORIAL_RADIUS * EQUATORIAL_RADIUS;

    let pp = (x * x + y * y) / a2;
    let q = (1.0 - E2) / a2 * z * z;
    let r = (pp + q - E4) / 6.0;

    if r <= 0.0 {
        return Err(GeodeticError::NearGeocentre);
    }

    let s = E4 * pp * q / (4.0 * r * r * r);
    let t = libm::cbrt(1.0 + s + libm::sqrt(s * (2.0 + s)));
    let u = r * (1.0 + t + 1.0 / t);
    let v = libm::sqrt(u * u + E4 * q);
    let w = E2 * (u + v - q) / (2.0 * v);
    let k = libm::sqrt(u + v + w * w) - w;

    let rho = libm::hypot(x, y);
    let d = k * rho / (k + E2);
    let hyp = libm::hypot(d, z);
    if rho == 0.0 || hyp == 0.0 {
        // On the polar axis the meridian is undefined but the normal is not: it is the axis itself,
        // pointing inward. `z.signum()` rather than a comparison, so `-0.0` on the axis still
        // resolves to a direction instead of a zero vector.
        return Ok(Vec3::new(0.0, 0.0, -libm::copysign(1.0, z)));
    }

    // `(d/hyp, z/hyp)` is `(cos φ, sin φ)`; the longitude enters only as `(x, y)/ρ`. Negated,
    // because this is the *inward* normal.
    Ok(Vec3::new(
        -(x / rho) * (d / hyp),
        -(y / rho) * (d / hyp),
        -(z / hyp),
    ))
}

/// Convert many positions, writing into a caller-provided slice.
///
/// The shape every batch entry point in this workspace takes: `&[In]`, `&mut [Out]`.
/// No allocation, no aliasing, and a parallel backend splits the slice without the
/// kernel changing. See `plans/08-parallelism.md`.
///
/// A point in the geocentric singularity yields `Err` in its own slot; the rest of
/// the batch is unaffected.
///
/// # Panics
///
/// If `out` is shorter than `positions`.
pub fn itrf_to_geodetic_into(
    positions: &[Position<Itrf>],
    out: &mut [Result<Geodetic, GeodeticError>],
) {
    assert!(
        out.len() >= positions.len(),
        "output slice holds {} of {} positions",
        out.len(),
        positions.len()
    );
    for (p, slot) in positions.iter().zip(out.iter_mut()) {
        *slot = itrf_to_geodetic(*p);
    }
}
