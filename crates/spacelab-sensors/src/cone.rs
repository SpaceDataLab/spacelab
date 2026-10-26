//! Conical field of view, about the geocentric nadir or an explicit boresight.
//!
//! The footprint edge `asin(L sin alpha) - alpha`, its tangency with the horizon, the
//! division-free `u~` and the rate-bound terms are derived in `derivations/cone_footprint.mac`
//! (120-digit values: `derivations/cone_footprint.gp`) and gated in `tests/cone_gates.rs`.

use spacelab_core::Vec3;

use crate::{AccessGeometry, Constraint, OrbitBound, SensorError};

/// `1/(1-e^2)` on WGS-84 -- how much faster the geodetic normal can turn than the geocentric one.
///
/// Written from the flattening rather than transcribed, so it cannot drift from
/// [`spacelab_earth::FLATTENING`]; the crate does not depend on `spacelab-earth` (see [`Boresight`])
/// and `flattening_agrees_with_the_earth_crate` asserts the two are the same bits.
const INVERSE_ONE_MINUS_E2: f64 = {
    let f = 1.0 / 298.257_223_563;
    let e2 = f * (2.0 - f);
    1.0 / (1.0 - e2)
};

/// Where a cone's axis points.
///
/// [`Cone`] measured off-nadir from `-r/|r|`, the direction to the Earth's centre, and nothing said
/// so at the call site. That is one of two "downs" and not the one a nadir-pointing attitude built
/// on the local vertical holds: the ellipsoid's own inward normal is deflected from it by up to
/// **0.192 424 3 deg** at the surface, at geodetic latitude **45.096 2 deg** (not 45 -- the maximum
/// of `phi - phi'` sits at `atan(1/sqrt(1-e^2))`), and by **0.177 125 deg** at 550 km, which is
/// **1.70 km** on the ground straight below and more off-nadir. `derivations/nadir_deflection.mac`
/// derives both the closed form `atan(e^2/(2 sqrt(1-e^2)))` and the numbers at height.
///
/// [`Boresight::Fixed`] is deliberately more general than a nadir convention. An *agile* optical
/// instrument slews, and the crate could not express a commanded pointing at all; the geodetic nadir
/// is then one direction a caller can supply rather than a second hard-wired special case. It also
/// keeps `spacelab-sensors` at `spacelab-core` + `libm`: computing the geodetic normal needs the
/// ellipsoid, and taking `spacelab-earth` as a dependency here would pull `spacelab-frames` and
/// `spacelab-fa` -- CIO tables and all -- into the crate the batch kernels live in.
/// `spacelab_earth::geodetic_nadir` produces the vector; this holds it -- it is not linked because
/// `spacelab-earth` is a dev-dependency here, which is the whole point of the paragraph above.
///
/// The direction is per-satellite, so it is supplied once per satellite position and never enters
/// the cell loop -- the same hoist `n2` already gets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boresight {
    /// `-r/|r|`, recomputed from each satellite position. What [`Cone::new`] gives, unchanged.
    GeocentricNadir,
    /// A fixed direction, normalised on construction. The geodetic nadir, or any commanded
    /// pointing.
    Fixed(Vec3),
}

/// A circular field of view of half-angle `alpha` about a [`Boresight`].
///
/// # The predicate, without a square root *or* a division
///
/// Let `r` be the satellite position, `c` the cell, `w = c - r` the line of sight, and
/// `bhat = -r/|r|` the nadir direction. Then
///
/// ```text
/// cos(theta) = (w · bhat) / |w|
/// ```
///
/// and the cell is inside the cone when `cos(theta) >= cos(alpha)`. Write `n2 = r · r` and
/// `utilde = |r| * (w · bhat) = n2 - c · r`. Then, since `cos(alpha) > 0` for `alpha < pi/2`:
///
/// ```text
/// theta <= alpha   <=>   utilde >= 0  &&  utilde^2 >= cos^2(alpha) * (w·w) * n2
/// ```
///
/// `n2` is per-satellite and hoists out of the cell loop. Per cell this costs two dot products,
/// three multiplies and two comparisons. No square root, no division, no trigonometry.
///
/// # The horizon is not a special case
///
/// A cell can lie inside the cone and behind the limb. `SaVi` handles that with a branch on
/// `L >= 1/sin(alpha)`, switching to a horizon-limited footprint radius. We do not need one:
/// `And(Cone, ElevationMask::HORIZON)` is the correct constraint, and `min(g_cone, g_horizon)`
/// has its roots on the boundary of the intersection. `derivations/footprint.mac` shows that
/// `SaVi`'s branch condition is exactly the vanishing of the discriminant of the cone root, i.e.
/// the disappearance of a real intersection — not a separate geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    cos_half: f64,
    boresight: Boresight,
}

impl Cone {
    /// Build from a half-angle in radians, inside `(0, pi/2]`, about the geocentric nadir.
    ///
    /// # Errors
    ///
    /// [`SensorError::HalfAngleOutOfRange`] outside `(0, pi/2]`. A half-angle above `pi/2` gives
    /// `cos(alpha) < 0`, and the sign handling below assumes it is not.
    pub fn new(half_angle: f64) -> Result<Self, SensorError> {
        Self::pointed(half_angle, Boresight::GeocentricNadir)
    }

    /// The same cone about an explicit [`Boresight`].
    ///
    /// # Errors
    ///
    /// [`SensorError::HalfAngleOutOfRange`] as [`Cone::new`], and
    /// [`SensorError::DegenerateBoresight`] if a [`Boresight::Fixed`] direction has no length or is
    /// not finite -- a zero axis has no cone around it, and normalising it would produce `NaN`
    /// silently rather than an answer.
    pub fn pointed(half_angle: f64, boresight: Boresight) -> Result<Self, SensorError> {
        if !(half_angle > 0.0 && half_angle <= core::f64::consts::FRAC_PI_2) {
            return Err(SensorError::HalfAngleOutOfRange { half_angle });
        }
        let boresight = match boresight {
            Boresight::GeocentricNadir => Boresight::GeocentricNadir,
            Boresight::Fixed(axis) => {
                let norm = axis.norm();
                if !norm.is_finite() || norm <= 0.0 {
                    return Err(SensorError::DegenerateBoresight { norm });
                }
                Boresight::Fixed(axis.scale(1.0 / norm))
            }
        };
        Ok(Self {
            cos_half: libm::cos(half_angle),
            boresight,
        })
    }

    /// `cos(alpha)`.
    #[must_use]
    pub const fn cos_half(self) -> f64 {
        self.cos_half
    }

    /// The axis this cone is measured about, normalised.
    #[must_use]
    pub const fn boresight(self) -> Boresight {
        self.boresight
    }
}

/// `utilde = n2 - c·r`, `q = w·w`, `n2 = r·r`, computed exactly as the batch kernel computes them.
/// See `elevation::dots` for why this does not use `Vec3::dot`.
#[inline]
#[must_use]
pub(crate) fn terms(geom: &AccessGeometry) -> (f64, f64, f64) {
    let (rx, ry, rz) = (geom.satellite.x, geom.satellite.y, geom.satellite.z);
    let (cx, cy, cz) = (geom.cell.x, geom.cell.y, geom.cell.z);
    let n2 = rx * rx + ry * ry + rz * rz;
    let utilde = n2 - (cx * rx + cy * ry + cz * rz);
    let (dx, dy, dz) = (rx - cx, ry - cy, rz - cz);
    let q = dx * dx + dy * dy + dz * dz;
    (utilde, q, n2)
}

/// Sign of the cone `g` for one cell. `utilde = n2 - c·r`, `q = w·w`, `n2 = r·r`.
#[inline]
#[must_use]
pub(crate) fn covered_from(utilde: f64, q: f64, n2: f64, cos_half: f64) -> bool {
    utilde >= 0.0 && utilde * utilde >= cos_half * cos_half * q * n2
}

/// `u = w·bhat` and `q = w·w` for a fixed unit boresight, computed as the batch kernel computes
/// them. See `elevation::dots` for why this does not use `Vec3::dot`.
#[inline]
#[must_use]
pub(crate) fn terms_pointed(geom: &AccessGeometry, boresight: Vec3) -> (f64, f64) {
    let (dx, dy, dz) = (
        geom.cell.x - geom.satellite.x,
        geom.cell.y - geom.satellite.y,
        geom.cell.z - geom.satellite.z,
    );
    let u = dx * boresight.x + dy * boresight.y + dz * boresight.z;
    let q = dx * dx + dy * dy + dz * dz;
    (u, q)
}

/// Sign of the cone `g` about a fixed unit boresight. The `n2` of [`covered_from`] is absent rather
/// than set to one: it was there only to avoid normalising `r`, and a supplied boresight is already
/// unit, so the pointed form is the cheaper of the two per cell as well as the more general.
#[inline]
#[must_use]
pub(crate) fn covered_from_pointed(u: f64, q: f64, cos_half: f64) -> bool {
    u >= 0.0 && u * u >= cos_half * cos_half * q
}

impl Constraint for Cone {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        match self.boresight {
            Boresight::GeocentricNadir => {
                let (utilde, q, n2) = terms(geom);
                // g = (w·bhat) - cos(alpha)|w| = utilde/|r| - cos(alpha)*sqrt(q)
                utilde / libm::sqrt(n2) - self.cos_half * libm::sqrt(q)
            }
            Boresight::Fixed(axis) => {
                let (u, q) = terms_pointed(geom, axis);
                u - self.cos_half * libm::sqrt(q)
            }
        }
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        match self.boresight {
            Boresight::GeocentricNadir => {
                let (utilde, q, n2) = terms(geom);
                covered_from(utilde, q, n2, self.cos_half)
            }
            Boresight::Fixed(axis) => {
                let (u, q) = terms_pointed(geom, axis);
                covered_from_pointed(u, q, self.cos_half)
            }
        }
    }

    /// `g = (w·bhat) - cos(alpha)|w|`. Both terms are Lipschitz in the satellite position with
    /// constant 1 once `bhat` is frozen; `bhat` itself turns at `|v_perp|/|r| <= v_rel/r_perigee`,
    /// and `|w| <= r_apogee + R_body <= 2*r_apogee`. Bounding crudely:
    ///
    /// ```text
    /// |dg/dt| <= (1 + cos(alpha)) * v_rel  +  |w| * |dbhat/dt|
    ///         <= (1 + cos(alpha)) * v_rel  +  2 * r_apogee * v_rel / r_perigee
    /// ```
    ///
    /// Deliberately loose. A step bound that is too small wastes evaluations; one that is too
    /// large loses passes, and only one of those is a bug.
    ///
    /// A [`Boresight::Fixed`] axis does not turn at all *within* one satellite position, but the
    /// caller re-supplies it as the satellite moves, so the bound has to cover the direction it will
    /// be re-supplied as. For the geodetic nadir that is the geocentric one turned by the deflection,
    /// and `derivations/nadir_deflection.mac` bounds the extra: `d(phi')/d(phi)` runs from `1-e^2` at
    /// the equator to `1/(1-e^2)` in the limit at the pole, so the geodetic normal turns at most
    /// `1/(1-e^2)` times as fast as the geocentric one -- **0.674 % more**, and the same factor
    /// covers any pointing a caller re-supplies by tracking the local vertical. It is applied to the
    /// whole turn term rather than argued away, because a step bound that is right by 0.674 % and
    /// unproven is worth less than one that is loose by 0.674 % and derived.
    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        let v_rel = orbit.max_relative_speed(frame_rate);
        let turn = 2.0 * orbit.apogee_radius() * v_rel / orbit.perigee_radius();
        let boresight_turn = match self.boresight {
            Boresight::GeocentricNadir => turn,
            Boresight::Fixed(_) => turn * INVERSE_ONE_MINUS_E2,
        };
        (1.0 + self.cos_half) * v_rel + boresight_turn
    }
}
