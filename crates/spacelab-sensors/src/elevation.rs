//! Elevation-mask constraint: the satellite must rise at least `eps_min` above the local horizon.

use spacelab_core::Vec3;

use crate::{AccessGeometry, Constraint, OrbitBound, SensorError};

/// The satellite is visible from the cell when its elevation above the local horizon is at least
/// `eps_min`.
///
/// # The predicate, without a square root
///
/// With `d = r_sat - r_cell`, `t = d · zhat`, `q = d · d` and `s = sin(eps_min)`:
///
/// ```text
/// eps >= eps_min   <=>   sin(eps) >= s   <=>   t >= s * sqrt(q)
/// ```
///
/// because `sin` is strictly increasing on `[-pi/2, pi/2]` and both elevations live there. The
/// square root is then removable by squaring, provided the sign of `t` is handled:
///
/// ```text
/// s >= 0:   t >= s*sqrt(q)   <=>   t >= 0  &&  t*t >= s*s*q
/// s <  0:   t >= s*sqrt(q)   <=>   t >= 0  ||  t*t <= s*s*q
/// ```
///
/// The boundary is preserved exactly: for `t, s >= 0`, `t*t == s*s*q` iff `t == s*sqrt(q)`.
/// `derivations/predicate_identical.m` checks the two forms agree pointwise over ±2000 ulp of the
/// boundary at five altitudes from LEO to GEO, and finds a single clean transition — no fuzzy
/// band. Squaring is also the *sharper* form near the root: if `t = s*sqrt(q)*(1+delta)` then
/// `t*t = s*s*q*(1+2*delta)`, so the comparison margin doubles in relative terms while the
/// rounding error stays at about one ulp.
///
/// # `eps_min` is a geometric elevation
///
/// The angle above is the one to the straight line of sight, before the atmosphere bends it. A
/// station's "visible above 10°" is normally the *apparent* elevation, and the two differ by 5.3' at
/// 10° and 34' at the horizon. Measured on the ISS day of `spacelab/tests/refraction_in_access.rs`,
/// that is **0.81–1.18 s** of pass edge at a 10° mask and **9.3–20.3 s** at the horizon — a bias with
/// the same sign on every edge of every pass, so unlike the TLE error it is the size of, it does not
/// average out.
///
/// Correcting it costs nothing per sample, because `h -> h + R(h)` is strictly increasing and an
/// apparent mask is therefore exactly one geometric mask. Convert the threshold once with
/// `spacelab_coverage::observability::Refraction::geometric` and hand the result to
/// [`ElevationMask::new`]; the same test pins that route against refracting every sample, and what
/// separates them is the 3.7" between the Saemundsson and Bennett fits, not an approximation in the
/// shift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElevationMask {
    sin_min: f64,
}

impl ElevationMask {
    /// Build from a mask elevation in radians, strictly inside `(-pi/2, pi/2)`.
    ///
    /// # Errors
    ///
    /// [`SensorError::ElevationOutOfRange`] if the mask is not strictly inside `(-pi/2, pi/2)`,
    /// where `sin` stops being invertible and the predicate stops meaning what it says.
    pub fn new(eps_min: f64) -> Result<Self, SensorError> {
        if !(eps_min > -core::f64::consts::FRAC_PI_2 && eps_min < core::f64::consts::FRAC_PI_2) {
            return Err(SensorError::ElevationOutOfRange { eps_min });
        }
        Ok(Self {
            sin_min: libm::sin(eps_min),
        })
    }

    /// The horizon itself: `eps_min = 0`.
    pub const HORIZON: Self = Self { sin_min: 0.0 };

    /// `sin(eps_min)`.
    #[must_use]
    pub const fn sin_min(self) -> f64 {
        self.sin_min
    }
}

/// `t = d·zhat` and `q = d·d`, computed **exactly as the batch kernels compute them**.
///
/// Deliberately not `Vec3::dot`, which fuses through `libm::fma`. Fusing is more accurate, and
/// inside the batch loop it costs 8x: `libm::fma` is a software routine, LLVM inlines it, and the
/// loop stops vectorising (measured — 7.48 ns/cell against 0.87). Since the kernel cannot fuse,
/// the scalar path must not either, or `covered_into` and `covered` would disagree within an ulp
/// of the boundary and no test would ever catch it.
///
/// The cost is about two ulp in `g`, i.e. ~1e-8 m, which at 7 km/s is 1e-12 s of event time
/// against a convergence threshold of 1e-6 s. Six orders of margin, spent to buy one arithmetic.
#[inline]
#[must_use]
pub(crate) fn dots(geom: &AccessGeometry) -> (f64, f64) {
    let (dx, dy, dz) = (
        geom.satellite.x - geom.cell.x,
        geom.satellite.y - geom.cell.y,
        geom.satellite.z - geom.cell.z,
    );
    let t = dx * geom.zenith.x + dy * geom.zenith.y + dz * geom.zenith.z;
    let q = dx * dx + dy * dy + dz * dz;
    (t, q)
}

/// The sign of `g` for one cell, without a square root. `t = d·zhat`, `q = d·d`.
#[inline]
#[must_use]
pub(crate) fn covered_from(t: f64, q: f64, sin_min: f64) -> bool {
    let lhs = t * t;
    let rhs = sin_min * sin_min * q;
    // `sin_min` is loop-invariant, so this branch is hoisted out of any batch loop.
    if sin_min >= 0.0 {
        t >= 0.0 && lhs >= rhs
    } else {
        t >= 0.0 || lhs <= rhs
    }
}

impl Constraint for ElevationMask {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        let (t, q) = dots(geom);
        t - self.sin_min * libm::sqrt(q)
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        let (t, q) = dots(geom);
        covered_from(t, q, self.sin_min)
    }

    /// `|dg/dt| <= (1 + |s|) * v_rel`, and `v_rel <= v_perigee + omega_frame * r_apogee`.
    ///
    /// Derived in `derivations/step_bound.mac`: `dg/dt = v·zhat - s*(d·v)/|d|`, and both terms are
    /// bounded by `|v|` through Cauchy–Schwarz because `|zhat| = 1`. Falsification-tested over
    /// 3.2e8 (state, cell) samples across 400 random orbits in
    /// `derivations/step_bound_check.m`: never violated, and tight to within 5% at the worst
    /// sample.
    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        (1.0 + libm::fabs(self.sin_min)) * orbit.max_relative_speed(frame_rate)
    }
}

/// Convenience: the elevation of `satellite` seen from `cell`, radians. Not used by the engine.
#[must_use]
pub fn elevation(satellite: Vec3, cell: Vec3, zenith: Vec3) -> f64 {
    let d = Vec3::new(
        satellite.x - cell.x,
        satellite.y - cell.y,
        satellite.z - cell.z,
    );
    libm::asin(d.dot(zenith) / d.norm())
}
