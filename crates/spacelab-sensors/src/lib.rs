//! Coverage constraints.
//!
//! A constraint is a continuous scalar function `g(satellite, cell, sensor)` whose sign says
//! whether the cell is covered and whose zeros are the boundary of an access window. Nothing
//! here returns `bool` where a `f64` will do: a boolean predicate cannot be bracketed, and a
//! root finder needs a sign change to bracket.
//!
//! The geometry is derived in `derivations/footprint.mac` and independently confirmed in
//! `derivations/footprint.input` (`FriCAS`). Both establish that the closed-form footprint radius
//! implied by `g = 0` is identical to the one published in `SaVi`, and that `SaVi`'s separate
//! "horizon-limited" branch is not a branch at all: it is the disappearance of a real root, and
//! it falls out of `Cone ∧ Elevation(0)` without a special case.
//!
//! # Two functions, not one
//!
//! [`Constraint::g`] is continuous and is what Brent refines. It costs a square root.
//!
//! [`Constraint::covered`] answers only the sign. It is exact — bit-for-bit the same decision as
//! `g(..) >= 0`, verified in `derivations/predicate_identical.m` over ±2000 ulp of the boundary
//! at five altitudes — and it costs no square root, no division and no trigonometry.
//!
//! That distinction is the whole performance story. `covered` runs once per
//! `(satellite, time, candidate cell)`; `g` runs a few dozen times per event. `Orekit` states the
//! underlying licence explicitly: `g` need only be continuous, correctly signed everywhere, and
//! *accurate near its roots*.

#![cfg_attr(not(test), no_std)]

pub mod agile;
pub mod comms;
pub mod cone;
pub mod cull;
pub mod elevation;
pub mod ellipsoid_grid;
pub mod geoloc;
pub mod grid;
pub mod kernel;
pub mod sar;
pub mod scan;

pub use agile::AgileOptical;
pub use comms::{CommsAntenna, CommsBudget};
pub use cone::{Boresight, Cone};
pub use cull::{CullBody, Reach, cap_cosine, cull_into, reachable_band};
pub use elevation::ElevationMask;
pub use ellipsoid_grid::{EllipsoidGrid, WGS84_GAMMA};
pub use geoloc::{
    Attitude, Ellipsoid, WGS84, geodetic_latitude, geolocate, intersect_ellipsoid, line_of_sight,
    nadir_direction, rotate_about, subpoint,
};
pub use grid::CellGrid;
pub use sar::{SideLookingSAR, look_side};
pub use scan::{Instrument, ScanSample};

use spacelab_core::Vec3;

/// The geometry a constraint needs, in a body-fixed frame.
///
/// `satellite` and `cell` share a frame; `zenith` is the outward unit normal of the ellipsoid at
/// `cell`, which on the WGS-84 ellipsoid is *not* parallel to `cell`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccessGeometry {
    /// Satellite position.
    pub satellite: Vec3,
    /// Cell centre on the reference ellipsoid.
    pub cell: Vec3,
    /// Outward unit normal of the ellipsoid at `cell`.
    pub zenith: Vec3,
}

impl AccessGeometry {
    /// Line of sight from the cell to the satellite.
    #[must_use]
    pub fn line_of_sight(&self) -> Vec3 {
        Vec3::new(
            self.satellite.x - self.cell.x,
            self.satellite.y - self.cell.y,
            self.satellite.z - self.cell.z,
        )
    }
}

/// A continuous scalar constraint on access geometry.
///
/// # Contract
///
/// * `g` is continuous everywhere the geometry is defined.
/// * `g(x) >= 0` exactly when the cell is covered.
/// * `covered(x) == (g(x) >= 0.0)` for every `x`. Implementations that cannot guarantee this
///   must not override `covered`.
///
/// The sign must carry meaning — *in* or *out* — or the [`And`] and [`Or`] combinators are
/// nonsense. This is `Orekit`'s caveat on `BooleanDetector`, and it is a real one.
pub trait Constraint {
    /// The continuous switching function. Zero on the boundary, positive inside.
    fn g(&self, geom: &AccessGeometry) -> f64;

    /// The sign of [`Constraint::g`], by whatever means is cheapest.
    ///
    /// The default defers to `g`. Implementations override it when they can decide the sign
    /// without the square root that `g` needs.
    fn covered(&self, geom: &AccessGeometry) -> bool {
        self.g(geom) >= 0.0
    }

    /// An upper bound on `|dg/dt|` given the satellite's semi-major axis, eccentricity and `mu`,
    /// plus the body-fixed frame's rotation rate.
    ///
    /// The coarse search step is `remaining_g / max_rate`: an interval `g` cannot traverse, so
    /// no sign change can hide inside it. This is the bound the whole event detector rests on,
    /// and it is derived, never fitted. See `derivations/step_bound.mac`.
    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64;
}

/// The orbit facts a rate bound needs. Keplerian, and deliberately crude: the bound must hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitBound {
    /// Semi-major axis, metres.
    pub semi_major_axis: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

impl OrbitBound {
    /// Perigee radius, `a(1-e)`.
    #[must_use]
    pub fn perigee_radius(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity)
    }

    /// Apogee radius, `a(1+e)`.
    #[must_use]
    pub fn apogee_radius(&self) -> f64 {
        self.semi_major_axis * (1.0 + self.eccentricity)
    }

    /// Maximum inertial speed, attained at perigee: `sqrt(mu (1+e) / (a(1-e)))`.
    #[must_use]
    pub fn perigee_speed(&self) -> f64 {
        libm::sqrt(self.mu * (1.0 + self.eccentricity) / self.perigee_radius())
    }

    /// Maximum angular rate about the body centre, attained at perigee: `v_p / r_p`.
    #[must_use]
    pub fn max_angular_rate(&self) -> f64 {
        self.perigee_speed() / self.perigee_radius()
    }

    /// Upper bound on the satellite's speed *in a frame rotating at `frame_rate`*.
    ///
    /// `|v - omega x r| <= |v| + |omega| |r| <= v_perigee + frame_rate * r_apogee`. Both maxima
    /// are attained at different points of the orbit, so the bound is never tight — but it is a
    /// bound, which is the only property the coarse step may rely on.
    #[must_use]
    pub fn max_relative_speed(&self, frame_rate: f64) -> f64 {
        self.perigee_speed() + libm::fabs(frame_rate) * self.apogee_radius()
    }
}

/// Why a sensor could not be constructed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorError {
    /// The mask elevation is outside `(-pi/2, pi/2)`, where `sin` stops being invertible.
    ElevationOutOfRange {
        /// The offending value, radians.
        eps_min: f64,
    },
    /// The cone half-angle is outside `(0, pi/2]`.
    HalfAngleOutOfRange {
        /// The offending value, radians.
        half_angle: f64,
    },
    /// A supplied boresight has no length or is not finite, so there is no axis to build a cone
    /// about. Refused rather than normalised, which would give `NaN` and no error.
    DegenerateBoresight {
        /// The offending direction's norm.
        norm: f64,
    },
    /// The SAR incidence band is empty or leaves `[0, pi/2)` — need `0 <= near < far < pi/2`.
    IncidenceBandOutOfRange {
        /// Near-range incidence, radians.
        near: f64,
        /// Far-range incidence, radians.
        far: f64,
    },
    /// The communications link margin is negative even at zenith: no access at any elevation.
    LinkNeverCloses,
    /// The six coordinate slices of a [`CellGrid`] have different lengths.
    RaggedGrid {
        /// Length of the shortest slice.
        shortest: usize,
        /// Length of the longest slice.
        longest: usize,
    },
    /// An output buffer does not match the grid it is written through.
    OutputLengthMismatch {
        /// Number of cells in the grid.
        cells: usize,
        /// Length of the buffer supplied.
        buffer: usize,
    },
}

impl core::fmt::Display for SensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ElevationOutOfRange { eps_min } => {
                write!(
                    f,
                    "mask elevation {eps_min} rad is not inside (-pi/2, pi/2)"
                )
            }
            Self::HalfAngleOutOfRange { half_angle } => {
                write!(
                    f,
                    "cone half-angle {half_angle} rad is not inside (0, pi/2]"
                )
            }
            Self::DegenerateBoresight { norm } => {
                write!(f, "boresight has norm {norm}, so it defines no axis")
            }
            Self::IncidenceBandOutOfRange { near, far } => {
                write!(
                    f,
                    "SAR incidence band [{near}, {far}] rad is not 0 <= near < far < pi/2"
                )
            }
            Self::LinkNeverCloses => {
                write!(f, "communications link margin is negative even at zenith")
            }
            Self::RaggedGrid { shortest, longest } => {
                write!(
                    f,
                    "cell grid slices differ in length: {shortest}..{longest}"
                )
            }
            Self::OutputLengthMismatch { cells, buffer } => {
                write!(f, "grid has {cells} cells but the buffer holds {buffer}")
            }
        }
    }
}

impl core::error::Error for SensorError {}

/// Logical AND of two constraints: `min(g1, g2)`.
///
/// Its zeros are exactly the boundary of the intersection. Composition happens *before*
/// root-finding, so an `And` costs one bracket and one Brent rather than two window sets to
/// intersect.
#[derive(Debug, Clone, Copy)]
pub struct And<A, B>(pub A, pub B);

/// Logical OR of two constraints: `max(g1, g2)`.
#[derive(Debug, Clone, Copy)]
pub struct Or<A, B>(pub A, pub B);

/// Logical NOT of a constraint: `-g`.
#[derive(Debug, Clone, Copy)]
pub struct Not<A>(pub A);

impl<A: Constraint, B: Constraint> Constraint for And<A, B> {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        self.0.g(geom).min(self.1.g(geom))
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        self.0.covered(geom) && self.1.covered(geom)
    }

    /// The composite's rate bound is the **maximum** of its components', because the composite's
    /// step must be the **minimum** of the steps each component would have chosen. An `AND` of a
    /// fast elevation constraint and a slow eclipse constraint, stepped at the eclipse
    /// constraint's leisure, silently loses short passes.
    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        self.0
            .max_rate(orbit, frame_rate)
            .max(self.1.max_rate(orbit, frame_rate))
    }
}

impl<A: Constraint, B: Constraint> Constraint for Or<A, B> {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        self.0.g(geom).max(self.1.g(geom))
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        self.0.covered(geom) || self.1.covered(geom)
    }

    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        self.0
            .max_rate(orbit, frame_rate)
            .max(self.1.max_rate(orbit, frame_rate))
    }
}

impl<A: Constraint> Constraint for Not<A> {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        -self.0.g(geom)
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        // Deliberately not `!self.0.covered(geom)`: the boundary belongs to `g >= 0`, and
        // negating the sign moves it. `-0.0 >= 0.0` is true, so `Not` on a cell exactly on the
        // boundary reports covered, which matches `-g >= 0`.
        -self.0.g(geom) >= 0.0
    }

    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        self.0.max_rate(orbit, frame_rate)
    }
}
