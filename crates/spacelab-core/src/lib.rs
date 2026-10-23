//! Typed epochs, angles and units.
//!
//! The two most expensive bug classes in this domain are confusing time scales
//! (UTC / TT / UT1) and confusing reference frames (TEME / ITRF / GCRF). Both are
//! preventable at compile time. Nothing in this crate carries a bare `f64` that
//! means something.

#![cfg_attr(not(test), no_std)]

pub mod calendar;
pub mod linalg;
pub mod time;

pub use calendar::{CalendarError, UtcDateTime};
pub use linalg::{Mat3, Vec3};
pub use time::{Epoch, Tai, Tdb, TimeScale, Tt, Ut1};

/// An angle, stored in radians.
///
/// Degrees exist only at the boundary, where a human reads or writes them.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Angle(f64);

impl Angle {
    /// Full turn, in radians.
    pub const TAU: f64 = core::f64::consts::TAU;

    /// Construct from radians.
    #[must_use]
    pub const fn from_radians(r: f64) -> Self {
        Self(r)
    }

    /// Construct from degrees.
    #[must_use]
    pub fn from_degrees(d: f64) -> Self {
        Self(d * (core::f64::consts::PI / 180.0))
    }

    /// The angle in radians.
    #[must_use]
    pub const fn radians(self) -> f64 {
        self.0
    }

    /// The angle in degrees.
    #[must_use]
    pub fn degrees(self) -> f64 {
        self.0 * (180.0 / core::f64::consts::PI)
    }

    /// Construct from seconds of arc.
    ///
    /// The unit Earth-orientation data arrives in, and the one in which
    /// milliarcseconds of pole wander are quoted.
    #[must_use]
    pub fn from_arcseconds(arcsec: f64) -> Self {
        Self(arcsec * (core::f64::consts::PI / (180.0 * 3600.0)))
    }

    /// The angle in seconds of arc.
    #[must_use]
    pub fn arcseconds(self) -> f64 {
        self.0 * ((180.0 * 3600.0) / core::f64::consts::PI)
    }
}

/// A length, stored in metres.
///
/// Kilometres exist only at the boundary. Mixing the two silently is a
/// three-order-of-magnitude error that looks plausible on a plot.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Length(f64);

impl Length {
    /// Construct from metres.
    #[must_use]
    pub const fn from_metres(m: f64) -> Self {
        Self(m)
    }

    /// Construct from kilometres.
    #[must_use]
    pub fn from_kilometres(km: f64) -> Self {
        Self(km * 1000.0)
    }

    /// The length in metres.
    #[must_use]
    pub const fn metres(self) -> f64 {
        self.0
    }

    /// The length in kilometres.
    #[must_use]
    pub fn kilometres(self) -> f64 {
        self.0 / 1000.0
    }
}

/// A NORAD catalog number.
///
/// A `u32` rather than a `u16`: the catalog has outgrown the five-digit range
/// that the legacy TLE format can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NoradId(u32);

impl NoradId {
    /// Construct from the raw catalog number.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The raw catalog number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}
