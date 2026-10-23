//! The WGS-84 ellipsoid.
//!
//! Converting a Cartesian position into latitude, longitude and height is the last
//! step of every ground track and the first step of every coverage calculation. It
//! is also a quartic equation, and the literature is full of ways to solve it.
//!
//! # Choice of algorithm, and why it lets us check ourselves
//!
//! We use **Vermeille's closed form**: an exact algebraic solution, no iteration,
//! no convergence criterion, the same number of operations for every point.
//!
//! ERFA uses **Fukushima's method**, which is iterative and accelerated by Halley's
//! rule. That is precisely the point. Two implementations of the *same* algorithm
//! agree because they were transcribed from each other. Two *different* algorithms
//! agreeing to `2.6×10⁻¹¹` rad — 0.17 mm on the surface — is evidence that both are
//! solving the equation they claim to solve.
//!
//! This is the first place in the project where the oracle is of a genuinely
//! different lineage. See `plans/04-validation-and-risks.md` §1.2.
//!
//! # References
//!
//! Vermeille, H. (2002), *Direct transformation from geocentric coordinates to
//! geodetic coordinates*, Journal of Geodesy 76, 451–454.

#![cfg_attr(not(test), no_std)]

mod wgs84;

pub use wgs84::{
    EQUATORIAL_RADIUS, FLATTENING, Geodetic, GeodeticError, POLAR_RADIUS, geodetic_nadir,
    geodetic_to_itrf, itrf_to_geodetic, itrf_to_geodetic_into,
};
