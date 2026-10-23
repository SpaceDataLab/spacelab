//! **spacelab-geometry** — convex hulls, Delaunay triangulations and spherical Voronoi diagrams.
//!
//! - [`predicates`] — the exact orientation and in-circle tests every other module is built on.
//! - [`hull`] — convex hulls in 2-D (monotone chain) and 3-D (incremental).
//! - [`delaunay`] — Bowyer–Watson incremental Delaunay triangulation in 2-D.
//! - [`sphere`] — spherical Delaunay and its dual Voronoi diagram, from the 3-D hull of unit vectors.
//!
//! `no_std` + `alloc` + `libm`, no external dependencies.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

pub mod delaunay;
pub mod hull;
pub mod predicates;
pub mod sphere;

pub use delaunay::Triangulation;
pub use hull::{ConvexHull, convex_hull_2d};
pub use sphere::{VoronoiCell, spherical_delaunay, spherical_voronoi};

/// Why a construction was refused. Every variant is a statement about the *input*: these algorithms
/// have no tolerance to tune and no iteration to fail to converge, so the only way they do not produce
/// an answer is that the input does not have one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryError {
    /// Fewer points than the construction needs: three for a 2-D triangulation or hull, four for a
    /// 3-D hull. The field is how many distinct points were supplied.
    TooFewPoints(usize),
    /// A coordinate is infinite or NaN. Every predicate here is exact for finite `f64` and meaningless
    /// otherwise, so this is refused at the boundary rather than propagated.
    NonFiniteCoordinate {
        /// Index of the offending point in the caller's slice.
        point: usize,
    },
    /// The points are all collinear (2-D) or all coplanar (3-D): they span too few dimensions for the
    /// construction to have an interior.
    Degenerate,
    /// A direction on the sphere has zero length, so it names no point on it.
    ZeroLengthDirection {
        /// Index of the offending direction in the caller's slice.
        point: usize,
    },
}

impl core::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewPoints(count) => write!(f, "{count} distinct points are not enough"),
            Self::NonFiniteCoordinate { point } => {
                write!(f, "point {point} has a non-finite coordinate")
            }
            Self::Degenerate => write!(f, "the points span too few dimensions"),
            Self::ZeroLengthDirection { point } => {
                write!(f, "direction {point} has zero length")
            }
        }
    }
}

impl core::error::Error for GeometryError {}
