//! Constellation design — generators that turn a family's parameters into the circular Keplerian
//! element sets a coverage or conjunction analysis consumes.
//!
//! spacelab has a coverage engine (`spacelab_coverage`) and a collision-probability engine, but no
//! way to *build* a standard constellation to feed them. This crate closes that loop: a family
//! generator emits [`spacelab_elements::KeplerianElements`] (plane-major), which the caller
//! propagates and hands to the coverage / conjunction machinery — *generate → propagate → coverage /
//! revisit / conjunctions*.
//!
//! Generators are `no_std` and allocation-free: each satellite is delivered through a `FnMut`
//! callback (as `spacelab_coverage` and the halo families do), so the caller decides whether to
//! collect into a `Vec`, stream, or filter.
//!
//! Families:
//! - [`Walker`] — Delta/Star, the standard `i: T/P/F` symmetric pattern (Walker 1984).
//! - [`StreetOfCoverage`] — near-polar streets tiled for continuous coverage (Lüders / Adams–Rider).
//! - [`FlowerConstellation`] — satellites on one repeating space track (Mortari et al. 2004).
//! - [`LatticeFlower`] — the 2-D integer-lattice unification of Walker and Flower (Avendaño, Davis &
//!   Mortari 2013).
//! - [`KinematicallyRegular`] — a constellation as the orbit of a seed under a finite rotation group
//!   (Mozhaev): cyclic (= Walker), dihedral, and tetrahedral symmetry.
//! - [`DraimTetrahedral`] — the four-satellite continuous global coverage constellation (Draim 1987).
//! - [`TwoTierShell`] — two-tier surveillance of a spherical shell of near-Earth space (SSA), with the
//!   space-visibility geometry and the launch-cost objective.
//!
//! Plus the Mozhaev classification of symmetry groups ([`PointGroup`], [`MozhaevSymbol`]) and, behind
//! the `std` feature, the α-characteristic coverage figure of merit ([`alpha_characteristic`]).

#![cfg_attr(not(any(test, feature = "std")), no_std)]

#[cfg(any(feature = "std", test))]
pub mod analysis;
pub mod draim;
pub mod flower;
pub mod lattice;
pub mod mozhaev;
pub mod shell;
pub mod soc;
pub mod symmetric;
pub mod walker;

#[cfg(any(feature = "std", test))]
pub use analysis::{alpha_characteristic, minimum_walker};
pub use draim::DraimTetrahedral;
pub use flower::FlowerConstellation;
pub use lattice::LatticeFlower;
pub use mozhaev::{GroupType, MozhaevSymbol, PointGroup};
pub use shell::{ShellTier, TwoTierShell, characteristic_velocity, observes};
pub use soc::{SocGeometry, StreetOfCoverage};
pub use symmetric::{KinematicallyRegular, Symmetry};
pub use walker::{Walker, WalkerPattern};
