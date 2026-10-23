//! Access-window detection.
//!
//! A constraint's coverage of a cell is a continuous scalar `g(t)`; an access window is the
//! interval where `g(t) ≥ 0`. This crate finds those intervals the way the architecture demands
//! and the way GMAT, Gpredict and PreviSat do not: bracket a sign change of `g` on a coarse grid
//! whose step is *derived* from a bound on `|dg/dt|`, then refine each root with a
//! bracket-preserving Brent to an abscissa tolerance in seconds.
//!
//! The coarse step is the guarantee against missing a short pass: if it is smaller than the
//! shortest possible access, no sign change can hide between samples. That bound comes from
//! `spacelab_sensors::Constraint::max_rate`, which is derived and falsification-tested, not fitted.

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(clippy::doc_markdown)] // program names and math notation, not identifiers

pub mod access;
#[cfg(any(feature = "std", test))]
pub mod combine;
pub mod detectors;
pub mod event;
pub mod fom;
pub mod metrics;
pub mod observability;
#[cfg(any(feature = "std", test))]
pub mod parallel;
pub mod photometry;
pub mod root;

pub use access::{Window, access_windows, windows};
#[cfg(any(feature = "std", test))]
pub use combine::{combined_metrics, merge_windows};
pub use detectors::{
    CircularFovDetector, EARTH_EQUATORIAL_RADIUS, EclipseDetector, EclipseKind, ElevationDetector,
    SUN_RADIUS,
};
pub use event::{Event, EventKind, find_events};
pub use fom::{ConstellationFom, constellation_fom};
pub use metrics::{CoverageMetrics, metrics};
pub use observability::{
    AirmassModel, ApparentAltitude, DarknessDetector, DiurnalArc, MoonSeparationDetector,
    Refraction, StandardAltitude, TrueAltitude, Twilight, diurnal_arc,
};
#[cfg(any(feature = "std", test))]
pub use parallel::revisit_map;
pub use photometry::{
    DetectabilityDetector, ObservationGeometry, OpticalSensor, PhaseFunction, SkyBackground,
    StandardMagnitude,
};
pub use root::{Bracket, RootError, brent};
