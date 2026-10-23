//! Osculating Keplerian orbital elements.
//!
//! Everything else in the workspace propagates and estimates Cartesian state vectors, because that
//! is what the integrators and the automatic-differentiation Jacobians want. But the domain reasons
//! in elements: an orbit *is* its `(a, e, i, Ω, ω)`, a Sun-synchronous orbit is a statement about
//! `Ω̇`, a repeat ground track is a resonance between the nodal period and Earth's rotation. This
//! crate is the bridge — state vector to the elements that fit it, and back — so results can be
//! expressed and reasoned about in the language of the domain.
//!
//! See [`KeplerianElements`]. The osculating set is kept distinct from the mean elements an OMM
//! carries: those go through SGP4, never through here.

#![cfg_attr(not(test), no_std)]

pub mod brouwer_lyddane;
pub mod design;
pub mod eckstein_hechler;
pub mod keplerian;

pub use brouwer_lyddane::{BrouwerLyddane, BrouwerLyddaneError, MAX_CORRECTIONS};
pub use design::{
    SUN_SYNCHRONOUS_NODE_RATE, critical_inclination, frozen_eccentricity, repeat_ground_track_sma,
    sun_synchronous_inclination,
};
pub use eckstein_hechler::{EcksteinHechler, EcksteinHechlerError};
pub use keplerian::KeplerianElements;
