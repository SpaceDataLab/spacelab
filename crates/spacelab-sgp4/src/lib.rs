//! SGP4 propagation, typed and shaped for many satellites at once.
//!
//! # Why we do not write our own
//!
//! The `sgp4` crate is a pure-Rust port of the reference Celestrak implementation,
//! validated against Vallado's published test vectors, MIT-licensed and maintained.
//! Rewriting it would buy no differentiation, and SGP4 is a model whose every quirk
//! *is* its specification: a "better" implementation is a wrong one.
//!
//! The consequence for verification is worth stating plainly, because it is easy to
//! overclaim. Every SGP4 implementation in circulation — the `sgp4` crate,
//! `python-sgp4`, Skyfield, gpredict, `PreviSat` — descends from Vallado's C++ or from
//! Spacetrack Report #3 before it. Agreement between any two of them establishes that
//! both transcribed the source correctly. It cannot establish that the model is right,
//! and there is no independent implementation with which to try: for SGP4, the
//! specification *is* a particular listing.
//!
//! What this crate adds is everything around it. Kilometres become metres. Bare
//! `[f64; 3]` becomes `StateVector<Teme>`, which cannot be handed to a
//! ground-station calculation without passing through a frame transform. And the
//! batch entry points have the shape the hot loop will want.
//!
//! When a profile eventually demands a structure-of-arrays kernel, it replaces the
//! inside of [`propagate_many_into`] and nothing else. Strangler fig, not rewrite.
//!
//! # On leap seconds, and why ignoring them is correct here
//!
//! A TLE epoch is a UTC *label*, and the elements were fitted by software that
//! subtracts UTC labels without regard for leap seconds. Propagating with a uniform
//! time scale would be more defensible in the abstract and less accurate in fact:
//! it would disagree with the convention the elements encode.
//!
//! So we propagate on the naive coordinate — every UTC day exactly 86400 seconds —
//! which is precisely what [`UtcDateTime::naive_ns_since_j2000`] already provides.
//! This is a deliberate agreement with the model, not an oversight.
//!
//! [`UtcDateTime::naive_ns_since_j2000`]: spacelab_core::UtcDateTime::naive_ns_since_j2000

#![forbid(unsafe_code)]

pub mod omm;
mod satellite;
pub mod tle;

pub use omm::{OmmError, parse_omm_csv, parse_omm_kvn};
pub use satellite::{Convention, PropagationError, Regime, Satellite, propagate_many_into};
/// The element-set type this crate is built on, and its classification marking.
///
/// Re-exported because they are in this crate's own signatures — [`parse_omm_csv`] returns them,
/// [`TleModel`] and [`TleBuilder`] take them — and a consumer should not have to name the vendor
/// crate to spell the type of a value this one handed it.
pub use sgp4::{Classification, Elements};
pub use tle::{
    MAX_CATALOG_NUMBER, TLE_FIT_STEPS, TLE_FIT_STEPS_WITH_DRAG, TLE_LINE_LENGTH, Tle, TleBuilder,
    TleError, TleModel, seed_elements,
};
