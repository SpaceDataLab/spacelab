//! Fundamental astronomy.
//!
//! # Provenance
//!
//! Every routine here is derived from published mathematics — the IERS
//! Conventions, the IAU resolutions, and the primary literature — and never
//! transcribed from another implementation's source. Each function names the
//! equation it implements.
//!
//! This is not only a licence rule. Transcribing a C implementation reproduces a
//! C API: out-parameters, integer status codes, reference frames tracked in the
//! programmer's head. The typed algebra that justifies this crate's existence is
//! reachable only by deriving from the mathematics.
//!
//! See `plans/07-symbolic-derivation.md`.

#![cfg_attr(not(test), no_std)]

pub mod cio;
mod cio_tables;
pub mod earth_rotation;
pub mod fundamental;
pub mod leap_seconds;
mod nut_tables;
pub mod nutation;
pub mod tdb;
mod tdb_tables;

pub use cio::{cio_locator_s, cip_xy, era};
pub use earth_rotation::{equinox_kinematic_1982, gmst82};
pub use leap_seconds::{LeapError, delta_at, tai_to_utc, utc_to_tai};
pub use nutation::{nutation_00a, nutation_06a};
pub use tdb::{tdb_minus_tt, tdb_to_tt, tt_to_tdb};
