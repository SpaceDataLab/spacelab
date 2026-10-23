//! CCSDS Navigation Data Messages: the file formats every operator interface actually speaks.
//!
//! An orbit product nobody else can read is not a product. CCSDS 502.0-B-3 (orbit data) and
//! 508.0-B-1 (conjunction data) are what GMAT, Orekit, NEPTUNE and every operations centre exchange,
//! and this crate implements the three that carry work in and out of this workspace:
//!
//! | Message | Direction | What it is |
//! |---|---|---|
//! | [`Opm`] | read and write | one Cartesian state, with optional elements, covariance and planned manoeuvres |
//! | [`Oem`] | read and write | a segmented ephemeris — the operator handover format |
//! | [`Cdm`] | read | a conjunction screening: two states, two covariances, one TCA |
//!
//! ADM, TDM and OCM are deliberately absent. Nothing in this workspace consumes them yet, and OCM in
//! particular is a large standard whose cost is not worth paying before something asks.
//!
//! # Where the OMM is
//!
//! The fourth ODM, the Orbit Mean-Elements Message, lives in `spacelab-sgp4` rather than here,
//! because the element set it carries *is* an SGP4 input and travels with the propagator that reads
//! it. That splits the CCSDS family across two crates, which is a real cost; the alternative was to
//! make this crate depend on a propagator, which is a worse one.
//!
//! # Units, and the one place a round trip is not exact
//!
//! Everything a reader returns is SI: metres, metres per second, radians, kilograms, seconds. CCSDS
//! states positions in kilometres and angles in degrees, and both conversions happen at the parse.
//!
//! That has a consequence worth stating rather than discovering. `parse(write(parse(text)))` returns
//! the middle parse **bit for bit** in every field scaled by a power of ten — position, velocity,
//! covariance, `GM` — because those values came from a decimal file and `x / 1000 * 1000` is the
//! identity on a double that was itself produced by scaling a decimal. It does *not* for the
//! Keplerian angles, because `π / 180` is not a power of ten and the composition is a rounding, not
//! a bijection. The round-trip test asserts exactly that split: exact where exactness is available,
//! and a stated bound of a few ulp where it is not.
//!
//! # Errors name the keyword, never the text
//!
//! These parsers read third-party files that arrive over a network. [`NdmError`]
//! carries a `&'static str` keyword from this crate's own table and an integer line number, and
//! never the offending bytes — an error message is a place untrusted input ends up in a log.

pub mod cdm;
pub mod kvn;
pub mod oem;
pub mod opm;

pub use cdm::{Cdm, CdmObject, parse_cdm};
pub use kvn::{COVARIANCE_ENTRIES, NdmError, symmetric_from_lower};
pub use oem::{EphemerisRecord, Oem, OemCovariance, OemMetadata, OemSegment, parse_oem, write_oem};
pub use opm::{Anomaly, Covariance, Keplerian, Maneuver, Opm, Spacecraft, parse_opm, write_opm};
