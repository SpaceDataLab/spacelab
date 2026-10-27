//! Clean-room implementation of the JB2008 thermospheric density model, written from the
//! published papers only:
//!
//! - Bowman et al. (2008), "A New Empirical Thermospheric Density Model JB2008 Using New Solar
//!   and Geomagnetic Indices", AIAA 2008-6438 (B08);
//! - Bowman et al. (2008), JASTP 70, 774-793, JB2006 (B06);
//! - Jacchia (1970), SAO Special Report 313 (SR313); Jacchia (1971), SAO Special Report 332
//!   (SR332).
//!
//! The base is Jacchia's static diffusion model in its 1970 edition, with the 1971 elements
//! named in `constants`. Every number carries its source there and in `PROVENANCE.md` beside this module.
//! Entry point: [`evaluate`].
//!
//! Gates: `tests/jacchia_bowman_g1_g2_static.rs`, `tests/jacchia_bowman_g2_model.rs`,
//! `tests/jacchia_bowman_g5_sr313_table5.rs` and `tests/jacchia_bowman_g5_sr313_examples.rs`.

// Written in a clean room from the papers (`plans/17` Ф18.A′ step 1) and moved here unchanged:
// literals stay digit for digit as printed so `PROVENANCE.md` can be read against the page, and the
// prose keeps the papers' notation.
#![allow(
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc
)]

pub mod calendar;
pub mod constants;
pub mod model;
pub mod profile;
pub mod quadrature;
pub mod static_model;
pub mod variations;

pub use model::{Error, Input, Output, Terms, evaluate};
pub use profile::Profile;
pub use static_model::{NumberDensities, StaticState, static_state, static_state_with_step};
pub use variations::SolarIndices;
