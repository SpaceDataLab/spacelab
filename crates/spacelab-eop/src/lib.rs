//! Earth-orientation parameters, as published by the IERS.
//!
//! The Earth is not a clock and its pole is not fixed. Turning a celestial frame
//! into a terrestrial one needs three measured quantities that no theory supplies:
//! the position of the pole (`xp`, `yp`), and how far the planet's rotation has
//! drifted from atomic time (`ΔUT1`). They are observed, published daily, and
//! predicted a year ahead with rapidly decaying confidence.
//!
//! # Two things this crate refuses to do quietly
//!
//! **Predicted values are not final values.** A `ΔUT1` predicted a month ahead
//! carries millisecond error, which is metres of longitude. [`EopQuality`] is part
//! of every sample and every interpolation result, so the distinction cannot be
//! lost by accident.
//!
//! **Bulletin B is preferred over Bulletin A when it exists.** `finals2000A`
//! carries both: the rapid service (columns 17–134) and the final combined
//! solution (columns 135–185). Most implementations read only the first. Before
//! 2000 the two disagree by tens of milliarcseconds.
//!
//! # Interpolation and sub-daily corrections
//!
//! The daily series is interpolated with four-point Lagrange, as the IERS Conventions
//! prescribe (a `ΔUT1` leap-second step is undone first so it is never smeared across the
//! window). On top of that, the Conventions add diurnal and semidiurnal corrections for ocean
//! tides and lunisolar libration (`ortho_eop`, `pmsdnut2`, `utlibr`). These live in
//! [`subdaily`] and are exposed as [`subdaily_correction`] — a caller with better-than-mas
//! inputs adds them to the interpolated [`Eop`]; for TLE-scale work they are far below the
//! input error and can be ignored.

#![forbid(unsafe_code)]

mod finals;
pub mod subdaily;
mod subdaily_tables;

pub use finals::{Eop, EopError, EopQuality, EopSample, EopSeries};
pub use subdaily::{SubdailyCorrection, subdaily_correction};
