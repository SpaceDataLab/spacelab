//! **spacelab-slr** — satellite laser ranging, from the published normal point to a modelled range.
//!
//! An SLR normal point is the cleanest range measurement in orbit determination — millimetre
//! precision on a passive sphere — and it is only as good as the corrections applied to it. The
//! station files say which ones the publisher already removed; everything else is the modeller's.
//! This crate is the part that is not the orbit: [`crd`] reads the measurement, [`sinex`] the
//! station, [`troposphere`] the refraction the beam met on the way, [`displacement`] where the
//! station actually was at that instant rather than where its catalogue entry says, and [`range`]
//! turns the four into a modelled two-way flight time.
//!
//! The satellite ephemeris is deliberately *not* here: [`range::two_way_light_time`] takes the
//! orbit as a callback, so the same model serves a published product, a fitted arc, or a propagated
//! one without this crate depending on any of them.
//!
//! `no_std` + `libm` for the models; the two text parsers need `std`.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

pub mod displacement;
pub mod loading;
mod loading_tables;
pub mod range;
pub mod troposphere;

#[cfg(any(feature = "std", test))]
pub mod blq;

#[cfg(any(feature = "std", test))]
pub mod crd;

#[cfg(any(feature = "std", test))]
pub mod dhf;

#[cfg(any(feature = "std", test))]
pub mod sinex;

pub use displacement::{pole_tide_displacement, solid_earth_tide_displacement};
pub use loading::{OceanLoading, ocean_loading_displacement, ocean_loading_topocentric};
pub use range::{TwoWayRange, shapiro_delay, two_way_light_time};
pub use troposphere::{
    ZenithDelay, fcul_a_mapping, mendes_pavlis_zenith_delay, water_vapour_pressure,
};
