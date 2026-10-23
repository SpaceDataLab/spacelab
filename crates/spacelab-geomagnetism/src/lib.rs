//! **spacelab-geomagnetism** — Earth's core magnetic field, and the coordinates built on it.
//!
//! [`igrf`] is the full degree/order-13 International Geomagnetic Reference Field; [`dipole`] is the
//! centred-dipole approximation to its first degree, and the magnetic latitude and magnetic local
//! time that empirical upper-atmosphere models are organized in.
//!
//! Two unrelated parts of this workspace need the geomagnetic field, and neither is about the other:
//! the higher-order ionospheric delay in `spacelab-gnss` needs the field vector along a signal path,
//! and the high-latitude thermospheric wind in `spacelab-propagation` needs magnetic coordinates,
//! because the auroral ion-drag jets are organized in magnetic local time and the dipole axis is
//! tilted about 11° from the rotation axis. This crate is the shared floor under both, so neither
//! has to depend on the other and the Gauss coefficients exist once.
//!
//! `no_std` + `libm`.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

pub mod dipole;
pub mod igrf;
pub(crate) mod igrf14_coeffs;
