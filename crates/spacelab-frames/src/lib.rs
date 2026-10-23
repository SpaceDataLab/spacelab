//! Reference frames, in the type system.
//!
//! Every existing implementation of this — SOFA, ERFA, `sofars`, and the rest —
//! hands you a bare 3×3 matrix and trusts you to remember what it connects. The
//! two most expensive bug classes in the field follow directly: composing a chain
//! in the wrong order, and applying a transform to the wrong frame's vector.
//!
//! Both are compile-time preventable, and here they are prevented:
//!
//! ```compile_fail
//! # use spacelab_frames::{Rotation, Gcrf, Itrf, Teme};
//! # fn get<A: spacelab_frames::Frame, B: spacelab_frames::Frame>() -> Rotation<A, B> { unimplemented!() }
//! let a: Rotation<Teme, Itrf> = get();
//! let b: Rotation<Gcrf, Itrf> = get();
//! let _ = a * b; // the chain does not join: Itrf is not Gcrf
//! ```
//!
//! All of it erases at monomorphization. A [`Rotation`] is a `Mat3` and nothing
//! more.
//!
//! # Velocity is not a vector you rotate
//!
//! A [`Rotation`] applies to a [`Position`]. Only a [`Transform`] applies to a
//! [`StateVector`], because moving a velocity between rotating frames requires the
//! `ω × r` term. Forgetting it costs 465 m/s at the equator, looks plausible, and
//! is among the most repeated mistakes in the field. Here the type system will not
//! let you forget: there is no `Mul<StateVector<A>> for Rotation<A, B>`.
//!
//! # Batch shape
//!
//! Precession, nutation, Earth rotation and polar motion are functions of the epoch
//! alone — not of the satellite. Over 10⁴ satellites and 10⁵ epochs the frame chain
//! is evaluated 10⁵ times, not 10⁹, provided it is hoisted. [`teme_to_itrf_into`]
//! writes into a caller-provided slice for exactly that reason: the four orders of
//! magnitude are won by a `Vec`, before any thread or lane is considered.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

mod cio;
#[cfg(any(feature = "std", test))]
mod cio_fast;
mod polar_motion;
mod rotation;
mod teme;

pub use cio::{cirs_to_tirs, gcrf_to_cirs, gcrf_to_itrf, gcrf_to_itrf_into, tirs_to_itrf};
#[cfg(any(feature = "std", test))]
pub use cio_fast::CioInterpolator;
pub use polar_motion::{pef_to_itrf, tio_locator};
pub use rotation::{
    Cirs, Frame, Gcrf, Itrf, Pef, Position, Rotation, StateVector, Teme, Tirs, Transform, Velocity,
};
pub use teme::{EARTH_ROTATION_RATE, Orientation, teme_to_itrf, teme_to_itrf_into, teme_to_pef};
