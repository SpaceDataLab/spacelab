//! Polar motion: the wobble of the crust about the rotation axis.
//!
//! What checks it: `derivations/frame_transform.mac` proves the transpose identity
//! [`pef_to_itrf`] asserts, and derives the commutator its own rejection test measures.
//! `derivations/polar_motion_rate.py` measures how fast the pole actually moves, which is what
//! [`crate::teme::teme_to_itrf`] neglects.

use spacelab_core::{Angle, Epoch, Mat3, Tt};

use crate::rotation::{Itrf, Pef, Rotation};

/// The TIO locator `s'`, the tiny angle positioning the Terrestrial Intermediate
/// Origin on the equator of the Celestial Intermediate Pole.
///
/// Dominated by the secular drift of the pole. Lambert & Bizouard (2002) show the
/// remaining terms are below a microarcsecond, so the IERS Conventions give simply
///
/// ```text
/// s' = −47 µas · t
/// ```
///
/// with `t` in Julian centuries of TT since J2000. Verified against `pyerfa`'s
/// `sp00`: identical to the last bit at every epoch tested.
///
/// A microarcsecond is three millimetres at the Earth's surface. It is included
/// because it costs one multiplication, not because anything we do yet can see it.
#[must_use]
pub fn tio_locator(tt: Epoch<Tt>) -> Angle {
    Angle::from_arcseconds(-47e-6 * tt.julian_centuries())
}

/// The polar-motion matrix, taking a pseudo-Earth-fixed vector into the ITRF.
///
/// The IERS Conventions define `W(t) = R₃(−s′)·R₂(x_p)·R₁(y_p)` as the matrix
/// carrying the ITRS into the intermediate frame. What we want is its inverse:
///
/// ```text
/// W(t)ᵀ = R₁(−y_p)·R₂(−x_p)·R₃(s′)
/// ```
///
/// # Provenance
///
/// The sign convention was not taken on faith. Four candidate orderings were built
/// and compared element-wise against `pyerfa`'s `pom00`. This one agrees to zero
/// difference in every element; the two that get a sign or a transpose wrong miss by
/// `5.9e-6`, which is twice the largest `|x_p|` or `|y_p|` in the fixture — a wrong
/// convention does not look wrong, it looks like a different pole.
///
/// The fourth candidate is the dangerous one. Keeping every factor and every sign but
/// reversing the product misses by only `6.2e-12`: rotations through microradians
/// almost commute, so the error is the commutator, second order in the angles and
/// four hundredths of a millimetre on the ground. All four are rebuilt and separated
/// by `polar_motion_rejects_the_other_three_orderings`, so these numbers are enforced
/// rather than remembered.
///
/// `derivations/frame_transform.mac` turns both halves of that paragraph into algebra.
/// The transpose identity is exact at every angle — `R₁(−y_p)·R₂(−x_p)·R₃(s′) − W(t)ᵀ`
/// simplifies to the zero matrix, not to something small. And the reversed product's
/// first-order term cancels identically; the leading miss is `x_p·y_p` in the `(1,3)`
/// element, with `∓y_p·s′` and `x_p·s′` beside it. So the number that separates the
/// dangerous candidate is a product of the two pole coordinates and does not involve
/// `s′` at all — 9e-12 at `|x_p| = |y_p| = 3 µrad`, which is where the fixture's
/// measured `6.2e-12` comes from.
#[must_use]
pub fn pef_to_itrf(xp: Angle, yp: Angle, sp: Angle) -> Rotation<Pef, Itrf> {
    Rotation::from_matrix(
        Mat3::rot1(-yp.radians()) * Mat3::rot2(-xp.radians()) * Mat3::rot3(sp.radians()),
    )
}
