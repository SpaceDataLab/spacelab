//! TEME to ITRF: the transform SGP4 forces on us, and nobody standardises.
//!
//! What checks it: `derivations/frame_transform.mac` proves the algebra — that `rot3` fixes `ẑ`, so
//! the single `ω·ẑ` this module attaches to a two-rotation chain is exact for the sidereal half —
//! and `derivations/polar_motion_rate.py` measures the half it is not, against the whole IERS
//! series. See [`teme_to_itrf`]'s "What is neglected".

use spacelab_core::{Angle, Epoch, Mat3, Tt, Ut1, Vec3};
use spacelab_fa::{equinox_kinematic_1982, gmst82};

use crate::polar_motion::{pef_to_itrf, tio_locator};
use crate::rotation::{Itrf, Pef, Rotation, Teme, Transform};

/// Nominal mean angular velocity of the Earth, rad/s.
///
/// The IERS conventional value. The instantaneous rate differs by the excess length
/// of day: `ω = ω₀·(1 − LOD/86400)`, and LOD reaches a couple of milliseconds, so
/// the correction is order `2×10⁻⁸` relative — a few micrometres per second at the
/// equator. [`teme_to_itrf`] applies it from the `lod_seconds` the caller supplies;
/// passing `0` recovers the nominal rate.
pub const EARTH_ROTATION_RATE: f64 = 7.292_115_146_706_979e-5;

/// Sidereal rotation from TEME into the pseudo-Earth-fixed frame.
///
/// A single rotation about the pole, by Greenwich Mean Sidereal Time 1982 plus the kinematic
/// terms of the 1982 equation of the equinoxes ([`equinox_kinematic_1982`]): Eq. (C-1) of Vallado,
/// Crawford, Hujsak and Kelso, "Revisiting Spacetrack Report #3", AIAA 2006-6753. TEME is not
/// defined by any standards body — it is whatever SGP4 emits — and that paper is the definition
/// the literature attaches to it.
///
/// The same paper's Eq. (1) rotates by GMST alone, and its Appendix C example follows (C-1); the
/// authors confirmed (2026-10-28) that (C-1) is the intended transform. The two differ by
/// 0.0027″ at most, 8 cm at LEO radius. `tt` is the time argument of the lunar node in those terms.
#[must_use]
pub fn teme_to_pef(ut1: Epoch<Ut1>, tt: Epoch<Tt>) -> Rotation<Teme, Pef> {
    let angle = gmst82(ut1).radians() + equinox_kinematic_1982(tt).radians();
    Rotation::from_matrix(Mat3::rot3(angle))
}

/// The full TEME to ITRF transform, rotation and angular velocity together.
///
/// ```text
/// TEME ──GMST82 + EqEq82_kin──▶ PEF ──polar motion──▶ ITRF
/// ```
///
/// The angular velocity is `ω·ẑ` expressed in TEME. Because the first rotation is
/// about `ẑ`, that vector is carried unchanged into PEF, which is what makes the
/// velocity transform in [`Transform::state`] correct there.
///
/// The rate is the instantaneous one, `ω₀·(1 − LOD/86400)`, from the excess length of
/// day `lod_seconds`. Pass `0` for the nominal rate.
///
/// # What is neglected
///
/// The rate of polar motion, and that is the *only* kinematic term dropped:
/// `derivations/frame_transform.mac` proves `rot3` fixes `ẑ`, so the sidereal axis
/// survives the first rotation unchanged and `ω·ẑ` in TEME is exact for it. What is
/// left over is the second rotation's own rate.
///
/// Its size is measured against the IERS series rather than inferred from the
/// wobble's amplitude — `derivations/polar_motion_rate.py`, 19 976 daily rows over
/// 54.7 years: the pole's angular rate is **1.3×10⁻¹³ rad/s** in the median and
/// **3.1×10⁻¹³** at its worst, so **2 µm/s** at the equator against the sidereal
/// term's 465 m/s. `ω` is 2.3×10⁸ times the worst of it. The wobble itself is
/// 0.57 arcsec peak to peak with a secular drift of 0.003 arcsec a year, and it is
/// the 433-day Chandler beat that keeps the *rate* this small. Vallado's reduction
/// neglects it too.
#[must_use]
pub fn teme_to_itrf(
    ut1: Epoch<Ut1>,
    tt: Epoch<Tt>,
    xp: Angle,
    yp: Angle,
    lod_seconds: f64,
) -> Transform<Teme, Itrf> {
    let sp = tio_locator(tt);
    let rotation: Rotation<Teme, Itrf> = pef_to_itrf(xp, yp, sp) * teme_to_pef(ut1, tt);
    // LOD is the day's excess over 86400 SI seconds, so that is the exact denominator.
    let omega = EARTH_ROTATION_RATE * (1.0 - lod_seconds / 86_400.0);
    Transform::new(rotation, Vec3::new(0.0, 0.0, omega))
}

/// Earth orientation at one epoch: everything [`teme_to_itrf`] needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation {
    /// Universal time, for sidereal rotation.
    pub ut1: Epoch<Ut1>,
    /// Terrestrial time, for the TIO locator.
    pub tt: Epoch<Tt>,
    /// Pole position along the IERS meridian.
    pub xp: Angle,
    /// Pole position, ninety degrees west.
    pub yp: Angle,
    /// Excess length of day, seconds, for the instantaneous rotation rate.
    pub lod_seconds: f64,
}

/// Build a transform for each epoch, writing into a caller-provided slice.
///
/// # Why this signature
///
/// The frame chain is a function of the epoch alone — not of any satellite. In a
/// coverage run over 10⁴ objects and 10⁵ epochs it must be evaluated 10⁵ times,
/// not 10⁹. Hoisting it into a slice, once, wins four orders of magnitude before a
/// single thread or SIMD lane is considered, and no amount of GPU offload would
/// have recovered them.
///
/// Writing into `&mut [Transform<..>]` rather than returning a `Vec` keeps the
/// allocation with the caller and lets Rust's non-aliasing guarantee stand in for
/// C's `restrict`. When a parallel backend arrives it will split this slice; the
/// kernel does not change.
///
/// # Panics
///
/// If `out` is shorter than `epochs`.
pub fn teme_to_itrf_into(epochs: &[Orientation], out: &mut [Transform<Teme, Itrf>]) {
    assert!(
        out.len() >= epochs.len(),
        "output slice holds {} of {} epochs",
        out.len(),
        epochs.len()
    );
    for (o, slot) in epochs.iter().zip(out.iter_mut()) {
        *slot = teme_to_itrf(o.ut1, o.tt, o.xp, o.yp, o.lod_seconds);
    }
}
