//! GCRF to ITRF: the rigorous IAU 2006/2000A, CIO-based transform.
//!
//! ```text
//! GCRF ──precession-nutation (X,Y,s)──▶ CIRS ──ERA──▶ TIRS ──polar motion──▶ ITRF
//! ```
//!
//! This is the IERS-recommended Celestial Intermediate Origin chain, the pole tracked
//! rigorously through precession and nutation rather than approximated by a fixed
//! axis. It is what the geopotential must be evaluated in for the pole-dependent
//! (tesseral) terms to be correct — the difference from a fixed-axis rotation is the
//! full accumulated precession, about 1.5 km a day at present epochs.
//!
//! The three angles come from [`spacelab_fa`] ([`cip_xy`], [`cio_locator_s`],
//! [`era`]); this module only assembles rotations, and every matrix construction is
//! validated element-wise against `pyerfa` in `tests/cio_oracle.rs`.

use spacelab_core::{Angle, Epoch, Mat3, Tt, Ut1, Vec3};
use spacelab_fa::{cio_locator_s, cip_xy, era};

use crate::polar_motion::tio_locator;
use crate::rotation::{Cirs, Gcrf, Itrf, Rotation, Tirs, Transform};
use crate::teme::{EARTH_ROTATION_RATE, Orientation};

/// GCRS to CIRS: the precession-nutation rotation `Q(t)` from the CIP coordinates
/// `X, Y` and the CIO locator `s`.
///
/// IERS Conventions 2010 Eq. 5.10:
///
/// ```text
/// Q(t) = R₃(−(E+s)) · R₂(d) · R₃(E),   E = atan2(Y, X),  d = atan √(r²/(1−r²)),  r² = X²+Y²
/// ```
#[allow(clippy::many_single_char_names)] // IERS notation: X, Y, s, E, d
#[must_use]
pub fn gcrf_to_cirs(x: f64, y: f64, s: f64) -> Rotation<Gcrf, Cirs> {
    let r2 = x * x + y * y;
    let e = if r2 > 0.0 { libm::atan2(y, x) } else { 0.0 };
    let d = libm::atan(libm::sqrt(r2 / (1.0 - r2)));
    Rotation::from_matrix(Mat3::rot3(-(e + s)) * Mat3::rot2(d) * Mat3::rot3(e))
}

/// CIRS to TIRS: Earth rotation about the Celestial Intermediate Pole, by the Earth
/// Rotation Angle. A single rotation about the pole, `R₃(ERA)`.
#[must_use]
pub fn cirs_to_tirs(era: Angle) -> Rotation<Cirs, Tirs> {
    Rotation::from_matrix(Mat3::rot3(era.radians()))
}

/// TIRS to ITRF: polar motion, `W(t)ᵀ = R₁(−y_p)·R₂(−x_p)·R₃(s′)`.
///
/// The matrix is identical to [`crate::pef_to_itrf`] — TIRS and PEF differ only by the
/// TIO locator `s′`, which is already this function's third argument — but it carries
/// the `Tirs` source frame in the type, so it composes only into the CIO chain.
#[must_use]
pub fn tirs_to_itrf(xp: Angle, yp: Angle, sp: Angle) -> Rotation<Tirs, Itrf> {
    Rotation::from_matrix(
        Mat3::rot1(-yp.radians()) * Mat3::rot2(-xp.radians()) * Mat3::rot3(sp.radians()),
    )
}

/// The full GCRF to ITRF transform, IAU 2006/2000A, rotation and angular velocity.
///
/// UT1 drives Earth rotation (ERA); TT drives the slowly-varying geometry (CIP
/// coordinates, CIO locator, TIO locator). Both are supplied by the caller — one is
/// never derived from the other inside the transform. The `dx`/`dy` celestial-pole
/// offsets (IERS EOP) correct the CIP position from the model; the CIO locator `s`
/// stays the model value. Pass zero for the pure IAU 2006/2000A pole.
///
/// The angular velocity is the Earth's spin about the Celestial Intermediate Pole.
/// The CIP is the CIRS `z`-axis by construction, so the rate vector is `(0,0,ω)` in
/// CIRS and is rotated back into the GCRF (the `From` frame) here — unlike TEME,
/// whose `z` already approximates the pole, GCRF's does not, and using it would tilt
/// `ω` by the precession-nutation angle. The rate is the instantaneous
/// `ω₀·(1 − LOD/86400)`; pass `0` for the nominal rate.
#[must_use]
pub fn gcrf_to_itrf(
    ut1: Epoch<Ut1>,
    tt: Epoch<Tt>,
    xp: Angle,
    yp: Angle,
    dx: Angle,
    dy: Angle,
    lod_seconds: f64,
) -> Transform<Gcrf, Itrf> {
    let (x, y) = cip_xy(tt);
    let s = cio_locator_s(tt, x, y);
    // dx/dy are the IERS celestial-pole offsets: correct the CIP position, keep the model s.
    let q = gcrf_to_cirs(x + dx.radians(), y + dy.radians(), s);
    let rotation: Rotation<Gcrf, Itrf> =
        tirs_to_itrf(xp, yp, tio_locator(tt)) * cirs_to_tirs(era(ut1)) * q;
    let rate = EARTH_ROTATION_RATE * (1.0 - lod_seconds / 86_400.0);
    let omega = q.inverse().matrix() * Vec3::new(0.0, 0.0, rate);
    Transform::new(rotation, omega)
}

/// Build a GCRF→ITRF transform for each epoch, writing into a caller-provided slice.
///
/// The CIO chain is a function of the epoch alone; hoisting it out of any per-object
/// loop is the same four-orders-of-magnitude win described on [`Orientation`] and
/// [`crate::teme_to_itrf_into`].
///
/// # Panics
///
/// If `out` is shorter than `epochs`.
pub fn gcrf_to_itrf_into(epochs: &[Orientation], out: &mut [Transform<Gcrf, Itrf>]) {
    assert!(
        out.len() >= epochs.len(),
        "output slice holds {} of {} epochs",
        out.len(),
        epochs.len()
    );
    // The coverage batch uses the pure model pole; sub-milliarcsecond CIP offsets are negligible there.
    let zero = Angle::from_radians(0.0);
    for (o, slot) in epochs.iter().zip(out.iter_mut()) {
        *slot = gcrf_to_itrf(o.ut1, o.tt, o.xp, o.yp, zero, zero, o.lod_seconds);
    }
}
