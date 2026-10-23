//! Properties of the CIO chain that hold for every epoch, not just the ones somebody wrote down.
//!
//! The oracle tests in this crate check the transform at a handful of epochs against Orekit and
//! ERFA. They cannot say anything about the epochs nobody chose. These three can, because they are
//! statements about what the transform *is* rather than about what it equals:
//!
//! 1. **The matrix is orthonormal.** `R·Rᵀ = I`. The chain is a product of four rotations, each
//!    assembled from a truncated series; if any factor is not a proper rotation — a small-angle
//!    approximation kept one term too few, a normalisation dropped — the product stops preserving
//!    length, and every distance computed through it is wrong by a factor no test of a single
//!    transformed vector would isolate.
//! 2. **The determinant is `+1`.** Orthonormality alone admits a reflection, which would flip
//!    handedness and pass property 1 exactly.
//! 3. **A state survives the round trip.** GCRF → ITRF → GCRF returns the position *and the
//!    velocity*. Position alone is a weak test — [`Transform::inverse`] is the transpose, so
//!    positions round-trip whenever property 1 holds. Velocity is not: it carries the `ω × r` term
//!    that `inverse` re-expresses in the target frame, and that bookkeeping has no other check.
//!
//! **Every bound below is the measured worst case with one order of headroom, not a round number.**
//! Over 20 000 draws spanning the whole domain: orthonormality defect 8.882e-16, determinant
//! departure 7.772e-16, relative position round-trip 7.976e-16, relative velocity round-trip
//! 7.905e-16 — all four within four ulp, which is round-off on a product of four matrices and not a
//! modelling error. A bound of 1e-6 would have passed just as green and said nothing.
//!
//! Ranges are the physical ones, not the representable ones: ±60 years of epoch either side of
//! J2000, polar motion inside the ±0.5″ the pole has ever wandered, celestial-pole offsets inside
//! the ±0.3 mas the IERS publishes, LOD inside ±5 ms. A property that only holds on inputs the
//! caller cannot supply is not worth asserting.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use proptest::prelude::*;
use spacelab_core::{Angle, Epoch, Tt, Ut1, Vec3};
use spacelab_frames::{Gcrf, StateVector, gcrf_to_itrf};

/// Nanoseconds in sixty years — the span the IAU 2006/2000A series is fitted over and a little more.
const SPAN_NS: i128 = 60 * 365 * 86_400 * 1_000_000_000;

/// Polar motion never leaves this, in arcseconds; the observed wander is under 0.4″.
const POLE_ARCSEC: f64 = 0.5;

/// IERS celestial-pole offsets are tens of microarcseconds; a milliarcsecond is generous.
const OFFSET_MAS: f64 = 0.3;

/// Length of day departs from 86 400 s by under 4 ms.
const LOD_SECONDS: f64 = 5e-3;

/// Geocentric distance range spanned: a station on the ground out to beyond geostationary.
const RADIUS_RANGE: (f64, f64) = (6.3e6, 4.3e7);

/// Orbital speed range spanned, m/s.
const SPEED_RANGE: (f64, f64) = (0.0, 1.1e4);

fn arcsec(x: f64) -> Angle {
    Angle::from_radians(x * core::f64::consts::PI / (180.0 * 3600.0))
}

/// The transform under test, from the raw parameters proptest generates.
fn transform(
    ns: i128,
    xp: f64,
    yp: f64,
    dx: f64,
    dy: f64,
    lod: f64,
) -> spacelab_frames::Transform<Gcrf, spacelab_frames::Itrf> {
    gcrf_to_itrf(
        Epoch::<Ut1>::from_ns_since_j2000(ns),
        Epoch::<Tt>::from_ns_since_j2000(ns),
        arcsec(xp),
        arcsec(yp),
        arcsec(dx * 1e-3),
        arcsec(dy * 1e-3),
        lod,
    )
}

/// Largest absolute deviation of `m·mᵀ` from the identity.
fn orthonormality_defect(m: [[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = (0..3).map(|k| m[i][k] * m[j][k]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - want).abs());
        }
    }
    worst
}

prop_compose! {
    /// The Earth-orientation parameters of one arbitrary but physically reachable instant.
    fn orientation()(
        ns in -SPAN_NS..SPAN_NS,
        xp in -POLE_ARCSEC..POLE_ARCSEC,
        yp in -POLE_ARCSEC..POLE_ARCSEC,
        dx in -OFFSET_MAS..OFFSET_MAS,
        dy in -OFFSET_MAS..OFFSET_MAS,
        lod in -LOD_SECONDS..LOD_SECONDS,
    ) -> (i128, f64, f64, f64, f64, f64) {
        (ns, xp, yp, dx, dy, lod)
    }
}

prop_compose! {
    /// A state with a plausible geocentric distance and speed, in an arbitrary direction.
    fn state()(
        r in RADIUS_RANGE.0..RADIUS_RANGE.1,
        v in SPEED_RANGE.0..SPEED_RANGE.1,
        ru in -1.0_f64..1.0, rv in -1.0_f64..1.0, rw in -1.0_f64..1.0,
        vu in -1.0_f64..1.0, vv in -1.0_f64..1.0, vw in -1.0_f64..1.0,
    ) -> (Vec3, Vec3) {
        let dir = |a: f64, b: f64, c: f64, scale: f64| {
            let n = (a * a + b * b + c * c).sqrt();
            if n == 0.0 {
                Vec3::new(scale, 0.0, 0.0)
            } else {
                let k = scale / n;
                Vec3::new(a * k, b * k, c * k)
            }
        };
        (dir(ru, rv, rw, r), dir(vu, vv, vw, v))
    }
}

proptest! {
    /// `R·Rᵀ = I` for every reachable epoch and pole. Worst measured defect 8.882e-16.
    #[test]
    fn the_cio_chain_is_orthonormal((ns, xp, yp, dx, dy, lod) in orientation()) {
        let defect = orthonormality_defect(transform(ns, xp, yp, dx, dy, lod).rotation.matrix().rows);
        prop_assert!(
            defect < 1e-14,
            "R·Rᵀ departs from the identity by {defect:.3e} at ns {ns}"
        );
    }

    /// A reflection satisfies `R·Rᵀ = I` too, so handedness needs its own statement. Worst
    /// measured departure 7.772e-16.
    #[test]
    fn the_cio_chain_is_a_proper_rotation((ns, xp, yp, dx, dy, lod) in orientation()) {
        let det = transform(ns, xp, yp, dx, dy, lod).rotation.matrix().determinant();
        prop_assert!(
            (det - 1.0).abs() < 1e-14,
            "determinant {det:.17} is not +1 at ns {ns}"
        );
    }

    /// GCRF → ITRF → GCRF returns the state. The velocity is the part that means something: it
    /// goes through `v − ω × r` one way and the negated, re-expressed `ω` the other, and nothing
    /// else in the crate checks that those two cancel. Worst measured relative error 7.976e-16 on
    /// position and 7.905e-16 on velocity.
    #[test]
    fn a_state_survives_the_round_trip_through_itrf(
        (ns, xp, yp, dx, dy, lod) in orientation(),
        (r, v) in state(),
    ) {
        let t = transform(ns, xp, yp, dx, dy, lod);
        let there = t.state(StateVector::<Gcrf>::new(r, v));
        let back = t.inverse().state(there);

        // Scaled by the magnitude carried through the rotation: the absolute error of an
        // orthonormal map is proportional to what it maps. The velocity scale carries the `ω × r`
        // term as well, because that is the quantity the inverse has to cancel — at geostationary
        // it is 3 km/s and dominates the orbital speed itself.
        let dr = (back.position.vector - r).norm() / r.norm();
        let dv = (back.velocity.vector - v).norm() / (v.norm() + 7.3e-5 * r.norm());
        prop_assert!(dr < 1e-14, "position moved {dr:.3e} of its own magnitude");
        prop_assert!(dv < 1e-14, "velocity moved {dv:.3e} of its own scale");
    }
}
