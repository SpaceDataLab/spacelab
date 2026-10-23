//! The batched GCRF→ITRF builder.
//!
//! `frames.rs` covers the TEME batch; this closes the CIO one: `gcrf_to_itrf_into` must equal the
//! scalar `gcrf_to_itrf` (with zero CIP offsets) epoch by epoch, and refuse a short output slice
//! rather than silently truncate. (The `CioInterpolator` accessors are covered inline in
//! `cio_fast.rs`, where the `std`/`test`-gated type is in scope.)

#![allow(clippy::float_cmp)]

use spacelab_core::{Angle, Epoch, Tt, Ut1};
use spacelab_frames::{Gcrf, Itrf, Orientation, Transform, gcrf_to_itrf, gcrf_to_itrf_into};

#[test]
fn gcrf_to_itrf_into_agrees_with_the_scalar_path() {
    let zero = Angle::from_radians(0.0);
    let epochs: Vec<Orientation> = (0..64)
        .map(|k| {
            let d = f64::from(k) * 0.05;
            Orientation {
                ut1: Epoch::<Ut1>::from_days_since_j2000(d),
                tt: Epoch::<Tt>::from_days_since_j2000(d),
                xp: Angle::from_arcseconds(0.1 + d * 1e-5),
                yp: Angle::from_arcseconds(0.3 - d * 1e-5),
                lod_seconds: 0.001,
            }
        })
        .collect();

    let identity = gcrf_to_itrf(Epoch::J2000, Epoch::J2000, zero, zero, zero, zero, 0.0);
    let mut out = vec![identity; epochs.len()];
    gcrf_to_itrf_into(&epochs, &mut out);

    // The batch is exactly the scalar path with zero CIP offsets, epoch by epoch.
    for (o, batched) in epochs.iter().zip(&out) {
        let scalar = gcrf_to_itrf(o.ut1, o.tt, o.xp, o.yp, zero, zero, o.lod_seconds);
        assert_eq!(*batched, scalar, "batch and scalar diverged");
    }
}

#[test]
#[should_panic(expected = "output slice holds")]
fn gcrf_to_itrf_into_refuses_a_short_output() {
    let zero = Angle::from_radians(0.0);
    let epochs = [Orientation {
        ut1: Epoch::J2000,
        tt: Epoch::J2000,
        xp: zero,
        yp: zero,
        lod_seconds: 0.0,
    }];
    let mut out: [Transform<Gcrf, Itrf>; 0] = [];
    gcrf_to_itrf_into(&epochs, &mut out);
}
