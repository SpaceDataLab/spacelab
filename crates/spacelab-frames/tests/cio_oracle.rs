//! GCRF→ITRF (IAU 2006/2000A CIO) validated element-wise against pyerfa.
//!
//! The reference matrices in the fixture come from the identical pipeline —
//! `xy06` series, `s06`, `c2ixys`, `era00`, `sp00`, `pom00`, `c2tcio` — evaluated by
//! pyerfa (ERFA, BSD-3-Clause). See `derivations/cio_frame.py`. Because both sides are
//! the same CIO algorithm (not Vallado-equinox vs IERS-CIO), this asserts a tight
//! numeric tolerance, not a convention floor.
//!
//! Four of the nine fixture epochs carry celestial-pole offsets `dX`, `dY`. Without them
//! the offsets reach the transform only as `+0`, where a sign error, a swapped pair and a
//! dropped term all produce the same matrix; with them the fixture resolves each.
//!
//! LOD does not enter the ERFA pipeline at all — ERFA has no Earth-rotation rate — so the
//! angular velocity is pinned against Orekit instead.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_range_loop,
    // The LOD references are carried at their derived precision, not at f64's: truncating them to
    // what clippy considers enough would make the literal, rather than the 60-digit evaluation,
    // the thing under test.
    clippy::excessive_precision
)]

use spacelab_core::{Angle, Epoch, Tt, Ut1};
use spacelab_frames::{EARTH_ROTATION_RATE, gcrf_to_itrf, teme_to_itrf};
use spacelab_testkit::nan_max;

const FIXTURE: &str = include_str!("fixtures/cio_gcrf_itrf_pyerfa.csv");

/// One milliarcsecond in radians — the unit the fixture stores the pole offsets in.
const MAS: f64 = core::f64::consts::PI / 648_000_000.0;

fn mas(v: f64) -> Angle {
    Angle::from_radians(v * MAS)
}

/// The largest element-wise gap between two rotation matrices.
fn max_element_gap(a: &[[f64; 3]; 3], b: &[f64; 9]) -> f64 {
    let mut d = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            d = nan_max(d, (a[i][j] - b[i * 3 + j]).abs());
        }
    }
    d
}

#[test]
fn gcrf_to_itrf_matches_pyerfa_cio_pipeline() {
    let mut max_diff = 0.0_f64;
    let mut count = 0;
    let mut with_offsets = 0;

    for line in FIXTURE.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        assert_eq!(v.len(), 15, "row must be 6 params + 9 matrix elements");

        let tt = Epoch::<Tt>::from_days_since_j2000(v[0]);
        let ut1 = Epoch::<Ut1>::from_days_since_j2000(v[1]);
        let xf = gcrf_to_itrf(
            ut1,
            tt,
            Angle::from_arcseconds(v[2]),
            Angle::from_arcseconds(v[3]),
            mas(v[4]),
            mas(v[5]),
            0.0,
        );

        let mut reference = [0.0; 9];
        reference.copy_from_slice(&v[6..15]);
        max_diff = nan_max(
            max_diff,
            max_element_gap(&xf.rotation.matrix().rows, &reference),
        );

        // The angular velocity is the pole-axis spin rotated into GCRF: a pure
        // rotation of (0,0,ω), so its magnitude is exactly the nominal rate at LOD 0.
        assert!(
            (xf.omega.norm() - EARTH_ROTATION_RATE).abs() < 1e-18,
            "|ω| = {} drifted from the nominal rate",
            xf.omega.norm()
        );
        if v[4] != 0.0 || v[5] != 0.0 {
            with_offsets += 1;
        }
        count += 1;
    }

    assert_eq!(count, 9, "expected 9 fixture epochs");
    assert!(
        with_offsets >= 4,
        "only {with_offsets} epochs carry celestial-pole offsets; dX/dY would go untested"
    );
    assert!(
        max_diff < 5e-13,
        "max GCRF→ITRF element diff {max_diff:.3e} exceeds 5e-13 (was ~4.9e-14)"
    );
    eprintln!(
        "gcrf_to_itrf vs pyerfa CIO pipeline: max element diff {max_diff:.3e} over {count} epochs"
    );
}

#[test]
fn each_celestial_pole_offset_moves_the_matrix_by_its_own_size() {
    // The fixture above resolves dX and dY only because they move the matrix by more than its
    // 5e-13 agreement floor. Measure that they do, and that each moves it *alone* — a transform
    // that applied dX twice and dropped dY would still reproduce a fixture row where the two
    // happen to be equal and opposite.
    let tt = Epoch::<Tt>::from_days_since_j2000(8766.0);
    let ut1 = Epoch::<Ut1>::from_days_since_j2000(8765.6);
    let zero = Angle::from_radians(0.0);
    let pole = Angle::from_arcseconds(0.1);

    let base = gcrf_to_itrf(ut1, tt, pole, pole, zero, zero, 0.0);
    let base_rows = base.rotation.matrix().rows;
    let mut flat = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            flat[i * 3 + j] = base_rows[i][j];
        }
    }
    let moved_by = |dx: Angle, dy: Angle| {
        let m = gcrf_to_itrf(ut1, tt, pole, pole, dx, dy, 0.0);
        max_element_gap(&m.rotation.matrix().rows, &flat)
    };

    // A 1 mas offset rotates the pole by 4.848e-9 rad, four orders above the fixture's agreement
    // floor. Each of dX, dY, and a negative dX must move the matrix by that, on its own.
    for (dx, dy, name) in [
        (mas(1.0), zero, "dX alone"),
        (zero, mas(1.0), "dY alone"),
        (mas(-1.0), zero, "−dX alone"),
    ] {
        let d = moved_by(dx, dy);
        assert!(
            (d - MAS).abs() < 0.01 * MAS,
            "{name} moved the matrix by {d:.4e}, not the expected {MAS:.4e}"
        );
    }
    // and the two axes are independent: both together move it further than either alone.
    assert!(moved_by(mas(1.0), mas(1.0)) > 1.2 * moved_by(mas(1.0), zero));
}

#[test]
fn the_lod_scaling_matches_orekit() {
    // ERFA carries no rotation rate, so LOD needs a different lineage. Orekit 13.1.7
    // (orekit-13.1.7.jar, SHA-1 dc3434a6936ae5decf96fcf58c02722c88ed50ff — the provenance
    // tools/orekit-oracle/README.md records) computes it in TIRFProvider.getRotationRate:
    //
    //     20: ldc2_w   // double 7.292115146706979E-5d
    //     23: dconst_1
    //     24: dload_2  // LOD, seconds
    //     25: ldc2_w   // double 86400.0d
    //     28: ddiv  29: dsub  30: dmul        =>  AVE * (1 - LOD/86400)
    //
    // about Vector3D.PLUS_K. That is read from the shipped bytecode, not from documentation.
    //
    // The references below are that expression evaluated to 60 digits over the exact binary value
    // of the constant (0x1.31da7d7cba179p-14), agreeing to every printed digit between mpmath and
    // Maxima. 1e-18 sits three orders above the rounding of the rotation that carries ω into GCRF,
    // and three orders below the ~1.5e-12 any wrong operator here would shift the rate by.
    let tt = Epoch::<Tt>::from_days_since_j2000(8766.0);
    let ut1 = Epoch::<Ut1>::from_days_since_j2000(8765.6);
    let zero = Angle::from_radians(0.0);

    // Both public entry points carry LOD, each writing the expression out for itself: the CIO
    // chain and the TEME reduction. Whichever is left unchecked is the one that drifts.
    // (`CioInterpolator` writes it a third time; being `std`-gated, it is held against these two
    // in `cio_fast.rs`.)
    for (lod, expected) in [
        (0.0, 7.292_115_146_706_979e-5),
        (0.0018, 7.292_114_994_787_913_583_337_152e-5),
        (-0.0005, 7.292_115_188_906_719_572_064_602e-5),
        (0.0035, 7.292_114_851_308_796_113_408_154e-5),
    ] {
        for (rate, who) in [
            (
                gcrf_to_itrf(ut1, tt, zero, zero, zero, zero, lod)
                    .omega
                    .norm(),
                "gcrf_to_itrf",
            ),
            (
                teme_to_itrf(ut1, tt, zero, zero, lod).omega.norm(),
                "teme_to_itrf",
            ),
        ] {
            assert!(
                (rate - expected).abs() < 1e-18,
                "{who} at LOD {lod} s gave |ω| = {rate:.20e}, \
                 Orekit's expression gives {expected:.20e}"
            );
        }
    }

    // A positive LOD is a long day and must *slow* the Earth. 1.8 ms of it is 1.5e-12 rad/s, which
    // is 0.46 m of equatorial displacement after one day — above the metre-level agreement the
    // frame chain is validated to, so it is not something the transform may quietly drop.
    let slowed = gcrf_to_itrf(ut1, tt, zero, zero, zero, zero, 0.0018)
        .omega
        .norm();
    assert!(
        EARTH_ROTATION_RATE - slowed > 1e-12,
        "a 1.8 ms LOD must slow the rate measurably, got {:.3e}",
        EARTH_ROTATION_RATE - slowed
    );
}
