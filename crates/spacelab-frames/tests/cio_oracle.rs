//! GCRF→ITRF (IAU 2006/2000A CIO) against pyerfa, element-wise: a comparator.
//!
//! The gate is `the_transform_is_the_iers_chain_at_six_epochs` in `src/cio.rs`, which holds the
//! transform to the IERS (2010) chain composed in 40-digit arithmetic by
//! `derivations/gcrf_itrf_chain.mac`. The reference matrices in the fixture here come from the
//! identical pipeline — `xy06` series, `s06`, `c2ixys`, `era00`, `sp00`, `pom00`, `c2tcio` —
//! evaluated by pyerfa (ERFA, BSD-3-Clause), see `derivations/cio_frame.py`; their distance is
//! recorded, not judged.
//!
//! Four of the nine fixture epochs carry celestial-pole offsets `dX`, `dY`, as five of the gate's
//! six epochs do.
//!
//! LOD does not enter the ERFA pipeline at all — ERFA has no Earth-rotation rate — so the
//! angular velocity is decided by `derivations/earth_rotation_rate.mac` instead.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_range_loop,
    // The rotation-rate goldens are carried at their derived precision, not at f64's: truncating
    // them to what clippy considers enough would make the literal, rather than the 40-digit
    // evaluation, the thing under test.
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

/// The distance to pyerfa's CIO pipeline over the nine fixture epochs.
#[test]
#[ignore = "comparator: the distance to pyerfa, not a verdict"]
fn gcrf_to_itrf_distance_to_pyerfa_cio_pipeline() {
    let mut distances = Vec::new();
    for line in FIXTURE.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        assert_eq!(v.len(), 15, "row must be 6 params + 9 matrix elements");

        let xf = gcrf_to_itrf(
            Epoch::<Ut1>::from_days_since_j2000(v[1]).unwrap(),
            Epoch::<Tt>::from_days_since_j2000(v[0]).unwrap(),
            Angle::from_arcseconds(v[2]),
            Angle::from_arcseconds(v[3]),
            mas(v[4]),
            mas(v[5]),
            0.0,
        );
        let mut reference = [0.0; 9];
        reference.copy_from_slice(&v[6..15]);
        distances.push(max_element_gap(&xf.rotation.matrix().rows, &reference));
    }
    assert_eq!(distances.len(), 9, "expected 9 fixture epochs");
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(clippy::cast_precision_loss, reason = "nine distances")]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | pyerfa | not recorded | the GCRF→ITRF matrix, worst entry | 1 | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// Each celestial-pole offset moves the matrix by its own size, alone. (S2)
///
/// A rotation of the pole through `δ` moves the matrix by `δ` to first order. The chain gate in
/// `src/cio.rs` resolves `dX` and `dY` because they move the matrix by more than its 3e-13 bound;
/// this measures that they do, and that each moves it *alone* — a transform that applied `dX`
/// twice and dropped `dY` would still reproduce a row where the two happen to be equal and opposite.
#[test]
fn each_celestial_pole_offset_moves_the_matrix_by_its_own_size() {
    let tt = Epoch::<Tt>::from_days_since_j2000(8766.0).unwrap();
    let ut1 = Epoch::<Ut1>::from_days_since_j2000(8765.6).unwrap();
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

/// The LODs of [`the_rotation_rate_is_the_era_rate_slowed_by_the_length_of_day`], seconds.
const LOD_SECONDS: [f64; 4] = [0.0, 0.0018, -0.0005, 0.0035];

/// `derivations/earth_rotation_rate.mac` (3): `EARTH_ROTATION_RATE · (1 − LOD/86400)` over the
/// shipped double at [`LOD_SECONDS`], rad/s.
const ROTATION_RATE_MAXIMA: [f64; 4] = [
    7.292_115_146_706_979_139_732_546_224_55e-5,
    7.292_114_994_787_913_583_337_147_480_12e-5,
    7.292_115_188_906_719_572_064_601_431_34e-5,
    7.292_114_851_308_796_113_408_159_777_05e-5,
];

/// The angular velocity is the Earth Rotation Angle's rate slowed by the length of day:
/// `derivations/earth_rotation_rate.mac` (1)–(3) at 40 digits. (G3)
///
/// ERFA carries no rotation rate, so the pyerfa fixture above cannot see it. The ERA of IERS
/// Conventions (2010) eq. (5.15) turns at `2π·1.00273781191135448/86400` rad per UT1 second, whose
/// nearest double is `EARTH_ROTATION_RATE`, and LOD's definition puts `1 − LOD/86400` UT1 seconds in
/// each SI second.
#[test]
fn the_rotation_rate_is_the_era_rate_slowed_by_the_length_of_day() {
    let tt = Epoch::<Tt>::from_days_since_j2000(8766.0).unwrap();
    let ut1 = Epoch::<Ut1>::from_days_since_j2000(8765.6).unwrap();
    let zero = Angle::from_radians(0.0);

    // Both public entry points carry LOD, each writing the expression out for itself: the CIO
    // chain and the TEME reduction. Whichever is left unchecked is the one that drifts.
    // (`CioInterpolator` writes it a third time; being `std`-gated, it is held against these two
    // in `cio_fast.rs`.)
    let mut worst = 0.0_f64;
    for (lod, expected) in LOD_SECONDS.into_iter().zip(ROTATION_RATE_MAXIMA) {
        for rate in [
            gcrf_to_itrf(ut1, tt, zero, zero, zero, zero, lod)
                .omega
                .norm(),
            teme_to_itrf(ut1, tt, zero, zero, lod).omega.norm(),
        ] {
            worst = worst.max((rate - expected).abs());
        }
    }
    println!("rotation rate against earth_rotation_rate.mac at 40 digits: {worst:.3e} rad/s");
    // Measured 1.4e-20 rad/s, one ulp of the rate.
    assert!(
        worst < 5e-20,
        "rotation rate against earth_rotation_rate.mac: {worst:.3e} rad/s"
    );

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
