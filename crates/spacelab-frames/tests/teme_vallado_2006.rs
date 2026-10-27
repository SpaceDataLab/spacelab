//! TEME is defined by one document: Vallado, Crawford, Hujsak and Kelso, "Revisiting Spacetrack
//! Report #3", AIAA 2006-6753 (Rev 3 as `CelesTrak` serves it, sha256 `3720341442fc…c1d07d`). Its Eq. (1) is
//! `r_PEF = ROT3(θ_GMST82) r_TEME`, which `teme_to_itrf` implements, polar motion then carrying PEF
//! to ITRF. Its Appendix C (p. 32) prints one conversion with every input it needs.
//!
//! The document disagrees with itself there, and this file records by how much. Appendix C's
//! Eq. (C-1) adds the kinematic terms of the 1982 equation of the equinoxes,
//! `0.00264″ sin Ω + 0.000063″ sin 2Ω`, which Eq. (1) leaves out, and the printed vector follows
//! (C-1). This is a tripwire, not a gate: neither the equation nor the vector is matched to the
//! other (plans/17 §1.6), and the erratum stays open until a primary source settles which one
//! defines TEME.

use spacelab_core::Angle;
use spacelab_core::Vec3;
use spacelab_core::calendar::UtcDateTime;
use spacelab_core::time::{Epoch, Tt, Ut1};
use spacelab_frames::{EARTH_ROTATION_RATE, StateVector, Teme, teme_to_itrf};

const TT_MINUS_TAI: f64 = 32.184;
/// UT1 seconds per sidereal radian are `1/(ω·k)`, `k` the ratio of sidereal to UT1 rate in GMST82.
const SIDEREAL_PER_UT1: f64 = 1.002_737_909_350_795;

/// Appendix C, p. 32: at 2004-04-06 07:51:28.386 UTC, with ΔUT1 = −0.439 961 s, ΔAT = 32 s,
/// xp = −0.140 682″, yp = 0.333 309″ and LOD = 0.001 556 3 s, the TEME state
/// r = (5094.180 107 20, 6127.644 705 20, 6380.344 532 70) km, v = (−4.746 131 494, 0.785 817 998,
/// 5.531 931 288) km/s is the ITRF state r = (−1033.479 383 00, 7901.295 275 40, 6380.356 595 80)
/// km, v = (−3.225 636 520, −2.872 451 450, 5.531 924 446) km/s.
///
/// Measured: Eq. (1) lands 8.48 cm from the printed position, a rotation about the pole by
/// 1.064e-8 rad. The kinematic terms at this epoch are 0.001 850″ = 8.97e-9 rad (Ω = 42.6°, the
/// IAU 1980 lunar node), and with them the gap is 1.35 cm, again about the pole: 2.3e-5 s of UT1,
/// the size of one ulp of a Julian date held in a single double (4.0e-5 s). The velocity gap is
/// 4.6e-5 m/s by Eq. (1) and 7.3e-6 m/s by Eq. (C-1).
#[test]
#[allow(clippy::cast_possible_truncation, clippy::expect_used)] // |offset| < 60 s
fn the_aiaa_2006_6753_example_follows_its_eq_c1_not_its_eq_1() {
    let utc = UtcDateTime::new(2004, 4, 6, 7, 51, 28.386).expect("valid utc");
    let ns = |offset_s: f64| utc.naive_ns_since_j2000() + (offset_s * 1e9).round() as i128;
    let tt = Epoch::<Tt>::from_ns_since_j2000(ns(32.0 + TT_MINUS_TAI));
    let teme = StateVector::<Teme>::new(
        Vec3::new(5_094_180.107_20, 6_127_644.705_20, 6_380_344.532_70),
        Vec3::new(-4_746.131_494, 785.817_998, 5_531.931_288),
    );
    let r_want = Vec3::new(-1_033_479.383_00, 7_901_295.275_40, 6_380_356.595_80);
    let v_want = Vec3::new(-3_225.636_520, -2_872.451_450, 5_531.924_446);
    let gap = |ut1_offset_s: f64| {
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(ns(ut1_offset_s));
        let itrf = teme_to_itrf(
            ut1,
            tt,
            Angle::from_arcseconds(-0.140_682),
            Angle::from_arcseconds(0.333_309),
            0.001_556_3,
        )
        .state(teme);
        (
            (itrf.position.vector - r_want).norm(),
            (itrf.velocity.vector - v_want).norm(),
        )
    };

    // IAU 1980 mean longitude of the lunar ascending node.
    let t = tt.julian_centuries();
    let node = ((125.0 + 2.0 / 60.0 + 40.280 / 3600.0)
        - (5.0 * 360.0 + 134.0 + 8.0 / 60.0 + 10.539 / 3600.0) * t
        + 7.455 / 3600.0 * t * t)
        .to_radians();
    let kinematic = ((0.00264 * node.sin() + 0.000_063 * (2.0 * node).sin()) / 3600.0).to_radians();
    // A rotation about the pole by `kinematic` is GMST82 that much later.
    let shift = kinematic / (EARTH_ROTATION_RATE * SIDEREAL_PER_UT1);

    let (eq1, v_eq1) = gap(-0.439_961);
    let (c1, v_c1) = gap(-0.439_961 + shift);
    std::println!(
        "AIAA 2006-6753 Appendix C: Eq. (1) {eq1:.3e} m, {v_eq1:.3e} m/s; Eq. (C-1) {c1:.3e} m, {v_c1:.3e} m/s"
    );
    assert!((0.084..0.086).contains(&eq1), "Eq. (1) gap {eq1:.4e} m");
    assert!(c1 < 0.014, "Eq. (C-1) gap {c1:.4e} m");
    assert!(
        v_eq1 < 1e-4 && v_c1 < 1e-4,
        "velocity {v_eq1:.3e}, {v_c1:.3e} m/s"
    );
}
