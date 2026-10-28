//! TEME is defined by one document: Vallado, Crawford, Hujsak and Kelso, "Revisiting Spacetrack
//! Report #3", AIAA 2006-6753 (Rev 3 as `CelesTrak` serves it, sha256 `3720341442fc…c1d07d`). Its
//! Eq. (C-1), `r_ITRF = W·ROT3(θ_GMST82)·ROT3(EqEquinox1982_kin)·r_TEME`, is what `teme_to_itrf`
//! implements. Its Appendix C (p. 32) prints one conversion with every input it needs.
//!
//! The paper also gives Eq. (1), which rotates by GMST alone. The Appendix C vector follows (C-1),
//! and the authors confirmed on 2026-10-28 that (C-1) is the intended transform. The gate is the
//! mathematics: `derivations/teme_c1.mac` evaluates (C-1) on the example's inputs at 40 digits, and
//! the double-precision transform must reproduce that state. The printed vector is a comparison
//! only (plans/17 §1.6): it sits 1.33 cm from (C-1), and the authors could not say whether that is
//! the interpolation of an input.

use spacelab_core::Angle;
use spacelab_core::Vec3;
use spacelab_core::calendar::UtcDateTime;
use spacelab_core::time::{Epoch, Tt, Ut1};
use spacelab_fa::equinox_kinematic_1982;
use spacelab_frames::{StateVector, Teme, teme_to_itrf};

const TT_MINUS_TAI: f64 = 32.184;
/// GMST82 advances `2π·k` per UT1 day, `k` the ratio of sidereal to UT1 rate.
const SIDEREAL_PER_UT1: f64 = 1.002_737_909_350_795;

/// Appendix C, p. 32: at 2004-04-06 07:51:28.386 UTC, with ΔUT1 = −0.439 961 s, ΔAT = 32 s,
/// xp = −0.140 682″, yp = 0.333 309″ and LOD = 0.001 556 3 s, the TEME state
/// r = (5094.180 107 20, 6127.644 705 20, 6380.344 532 70) km, v = (−4.746 131 494, 0.785 817 998,
/// 5.531 931 288) km/s is the ITRF state r = (−1033.479 383 00, 7901.295 275 40, 6380.356 595 80)
/// km, v = (−3.225 636 520, −2.872 451 450, 5.531 924 446) km/s.
///
/// `derivations/teme_c1.mac` at 40 digits: (C-1) gives r = (−1 033 479.396 221 834,
/// 7 901 295.273 710 045, 6 380 356.595 751 121) m, v = (−3 225.636 514 890 048,
/// −2 872.451 455 023 926, 5 531.924 446 303 694) m/s, 1.333 cm and 7.17e-6 m/s from the printed
/// state; Eq. (1) is 8.480 cm from it. The kinematic terms are 0.001 850″ (Ω = 42.6°).
#[test]
#[allow(clippy::cast_possible_truncation, clippy::expect_used)] // |offset| < 60 s
fn teme_to_itrf_is_eq_c1_of_aiaa_2006_6753_to_the_cas() {
    let utc = UtcDateTime::new(2004, 4, 6, 7, 51, 28.386).expect("valid utc");
    let ns = |offset_s: f64| utc.naive_ns_since_j2000() + (offset_s * 1e9).round() as i128;
    let tt = Epoch::<Tt>::from_ns_since_j2000(ns(32.0 + TT_MINUS_TAI));
    let teme = StateVector::<Teme>::new(
        Vec3::new(5_094_180.107_20, 6_127_644.705_20, 6_380_344.532_70),
        Vec3::new(-4_746.131_494, 785.817_998, 5_531.931_288),
    );
    let itrf_at = |ut1_offset_s: f64| {
        teme_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(ns(ut1_offset_s)),
            tt,
            Angle::from_arcseconds(-0.140_682),
            Angle::from_arcseconds(0.333_309),
            0.001_556_3,
        )
        .state(teme)
    };
    let itrf = itrf_at(-0.439_961);

    // The gate: the CAS's (C-1). Double precision carries GMST to ~1e-13 rad, 1e-6 m here.
    let r_cas = Vec3::new(
        -1_033_479.396_221_833_7,
        7_901_295.273_710_045,
        6_380_356.595_751_121,
    );
    let v_cas = Vec3::new(
        -3_225.636_514_890_048,
        -2_872.451_455_023_926,
        5_531.924_446_303_694,
    );
    let (dr, dv) = (
        (itrf.position.vector - r_cas).norm(),
        (itrf.velocity.vector - v_cas).norm(),
    );
    assert!(dr < 1e-5, "(C-1) against the CAS: {dr:.3e} m");
    assert!(dv < 1e-8, "(C-1) against the CAS: {dv:.3e} m/s");

    // The comparisons: the printed vector, and Eq. (1) as a rotation that much earlier.
    let r_printed = Vec3::new(-1_033_479.383_00, 7_901_295.275_40, 6_380_356.595_80);
    let kinematic = equinox_kinematic_1982(tt).radians();
    let shift = kinematic / (core::f64::consts::TAU / 86_400.0 * SIDEREAL_PER_UT1);
    let eq1 = (itrf_at(-0.439_961 - shift).position.vector - r_printed).norm();
    let c1 = (itrf.position.vector - r_printed).norm();
    std::println!(
        "AIAA 2006-6753 Appendix C: (C-1) {dr:.2e} m from the CAS; printed vector {c1:.4e} m by (C-1), {eq1:.4e} m by Eq. (1)"
    );
    assert!(
        (0.0133..0.0134).contains(&c1),
        "(C-1) to printed {c1:.4e} m"
    );
    assert!(
        (0.0847..0.0849).contains(&eq1),
        "Eq. (1) to printed {eq1:.4e} m"
    );
}
