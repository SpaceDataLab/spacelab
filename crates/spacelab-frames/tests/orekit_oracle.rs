//! Our frame chain against Orekit — a third, independent lineage.
//!
//! Every other oracle we run descends from SOFA (`pyerfa`) or from Vallado (`skyfield`,
//! `python-sgp4`). `plans/04-validation-and-risks.md` §1.2 says two same-lineage oracles catch our
//! *transcription* errors and cannot catch an *inherited* one. Orekit implements the IERS
//! Conventions directly and descends from neither.
//!
//! The fixtures in `fixtures/orekit_teme_itrf.csv` were produced by
//! `tools/orekit-oracle/OrekitTemeItrfOracle.java` against **Orekit 13.1.7 / Hipparchus 4.0.3**,
//! with the polar motion, UT1−UTC and TAI−UTC *injected* per epoch and asserted at run time, so
//! the inputs are exactly reproducible here. Nutation corrections are zero (pure IAU model), LOD
//! is zero (nominal Earth rate), and no ITRF datum (Helmert) step is applied — this is the raw
//! CIO-based terrestrial frame, which is what a from-scratch TEME→ITRF routine produces.
//!
//! Two comparisons, and they measure different things:
//!
//! * **GMST82** is the same IAU-1982 series in both codes, so agreement is a *transcription*
//!   check across two independent implementations (our Rust, Orekit's Java). It should be tight.
//! * **TEME→ITRF** is *not* the same algorithm. Ours is Vallado's equinox path (TEME→PEF by
//!   GMST82, PEF→ITRF by polar motion). Orekit's is the rigorous IERS chain through the CIO. The
//!   gap between them is the inherited-convention error §1.2 warns about, and this test measures
//!   its size rather than asserting it away.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_core::calendar::UtcDateTime;
use spacelab_core::time::{Epoch, Tt, Ut1};
use spacelab_core::{Angle, Mat3, Vec3};
use spacelab_fa::gmst82;
use spacelab_frames::{EARTH_ROTATION_RATE, Teme, teme_to_itrf};
use spacelab_testkit::nan_max;

const FIXTURES: &str = include_str!("fixtures/orekit_teme_itrf.csv");

/// TT − TAI, seconds, exact by definition.
const TT_MINUS_TAI: f64 = 32.184;

/// The AIAA-2006-6753 TEME example position, metres. Matches the Java tool.
const R_TEME_AIAA: Vec3 = Vec3::new(-9_060_473.735_69, 4_658_709.525_02, 813_686.731_53);
/// An ISS-like TEME position, metres.
const R_TEME_ISS: Vec3 = Vec3::new(-2_932_912.880, -6_041_930.794, -509_897.582);
/// The TEME velocity carried through the transform, m/s.
const V_TEME: Vec3 = Vec3::new(-2_233.348_094, -4_110.136_162, -3_157.394_074);

struct Row {
    utc: UtcDateTime,
    dut1: f64,
    dat: f64,
    xp: Angle,
    yp: Angle,
    gmst82_rad: f64,
    diff_2010_1996: f64,
    matrix: Mat3,
    r_aiaa: Vec3,
    r_iss: Vec3,
    v_aiaa: Vec3,
}

fn parse_utc(s: &str) -> UtcDateTime {
    // "2000-01-01T12:00:00.000"
    let (date, time) = s.split_once('T').expect("iso datetime");
    let mut d = date.split('-');
    let year: i32 = d.next().unwrap().parse().unwrap();
    let month: u8 = d.next().unwrap().parse().unwrap();
    let day: u8 = d.next().unwrap().parse().unwrap();
    let mut t = time.split(':');
    let hour: u8 = t.next().unwrap().parse().unwrap();
    let minute: u8 = t.next().unwrap().parse().unwrap();
    let second: f64 = t.next().unwrap().parse().unwrap();
    UtcDateTime::new(year, month, day, hour, minute, second).expect("valid utc")
}

fn parse_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for line in FIXTURES.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let g = |i: usize| f[i].parse::<f64>().expect("float field");
        let m = Mat3::from_rows([
            [g(7), g(8), g(9)],
            [g(10), g(11), g(12)],
            [g(13), g(14), g(15)],
        ]);
        rows.push(Row {
            utc: parse_utc(f[0]),
            dut1: g(1),
            dat: g(2),
            xp: Angle::from_arcseconds(g(3)),
            yp: Angle::from_arcseconds(g(4)),
            gmst82_rad: g(5),
            diff_2010_1996: g(6),
            matrix: m,
            r_aiaa: Vec3::new(g(16), g(17), g(18)),
            r_iss: Vec3::new(g(19), g(20), g(21)),
            v_aiaa: Vec3::new(g(22), g(23), g(24)),
        });
    }
    rows
}

/// `ns` since J2000 for the naive coordinate plus an offset in seconds. UT1 = naive + (UT1−UTC);
/// TT = naive + (TAI−UTC) + (TT−TAI). Both are additive in the naive coordinate by construction —
/// that is the whole point of the naive coordinate (`plans/06-fundamental-astronomy.md`).
#[allow(clippy::cast_possible_truncation)] // |offset_seconds| < 60, so offset*1e9 < 6e10 << i128::MAX
fn epoch_ns(utc: UtcDateTime, offset_seconds: f64) -> i128 {
    utc.naive_ns_since_j2000() + (offset_seconds * 1e9).round() as i128
}

fn ut1_of(r: &Row) -> Epoch<Ut1> {
    Epoch::<Ut1>::from_ns_since_j2000(epoch_ns(r.utc, r.dut1))
}

fn tt_of(r: &Row) -> Epoch<Tt> {
    Epoch::<Tt>::from_ns_since_j2000(epoch_ns(r.utc, r.dat + TT_MINUS_TAI))
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > core::f64::consts::PI {
        d -= core::f64::consts::TAU;
    }
    while d < -core::f64::consts::PI {
        d += core::f64::consts::TAU;
    }
    d.abs()
}

/// GMST82 outside a leap second: the same IAU-1982 series in both codes, so agreement is a
/// transcription cross-check across two independent implementations of two *different valid
/// evaluation forms* (ours is ERFA-compact; Orekit's is the Aoki day-fraction split).
///
/// The ceiling is derived, not tuned. Two forms of the same polynomial differ by roundings of its
/// dominant term `B·T` (`B = 8640184.812866` s/century); in radians that is `ε·|B·T|·τ/86400`,
/// which at 2029 (`T ≈ 0.3`) is about `4e-12` rad. Round up to `1e-11` rad — **2 nanometres of
/// rotation at the equator**, far below anything a `f64` GMST resolves, and a wrong coefficient
/// anywhere would miss by orders of magnitude more.
#[test]
fn gmst82_matches_orekit_iers1996_outside_leap_seconds() {
    let rows = parse_rows();
    assert!(rows.len() >= 12, "fixtures did not load");

    let mut worst = 0.0_f64;
    let mut checked = 0;
    for r in &rows {
        if r.utc.is_leap_second_label() {
            continue;
        }
        let ours = gmst82(ut1_of(r)).radians();
        let d = angle_diff(ours, r.gmst82_rad);
        assert!(
            d < 1e-11,
            "GMST82 at {:?}: ours {ours:.15}, Orekit {:.15}, diff {d:.3e} rad exceeds 1e-11",
            r.utc,
            r.gmst82_rad
        );
        worst = nan_max(worst, d);
        checked += 1;
    }
    assert!(
        checked >= 10,
        "expected at least 10 non-leap epochs, got {checked}"
    );
    println!(
        "GMST82 vs Orekit over {checked} non-leap epochs: worst {worst:.3e} rad \
         ({:.4} nm at the equator)",
        worst * 6_378_137.0 * 1e9
    );
}

/// GMST82 inside a leap second: the naive coordinate is off by *exactly* one leap second of Earth
/// rotation, and this test proves it is exactly that and nothing more.
///
/// This is not a defect to hide. It is the measurable price of the naive coordinate's invariant,
/// and pinning it to `EARTH_ROTATION_RATE × 1 s` is what distinguishes "a known 1-second aliasing"
/// from "an unexplained kilometre." A caller who must be correct *inside* a leap second has to pass
/// the calendar's `extra_second` disambiguation; the bare frame chain cannot know.
#[test]
fn gmst82_inside_a_leap_second_is_off_by_exactly_one_second_of_rotation() {
    let rows = parse_rows();
    let mut n = 0;
    for r in rows.iter().filter(|r| r.utc.is_leap_second_label()) {
        let d = angle_diff(gmst82(ut1_of(r)).radians(), r.gmst82_rad);
        // `EARTH_ROTATION_RATE` is already ω⊕, the sidereal rate (7.292115e-5 rad/s). GMST advances
        // by exactly `dGMST/dUT1 · 1 s = ω⊕ · 1 s` for one UT1-second of aliasing — the 1.00273790935
        // day-length factor is what makes ω⊕ differ from 2π/86400 in the first place, so applying it
        // again would double-count.
        let one_second_of_rotation = EARTH_ROTATION_RATE * 1.0;
        assert!(
            (d - one_second_of_rotation).abs() < 1e-7,
            "expected the leap-second aliasing to be exactly one second of rotation \
             ({one_second_of_rotation:.6e} rad), measured {d:.6e} rad at {:?}",
            r.utc
        );
        n += 1;
    }
    assert!(n >= 2, "expected the two mid-leap fixtures, got {n}");
    println!(
        "confirmed: an instant inside a leap second aliases by exactly 1 s of rotation ({n} fixtures)"
    );
}

/// TEME→ITRF: a *different* algorithm from Orekit's, so this measures the convention gap.
///
/// It does not assert a tight tolerance, because the two chains are not the same transformation.
/// It asserts the gap is where physics says it should be — sub-metre in position, matching the
/// centimetre-to-decimetre disagreement Orekit's own `ITRFProviderTest` documents against Vallado
/// — and it *prints* the number, because the number is the finding.
#[test]
fn teme_to_itrf_agrees_with_orekit_to_the_convention_floor() {
    let rows = parse_rows();
    let mut worst_pos = 0.0_f64;
    let mut worst_mat = 0.0_f64;
    let mut worst_vel = 0.0_f64;
    let mut orekit_spread = 0.0_f64;

    for r in &rows {
        let t = teme_to_itrf(ut1_of(r), tt_of(r), r.xp, r.yp, 0.0);
        let m = t.rotation.matrix();

        // matrix agreement, worst entry
        let mut me = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                me = nan_max(me, (m.rows[i][j] - r.matrix.rows[i][j]).abs());
            }
        }

        // position agreement on both test vectors, metres
        let ours_aiaa = t.position(spacelab_frames::Position::<Teme>::new(R_TEME_AIAA));
        let ours_iss = t.position(spacelab_frames::Position::<Teme>::new(R_TEME_ISS));
        let pe = (ours_aiaa.vector - r.r_aiaa)
            .norm()
            .max((ours_iss.vector - r.r_iss).norm());

        // velocity agreement — this exercises the ω × r term, the part a naive rotation drops
        let ours_state = t.state(spacelab_frames::StateVector::<Teme>::new(
            R_TEME_AIAA,
            V_TEME,
        ));
        let ve = (ours_state.velocity.vector - r.v_aiaa).norm();

        if r.utc.is_leap_second_label() {
            // Inside a leap second the naive coordinate is off by one second of rotation, which on
            // the horizontal part of the test vector is |r_h|·ω·1s. Confirm the position error is
            // exactly that and nothing more — same finding as the GMST test, projected onto a vector.
            let r_h = (r.r_aiaa.x * r.r_aiaa.x + r.r_aiaa.y * r.r_aiaa.y).sqrt();
            let expected = r_h * EARTH_ROTATION_RATE * 1.0;
            assert!(
                (pe - expected).abs() < 5.0,
                "leap-second aliasing at {:?}: expected ~{expected:.1} m (|r_h|·ω·1s), got {pe:.1} m",
                r.utc
            );
            continue;
        }

        worst_mat = nan_max(worst_mat, me);
        worst_pos = nan_max(worst_pos, pe);
        worst_vel = nan_max(worst_vel, ve);
        orekit_spread = nan_max(orekit_spread, r.diff_2010_1996);
    }

    // Outside leap seconds, our Vallado equinox chain and Orekit's IERS CIO chain are two *different*
    // algorithms for the same transformation. They agree to sub-metre — the known TEME-definition
    // ambiguity Vallado documents at the metre level — with the disagreement flat across three
    // decades rather than growing, which is the signature of a fixed convention offset, not a bug.
    assert!(
        worst_pos < 1.5,
        "TEME→ITRF position disagreement {worst_pos:.3} m exceeds the 1.5 m convention floor — \
         a growing or large gap would mean our chain is wrong, not merely a different convention"
    );
    assert!(
        worst_vel < 5.0,
        "velocity disagreement {worst_vel:.3} m/s is too large — the ω×r term is wrong"
    );

    println!(
        "TEME→ITRF vs Orekit IERS_1996, 10 non-leap epochs: worst position {worst_pos:.4} m, \
         velocity {worst_vel:.4e} m/s, matrix entry {worst_mat:.3e}"
    );
    println!(
        "  (Orekit's own IERS_2010 vs IERS_1996 differs up to {orekit_spread:.3} m on this vector; \
         TLE drifts 1–3 km/day)"
    );
}
