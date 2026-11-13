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
//! Every test here is a comparator: it prints the distance to Orekit into `docs/comparisons.md`
//! and decides nothing. The mathematics decides elsewhere:
//!
//! * **GMST82** is `derivations/gmst82.mac`, held by spacelab-fa's
//!   `gmst_is_the_exact_rational_polynomial_reduced_to_a_turn`. Orekit evaluates the same IAU-1982
//!   series in another form, so the distance is a few roundings of its `B·T` term.
//! * **TEME→ITRF** is Eq. (C-1) of AIAA 2006-6753, `derivations/teme_c1.mac`, held by
//!   `tests/teme_vallado_2006.rs`. Orekit's chain runs through the CIO instead, so the distance is
//!   the convention gap §1.2 warns about.
//! * **Inside a leap second** the naive coordinate puts `23:59:60.5` on the next day's
//!   `00:00:00.5`, one second past the instant (spacelab-core's
//!   `a_leap_second_label_is_exactly_one_day_in`); the row compares Orekit with our GMST82 at the
//!   instant itself.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_core::calendar::UtcDateTime;
use spacelab_core::time::{Epoch, Tt, Ut1};
use spacelab_core::{Angle, Mat3, Vec3};
use spacelab_fa::gmst82;
use spacelab_frames::{Teme, teme_to_itrf};
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

fn comparison_row(quantity: &str, unit: &str, distances: &[f64]) {
    let max = distances.iter().copied().fold(0.0, nan_max);
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | Orekit | 13.1.7 | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// The distance to Orekit's GMST82 (`IERS_1996`) outside leap seconds.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn gmst82_distance_to_orekit_outside_leap_seconds() {
    let d: Vec<f64> = parse_rows()
        .iter()
        .filter(|r| !r.utc.is_leap_second_label())
        .map(|r| angle_diff(gmst82(ut1_of(r)).radians(), r.gmst82_rad))
        .collect();
    comparison_row(
        "GMST82, IERS_1996, epochs 2000-2029 outside leap seconds",
        "rad",
        &d,
    );
}

/// The distance to Orekit's GMST82 inside a leap second, ours taken at the instant itself: one
/// second of UT1 before the naive coordinate's.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn gmst82_distance_to_orekit_inside_a_leap_second() {
    let d: Vec<f64> = parse_rows()
        .iter()
        .filter(|r| r.utc.is_leap_second_label())
        .map(|r| {
            let instant = Epoch::<Ut1>::from_ns_since_j2000(epoch_ns(r.utc, r.dut1 - 1.0));
            angle_diff(gmst82(instant).radians(), r.gmst82_rad)
        })
        .collect();
    comparison_row("GMST82 inside a leap second, at the instant", "rad", &d);
}

/// The distance from our TEME→ITRF (AIAA 2006-6753 Eq. (C-1)) to Orekit's `IERS_1996` chain through
/// the CIO, outside leap seconds: position on the AIAA and ISS vectors, velocity on the AIAA state,
/// and the worst matrix entry.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn teme_to_itrf_distance_to_orekit() {
    let (mut position, mut velocity, mut matrix) = (Vec::new(), Vec::new(), Vec::new());
    for r in parse_rows()
        .iter()
        .filter(|r| !r.utc.is_leap_second_label())
    {
        let t = teme_to_itrf(ut1_of(r), tt_of(r), r.xp, r.yp, 0.0);
        let m = t.rotation.matrix();
        let mut me = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                me = nan_max(me, (m.rows[i][j] - r.matrix.rows[i][j]).abs());
            }
        }
        matrix.push(me);
        let ours_aiaa = t.position(spacelab_frames::Position::<Teme>::new(R_TEME_AIAA));
        let ours_iss = t.position(spacelab_frames::Position::<Teme>::new(R_TEME_ISS));
        position.push((ours_aiaa.vector - r.r_aiaa).norm());
        position.push((ours_iss.vector - r.r_iss).norm());
        let ours_state = t.state(spacelab_frames::StateVector::<Teme>::new(
            R_TEME_AIAA,
            V_TEME,
        ));
        velocity.push((ours_state.velocity.vector - r.v_aiaa).norm());
    }
    comparison_row(
        "TEME to ITRF position, AIAA and ISS vectors",
        "m",
        &position,
    );
    comparison_row("TEME to ITRF velocity, AIAA state", "m/s", &velocity);
    comparison_row("TEME to ITRF matrix, worst entry", "1", &matrix);
}
