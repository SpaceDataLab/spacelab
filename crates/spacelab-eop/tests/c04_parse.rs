//! Tests against a committed excerpt of the IERS EOP 20 C04 series, `eopc04.1962-now`.
//!
//! The excerpt keeps the six header lines verbatim and the rows for every day the `finals2000A`
//! excerpt also holds and C04 has reached: 1973, the turn of 2000, the 2017 leap second and July
//! 2026. It adds 2024-12-29 to 2025-01-08 for the IGS orbit fit in `crates/spacelab`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::{Itrs, ReferenceFrame};
use spacelab_eop::{EopError, EopQuality, EopSeries};

const C04: &str = include_str!("fixtures/eopc04_excerpt.txt");
const FINALS: &str = include_str!("fixtures/finals2000A_excerpt.txt");

fn c04() -> EopSeries {
    EopSeries::parse_c04(C04).expect("the C04 excerpt parses")
}

/// The reason to read C04 at all: it says which ITRF its pole is in, and `finals2000A` cannot.
#[test]
fn the_header_states_itrf2020_and_finals_states_nothing() {
    let c04 = c04();
    let finals = EopSeries::parse_finals2000a(FINALS).expect("the finals excerpt parses");

    assert_eq!(c04.frame(), ReferenceFrame::Itrs(Itrs::Itrf2020));
    assert_eq!(finals.frame(), ReferenceFrame::Unstated);
    // An IGS20 orbit may be differenced against a chain built on this series, and only on it.
    let igs20 = ReferenceFrame::Itrs(Itrs::Igs20);
    assert!(c04.frame().agrees_with(igs20));
    assert!(!finals.frame().agrees_with(igs20));
    assert!(!c04.frame().agrees_with(ReferenceFrame::Itrs(Itrs::Igs14)));
}

/// One row, every field read, against the digits in the file (2017-01-01, the first day after the
/// leap second). The units are the trap: C04 prints seconds of arc and seconds throughout, where
/// `finals2000A` prints `dX, dY` in milliarcseconds and LOD in milliseconds.
#[test]
fn the_columns_are_the_published_byte_ranges() {
    // 2017   1   1   0  57754.00    0.080549    0.263128   0.5912870    0.000120   -0.000168 ...
    let s = c04();
    let row = s
        .samples()
        .iter()
        .find(|x| (x.mjd - 57_754.0).abs() < 1e-9)
        .expect("the excerpt carries 2017-01-01");
    let e = row.best();
    let close = |got: f64, want: f64| (got - want).abs() < 1e-12;

    assert!(
        close(e.xp.arcseconds(), 0.080_549),
        "xp {}",
        e.xp.arcseconds()
    );
    assert!(
        close(e.yp.arcseconds(), 0.263_128),
        "yp {}",
        e.yp.arcseconds()
    );
    assert!(
        close(e.dut1_seconds, 0.591_287_0),
        "dut1 {}",
        e.dut1_seconds
    );
    assert!(
        close(e.dx.arcseconds(), 0.000_120),
        "dX {}",
        e.dx.arcseconds()
    );
    assert!(
        close(e.dy.arcseconds(), -0.000_168),
        "dY {}",
        e.dy.arcseconds()
    );
    assert!(close(e.lod_seconds, 0.000_996_2), "LOD {}", e.lod_seconds);
    assert!(close(e.xp_sigma.arcseconds(), 0.000_069));
    assert!(close(e.yp_sigma.arcseconds(), 0.000_058));
    assert!(close(e.dut1_sigma_seconds, 0.000_014_6));
    assert_eq!(e.quality, EopQuality::Final);
    assert_eq!(row.bulletin_b, Some(row.bulletin_a));
}

/// C04 against `finals2000A`'s Bulletin B, day by day: two IERS products, the second built from an
/// earlier combination, so they agree as closely as their vintages do and no closer.
///
/// The bounds are the excerpt's measured worst cases, rounded outward. Since 2023 the B columns are
/// C04 of an earlier date: `ΔUT1` and `dX, dY` identical in 2025 (`crates/spacelab` holds those
/// days), the pole within 0.05 mas. Around 2000 and 2017, a tenth of a milliarcsecond. In 1973 the
/// B columns are an older combination than 20 C04, and the series differ by 20 mas and 1.6 ms. A
/// column read one place off, or `dX` taken as milliarcseconds, misses these by orders of magnitude.
#[test]
fn c04_and_bulletin_b_describe_the_same_pole_to_their_vintage() {
    let c04 = c04();
    let finals = EopSeries::parse_finals2000a(FINALS).expect("the finals excerpt parses");
    let mut compared = [0_u32; 2];

    for sample in finals.samples() {
        let Some(b) = sample.bulletin_b else { continue };
        let Some(row) = c04
            .samples()
            .iter()
            .find(|x| (x.mjd - sample.mjd).abs() < 1e-9)
        else {
            continue;
        };
        let c = row.best();
        let mas = |x: f64, y: f64| (x - y).abs() * 1e3;
        let pole = mas(b.xp.arcseconds(), c.xp.arcseconds())
            .max(mas(b.yp.arcseconds(), c.yp.arcseconds()));
        let offsets = mas(b.dx.arcseconds(), c.dx.arcseconds())
            .max(mas(b.dy.arcseconds(), c.dy.arcseconds()));
        let ut1_us = (b.dut1_seconds - c.dut1_seconds).abs() * 1e6;

        let (era, pole_max, ut1_max, offsets_max) = if sample.mjd < 50_000.0 {
            (0, 25.0, 2_000.0, 25.0)
        } else {
            (1, 0.5, 50.0, 0.3)
        };
        assert!(
            pole < pole_max,
            "MJD {}: pole differs by {pole:.4} mas",
            sample.mjd
        );
        assert!(
            ut1_us < ut1_max,
            "MJD {}: dUT1 differs by {ut1_us:.2} us",
            sample.mjd
        );
        assert!(
            offsets < offsets_max,
            "MJD {}: dX, dY differ by {offsets:.4} mas",
            sample.mjd
        );
        compared[era] += 1;
    }
    assert_eq!(compared, [6, 18], "rows compared, 1973 and 1999-2017");
}

/// The 2017 leap second puts a one-second step in C04's `ΔUT1` at a tabulated midnight, exactly as
/// in `finals2000A`. Interpolated inside the day before it, the two products must agree at the
/// level their rows do -- a smeared step would put them half a second apart.
#[test]
fn the_leap_second_is_not_smeared() {
    let c04 = c04();
    let finals = EopSeries::parse_finals2000a(FINALS).expect("the finals excerpt parses");
    for mjd in [57_752.5, 57_753.25, 57_753.5, 57_753.75, 57_754.5] {
        let a = c04.at_mjd(mjd).expect("inside the excerpt");
        let b = finals.at_mjd(mjd).expect("inside the excerpt");
        let us = (a.dut1_seconds - b.dut1_seconds).abs() * 1e6;
        assert!(us < 20.0, "MJD {mjd}: dUT1 differs by {us:.2} us");
    }
}

#[test]
fn every_row_is_final() {
    assert!(
        c04()
            .samples()
            .iter()
            .all(|x| x.best().quality == EopQuality::Final)
    );
}

/// The excerpt is two runs of days with years between them, and interpolating across a hole is an
/// invention rather than a measurement.
#[test]
fn a_hole_between_runs_is_refused() {
    assert!(matches!(
        c04().at_mjd(50_000.0),
        Err(EopError::Gap {
            after: 41_689.0,
            before: 51_540.0
        })
    ));
}

/// The IERS publishes 20 C04 against IAU 1980 as well, where columns 63-86 are `dψ, dε`. Read as
/// `dX, dY` they would be wrong by their whole size, so a file that names that model is refused at
/// its model line.
#[test]
fn a_series_against_another_nutation_model_is_refused() {
    let iau1980 = C04.replace(
        "Reference Precession-Nutation Model: IAU 2000",
        "Reference Precession-Nutation Model: IAU 1980",
    );
    assert!(matches!(
        EopSeries::parse_c04(&iau1980),
        Err(EopError::Header { line: 4, .. })
    ));
}

/// A file whose header was stripped says nothing about which model its offsets refer to, and is
/// refused at its first record rather than read on trust.
#[test]
fn records_without_the_model_line_are_refused() {
    let stripped = C04
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(matches!(
        EopSeries::parse_c04(&stripped),
        Err(EopError::Header { line: 1, .. })
    ));
}

/// The ITRF line is a statement the file makes; without it the frame is absent, not assumed.
#[test]
fn a_file_without_the_itrf_line_leaves_the_frame_unstated() {
    let unstated = C04
        .lines()
        .filter(|l| !l.contains("consistent with ITRF"))
        .collect::<Vec<_>>()
        .join("\n");
    let s = EopSeries::parse_c04(&unstated).expect("the records still parse");
    assert_eq!(s.frame(), ReferenceFrame::Unstated);
    assert_eq!(s.samples().len(), c04().samples().len());
}

#[test]
fn malformed_and_empty_files_are_refused() {
    let header = C04.lines().take(6).collect::<Vec<_>>().join("\n");
    assert_eq!(EopSeries::parse_c04(&header), Err(EopError::Empty));

    // Line 7 is the first record; its MJD field is columns 17-26.
    let corrupt = C04.replacen("41684.00", "41684.x0", 1);
    assert_eq!(
        EopSeries::parse_c04(&corrupt),
        Err(EopError::Field {
            line: 7,
            columns: 16..26
        })
    );

    let mut lines: Vec<&str> = C04.lines().collect();
    lines.swap(7, 8);
    let swapped = lines.join("\n");
    assert_eq!(
        EopSeries::parse_c04(&swapped),
        Err(EopError::NotMonotonic { line: 9 })
    );
}
