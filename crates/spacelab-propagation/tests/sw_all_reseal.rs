//! `SW-All.csv` as Celestrak publishes it after moving its sources to the primary providers on
//! 2026-07-06 (`plans/17` Ф20.N, F51).
//!
//! [`parse_space_weather`] reads its columns by position. These tests seal those positions, the
//! blank-field conventions and the boundary between measured and forecast days on an excerpt of a file
//! fetched after the switch:
//!
//! * `celestrak_sw_2026-11-12.csv` — 19 rows and the header of `https://celestrak.org/SpaceData/SW-All.csv`
//!   retrieved 2026-11-12T07:21:35Z (25 468 lines, sha256 `936351953c3b…a36d01bf`), byte for byte with
//!   the provider's CRLF line ends: the last interpolated day, the week of the switch, the last measured
//!   days and the first forecast ones, the end of the daily forecast, and the monthly means after it.
//!
//! A file fetched on 2026-08-10 had one more convention, not in this one: its last day before the daily
//! forecast was a `PRM` row with full 3-hourly indices and a flux of 122.7 between days of 93.7 and 87.6.
//! [`FluxQuality`] keeps such a day visible, as it does every row this excerpt holds.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use spacelab_propagation::space_weather::{FluxQuality, find, parse_space_weather};
use spacelab_testkit::sha256_hex;

const EXCERPT: &str = include_str!("fixtures/celestrak_sw_2026-11-12.csv");

/// The 31 columns of the header, in order.
const HEADER: [&str; 31] = [
    "DATE",
    "BSRN",
    "ND",
    "KP1",
    "KP2",
    "KP3",
    "KP4",
    "KP5",
    "KP6",
    "KP7",
    "KP8",
    "KP_SUM",
    "AP1",
    "AP2",
    "AP3",
    "AP4",
    "AP5",
    "AP6",
    "AP7",
    "AP8",
    "AP_AVG",
    "CP",
    "C9",
    "ISN",
    "F10.7_OBS",
    "F10.7_ADJ",
    "F10.7_DATA_TYPE",
    "F10.7_OBS_CENTER81",
    "F10.7_OBS_LAST81",
    "F10.7_ADJ_CENTER81",
    "F10.7_ADJ_LAST81",
];

/// Each data row as `(name, value)` pairs, looked up by the header's names rather than by position.
fn rows_by_name() -> Vec<Vec<(&'static str, &'static str)>> {
    let mut lines = EXCERPT.lines();
    let header: Vec<&str> = lines.next().unwrap().trim_end().split(',').collect();
    lines
        .map(|line| {
            header
                .iter()
                .copied()
                .zip(line.trim_end().split(','))
                .collect()
        })
        .collect()
}

fn field<'a>(row: &[(&str, &'a str)], name: &str) -> &'a str {
    row.iter().find(|(n, _)| *n == name).unwrap().1
}

fn number(row: &[(&str, &str)], name: &str) -> f64 {
    field(row, name).parse().unwrap()
}

fn date(row: &[(&str, &str)]) -> (i32, u32, u32) {
    let mut parts = field(row, "DATE").split('-');
    let mut next = || parts.next().unwrap();
    (
        next().parse().unwrap(),
        next().parse().unwrap(),
        next().parse().unwrap(),
    )
}

#[test]
fn the_excerpt_is_the_sealed_cut_of_2026_11_12() {
    assert_eq!(
        &sha256_hex(EXCERPT.as_bytes()),
        b"7d52d42ccaaff5da8e3dc441d8268195c149f295e44b9ad492521e97a65d9b51"
    );
    assert!(EXCERPT.split_inclusive('\n').all(|l| l.ends_with("\r\n")));
    let header: Vec<&str> = EXCERPT
        .lines()
        .next()
        .unwrap()
        .trim_end()
        .split(',')
        .collect();
    assert_eq!(header, HEADER);
    assert!(rows_by_name().iter().all(|row| row.len() == HEADER.len()));
}

/// The seal on the positions: every field the parser returns is the value in the column its header
/// names, for every row it reads.
#[test]
fn the_parser_reads_each_field_from_the_column_its_header_names() {
    let days = parse_space_weather(EXCERPT);
    let read: Vec<_> = rows_by_name()
        .into_iter()
        .filter(|row| !field(row, "AP_AVG").is_empty())
        .collect();
    assert_eq!(days.len(), read.len());
    for row in &read {
        let (y, m, d) = date(row);
        let day = find(&days, y, m, d).unwrap();
        assert_eq!(day.ap, number(row, "AP_AVG"));
        for i in 0..8 {
            assert_eq!(day.ap3h[i], number(row, &format!("AP{}", i + 1)));
            assert_eq!(day.kp3h[i], number(row, &format!("KP{}", i + 1)) / 10.0);
        }
        assert_eq!(day.f107_observed, number(row, "F10.7_OBS"));
        assert_eq!(day.f107_adjusted, number(row, "F10.7_ADJ"));
        assert_eq!(day.f107_center81, number(row, "F10.7_OBS_CENTER81"));
        assert_eq!(day.f107_last81, number(row, "F10.7_OBS_LAST81"));
    }
}

/// The measured days end on 2026-11-11 and the daily forecast starts the next day; no day after the
/// first forecast one is measured, and the interpolated day is not counted as measured.
#[test]
fn the_measured_series_ends_where_the_daily_forecast_begins() {
    let days = parse_space_weather(EXCERPT);
    let quality = |y, m, d| find(&days, y, m, d).unwrap().quality;
    assert_eq!(quality(2026, 5, 9), FluxQuality::Interpolated);
    assert!(!FluxQuality::Interpolated.is_measured());
    for d in 3..=9 {
        assert_eq!(quality(2026, 7, d), FluxQuality::Observed);
    }
    assert_eq!(quality(2026, 11, 11), FluxQuality::Observed);
    assert_eq!(quality(2026, 11, 12), FluxQuality::Predicted);
    let first_forecast = days
        .iter()
        .position(|d| d.quality == FluxQuality::Predicted)
        .unwrap();
    assert!(
        days[first_forecast..]
            .iter()
            .all(|d| !d.quality.is_measured())
    );
    assert_eq!(
        days.iter()
            .filter(|d| d.quality == FluxQuality::Predicted)
            .count(),
        5
    );
}

/// A monthly mean (`PRM`) carries a flux and no Kp or ap: every geomagnetic column is blank. The parser
/// does not read it, rather than fill its `ap` with a value the file does not give.
#[test]
fn a_monthly_mean_has_no_geomagnetic_index_and_is_not_read() {
    let days = parse_space_weather(EXCERPT);
    let monthly: Vec<_> = rows_by_name()
        .into_iter()
        .filter(|row| field(row, "F10.7_DATA_TYPE") == "PRM")
        .collect();
    assert_eq!(monthly.len(), 3);
    for row in &monthly {
        for name in &HEADER[3..=22] {
            assert_eq!(field(row, name), "", "{name}");
        }
        assert!(number(row, "F10.7_OBS") > 0.0);
        let (y, m, d) = date(row);
        assert!(find(&days, y, m, d).is_none());
    }
    assert_eq!(days.len(), 16);
}
