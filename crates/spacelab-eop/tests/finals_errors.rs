//! The `finals2000A` error paths the happy-path parser tests don't reach: every `EopError` message,
//! the empty-file and out-of-order refusals, and an out-of-span lookup.

#![allow(clippy::unwrap_used)]

use spacelab_eop::{EopError, EopSeries};

const EXCERPT: &str = include_str!("fixtures/finals2000A_excerpt.txt");

#[test]
fn every_eop_error_prints_a_distinct_non_empty_message() {
    let errors = [
        EopError::Field {
            line: 5,
            columns: 10..18,
        },
        EopError::Empty,
        EopError::Header {
            line: 4,
            expected: "Reference Precession-Nutation Model: IAU 2000",
        },
        EopError::NotMonotonic { line: 12 },
        EopError::Gap {
            after: 58_000.0,
            before: 58_002.0,
        },
        EopError::OutOfRange {
            mjd: 0.0,
            first: 41_684.0,
            last: 60_000.0,
        },
    ];
    let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
    assert!(messages.iter().all(|m| !m.is_empty()));
    let distinct: std::collections::HashSet<&String> = messages.iter().collect();
    assert_eq!(
        distinct.len(),
        errors.len(),
        "duplicate message: {messages:?}"
    );
}

#[test]
fn a_file_with_no_usable_rows_is_rejected() {
    assert!(matches!(
        EopSeries::parse_finals2000a(""),
        Err(EopError::Empty)
    ));
    // Lines with no pole flag are reserved-but-empty days and are skipped, so a file of only those is
    // Empty too.
    assert!(matches!(
        EopSeries::parse_finals2000a("\n\n"),
        Err(EopError::Empty)
    ));
}

#[test]
fn out_of_order_rows_are_rejected() {
    let lines: Vec<&str> = EXCERPT.lines().collect();
    // The first two data rows, reversed: the second now steps backwards in MJD.
    let reversed = format!("{}\n{}", lines[1], lines[0]);
    assert!(matches!(
        EopSeries::parse_finals2000a(&reversed),
        Err(EopError::NotMonotonic { line: 2 }),
    ));
}

#[test]
fn a_date_outside_the_span_is_refused() {
    let series = EopSeries::parse_finals2000a(EXCERPT).unwrap();
    assert!(matches!(
        series.at_mjd(0.0),
        Err(EopError::OutOfRange { .. })
    ));
    let (_, last) = series.span();
    assert!(matches!(
        series.at_mjd(last + 1000.0),
        Err(EopError::OutOfRange { .. }),
    ));
}
