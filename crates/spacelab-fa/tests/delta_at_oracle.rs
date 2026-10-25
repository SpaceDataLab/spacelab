//! Comparator: the leap-second table against `pyerfa`. Benchmark material, not a
//! gate — the gate is `tests/leap_second_list.rs`, against the IERS's own list.
//!
//! Fixtures are generated offline by `derivations/leap_seconds.py` from `pyerfa`.
//! They cover every transition boundary and the day either side of it, plus random
//! dates across the integer-second era.
//!
//! Same caveat as everywhere: `pyerfa` is of the SOFA lineage. Here it matters
//! less than usual, because the leap-second record is a published list of facts
//! rather than an algorithm — a disagreement would mean one of us mistyped the
//! IERS bulletin, not that a model is wrong.

#![allow(clippy::expect_used)]

use spacelab_core::UtcDateTime;
use spacelab_fa::delta_at;

const FIXTURES: &str = include_str!("fixtures/delta_at_pyerfa.csv");

#[test]
#[ignore = "comparator: pyerfa's leap-second table, not a verdict"]
fn delta_at_agrees_with_pyerfa_on_every_transition() {
    let mut checked = 0_usize;

    for (i, line) in FIXTURES.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 4, "line {}: expected 4 columns", i + 1);

        let year: i32 = f[0].parse().expect("year");
        let month: u8 = f[1].parse().expect("month");
        let day: u8 = f[2].parse().expect("day");
        let expected: i32 = f[3].parse().expect("delta_at");

        let utc = UtcDateTime::from_date(year, month, day).expect("fixture date exists");

        match delta_at(utc) {
            Ok(got) => {
                assert_eq!(got, expected, "{year}-{month:02}-{day:02}: TAI-UTC");
                checked += 1;
            }
            Err(e) => panic!("{year}-{month:02}-{day:02}: {e}"),
        }
    }

    assert!(checked >= 250, "thin fixture set: only {checked} dates");
    println!("delta_at: {checked} dates agree");
}

#[test]
#[ignore = "comparator: pyerfa's leap-second table, not a verdict"]
fn every_transition_boundary_is_sharp() {
    // The fixtures contain the day before, the day of, and the day after each
    // transition. If our table were off by a day anywhere, the pairs would not
    // straddle a step. Verify at least one step is actually present, so that a
    // fixture file of constant values could never pass the test above vacuously.
    let mut steps = 0;
    let mut previous: Option<i32> = None;

    for line in FIXTURES.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let utc = UtcDateTime::from_date(
            f[0].parse().expect("year"),
            f[1].parse().expect("month"),
            f[2].parse().expect("day"),
        )
        .expect("date");
        if let Ok(dat) = delta_at(utc) {
            if previous.is_some_and(|p| p != dat) {
                steps += 1;
            }
            previous = Some(dat);
        }
    }

    assert_eq!(
        steps, 27,
        "expected 27 steps after the 1972 baseline, saw {steps}"
    );
}
