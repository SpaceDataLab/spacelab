//! UTC↔TAI conversion round-trips and the out-of-table refusals.
//!
//! `delta_at`/`utc_to_tai` are exercised by the umbrella crate, but the reverse `tai_to_utc`, its
//! refusal before the 1972 integer-second era, its answer past the end of leap seconds and the
//! `LeapError` messages had no direct test. (The `const fn e` table builder runs only at compile time, so it is
//! not reachable by runtime coverage.)

#![allow(clippy::unwrap_used)]

use spacelab_core::{Epoch, Tai, UtcDateTime};
use spacelab_fa::{LeapError, tai_to_utc, utc_to_tai};

#[test]
fn utc_and_tai_round_trip() {
    for (y, mo, d, h, mi, s) in [
        (1972, 1, 1, 0, 0, 0.0),     // the first table entry
        (2000, 1, 1, 12, 0, 0.0),    // J2000
        (2016, 12, 30, 6, 0, 0.0),   // a recent measured day
        (2020, 6, 15, 18, 30, 45.0), // an arbitrary instant
    ] {
        let utc = UtcDateTime::new(y, mo, d, h, mi, s).unwrap();
        let tai = utc_to_tai(utc).unwrap();
        let back = tai_to_utc(tai).unwrap();
        assert_eq!(back, utc, "round trip at {y}-{mo:02}-{d:02}");
    }
}

#[test]
fn tai_to_utc_refuses_before_the_table_and_answers_after_it() {
    // Before the 1972 integer-second era: no entry applies.
    assert_eq!(
        tai_to_utc(Epoch::<Tai>::from_ns_since_j2000(
            -2_000_000_000_000_000_000
        )),
        Err(LeapError::BeforeIntegerSecondEra),
    );
    // 2031-09, after UTC becomes continuous on 2027-05-20: exactly TAI - 37 s.
    let tai_ns = 1_000_000_000_000_000_000;
    assert_eq!(
        tai_to_utc(Epoch::<Tai>::from_ns_since_j2000(tai_ns)),
        Ok(UtcDateTime::from_naive_ns_since_j2000(
            tai_ns - 37_000_000_000,
            false
        )),
    );
}

#[test]
fn each_leap_error_prints_a_distinct_non_empty_message() {
    let errors = [
        LeapError::BeforeIntegerSecondEra,
        LeapError::SpuriousLeapSecond,
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
