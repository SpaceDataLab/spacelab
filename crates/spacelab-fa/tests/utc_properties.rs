//! UTC ↔ TAI ↔ TT on integer nanoseconds, across every leap second and past the end of leap
//! seconds on 2027-05-20.
//!
//! The comparator in `delta_at_oracle.rs` checks the table at dates somebody chose. These check
//! what the conversions *are*, wherever the draw lands:
//!
//! 1. **TAI → TT → TAI → UTC → TAI is the identity** on every integer nanosecond from 1972 on,
//!    including the nanoseconds inside each leap second, and `TT − TAI` is 32.184 s exactly.
//! 2. **UTC → TAI → UTC is the identity** on every label, including each `23:59:60.xxx`.
//! 3. **UTC labels increase with TAI.** With 1 and 2 this makes the conversion an order-preserving
//!    bijection, so it is fixed up to a constant shift; a swap of the leap second with the second
//!    after it passes both round trips and fails here.
//! 4. **The shift and the day lengths.** `TAI − UTC` is 10 s at 1972-01-01, by the definition of
//!    the integer-second era. Every UTC day before 2500 lasts 86 400 SI seconds, except exactly 27
//!    that last 86 401: each a June 30 or a December 31, none after 2016, and only these accept a
//!    `23:59:60` label. With 3 this fixes every label.
//! 5. **Continuity.** From 2017-01-01 on, UTC is TAI − 37 s exactly, across 2027-05-20.
//! 6. **The edges.** Each leap second runs from `23:59:60` to the next midnight, nanosecond-exact.
//!
//! A uniform draw over the 55 years of leap seconds lands inside one with probability 1.6e-8, so
//! half the draws sit within three seconds of a transition, and a sixth of those fall inside it.
//! A draw lands on an edge with probability 1e-9, so 6 enumerates the edges instead.

#![allow(clippy::unwrap_used)]

use proptest::prelude::*;
use spacelab_core::{
    Epoch, Tai, UtcDateTime,
    calendar::civil_from_days,
    time::{tai_to_tt, tt_to_tai},
};
use spacelab_fa::{LeapError, delta_at, tai_to_utc, utc_to_tai};

const S: i128 = 1_000_000_000;

/// The exhaustive day scan and the wide draws stop before this year.
const END_YEAR: i32 = 2500;

fn naive(year: i32, month: u8, day: u8) -> i128 {
    UtcDateTime::from_date(year, month, day)
        .unwrap()
        .naive_ns_since_j2000()
}

fn tai(utc: UtcDateTime) -> i128 {
    utc_to_tai(utc).unwrap().ns_since_j2000()
}

fn utc(tai_ns: i128) -> UtcDateTime {
    tai_to_utc(Epoch::<Tai>::from_ns_since_j2000(tai_ns)).unwrap()
}

/// Each month start where `TAI − UTC` steps, as `(naive, TAI)` of its 00:00:00 UTC.
fn transitions() -> Vec<(i128, i128)> {
    let mut found = Vec::new();
    let mut previous = delta_at(UtcDateTime::from_date(1972, 1, 1).unwrap()).unwrap();
    for year in 1972..=2027 {
        for month in 1..=12 {
            let start = UtcDateTime::from_date(year, month, 1).unwrap();
            let d = delta_at(start).unwrap();
            if d != previous {
                found.push((start.naive_ns_since_j2000(), tai(start)));
                previous = d;
            }
        }
    }
    found
}

/// Orders labels as the calendar does; the second is exact to the nanosecond.
#[allow(clippy::cast_possible_truncation)] // second < 61, so the product is far below i64
fn key(u: UtcDateTime) -> (i32, u8, u8, u8, u8, i64) {
    let second_ns = (u.second() * 1e9).round() as i64;
    (
        u.year(),
        u.month(),
        u.day(),
        u.hour(),
        u.minute(),
        second_ns,
    )
}

/// Within three seconds of a transition, within 400 days of 2027-05-20, or anywhere before 2500,
/// in the coordinate `at` picks from a transition's `(naive, TAI)` pair.
fn near(at: fn((i128, i128)) -> i128, continuity: i128, from: i128) -> impl Strategy<Value = i128> {
    let steps: Vec<i128> = transitions().into_iter().map(at).collect();
    let end = at((naive(END_YEAR, 1, 1), naive(END_YEAR, 1, 1) + 37 * S));
    prop_oneof![
        2 => (0..steps.len(), -3 * S..3 * S).prop_map(move |(k, offset)| steps[k] + offset),
        1 => (-400 * 86_400 * S..400 * 86_400 * S).prop_map(move |offset| continuity + offset),
        1 => from..end,
    ]
}

/// A TAI instant from the start of the integer-second era on.
fn instant() -> impl Strategy<Value = i128> {
    near(
        |(_, t)| t,
        naive(2027, 5, 20) + 37 * S,
        naive(1972, 1, 1) + 10 * S,
    )
}

/// A label that is not inside a leap second.
fn ordinary_label() -> impl Strategy<Value = UtcDateTime> {
    near(|(n, _)| n, naive(2027, 5, 20), naive(1972, 1, 1))
        .prop_map(|n| UtcDateTime::from_naive_ns_since_j2000(n, false))
}

/// A label inside a leap second: `23:59:60` plus a nanosecond fraction, on a day that had one.
fn leap_label() -> impl Strategy<Value = UtcDateTime> {
    let starts: Vec<i128> = transitions().into_iter().map(|(n, _)| n).collect();
    (0..starts.len(), 0..S).prop_map(move |(k, fraction)| {
        UtcDateTime::from_naive_ns_since_j2000(starts[k] - S + fraction, true)
    })
}

proptest! {
    // About twelve draws land inside each leap second at 4096 cases; under one at the default 256.
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn tai_tt_tai_utc_tai_is_the_identity(t in instant()) {
        let tt = tai_to_tt(Epoch::<Tai>::from_ns_since_j2000(t));
        prop_assert_eq!(tt.ns_since_j2000() - t, 32_184_000_000);
        let back = tt_to_tai(tt).ns_since_j2000();
        prop_assert_eq!(tai(utc(back)), t);
    }

    #[test]
    fn utc_tai_utc_is_the_identity(label in ordinary_label(), leap in leap_label()) {
        prop_assert_eq!(utc(tai(label)), label);
        prop_assert!(leap.is_leap_second_label());
        prop_assert_eq!(utc(tai(leap)), leap);
    }

    #[test]
    fn utc_labels_increase_with_tai(t in instant(), dt in 1..2 * S) {
        let (a, b) = (utc(t), utc(t + dt));
        prop_assert!(key(a) < key(b), "{} then {}", a, b);
    }

    #[test]
    fn utc_is_tai_minus_37_s_from_2017(
        t in naive(2017, 1, 1) + 37 * S..naive(END_YEAR, 1, 1) + 37 * S,
    ) {
        prop_assert_eq!(utc(t), UtcDateTime::from_naive_ns_since_j2000(t - 37 * S, false));
    }
}

#[test]
fn every_leap_second_runs_from_23_59_60_to_midnight() {
    let steps = transitions();
    assert_eq!(steps.len(), 27);
    for (n, t) in steps {
        let next = UtcDateTime::from_naive_ns_since_j2000(n, false);
        let (y, m, d) = civil_from_days(next.days_from_civil() - 1);
        let eve = |second_ns: i64| (y, m, d, 23, 59, second_ns);
        let edges = [
            (t - S - 1, eve(59_999_999_999)),
            (t - S, eve(60_000_000_000)),
            (t - 1, eve(60_999_999_999)),
            (t, (next.year(), next.month(), 1, 0, 0, 0)),
        ];
        for (at, label) in edges {
            assert_edge(at, label);
        }
    }
}

fn assert_edge(at: i128, label: (i32, u8, u8, u8, u8, i64)) {
    let u = utc(at);
    assert_eq!(key(u), label, "TAI {at} ns");
    assert_eq!(tai(u), at, "TAI {at} ns came back from {u}");
}

#[test]
fn twenty_seven_utc_days_are_one_second_long_and_none_after_2016() {
    let first = UtcDateTime::from_date(1972, 1, 1).unwrap();
    let mut midnight = tai(first);
    assert_eq!(midnight, naive(1972, 1, 1) + 10 * S);

    let last = UtcDateTime::from_date(END_YEAR, 1, 1)
        .unwrap()
        .days_from_civil();
    let mut long_days = Vec::new();
    for day in first.days_from_civil()..last {
        let (y, m, d) = civil_from_days(day);
        let (ny, nm, nd) = civil_from_days(day + 1);
        let next = tai(UtcDateTime::from_date(ny, nm, nd).unwrap());
        let sixty = utc_to_tai(UtcDateTime::new(y, m, d, 23, 59, 60.0).unwrap());
        match next - midnight {
            l if l == 86_400 * S => assert_eq!(sixty, Err(LeapError::SpuriousLeapSecond)),
            l if l == 86_401 * S => {
                assert!(sixty.is_ok(), "{y}-{m:02}-{d:02}");
                long_days.push((y, m, d));
            }
            l => panic!("{y}-{m:02}-{d:02} lasts {l} ns"),
        }
        midnight = next;
    }
    assert_eq!(long_days.len(), 27, "{long_days:?}");
    assert!(
        long_days
            .iter()
            .all(|&(y, m, d)| y <= 2016 && matches!((m, d), (6, 30) | (12, 31))),
        "{long_days:?}"
    );
}
