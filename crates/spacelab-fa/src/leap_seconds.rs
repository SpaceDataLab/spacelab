//! Leap seconds, and the conversion between the UTC calendar and TAI.
//!
//! # Provenance
//!
//! The table below is the IERS's own list of leap seconds, `leap-seconds.list`
//! (Earth Orientation Centre, Paris Observatory; in the public domain), updated
//! through Bulletin C 72 and valid until 2027-06-28. It is a list of measurements
//! of the Earth's rotation — facts, not expression. `tests/leap_second_list.rs`
//! holds the table to the list on every day from 1972-01-01 to its expiry.
//! `tests/utc_properties.rs` holds the conversions to their definitions, to the
//! nanosecond, across every leap second and past 2027-05-20.
//!
//! # The table is closed
//!
//! The last row, `TAI − UTC = 37 s` from 2017-01-01, holds for every later date.
//! Two documents close the table:
//!
//! 1. The IERS list has no leap second after 2016 and is valid until 2027-06-28.
//! 2. CGPM 2026 Resolution 5: "continuous UTC will become effective on 20 May
//!    2027". From that date no leap second is inserted.
//!
//! The list is valid past the start of continuity, so no date is left open. If
//! the IERS issues a list with a later row, replace the fixture:
//! `tests/leap_second_list.rs` then fails until the table has that row. A
//! negative leap second would need code as well as a row: [`tai_to_utc`] assumes
//! every step adds one second.
//!
//! After 2027-05-20, `UT1 − UTC` is no longer held below 0.9 s. The resolution
//! bounds `|UT1 − UTC|` by 3600 s and expects the bound to hold for several
//! centuries. It defines no adjustment after that, and this table has none.

use core::fmt;

use spacelab_core::{
    Epoch, Tai, UtcDateTime,
    calendar::{civil_from_days, days_from_civil},
    time::NS_PER_DAY,
};

/// One entry: from 00:00:00 UTC on this date, `TAI - UTC` takes this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    year: i32,
    month: u8,
    delta_at: i32,
}

const fn e(year: i32, month: u8, delta_at: i32) -> Entry {
    Entry {
        year,
        month,
        delta_at,
    }
}

/// Every leap second ever inserted, from the start of the integer-second era.
///
/// All 28 take effect on the first of the month. Before 1972, UTC ran at a
/// different rate from TAI and `TAI - UTC` was a fractional, drifting quantity;
/// we decline to model that era rather than model it badly.
const TABLE: [Entry; 28] = [
    e(1972, 1, 10),
    e(1972, 7, 11),
    e(1973, 1, 12),
    e(1974, 1, 13),
    e(1975, 1, 14),
    e(1976, 1, 15),
    e(1977, 1, 16),
    e(1978, 1, 17),
    e(1979, 1, 18),
    e(1980, 1, 19),
    e(1981, 7, 20),
    e(1982, 7, 21),
    e(1983, 7, 22),
    e(1985, 7, 23),
    e(1988, 1, 24),
    e(1990, 1, 25),
    e(1991, 1, 26),
    e(1992, 7, 27),
    e(1993, 7, 28),
    e(1994, 7, 29),
    e(1996, 1, 30),
    e(1997, 7, 31),
    e(1999, 1, 32),
    e(2006, 1, 33),
    e(2009, 1, 34),
    e(2012, 7, 35),
    e(2015, 7, 36),
    e(2017, 1, 37),
];

/// Why a UTC instant could not be converted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeapError {
    /// Before 1972, `TAI - UTC` was fractional and drifting. Not modelled.
    BeforeIntegerSecondEra,
    /// A `23:59:60` label on a day that had no leap second.
    SpuriousLeapSecond,
}

impl fmt::Display for LeapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BeforeIntegerSecondEra => {
                write!(f, "before 1972-01-01, TAI-UTC was fractional; not modelled")
            }
            Self::SpuriousLeapSecond => {
                write!(f, "second 60 labelled on a day with no leap second")
            }
        }
    }
}

impl core::error::Error for LeapError {}

/// Naive nanoseconds from J2000 to 00:00:00 UTC on a date.
fn midnight_naive_ns(year: i32, month: u8, day: u8) -> i128 {
    const UNIX_DAYS_TO_J2000: i64 = 10_957;
    const NS_MIDNIGHT_TO_NOON: i128 = 12 * 3_600 * 1_000_000_000;
    i128::from(days_from_civil(year, month, day) - UNIX_DAYS_TO_J2000) * NS_PER_DAY
        - NS_MIDNIGHT_TO_NOON
}

/// Naive nanoseconds from J2000 to 00:00:00 on the first of a month.
fn month_start_naive_ns(year: i32, month: u8) -> i128 {
    midnight_naive_ns(year, month, 1)
}

/// `TAI - UTC`, in whole seconds, for a UTC date.
///
/// Keyed on the **calendar date**, never on the naive coordinate. A `23:59:60`
/// label deliberately shares its naive coordinate with the following midnight —
/// that aliasing is what lets the leap second fall out of the arithmetic — so
/// using the coordinate here would look up tomorrow's `ΔAT` and be wrong by
/// exactly one second, once a year.
///
/// Every date from 1972-01-01 on has an answer: the table is closed, and its last
/// row holds past 2027-05-20, when UTC becomes continuous (module documentation).
///
/// # Errors
///
/// [`LeapError::BeforeIntegerSecondEra`] before 1972-01-01.
pub fn delta_at(utc: UtcDateTime) -> Result<i32, LeapError> {
    let midnight = midnight_naive_ns(utc.year(), utc.month(), utc.day());

    if midnight < month_start_naive_ns(TABLE[0].year, TABLE[0].month) {
        return Err(LeapError::BeforeIntegerSecondEra);
    }

    let mut delta = TABLE[0].delta_at;
    for entry in TABLE {
        if midnight >= month_start_naive_ns(entry.year, entry.month) {
            delta = entry.delta_at;
        } else {
            break;
        }
    }
    Ok(delta)
}

/// Whether a leap second was inserted at the end of this UTC day.
fn day_ends_with_leap_second(utc: UtcDateTime) -> bool {
    let tomorrow = civil_from_days(utc.days_from_civil() + 1);
    TABLE
        .iter()
        .any(|entry| entry.year == tomorrow.0 && entry.month == tomorrow.1 && tomorrow.2 == 1)
}

/// The UTC calendar to TAI.
///
/// `TAI = naive(UTC) + ΔAT`, where `naive` counts every UTC day as exactly 86400
/// seconds. A `23:59:60` label lands 86400 seconds into its day, so the leap
/// second requires no special case in the arithmetic — only in the validation.
///
/// # Errors
///
/// [`LeapError`], including [`LeapError::SpuriousLeapSecond`] for a `:60` label on
/// a day that had none.
pub fn utc_to_tai(utc: UtcDateTime) -> Result<Epoch<Tai>, LeapError> {
    if utc.is_leap_second_label() && !day_ends_with_leap_second(utc) {
        return Err(LeapError::SpuriousLeapSecond);
    }
    let delta = delta_at(utc)?;
    let ns = utc.naive_ns_since_j2000() + i128::from(delta) * 1_000_000_000;
    Ok(Epoch::from_ns_since_j2000(ns))
}

/// TAI to the UTC calendar.
///
/// The inverse is not a function of the naive coordinate alone. During a leap
/// second, two distinct TAI instants map to naive coordinates one second apart
/// but must be labelled `23:59:60` and `00:00:00` of the following day. We detect
/// this by asking whether the naive coordinate has already reached the next
/// entry's effective instant while `ΔAT` still holds its old value.
///
/// # Errors
///
/// [`LeapError`] as for [`delta_at`].
pub fn tai_to_utc(tai: Epoch<Tai>) -> Result<UtcDateTime, LeapError> {
    let tai_ns = tai.ns_since_j2000();

    // Each entry becomes effective at TAI = naive(effective date) + its own ΔAT.
    // ΔAT is monotone, so the last entry whose TAI instant we have passed is ours.
    let mut chosen: Option<Entry> = None;
    for entry in TABLE {
        let effective_tai = month_start_naive_ns(entry.year, entry.month)
            + i128::from(entry.delta_at) * 1_000_000_000;
        if tai_ns >= effective_tai {
            chosen = Some(entry);
        } else {
            break;
        }
    }
    let Some(entry) = chosen else {
        return Err(LeapError::BeforeIntegerSecondEra);
    };

    let naive = tai_ns - i128::from(entry.delta_at) * 1_000_000_000;

    // Are we inside the leap second itself? That is: has the naive coordinate
    // reached the *next* entry's effective date, while ΔAT is still the old value?
    let next = TABLE.iter().find(|c| {
        month_start_naive_ns(c.year, c.month) + i128::from(c.delta_at) * 1_000_000_000 > tai_ns
    });
    let inside_leap = next.is_some_and(|c| naive >= month_start_naive_ns(c.year, c.month));

    Ok(if inside_leap {
        // Render one second earlier, then push the label from :59 to :60.
        UtcDateTime::from_naive_ns_since_j2000(naive - 1_000_000_000, true)
    } else {
        UtcDateTime::from_naive_ns_since_j2000(naive, false)
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // a panicking test is a failing test

    use super::*;

    fn utc(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> UtcDateTime {
        UtcDateTime::new(year, month, day, hour, minute, second).unwrap()
    }

    #[test]
    fn the_era_begins_at_ten_seconds() {
        assert_eq!(delta_at(utc(1972, 1, 1, 0, 0, 0.0)), Ok(10));
        assert_eq!(
            delta_at(utc(1971, 12, 31, 23, 59, 59.0)),
            Err(LeapError::BeforeIntegerSecondEra)
        );
    }

    #[test]
    fn delta_at_steps_exactly_at_midnight() {
        // The last instant of 2016 still has 36; the first of 2017 has 37.
        assert_eq!(delta_at(utc(2016, 12, 31, 23, 59, 59.0)), Ok(36));
        assert_eq!(delta_at(utc(2017, 1, 1, 0, 0, 0.0)), Ok(37));
    }

    #[test]
    fn the_last_row_holds_past_the_start_of_continuous_utc() {
        for (year, month, day) in [
            (2027, 5, 19),
            (2027, 5, 20),
            (2027, 7, 1),
            (2100, 1, 1),
            (9999, 12, 31),
        ] {
            assert_eq!(
                delta_at(utc(year, month, day, 0, 0, 0.0)),
                Ok(37),
                "{year}-{month:02}-{day:02}"
            );
        }
        // Bulletin C 72 excluded the first; continuity excludes the second.
        for (year, month, day) in [(2026, 12, 31), (2027, 6, 30)] {
            assert_eq!(
                utc_to_tai(utc(year, month, day, 23, 59, 60.0)),
                Err(LeapError::SpuriousLeapSecond),
                "{year}-{month:02}-{day:02}"
            );
        }
    }

    /// Every step adds one second, which [`tai_to_utc`] assumes, and every leap second ended
    /// June or December.
    #[test]
    fn every_leap_second_adds_one_second_at_the_end_of_june_or_december() {
        for pair in TABLE.windows(2) {
            let (before, after) = (pair[0], pair[1]);
            assert_eq!(after.delta_at - before.delta_at, 1, "{after:?}");
            assert!(matches!(after.month, 1 | 7), "{after:?}");
            assert!(
                (after.year, after.month) > (before.year, before.month),
                "{after:?}"
            );
        }
    }

    #[test]
    fn a_leap_second_is_one_tai_second_before_the_new_year() {
        // 2016-12-31T23:59:60 UTC is 2017-01-01T00:00:36 TAI, one second before
        // 2017-01-01T00:00:00 UTC, which is 00:00:37 TAI.
        let leap = utc_to_tai(utc(2016, 12, 31, 23, 59, 60.0)).unwrap();
        let after = utc_to_tai(utc(2017, 1, 1, 0, 0, 0.0)).unwrap();
        assert_eq!(
            after.ns_since_j2000() - leap.ns_since_j2000(),
            1_000_000_000
        );
    }

    #[test]
    fn a_spurious_leap_second_is_rejected() {
        // 2018 ended without one.
        assert_eq!(
            utc_to_tai(utc(2018, 12, 31, 23, 59, 60.0)),
            Err(LeapError::SpuriousLeapSecond)
        );
        assert!(utc_to_tai(utc(2016, 12, 31, 23, 59, 60.0)).is_ok());
    }

    #[test]
    fn tai_to_utc_recovers_the_leap_second_label() {
        let leap = utc_to_tai(utc(2016, 12, 31, 23, 59, 60.0)).unwrap();
        let back = tai_to_utc(leap).unwrap();
        assert_eq!(back.day(), 31);
        assert_eq!(back.month(), 12);
        assert_eq!(back.year(), 2016);
        assert!((back.second() - 60.0).abs() < 1e-9, "got {}", back.second());
    }

    #[test]
    fn utc_tai_round_trips_across_every_transition() {
        for entry in TABLE.iter().skip(1) {
            for offset_s in [-2_i8, -1, 0, 1, 2] {
                let base = utc(entry.year, entry.month, 1, 0, 0, 0.0);
                let ns = base.naive_ns_since_j2000() + i128::from(offset_s) * 1_000_000_000;
                let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
                let Ok(tai) = utc_to_tai(label) else { continue };
                let back = tai_to_utc(tai).unwrap();
                assert_eq!(back.year(), label.year());
                assert_eq!(back.day(), label.day());
                assert!(
                    (back.second() - label.second()).abs() < 1e-9,
                    "at {label}: got {back}"
                );
            }
        }
    }
}
