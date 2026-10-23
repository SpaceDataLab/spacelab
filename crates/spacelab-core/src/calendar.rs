//! The Gregorian calendar, and UTC as a calendar rather than a scale.
//!
//! # Why UTC is not an [`Epoch`](crate::Epoch)
//!
//! Every other time scale we use is uniform: TAI, TT, TDB and UT1 tick without
//! interruption, so an integer count of nanoseconds from a fixed instant names
//! each of their moments exactly once.
//!
//! UTC does not. A day carrying a leap second holds 86401 seconds, and the label
//! `23:59:60` exists. A linear nanosecond counter cannot express it, and a linear
//! counter that ignores it silently mislabels the second either side. Giving UTC
//! an `Epoch<Utc>` would be applying a uniform representation to a non-uniform
//! scale — a lie the type system would then enforce.
//!
//! So UTC is a *calendar*, not a scale. It is [`UtcDateTime`], it is converted to
//! and from [`Epoch<Tai>`](crate::Epoch) with a leap-second table, and that
//! conversion is fallible because the table is finite and the calendar is not.

use core::fmt;

use crate::time::NS_PER_DAY;

/// Days from 1970-01-01 to 2000-01-01. Thirty years and seven leap days.
const UNIX_DAYS_TO_J2000: i64 = 10_957;

/// Nanoseconds from midnight to the J2000 epoch instant, which is at 12:00.
const NS_MIDNIGHT_TO_NOON: i128 = 12 * 3_600 * 1_000_000_000;

/// Something the Gregorian calendar does not admit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalendarError {
    /// Month outside `1..=12`.
    Month(u8),
    /// Day outside `1..=len_of_month`.
    Day {
        /// The year the day was claimed to fall in.
        year: i32,
        /// The month the day was claimed to fall in.
        month: u8,
        /// The offending day.
        day: u8,
    },
    /// Hour outside `0..=23`.
    Hour(u8),
    /// Minute outside `0..=59`.
    Minute(u8),
    /// Second outside `[0, 61)`.
    ///
    /// The upper bound is 61 rather than 60 because a UTC minute may hold a leap
    /// second. Whether *this* minute does is not a calendar question — it needs a
    /// leap-second table, and is checked at conversion.
    Second(f64),
}

impl fmt::Display for CalendarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Month(m) => write!(f, "month {m} is not in 1..=12"),
            Self::Day { year, month, day } => {
                write!(f, "day {day} does not exist in {year}-{month:02}")
            }
            Self::Hour(h) => write!(f, "hour {h} is not in 0..=23"),
            Self::Minute(m) => write!(f, "minute {m} is not in 0..=59"),
            Self::Second(s) => write!(f, "second {s} is not in [0, 61)"),
        }
    }
}

impl core::error::Error for CalendarError {}

/// Days from 1970-01-01 to the given Gregorian date, proleptic, negative before.
///
/// Howard Hinnant, *chrono-Compatible Low-Level Date Algorithms*, released to the
/// public domain. <https://howardhinnant.github.io/date_algorithms.html>
///
/// Exact for every year in `i32`, with no division by a non-constant and no table.
#[must_use]
pub const fn days_from_civil(year: i32, month: u8, day: u8) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let m = month as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era as i64 * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`].
#[must_use]
// `m` lies in 1..=12 and `d` in 1..=31 by construction of the algorithm, and `y`
// is bounded by the input, which came from an `i32` year. The casts cannot lose
// anything; the round-trip property test over a full 400-year cycle proves it.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub const fn civil_from_days(days: i64) -> (i32, u8, u8) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u8, d as u8)
}

/// Whether a proleptic Gregorian year is a leap year.
#[must_use]
pub const fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Number of days in a Gregorian month.
#[must_use]
pub const fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// An instant labelled on the UTC calendar.
///
/// `second` may reach 60 during a leap second. Whether it legitimately does is
/// decided by a leap-second table, not by this type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtcDateTime {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
}

impl UtcDateTime {
    /// Construct, validating against the calendar.
    ///
    /// # Errors
    ///
    /// [`CalendarError`] if any field is outside the range the calendar admits.
    pub fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
    ) -> Result<Self, CalendarError> {
        if !(1..=12).contains(&month) {
            return Err(CalendarError::Month(month));
        }
        if day < 1 || day > days_in_month(year, month) {
            return Err(CalendarError::Day { year, month, day });
        }
        if hour > 23 {
            return Err(CalendarError::Hour(hour));
        }
        if minute > 59 {
            return Err(CalendarError::Minute(minute));
        }
        if !(0.0..61.0).contains(&second) {
            return Err(CalendarError::Second(second));
        }
        Ok(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        })
    }

    /// Midnight on the given date.
    ///
    /// # Errors
    ///
    /// [`CalendarError`] if the date does not exist.
    pub fn from_date(year: i32, month: u8, day: u8) -> Result<Self, CalendarError> {
        Self::new(year, month, day, 0, 0, 0.0)
    }

    /// The calendar year.
    #[must_use]
    pub const fn year(self) -> i32 {
        self.year
    }
    /// The calendar month, `1..=12`.
    #[must_use]
    pub const fn month(self) -> u8 {
        self.month
    }
    /// The day of the month.
    #[must_use]
    pub const fn day(self) -> u8 {
        self.day
    }
    /// The hour, `0..=23`.
    #[must_use]
    pub const fn hour(self) -> u8 {
        self.hour
    }
    /// The minute, `0..=59`.
    #[must_use]
    pub const fn minute(self) -> u8 {
        self.minute
    }
    /// The second, `[0, 61)`. Reaches 60 only during a leap second.
    #[must_use]
    pub const fn second(self) -> f64 {
        self.second
    }

    /// Whether this label sits inside a leap second.
    ///
    /// Says nothing about whether such a leap second was actually inserted.
    #[must_use]
    pub fn is_leap_second_label(self) -> bool {
        self.second >= 60.0
    }

    /// Days from 1970-01-01 to this date.
    #[must_use]
    pub const fn days_from_civil(self) -> i64 {
        days_from_civil(self.year, self.month, self.day)
    }

    /// The Modified Julian Date of this UTC instant.
    ///
    /// `MJD = JD - 2400000.5`, so an MJD day begins at midnight rather than noon.
    /// Earth-orientation data is tabulated on this grid, at 0h UTC.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // the day number is far below 2^53
    pub fn modified_julian_date(self) -> f64 {
        const UNIX_DAYS_TO_MJD_EPOCH: i64 = 40_587;
        let day = self.days_from_civil() + UNIX_DAYS_TO_MJD_EPOCH;
        let fraction =
            (f64::from(self.hour) * 3600.0 + f64::from(self.minute) * 60.0 + self.second)
                / 86_400.0;
        day as f64 + fraction
    }

    /// Nanoseconds from the J2000 epoch instant, **counting every UTC day as
    /// exactly 86400 seconds**.
    ///
    /// This is not an instant on any physical time scale. It is the coordinate in
    /// which `TAI = naive + ΔAT` holds, and it is the only sane place to put a
    /// `23:59:60` label: that label yields exactly 86400 seconds into its day, so
    /// the leap second falls out of the arithmetic rather than being special-cased.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // second < 61, so the product is far below i128
    pub fn naive_ns_since_j2000(self) -> i128 {
        let days = i128::from(self.days_from_civil() - UNIX_DAYS_TO_J2000);
        let hms_ns = i128::from(self.hour) * 3_600 * 1_000_000_000
            + i128::from(self.minute) * 60 * 1_000_000_000
            + libm::round(self.second * 1e9) as i128;
        days * NS_PER_DAY + hms_ns - NS_MIDNIGHT_TO_NOON
    }

    /// Reconstruct a label from the naive coordinate of [`Self::naive_ns_since_j2000`].
    ///
    /// `extra_second` shifts a `23:59:59` label to `23:59:60`, which is how a leap
    /// second is rendered. The caller decides; this type cannot know.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // `rem` is non-negative by `rem_euclid`, and each field is bounded by its unit
    pub fn from_naive_ns_since_j2000(ns: i128, extra_second: bool) -> Self {
        let ns = ns + NS_MIDNIGHT_TO_NOON;
        let day = ns.div_euclid(NS_PER_DAY);
        let rem = ns.rem_euclid(NS_PER_DAY);

        let (year, month, dom) = civil_from_days(day as i64 + UNIX_DAYS_TO_J2000);
        let hour = (rem / (3_600 * 1_000_000_000)) as u8;
        let minute = (rem / (60 * 1_000_000_000) % 60) as u8;
        let second = (rem % (60 * 1_000_000_000)) as f64 / 1e9 + f64::from(u8::from(extra_second));

        Self {
            year,
            month,
            day: dom,
            hour,
            minute,
            second,
        }
    }
}

impl fmt::Display for UtcDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:09.6}Z",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // a panicking test is a failing test

    use super::*;

    #[test]
    fn epochs_are_where_we_think() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 1, 1), UNIX_DAYS_TO_J2000);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(UNIX_DAYS_TO_J2000), (2000, 1, 1));
    }

    #[test]
    fn civil_round_trips_over_four_centuries() {
        // One full Gregorian cycle either side of the Unix epoch: every leap rule
        // fires, including the 100- and 400-year exceptions.
        for z in -146_097_i64..=146_097 {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z, "round trip failed at day {z}");
        }
    }

    #[test]
    fn civil_round_trips_through_the_proleptic_era() {
        // The cycle above spans 1570..2370, so `y` and `z` are positive throughout and the
        // negative branch of each era division -- `y - 399`, `z - 146_096`, the correction that
        // makes truncating division floor -- never runs. These days are all before year 0, which
        // is what "proleptic, negative before" in the doc comment claims to handle.
        assert_eq!(civil_from_days(-719_468), (0, 3, 1));
        assert_eq!(days_from_civil(0, 3, 1), -719_468);
        assert_eq!(civil_from_days(-719_469), (0, 2, 29)); // year 0 is a leap year
        assert_eq!(days_from_civil(-1, 12, 31), -719_529);

        assert!(
            civil_from_days(-1_200_000).0 < -1_300,
            "the sweep must reach negative years"
        );
        for z in -1_200_000_i64..=-700_000 {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z, "round trip failed at day {z}");
        }
    }

    #[test]
    fn the_accessors_report_the_fields_they_were_built_from() {
        let t = UtcDateTime::new(2017, 7, 23, 4, 5, 6.5).unwrap();
        assert_eq!(t.year(), 2017);
        assert_eq!(t.month(), 7);
        assert_eq!(t.day(), 23);
        assert_eq!(t.hour(), 4);
        assert_eq!(t.minute(), 5);
        assert!((t.second() - 6.5).abs() < 1e-12);
    }

    #[test]
    fn modified_julian_date_carries_hours_minutes_and_seconds() {
        // MJD 51544.0 is 2000-01-01 00:00 UTC, the anchor every EOP table is indexed on.
        let midnight = UtcDateTime::new(2000, 1, 1, 0, 0, 0.0).unwrap();
        assert!((midnight.modified_julian_date() - 51_544.0).abs() < 1e-12);

        // A time whose hour, minute and second are all distinct and non-zero: at 00:00 or 12:00
        // the three terms of the day fraction collapse and the arithmetic joining them is
        // untested. 03:25:45 is 12345 s, so the fraction is exactly 12345/86400.
        let odd = UtcDateTime::new(2000, 1, 1, 3, 25, 45.0).unwrap();
        let expected = 51_544.0 + 12_345.0 / 86_400.0;
        assert!(
            (odd.modified_julian_date() - expected).abs() < 1e-12,
            "{} != {expected}",
            odd.modified_julian_date()
        );
    }

    #[test]
    fn the_leap_second_label_adds_its_second() {
        // A positive leap second is labelled 23:59:60, one second past the last ordinary second
        // of the day. The flag is the only thing that distinguishes them, so reconstructing the
        // same instant with and without it must differ by exactly one second in the label.
        let t = UtcDateTime::new(2016, 12, 31, 23, 59, 59.0).unwrap();
        let ns = t.naive_ns_since_j2000();
        let ordinary = UtcDateTime::from_naive_ns_since_j2000(ns, false);
        let leap = UtcDateTime::from_naive_ns_since_j2000(ns, true);
        assert_eq!((leap.hour(), leap.minute()), (23, 59));
        assert!((leap.second() - ordinary.second() - 1.0).abs() < 1e-9);
        assert!((leap.second() - 60.0).abs() < 1e-9, "{}", leap.second());
    }

    #[test]
    fn j2000_noon_is_the_origin() {
        let noon = UtcDateTime::new(2000, 1, 1, 12, 0, 0.0).unwrap();
        assert_eq!(noon.naive_ns_since_j2000(), 0);
    }

    #[test]
    fn naive_coordinate_round_trips() {
        let t = UtcDateTime::new(2016, 12, 31, 23, 59, 59.25).unwrap();
        let back = UtcDateTime::from_naive_ns_since_j2000(t.naive_ns_since_j2000(), false);
        assert_eq!(back, t);
    }

    #[test]
    fn a_leap_second_label_is_exactly_one_day_in() {
        // 23:59:60 sits 86400 s into its day. That is the whole trick: the leap
        // second needs no special case in the arithmetic, only in the label.
        let leap = UtcDateTime::new(2016, 12, 31, 23, 59, 60.0).unwrap();
        let next = UtcDateTime::from_date(2017, 1, 1).unwrap();
        assert_eq!(leap.naive_ns_since_j2000(), next.naive_ns_since_j2000());
    }

    #[test]
    fn the_calendar_rejects_what_it_does_not_have() {
        assert!(UtcDateTime::from_date(2023, 2, 29).is_err());
        assert!(UtcDateTime::from_date(2024, 2, 29).is_ok());
        assert!(UtcDateTime::from_date(1900, 2, 29).is_err());
        assert!(UtcDateTime::from_date(2000, 2, 29).is_ok());
        assert!(UtcDateTime::new(2024, 1, 1, 24, 0, 0.0).is_err());
        assert!(UtcDateTime::new(2024, 1, 1, 0, 0, 61.0).is_err());
    }
}
