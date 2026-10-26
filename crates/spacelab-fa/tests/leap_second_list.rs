//! The leap-second table against the IERS's own list, `fixtures/leap-seconds.list` (G5).
//!
//! The list is the IERS Earth Orientation Centre's, in the public domain, updated through
//! Bulletin C 72 and valid until 2027-06-28. The fixture is the copy the IANA time-zone database
//! redistributes, byte for byte; `docs/provenance.md` (`iers-leap-seconds`) records its sha256 and
//! that its own SHA-1 line matches its data.

#![allow(clippy::expect_used)]

use spacelab_core::{
    UtcDateTime,
    calendar::{civil_from_days, days_from_civil},
};
use spacelab_fa::delta_at;

const LIST: &str = include_str!("fixtures/leap-seconds.list");

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// The list's day of an NTP timestamp, in days from 1970-01-01. The list defines
/// `MJD = X/86400 + 15020`, and MJD 0 is 1858-11-17.
fn ntp_day(x: i64) -> i64 {
    x.div_euclid(86_400) + 15_020 + days_from_civil(1858, 11, 17)
}

struct List {
    /// `(day, TAI − UTC)` of each row, the day in days from 1970-01-01.
    rows: Vec<(i64, i32)>,
    updated: i64,
    expires: i64,
}

/// Each row's timestamp is a midnight, and the date the row writes out in clear is that
/// midnight's date.
fn parse() -> List {
    let (mut rows, mut updated, mut expires) = (Vec::new(), None, None);
    for line in LIST.lines() {
        let stamp = |s: &str| ntp_day(s.trim().parse().expect("an NTP timestamp"));
        if let Some(s) = line.strip_prefix("#$") {
            updated = Some(stamp(s));
        } else if let Some(s) = line.strip_prefix("#@") {
            expires = Some(stamp(s));
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            let (fields, clear) = line
                .split_once('#')
                .expect("a row writes its date in clear");
            let [x, dtai] = fields.split_whitespace().collect::<Vec<_>>()[..] else {
                panic!("{line}: expected a timestamp and TAI − UTC");
            };
            let x: i64 = x.parse().expect("a timestamp");
            assert_eq!(x % 86_400, 0, "{line}: not a midnight");
            let [d, m, y] = clear.split_whitespace().collect::<Vec<_>>()[..] else {
                panic!("{line}: expected `day month year` in clear");
            };
            let month = MONTHS.iter().position(|&n| n == m).expect("a month name") + 1;
            let date = (
                y.parse().expect("year"),
                u8::try_from(month).expect("month"),
                d.parse().expect("day"),
            );
            assert_eq!(civil_from_days(ntp_day(x)), date, "{line}");
            rows.push((ntp_day(x), dtai.parse().expect("TAI − UTC")));
        }
    }
    List {
        rows,
        updated: updated.expect("the list states its last update"),
        expires: expires.expect("the list states its expiry"),
    }
}

fn delta_at_on(day: i64) -> i32 {
    let (y, m, d) = civil_from_days(day);
    delta_at(UtcDateTime::from_date(y, m, d).expect("a date")).expect("inside the table")
}

/// The list is the table: every day from 1972-01-01 to the list's expiry, `TAI − UTC` changes on
/// exactly the days the list has a row for, to exactly the row's value. (G5)
#[test]
fn the_table_is_the_iers_list_until_it_expires() {
    let list = parse();
    let first = days_from_civil(1972, 1, 1);
    let mut steps = vec![(first, delta_at_on(first))];
    for day in first + 1..list.expires {
        let value = delta_at_on(day);
        if value != steps.last().expect("a first row").1 {
            steps.push((day, value));
        }
    }
    assert_eq!(steps, list.rows);
}

/// The list was updated through Bulletin C 72 and is valid past 2027-05-20, when CGPM 2026
/// Resolution 5 makes UTC continuous. No date before continuity is left unannounced, so its last
/// row holds for every later date.
#[test]
fn the_list_is_valid_past_the_start_of_continuous_utc() {
    let list = parse();
    assert_eq!(civil_from_days(list.updated), (2026, 7, 6));
    assert_eq!(civil_from_days(list.expires), (2027, 6, 28));
    assert!(LIST.contains("File expires on 28 June 2027"));
    assert!(list.expires > days_from_civil(2027, 5, 20));
}
