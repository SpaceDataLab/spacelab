//! The day-boundary rounding class (plan 18, R9 P7; asserted under Ф20.Q): no epoch near midnight
//! changes its day on the way through this crate.
//!
//! The class is a calendar field computed from a floating day count, where a label 1 ns before
//! midnight rounds into the next day. An epoch passes through a float twice here: the TLE writer
//! prints a fractional day of year, and `Satellite::from_elements` hands the second of its label to
//! `UtcDateTime` as an `f64`. The check is exhaustive over the first and last 1 000 ns of every day
//! on a grid chosen where a carry has somewhere to go. Across both ends, the satellite's epoch is
//! the label's own integer nanosecond, and the printed TLE names the label's own day and parses back
//! to it. The last label of a day prints as the last instant eight digits can name, `.99999999`,
//! because the format has no way to write day zero of the next day.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use sgp4::chrono::{Datelike, NaiveDate};
use spacelab_core::UtcDateTime;
use spacelab_sgp4::{Convention, Elements, Satellite, TleBuilder};

/// ISS, epoch 2008-264: the `sgp4` crate's documented element set. Only its epoch is replaced.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// Year ends of 365 and 366 days, the leap day and both its neighbours, a day that ended in a leap
/// second, an ordinary day, and the first and last years the two-digit TLE year can name.
const DAYS: [(i32, u8, u8); 10] = [
    (1957, 1, 1),
    (1999, 12, 31),
    (2000, 1, 1),
    (2000, 2, 28),
    (2000, 2, 29),
    (2000, 12, 31),
    (2016, 12, 31),
    (2025, 12, 31),
    (2026, 8, 14),
    (2056, 12, 31),
];

const NS_PER_DAY: i128 = 86_400_000_000_000;

/// Nanoseconds tested at each end of a day.
const EDGE_NS: u32 = 1_000;

#[test]
fn no_epoch_within_a_microsecond_of_midnight_changes_its_day() {
    let template = Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).unwrap();
    for (year, month, day) in DAYS {
        let date = NaiveDate::from_ymd_opt(year, u32::from(month), u32::from(day)).unwrap();
        let midnight = UtcDateTime::from_date(year, month, day)
            .unwrap()
            .naive_ns_since_j2000();
        let day_field = format!("{:02}{:03}", year % 100, date.ordinal());
        for k in 0..EDGE_NS {
            let first = date.and_hms_nano_opt(0, 0, 0, k).unwrap();
            let last = date
                .and_hms_nano_opt(23, 59, 59, 1_000_000_000 - EDGE_NS + k)
                .unwrap();
            for (label, ns_of_day, fraction) in [
                (first, i128::from(k), "00000000"),
                (
                    last,
                    NS_PER_DAY - i128::from(EDGE_NS) + i128::from(k),
                    "99999999",
                ),
            ] {
                let elements = Elements {
                    datetime: label,
                    ..template.clone()
                };

                let epoch = Satellite::from_elements(&elements, Convention::Reference)
                    .unwrap()
                    .epoch();
                assert_eq!(
                    epoch.naive_ns_since_j2000(),
                    midnight + ns_of_day,
                    "{label}"
                );
                assert_eq!(
                    (epoch.year(), epoch.month(), epoch.day()),
                    (year, month, day),
                    "{label}"
                );

                let tle = TleBuilder::from_elements(&elements).build().unwrap();
                assert_eq!(
                    &tle.line1()[18..32],
                    format!("{day_field}.{fraction}"),
                    "{label}"
                );
                let reread =
                    Elements::from_tle(None, tle.line1().as_bytes(), tle.line2().as_bytes())
                        .unwrap();
                assert_eq!(reread.datetime.date(), date, "{label}");
            }
        }
    }
}
