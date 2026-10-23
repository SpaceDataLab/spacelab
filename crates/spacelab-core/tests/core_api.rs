//! The value types, the calendar, and the small vector/time helpers.
//!
//! `spacelab-core` is exercised heavily by every dependent crate but had almost no tests of its own,
//! so its public surface — the `Angle`/`Length`/`NoradId` unit wrappers, the UTC calendar (validation,
//! leap years, `days_from_civil`, Modified Julian Date), the `Vec3` operators and
//! `Epoch::from_days_since_j2000` — went unverified in isolation. Pin them to known values.

#![allow(clippy::float_cmp, clippy::unwrap_used)] // exact unit conversions and calendar constants

use core::f64::consts::{PI, TAU};

use spacelab_core::calendar::{days_in_month, is_leap_year};
use spacelab_core::{Angle, CalendarError, Epoch, Length, NoradId, Tt, UtcDateTime, Vec3};

#[test]
fn angle_and_length_conversions_round_trip() {
    assert!((Angle::from_degrees(180.0).radians() - PI).abs() < 1e-15);
    assert!((Angle::from_radians(PI).degrees() - 180.0).abs() < 1e-13);
    // 3600 arcseconds is one degree.
    assert!(
        (Angle::from_arcseconds(3600.0).radians() - Angle::from_degrees(1.0).radians()).abs()
            < 1e-18
    );
    assert!((Angle::from_radians(0.37).arcseconds() - 0.37 * (180.0 * 3600.0) / PI).abs() < 1e-9);
    assert_eq!(Angle::TAU, TAU);

    assert_eq!(Length::from_kilometres(1.0).metres(), 1000.0);
    assert_eq!(Length::from_metres(2500.0).kilometres(), 2.5);
}

#[test]
fn norad_id_wraps_and_orders_the_catalog_number() {
    assert_eq!(NoradId::new(25544).get(), 25544);
    assert!(NoradId::new(5) < NoradId::new(25544)); // derived Ord
}

#[test]
fn leap_years_and_month_lengths_follow_the_gregorian_rules() {
    assert!(is_leap_year(2000) && is_leap_year(2004));
    assert!(!is_leap_year(1900) && !is_leap_year(2001));
    assert_eq!(days_in_month(2000, 2), 29); // divisible by 400
    assert_eq!(days_in_month(1900, 2), 28); // divisible by 100, not 400
    assert_eq!(days_in_month(2001, 2), 28);
    assert_eq!(days_in_month(2000, 1), 31);
    assert_eq!(days_in_month(2000, 4), 30);
    assert_eq!(days_in_month(2000, 13), 0); // no such month
}

#[test]
fn utc_datetime_validates_its_fields_and_reports_them_back() {
    let t = UtcDateTime::new(2000, 1, 1, 12, 30, 15.5).unwrap();
    assert_eq!(t.year(), 2000);
    assert_eq!(t.month(), 1);
    assert_eq!(t.day(), 1);
    assert_eq!(t.hour(), 12);
    assert_eq!(t.minute(), 30);
    assert_eq!(t.second(), 15.5);
    assert!(!t.is_leap_second_label());

    // A leap-second label (23:59:60) is admitted and recognised.
    let leap = UtcDateTime::new(2016, 12, 31, 23, 59, 60.0).unwrap();
    assert!(leap.is_leap_second_label());

    // from_date is midnight.
    let midnight = UtcDateTime::from_date(2020, 2, 29).unwrap();
    assert_eq!(
        (midnight.hour(), midnight.minute(), midnight.second()),
        (0, 0, 0.0)
    );

    // Each out-of-range field is refused with its own error.
    assert!(matches!(
        UtcDateTime::new(2000, 13, 1, 0, 0, 0.0),
        Err(CalendarError::Month(13))
    ));
    assert!(matches!(
        UtcDateTime::new(2001, 2, 29, 0, 0, 0.0), // 2001 is not a leap year
        Err(CalendarError::Day { .. })
    ));
    assert!(matches!(
        UtcDateTime::new(2000, 1, 1, 24, 0, 0.0),
        Err(CalendarError::Hour(24))
    ));
    assert!(matches!(
        UtcDateTime::new(2000, 1, 1, 0, 60, 0.0),
        Err(CalendarError::Minute(60))
    ));
    assert!(matches!(
        UtcDateTime::new(2000, 1, 1, 0, 0, 61.0),
        Err(CalendarError::Second(_))
    ));
}

#[test]
fn days_from_civil_and_mjd_match_known_epochs() {
    // The civil-days count is anchored at the Unix epoch.
    assert_eq!(
        UtcDateTime::from_date(1970, 1, 1)
            .unwrap()
            .days_from_civil(),
        0
    );
    assert_eq!(
        UtcDateTime::from_date(2000, 1, 1)
            .unwrap()
            .days_from_civil(),
        10957
    );

    // 2000-01-01 00:00 UTC is MJD 51544.0; noon is 51544.5 (the J2000 epoch on the MJD grid).
    assert_eq!(
        UtcDateTime::new(2000, 1, 1, 0, 0, 0.0)
            .unwrap()
            .modified_julian_date(),
        51544.0
    );
    assert_eq!(
        UtcDateTime::new(2000, 1, 1, 12, 0, 0.0)
            .unwrap()
            .modified_julian_date(),
        51544.5
    );
}

#[test]
fn errors_and_datetimes_print_a_non_empty_message() {
    let errors = [
        CalendarError::Month(0),
        CalendarError::Day {
            year: 2001,
            month: 2,
            day: 29,
        },
        CalendarError::Hour(24),
        CalendarError::Minute(60),
        CalendarError::Second(61.0),
    ];
    for e in errors {
        assert_ne!(e.to_string(), "");
    }
    assert_ne!(
        UtcDateTime::new(2000, 1, 1, 12, 0, 0.0)
            .unwrap()
            .to_string(),
        ""
    );
}

#[test]
fn vec3_operators_are_componentwise() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, -1.0, 0.5);
    let scaled = a.scale(2.0);
    assert_eq!((scaled.x, scaled.y, scaled.z), (2.0, 4.0, 6.0));
    let sum = a + b;
    assert_eq!((sum.x, sum.y, sum.z), (5.0, 1.0, 3.5));
    let diff = a - b;
    assert_eq!((diff.x, diff.y, diff.z), (-3.0, 3.0, 2.5));
    let neg = -a;
    assert_eq!((neg.x, neg.y, neg.z), (-1.0, -2.0, -3.0));
}

#[test]
fn epoch_from_days_since_j2000_splits_before_scaling() {
    // 1.5 days is exactly 129 600 seconds = 1.296e14 ns from J2000.
    let e = Epoch::<Tt>::from_days_since_j2000(1.5);
    assert_eq!(e.ns_since_j2000(), 129_600_000_000_000);
}
