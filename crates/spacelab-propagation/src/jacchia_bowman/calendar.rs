//! Day of year from a modified Julian date, Gregorian calendar, anchored on the printed epoch
//! MJD 36204 = 1958 January 1 (SR332 p. 44).
//!
//! Gate: the calendar `(G2)` tests in `tests/jacchia_bowman_g2_model.rs`.

use super::constants::MJD_1958_JAN_1;

const ANCHOR_YEAR: i64 = 1958;
/// Days in a 400-year Gregorian cycle: 400 * 365 + 97 leap days (calendar definition).
const DAYS_PER_400_YEARS: i64 = 146_097;

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_year(year: i64) -> i64 {
    if is_leap(year) { 366 } else { 365 }
}

/// Calendar year and day of year t (1.0 at 0h UT on January 1, fractional) for an MJD (UT).
pub fn year_and_day_of_year(mjd: f64) -> (i64, f64) {
    let whole = mjd.floor();
    let frac = mjd - whole;
    let mut days = whole as i64 - MJD_1958_JAN_1 as i64;
    let mut year = ANCHOR_YEAR;
    // The Gregorian calendar repeats every 400 years.
    year += 400 * days.div_euclid(DAYS_PER_400_YEARS);
    days = days.rem_euclid(DAYS_PER_400_YEARS);
    while days >= days_in_year(year) {
        days -= days_in_year(year);
        year += 1;
    }
    (year, (days + 1) as f64 + frac)
}
