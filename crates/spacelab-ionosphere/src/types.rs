//! Small value types shared across the model.

/// A point on or above the Earth, expressed in geodetic coordinates.
///
/// The NeQuick-G STEC path only ever consumes the raw latitude, longitude and
/// altitude components (the ray geometry uses a spherical mean-Earth radius, not
/// the reference ellipsoid), so no ellipsoid is carried here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodeticPoint {
    /// Geodetic latitude, radians.
    pub latitude: f64,
    /// Geodetic longitude, radians.
    pub longitude: f64,
    /// Altitude above the reference surface, metres.
    pub altitude: f64,
}

impl GeodeticPoint {
    /// Build a geodetic point from latitude/longitude (radians) and altitude (metres).
    #[must_use]
    pub const fn new(latitude: f64, longitude: f64, altitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude,
        }
    }
}

/// The subset of a UTC epoch that the NeQuick-G algorithm actually depends on.
///
/// The model reads the date only through the calendar month (selecting the monthly
/// CCIR coefficient file and the seasonal terms) and the universal time of day
/// (the Fourier-in-UT term and the solar declination). The day-of-month never
/// enters: the solar declination uses the mid-month approximation
/// `dy = 30.5 * month - 15`. Both reference test epochs therefore reduce to a
/// `(month, seconds_in_utc_day)` pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DateTimeComponents {
    /// Calendar month, 1 (January) through 12 (December).
    month: u32,
    /// Seconds elapsed since 00:00:00 UTC on the current day.
    seconds_in_utc_day: f64,
}

impl DateTimeComponents {
    /// Build the epoch from a calendar month and the number of seconds since UTC midnight.
    ///
    /// # Panics
    ///
    /// Panics if `month` is not in `1..=12`.
    #[must_use]
    pub fn new(month: u32, seconds_in_utc_day: f64) -> Self {
        assert!(
            (1..=12).contains(&month),
            "month must be in 1..=12, got {month}"
        );
        Self {
            month,
            seconds_in_utc_day,
        }
    }

    /// Calendar month, 1 through 12.
    pub(crate) const fn month(self) -> u32 {
        self.month
    }

    /// Universal time of day, in hours.
    pub(crate) fn hours(self) -> f64 {
        self.seconds_in_utc_day / 3600.0
    }
}

/// Sine and cosine of an angle, mirroring Hipparchus' `SinCos`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SinCos {
    /// Sine of the angle.
    pub s: f64,
    /// Cosine of the angle.
    pub c: f64,
}

impl SinCos {
    /// Evaluate sine and cosine of `angle` (radians).
    pub(crate) fn of(angle: f64) -> Self {
        let (s, c) = libm::sincos(angle);
        Self { s, c }
    }
}
