//! Load measured thermospheric **crosswinds** into fit samples for [`crate::wind_model`].
//!
//! Reads the TU Delft thermosphere crosswind ASCII product (CHAMP / GRACE / GOCE / GRACE-FO; Siemes
//! et al., *J. Space Weather Space Clim.* 2023; <http://thermosphere.tudelft.nl>, CC BY 4.0). It is
//! the same file family [`crate::density_data`] reads — same header, same first eight columns — with
//! the density replaced by a wind speed and a north/east/down unit vector:
//! `date time time_system altitude(m) longitude(deg) latitude(deg) local_solar_time(h)
//! argument_of_latitude(deg) speed(m/s) dir_north dir_east dir_down flag`. Comment lines start
//! with `#`. The bulk data are an external download (`tools/fetch-wind-data.sh`, open and keyless)
//! and are not vendored; a one-day GRACE-FO excerpt is kept as a parser fixture.
//!
//! **What the product measures is a projection, not a wind vector.** An accelerometer sees the
//! component of the neutral wind across the satellite's own velocity; the along-track component is
//! inseparable from the density. So each row is one scalar equation
//! `w · d̂ = speed`, with `d̂` the reported unit vector and `speed ≥ 0` its magnitude — never a
//! `(u, v)` pair. [`CrosswindObservation::projection`] is that equation and is what
//! [`crate::wind_model`] fits; [`CrosswindObservation::velocity_ned`] is the measured vector
//! `speed · d̂`, which is the *projection of* the wind and not the wind.
//!
//! Every mission in the product flies a near-polar orbit — 87.3° (CHAMP), 89.0° (GRACE, GRACE-FO),
//! 96.7° (GOCE) — so the velocity is near-meridional and `d̂` is mostly zonal. Mostly, not only, and
//! the difference is the reason to load more than one mission. Measured over the arcs
//! `tools/fetch-wind-data.sh` downloads: `|d̂ · ê| ≥ 0.99` in 79–93 % of CHAMP / GRACE / GRACE-FO
//! rows, yet the polar passes swing the cross-track direction through north-south, and GOCE's
//! sun-synchronous inclination gives `|d̂ · n̂| > 0.1` in *every* one of its rows and `> 0.3` in 27 %
//! of them. The zonal wind is what this data pins down; the meridional wind is weakly but genuinely
//! observed, and mostly by GOCE. [`zonal_fraction`] measures that split on whatever was actually
//! loaded rather than assuming it.

use alloc::vec::Vec;

use crate::density_data::TimeScale;

/// One crosswind observation from the TU Delft product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrosswindObservation {
    /// Calendar year.
    pub year: i32,
    /// Calendar month, `1..=12`.
    pub month: u32,
    /// Day of month, `1..=31`.
    pub day: u32,
    /// Seconds elapsed since midnight, in [`Self::time_scale`].
    pub seconds_of_day: f64,
    /// The scale [`Self::seconds_of_day`] is measured in.
    pub time_scale: TimeScale,
    /// Geodetic altitude, metres.
    pub altitude: f64,
    /// Geodetic longitude, radians.
    pub longitude: f64,
    /// Geodetic latitude, radians.
    pub latitude: f64,
    /// Local solar time, hours in `[0, 24)`.
    pub local_solar_time: f64,
    /// Argument of latitude, radians — the along-orbit angle from the ascending node, which is what
    /// separates the ascending and descending passes that sample opposite local solar times.
    pub argument_of_latitude: f64,
    /// Magnitude of the measured crosswind, m/s. Non-negative: the direction carries the sign.
    pub speed: f64,
    /// Crosswind direction as a unit vector in north / east / down components.
    pub direction: [f64; 3],
    /// `true` when the data flag is 0 (nominal).
    pub nominal: bool,
}

impl CrosswindObservation {
    /// The measured crosswind vector `speed · d̂` in north / east / down components, m/s.
    ///
    /// This is the *projection of* the neutral wind onto the cross-track direction, not the wind:
    /// the component along the satellite's velocity is not observed at all.
    #[must_use]
    pub fn velocity_ned(&self) -> [f64; 3] {
        [
            self.speed * self.direction[0],
            self.speed * self.direction[1],
            self.speed * self.direction[2],
        ]
    }

    /// The scalar equation this row imposes on a wind field: the direction `d̂` and the projection
    /// `w · d̂ = speed` along it.
    #[must_use]
    pub const fn projection(&self) -> ([f64; 3], f64) {
        (self.direction, self.speed)
    }

    /// Day of year, `1..=366`.
    #[must_use]
    pub const fn day_of_year(&self) -> u32 {
        let leap = self.year % 4 == 0 && (self.year % 100 != 0 || self.year % 400 == 0);
        let before = [0u32, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let m = if self.month >= 1 && self.month <= 12 {
            self.month
        } else {
            1
        };
        before[m as usize - 1] + self.day + if leap && m > 2 { 1 } else { 0 }
    }
}

/// Parse the TU Delft thermosphere crosswind ASCII text. Comment lines (`#`) and malformed rows are
/// skipped; degrees are converted to radians.
///
/// A row whose direction is not a unit vector to `1e-6` is dropped: the fit reads `d̂` as a direction
/// cosine, and a row where it is not one is a row where the product's own geometry is unusable.
#[must_use]
pub fn parse_crosswind(text: &str) -> Vec<CrosswindObservation> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 12 {
            continue;
        }
        let d: Vec<&str> = t[0].split('-').collect();
        if d.len() != 3 {
            continue;
        }
        let (Ok(year), Ok(month), Ok(day)) = (d[0].parse(), d[1].parse(), d[2].parse()) else {
            continue;
        };
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            continue;
        }
        let num = |s: &str| s.parse::<f64>().ok();
        let hms: Vec<&str> = t[1].split(':').collect();
        if hms.len() != 3 {
            continue;
        }
        let (Some(hh), Some(mm), Some(ss)) = (num(hms[0]), num(hms[1]), num(hms[2])) else {
            continue;
        };
        let time_scale = match t[2] {
            "UTC" => TimeScale::Utc,
            "GPS" => TimeScale::Gps,
            _ => TimeScale::Other,
        };
        let (Some(alt), Some(lon), Some(lat), Some(lst), Some(arglat), Some(speed)) = (
            num(t[3]),
            num(t[4]),
            num(t[5]),
            num(t[6]),
            num(t[7]),
            num(t[8]),
        ) else {
            continue;
        };
        let (Some(dn), Some(de), Some(dd)) = (num(t[9]), num(t[10]), num(t[11])) else {
            continue;
        };
        if libm::fabs(libm::sqrt(dn * dn + de * de + dd * dd) - 1.0) > 1e-6 {
            continue;
        }
        let nominal = t.get(12).and_then(|s| num(s)).is_none_or(|f| f == 0.0);
        out.push(CrosswindObservation {
            year,
            month,
            day,
            seconds_of_day: 3600.0 * hh + 60.0 * mm + ss,
            time_scale,
            altitude: alt,
            longitude: lon.to_radians(),
            latitude: lat.to_radians(),
            local_solar_time: lst,
            argument_of_latitude: arglat.to_radians(),
            speed,
            direction: [dn, de, dd],
            nominal,
        });
    }
    out
}

/// The fraction of `observations` whose crosswind direction is within `tolerance` of purely zonal,
/// i.e. `|d̂ · ê| ≥ 1 − tolerance`.
///
/// This is the observability of the two horizontal components, measured rather than assumed: a value
/// near 1 says the loaded arcs pin the zonal wind and leave the meridional one to the regularizer.
/// Returns `0.0` for an empty slice.
#[must_use]
pub fn zonal_fraction(observations: &[CrosswindObservation], tolerance: f64) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let n = observations
        .iter()
        .filter(|o| libm::fabs(o.direction[1]) >= 1.0 - tolerance)
        .count();
    let total = observations.len();
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
        / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp, clippy::unreadable_literal)]

    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/gracefo_crosswind_20240501.txt");

    #[test]
    fn parses_the_real_grace_fo_fixture() {
        let obs = parse_crosswind(FIXTURE);
        assert_eq!(obs.len(), 30);
        // First data row: 2024-05-01 00:00:00.000 GPS, alt 499936.602 m, lon 83.693°, lat −68.358°,
        // lst 5.623 h, arglat 291.665°, speed 326.60 m/s, d̂ = (0.0155443312, −0.9997235722,
        // −0.017639530338).
        let o = obs[0];
        assert_eq!((o.year, o.month, o.day), (2024, 5, 1));
        assert!((o.seconds_of_day - 0.0).abs() < 1e-12);
        assert_eq!(o.time_scale, TimeScale::Gps);
        assert_eq!(o.day_of_year(), 122);
        assert!((o.altitude - 499_936.602).abs() < 1e-3);
        assert!((o.longitude - 83.693_f64.to_radians()).abs() < 1e-12);
        assert!((o.latitude + 68.358_f64.to_radians()).abs() < 1e-12);
        assert!((o.local_solar_time - 5.623).abs() < 1e-12);
        assert!((o.argument_of_latitude - 291.665_f64.to_radians()).abs() < 1e-12);
        assert!((o.speed - 326.60).abs() < 1e-12);
        assert!((o.direction[0] - 0.0155443312).abs() < 1e-12);
        assert!((o.direction[1] + 0.9997235722).abs() < 1e-12);
        assert!((o.direction[2] + 0.017639530338).abs() < 1e-12);
        assert!(o.nominal);
    }

    #[test]
    fn the_reported_direction_is_a_unit_vector_and_the_speed_is_its_magnitude() {
        // Both are load-bearing: the fit reads `d̂` as direction cosines and `speed` as the projection
        // along it, so a product that packed a signed component into the speed, or a direction that
        // was not normalized, would be silently rescaling every residual.
        for o in parse_crosswind(FIXTURE) {
            let d = o.direction;
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (n - 1.0).abs() < 1e-9,
                "direction is not a unit vector: {n}"
            );
            assert!(o.speed >= 0.0, "speed carries a sign: {}", o.speed);
            let v = o.velocity_ned();
            let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((m - o.speed).abs() < 1e-9 * o.speed.max(1.0));
        }
    }

    #[test]
    fn a_near_polar_orbit_measures_mostly_but_not_only_the_zonal_wind() {
        // GRACE-FO's inclination is 89°, so 27 of these 30 rows sit within 8° of east-west. The
        // exception is the point of the test: over the pole the cross-track direction swings right
        // through north-south, and one row of the thirty is 83° from zonal. That row is why
        // `wind_model` fits a coupled vector field instead of a scalar zonal one — a model that
        // could only produce an eastward wind would have nothing to say about it.
        let obs = parse_crosswind(FIXTURE);
        let f = zonal_fraction(&obs, 0.01);
        assert!((f - 27.0 / 30.0).abs() < 1e-12, "zonal fraction is {f:.3}");
        let meridional = obs
            .iter()
            .filter(|o| libm::fabs(o.direction[0]) > 0.5)
            .count();
        assert_eq!(meridional, 1, "the polar pass should contribute one row");
    }

    /// A row of the product's own layout, with the pieces a test wants to vary spelled out.
    fn row(time: &str, scale: &str, tail: &str) -> alloc::string::String {
        alloc::format!(
            "2024-05-01 {time} {scale} 499936.602 83.693 -68.358 5.623 291.665 326.60 {tail}"
        )
    }

    fn dated(year: i32, month: u32, day: u32) -> CrosswindObservation {
        CrosswindObservation {
            year,
            month,
            day,
            seconds_of_day: 0.0,
            time_scale: TimeScale::Utc,
            altitude: 400_000.0,
            longitude: 0.0,
            latitude: 0.0,
            local_solar_time: 12.0,
            argument_of_latitude: 0.0,
            speed: 0.0,
            direction: [0.0, 1.0, 0.0],
            nominal: true,
        }
    }

    #[test]
    fn the_projection_is_the_direction_and_the_speed_along_it() {
        // `wind_model` fits `w · d̂ = speed` and reads both sides from here, so this pins the pair
        // rather than trusting that the accessor returns the fields it names.
        for o in parse_crosswind(FIXTURE) {
            let (d, s) = o.projection();
            assert_eq!(d, o.direction);
            assert_eq!(s, o.speed);
            let v = o.velocity_ned();
            let along = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
            assert!((along - s).abs() < 1e-9 * s.max(1.0), "{along} vs {s}");
        }
    }

    #[test]
    fn day_of_year_counts_the_leap_day_only_in_leap_years() {
        // The Gregorian rule in full: divisible by four is a leap year unless it is a century that is
        // not divisible by four hundred. 1900 and 2000 are the two cases that separate the rule from
        // its four-year approximation, and March is the first month that can tell them apart.
        for (year, month, day, expected) in [
            (2024, 1, 1, 1),
            (2024, 2, 15, 46),
            (2024, 2, 29, 60),
            (2024, 3, 1, 61),
            (2024, 5, 1, 122),
            (2024, 12, 31, 366),
            (2023, 3, 1, 60),
            (2023, 12, 31, 365),
            (2000, 3, 1, 61),
            (1900, 3, 1, 60),
        ] {
            let got = dated(year, month, day).day_of_year();
            assert_eq!(got, expected, "{year}-{month:02}-{day:02}");
        }
    }

    #[test]
    fn day_of_year_reads_a_month_off_the_calendar_as_january() {
        // The parser cannot produce one, but the fields are public and this is what keeps the
        // accessor total: it falls back to January instead of indexing past the month table.
        assert_eq!(dated(2024, 13, 7).day_of_year(), 7);
        assert_eq!(dated(2024, 0, 7).day_of_year(), 7);
    }

    #[test]
    fn the_clock_and_the_time_scale_columns_are_read() {
        // Every row of the fixture is at 00:00-something, so the hour and minute terms are multiplied
        // by zero there and any arithmetic at all reproduces the seconds.
        let obs = parse_crosswind(&row("12:34:56.500", "UTC", "0.0 1.0 0.0 0.0"));
        assert_eq!(obs.len(), 1);
        assert!((obs[0].seconds_of_day - 45_296.5).abs() < 1e-9);
        assert_eq!(obs[0].time_scale, TimeScale::Utc);
        assert_eq!(
            parse_crosswind(&row("00:00:00.000", "GPS", "0.0 1.0 0.0 0.0"))[0].time_scale,
            TimeScale::Gps
        );
        assert_eq!(
            parse_crosswind(&row("00:00:00.000", "TAI", "0.0 1.0 0.0 0.0"))[0].time_scale,
            TimeScale::Other
        );
    }

    #[test]
    fn the_flag_column_is_optional_and_a_row_short_of_the_geometry_is_dropped() {
        // The product writes the flag; the twelve-column form without it is still a complete
        // observation and reads as nominal. A row missing part of the direction vector is not, and
        // must be dropped rather than read past its end.
        let twelve = parse_crosswind(&row("00:00:00.000", "GPS", "0.0 1.0 0.0"));
        assert_eq!(twelve.len(), 1);
        assert!(twelve[0].nominal);
        assert_eq!(
            parse_crosswind(&row("00:00:00.000", "GPS", "0.0 1.0")).len(),
            0
        );
        let flagged = parse_crosswind(&row("00:00:00.000", "GPS", "0.0 1.0 0.0 1.0"));
        assert!(!flagged[0].nominal);
    }

    #[test]
    fn a_row_whose_date_is_off_the_calendar_is_dropped() {
        // Both halves of the range check, separately: a month of 13 with a good day, and a day of 32
        // with a good month. Either one alone makes the row unusable.
        let tail = "0.0 1.0 0.0 0.0";
        for date in ["2024-13-01", "2024-05-32", "2024-00-01", "2024-05-00"] {
            let line = alloc::format!(
                "{date} 00:00:00.000 GPS 499936.602 83.693 -68.358 5.623 291.665 326.60 {tail}"
            );
            assert_eq!(parse_crosswind(&line).len(), 0, "{date} was accepted");
        }
    }

    #[test]
    fn a_row_whose_direction_is_not_a_unit_vector_is_dropped() {
        let good = "2024-05-01 00:00:00.000 GPS 499936.602 83.693 -68.358 5.623 291.665 326.60 \
                    0.0 1.0 0.0 0.0";
        let bad = "2024-05-01 00:00:10.000 GPS 499824.727 83.732 -67.726 5.628 292.301 327.41 \
                   0.0 0.5 0.0 0.0";
        assert_eq!(parse_crosswind(good).len(), 1);
        assert_eq!(parse_crosswind(bad).len(), 0);
        assert_eq!(parse_crosswind(&alloc::format!("{good}\n{bad}\n")).len(), 1);
    }
}
