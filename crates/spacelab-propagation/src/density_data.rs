//! Load real accelerometer-derived thermospheric densities into calibration samples for
//! [`crate::density_calibration`] / [`crate::density_rom`].
//!
//! Reads the TU Delft thermosphere density ASCII product (CHAMP / GRACE / GRACE-FO / Swarm; Siemes et
//! al., *J. Space Weather Space Clim.* 2023; <http://thermosphere.tudelft.nl>, CC BY 4.0). The bulk
//! data are an external download (open, keyless) and are *not* vendored here — only a small sample is
//! kept as a parser fixture, with its attribution header intact. Each data row is
//! `date time time_system altitude(m) longitude(deg) latitude(deg) local_solar_time(h)
//! argument_of_latitude(deg) density(kg/m³) [orbit_mean_density(kg/m³) flag flag]`, comment lines
//! start with `#`.
//!
//! **The trailing three fields are not in every mission's product.** The accelerometer files
//! (CHAMP, GRACE, GRACE-FO) carry twelve columns; the Swarm files, whose density comes from precise
//! orbit determination rather than from an accelerometer, carry nine and stop at the density. A
//! parser that insists on twelve reads the whole Swarm archive as zero rows without saying so, so
//! the orbit mean is [`Option`] and the flags default to nominal when absent.
//!
//! Where it is present, the running orbit-mean density makes `ln(ρ / ρ_orbit_mean)` the
//! diurnal/latitudinal density structure — a calibration signal needing no base model.

// Grid-cell indexing casts small counts/indices between usize and f64; `n_lat`/`n_lst` are the
// natural grid-dimension names.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use alloc::vec;
use alloc::vec::Vec;

use crate::density_calibration::DensitySample;

/// The scale a product's timestamps are kept in. The file header says "Time system: UTC or GPS
/// (differs per mission)" and it means it — CHAMP and GRACE-FO are stamped GPS, Swarm UTC — so the
/// column is read rather than assumed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeScale {
    /// Coordinated Universal Time.
    #[default]
    Utc,
    /// GPS time: ahead of UTC by the leap seconds accumulated since 1980 (18 s from 2017 onward).
    Gps,
    /// A time system the product named and this enum does not know.
    Other,
}

/// One accelerometer density observation from the TU Delft product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityObservation {
    /// Calendar year.
    pub year: i32,
    /// Calendar month, `1..=12`.
    pub month: u32,
    /// Day of month, `1..=31`.
    pub day: u32,
    /// Seconds elapsed since midnight, in [`Self::time_scale`].
    ///
    /// Every empirical thermosphere model wants a universal time — NRLMSISE-00 takes it directly as
    /// `sec`, JB2008 as a fraction of the day since 1950, DTM2020 through the day-of-year fraction —
    /// so without it an observation cannot be handed to the models it exists to test.
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
    /// Instantaneous neutral mass density, kg/m³.
    pub density: f64,
    /// Running orbit-mean density, kg/m³, where the mission's product carries one — the
    /// accelerometer files do, the Swarm orbit-determination files do not.
    pub orbit_mean_density: Option<f64>,
    /// `true` when the data flag is 0 (nominal).
    pub nominal: bool,
}

impl DensityObservation {
    /// The density height-normalized to `reference_altitude` (metres) through an exponential scale
    /// height `scale_height` (metres), as a natural log:
    /// `ln ρ_ref = ln ρ + (altitude − reference_altitude) / scale_height`. Unlike the orbit-mean ratio
    /// ([`orbit_mean_samples`]), this keeps the **absolute amplitude** — the part of the density that
    /// tracks solar flux — after removing the first-order altitude dependence, so a multi-year series
    /// of it is what predictive space-weather dynamics is fit against.
    #[must_use]
    pub fn normalized_log_density(&self, reference_altitude: f64, scale_height: f64) -> f64 {
        libm::log(self.density) + (self.altitude - reference_altitude) / scale_height
    }

    /// Day of year, `1..=366`, the seasonal argument every empirical model takes.
    #[must_use]
    pub const fn day_of_year(&self) -> u32 {
        let leap = self.year % 4 == 0 && (self.year % 100 != 0 || self.year % 400 == 0);
        // Cumulative days before each month in a common year.
        let before = [0u32, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let m = if self.month >= 1 && self.month <= 12 {
            self.month as usize - 1
        } else {
            0
        };
        before[m] + self.day + if leap && self.month > 2 { 1 } else { 0 }
    }
}

/// Parse the TU Delft thermosphere density ASCII text. Comment lines (`#`) and malformed rows are
/// skipped; degrees are converted to radians.
#[must_use]
pub fn parse_density(text: &str) -> Vec<DensityObservation> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 9 {
            continue;
        }
        let d: Vec<&str> = t[0].split('-').collect();
        if d.len() != 3 {
            continue;
        }
        let (Ok(year), Ok(month), Ok(day)) = (d[0].parse(), d[1].parse(), d[2].parse()) else {
            continue;
        };
        // A date that parses is not yet a date. `day_of_year` indexes a twelve-entry table by month,
        // so a corrupt `2000-13-05` is malformed in the sense this parser already promises to skip.
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
        let (Some(alt), Some(lon), Some(lat), Some(lst), Some(rho)) =
            (num(t[3]), num(t[4]), num(t[5]), num(t[6]), num(t[8]))
        else {
            continue;
        };
        let rho_mean = t.get(9).and_then(|s| num(s));
        let nominal = t.get(10).and_then(|s| num(s)).is_none_or(|f| f == 0.0);
        out.push(DensityObservation {
            year,
            month,
            day,
            seconds_of_day: 3600.0 * hh + 60.0 * mm + ss,
            time_scale,
            altitude: alt,
            longitude: lon.to_radians(),
            latitude: lat.to_radians(),
            local_solar_time: lst,
            density: rho,
            orbit_mean_density: rho_mean,
            nominal,
        });
    }
    out
}

/// Turn observations into calibration samples of `ln(ρ / ρ_orbit_mean)` at their (latitude, local
/// solar time) — the density structure relative to the orbit average, ready for
/// [`crate::density_calibration::DensityCorrection::fit`]. Keeps only nominal observations with
/// positive densities and a positive orbit mean, so a product that carries no orbit mean at all
/// (Swarm) yields no samples rather than a division by a default.
#[must_use]
pub fn orbit_mean_samples(observations: &[DensityObservation]) -> Vec<DensitySample> {
    observations
        .iter()
        .filter(|o| o.nominal && o.density > 0.0)
        .filter_map(|o| {
            let mean = o.orbit_mean_density?;
            (mean > 0.0).then_some(DensitySample {
                latitude: o.latitude,
                local_solar_time: o.local_solar_time,
                log_ratio: libm::log(o.density / mean),
                weight: 1.0,
            })
        })
        .collect()
}

/// Bin samples into an `n_lat × n_lst` field of the mean `log_ratio` per (latitude, local-solar-time)
/// cell, flattened row-major as `lat_bin * n_lst + lst_bin`. Latitude maps `[-π/2, π/2] → [0, n_lat)`,
/// local solar time `[0, 24) → [0, n_lst)`; empty cells are `0.0`. This is a density-field snapshot for
/// [`crate::density_rom::DensityRom`] POD, or a coverage view of a data window. Returns empty if either
/// dimension is zero.
#[must_use]
pub fn grid_samples(samples: &[DensitySample], n_lat: usize, n_lst: usize) -> Vec<f64> {
    if n_lat == 0 || n_lst == 0 {
        return Vec::new();
    }
    let cells = n_lat * n_lst;
    let mut sum = vec![0.0_f64; cells];
    let mut count = vec![0_usize; cells];
    for s in samples {
        let lat_frac = (s.latitude + core::f64::consts::FRAC_PI_2) / core::f64::consts::PI;
        let lst_frac = s.local_solar_time / 24.0;
        let li = ((lat_frac * n_lat as f64) as usize).min(n_lat - 1);
        let lj = ((lst_frac * n_lst as f64) as usize).min(n_lst - 1);
        let idx = li * n_lst + lj;
        sum[idx] += s.log_ratio;
        count[idx] += 1;
    }
    (0..cells)
        .map(|i| {
            if count[i] > 0 {
                sum[i] / count[i] as f64
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp, clippy::unreadable_literal)]

    use super::*;
    use crate::density_calibration::DensityCorrection;

    const FIXTURE: &str = include_str!("../tests/fixtures/champ_density_sample.txt");

    #[test]
    fn parses_the_real_champ_fixture() {
        let obs = parse_density(FIXTURE);
        assert_eq!(obs.len(), 30);
        // First data row: 2000-07-29, alt 471231.023 m, lat 39.87860857°, lst 0.963 h, ρ 0.11632377e-11.
        let o = obs[0];
        assert_eq!((o.year, o.month, o.day), (2000, 7, 29));
        // 01:00:00.000, stamped GPS — CHAMP's clock is the GPS receiver's, not UTC.
        assert!((o.seconds_of_day - 3600.0).abs() < 1e-12);
        assert_eq!(o.time_scale, TimeScale::Gps);
        assert_eq!(o.day_of_year(), 211);
        // Height-normalizing to its own altitude is a no-op; +1 scale height adds ln e = 1.
        assert!(
            (o.normalized_log_density(o.altitude, 60_000.0) - libm::log(o.density)).abs() < 1e-12
        );
        assert!(
            (o.normalized_log_density(o.altitude - 60_000.0, 60_000.0)
                - libm::log(o.density)
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!((o.altitude - 471_231.023).abs() < 1e-3);
        assert!((o.latitude - 39.878_608_57_f64.to_radians()).abs() < 1e-12);
        assert!((o.local_solar_time - 0.963).abs() < 1e-12);
        assert!((o.density - 0.11632377e-11).abs() < 1e-22);
        assert!((o.orbit_mean_density.unwrap() - 0.19451640e-11).abs() < 1e-22);
        assert!(o.nominal);
    }

    #[test]
    fn orbit_mean_samples_are_log_density_ratios() {
        let obs = parse_density(FIXTURE);
        let samples = orbit_mean_samples(&obs);
        assert_eq!(samples.len(), 30);
        assert!((samples[0].log_ratio - libm::log(0.11632377e-11 / 0.19451640e-11)).abs() < 1e-12);
    }

    #[test]
    fn fits_a_low_degree_correction_to_the_real_samples() {
        let obs = parse_density(FIXTURE);
        let samples = orbit_mean_samples(&obs);
        // 30 real samples determine a degree-1 (4-coefficient) correction.
        let correction = DensityCorrection::<4>::fit(&samples).unwrap();
        let factor = correction.factor(0.0, 12.0);
        assert!(factor.is_finite() && factor > 0.0, "factor {factor}");
    }

    #[test]
    fn grid_samples_averages_per_cell() {
        let s = [
            DensitySample {
                latitude: (-45.0_f64).to_radians(),
                local_solar_time: 3.0,
                log_ratio: 1.0,
                weight: 1.0,
            },
            DensitySample {
                latitude: (-45.0_f64).to_radians(),
                local_solar_time: 3.0,
                log_ratio: 3.0,
                weight: 1.0,
            },
            DensitySample {
                latitude: 45.0_f64.to_radians(),
                local_solar_time: 18.0,
                log_ratio: -1.0,
                weight: 1.0,
            },
        ];
        let g = grid_samples(&s, 2, 2);
        assert_eq!(g.len(), 4);
        assert!((g[0] - 2.0).abs() < 1e-12); // (south, morning): mean of 1 and 3
        assert!((g[3] + 1.0).abs() < 1e-12); // (north, evening): -1
        assert!(g[1].abs() < 1e-12); // empty cell
    }

    /// Ten fields is exactly enough — the parser reads `t[0..=6]`, `t[8]` and `t[9]`, and treats
    /// `t[10]` as optional. The CHAMP fixture is eleven wide throughout, so the boundary is untested
    /// in both directions: a ten-field row has to be kept, and a row too narrow to index has to be
    /// dropped rather than run off the end of `t`.
    ///
    /// A row whose clock is unreadable is dropped for the same reason a row too narrow to index is:
    /// the fields the caller came for are not there. The alternative — defaulting to midnight —
    /// would put a real density at a fabricated universal time.
    #[test]
    fn parse_density_takes_a_row_at_the_minimum_width_and_drops_a_narrower_one() {
        let text = "\
2000-07-29 01:02:03.500 GPS 471231.023 10.0 39.87860857 0.963 0 0.11632377e-11 0.19451640e-11
2000-07-30 01:02:03.500 GPS 471000.0
2000-07-31 midnight GPS 471000.0 10.0 39.0 0.9 0 0.1e-11 0.2e-11
";
        let obs = parse_density(text);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].day, 29);
        assert!((obs[0].seconds_of_day - 3723.5).abs() < 1e-12);
        assert_eq!(obs[0].time_scale, TimeScale::Gps);
        assert!(obs[0].nominal, "a row with no flag field is nominal");
    }

    /// Three things about a row that the CHAMP fixture cannot say, because it is eleven fields wide,
    /// stamped GPS throughout and carries no date the mission did not fly.
    ///
    /// A **nine-field** row is the Swarm width — the narrowest the parser accepts, and the module
    /// header's reason for accepting it at all — so the width test has to be strict at exactly nine
    /// or the whole Swarm archive reads as zero rows. **UTC** is the other half of the time-system
    /// column: reading it as `Other` would lose the eighteen seconds that separate the two scales at
    /// the moment the observation is handed to a model. And a date that *parses* is not yet a date:
    /// month 13 is what a corrupt file offers a table with twelve entries.
    #[test]
    fn parse_density_takes_the_swarm_width_reads_utc_and_drops_an_impossible_date() {
        let text = "\
2000-07-29 01:02:03.500 UTC 471231.023 10.0 39.87860857 0.963 0 0.11632377e-11
2000-13-05 01:02:03.500 UTC 471231.023 10.0 39.87860857 0.963 0 0.11632377e-11
2000-00-05 01:02:03.500 UTC 471231.023 10.0 39.87860857 0.963 0 0.11632377e-11
2000-07-32 01:02:03.500 UTC 471231.023 10.0 39.87860857 0.963 0 0.11632377e-11
2000-07-00 01:02:03.500 UTC 471231.023 10.0 39.87860857 0.963 0 0.11632377e-11
";
        let obs = parse_density(text);
        assert_eq!(
            obs.len(),
            1,
            "nine fields is a row; a bad month or day is not"
        );
        assert_eq!((obs[0].year, obs[0].month, obs[0].day), (2000, 7, 29));
        assert_eq!(obs[0].time_scale, TimeScale::Utc);
        assert_eq!(
            obs[0].orbit_mean_density, None,
            "the Swarm width stops at the density"
        );
    }

    /// The day-of-year the models take as their seasonal argument, at the four places the month
    /// table and the leap rule can disagree: the day before and the day after 29 February in a leap
    /// year, and 1 March in a century that is not one.
    #[test]
    fn the_day_of_year_follows_the_gregorian_leap_rule() {
        let at = |year, month, day| {
            DensityObservation {
                year,
                month,
                day,
                seconds_of_day: 0.0,
                time_scale: TimeScale::Utc,
                altitude: 4.7e5,
                longitude: 0.0,
                latitude: 0.0,
                local_solar_time: 0.0,
                density: 1.0e-12,
                orbit_mean_density: Some(1.0e-12),
                nominal: true,
            }
            .day_of_year()
        };

        assert_eq!(at(2024, 2, 28), 59);
        assert_eq!(at(2024, 2, 29), 60);
        assert_eq!(at(2024, 3, 1), 61);
        assert_eq!(at(2024, 12, 31), 366);
        // 1900 is divisible by 4 and by 100 but not by 400, so it is not a leap year; 2000 is.
        assert_eq!(at(1900, 3, 1), 60);
        assert_eq!(at(2000, 3, 1), 61);
        assert_eq!(at(2019, 1, 1), 1);
        assert_eq!(at(2019, 12, 31), 365);

        // `month` is a public `u32` and the table it indexes has twelve entries, so the range check
        // is what stands between a hand-built observation and reading off the end of it: month 13
        // indexes `before[12]` and month 0 underflows the `- 1` into `usize::MAX`. Both fall back to
        // January's offset. That is a floor, not a calendar answer, and it is all this asserts —
        // `parse_density` refuses such a date outright, so nothing the parser produces reaches it.
        assert_eq!(at(2019, 0, 5), 5);
        assert_eq!(at(2019, 13, 5), 5);
    }

    /// The filter drops non-nominal observations and non-positive densities, and the fixture is
    /// thirty rows of clean CHAMP data with none of either — so nothing exercised any of the three
    /// conjuncts, nor the strictness of the two comparisons. A zero density is the case that
    /// matters: it is what the instrument writes for a rejected measurement, and `ln(0/ρ̄)` is −∞.
    #[test]
    fn orbit_mean_samples_keeps_only_positive_nominal_rows() {
        let good = DensityObservation {
            year: 2000,
            month: 7,
            day: 29,
            seconds_of_day: 3600.0,
            time_scale: TimeScale::Gps,
            altitude: 471_231.0,
            longitude: 0.1,
            latitude: 0.2,
            local_solar_time: 3.0,
            density: 2.0e-12,
            orbit_mean_density: Some(1.0e-12),
            nominal: true,
        };
        let all = [
            good,
            DensityObservation {
                nominal: false,
                ..good
            },
            DensityObservation {
                density: 0.0,
                ..good
            },
            DensityObservation {
                orbit_mean_density: Some(0.0),
                ..good
            },
            // Swarm's product stops at the density column, so there is no orbit mean to divide by.
            DensityObservation {
                orbit_mean_density: None,
                ..good
            },
        ];
        let samples = orbit_mean_samples(&all);
        assert_eq!(samples.len(), 1);
        assert!((samples[0].log_ratio - core::f64::consts::LN_2).abs() < 1e-15);
    }

    /// The bin index is a truncation, so a sample at the top of either range — the north pole, or
    /// 24 h local solar time — computes an index one past the last cell and is saved only by the
    /// clamp. Real data never reaches either edge, which leaves the clamp and the zero-dimension
    /// guard as the two ways this function indexes off the end of its own buffer.
    #[test]
    fn grid_samples_holds_its_domain_edges() {
        let at = |latitude, local_solar_time| DensitySample {
            latitude,
            local_solar_time,
            log_ratio: 1.0,
            weight: 1.0,
        };
        let g = grid_samples(
            &[at(core::f64::consts::FRAC_PI_2, 24.0), at(0.0, 0.0)],
            2,
            3,
        );
        assert_eq!(g.len(), 6);
        assert!(
            (g[5] - 1.0).abs() < 1e-12,
            "the pole at 24 h is the last cell"
        );
        assert!((g[3] - 1.0).abs() < 1e-12, "the equator at 0 h");

        // Either dimension zero is refused before anything is indexed.
        assert_eq!(grid_samples(&[at(0.0, 0.0)], 0, 3).len(), 0);
        assert_eq!(grid_samples(&[at(0.0, 0.0)], 3, 0).len(), 0);
    }
}
