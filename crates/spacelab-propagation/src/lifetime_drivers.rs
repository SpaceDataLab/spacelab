//! Empirical atmospheres for a decay, driven by a measured index history or a guess at its future.
//!
//! [`crate::lifetime`] takes any [`DecayAtmosphere`], and the question a lifetime is bought to answer
//! is what the Sun will do. This module holds the three answers the campaign in
//! `docs/verification.md` §10.37 scores. [`Scenario::Observed`] reads the history: the model's own
//! error, with the driver known. [`Scenario::Persistence`] holds the state at the epoch: the baseline.
//! [`Scenario::Analog`] replays a past solar cycle at the same phase, its flux scaled to meet the
//! present one: a member of an ensemble that knows nothing after its epoch.
//!
//! # The geometry
//!
//! The mean-element quadrature hands over an inertial position. Altitude and latitude are geodetic on
//! the WGS-84 ellipsoid by Bowring's closed form, because the models are fitted on geodetic altitude
//! and a sphere would be 21 km wrong at the equator. Longitude comes from the Earth Rotation Angle.
//! Precession and polar motion are left out: over a decay they move the density's local solar time
//! by far less than the orbit's own node does, and the position is mean-element-grade anyway.
//!
//! # The interval
//!
//! [`conformal_radius`] turns a calibration set of absolute log errors into a window with
//! distribution-free coverage. The law is proved in `derivations/split_conformal_coverage.mac`.
//!
//! Gates: the tests below hold the sampling to `SdtmDrivers` and `gtd7d` bit for bit, the analog's
//! flux to its closed form at `t = 0`, `t = τ` and `t ≫ τ`, the geodetic map to the forward ellipsoid
//! formula, and the conformal radius to the exact exchangeable coverage `k/(n + 1)`.

use alloc::vec::Vec;

use crate::Tolerance;
use crate::lifetime::{DecayAtmosphere, smoother_step};
use crate::nrlmsise00::{Nrlmsise00Flags, Nrlmsise00Input, gtd7d};
use crate::sdtm::{
    BOUNDARY_ALTITUDE, Envelope, SdtmDrivers, SdtmPoint, StormResponse, Thermosphere,
};
use crate::space_weather::{SpaceWeatherDay, THREE_HOURLY_CADENCE, days_from_civil};

const SECONDS_PER_DAY: f64 = 86_400.0;
const INTERVALS_PER_DAY: usize = 8;
const UNIX_DAYS_AT_J2000: f64 = 10_957.5;
/// WGS-84 semi-major axis, m.
const WGS84_A: f64 = 6_378_137.0;
/// WGS-84 inverse flattening.
const WGS84_INVERSE_F: f64 = 298.257_223_563;
/// IERS 2010 Earth rotation rate, rad/s.
const EARTH_ROTATION: f64 = 7.292_115_0e-5;
/// Far above the rounding of a Unix time, 1.2e-7 s, and far below the 3 h grid.
const GRID_SLACK: f64 = 1.0e-3;
/// Below this height the density stops rising, m. A decay ends at
/// [`crate::lifetime::DEFAULT_REENTRY_ALTITUDE`], 10 km above it, so nothing reported reads this
/// region; it only spares the bracketing run the stiff last kilometres, where an unbounded density
/// collapses the step until the run is abandoned.
const DENSITY_FLOOR_HEIGHT: f64 = 110_000.0;
/// How far inside the envelope's height limits the fitted field blends into NRLMSISE-00, m.
const ENVELOPE_BLEND: f64 = 10_000.0;

/// One day of the indices both models read.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Day {
    f107_previous: f64,
    f107a_trailing: f64,
    f107a_centred: f64,
    ap: f64,
    ordinal: f64,
}

/// The indices at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriverState {
    /// F10.7 of the previous day.
    pub f107: f64,
    /// 81-day trailing mean: SDTM-1's flux.
    pub f107a_trailing: f64,
    /// 81-day centred mean: NRLMSISE-00's flux.
    pub f107a_centred: f64,
    /// Daily `Ap`: NRLMSISE-00's geomagnetic index.
    pub ap: f64,
    /// SDTM-1's storm response.
    pub storm: f64,
}

/// A contiguous daily index history with the storm state already run over it.
#[derive(Debug, Clone)]
pub struct DriverHistory {
    first_day: f64,
    days: Vec<Day>,
    storm: Vec<f64>,
}

impl DriverHistory {
    /// Build from days ascending without a gap, running `storm` over `warmup` three-hourly intervals.
    ///
    /// The table starts on the first day whose storm state has its whole warm-up. `None` for a gap in
    /// the dates or a feed too short to warm up: a shortened history reads as a quieter storm.
    #[must_use]
    pub fn from_days(
        days: &[SpaceWeatherDay],
        storm: &StormResponse,
        warmup: usize,
    ) -> Option<Self> {
        let number = |d: &SpaceWeatherDay| days_from_civil(d.year, d.month, d.day);
        if days.windows(2).any(|w| number(&w[1]) != number(&w[0]) + 1) {
            return None;
        }
        let skip = warmup.div_ceil(INTERVALS_PER_DAY).max(1);
        if days.len() <= skip {
            return None;
        }
        let ap: Vec<f64> = days.iter().flat_map(|d| d.ap3h).collect();
        let table = days[skip..]
            .iter()
            .zip(&days[skip - 1..])
            .map(|(today, yesterday)| Day {
                f107_previous: yesterday.f107_observed,
                f107a_trailing: today.f107_last81,
                f107a_centred: today.f107_center81,
                ap: today.ap,
                ordinal: small_integer(number(today) - days_from_civil(today.year, 1, 1) + 1),
            })
            .collect();
        let first_interval = skip * INTERVALS_PER_DAY;
        let response = (first_interval..ap.len())
            .map(|k| storm.response(&ap[k - warmup..k], THREE_HOURLY_CADENCE))
            .collect();
        Some(Self {
            first_day: small_integer(number(&days[skip])),
            days: table,
            storm: response,
        })
    }

    /// The indices at `unix_seconds`, holding the first or last value outside the table.
    #[must_use]
    pub fn state(&self, unix_seconds: f64) -> DriverState {
        let (day, _) = self.day(unix_seconds);
        let interval =
            libm::floor((unix_seconds - self.first_day * SECONDS_PER_DAY) / THREE_HOURLY_CADENCE);
        let last = self.storm.len() - 1;
        DriverState {
            f107: day.f107_previous,
            f107a_trailing: day.f107a_trailing,
            f107a_centred: day.f107a_centred,
            ap: day.ap,
            storm: self.storm[clamp_index(interval, last)],
        }
    }

    fn day(&self, unix_seconds: f64) -> (Day, f64) {
        let number = libm::floor(unix_seconds / SECONDS_PER_DAY);
        let index = clamp_index(number - self.first_day, self.days.len() - 1);
        (self.days[index], unix_seconds - number * SECONDS_PER_DAY)
    }
}

/// A day number or an ordinal: far below 2⁵³, so it converts exactly.
#[expect(clippy::cast_precision_loss, reason = "a day number, about 2e4")]
const fn small_integer(value: i64) -> f64 {
    value as f64
}

/// The index of a value that is finite by construction, clamped into `0..=last`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the value is positive and below `last` where it is cast, and `last` is a table length"
)]
fn clamp_index(value: f64, last: usize) -> usize {
    if value <= 0.0 {
        0
    } else if value >= last as f64 {
        last
    } else {
        value as usize
    }
}

/// What the indices do after the epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scenario {
    /// The measured history. Not a forecast: it scores the model with the driver known.
    Observed,
    /// The state of the epoch, held. The baseline any forecast has to beat.
    Persistence(DriverState),
    /// A past stretch of the history, `shift_days` earlier, with both flux means and the daily flux
    /// multiplied by `1 + (ratio − 1)·exp(−t/tau)`.
    Analog {
        /// How far back the member reads, days.
        shift_days: f64,
        /// Present flux over the analog's at the epoch.
        ratio: f64,
        /// E-folding time of that ratio, s.
        tau: f64,
    },
}

impl Scenario {
    /// The indices `t` seconds after `epoch`, a Unix time.
    #[must_use]
    pub fn state(&self, history: &DriverHistory, epoch: f64, t: f64) -> DriverState {
        match *self {
            Self::Observed => history.state(epoch + t),
            Self::Persistence(state) => state,
            Self::Analog {
                shift_days,
                ratio,
                tau,
            } => {
                let past = history.state(epoch + t - shift_days * SECONDS_PER_DAY);
                let factor = 1.0 + (ratio - 1.0) * libm::exp(-t / tau);
                DriverState {
                    f107: past.f107 * factor,
                    f107a_trailing: past.f107a_trailing * factor,
                    f107a_centred: past.f107a_centred * factor,
                    ..past
                }
            }
        }
    }
}

/// Geodetic latitude (rad) and altitude (m) of a point at distance `rho` from the axis and `z` from the
/// equatorial plane, by Bowring's closed form.
#[must_use]
pub fn geodetic_latitude_altitude(rho: f64, z: f64) -> (f64, f64) {
    let f = 1.0 / WGS84_INVERSE_F;
    let b = WGS84_A * (1.0 - f);
    let e2 = f * (2.0 - f);
    let ep2 = e2 / (1.0 - e2);
    let beta = libm::atan2(WGS84_A * z, b * rho);
    let (sin_beta, cos_beta) = libm::sincos(beta);
    let latitude = libm::atan2(
        z + ep2 * b * sin_beta * sin_beta * sin_beta,
        rho - e2 * WGS84_A * cos_beta * cos_beta * cos_beta,
    );
    let (sin_lat, cos_lat) = libm::sincos(latitude);
    let normal = WGS84_A / libm::sqrt(1.0 - e2 * sin_lat * sin_lat);
    // Projecting on the normal stays conditioned at both the equator and the pole, which `rho/cos` and
    // `z/sin` do not.
    let altitude = rho * cos_lat + z * sin_lat - normal * (1.0 - e2 * sin_lat * sin_lat);
    (latitude, altitude)
}

/// Earth Rotation Angle at `unix_seconds`, rad, IAU 2000, with UTC standing in for UT1.
#[must_use]
pub fn earth_rotation_angle(unix_seconds: f64) -> f64 {
    let days = unix_seconds / SECONDS_PER_DAY - UNIX_DAYS_AT_J2000;
    #[allow(clippy::excessive_precision)]
    let turns = 0.779_057_273_264_0 + 1.002_737_811_911_354_48 * days;
    core::f64::consts::TAU * (turns - libm::floor(turns))
}

/// A tolerance for a decay under an empirical model: 7 mm of semi-major axis a step, a thousandth of
/// the scatter of `a` read from element sets. `tools/lifetime-campaign.py converge` holds the
/// lifetimes to this against a tolerance a hundred times tighter.
pub const SCENARIO_TOLERANCE: Tolerance = Tolerance::new(1e-9, 1e-9);

/// A co-rotating empirical atmosphere reading one [`Scenario`].
///
/// With a `field`, it is SDTM-1 inside the envelope it is given and NRLMSISE-00 outside, blended over the
/// envelope's outer 10 km of height. A decay gives it [`decay_envelope`]. Without one, it is NRLMSISE-00
/// alone.
#[derive(Debug, Clone)]
pub struct ScenarioAtmosphere<'a, const K: usize> {
    /// The index history the scenario reads.
    pub history: &'a DriverHistory,
    /// What the indices do after the epoch.
    pub scenario: Scenario,
    /// Unix time of `t = 0`.
    pub epoch: f64,
    /// The fitted field and the envelope to read it in, or `None` for NRLMSISE-00 alone.
    pub field: Option<(&'a Thermosphere<K>, &'a Envelope)>,
    /// What the propagation is held to, usually [`SCENARIO_TOLERANCE`].
    pub tolerance: Tolerance,
}

impl<const K: usize> DecayAtmosphere for ScenarioAtmosphere<'_, K> {
    fn density(&self, t: f64, position: [f64; 3]) -> f64 {
        self.density_held(t, t, position)
    }

    fn next_jump(&self, t: f64) -> f64 {
        let now = self.epoch + t + GRID_SLACK;
        let grid = libm::floor(now / THREE_HOURLY_CADENCE) + 1.0;
        grid * THREE_HOURLY_CADENCE - self.epoch
    }

    fn tolerance(&self) -> Tolerance {
        self.tolerance
    }

    fn density_held(&self, segment_start: f64, t: f64, position: [f64; 3]) -> f64 {
        let now = self.epoch + t;
        let [x, y, z] = position;
        let (latitude, altitude) = geodetic_latitude_altitude(libm::hypot(x, y), z);
        let altitude = altitude.max(DENSITY_FLOOR_HEIGHT);
        let east = libm::atan2(y, x) - earth_rotation_angle(now);
        // The slack puts a segment that starts on the grid on the grid's far side, whichever way the
        // sum `epoch + t` rounded.
        let held = segment_start + GRID_SLACK;
        let (day, _) = self.history.day(self.epoch + held);
        let second = now - libm::floor((self.epoch + held) / SECONDS_PER_DAY) * SECONDS_PER_DAY;
        let state = self.scenario.state(self.history, self.epoch, held);
        let hours = second / 3_600.0 + east.to_degrees() / 15.0;
        let local = hours - 24.0 * libm::floor(hours / 24.0);
        if let Some((field, envelope)) = self.field {
            let point = SdtmPoint {
                altitude,
                latitude,
                local_solar_time: local,
                day_of_year: day.ordinal + second / SECONDS_PER_DAY,
            };
            let drivers = SdtmDrivers {
                f107: state.f107,
                f107a: state.f107a_trailing,
                storm: state.storm,
            };
            if envelope.contains(&point, &drivers) {
                let weight = field_weight(envelope, altitude);
                let density = field.density(&point, &drivers);
                if weight >= 1.0 {
                    return density;
                }
                let outside = Self::nrlmsise(&day, second, altitude, latitude, east, local, &state);
                return libm::exp(
                    weight * libm::log(density) + (1.0 - weight) * libm::log(outside),
                );
            }
        }
        Self::nrlmsise(&day, second, altitude, latitude, east, local, &state)
    }

    fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
        [
            -EARTH_ROTATION * position[1],
            EARTH_ROTATION * position[0],
            0.0,
        ]
    }
}

impl<const K: usize> ScenarioAtmosphere<'_, K> {
    fn nrlmsise(
        day: &Day,
        second: f64,
        altitude: f64,
        latitude: f64,
        east: f64,
        local: f64,
        state: &DriverState,
    ) -> f64 {
        let wrapped = east - core::f64::consts::TAU * libm::floor(east / core::f64::consts::TAU);
        let input = Nrlmsise00Input {
            year: 0,
            // A day of the year, 1..=366.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "an ordinal day, at most 366"
            )]
            doy: day.ordinal as i32,
            sec: second,
            alt: altitude / 1_000.0,
            g_lat: latitude.to_degrees(),
            g_long: wrapped.to_degrees(),
            lst: local,
            f107a: state.f107a_centred,
            f107: state.f107,
            ap: state.ap,
            ap_array: None,
        };
        // d[5] is g/cm3 from gtd7d, anomalous oxygen included.
        gtd7d(&input, &Nrlmsise00Flags::default()).d[5] * 1_000.0
    }
}

/// Where a decay reads a field fitted in `fitted`: down to the field's own boundary, and as far up and
/// over the same fluxes as it was fitted.
///
/// `docs/verification.md` §10.28 recommends NRLMSISE-00 below the fitted limit, and for a density at one
/// point that stands. A decay is different, because it carries a ballistic coefficient fitted through
/// the density above the limit. The field's density is lower than NRLMSISE-00's there, so the fitted
/// coefficient is larger, and handing it to NRLMSISE-00 below the limit brings the object down too fast
/// (§10.37). Below the limit the field is checked only through the lifetimes it gives.
#[must_use]
pub const fn decay_envelope(fitted: &Envelope) -> Envelope {
    Envelope {
        altitude: core::ops::RangeInclusive::new(BOUNDARY_ALTITUDE, *fitted.altitude.end()),
        f107a: core::ops::RangeInclusive::new(*fitted.f107a.start(), *fitted.f107a.end()),
    }
}

/// The weight of the fitted field at `altitude`: 1 inside, falling to 0 at the envelope's height limits.
///
/// The recommendation is a switch, and the two models differ at it. An orbit's quadrature nodes cross
/// the switch one at a time, so each crossing is a jump in the mean rate, and the propagation's step
/// collapses on every one of them. The log density therefore blends with a C² polynomial over the outer
/// [`ENVELOPE_BLEND`] of the envelope, and a blended point reads both models.
fn field_weight(envelope: &Envelope, altitude: f64) -> f64 {
    let lower = (altitude - envelope.altitude.start()) / ENVELOPE_BLEND;
    let upper = (envelope.altitude.end() - altitude) / ENVELOPE_BLEND;
    smoother_step(lower.min(upper))
}

/// The split-conformal radius of level `1 − alpha`: the `k`-th smallest score, `k = ⌈(n + 1)(1 − alpha)⌉`.
///
/// For a new score exchangeable with these, it is at or below the radius with probability exactly
/// `k/(n + 1)` when the scores are distinct, which lies in `[1 − alpha, 1 − alpha + 1/(n + 1))`.
///
/// `None` when `k > n`, because no finite radius then has the coverage, and for an `alpha` outside
/// `(0, 1)` or a score that is not a number.
#[must_use]
pub fn conformal_radius(scores: &[f64], alpha: f64) -> Option<f64> {
    if scores.iter().any(|s| s.is_nan()) {
        return None;
    }
    let n = scores.len();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "a saturating cast: an `alpha` outside (0, 1), or NaN, lands on 0 or past `n`, refused below"
    )]
    let k = libm::ceil((n as f64 + 1.0) * (1.0 - alpha)) as usize;
    if k > n || k == 0 {
        return None;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[k - 1])
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::unwrap_used,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::assertions_on_constants
    )]

    use super::*;
    use crate::sdtm::coefficients::{ENVELOPE, FIELD, HARMONICS};

    fn feed(count: usize) -> Vec<SpaceWeatherDay> {
        (0..count)
            .map(|i| {
                let x = i as f64;
                SpaceWeatherDay {
                    year: 2001,
                    month: 1 + (i / 28) as u32,
                    day: 1 + (i % 28) as u32,
                    ap: 5.0 + (i % 7) as f64,
                    ap3h: core::array::from_fn(|j| 4.0 + ((i + j) % 9) as f64),
                    kp3h: [0.0; 8],
                    f107_observed: 100.0 + x,
                    f107_adjusted: 100.0 + x,
                    f107_center81: 120.0 + x,
                    f107_last81: 110.0 + x,
                    quality: crate::space_weather::FluxQuality::Observed,
                }
            })
            .collect()
    }

    /// 28 days a month keeps the fake feed contiguous only inside one month, so these use one.
    fn history() -> DriverHistory {
        DriverHistory::from_days(&feed(28), FIELD.storm(), 16).unwrap()
    }

    #[test]
    fn a_gap_or_a_feed_too_short_to_warm_up_is_refused() {
        let mut days = feed(28);
        days.remove(10);
        assert!(DriverHistory::from_days(&days, FIELD.storm(), 16).is_none());
        assert!(DriverHistory::from_days(&feed(2), FIELD.storm(), 16).is_none());
        assert!(DriverHistory::from_days(&feed(3), FIELD.storm(), 16).is_some());
    }

    #[test]
    fn the_table_applies_each_models_own_index_conventions() {
        let days = feed(28);
        let h = history();
        let unix = days_from_civil(2001, 1, 11) as f64 * SECONDS_PER_DAY + 10_800.0;
        let s = h.state(unix);
        assert_eq!(s.f107, days[9].f107_observed);
        assert_eq!(s.f107a_trailing, days[10].f107_last81);
        assert_eq!(s.f107a_centred, days[10].f107_center81);
        assert_eq!(s.ap, days[10].ap);
        let flat: Vec<f64> = days.iter().flat_map(|d| d.ap3h).collect();
        let k = 10 * 8 + 1;
        let want = FIELD
            .storm()
            .response(&flat[k - 16..k], THREE_HOURLY_CADENCE);
        assert_eq!(s.storm, want);
    }

    #[test]
    fn outside_the_table_it_holds_the_nearest_value() {
        let h = history();
        let first = h.days[0];
        let last = *h.days.last().unwrap();
        assert_eq!(h.state(0.0).f107, first.f107_previous);
        assert_eq!(h.state(1.0e12).f107, last.f107_previous);
        assert_eq!(h.state(0.0).storm, h.storm[0]);
        assert_eq!(h.state(1.0e12).storm, *h.storm.last().unwrap());
    }

    #[test]
    fn an_analog_with_no_shift_and_no_ratio_is_the_observed_history() {
        let h = history();
        let epoch = days_from_civil(2001, 1, 9) as f64 * SECONDS_PER_DAY;
        let analog = Scenario::Analog {
            shift_days: 0.0,
            ratio: 1.0,
            tau: 1.0e6,
        };
        for t in [0.0, 1.0e4, 3.0e5, 1.0e6] {
            assert_eq!(
                analog.state(&h, epoch, t),
                Scenario::Observed.state(&h, epoch, t)
            );
        }
    }

    #[test]
    fn the_analogs_flux_factor_is_its_closed_form() {
        let h = history();
        let epoch = days_from_civil(2001, 1, 15) as f64 * SECONDS_PER_DAY;
        let tau = 2.0e5;
        let scenario = Scenario::Analog {
            shift_days: 5.0,
            ratio: 1.5,
            tau,
        };
        let past = |t: f64| h.state(epoch + t - 5.0 * SECONDS_PER_DAY);
        let at_zero = scenario.state(&h, epoch, 0.0);
        assert_eq!(at_zero.f107, past(0.0).f107 * 1.5);
        assert_eq!(at_zero.f107a_trailing, past(0.0).f107a_trailing * 1.5);
        assert_eq!(at_zero.f107a_centred, past(0.0).f107a_centred * 1.5);
        assert_eq!(at_zero.ap, past(0.0).ap);
        assert_eq!(at_zero.storm, past(0.0).storm);
        let at_tau = scenario.state(&h, epoch, tau);
        let want = past(tau).f107 * (1.0 + 0.5 * core::f64::consts::E.recip());
        assert!((at_tau.f107 / want - 1.0).abs() < 1e-15);
        let late = scenario.state(&h, epoch, 200.0 * tau);
        assert_eq!(late.f107, past(200.0 * tau).f107);
    }

    #[test]
    fn persistence_is_the_state_it_was_given_at_any_time() {
        let h = history();
        let held = h.state(days_from_civil(2001, 1, 9) as f64 * SECONDS_PER_DAY);
        let scenario = Scenario::Persistence(held);
        assert_eq!(scenario.state(&h, 0.0, 1.0e9), held);
    }

    /// The forward ellipsoid formula, so the inverse is held to something that is not itself.
    fn forward(latitude: f64, altitude: f64) -> (f64, f64) {
        let f = 1.0 / WGS84_INVERSE_F;
        let e2 = f * (2.0 - f);
        let (s, c) = libm::sincos(latitude);
        let normal = WGS84_A / libm::sqrt(1.0 - e2 * s * s);
        (
            (normal + altitude) * c,
            (normal * (1.0 - e2) + altitude) * s,
        )
    }

    #[test]
    fn the_geodetic_map_inverts_the_forward_ellipsoid_formula() {
        let (mut worst_lat, mut worst_alt) = (0.0_f64, 0.0_f64);
        for lat_deg in [-90.0_f64, -67.0, -45.0, -30.0, 0.0, 12.0, 45.0, 80.0, 90.0] {
            for altitude in [120e3, 225e3, 400e3, 900e3] {
                let (rho, z) = forward(lat_deg.to_radians(), altitude);
                let (lat, alt) = geodetic_latitude_altitude(rho, z);
                worst_lat = worst_lat.max((lat - lat_deg.to_radians()).abs());
                worst_alt = worst_alt.max((alt - altitude).abs());
            }
        }
        // One Bowring step: 7.6e-10 rad measured, 5 mm on the ground, and the altitude is second order.
        assert!(worst_lat < 1e-9, "{worst_lat}");
        assert!(worst_alt < 1e-8, "{worst_alt}");
    }

    #[test]
    fn the_rotation_angle_is_the_iau_2000_definition() {
        let j2000 = UNIX_DAYS_AT_J2000 * SECONDS_PER_DAY;
        let want = core::f64::consts::TAU * 0.779_057_273_264_0;
        assert!((earth_rotation_angle(j2000) - want).abs() < 1e-12);
        let one_day = earth_rotation_angle(j2000 + SECONDS_PER_DAY);
        let want = core::f64::consts::TAU * (0.779_057_273_264_0 + 0.002_737_811_911_354_48);
        assert!((one_day - want).abs() < 1e-9);
        for k in 0..50 {
            let angle = earth_rotation_angle(j2000 + 7.3e6 * f64::from(k));
            assert!((0.0..core::f64::consts::TAU).contains(&angle));
        }
    }

    fn atmosphere(field: bool) -> ScenarioAtmosphere<'static, HARMONICS> {
        static ENV: Envelope = ENVELOPE;
        let history = alloc::boxed::Box::leak(alloc::boxed::Box::new(history()));
        ScenarioAtmosphere {
            history,
            scenario: Scenario::Observed,
            epoch: days_from_civil(2001, 1, 15) as f64 * SECONDS_PER_DAY + 3_600.0,
            field: field.then_some((&FIELD, &ENV)),
            tolerance: SCENARIO_TOLERANCE,
        }
    }

    /// The point `position` is, by the same arithmetic the atmosphere must use.
    fn sample(
        a: &ScenarioAtmosphere<'_, HARMONICS>,
        position: [f64; 3],
    ) -> (SdtmPoint, DriverState) {
        let (latitude, altitude) =
            geodetic_latitude_altitude(libm::hypot(position[0], position[1]), position[2]);
        let (day, second) = a.history.day(a.epoch);
        let east = libm::atan2(position[1], position[0]) - earth_rotation_angle(a.epoch);
        let hours = second / 3_600.0 + east.to_degrees() / 15.0;
        (
            SdtmPoint {
                altitude,
                latitude,
                local_solar_time: hours - 24.0 * libm::floor(hours / 24.0),
                day_of_year: day.ordinal + second / SECONDS_PER_DAY,
            },
            a.history.state(a.epoch),
        )
    }

    #[test]
    fn the_next_jump_is_the_next_line_of_the_three_hour_grid() {
        // The epoch is 01:00, so the first line is two hours on and the next three after it.
        let a = atmosphere(false);
        assert_eq!(a.next_jump(0.0), 7_200.0);
        assert_eq!(a.next_jump(3_600.0), 7_200.0);
        assert_eq!(a.next_jump(7_200.0), 18_000.0);
        assert_eq!(a.next_jump(30_000.0), 39_600.0);
        // A sum `epoch + t` that rounded to just short of a line is on it; a real gap is not.
        assert_eq!(a.next_jump(7_200.0 - 0.1 * GRID_SLACK), 18_000.0);
        assert_eq!(a.next_jump(7_200.0 - 10.0 * GRID_SLACK), 7_200.0);
    }

    #[test]
    fn it_asks_for_the_tolerance_it_was_given() {
        let tolerance = Tolerance::new(3.0e-7, 5.0e-8);
        let a = ScenarioAtmosphere {
            tolerance,
            ..atmosphere(false)
        };
        assert_eq!(a.tolerance().atol.to_bits(), tolerance.atol.to_bits());
        assert_eq!(a.tolerance().rtol.to_bits(), tolerance.rtol.to_bits());
    }

    /// A segment that starts a minute before midnight and is read five minutes on: the drivers and
    /// the day are the segment start's, the clock and the position are `t`'s, and the clock runs on
    /// past 86 400 s rather than jumping back inside the segment.
    #[test]
    fn the_drivers_are_held_at_the_segment_start_and_the_rest_is_read_at_t() {
        let a = atmosphere(false);
        let segment_start = 86_400.0 - 60.0 - 3_600.0;
        let t = segment_start + 300.0;
        let now = a.epoch + t;
        let angle = earth_rotation_angle(now) + 0.5;
        let position = [6.75e6 * libm::cos(angle), 6.75e6 * libm::sin(angle), 2.0e5];
        let (latitude, altitude) =
            geodetic_latitude_altitude(libm::hypot(position[0], position[1]), position[2]);
        let held = a.epoch + segment_start + GRID_SLACK;
        let (day, _) = a.history.day(held);
        let second = now - libm::floor(held / SECONDS_PER_DAY) * SECONDS_PER_DAY;
        assert!(second > SECONDS_PER_DAY, "{second}");
        let state = a
            .scenario
            .state(a.history, a.epoch, segment_start + GRID_SLACK);
        let east = libm::atan2(position[1], position[0]) - earth_rotation_angle(now);
        let east = east - core::f64::consts::TAU * libm::floor(east / core::f64::consts::TAU);
        assert!((east - 0.5).abs() < 1.0e-9, "{east}");
        let hours = second / 3_600.0 + east.to_degrees() / 15.0;
        let local = hours - 24.0 * libm::floor(hours / 24.0);
        assert!((0.0..24.0).contains(&local), "{local}");
        let input = Nrlmsise00Input {
            year: 0,
            doy: day.ordinal as i32,
            sec: second,
            alt: altitude / 1_000.0,
            g_lat: latitude.to_degrees(),
            g_long: east.to_degrees(),
            lst: local,
            f107a: state.f107a_centred,
            f107: state.f107,
            ap: state.ap,
            ap_array: None,
        };
        let want = gtd7d(&input, &Nrlmsise00Flags::default()).d[5] * 1_000.0;
        let got = a.density_held(segment_start, t, position);
        assert!((got / want - 1.0).abs() < 1.0e-12, "{got} {want}");
        // Past the jump, the held read and the read at `t` differ: the next day's flux has arrived.
        assert_ne!(got, a.density(t, position));
    }

    /// A segment that starts exactly on a day's first second reads that day, not the one before it,
    /// and the same is true of a line of the three-hour grid.
    #[test]
    fn a_segment_starting_on_a_line_of_the_grid_reads_the_far_side_of_it() {
        // The field: its storm state is what moves at a line inside the day.
        let a = atmosphere(true);
        let position = [6.75e6, 1.0e5, 2.0e5];
        for line in [23.0 * 3_600.0, 2.0 * 3_600.0] {
            let on = a.density_held(line, line, position);
            assert_eq!(on, a.density_held(line + 1.0, line, position));
            assert_ne!(on, a.density_held(line - 1.0, line, position));
        }
    }

    /// `hours` runs from −12 to 36, and each side of the wrap has a test of its own: this is the middle.
    #[test]
    fn local_time_inside_the_day_is_the_hours_as_they_stand() {
        let a = atmosphere(true);
        let angle = earth_rotation_angle(a.epoch) + 0.5;
        let position = [6.75e6 * libm::cos(angle), 6.75e6 * libm::sin(angle), 0.0];
        let (point, state) = sample(&a, position);
        // 01:00 and half a radian east, so the hours need no wrap.
        let want = 1.0 + 0.5_f64.to_degrees() / 15.0;
        assert!((point.local_solar_time - want).abs() < 1.0e-9);
        let drivers = SdtmDrivers {
            f107: state.f107,
            f107a: state.f107a_trailing,
            storm: state.storm,
        };
        assert_eq!(a.density(0.0, position), FIELD.density(&point, &drivers));
    }

    #[test]
    fn a_day_knows_its_ordinal_in_the_year() {
        let h = history();
        for (date, want) in [(9, 9.0), (20, 20.0), (28, 28.0)] {
            let (day, _) = h.day(days_from_civil(2001, 1, date) as f64 * SECONDS_PER_DAY);
            assert_eq!(day.ordinal, want);
        }
    }

    #[test]
    fn inside_the_envelope_it_is_the_field_and_outside_it_is_nrlmsise() {
        let a = atmosphere(true);
        let inside = [6.75e6, 1.0e5, 2.0e5];
        let (point, state) = sample(&a, inside);
        assert!(ENVELOPE.altitude.contains(&point.altitude));
        let drivers = SdtmDrivers {
            f107: state.f107,
            f107a: state.f107a_trailing,
            storm: state.storm,
        };
        assert_eq!(a.density(0.0, inside), FIELD.density(&point, &drivers));

        let below = [6.50e6, 1.0e5, 2.0e5];
        let (point, _) = sample(&a, below);
        assert!(point.altitude < *ENVELOPE.altitude.start());
        let alone = atmosphere(false);
        assert_eq!(a.density(0.0, below), alone.density(0.0, below));
        assert_ne!(alone.density(0.0, inside), a.density(0.0, inside));
    }

    #[test]
    fn a_decay_reads_the_field_down_to_its_boundary_and_no_further() {
        static DECAY: Envelope = decay_envelope(&ENVELOPE);
        assert_eq!(*DECAY.altitude.start(), BOUNDARY_ALTITUDE);
        assert_eq!(DECAY.altitude.end(), ENVELOPE.altitude.end());
        assert_eq!(DECAY.f107a, ENVELOPE.f107a);

        let fitted = atmosphere(true);
        let a = ScenarioAtmosphere {
            field: Some((&FIELD, &DECAY)),
            ..atmosphere(true)
        };
        let alone = atmosphere(false);
        let at = |height: f64| [WGS84_A + height, 0.0, 0.0];
        // Under the fitted limit: the field, where the fitted envelope gives NRLMSISE-00.
        let under = at(*ENVELOPE.altitude.start() - 40_000.0);
        let (point, state) = sample(&a, under);
        let drivers = SdtmDrivers {
            f107: state.f107,
            f107a: state.f107a_trailing,
            storm: state.storm,
        };
        assert_eq!(a.density(0.0, under), FIELD.density(&point, &drivers));
        assert_eq!(fitted.density(0.0, under), alone.density(0.0, under));
        assert_ne!(a.density(0.0, under), alone.density(0.0, under));
        // Under the boundary: NRLMSISE-00. Over the fitted limit: NRLMSISE-00.
        for height in [BOUNDARY_ALTITUDE - 1.0, *ENVELOPE.altitude.end() + 1.0] {
            assert_eq!(a.density(0.0, at(height)), alone.density(0.0, at(height)));
        }
    }

    #[test]
    fn nrlmsise_alone_reads_the_centred_flux_and_the_daily_ap() {
        let a = atmosphere(false);
        let position = [6.75e6, 1.0e5, 2.0e5];
        let (point, state) = sample(&a, position);
        let (day, second) = a.history.day(a.epoch);
        let east = libm::atan2(position[1], position[0]) - earth_rotation_angle(a.epoch);
        let wrapped = east - core::f64::consts::TAU * libm::floor(east / core::f64::consts::TAU);
        let input = Nrlmsise00Input {
            year: 0,
            doy: day.ordinal as i32,
            sec: second,
            alt: point.altitude / 1_000.0,
            g_lat: point.latitude.to_degrees(),
            g_long: wrapped.to_degrees(),
            lst: point.local_solar_time,
            f107a: state.f107a_centred,
            f107: state.f107,
            ap: state.ap,
            ap_array: None,
        };
        let want = gtd7d(&input, &Nrlmsise00Flags::default()).d[5] * 1_000.0;
        assert_eq!(a.density(0.0, position), want);
    }

    #[test]
    fn the_field_blends_into_nrlmsise_inside_the_height_limits_and_nowhere_else() {
        let (with, without) = (atmosphere(true), atmosphere(false));
        let (low, high) = (*ENVELOPE.altitude.start(), *ENVELOPE.altitude.end());
        assert_eq!(field_weight(&ENVELOPE, low), 0.0);
        assert_eq!(field_weight(&ENVELOPE, high), 0.0);
        assert_eq!(field_weight(&ENVELOPE, low + ENVELOPE_BLEND), 1.0);
        assert_eq!(field_weight(&ENVELOPE, high - ENVELOPE_BLEND), 1.0);
        assert_eq!(field_weight(&ENVELOPE, low + 0.5 * ENVELOPE_BLEND), 0.5);
        assert_eq!(field_weight(&ENVELOPE, high - 0.5 * ENVELOPE_BLEND), 0.5);
        // The polynomial's ends are flat to second order: 6x⁵ − 15x⁴ + 10x³ near x = 0.
        let small = 1.0e-3 * ENVELOPE_BLEND;
        assert!(field_weight(&ENVELOPE, low + small) < 1.1e-8);
        // Continuous across the limit: the blended density meets NRLMSISE-00 from inside.
        let on = |a: &ScenarioAtmosphere<'static, HARMONICS>, height: f64| {
            a.density(0.0, [WGS84_A + height, 0.0, 0.0])
        };
        let (inside, outside) = (on(&with, low + 1.0), on(&without, low + 1.0));
        assert!(
            (inside / outside - 1.0).abs() < 1.0e-9,
            "{inside} {outside}"
        );
        // Halfway, the log density is the mean of the two models' at that height.
        let middle = low + 0.5 * ENVELOPE_BLEND;
        let (point, state) = sample(&with, [WGS84_A + middle, 0.0, 0.0]);
        let field = FIELD.density(
            &point,
            &SdtmDrivers {
                f107: state.f107,
                f107a: state.f107a_trailing,
                storm: state.storm,
            },
        );
        let expected = (0.5 * (field.ln() + on(&without, middle).ln())).exp();
        let blended = on(&with, middle);
        assert_ne!(field, on(&without, middle));
        assert!(
            (blended / expected - 1.0).abs() < 1.0e-12,
            "{blended} {expected}"
        );
    }

    #[test]
    fn the_density_stops_rising_below_its_floor_and_not_above_it() {
        let a = atmosphere(false);
        let at = |height: f64| a.density(1_000.0, [WGS84_A + height, 0.0, 0.0]);
        assert_eq!(at(60_000.0), at(DENSITY_FLOOR_HEIGHT));
        assert!(at(DENSITY_FLOOR_HEIGHT + 5_000.0) < at(DENSITY_FLOOR_HEIGHT));
        assert!(DENSITY_FLOOR_HEIGHT < crate::lifetime::DEFAULT_REENTRY_ALTITUDE);
    }

    #[test]
    fn the_atmosphere_co_rotates_at_the_iers_rate() {
        let a = atmosphere(false);
        assert_eq!(
            a.velocity([7.0e6, 1.0e6, 2.0e6]),
            [-EARTH_ROTATION * 1.0e6, EARTH_ROTATION * 7.0e6, 0.0]
        );
    }

    #[test]
    fn the_conformal_radius_covers_exactly_k_over_n_plus_one() {
        // Twenty distinct scores: every one held out in turn is the finite-sample law, exactly.
        let scores: Vec<f64> = (0..20).map(|i| 0.1 + f64::from((i * 7) % 20)).collect();
        for alpha in [0.1_f64, 0.2, 0.31, 0.5] {
            let n = scores.len() - 1;
            let k = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
            let covered = (0..scores.len())
                .filter(|&i| {
                    let mut rest = scores.clone();
                    let held = rest.remove(i);
                    held <= conformal_radius(&rest, alpha).unwrap()
                })
                .count();
            assert_eq!(covered, k, "alpha {alpha}");
            let coverage = k as f64 / (n as f64 + 1.0);
            assert!(coverage >= 1.0 - alpha && coverage < 1.0 - alpha + 1.0 / (n as f64 + 1.0));
        }
    }

    #[test]
    fn the_conformal_radius_is_the_kth_smallest_and_refuses_what_has_no_coverage() {
        // The last score is an answer, not a refusal: k = n.
        assert_eq!(conformal_radius(&[3.0], 0.5), Some(3.0));
        for alpha in [0.0, 1.0, -1.0, 2.0, f64::NAN, f64::INFINITY] {
            assert_eq!(conformal_radius(&[1.0, 2.0, 3.0], alpha), None, "{alpha}");
        }
        let scores = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(conformal_radius(&scores, 0.5), Some(3.0));
        assert_eq!(conformal_radius(&scores, 0.34), Some(4.0));
        // (5 + 1)(1 - 0.1) = 5.4 rounds up to 6, more than there are.
        assert_eq!(conformal_radius(&scores, 0.1), None);
        assert_eq!(conformal_radius(&scores, 0.0), None);
        assert_eq!(conformal_radius(&scores, 1.0), None);
        assert_eq!(conformal_radius(&scores, f64::NAN), None);
        assert_eq!(conformal_radius(&[1.0, f64::NAN], 0.5), None);
        assert_eq!(conformal_radius(&[], 0.5), None);
    }
}
