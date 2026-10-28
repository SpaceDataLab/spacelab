//! Space-weather index feed — daily F10.7 solar flux and Ap geomagnetic index, the drivers every
//! empirical thermosphere model (NRLMSISE-00 / JB2008 / DTM2020) and the [`crate::density_rom`]
//! reduced dynamics need.
//!
//! Parses the CelesTrak `SW-All.csv` product (<https://celestrak.org/SpaceData/>, keyless, freely
//! redistributable; underlying data from NOAA/SWPC and GFZ Potsdam). One comma-separated row per UTC
//! day; the columns used here are `DATE` (0), the eight 3-hourly `KP1..KP8` (3–10) and `AP1..AP8`
//! (12–19), `AP_AVG` (20), `F10.7_OBS` (24), `F10.7_ADJ` (25), `F10.7_DATA_TYPE` (26),
//! `F10.7_OBS_CENTER81` (27) and `F10.7_OBS_LAST81` (28). The bulk file is an external download and
//! is not vendored — only a small sample is kept as a fixture.
//!
//! **The file is not all measurements.** It runs decades past the present on NOAA/NASA forecasts, and
//! the only thing separating a measured day from a forecast one is `F10.7_DATA_TYPE` — 25 089 `OBS`
//! against 60 `INT`, 45 `PRD` and 182 `PRM` in the 2026-08 snapshot, with every non-observation in
//! the forward tail out to 2041. A caller propagating into next decade is entirely inside forecast
//! territory, so [`SpaceWeatherDay::quality`] is part of the reading, not metadata about it.

// `d`/`day` and the product/agency names (CelesTrak, NOAA, SWPC, GFZ) are the domain vocabulary; the
// ordinal/subinterval arithmetic in the ap-array builder casts small non-negative integers.
#![allow(
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use alloc::vec::Vec;

/// Where a day's F10.7 came from — the `F10.7_DATA_TYPE` column.
///
/// A model cannot tell these apart from the flux value, and the difference is not a detail: a
/// `Predicted` day carries the NOAA forecast's uncertainty, and a `PredictedMean` day is a
/// solar-cycle *climatology* with no daily structure at all, so a storm-time density computed on one
/// is a statement about the average cycle rather than about that day.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxQuality {
    /// `OBS` — measured at Penticton.
    Observed,
    /// `INT` — interpolated across a gap in the observed series.
    Interpolated,
    /// `PRD` — NOAA/SWPC short-term daily forecast.
    Predicted,
    /// `PRM` — monthly predicted mean, the long-range solar-cycle projection.
    PredictedMean,
    /// Synthesised by [`extend_forecast`] — one of this module's own baselines, not a provider's
    /// forecast. Kept distinct from [`Self::Predicted`] because the two carry different errors and
    /// only one of them is NOAA's: `PRD` is a physical forecast, this is a persistence or recurrence
    /// extrapolation whose skill by lead time is measured in `docs/verification.md` §10.13.
    Extrapolated,
    /// Anything else, including an empty cell. Not silently promoted to `Observed`: a value whose
    /// provenance the file declines to state is exactly the case this type exists to keep visible.
    Unknown,
}

impl FluxQuality {
    /// Whether this day's flux was measured rather than filled in or forecast. `Interpolated` is
    /// **not** measured: it is a value the series did not have.
    #[must_use]
    pub fn is_measured(self) -> bool {
        self == Self::Observed
    }

    fn parse(field: &str) -> Self {
        match field.trim() {
            "OBS" => Self::Observed,
            "INT" => Self::Interpolated,
            "PRD" => Self::Predicted,
            "PRM" => Self::PredictedMean,
            _ => Self::Unknown,
        }
    }
}

/// One UTC day of space-weather indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpaceWeatherDay {
    /// Calendar year.
    pub year: i32,
    /// Calendar month, `1..=12`.
    pub month: u32,
    /// Day of month, `1..=31`.
    pub day: u32,
    /// Daily average geomagnetic `Ap` index.
    pub ap: f64,
    /// The eight 3-hourly `Ap` values of the day (`AP1..AP8`, covering `00–03, 03–06, …, 21–24` UT).
    /// The storm-time driver [`nrlmsise_ap_array`] needs this sub-daily resolution; the daily average
    /// [`Self::ap`] smooths the sharp density response to a geomagnetic storm.
    pub ap3h: [f64; 8],
    /// The eight 3-hourly `Kp` values of the day (`KP1..KP8`, same intervals as [`Self::ap3h`]),
    /// on the quasi-logarithmic 0–9 scale in thirds.
    ///
    /// `Kp` and `ap` carry the same measurement, but the map between them is a quantised table, so
    /// a model written against one cannot be driven from the other without loss: DTM2020 takes
    /// `Kp`, NRLMSISE-00 takes `ap`, and the file publishes both.
    pub kp3h: [f64; 8],
    /// Observed F10.7 solar radio flux (10⁻²² W·m⁻²·Hz⁻¹), at the Earth's actual distance from the
    /// Sun.
    ///
    /// This is what NRLMSISE-00 asks for, in as many words: *"F107 and F107A values used to generate
    /// the model correspond to the 10.7 cm radio flux at the actual distance of the Earth from the
    /// Sun rather than the radio flux at 1 AU."* DTM2020's documentation does not say, and JB2008
    /// does not read this file at all — its `F10` comes from the SOLFSMY product.
    pub f107_observed: f64,
    /// F10.7 adjusted to 1 AU — the same measurement with the annual 1/r² modulation removed.
    ///
    /// Roughly ±3.4% away from [`Self::f107_observed`] over a year, which is why the choice is a
    /// model-specification question and not a preference. Nothing here defaults to it.
    pub f107_adjusted: f64,
    /// 81-day *centred* average of the observed F10.7.
    ///
    /// The quantity NRLMSISE-00 specifies — `F107A - 81 day AVERAGE OF F10.7 FLUX (centered on day
    /// DDD)` — and **not** the one DTM2020 does. Past the last observed day it cannot be an
    /// observation: a centred mean needs 40 days on each side, so within 40 days of the present it is
    /// half forecast and beyond that wholly one. [`Self::quality`] says which.
    pub f107_center81: f64,
    /// 81-day *trailing* average of the observed F10.7 — the same window ending today rather than
    /// centred on it.
    ///
    /// The quantity DTM2020 specifies: its reference subroutine's header reads `fbar(1)=mean flux of
    /// last 81 days at t`. It is not a worse version of [`Self::f107_center81`] — it is a different
    /// driver, and separately it is the one that stays an observation at the end of the series. Both
    /// are kept because the file publishes both and the models disagree about which they want.
    pub f107_last81: f64,
    /// Whether this day's flux was measured, interpolated or forecast.
    pub quality: FluxQuality,
}

/// Parse a CelesTrak `SW-All.csv` file. The header line and malformed rows are skipped.
#[must_use]
pub fn parse_space_weather(csv: &str) -> Vec<SpaceWeatherDay> {
    let mut out = Vec::new();
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("DATE") {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        // Through `F10.7_OBS_LAST81` (28), the last column this reader needs.
        if f.len() < 29 {
            continue;
        }
        let d: Vec<&str> = f[0].split('-').collect();
        if d.len() != 3 {
            continue;
        }
        let (Ok(year), Ok(month), Ok(day)) = (d[0].parse(), d[1].parse(), d[2].parse()) else {
            continue;
        };
        let num = |s: &str| s.trim().parse::<f64>().ok();
        let (Some(ap), Some(f107)) = (num(f[20]), num(f[24])) else {
            continue;
        };
        // AP1..AP8 at columns 12–19; a blank sub-daily cell falls back to the daily average.
        let mut ap3h = [ap; 8];
        for (i, slot) in ap3h.iter_mut().enumerate() {
            if let Some(v) = num(f[12 + i]) {
                *slot = v;
            }
        }
        // KP1..KP8 at columns 3–10, published as Kp×10 (`43` is Kp 4+, i.e. 4.3). A blank cell
        // leaves the quietest value the scale has rather than a fabricated one.
        let mut kp3h = [0.0; 8];
        for (i, slot) in kp3h.iter_mut().enumerate() {
            if let Some(v) = num(f[3 + i]) {
                *slot = v / 10.0;
            }
        }
        // The two 81-day means back each other up, and neither backs up to the daily value: an
        // 81-day average that has quietly become a single day's flux is a driver an order of
        // magnitude more variable than the one the model was fitted against, and nothing downstream
        // can see the substitution. A row with neither is dropped instead.
        let (center81, last81) = (num(f[27]), num(f[28]));
        let (Some(center81), Some(last81)) = (center81.or(last81), last81.or(center81)) else {
            continue;
        };
        out.push(SpaceWeatherDay {
            year,
            month,
            day,
            ap,
            ap3h,
            kp3h,
            f107_observed: f107,
            f107_adjusted: num(f[25]).unwrap_or(f107),
            f107_center81: center81,
            f107_last81: last81,
            quality: FluxQuality::parse(f[26]),
        });
    }
    out
}

/// The indices for a given calendar day, if present.
#[must_use]
pub fn find(days: &[SpaceWeatherDay], year: i32, month: u32, day: u32) -> Option<SpaceWeatherDay> {
    days.iter()
        .find(|d| d.year == year && d.month == month && d.day == day)
        .copied()
}

/// Proleptic-Gregorian day number (days since 1970-01-01), Howard Hinnant's `days_from_civil`. Exact
/// integer arithmetic, so consecutive calendar days differ by exactly 1 across month and leap-year
/// boundaries — the key the storm-array look-back is indexed by, robust to gaps in the feed.
const fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = year as i64 - if month <= 2 { 1 } else { 0 };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (if month > 2 { month - 3 } else { month + 9 }) as i64; // Mar=0 … Feb=11
    let doy = (153 * mp + 2) / 5 + day as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Cadence of the feed's sub-daily indices, seconds — three hours, eight intervals to the day.
///
/// Public because it is the `cadence` argument [`crate::sdtm::StormKernel::response`] must be given
/// for the history [`sdtm_ap_history`] returns: the kernel integrates a step function, and a cadence
/// that does not match the sampling rescales the storm response rather than failing.
pub const THREE_HOURLY_CADENCE: f64 = 10_800.0;

/// The inverse of [`days_from_civil`], Hinnant's `civil_from_days` — the calendar date a synthesised
/// forecast row has to carry. Exact integer arithmetic in both directions, so the round trip is the
/// identity and `tests` checks that it is.
const fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let shifted = z + 719_468;
    let era = (if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    }) / 146_097;
    let doe = shifted - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], March-based
    let mp = (5 * doy + 2) / 153; // [0, 11], Mar=0 … Feb=11
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

/// One of a day's eight 3-hourly indices, addressed by a global interval index
/// (day ordinal × 8 + subinterval) so that a look-back crosses month and year boundaries by
/// integer arithmetic. `days` must be ascending by date. `None` when the day is not in the feed.
fn three_hourly(
    days: &[SpaceWeatherDay],
    global: i64,
    pick: impl Fn(&SpaceWeatherDay) -> &[f64; 8],
) -> Option<f64> {
    let ordinal = global.div_euclid(8);
    let sub = global.rem_euclid(8) as usize;
    days.binary_search_by(|d| days_from_civil(d.year, d.month, d.day).cmp(&ordinal))
        .ok()
        .map(|i| pick(&days[i])[sub])
}

/// The seven-element NRLMSISE-00 storm-time `ap_array` (the [`SpaceWeatherDay::ap3h`] history that
/// feeds [`crate::Nrlmsise00Input::ap_array`]) for the instant `sec` UT seconds into the day
/// `year-month-day`, following the model's definition:
///
/// * slot 0 — daily `Ap`;
/// * slot 1 — the 3-hourly `ap` of the current interval;
/// * slots 2–4 — the 3-hourly `ap` 3, 6 and 9 hours earlier;
/// * slot 5 — the average of the eight 3-hourly `ap` from 12 to 33 hours before;
/// * slot 6 — the average of the eight 3-hourly `ap` from 36 to 57 hours before.
///
/// Resolving the storm history at 3-hour cadence, rather than replaying one daily `Ap` into every
/// slot, is what lets the model track the sharp density surge and decay of a geomagnetic storm.
///
/// `days` must be ascending by date (as [`parse_space_weather`] returns). Returns `None` if the
/// target day or any required historical interval (up to ~2.5 days back) is missing from `days`.
#[must_use]
pub fn nrlmsise_ap_array(
    days: &[SpaceWeatherDay],
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
) -> Option<[f64; 7]> {
    let target = find(days, year, month, day)?;
    let subinterval = if sec.is_finite() {
        (sec / 10_800.0).floor().clamp(0.0, 7.0) as i64
    } else {
        0
    };
    let current = days_from_civil(year, month, day) * 8 + subinterval;

    let at = |global: i64| three_hourly(days, global, |d| &d.ap3h);
    let mean8 = |first_back: i64| -> Option<f64> {
        let mut sum = 0.0;
        for k in first_back..first_back + 8 {
            sum += at(current - k)?;
        }
        Some(sum / 8.0)
    };

    Some([
        target.ap,
        at(current)?,
        at(current - 1)?,
        at(current - 2)?,
        at(current - 3)?,
        mean8(4)?,  // 12–33 h before → intervals 4..=11 back
        mean8(12)?, // 36–57 h before → intervals 12..=19 back
    ])
}

/// The 3-hourly `ap` history SDTM-1's storm kernel integrates, oldest first, ending at the interval
/// **before** the one containing `sec` UT seconds into `year-month-day`.
///
/// The exclusion of the sample's own interval is causality, not caution: `ap` for the interval a
/// sample falls in is not published at the sample's epoch, so a model that used it could not be run
/// forward. It is also what `tools/gen_sdtm_coefficients.py` fitted the kernel on, which is what makes
/// the two the same model.
///
/// `N` is how far back the recursion is warmed up. The kernel forgets its initial state as
/// `exp(-3 h/τ_r)^N`, so with the fitted `τ_r` under 2 h even `N = 8` leaves a part in a million;
/// [`crate::sdtm::coefficients`] is evaluated with 16. Returns `None` if any of those intervals is
/// missing from the feed, because a shortened history reads as a quieter storm.
#[must_use]
pub fn sdtm_ap_history<const N: usize>(
    days: &[SpaceWeatherDay],
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
) -> Option<[f64; N]> {
    let subinterval = if sec.is_finite() {
        (sec / THREE_HOURLY_CADENCE).floor().clamp(0.0, 7.0) as i64
    } else {
        0
    };
    let previous = days_from_civil(year, month, day) * 8 + subinterval - 1;
    let mut out = [0.0; N];
    for (k, slot) in out.iter_mut().enumerate() {
        let back = i64::try_from(N - 1 - k).ok()?;
        *slot = three_hourly(days, previous - back, |d| &d.ap3h)?;
    }
    Some(out)
}

/// The three SDTM-1 drivers at the instant `sec` UT seconds into `year-month-day`, with the model's
/// own index conventions applied: F10.7 of the **previous** day, the **trailing** 81-day mean of the
/// same day, and the storm kernel's response over the `ap` history ending at the interval before the
/// sample.
///
/// The conventions belong here rather than at the call site for the same reason the JB2008 drivers'
/// lags do: each of the three is part of the model definition, and every one of them has a plausible
/// wrong answer in the same file. The centred 81-day mean is the quantity NRLMSISE-00 asks for and it
/// is *not* this model's — `tools/gen_sdtm_coefficients.py` fitted the trailing one, because it is the
/// only one an operational caller has at the end of the series. Sampling F10.7 on the sample's own day
/// would read as spuriously good on a flare day, and giving the kernel a cadence other than
/// [`THREE_HOURLY_CADENCE`] rescales the storm term silently.
///
/// `N` is the kernel's warm-up length — see [`sdtm_ap_history`], which is where the `None` cases are:
/// the previous day, the sample's day or any of the `N` history intervals missing from the feed.
#[must_use]
pub fn sdtm_drivers<const N: usize>(
    days: &[SpaceWeatherDay],
    kernel: &crate::sdtm::StormKernel,
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
) -> Option<crate::sdtm::SdtmDrivers> {
    let yesterday = days_from_civil(year, month, day) - 1;
    let previous = days
        .binary_search_by(|d| days_from_civil(d.year, d.month, d.day).cmp(&yesterday))
        .ok()
        .map(|i| days[i])?;
    let today = find(days, year, month, day)?;
    let history = sdtm_ap_history::<N>(days, year, month, day, sec)?;
    Some(crate::sdtm::SdtmDrivers {
        f107: previous.f107_observed,
        f107a: today.f107_last81,
        storm: kernel.response(&history, THREE_HOURLY_CADENCE),
    })
}

/// Length of the 81-day mean window the flux drivers are averaged over.
pub const FLUX_WINDOW: i64 = 81;

/// Lag of the recurrence forecast, days — one synodic solar rotation as seen from the Earth.
///
/// The rotation is 27.2753 days, so an integer lag is a choice: 27 runs a quarter day early per
/// rotation and 28 three quarters late. 27 is the operational convention (NOAA's own 27-day outlook)
/// and the one this module's skill measurement was run on.
pub const SOLAR_ROTATION: i64 = 27;

/// How a feed is carried past its last measurement.
///
/// Both are *baselines* in the forecasting sense — the thing a real forecast has to beat, not a
/// forecast itself. Which of the two wins depends on the lead time, and the crossover is measured
/// rather than argued (`docs/verification.md` §10.13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForecastMethod {
    /// Hold the last measured day's flux and `ap` intervals. Exact at lead zero and degrades as the
    /// Sun rotates: it cannot know that an active region is about to come back over the limb.
    Persistence,
    /// Copy the day one solar rotation earlier — [`SOLAR_ROTATION`] days back in the series being
    /// built, which past 27 days of lead is itself a copy. Long-lived active regions and coronal
    /// holes recur, so this is the standard cheap skill at multi-week leads.
    Recurrence,
}

/// Carry a feed past its last measured day with one of the [`ForecastMethod`] baselines, through the
/// UTC day `year-month-day` inclusive.
///
/// **The provider's own forward tail is discarded.** CelesTrak publishes NOAA `PRD` and `PRM` rows out
/// to 2041, and a baseline built on top of those would be a forecast of a forecast — so the returned
/// series is the measured prefix (through the last [`FluxQuality::is_measured`] day) followed by
/// synthesised rows carrying [`FluxQuality::Extrapolated`]. A caller who wants NOAA's forecast should
/// use the file as parsed and read [`SpaceWeatherDay::quality`].
///
/// **The 81-day means are recomputed, not carried.** Holding the last measured day's `F10.7_OBS_LAST81`
/// forward is the mistake this function exists to prevent: the window keeps rolling whatever the daily
/// value does, so a trailing mean under persistence *still moves* — towards the persisted value —
/// until the whole window is synthetic. Both means are averaged over the extended daily series, and
/// the centred one is exact through the last returned day because the fluxes are synthesised forty
/// days further and then dropped.
///
/// Returns `None` if `days` is empty, has no measured day, is not ascending and contiguous over the
/// [`FLUX_WINDOW`] days ending at the last measured one, or if that day is already at or past
/// `year-month-day` — each of which would make a mean this function computes an average of a window it
/// does not have.
#[must_use]
pub fn extend_forecast(
    days: &[SpaceWeatherDay],
    method: ForecastMethod,
    year: i32,
    month: u32,
    day: u32,
) -> Option<Vec<SpaceWeatherDay>> {
    let last = days.iter().rposition(|d| d.quality.is_measured())?;
    let origin = days_from_civil(days[last].year, days[last].month, days[last].day);
    let through = days_from_civil(year, month, day);
    if through <= origin {
        return None;
    }
    // The window the first synthesised day's trailing mean averages over is the last FLUX_WINDOW - 1
    // measured days plus itself, so those days have to be there and consecutive.
    let first = last.checked_sub(usize::try_from(FLUX_WINDOW).ok()? - 1)?;
    if days[first..=last]
        .iter()
        .zip(origin - (FLUX_WINDOW - 1)..=origin)
        .any(|(d, ordinal)| days_from_civil(d.year, d.month, d.day) != ordinal)
    {
        return None;
    }

    // Synthesised forty days past `through` so that every returned day's centred mean is an average of
    // eighty-one values rather than of the ones that happen to exist.
    let horizon = through + FLUX_WINDOW / 2;
    let mut out = days[..=last].to_vec();
    for ordinal in origin + 1..=horizon {
        let source = match method {
            ForecastMethod::Persistence => out[last],
            ForecastMethod::Recurrence => out[out.len() - usize::try_from(SOLAR_ROTATION).ok()?],
        };
        let (y, m, d) = civil_from_days(ordinal);
        out.push(SpaceWeatherDay {
            year: y,
            month: m,
            day: d,
            quality: FluxQuality::Extrapolated,
            ..source
        });
    }
    // Both means over the extended dailies. Done in one pass after the fact because the centred window
    // of a synthesised day reaches into days that did not exist when it was pushed.
    let flux: Vec<f64> = out.iter().map(|d| d.f107_observed).collect();
    let half = usize::try_from(FLUX_WINDOW / 2).ok()?;
    let width = usize::try_from(FLUX_WINDOW).ok()?;
    // Counted in `f64` rather than cast from the window length, as `sdtm::DensityUncertainty` does:
    // the count is an exact small integer either way and the cast is one the lint is right to flag.
    let mean = |end: usize| -> Option<f64> {
        let start = (end + 1).checked_sub(width)?;
        let (mut sum, mut n) = (0.0_f64, 0.0_f64);
        for x in flux.get(start..=end)? {
            sum += *x;
            n += 1.0;
        }
        Some(sum / n)
    };
    for i in out.len() - usize::try_from(horizon - origin).ok()?..out.len() {
        if let Some(trailing) = mean(i) {
            out[i].f107_last81 = trailing;
        }
        if let Some(centred) = i.checked_add(half).and_then(mean) {
            out[i].f107_center81 = centred;
        }
    }
    out.truncate(out.len() - usize::try_from(horizon - through).ok()?);
    Some(out)
}

/// One interval of a GFZ Hpo-family series — `Hp30`/`ap30` or `Hp60`/`ap60`.
///
/// The Hpo indices are Kp's construction run at 30- and 60-minute cadence instead of three hours,
/// from the same 13 observatories. Two properties make them a different measurement rather than a
/// finer sampling of the same one, and both are visible in the May 2024 storm this module's fixture
/// carries:
///
/// * **They resolve the onset.** Kp assigns one number to 15:00–18:00 UT; `ap60` assigns three, and
///   on 2024-05-10 they are 80, 94 and 294. Any single value over that window is wrong by a factor
///   of three at one end or the other, and a thermospheric model driven by it heats too early and
///   then not enough.
/// * **They are not capped.** Kp saturates at 9, `ap` at 400, and the Gannon storm reached both --
///   so the published 3-hourly series says "as disturbed as the scale can express" and stops.
///   `Hp60` reads 10.667 and `ap60` 456 in the same hours, which is the only way the record can say
///   *how far past* the ceiling the storm went.
///
/// Held as the file publishes it: a calendar date and the interval's start hour, rather than a
/// single instant, because the index *is* an interval average and collapsing it to its midpoint
/// would silently move a driver by half a cadence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HpoInterval {
    /// Calendar year, UT.
    pub year: i32,
    /// Calendar month, `1..=12`.
    pub month: u32,
    /// Day of month, `1..=31`.
    pub day: u32,
    /// Start of the interval in hours UT: `0.0, 0.5, 1.0, …` for Hp30, `0.0, 1.0, …` for Hp60.
    pub start_hour: f64,
    /// The quasi-logarithmic index, in thirds. **Not capped at 9** -- see the type's documentation.
    pub hp: f64,
    /// The linear equivalent, on the same table Kp↔ap uses below 9 and extended above it.
    pub ap: f64,
}

/// Parse a GFZ `Hp30_ap30_*.txt` or `Hp60_ap60_*.txt` file.
///
/// Comment lines (`#`) and malformed rows are skipped, and so are the intervals the file marks
/// absent -- `-1.000` for `Hp`, `-1` for `ap`. Skipping rather than substituting is the point: a
/// missing geomagnetic index is not a quiet one, and the lookup below returning `None` is what a
/// caller has to handle. The alternative, filling with zero, reads downstream as the quietest hour
/// the scale has.
///
/// The file's own `days since 1932-01-01` columns are ignored in favour of re-deriving the ordinal
/// from the calendar date, so this shares `days_from_civil` with every other series here and
/// cannot disagree with them about what a day is.
#[must_use]
pub fn parse_hpo(text: &str) -> Vec<HpoInterval> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        // `YYYY MM DD hh.h hh._m days days_m Hp ap D` — nine fields, and the reader needs through
        // the eighth.
        if f.len() < 9 {
            continue;
        }
        let (Ok(year), Ok(month), Ok(day)) = (f[0].parse(), f[1].parse(), f[2].parse()) else {
            continue;
        };
        let (Ok(start_hour), Ok(hp), Ok(ap)) = (
            f[3].parse::<f64>(),
            f[7].parse::<f64>(),
            f[8].parse::<f64>(),
        ) else {
            continue;
        };
        if hp < 0.0 || ap < 0.0 {
            continue;
        }
        out.push(HpoInterval {
            year,
            month,
            day,
            start_hour,
            hp,
            ap,
        });
    }
    out
}

/// The Hpo interval containing `sec` UT seconds into `year-month-day`, if the series has it.
///
/// `intervals` must be ascending, as [`parse_hpo`] returns. The cadence is read from the data rather
/// than passed in: the file states each interval's start hour, so one lookup serves Hp30 and Hp60
/// and a caller cannot pair a 30-minute series with a 60-minute assumption.
#[must_use]
pub fn hpo_at(
    intervals: &[HpoInterval],
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
) -> Option<HpoInterval> {
    let hour = if sec.is_finite() {
        (sec / 3600.0).clamp(0.0, 23.999_999)
    } else {
        0.0
    };
    // The last interval of the day that starts at or before the instant. Scanning backwards from
    // the day's rows keeps this correct for both cadences without naming either.
    intervals
        .iter()
        .rfind(|i| i.year == year && i.month == month && i.day == day && i.start_hour <= hour)
        .copied()
}

/// One hour of the USGS Dst series, as `tools/fetch-usgs-dst.sh` writes it.
///
/// USGS publishes two disturbance-storm-time indices from late September 2015 (Rigler 2017,
/// doi:10.3133/ofr20171037), both U.S. public domain: `Dst4` from the four observatories of the
/// traditional Dst, and `Dst3` from USGS observatories alone, more complete and with less latency.
/// They are a separate construction from Kyoto's Dst, whose baseline sits lower, so a model fitted
/// on Kyoto is refitted on these rather than fed them.
///
/// Each value is the mean of the hour's sixty 1-minute samples; the file stamps it at the hour's
/// centre, `hh:29:30`, and it is held here by the hour it covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UsgsDstHour {
    /// Calendar year, UT.
    pub year: i32,
    /// Calendar month, `1..=12`.
    pub month: u32,
    /// Day of month, `1..=31`.
    pub day: u32,
    /// The hour UT the value averages, `0..=23`.
    pub hour: u32,
    /// `Dst4` in nT, `None` where the service has no value for the hour.
    pub dst4: Option<f64>,
    /// `Dst3` in nT, `None` where the service has no value for the hour.
    pub dst3: Option<f64>,
}

/// Parse the hourly file `tools/fetch-usgs-dst.sh` assembles (`usgs_dst_hourly.txt` or an excerpt).
///
/// A row is `YYYY-MM-DDThh:29:30Z Dst4 Dst3`, each value a number or `null`. Comment lines and rows
/// of any other shape are skipped, a stamp off the hour's centre among them. A `null` stays `None`:
/// a missing storm index is not a quiet hour, and Dst 0 would read as one.
#[must_use]
pub fn parse_usgs_dst(text: &str) -> Vec<UsgsDstHour> {
    let value = |s: &str| -> Option<Option<f64>> {
        if s == "null" {
            Some(None)
        } else {
            s.parse::<f64>().ok().filter(|v| v.is_finite()).map(Some)
        }
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 3 || f[0].starts_with('#') {
            continue;
        }
        let t = f[0];
        if t.len() != 20
            || !t.is_ascii()
            || &t[4..5] != "-"
            || &t[7..8] != "-"
            || &t[10..11] != "T"
            || &t[13..] != ":29:30Z"
        {
            continue;
        }
        let (Ok(year), Ok(month), Ok(day), Ok(hour)) = (
            t[0..4].parse::<i32>(),
            t[5..7].parse::<u32>(),
            t[8..10].parse::<u32>(),
            t[11..13].parse::<u32>(),
        ) else {
            continue;
        };
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 {
            continue;
        }
        let (Some(dst4), Some(dst3)) = (value(f[1]), value(f[2])) else {
            continue;
        };
        out.push(UsgsDstHour {
            year,
            month,
            day,
            hour,
            dst4,
            dst3,
        });
    }
    out
}

/// The USGS Dst hour containing `sec` UT seconds into `year-month-day`, if the series has it.
///
/// `hours` must be ascending, as [`parse_usgs_dst`] returns; the lookup is a binary search, since the
/// full series is ~10⁵ rows and a propagator asks once per step. A non-finite `sec` has no hour.
#[must_use]
pub fn usgs_dst_at(
    hours: &[UsgsDstHour],
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
) -> Option<UsgsDstHour> {
    if !sec.is_finite() {
        return None;
    }
    let hour = (sec / 3600.0).clamp(0.0, 23.0) as u32;
    let key = (year, month, day, hour);
    let i = hours.partition_point(|h| (h.year, h.month, h.day, h.hour) < key);
    hours
        .get(i)
        .filter(|h| (h.year, h.month, h.day, h.hour) == key)
        .copied()
}

/// One day of the Space Environment Technologies `SOLFSMY.TXT` series — the four solar indices
/// JB2008 is fitted against and their 81-day centred means.
///
/// F10 is the familiar 10.7 cm radio flux; S10 (26–34 nm EUV), M10 (Mg II core-to-wing) and Y10
/// (X-ray + Lyman-α) are separate proxies for the parts of the spectrum that heat different layers,
/// each scaled to F10 units. They are what JB2008 has and NRLMSISE-00 and DTM2020 do not, and no
/// other publisher carries them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolarIndexDay {
    /// Calendar year.
    pub year: i32,
    /// Day of year, `1..=366`.
    pub day_of_year: u32,
    /// F10.7 daily value, solar flux units.
    pub f10: f64,
    /// F10.7 81-day centred mean.
    pub f81c: f64,
    /// S10 EUV index.
    pub s10: f64,
    /// S10 81-day centred mean.
    pub s81c: f64,
    /// M10 Mg II index.
    pub m10: f64,
    /// M10 81-day centred mean.
    pub m81c: f64,
    /// Y10 X-ray + Lyman-α index.
    pub y10: f64,
    /// Y10 81-day centred mean.
    pub y81c: f64,
}

/// One day of the Space Environment Technologies `DTCFILE.TXT` series: the exospheric temperature
/// change `ΔTc` (K) attributed to geomagnetic storming, at one-hour resolution.
///
/// This is JB2008's whole geomagnetic response — it takes no `ap` or `Kp` — and it is derived from
/// `Dst` rather than from a planetary range index, which is why the model tracks a storm's main
/// phase more sharply than the `ap`-driven models do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DstTemperatureDay {
    /// Calendar year.
    pub year: i32,
    /// Day of year, `1..=366`.
    pub day_of_year: u32,
    /// `ΔTc` (K) for each UT hour `00..=23`.
    pub dtc: [f64; 24],
}

/// Parse `SOLFSMY.TXT`. Comment lines (`#`) and malformed rows are skipped; the returned days are in
/// file order, which the product publishes ascending.
#[must_use]
pub fn parse_solfsmy(text: &str) -> Vec<SolarIndexDay> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // year, doy, julian day, then the eight indices; a trailing source-code field is ignored.
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 11 {
            continue;
        }
        let (Ok(year), Ok(day_of_year)) = (t[0].parse(), t[1].parse()) else {
            continue;
        };
        let mut v = [0.0; 8];
        let mut ok = true;
        for (i, slot) in v.iter_mut().enumerate() {
            match t[3 + i].parse::<f64>() {
                Ok(x) => *slot = x,
                Err(_) => ok = false,
            }
        }
        if !ok {
            continue;
        }
        out.push(SolarIndexDay {
            year,
            day_of_year,
            f10: v[0],
            f81c: v[1],
            s10: v[2],
            s81c: v[3],
            m10: v[4],
            m81c: v[5],
            y10: v[6],
            y81c: v[7],
        });
    }
    out
}

/// Parse `DTCFILE.TXT`. Each kept row is `DTC year doy` followed by 24 hourly `ΔTc` values.
#[must_use]
pub fn parse_dtcfile(text: &str) -> Vec<DstTemperatureDay> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 27 || t[0] != "DTC" {
            continue;
        }
        let (Ok(year), Ok(day_of_year)) = (t[1].parse(), t[2].parse()) else {
            continue;
        };
        let mut dtc = [0.0; 24];
        let mut ok = true;
        for (i, slot) in dtc.iter_mut().enumerate() {
            match t[3 + i].parse::<f64>() {
                Ok(x) => *slot = x,
                Err(_) => ok = false,
            }
        }
        if ok {
            out.push(DstTemperatureDay {
                year,
                day_of_year,
                dtc,
            });
        }
    }
    out
}

/// Nonlinear space-weather driver features for thermospheric-density regression (feed a
/// [`crate::density_rom::ReducedDynamics`]): `[(F + F̄)/2, Ap, ((F + F̄)/2)², √Ap, 1]`, with `F` the
/// observed F10.7 and `F̄` its 81-day centred mean. The combined flux and its square capture the
/// high-flux saturation that a bare linear F10.7 term misses — the same parameterization family JB2008
/// uses. On real CHAMP data these lift the held-out R² of the density fit from ~0.955 (linear) to
/// ~0.975; the trailing `1` is the intercept.
#[must_use]
pub fn density_drivers(f107: f64, f107_81: f64, ap: f64) -> [f64; 5] {
    let flux = f64::midpoint(f107, f107_81);
    [flux, ap, flux * flux, libm::sqrt(ap.max(0.0)), 1.0]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/celestrak_sw_sample.csv");

    #[test]
    fn parses_known_celestrak_rows() {
        let days = parse_space_weather(FIXTURE);
        assert_eq!(days.len(), 7);

        // 1957-10-01: AP_AVG 21, F10.7_OBS 269.3, F10.7_ADJ 269.8, F10.7_OBS_CENTER81 266.6.
        let d = find(&days, 1957, 10, 1).unwrap();
        assert!((d.ap - 21.0).abs() < 1e-9);
        assert!((d.f107_observed - 269.3).abs() < 1e-9);
        assert!((d.f107_adjusted - 269.8).abs() < 1e-9);
        assert!((d.f107_center81 - 266.6).abs() < 1e-9);

        // 2000-07-29 (start of the CHAMP demo month): AP_AVG 27, F10.7_OBS 153.2.
        let c = find(&days, 2000, 7, 29).unwrap();
        assert!((c.ap - 27.0).abs() < 1e-9);
        assert!((c.f107_observed - 153.2).abs() < 1e-9);
        assert!((c.f107_center81 - 181.1).abs() < 1e-9);
    }

    #[test]
    fn find_returns_none_for_a_missing_day() {
        let days = parse_space_weather(FIXTURE);
        assert!(find(&days, 1999, 1, 1).is_none());
    }

    #[test]
    fn parses_three_hourly_ap() {
        let days = parse_space_weather(FIXTURE);
        let d = find(&days, 2000, 7, 31).unwrap();
        assert_eq!(d.ap3h, [5.0, 7.0, 18.0, 7.0, 12.0, 15.0, 56.0, 48.0]);
        assert!((d.ap - 21.0).abs() < 1e-9); // AP_AVG of the same row
    }

    #[test]
    fn day_ordinal_is_contiguous_across_boundaries() {
        assert_eq!(
            days_from_civil(2000, 8, 1) - days_from_civil(2000, 7, 29),
            3
        );
        assert_eq!(
            days_from_civil(2000, 3, 1) - days_from_civil(2000, 2, 28),
            2
        ); // 2000 leap
        assert_eq!(
            days_from_civil(2001, 1, 1) - days_from_civil(2000, 12, 31),
            1
        );
    }

    /// The proleptic-Gregorian claim in `days_from_civil`'s doc comment covers negative years, and
    /// the `y - 399` in the era term is there for exactly them: Rust's `/` truncates toward zero, so
    /// a negative `y` needs the bias to land on the floor the algorithm assumes. Every fixture and
    /// every test above is CE — the feed starts in 1932 — so that branch was never evaluated, and
    /// neither `y + 399` nor `y / 399` moved an assertion.
    ///
    /// Two oracles, because the invariant alone would not catch a calendar that is self-consistently
    /// wrong. CPython's `date.toordinal()` fixes the CE anchors, and the Gregorian calendar's exact
    /// 400-year, 146 097-day cycle carries them across zero — which is what forces the branch.
    #[test]
    fn the_civil_day_number_is_gregorian_on_both_sides_of_year_zero() {
        // CPython: `date(y, m, d).toordinal() - date(1970, 1, 1).toordinal()`.
        for (year, month, day, expected) in [
            (1970, 1, 1, 0_i64),
            (1957, 10, 1, -4475),
            (2000, 2, 29, 11_016),
            (4, 2, 29, -718_008),
            (1, 3, 1, -719_103),
            (1, 1, 1, -719_162),
        ] {
            assert_eq!(
                days_from_civil(year, month, day),
                expected,
                "{year:04}-{month:02}-{day:02}"
            );
        }

        // 400 Gregorian years are exactly 146 097 days, so the same civil date four centuries
        // earlier is exactly that many days earlier — including where the subtraction lands the
        // era term on a negative year, which the January and February cases do.
        for (year, month, day) in [
            (1, 1, 1),
            (1, 2, 28),
            (1, 3, 1),
            (4, 2, 29),
            (100, 7, 15),
            (399, 12, 31),
            (400, 2, 29),
        ] {
            assert_eq!(
                days_from_civil(year - 400, month, day),
                days_from_civil(year, month, day) - 146_097,
                "the 400-year cycle breaks at {year:04}-{month:02}-{day:02}"
            );
        }

        // And the ordinal stays contiguous stepping backwards through 1 BCE, a Gregorian leap year.
        assert_eq!(days_from_civil(1, 1, 1) - days_from_civil(0, 12, 31), 1);
        assert_eq!(days_from_civil(0, 3, 1) - days_from_civil(0, 2, 29), 1);
        assert_eq!(days_from_civil(0, 1, 1) - days_from_civil(-1, 12, 31), 1);
    }

    /// `f.len() < 28` is the parser's bounds proof: the row is read at indices 0, 12–19, 20, 24, 25
    /// and 27, so 28 fields is exactly the minimum it can serve, and CelesTrak ships 31. That gap is
    /// why both directions of the guard were invisible — widening it to `<= 28` rejects only a width
    /// the feed never emits, and narrowing it to `== 28` accepts *short* rows and then indexes them.
    ///
    /// The second is the one that matters: `parse_space_weather` takes a downloaded file and is
    /// re-exported across the PyO3 boundary, where an out-of-bounds index is a process abort rather
    /// than an exception. "Malformed rows are skipped" is the documented contract and now has a test.
    #[test]
    fn the_parser_serves_the_narrowest_row_it_can_read_and_skips_anything_shorter() {
        let narrow: String = {
            let mut f = vec!["0"; 29];
            f[0] = "2000-08-02";
            for (i, slot) in f.iter_mut().enumerate().skip(12).take(8) {
                *slot = ["1", "2", "3", "4", "5", "6", "7", "8"][i - 12];
            }
            f[20] = "4.5"; // AP_AVG
            f[24] = "150.5"; // F10.7_OBS
            f[25] = "151.5"; // F10.7_ADJ
            f[26] = "OBS"; // F10.7_DATA_TYPE
            f[27] = "149.5"; // F10.7_OBS_CENTER81
            f[28] = "148.5"; // F10.7_OBS_LAST81
            f.join(",")
        };
        let days = parse_space_weather(&narrow);
        assert_eq!(
            days.len(),
            1,
            "a 29-field row is the narrowest the parser reads"
        );
        let d = days[0];
        assert_eq!(d.ap3h, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!((d.ap - 4.5).abs() < 1e-12);
        assert!((d.f107_observed - 150.5).abs() < 1e-12);
        assert!((d.f107_adjusted - 151.5).abs() < 1e-12);
        assert!((d.f107_center81 - 149.5).abs() < 1e-12);
        assert!((d.f107_last81 - 148.5).abs() < 1e-12);
        assert_eq!(d.quality, FluxQuality::Observed);

        // Anything shorter is skipped rather than indexed, at every width the row could be truncated
        // to — including the ones that would reach past `f[12]`, `f[20]`, `f[24]` and `f[28]`.
        for width in 1..29 {
            let short = narrow.split(',').take(width).collect::<Vec<_>>().join(",");
            assert!(
                parse_space_weather(&short).is_empty(),
                "a {width}-field row was not skipped"
            );
        }
    }

    #[test]
    fn nrlmsise_ap_array_matches_the_hand_built_storm_history() {
        let days = parse_space_weather(FIXTURE);
        // 2000-08-01, first 3-hour interval (sec = 0), assembled by hand from the 07-29..08-01 run.
        let a = nrlmsise_ap_array(&days, 2000, 8, 1, 0.0).unwrap();
        assert!((a[0] - 12.0).abs() < 1e-12); // daily Ap of 08-01
        assert!((a[1] - 9.0).abs() < 1e-12); // 08-01 interval 0 (current)
        assert!((a[2] - 48.0).abs() < 1e-12); // 07-31 interval 7 (−3 h)
        assert!((a[3] - 56.0).abs() < 1e-12); // 07-31 interval 6 (−6 h)
        assert!((a[4] - 15.0).abs() < 1e-12); // 07-31 interval 5 (−9 h)
        assert!((a[5] - 8.625).abs() < 1e-12); // mean 07-31[4..0] + 07-30[7..5]
        assert!((a[6] - 10.5).abs() < 1e-12); // mean 07-30[4..0] + 07-29[7..5]
    }

    /// The storm array is indexed by a *global* 3-hour interval, `day ordinal × 8 + subinterval`,
    /// and every other test here calls it at `sec = 0.0` — so the subinterval is always zero and
    /// neither the division that derives it nor the addition that folds it into the index is ever
    /// observed. Walking all eight intervals of 2000-08-01 pins both: the whole window has to slide
    /// by exactly one slot per step, against the flattened 07-29..08-01 history.
    #[test]
    fn the_storm_window_slides_one_slot_per_three_hours() {
        let days = parse_space_weather(FIXTURE);
        // 07-29, 07-30, 07-31, 08-01 laid end to end; 08-01 interval 0 is at index 24.
        #[rustfmt::skip]
        let flat = [
            27.0, 22.0, 22.0, 67.0, 39.0, 18.0, 9.0, 12.0,
            12.0, 15.0, 4.0, 5.0, 9.0, 6.0, 7.0, 7.0,
            5.0, 7.0, 18.0, 7.0, 12.0, 15.0, 56.0, 48.0,
            9.0, 7.0, 6.0, 9.0, 7.0, 22.0, 15.0, 18.0,
        ];
        let mean8 = |first_back: usize, now: usize| -> f64 {
            (1..=8).map(|k| flat[now - first_back - k + 1]).sum::<f64>() / 8.0
        };

        for s in 0..8_u32 {
            let now = 24 + s as usize;
            // Mid-interval, so the floor is doing work rather than landing on a boundary.
            let sec = f64::from(s) * 10_800.0 + 4_321.0;
            let a = nrlmsise_ap_array(&days, 2000, 8, 1, sec).unwrap();
            assert_eq!(a[0], 12.0, "daily Ap at interval {s}");
            for (slot, back) in (1..=4).zip(0..4) {
                assert_eq!(
                    a[slot],
                    flat[now - back],
                    "slot {slot} at interval {s} is not {back} intervals back"
                );
            }
            assert_eq!(a[5], mean8(4, now), "12–33 h mean at interval {s}");
            assert_eq!(a[6], mean8(12, now), "36–57 h mean at interval {s}");
        }

        // The subinterval is a floor, not a round: 10 799 s is still interval 0, 10 800 s is 1.
        assert_eq!(
            nrlmsise_ap_array(&days, 2000, 8, 1, 10_799.0).unwrap()[1],
            flat[24]
        );
        assert_eq!(
            nrlmsise_ap_array(&days, 2000, 8, 1, 10_800.0).unwrap()[1],
            flat[25]
        );
        // And it is clamped into the day rather than running off the end of it.
        assert_eq!(
            nrlmsise_ap_array(&days, 2000, 8, 1, 86_400.0).unwrap()[1],
            flat[31]
        );
        assert_eq!(
            nrlmsise_ap_array(&days, 2000, 8, 1, -1.0).unwrap()[1],
            flat[24]
        );
        assert_eq!(
            nrlmsise_ap_array(&days, 2000, 8, 1, f64::NAN).unwrap()[1],
            flat[24]
        );
    }

    /// The kernel history is the same eight columns, ordered oldest-first and ending one interval
    /// *before* the sample — which is the difference between a model that can be run forward and one
    /// that cannot, and it is off by one in the direction that would be invisible in a fit.
    #[test]
    fn the_storm_history_ends_at_the_interval_before_the_sample() {
        let days = parse_space_weather(FIXTURE);
        // 07-29, 07-30, 07-31, 08-01 laid end to end; 08-01 interval 0 is at index 24.
        #[rustfmt::skip]
        let flat = [
            27.0, 22.0, 22.0, 67.0, 39.0, 18.0, 9.0, 12.0,
            12.0, 15.0, 4.0, 5.0, 9.0, 6.0, 7.0, 7.0,
            5.0, 7.0, 18.0, 7.0, 12.0, 15.0, 56.0, 48.0,
            9.0, 7.0, 6.0, 9.0, 7.0, 22.0, 15.0, 18.0,
        ];
        for s in 0..8_u32 {
            let sec = f64::from(s) * 10_800.0 + 4_321.0;
            let h = sdtm_ap_history::<4>(&days, 2000, 8, 1, sec).unwrap();
            let now = 24 + s as usize;
            assert_eq!(h[3], flat[now - 1], "newest value at interval {s}");
            assert_eq!(h[0], flat[now - 4], "oldest value at interval {s}");
            assert_eq!(
                h.to_vec(),
                flat[now - 4..now],
                "the whole window at interval {s}"
            );
        }
        // Nothing of the sample's own interval reaches the history: interval 0 of 08-01 is 9.0 and
        // does not appear where the window ends at it.
        let h = sdtm_ap_history::<3>(&days, 2000, 8, 1, 1.0).unwrap();
        assert_eq!(h, [flat[21], flat[22], flat[23]]);
        // And the day boundary is crossed by the ordinal arithmetic, not by clamping.
        assert!(sdtm_ap_history::<200>(&days, 2000, 8, 1, 0.0).is_none());
        assert!(sdtm_ap_history::<4>(&days, 1999, 1, 1, 0.0).is_none());
    }

    #[test]
    fn nrlmsise_ap_array_is_none_without_enough_history() {
        let days = parse_space_weather(FIXTURE);
        // First day of the run: no 57 h of preceding 3-hourly history in the feed.
        assert!(nrlmsise_ap_array(&days, 2000, 7, 29, 0.0).is_none());
        // A day absent from the feed entirely.
        assert!(nrlmsise_ap_array(&days, 1999, 1, 1, 0.0).is_none());
    }

    /// `Kp` and `ap` come from the same eight columns of the same row, so the check that matters is
    /// that they are the *same* eight intervals: `Kp` published as tenths, `ap` as itself, and the
    /// published `KP_SUM` reconciling the row.
    #[test]
    fn the_three_hourly_kp_is_read_as_tenths_alongside_ap() {
        let days = parse_space_weather(FIXTURE);
        // 1957-10-01: KP1..KP8 = 43 40 30 20 37 23 43 37, KP_SUM 273; AP1..AP8 = 32 27 15 7 22 9 32 22.
        let d = find(&days, 1957, 10, 1).unwrap();
        assert_eq!(d.kp3h, [4.3, 4.0, 3.0, 2.0, 3.7, 2.3, 4.3, 3.7]);
        assert_eq!(d.ap3h, [32.0, 27.0, 15.0, 7.0, 22.0, 9.0, 32.0, 22.0]);
        let sum: f64 = d.kp3h.iter().sum();
        assert!(
            (sum * 10.0 - 273.0).abs() < 1e-9,
            "the row's own KP_SUM does not reconcile: {sum}"
        );
    }

    /// What the two Space Environment Technologies index parsers do with a line that is not a data row.
    ///
    /// Both read a file nobody in this repository writes, and both index `t[3 + i]` --- to 10 in one
    /// and to 26 in the other --- without a second bounds check, so a guard that lets a short line
    /// through is a panic and not a wrong number. The validation fixtures cannot see any of this:
    /// every `SOLFSMY` row in them carries the optional twelfth `Ssrc` field, every `DTCFILE` row
    /// carries exactly twenty-seven, and the only comment lines are prose rather than data.
    #[test]
    fn the_set_index_parsers_reject_every_shape_of_line_that_is_not_a_row() {
        // A commented-out data row is a comment. The `#` disqualifies it, not the field count ---
        // which is the difference between the two tests in the guard being `||` and being `&&`.
        let commented =
            "# 2018 358   2458477.0  69.5  70.5  57.9  57.8  65.5  67.6  63.6  65.1  1E11";
        assert!(
            parse_solfsmy(commented).is_empty(),
            "a commented-out row was read as data"
        );

        // Eleven fields is a row: the twelfth is the source code, which the parser ignores.
        let eleven = "  2018 358   2458477.0  69.5  70.5  57.9  57.8  65.5  67.6  63.6  65.1";
        let kept = parse_solfsmy(eleven);
        assert_eq!(kept.len(), 1, "an eleven-field row was dropped");
        assert_eq!(kept[0].y81c, 65.1);

        // Ten is not, and has to be dropped rather than indexed past the end.
        let ten = "  2018 358   2458477.0  69.5  70.5  57.9  57.8  65.5  67.6  63.6";
        assert!(
            parse_solfsmy(ten).is_empty(),
            "a short row was read as data"
        );

        // `DTCFILE`: a truncated `DTC` line satisfies the tag but not the width.
        let short_dtc = "DTC 2018 358    0  24  24  24  24  24  24  24  24  24   0   0   0   0   0   0  24  24  24  24  24  24";
        assert!(
            parse_dtcfile(short_dtc).is_empty(),
            "a truncated DTC row was read as data"
        );

        // And a full-width line that is not a `DTC` row is not one, however well its fields parse.
        let not_dtc = short_dtc.replacen("DTC", "GEO", 1) + "  24  24";
        assert!(
            parse_dtcfile(&not_dtc).is_empty(),
            "a full-width non-DTC row was read as data"
        );
    }

    #[test]
    fn density_drivers_build_the_expected_features() {
        let d = density_drivers(150.0, 130.0, 16.0);
        let flux = 140.0; // (150 + 130) / 2
        assert!((d[0] - flux).abs() < 1e-12);
        assert!((d[1] - 16.0).abs() < 1e-12);
        assert!((d[2] - flux * flux).abs() < 1e-9);
        assert!((d[3] - 4.0).abs() < 1e-12); // √16
        assert!((d[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn parse_hpo_keeps_the_row_it_can_use_and_drops_every_shape_that_is_not_data() {
        // One good row, so the refusals below cannot pass vacuously, and then each shape the GFZ
        // files are known to contain that is not an index: the header, an interval marked absent
        // by `Hp = -1.000`, one marked absent by `ap = -1`, a truncated row, and a row whose date
        // or index does not parse.
        let text = "\
#YYY MM DD hh.h hh._m        days      days_m   Hp60 ap60 D
2024 05 10 15.0 15.50 33733.62500 33733.64583  8.000   80 0
2024 05 08 00.0 00.50 33731.00000 33731.02083  0.000    0 0
2024 05 10 21.0 21.50 33733.87500 33733.89583  9.000  154
2024 05 10 16.0 16.50 33733.66667 33733.68750 -1.000    4 0
2024 05 10 17.0 17.50 33733.70833 33733.72917  9.000   -1 0
2024 05 10 18.0 18.50 33733.75000 33733.77083  9.000
XXXX 05 10 19.0 19.50 33733.79167 33733.81250  9.000  154 0
2024 05 10 20.0 20.50 33733.83333 33733.85417  9.00O  154 0

";
        let got = parse_hpo(text);
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(
            got[0],
            HpoInterval {
                year: 2024,
                month: 5,
                day: 10,
                start_hour: 15.0,
                hp: 8.0,
                ap: 80.0,
            }
        );
        // The quiet end of the scale is data, not a marker: `Hp = 0.000` with `ap = 0` is the
        // calmest hour the index can report, and only the file's own `-1` means absent.
        assert_eq!(got[1].hp, 0.0);
        assert_eq!(got[1].ap, 0.0);
        // Nine fields is a row -- the trailing `D` definitiveness flag is not read, and the width
        // guard is the bounds proof for `f[8]` rather than a shape assertion about the file.
        assert_eq!(got[2].start_hour, 21.0);
        assert_eq!(got[2].ap, 154.0);
    }

    #[test]
    fn the_flux_provenance_code_is_read_and_only_a_measured_day_counts_as_one() {
        // The four codes CelesTrak writes, the empty cell, and a code that is none of them. An
        // interpolated or forecast day is a number the series did not measure, so `is_measured` has
        // to separate `OBS` from the other four rather than "has a value" from "has none".
        assert_eq!(FluxQuality::parse("OBS"), FluxQuality::Observed);
        assert_eq!(FluxQuality::parse("INT"), FluxQuality::Interpolated);
        assert_eq!(FluxQuality::parse("PRD"), FluxQuality::Predicted);
        assert_eq!(FluxQuality::parse("PRM"), FluxQuality::PredictedMean);
        assert_eq!(FluxQuality::parse(""), FluxQuality::Unknown);
        assert_eq!(FluxQuality::parse("OBSERVED"), FluxQuality::Unknown);
        // The column is padded in the file, so the trim is load-bearing.
        assert_eq!(FluxQuality::parse("  OBS "), FluxQuality::Observed);

        assert!(FluxQuality::Observed.is_measured());
        for q in [
            FluxQuality::Interpolated,
            FluxQuality::Predicted,
            FluxQuality::PredictedMean,
            FluxQuality::Unknown,
        ] {
            assert!(!q.is_measured(), "{q:?} is not a measurement");
        }
    }

    #[test]
    fn hpo_at_reads_the_interval_that_contains_the_instant_and_none_where_the_series_stops() {
        let at = |day, start_hour, ap| HpoInterval {
            year: 2024,
            month: 5,
            day,
            start_hour,
            hp: 0.0,
            ap,
        };
        // 2024-05-10 hourly from 02 UT, so the day has a start as well as an end; 05-11 exists to
        // prove the lookup filters by date and not only by hour.
        let mut series: Vec<HpoInterval> = (2..24)
            .map(|h| at(10, f64::from(h), f64::from(h)))
            .collect();
        series.push(at(11, 0.0, 999.0));

        let ap = |sec| hpo_at(&series, 2024, 5, 10, sec).map(|i| i.ap);

        // The interval that contains the instant, not the nearest one: 15:59:59 is still 15 UT.
        assert_eq!(ap(15.0 * 3600.0), Some(15.0));
        assert_eq!(ap(16.0 * 3600.0 - 1.0), Some(15.0));
        assert_eq!(ap(16.0 * 3600.0), Some(16.0));

        // Before the day's first interval there is no answer to give, and the next day's row is
        // not one -- filling from it would read as a driver the series never published.
        assert_eq!(ap(0.0), None);
        assert_eq!(ap(2.0 * 3600.0), Some(2.0));

        // Out-of-range seconds clamp into the day rather than reaching the neighbouring one.
        assert_eq!(ap(86_400.0), Some(23.0));
        assert_eq!(ap(1.0e9), Some(23.0));
        assert_eq!(ap(-1.0), None); // clamped to 00:00 UT, which this day does not carry
        assert_eq!(ap(f64::NAN), None);
        assert_eq!(ap(f64::INFINITY), None);

        // A day the series does not cover.
        assert!(hpo_at(&series, 2024, 5, 9, 12.0 * 3600.0).is_none());
        assert!(hpo_at(&series, 2023, 5, 10, 12.0 * 3600.0).is_none());
        assert!(hpo_at(&series, 2024, 6, 10, 12.0 * 3600.0).is_none());
    }

    #[test]
    fn hpo_at_reads_the_cadence_from_the_data_rather_than_assuming_one() {
        // The same lookup over a 30-minute series has to resolve the half hour; an implementation
        // that rounded to whole hours would return the 15:00 row at 15:45.
        let at = |start_hour, ap| HpoInterval {
            year: 2024,
            month: 5,
            day: 10,
            start_hour,
            hp: 0.0,
            ap,
        };
        let series = [at(15.0, 80.0), at(15.5, 85.0), at(16.0, 94.0)];
        assert_eq!(
            hpo_at(&series, 2024, 5, 10, 15.75 * 3600.0).map(|i| i.ap),
            Some(85.0)
        );
        assert_eq!(
            hpo_at(&series, 2024, 5, 10, 15.5 * 3600.0).map(|i| i.ap),
            Some(85.0)
        );
        assert_eq!(
            hpo_at(&series, 2024, 5, 10, 15.49 * 3600.0).map(|i| i.ap),
            Some(80.0)
        );
    }

    #[test]
    fn parse_usgs_dst_keeps_nulls_as_absent_and_drops_every_shape_that_is_not_a_row() {
        // Two good rows (one with a null), then the header, a stamp off the hour's centre, a
        // missing column, an unparsable value, a non-finite value, an impossible hour, a
        // non-ASCII stamp of the right byte length, and stamps each wrong in exactly one place:
        // one separator, the month, the day at either end.
        let text = "\
#time                         Dst4                 Dst3
2024-05-11T02:29:30Z    -399.30463333333336   -421.92495
2024-05-31T23:29:30Z                  null     -7.45295
2024-05-11T03:00:00Z   -380.0 -400.0
2024-05-11T04:29:30Z   -380.0
2024-05-11T05:29:30Z   -38O.0 -400.0
2024-05-11T06:29:30Z   inf -400.0
2024-05-11T24:29:30Z   -1.0 -1.0
2024é5-11T07:29:30Z   -1.0 -1.0
2024/05-11T08:29:30Z   -1.0 -1.0
2024-05/11T09:29:30Z   -1.0 -1.0
2024-05-11X10:29:30Z   -1.0 -1.0
2024-13-11T11:29:30Z   -1.0 -1.0
2024-05-00T12:29:30Z   -1.0 -1.0
2024-05-32T13:29:30Z   -1.0 -1.0
";
        let got = parse_usgs_dst(text);
        assert_eq!(
            got,
            [
                UsgsDstHour {
                    year: 2024,
                    month: 5,
                    day: 11,
                    hour: 2,
                    dst4: Some(-399.304_633_333_333_36),
                    dst3: Some(-421.924_95),
                },
                UsgsDstHour {
                    year: 2024,
                    month: 5,
                    day: 31,
                    hour: 23,
                    dst4: None,
                    dst3: Some(-7.452_95),
                },
            ]
        );
    }

    #[test]
    fn usgs_dst_at_reads_the_hour_that_contains_the_instant_and_none_where_there_is_no_row() {
        let at = |day, hour| UsgsDstHour {
            year: 2024,
            month: 5,
            day,
            hour,
            dst4: Some(-f64::from(100 * day + hour)),
            dst3: None,
        };
        // 2024-05-10 from 02 UT with 05 UT absent, then 05-11 00 UT.
        let series: Vec<UsgsDstHour> = (2..24)
            .filter(|&h| h != 5)
            .map(|h| at(10, h))
            .chain([at(11, 0)])
            .collect();
        let dst = |sec| usgs_dst_at(&series, 2024, 5, 10, sec).and_then(|h| h.dst4);

        assert_eq!(dst(15.0 * 3600.0), Some(-1015.0));
        assert_eq!(dst(16.0 * 3600.0 - 1.0), Some(-1015.0));
        assert_eq!(dst(16.0 * 3600.0), Some(-1016.0));
        // An hour the series lacks is not filled from a neighbour.
        assert_eq!(dst(5.5 * 3600.0), None);
        assert_eq!(dst(0.0), None);
        // Seconds past the day clamp into its last hour, not the next day's first.
        assert_eq!(dst(86_400.0), Some(-1023.0));
        assert_eq!(dst(f64::NAN), None);
        assert_eq!(dst(f64::INFINITY), None);
        assert_eq!(
            usgs_dst_at(&series, 2024, 5, 11, 600.0).and_then(|h| h.dst4),
            Some(-1100.0)
        );
        assert!(usgs_dst_at(&series, 2024, 5, 12, 0.0).is_none());
    }

    /// A synthetic feed of `n` consecutive measured days whose flux is `100 + k` and whose 3-hourly
    /// `ap` is `k`, so that a copied day is identifiable by its value alone.
    fn ramp(n: u16) -> Vec<SpaceWeatherDay> {
        (0..n)
            .map(|k| {
                let (year, month, day) =
                    civil_from_days(days_from_civil(2024, 1, 1) + i64::from(k));
                SpaceWeatherDay {
                    year,
                    month,
                    day,
                    ap: f64::from(k),
                    ap3h: [f64::from(k); 8],
                    kp3h: [1.0; 8],
                    f107_observed: 100.0 + f64::from(k),
                    f107_adjusted: 100.0 + f64::from(k),
                    f107_center81: 0.0,
                    f107_last81: 0.0,
                    quality: FluxQuality::Observed,
                }
            })
            .collect()
    }

    /// The calendar round trip is the identity across leap days, century and era boundaries.
    ///
    /// A forecast row's date comes out of `civil_from_days` and is looked up again by
    /// `days_from_civil`, so a single-day error anywhere here reads as a missing day in every
    /// driver assembler rather than as a wrong date.
    #[test]
    fn the_calendar_round_trip_is_exact_over_four_centuries() {
        // The second range is the era before year 0, the branch a negative day number takes.
        for ordinal in (days_from_civil(1800, 1, 1)..days_from_civil(2200, 1, 1))
            .chain(days_from_civil(-401, 1, 1)..days_from_civil(1, 1, 1))
        {
            let (y, m, d) = civil_from_days(ordinal);
            assert_eq!(days_from_civil(y, m, d), ordinal, "{y}-{m}-{d}");
        }
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
        assert_eq!(civil_from_days(days_from_civil(1900, 3, 1)), (1900, 3, 1));
        assert_eq!(civil_from_days(days_from_civil(2000, 2, 29)), (2000, 2, 29));
    }

    /// Persistence holds the daily values and *moves* the trailing mean, which is the whole point of
    /// recomputing it: the window keeps rolling after the last measurement.
    #[test]
    fn persistence_holds_the_day_and_rolls_the_eighty_one_day_mean() {
        let days = ramp(100);
        let last = days[99];
        let out = extend_forecast(&days, ForecastMethod::Persistence, 2024, 5, 9).unwrap();
        assert_eq!(
            out.len(),
            130,
            "through 2024-05-09 is 30 days past 2024-04-09"
        );
        let forecast = &out[100..];
        assert!(
            forecast
                .iter()
                .all(|d| d.quality == FluxQuality::Extrapolated)
        );
        assert!(
            forecast
                .iter()
                .all(|d| d.f107_observed == last.f107_observed && d.ap3h == last.ap3h)
        );
        // The 81-day mean is not held: it climbs towards the persisted 199 as the window fills with
        // it, and the first forecast day's is the mean of the last 80 measured plus one copy. Written
        // as the sum it is rather than a closed form, so that the test cannot be wrong in the same
        // way the code could.
        let window: f64 = (20..100).map(|k| 100.0 + f64::from(k)).sum::<f64>() + 199.0;
        let expected = window / 81.0;
        assert!(
            (forecast[0].f107_last81 - expected).abs() < 1e-12,
            "{} != {expected}",
            forecast[0].f107_last81
        );
        assert!(
            forecast
                .windows(2)
                .all(|w| w[1].f107_last81 > w[0].f107_last81),
            "the rolling mean has to keep moving"
        );
        assert!(forecast.last().unwrap().f107_last81 < last.f107_observed);
    }

    /// Recurrence copies the day one solar rotation earlier, and past a rotation of lead it copies its
    /// own earlier copy — stated in the doc and checked here, because it is what bounds the method.
    #[test]
    fn recurrence_copies_one_solar_rotation_back_and_then_itself() {
        let days = ramp(100);
        let out = extend_forecast(&days, ForecastMethod::Recurrence, 2024, 6, 8).unwrap();
        assert_eq!(out.len(), 160);
        // Day 100 is 27 days after day 73.
        assert_eq!(out[100].f107_observed, days[73].f107_observed);
        assert_eq!(out[126].f107_observed, days[99].f107_observed);
        // And day 127 is 27 days after the synthesised day 100, not after a measured day.
        assert_eq!(out[127].f107_observed, days[73].f107_observed);
        assert_eq!(out[100].year, 2024);
        assert_eq!((out[100].month, out[100].day), (4, 10));
    }

    /// The measured prefix comes back as it went in: only synthesised days get recomputed means.
    #[test]
    fn the_measured_days_are_returned_unchanged() {
        let days = ramp(100);
        let out = extend_forecast(&days, ForecastMethod::Persistence, 2024, 5, 9).unwrap();
        assert_eq!(out[..100], days[..]);
    }

    /// A forecast across a new year before 1970 — a negative day number — carries the dates the
    /// calendar gives, not the last measured day's year.
    #[test]
    fn a_forecast_across_a_year_boundary_before_1970_carries_its_own_dates() {
        let start = days_from_civil(1959, 12, 25) - 99;
        let days: Vec<SpaceWeatherDay> = ramp(100)
            .into_iter()
            .zip(start..)
            .map(|(d, ordinal)| {
                let (year, month, day) = civil_from_days(ordinal);
                SpaceWeatherDay {
                    year,
                    month,
                    day,
                    ..d
                }
            })
            .collect();
        let out = extend_forecast(&days, ForecastMethod::Persistence, 1960, 1, 5).unwrap();
        assert_eq!(out.len(), 111);
        for (d, ordinal) in out.iter().zip(start..) {
            assert_eq!(days_from_civil(d.year, d.month, d.day), ordinal, "{d:?}");
        }
        assert_eq!((out[110].year, out[110].month, out[110].day), (1960, 1, 5));
    }

    /// The refusals, each of which would otherwise produce a mean over a window that is not there.
    #[test]
    fn the_forecast_refuses_a_feed_it_cannot_average() {
        assert!(extend_forecast(&[], ForecastMethod::Persistence, 2024, 5, 1).is_none());
        // No measured day at all.
        let mut forecast_only = ramp(100);
        for d in &mut forecast_only {
            d.quality = FluxQuality::Predicted;
        }
        assert!(extend_forecast(&forecast_only, ForecastMethod::Persistence, 2024, 5, 1).is_none());
        // Fewer than 81 measured days.
        assert!(extend_forecast(&ramp(80), ForecastMethod::Persistence, 2024, 5, 1).is_none());
        // A gap inside the window.
        let mut gapped = ramp(100);
        gapped.remove(50);
        assert!(extend_forecast(&gapped, ForecastMethod::Persistence, 2024, 5, 1).is_none());
        // `through` at or before the last measured day.
        let days = ramp(100);
        assert!(extend_forecast(&days, ForecastMethod::Persistence, 2024, 4, 9).is_none());
        assert!(extend_forecast(&days, ForecastMethod::Persistence, 2024, 4, 8).is_none());
        assert!(extend_forecast(&days, ForecastMethod::Persistence, 2024, 4, 10).is_some());
    }

    /// The provider's own forward tail is replaced, not built on.
    #[test]
    fn the_forecast_starts_at_the_last_measurement_and_drops_the_providers_tail() {
        let mut days = ramp(100);
        for d in &mut days[90..] {
            d.quality = FluxQuality::Predicted;
            d.f107_observed = 999.0;
        }
        // 2024-04-10 is ordinal 100, and the last measured day is now ordinal 89.
        let out = extend_forecast(&days, ForecastMethod::Persistence, 2024, 4, 10).unwrap();
        assert_eq!(out.len(), 101, "90 measured days and 11 synthesised");
        assert!(out.iter().all(|d| d.f107_observed != 999.0));
        assert_eq!(out[89].quality, FluxQuality::Observed);
        assert_eq!(out[100].quality, FluxQuality::Extrapolated);
    }

    /// The centred mean of a forecast day is an average of eighty-one values, not of the ones that
    /// happen to exist — the forty days past the horizon are synthesised and then dropped.
    #[test]
    fn the_centred_mean_of_a_forecast_day_is_not_truncated() {
        let days = ramp(100);
        let out = extend_forecast(&days, ForecastMethod::Persistence, 2024, 4, 15).unwrap();
        assert_eq!(out.len(), 106);
        let horizon = out.last().unwrap();
        // The last returned day is ordinal 105, so its centred window is 65..=145: the measured days
        // 65..=99 and then 46 copies of the persisted 199.
        let measured: f64 = (65..100).map(|k| 100.0 + f64::from(k)).sum();
        let expected = (measured + 46.0 * 199.0) / 81.0;
        assert!(
            (horizon.f107_center81 - expected).abs() < 1e-12,
            "{} != {expected}",
            horizon.f107_center81
        );
    }

    /// The SDTM-1 drivers are the model's conventions and not the file's defaults.
    #[test]
    fn the_sdtm_drivers_take_yesterdays_flux_and_the_trailing_mean() {
        let mut days = ramp(100);
        for (d, k) in days.iter_mut().zip(0_u16..) {
            d.f107_last81 = 150.0 + f64::from(k);
            d.f107_center81 = 900.0; // the quantity this model must not read
        }
        let kernel = crate::sdtm::coefficients::FIELD.kernel();
        let drivers = sdtm_drivers::<16>(&days, kernel, 2024, 4, 9, 12.0 * 3600.0).unwrap();
        assert_eq!(drivers.f107, days[98].f107_observed);
        assert_eq!(drivers.f107a, days[99].f107_last81);
        // The history ends at the interval before noon, all of it inside day 99 at `ap` 99, so the
        // kernel's response to a constant history is that constant times its own gain.
        let history = sdtm_ap_history::<16>(&days, 2024, 4, 9, 12.0 * 3600.0).unwrap();
        assert_eq!(
            drivers.storm,
            kernel.response(&history, THREE_HOURLY_CADENCE)
        );
        assert!(drivers.storm > 0.0);
        // And it refuses what it cannot assemble rather than shortening the history.
        assert!(sdtm_drivers::<16>(&days, kernel, 2024, 1, 1, 0.0).is_none());
    }
}
