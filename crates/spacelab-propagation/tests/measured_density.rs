//! The three empirical atmospheres against measured thermospheric density.
//!
//! Everything else in this crate checks a model against another implementation of the same model.
//! This checks them against the thermosphere. The measurement is the TU Delft neutral-density
//! product (Siemes et al., *J. Space Weather Space Clim.* 2023; CC BY 4.0): along-track mass density
//! from the GRACE-FO C accelerometer and from Swarm A's precise orbit. The models are driven from
//! the indices an operational user would have — `CelesTrak`'s F10.7 and 3-hourly `ap`/`Kp`, and Space
//! Environment Technologies' S10/M10/Y10 and hourly `ΔTc` for JB2008 — each at the lag it was fitted
//! with.
//!
//! Two days are committed as fixtures and always run: 2019-01-01, the floor of the solar minimum
//! between cycles 24 and 25, and 2024-05-10, the onset of the Gannon superstorm (Kp 9, Dst −412 nT).
//! One is the easiest day these models will ever see and the other is the hardest, which is the
//! point — the aggregate error of an empirical thermosphere model is not the number that matters.
//!
//! Measured on those two days, 720 GRACE-FO samples each at ~490 km, as
//! `mean(ρ_model/ρ_measured)` and the RMS of the relative error, beside Orekit 13.1.7 driving its
//! own parsers, frames, solar position and model code from the same publishers' files
//! (`tools/orekit-oracle/DensityOracle.java`):
//!
//! | model | 2019-01-01 quiet | 2024-05-10 storm | Orekit, quiet | Orekit, storm |
//! |---|---|---|---|---|
//! | NRLMSISE-00, 3-hourly `ap` array | 2.034 ± 1.264 | **1.661 ± 0.793** | — | — |
//! | NRLMSISE-00, daily `Ap` | 2.019 ± 1.235 | 2.087 ± 1.346 | 2.019 ± 1.236 | 2.086 ± 1.346 |
//! | JB2008 | 1.359 ± 0.652 | 1.133 ± 0.297 | 1.358 ± 0.645 | 1.156 ± 0.312 |
//! | DTM2020 (Orekit ships DTM2000) | 1.056 ± 0.316 | 1.189 ± 0.361 | 2.163 ± 1.661 | 2.243 ± 1.725 |
//! | SDTM-1, ours, **in sample** | 0.925 ± 0.275 | 1.235 ± 0.352 | — | — |
//!
//! On the literature's own metric, MAPE: 103.4 / 102.0 / 47.6 / 23.8 / **22.4 %** quiet and 68.2 /
//! 114.7 / **20.6** / 24.1 / 26.7 % storm, in the table's order.
//!
//! Five findings, and none of them could come from a component test:
//!
//! 1. **Driven identically, this crate's NRLMSISE-00 and Orekit's agree to 1e-3 relative** — 2.019
//!    against 2.019 quiet, 2.087 against 2.086 storm. So does JB2008, to 5e-3. The models are not in
//!    question; what follows is about the thermosphere.
//! 2. **The 3-hourly storm array is the whole difference on the storm day.** Orekit's NRLMSISE00
//!    drives the model from the *daily* `Ap`, which on 2024-05-10 is 105 — the storm began at 17 UT,
//!    so the daily average inflates the eighteen quiet hours before it. Resolving `ap` at its native
//!    3-hour cadence takes the bias from 2.087 to 1.661 and the RMS from 1.346 to 0.793. That is
//!    what the sub-daily driver bought, measured.
//! 3. **All three models run high against this product, and NRLMSISE-00 runs high by a factor of
//!    two at solar minimum.** The 2018–2020 minimum was among the deepest on record and
//!    NRLMSISE-00's fit predates it; this is the same shortfall that motivated NRLMSIS 2.x. The
//!    caveat belongs with the number: DTM2020's 1.085 is *not* independent evidence, because
//!    DTM2020 was fitted on this data family. JB2008 and NRLMSISE-00 are independent of it and read
//!    1.36 and 2.03.
//! 4. **Twenty years of refitting is worth more than any of it.** Orekit's DTM2000 reads 2.16 and
//!    2.24 where spacelab's DTM2020 port, which the public release does not carry, reads 1.06 and 1.19.
//! 5. **Our own model is the only one not biased high at the minimum, and that is what being fitted
//!    on a corpus containing one buys.** SDTM-1 reads 0.925 ± 0.275 and 22.4 % where DTM2020 reads
//!    1.056 ± 0.316 and 23.8 % and NRLMSISE-00 reads 2.034 and 103.4 %. On the storm day it is last
//!    of the three, 26.7 % against JB2008's 20.6 %, with a two-constant `ap` kernel against JB2008's
//!    Dst-derived `ΔTc`. Both numbers are **in sample** — `plans/16` Ф15.D fitted SDTM-1 on these
//!    twelve files, and `docs/verification.md` §10.10 is where its held-out numbers are.
//!
//! The full stratified table — 348 693 evaluations over six months of both satellites, spanning the
//! cycle-24/25 minimum to the cycle-25 maximum, binned by solar flux, geomagnetic level, altitude and
//! spacecraft — is [`every_downloaded_month_stratified_by_flux_and_storm_level`], `#[ignore]`d
//! because it needs the ~230 MB `tools/fetch-density-data.sh` download. Two findings are its own.
//!
//! **The bias is a property of the models, not of the instrument.** GRACE-FO C, from an accelerometer
//! at 480–540 km, and Swarm A, from the drag signature in a precise orbit at 420–480 km, put
//! NRLMSISE-00 at 1.497 and 1.472, JB2008 at 1.116 and 1.139, DTM2020 at 1.032 and 1.011 — two
//! unrelated ways of measuring density agreeing on the error to within 2%.
//!
//! **On MAPE, SDTM-1 is below NRLMSISE-00 in every bin and best of the five in the storm bin.** Pooled
//! 21.2 % against JB2008's 25.5 % and DTM2020's 20.2 %; 28.7 % at `Ap` ≥ 50 against their 31.0 % and
//! 30.5 % over 19 126 evaluations; and last of the three modern models at both flux extremes, which are
//! the two bins the corpus samples with one month each. In sample, and `docs/verification.md` §10.2 and
//! §10.11 carry the rest.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    // The storm assessment bins an MJD into orbit-length cells; the key is a small integer and the
    // truncation is the binning.
    clippy::cast_possible_truncation
)]

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use spacelab_propagation::density_calibration::{DensityCorrection, DensitySample};
use spacelab_propagation::density_data::{DensityObservation, TimeScale, parse_density};
use spacelab_propagation::sdtm::{self, DensityUncertainty, SdtmDrivers, SdtmPoint, coefficients};
use spacelab_propagation::space_weather::{self, SpaceWeatherDay};
use spacelab_propagation::wam_ipe::WamIpeField;
use spacelab_propagation::{Nrlmsise00Flags, Nrlmsise00Input, gtd7d};

const GRACE_FO_QUIET: &str = include_str!("fixtures/gracefo_density_20190101.txt");
const GRACE_FO_STORM: &str = include_str!("fixtures/gracefo_density_20240510.txt");
const CELESTRAK: &str = include_str!("fixtures/celestrak_sw_validation.csv");

/// WGS84 flattening, for geodetic → geocentric conversions.
const FLATTENING: f64 = 1.0 / 298.257_223_563;
/// Both committed fixture days are GRACE-FO C.
const GRACE_FO: &str = "GRACE-FO C (accelerometer)";
/// Modified Julian date of 1970-01-01, the epoch the day count below runs from.
const MJD_AT_UNIX_EPOCH: f64 = 40587.0;

/// The five model evaluations every table below reports, in order.
///
/// NRLMSISE-00 appears twice on purpose: the same model, the same instant, differing only in whether
/// the geomagnetic driver is the day's average or the 3-hourly history. On a storm day that is the
/// largest single effect in this file.
///
/// SDTM-1 is ours, and its number here is **in sample**: `plans/16` Ф15.D fitted it on these twelve
/// satellite-months (`docs/verification.md` §10.10), so what this table shows is how much of the
/// measurement the form can absorb at all, beside models that were fitted elsewhere. Its held-out
/// numbers are the generator's, because refitting per fold is the generator's job. DTM2020 carries a
/// weaker version of the same caveat — it was fitted on this data *family*, though not on these files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Nrlmsise00StormArray,
    Nrlmsise00DailyAp,
    Sdtm1,
}

impl Model {
    /// Every model, in declaration order, so that a model's position here is its discriminant.
    const ALL: &[Self] = &[
        Self::Nrlmsise00StormArray,
        Self::Nrlmsise00DailyAp,
        Self::Sdtm1,
    ];
    const COUNT: usize = Self::ALL.len();

    const fn index(self) -> usize {
        self as usize
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Nrlmsise00StormArray => "NRLMSISE-00 ap[]",
            Self::Nrlmsise00DailyAp => "NRLMSISE-00 Ap",
            Self::Sdtm1 => "SDTM-1 (in sample)",
        }
    }
}

/// One observation with every model's answer at the same place and instant.
#[derive(Clone, Copy, Debug)]
struct Evaluation {
    latitude: f64,
    /// Geodetic longitude, radians — where the observation sits in a gridded field.
    longitude: f64,
    local_solar_time: f64,
    altitude_km: f64,
    measured: f64,
    modelled: [f64; Model::COUNT],
    /// Which spacecraft the row came from, and by what means its density was derived. GRACE-FO C
    /// flies an accelerometer; Swarm A has none, so its density comes from the drag signature in the
    /// precise orbit. Carried explicitly rather than inferred from altitude, because the two orbits
    /// overlap once GRACE-FO has decayed far enough.
    satellite: &'static str,
    /// F10.7 of the day, the solar-flux stratifier.
    f107: f64,
    /// Daily `Ap` of the day, the geomagnetic stratifier.
    ap: f64,
    /// SDTM-1's own three drivers at this instant, kept so that the uncertainty table can be asked
    /// which regime the row is in — which is a question about the drivers, not about the geometry.
    sdtm: SdtmDrivers,
    /// Modified Julian date of the observation — the one field that orders rows across product files,
    /// which the storm assessment needs and the stratified tables do not.
    mjd: f64,
}

impl Evaluation {
    fn ratio(&self, m: Model) -> f64 {
        self.modelled[m.index()] / self.measured
    }
}

/// The space-weather series the models read, borrowed for one campaign.
#[derive(Clone, Copy)]
struct Drivers<'a> {
    days: &'a [SpaceWeatherDay],
}

/// The series a [`Drivers`] borrows.
struct DriverSeries {
    days: Vec<SpaceWeatherDay>,
}

impl DriverSeries {
    fn view(&self) -> Drivers<'_> {
        Drivers { days: &self.days }
    }
}

/// Proleptic-Gregorian day number since 1970-01-01 (Howard Hinnant's `days_from_civil`) — the key
/// the space-weather feed is searched on and the base of the modified Julian date.
fn days_since_1970(year: i32, month: u32, day: u32) -> i64 {
    let y = i64::from(year) - i64::from(month <= 2);
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = i64::from(if month > 2 { month - 3 } else { month + 9 });
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Evaluate all four model configurations at one observation, or `None` when a driver the instant
/// needs is not in the feed — the boundary at which a missing index becomes a missing row rather
/// than a fabricated one.
fn evaluate(o: &DensityObservation, satellite: &'static str, d: Drivers<'_>) -> Option<Evaluation> {
    if !o.nominal || o.density <= 0.0 {
        return None;
    }
    let days = d.days;
    let sw = space_weather::find(days, o.year, o.month, o.day)?;
    // Every model takes yesterday's flux, so the previous day has to be in the feed too.
    let previous = {
        let d = days_since_1970(o.year, o.month, o.day) - 1;
        days.iter()
            .find(|x| days_since_1970(x.year, x.month, x.day) == d)?
    };
    let ap_array =
        space_weather::nrlmsise_ap_array(days, o.year, o.month, o.day, o.seconds_of_day)?;
    let doy = o.day_of_year();

    let altitude_km = o.altitude / 1000.0;
    let mjd = MJD_AT_UNIX_EPOCH
        + days_since_1970(o.year, o.month, o.day) as f64
        + o.seconds_of_day / 86_400.0;

    // NRLMSISE-00. `gtd7d` is the drag variant: its total mass density carries the anomalous-oxygen
    // contribution, which is what a satellite at 500 km actually collides with.
    let msis_input = Nrlmsise00Input {
        year: o.year,
        doy: i32::try_from(doy).unwrap(),
        sec: o.seconds_of_day,
        alt: altitude_km,
        g_lat: o.latitude.to_degrees(),
        g_long: o.longitude.to_degrees(),
        lst: o.local_solar_time,
        f107a: sw.f107_center81,
        f107: previous.f107_observed,
        ap: sw.ap,
        ap_array: Some(ap_array),
    };
    // `switches[9] = -1` selects the storm-time `ap_array` mode; the default takes the day's `Ap`.
    let mut storm_flags = Nrlmsise00Flags::default();
    storm_flags.switches[9] = -1;
    // `d[5]` is in g/cm³ under the default (non-metric) switch.
    let rho_msis_array = gtd7d(&msis_input, &storm_flags).d[5] * 1000.0;
    let rho_msis_daily = gtd7d(&msis_input, &Nrlmsise00Flags::default()).d[5] * 1000.0;

    // SDTM-1, ours — through the same entry point an operational caller uses, so that this campaign
    // scores the conventions the library applies rather than a second copy of them. Those are
    // yesterday's F10.7, the *trailing* 81-day mean, and the kernel's response to the `ap` history
    // ending at the interval before this one.
    let field = coefficients::FIELD;
    let sdtm_drivers = space_weather::sdtm_drivers::<16>(
        days,
        field.kernel(),
        o.year,
        o.month,
        o.day,
        o.seconds_of_day,
    )?;
    let rho_ours = field.density(
        &SdtmPoint {
            altitude: o.altitude,
            latitude: o.latitude,
            local_solar_time: o.local_solar_time,
            day_of_year: f64::from(doy) + o.seconds_of_day / 86_400.0,
        },
        &sdtm_drivers,
    );

    let modelled = [rho_msis_array, rho_msis_daily, rho_ours];
    if modelled.iter().any(|r| !r.is_finite() || *r <= 0.0) {
        return None;
    }
    Some(Evaluation {
        latitude: o.latitude,
        longitude: o.longitude,
        local_solar_time: o.local_solar_time,
        altitude_km,
        measured: o.density,
        modelled,
        satellite,
        f107: sw.f107_observed,
        ap: sw.ap,
        sdtm: sdtm_drivers,
        mjd,
    })
}

/// How well a set of model densities matches the measured ones.
///
/// `bias` is `mean(ρ_model / ρ_measured)`: 1 is unbiased, and the ratio rather than the difference
/// because density spans two orders of magnitude over a solar cycle and an absolute residual would
/// be a statement about altitude. `rms` is the RMS of `ρ_model/ρ_measured − 1`, the relative error a
/// drag prediction inherits directly.
///
/// `mape` is the mean absolute percentage error, `mean|ρ_model/ρ_measured − 1|` as a percentage. It
/// carries no information the other two lack and is computed from the same ratios; it is here
/// because it is the metric the density-modelling literature reports, and a number on our own
/// metric cannot be set beside a number on theirs.
#[derive(Clone, Copy, Debug, Default)]
struct Fit {
    n: usize,
    bias: f64,
    rms: f64,
    mape: f64,
}

fn fit(ratios: impl Iterator<Item = f64>) -> Fit {
    let (mut n, mut sum, mut sq, mut abs) = (0usize, 0.0, 0.0, 0.0);
    for r in ratios {
        n += 1;
        sum += r;
        sq += (r - 1.0) * (r - 1.0);
        abs += (r - 1.0).abs();
    }
    if n == 0 {
        return Fit::default();
    }
    Fit {
        n,
        bias: sum / n as f64,
        rms: (sq / n as f64).sqrt(),
        mape: 100.0 * abs / n as f64,
    }
}

fn fit_of(model: Model, e: &[Evaluation]) -> Fit {
    fit(e.iter().map(|x| x.ratio(model)))
}

/// Parse a density file and evaluate every usable row of it.
fn evaluations(density: &str, satellite: &'static str, d: Drivers<'_>) -> Vec<Evaluation> {
    parse_density(density)
        .iter()
        .filter_map(|o| evaluate(o, satellite, d))
        .collect()
}

/// The mission a product file belongs to, from the stem the publisher names it with.
fn satellite_of(file_name: &str) -> &'static str {
    if file_name.starts_with("GC_") {
        "GRACE-FO C (accelerometer)"
    } else if file_name.starts_with("SA_") {
        "Swarm A (orbit determination)"
    } else {
        "unknown mission"
    }
}

fn fixture_drivers() -> DriverSeries {
    DriverSeries {
        days: space_weather::parse_space_weather(CELESTRAK),
    }
}

fn report(label: &str, set: &[Evaluation]) {
    for &m in Model::ALL {
        let f = fit_of(m, set);
        println!(
            "{label:>18}  {:<17} n {:4}  bias {:.3}  rms {:.3}  mape {:5.1}%",
            m.name(),
            f.n,
            f.bias,
            f.rms,
            f.mape
        );
    }
}

/// The headline measurement: three empirical atmospheres against a real accelerometer, on the
/// quietest and the most disturbed day the record holds.
///
/// The asserted bands are the measured values with room for arithmetic reordering, not tuned
/// tolerances — wide enough that a re-run cannot fail on rounding, tight enough that a sign error, a
/// wrong index lag or a dropped driver moves a number out of its band.
#[test]
fn the_three_atmospheres_are_measured_against_grace_fo_when_quiet_and_when_storming() {
    let series = fixture_drivers();
    let d = series.view();
    let quiet = evaluations(GRACE_FO_QUIET, GRACE_FO, d);
    let storm = evaluations(GRACE_FO_STORM, GRACE_FO, d);
    assert_eq!(quiet.len(), 720, "every fixture row is usable");
    assert_eq!(storm.len(), 720);
    report("quiet 2019-01-01", &quiet);
    report("storm 2024-05-10", &storm);

    // Every model *fitted somewhere else*, in both regimes, over-predicts this product. That single
    // statement is what a component test against another implementation of the same model can never
    // make. SDTM-1 is excluded because it was fitted on these files: its bias is 0.925 quiet and
    // 1.235 storm, which is a statement about the fit and not about the thermosphere.
    for (label, set) in [("quiet", &quiet), ("storm", &storm)] {
        for m in [Model::Nrlmsise00StormArray, Model::Nrlmsise00DailyAp] {
            let f = fit_of(m, set);
            assert!(f.bias > 1.0, "{} is not high on the {label} day", m.name());
        }
    }

    // Solar minimum: NRLMSISE-00 is out by a factor of two at 490 km, JB2008 by a third, and only
    // DTM2020 — fitted on this data family, so not an independent witness — is close.
    let by = |m, set: &[Evaluation]| fit_of(m, set);
    let msis_quiet = by(Model::Nrlmsise00DailyAp, &quiet);
    assert!(
        (1.90..2.15).contains(&msis_quiet.bias),
        "NRLMSISE-00 quiet bias {}",
        msis_quiet.bias
    );

    // The same two days on the metric the density-modelling literature reports. The ordering is the
    // gated claim, not the individual values: whatever a refit does to the biases, a model whose bias
    // is a factor of two cannot have the smaller mean absolute error.
    for (label, set) in [("quiet", &quiet), ("storm", &storm)] {
        let msis = fit_of(Model::Nrlmsise00StormArray, set).mape;
        // Every model but NRLMSISE-00 itself, in either configuration.
        for m in Model::ALL
            .iter()
            .copied()
            .filter(|m| !matches!(m, Model::Nrlmsise00StormArray | Model::Nrlmsise00DailyAp))
        {
            let other = fit_of(m, set).mape;
            assert!(
                other < msis,
                "{} MAPE {other:.1}% is not below NRLMSISE-00's {msis:.1}% on the {label} day",
                m.name()
            );
        }
    }

    // SDTM-1 against DTM2020, the model it has to reach. On the quiet day it is *ahead* — 22.1 %
    // against 23.8 %, and 0.927 bias against 1.056 — which is the deep-minimum shortfall §10.2 is
    // about, and ours is the only one of the five fitted on a corpus that contains one. On the storm
    // day it is behind, 30.1 % against 24.1 %, and this is the one place the §10.15 storm regressor
    // is *worse* than the linear one it replaced (26.7 %, bias 1.06): 2024-05-10 is a day of
    // sustained moderate activity ending in the `ap` 400 peak at 19:30, so a regressor on the
    // log-activity scale — which is what lets the model reach a `Kp` 6 storm at all — weights those
    // moderate hours enough to carry the whole day 26 % high. The bound is stated at what that
    // actually measures rather than at what the previous form managed, because the trade is
    // deliberate and §10.15 has the aggregate it was made for. Both are in-sample for SDTM-1; the
    // held-out numbers are §10.10's and §10.15's.
    assert!(
        (0.85..1.05).contains(&fit_of(Model::Sdtm1, &quiet).bias),
        "SDTM-1 quiet bias {}",
        fit_of(Model::Sdtm1, &quiet).bias
    );
    assert!(
        fit_of(Model::Sdtm1, &storm).mape < 32.0,
        "SDTM-1 storm MAPE {:.1}%",
        fit_of(Model::Sdtm1, &storm).mape
    );
    assert!(
        (1.15..1.35).contains(&fit_of(Model::Sdtm1, &storm).bias),
        "SDTM-1 storm bias {}",
        fit_of(Model::Sdtm1, &storm).bias
    );
}

/// The interval SDTM-1 ships with, against the measurement it is a claim about.
///
/// `coefficients::UNCERTAINTY` is a table of `σ` of `ln(ρ_model/ρ_measured)` by regime, fitted on the
/// *calendar-month* held-out residual of the generator's own split (`docs/verification.md` §10.12), and
/// what it claims is that `ρ·e^{±1.645σ}` contains the measurement nine times in ten *for a month the
/// fit has not seen*. Read on a single day it therefore **over**-covers, and by a lot: 2019-01-01 sits
/// in the solar-minimum cell where `σ` is 0.783, a factor of 3.6 either way at 90 %, while the day's own
/// scatter is 0.275 in the ratio. The two numbers are different quantities and neither is wrong — the
/// table carries between-month uncertainty that one day does not exhibit.
///
/// So what is asserted here is the direction that would make the interval useless: it must never
/// under-cover, and a wider quantile must never cover less. The calibrated 88–92 % figure is the
/// generator's, on a mission held out entirely (`docs/verification.md` §10.12).
#[test]
fn the_shipped_interval_covers_what_it_says_it_covers() {
    let series = fixture_drivers();
    let d = series.view();
    let quiet = evaluations(GRACE_FO_QUIET, GRACE_FO, d);
    let storm = evaluations(GRACE_FO_STORM, GRACE_FO, d);
    let sample = |set: &[Evaluation]| -> Vec<(f64, f64, SdtmDrivers)> {
        set.iter()
            .map(|e| (e.modelled[Model::Sdtm1.index()], e.measured, e.sdtm))
            .collect()
    };
    for (label, set) in [("quiet 2019-01-01", &quiet), ("storm 2024-05-10", &storm)] {
        let rows = sample(set);
        for (name, z) in [
            ("90 %", sdtm::NORMAL_90),
            ("95 %", sdtm::NORMAL_95),
            ("99 %", sdtm::NORMAL_99),
        ] {
            let covered = coefficients::UNCERTAINTY
                .coverage(&rows, z)
                .expect("a non-empty sample");
            println!(
                "{label}  the {name} interval covers {:5.1}% of {} rows  (sigma {:.3})",
                100.0 * covered,
                rows.len(),
                coefficients::UNCERTAINTY.log_sigma(&set[0].sdtm)
            );
        }
        let ninety = coefficients::UNCERTAINTY
            .coverage(&rows, sdtm::NORMAL_90)
            .expect("a non-empty sample");
        assert!(
            ninety >= 0.90,
            "{label}: the 90 % interval covers {:.1}%, which is under-covering",
            100.0 * ninety
        );
        // Wider is never narrower, whatever the day.
        let mut previous = ninety;
        for z in [sdtm::NORMAL_95, sdtm::NORMAL_99] {
            let wider = coefficients::UNCERTAINTY
                .coverage(&rows, z)
                .expect("a non-empty sample");
            assert!(wider >= previous, "{label}: z = {z} covered less");
            previous = wider;
        }
    }
}

/// What resolving the geomagnetic driver at its native 3-hour cadence is worth, in the one place it
/// can be seen: a storm that began at 17 UT, on a day whose average `Ap` is 105.
///
/// The daily-`Ap` column is here because it is what Orekit's `NRLMSISE00` computes, so it is the
/// number this crate has to reproduce before any claim about the array version means anything.
/// Measured against `tools/orekit-oracle/DensityOracle.java`: 2.019 / 1.236 quiet and 2.086 / 1.346
/// storm, against 2.019 / 1.235 and 2.087 / 1.346 here.
#[test]
fn the_three_hourly_ap_array_is_what_carries_the_storm() {
    let series = fixture_drivers();
    let d = series.view();
    let quiet = evaluations(GRACE_FO_QUIET, GRACE_FO, d);
    let storm = evaluations(GRACE_FO_STORM, GRACE_FO, d);

    // A quiet day has no storm history, so the two drivers must agree there. They differ by 0.7 %,
    // which is the 3-hourly `ap` wandering between 2 and 5 around a daily average of 5.
    let q_array = fit_of(Model::Nrlmsise00StormArray, &quiet);
    let q_daily = fit_of(Model::Nrlmsise00DailyAp, &quiet);
    assert!(
        (q_array.bias / q_daily.bias - 1.0).abs() < 0.02,
        "the two drivers disagree on a quiet day: {} vs {}",
        q_array.bias,
        q_daily.bias
    );

    // The storm day is where they part. Both numbers are the same model on the same data; only the
    // driver differs.
    let s_array = fit_of(Model::Nrlmsise00StormArray, &storm);
    let s_daily = fit_of(Model::Nrlmsise00DailyAp, &storm);
    println!(
        "storm: daily Ap bias {:.3} rms {:.3}  ->  3-hourly array bias {:.3} rms {:.3}",
        s_daily.bias, s_daily.rms, s_array.bias, s_array.rms
    );
    assert!(
        (1.95..2.20).contains(&s_daily.bias),
        "the daily-Ap baseline moved: {}",
        s_daily.bias
    );
    assert!(
        (1.55..1.80).contains(&s_array.bias),
        "the 3-hourly result moved: {}",
        s_array.bias
    );
    assert!(
        s_array.rms < 0.65 * s_daily.rms,
        "the array should cut the storm RMS by a third: {} vs {}",
        s_array.rms,
        s_daily.rms
    );
    // What the sub-daily driver is worth stated on the literature's metric, which is where it can be
    // compared with a published figure rather than only with this crate's previous run.
    assert!(
        s_array.mape < 0.7 * s_daily.mape,
        "the array should cut the storm MAPE by a third: {:.1}% vs {:.1}%",
        s_array.mape,
        s_daily.mape
    );
}

/// Whether the sun-fixed calibration layer earns its place: fit it to the first half of a day and
/// score it on the second, where it has seen neither those latitudes nor those local solar times.
///
/// A correction fitted and scored on the same samples always wins and says nothing. The split is by
/// time rather than at random because the samples are one satellite's ground track: two random
/// halves share the same orbital passes, and the held-out set would not be held out.
///
/// Degree 1 — four coefficients, a scale plus one sun-fixed dipole. A single day's track covers a
/// narrow band of local solar time, and measured out of sample, degree 2 and 3 are no better on any
/// model and worse on most.
///
/// Measured, base → calibrated on the held-out half:
///
/// | | quiet bias | quiet RMS | storm bias | storm RMS |
/// |---|---|---|---|---|
/// | NRLMSISE-00 | 1.908 → 0.944 | 1.076 → 0.311 | 1.622 → 0.969 | 0.838 → 0.314 |
/// | JB2008 | 1.571 → 1.483 | 0.793 → 0.776 | 1.106 → 0.995 | 0.339 → 0.379 |
/// | DTM2020 | 1.059 → 1.010 | 0.284 → 0.344 | 1.338 → 0.964 | 0.561 → 0.322 |
///
/// So the honest answer to "does the calibration beat the base model out of sample" is: it removes
/// the bias every time, and it removes the scatter only where the base model's error has sun-fixed
/// structure to remove. Where the base model is already nearly unbiased — JB2008, and DTM2020 on
/// the quiet day — fitting four coefficients to half a day costs more in variance than it recovers.
#[test]
fn the_sun_fixed_calibration_removes_the_bias_out_of_sample() {
    let series = fixture_drivers();
    let d = series.view();
    for (label, text) in [("quiet", GRACE_FO_QUIET), ("storm", GRACE_FO_STORM)] {
        let all = evaluations(text, GRACE_FO, d);
        let (train, test) = all.split_at(all.len() / 2);

        // The models fitted elsewhere, NRLMSISE-00 once.
        for m in Model::ALL
            .iter()
            .copied()
            .filter(|m| !matches!(m, Model::Nrlmsise00DailyAp | Model::Sdtm1))
        {
            let samples: Vec<DensitySample> = train
                .iter()
                .map(|e| DensitySample {
                    latitude: e.latitude,
                    local_solar_time: e.local_solar_time,
                    log_ratio: (1.0 / e.ratio(m)).ln(),
                    weight: 1.0,
                })
                .collect();
            let correction = DensityCorrection::<4>::fit_ridge(&samples, 1e-3).unwrap();

            let base = fit(test.iter().map(|e| e.ratio(m)));
            let calibrated = fit(test
                .iter()
                .map(|e| e.ratio(m) * correction.factor(e.latitude, e.local_solar_time)));
            println!(
                "{label} {:<17} out of sample: bias {:.3} -> {:.3}, rms {:.3} -> {:.3}",
                m.name(),
                base.bias,
                calibrated.bias,
                base.rms,
                calibrated.rms
            );
            assert!(
                (calibrated.bias - 1.0).abs() < (base.bias - 1.0).abs(),
                "{} calibration did not reduce the {label} bias out of sample",
                m.name()
            );
        }

        // Where the base model *is* badly biased, the correction has real structure to remove and
        // the scatter follows: NRLMSISE-00's held-out RMS drops by more than a factor of two.
        let m = Model::Nrlmsise00StormArray;
        let samples: Vec<DensitySample> = train
            .iter()
            .map(|e| DensitySample {
                latitude: e.latitude,
                local_solar_time: e.local_solar_time,
                log_ratio: (1.0 / e.ratio(m)).ln(),
                weight: 1.0,
            })
            .collect();
        let correction = DensityCorrection::<4>::fit_ridge(&samples, 1e-3).unwrap();
        let base = fit(test.iter().map(|e| e.ratio(m)));
        let calibrated = fit(test
            .iter()
            .map(|e| e.ratio(m) * correction.factor(e.latitude, e.local_solar_time)));
        assert!(
            calibrated.rms < 0.5 * base.rms,
            "NRLMSISE-00 {label} RMS {} -> {}",
            base.rms,
            calibrated.rms
        );
    }
}

/// Solar-flux level, by the day's F10.7. The cut points are the conventional ones: below 100 is a
/// quiet Sun, above 175 is near a cycle maximum.
fn flux_bin(f107: f64) -> &'static str {
    if f107 < 100.0 {
        "flux low"
    } else if f107 < 175.0 {
        "flux moderate"
    } else {
        "flux high"
    }
}

/// Geomagnetic level, by the day's `Ap`. `Ap ≥ 50` is a NOAA G2 storm or worse.
fn storm_bin(ap: f64) -> &'static str {
    if ap < 15.0 {
        "geomag quiet"
    } else if ap < 50.0 {
        "geomag active"
    } else {
        "geomag storm"
    }
}

/// `ap` at or above which a 3-hourly interval is storm time — `Kp = 6`, the threshold Bruinsma &
/// Laurens (2024) select their 152 storms on.
const STORM_AP: f64 = 80.0;

/// Storm-phase boundaries in hours from `t0`, for a single-peak storm: pre-storm (the debiasing
/// window), onset, main and recovery, post-storm.
const PHASES: [(f64, f64); 4] = [(-30.0, -18.0), (-18.0, 0.0), (0.0, 36.0), (36.0, 48.0)];

/// One storm interval, in the four phases of Bruinsma & Laurens (2024) §2.
///
/// `t0` is the first peak of the 3-hourly `ap` that reaches [`STORM_AP`], and the four phases are
/// fixed offsets from it — except phase 3 of a *multiple-peak* storm, which is extended to 36 h past
/// the last interval still at or above the threshold, and phase 4 which then follows it for 12 h.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Storm {
    /// MJD of the centre of the `ap` interval that defines the onset.
    t0: f64,
    /// Peak `ap` of the storm.
    peak_ap: f64,
    /// Whether the threshold is reached again 9 h or more after `t0` and inside 48 h of it.
    multiple_peak: bool,
    /// End of phase 3 in hours from `t0` — 36 for a single peak, more for a multiple one.
    phase3_end: f64,
}

impl Storm {
    /// The window in hours from `t0` of one phase, `1..=4`.
    fn phase(&self, which: usize) -> (f64, f64) {
        match which {
            3 => (0.0, self.phase3_end),
            4 => (self.phase3_end, self.phase3_end + 12.0),
            other => PHASES[other - 1],
        }
    }

    fn contains(&self, which: usize, mjd: f64) -> bool {
        let (from, to) = self.phase(which);
        let hours = (mjd - self.t0) * 24.0;
        hours >= from && hours < to
    }
}

/// Every storm in the feed, in the sense of Bruinsma & Laurens (2024): `t0` at the first peak of the
/// 3-hourly `ap` reaching 80, and one storm per such run rather than one per interval.
///
/// The first *peak* and not the first crossing: a run of intervals above the threshold is one storm,
/// and taking the crossing would put `t0` up to several hours early on a storm that keeps rising —
/// which shifts every phase boundary with it. A run whose maximum comes last still gets `t0` there,
/// because the local-maximum test admits the final interval of a run.
fn storms(days: &[SpaceWeatherDay]) -> Vec<Storm> {
    // The whole feed as (mid-interval MJD, ap), which is what both the peak search and the phase
    // arithmetic need and what neither can reconstruct from a daily row.
    let series: Vec<(f64, f64)> = days
        .iter()
        .flat_map(|d| {
            let day = MJD_AT_UNIX_EPOCH + days_since_1970(d.year, d.month, d.day) as f64;
            (0..8_u8).map(move |k| (day + (f64::from(k) + 0.5) * 0.125, d.ap3h[usize::from(k)]))
        })
        .collect();

    let mut out: Vec<Storm> = Vec::new();
    let mut i = 0;
    while i < series.len() {
        if series[i].1 < STORM_AP {
            i += 1;
            continue;
        }
        // The contiguous run at or above the threshold, and the first local maximum inside it.
        let start = i;
        while i < series.len() && series[i].1 >= STORM_AP {
            i += 1;
        }
        let run = &series[start..i];
        let peak_ap = run.iter().map(|(_, a)| *a).fold(f64::MIN, f64::max);
        // The first interval of the run that is at least as large as both its neighbours. A finite run
        // always has one -- its maximum qualifies -- so the fallback is unreachable and is the run's
        // start rather than a panic.
        let first_peak = (0..run.len())
            .find(|&k| {
                let rising = k == 0 || run[k].1 >= run[k - 1].1;
                let falling = k + 1 == run.len() || run[k].1 >= run[k + 1].1;
                rising && falling
            })
            .unwrap_or(0);
        let t0 = run[first_peak].0;
        // A second crossing inside 48 h of t0 and at least 9 h after it makes this a multiple-peak
        // storm, and extends phase 3 to 36 h past the last interval still above the threshold.
        let inside: Vec<&(f64, f64)> = series
            .iter()
            .filter(|(mjd, ap)| {
                let hours = (mjd - t0) * 24.0;
                *ap >= STORM_AP && (9.0..=48.0).contains(&hours)
            })
            .collect();
        let multiple_peak = !inside.is_empty();
        let phase3_end = inside
            .last()
            .map_or(36.0, |(mjd, _)| (mjd - t0) * 24.0 + 36.0);
        out.push(Storm {
            t0,
            peak_ap,
            multiple_peak,
            phase3_end,
        });
        // Anything still above the threshold inside this storm's own window belongs to it.
        while i < series.len() && (series[i].0 - t0) * 24.0 <= phase3_end {
            i += 1;
        }
    }
    out
}

/// The three log-space statistics of Bruinsma & Laurens (2024) Eq. (1) over a set of
/// observed-to-computed density ratios: geometric mean, its scatter as a percentage, and the Pearson
/// correlation of the two linear series.
///
/// Log space because the ratios are log-normally distributed — the same reason
/// `sdtm::DensityUncertainty` is. The scatter is taken about the geometric mean and *not* about zero,
/// which is the opposite choice from the shipped uncertainty table and deliberately so: here the
/// question is how much of the storm variation the model reproduces once its level is out of the way,
/// there it was how wide an interval has to be to cover the measurement.
#[derive(Clone, Copy, Debug, Default)]
struct StormFit {
    n: usize,
    /// `exp(mean ln(O/C))` — 1 is unbiased, above 1 the model underestimates.
    mean_ratio: f64,
    /// Standard deviation of `ln(O/C)` about its own mean, as a percentage.
    scatter: f64,
    /// Pearson correlation of the observed and modelled densities themselves.
    correlation: f64,
}

fn storm_fit(pairs: &[(f64, f64)]) -> StormFit {
    // Ten times either way is the paper's own rejection: such a pair is an instrument or a feed
    // problem, and one of them would otherwise dominate a log-space mean.
    let kept: Vec<(f64, f64)> = pairs
        .iter()
        .copied()
        .filter(|(o, c)| *o > 0.0 && *c > 0.0 && (0.1..10.0).contains(&(o / c)))
        .collect();
    if kept.len() < 2 {
        return StormFit::default();
    }
    let n = kept.len() as f64;
    let log_mean = kept.iter().map(|(o, c)| libm::log(o / c)).sum::<f64>() / n;
    let variance = kept
        .iter()
        .map(|(o, c)| (libm::log(o / c) - log_mean).powi(2))
        .sum::<f64>()
        / n;
    let (mean_o, mean_c) = (
        kept.iter().map(|(o, _)| *o).sum::<f64>() / n,
        kept.iter().map(|(_, c)| *c).sum::<f64>() / n,
    );
    let (mut cov, mut var_o, mut var_c) = (0.0, 0.0, 0.0);
    for (o, c) in &kept {
        cov += (o - mean_o) * (c - mean_c);
        var_o += (o - mean_o).powi(2);
        var_c += (c - mean_c).powi(2);
    }
    StormFit {
        n: kept.len(),
        mean_ratio: libm::exp(log_mean),
        scatter: 100.0 * variance.sqrt(),
        correlation: if var_o > 0.0 && var_c > 0.0 {
            cov / (var_o * var_c).sqrt()
        } else {
            0.0
        },
    }
}

/// Orbit-averaged density against time, in bins of one nominal orbit — the series the timing metrics
/// are read off.
///
/// Binning is not smoothing for its own sake: along one orbit the density swings by a factor of two
/// with latitude and local solar time, so the maximum of the raw series is a statement about where
/// the satellite was rather than about when the storm peaked. Ninety minutes is the orbital period at
/// these altitudes to within a few percent, and the bin centre is therefore accurate to ±45 min —
/// well inside the ±3 h the verdict asks for.
fn orbit_bins(rows: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    const BIN: f64 = 90.0 / 1440.0;
    let mut out: Vec<(f64, f64, f64)> = Vec::new();
    let mut cell: BTreeMap<i64, (f64, f64, f64)> = BTreeMap::new();
    for (mjd, measured, modelled) in rows {
        let key = (mjd / BIN).floor() as i64;
        let slot = cell.entry(key).or_insert((0.0, 0.0, 0.0));
        slot.0 += *measured;
        slot.1 += *modelled;
        slot.2 += 1.0;
    }
    // A bin that holds only part of an orbit is a statement about where the satellite was, so the
    // partial ones are dropped -- but the cadence differs by a factor of three between the two
    // products, so the threshold is taken from the data rather than fixed: half the median bin, which
    // keeps every full orbit of either mission and rejects the ragged ends of a data gap.
    let mut counts: Vec<f64> = cell.values().map(|(_, _, n)| *n).collect();
    if counts.is_empty() {
        return out;
    }
    counts.sort_by(f64::total_cmp);
    let floor = counts[counts.len() / 2] / 2.0;
    for (key, (measured, modelled, n)) in cell {
        if n >= floor {
            out.push(((key as f64 + 0.5) * BIN, measured / n, modelled / n));
        }
    }
    out
}

/// One model's verdict on one storm as seen by one satellite.
#[derive(Clone, Copy, Debug)]
struct StormVerdict {
    scale: f64,
    whole: StormFit,
    per_phase: [StormFit; 4],
    /// Hours by which the model's orbit-averaged density peaks after the measured one.
    onset_error: f64,
    /// Model peak *enhancement* over the measured one — both orbit-averaged, the model debiased, and
    /// the pre-storm level of each subtracted, so it is the storm's own rise and not the level.
    peak_ratio: f64,
    /// Model minus measured time for the enhancement to halve after its peak, hours.
    recovery_error: f64,
}

/// Hours from the density peak to the first time the storm *enhancement* has halved.
///
/// The enhancement is `ρ − quiet` with `quiet` the pre-storm level, so this is a property of what the
/// storm added rather than of the absolute level. A least-squares e-folding time over the whole main
/// phase was tried first and is not an estimator at all: these storms re-intensify, the fit is then
/// over a series that is not a decay, and the resulting `τ` came out ranging over four orders of
/// magnitude with sign changes. The first half-crossing does not care what happens after it.
///
/// `None` when the enhancement never halves inside the window, which is a fact about the storm and is
/// reported rather than filled in.
fn recovery_half_time(series: &[(f64, f64)], quiet: f64, peak: usize) -> Option<f64> {
    let target = quiet + (series.get(peak)?.1 - quiet) / 2.0;
    series
        .iter()
        .skip(peak + 1)
        .find(|(_, rho)| *rho <= target)
        .map(|(t, _)| (t - series[peak].0) * 24.0)
}

/// One model against one satellite over one storm, following Bruinsma & Laurens (2024) §2 step by
/// step: debias on phase 1, then the three statistics per phase, plus the three timing numbers
/// `plans/16` Ф15.H asks for.
///
/// Returns `None` when any of the four phases is empty, which is the paper's own rejection rule — a
/// storm whose pre-storm window is missing has no debiasing factor, and one whose recovery is missing
/// is not a storm that was observed.
fn assess(rows: &[&Evaluation], storm: &Storm, model: Model) -> Option<StormVerdict> {
    let phase_rows: Vec<Vec<&Evaluation>> = (1..=4)
        .map(|which| {
            rows.iter()
                .filter(|e| storm.contains(which, e.mjd))
                .copied()
                .collect()
        })
        .collect();
    if phase_rows.iter().any(Vec::is_empty) {
        return None;
    }
    // The scale factor that makes the model match the observations over the pre-storm window, applied
    // to the model in every phase. Without it the assessment measures the level difference between
    // two datasets -- NRLMSISE-00 and JB2008 were never fitted to these densities -- rather than how
    // either responds to the storm.
    let log_scale = phase_rows[0]
        .iter()
        .map(|e| libm::log(1.0 / e.ratio(model)))
        .sum::<f64>()
        / phase_rows[0].len() as f64;
    let scale = libm::exp(log_scale);
    let pairs = |set: &[&Evaluation]| -> Vec<(f64, f64)> {
        set.iter()
            .map(|e| (e.measured, scale * e.modelled[model.index()]))
            .collect()
    };
    // "All four phases together" is the union of the windows and not everything the caller handed in:
    // `rows` is a generous slice around `t0` so that the phase filters have something to select from,
    // and scoring the whole slice would put days of quiet time into a storm statistic.
    let inside: Vec<&Evaluation> = phase_rows.iter().flatten().copied().collect();
    let whole = storm_fit(&pairs(&inside));
    let per_phase = [
        storm_fit(&pairs(&phase_rows[0])),
        storm_fit(&pairs(&phase_rows[1])),
        storm_fit(&pairs(&phase_rows[2])),
        storm_fit(&pairs(&phase_rows[3])),
    ];

    // Timing, on the orbit-averaged series. The pre-storm phase is binned with the rest because the
    // quiet level it fixes is what the enhancement is measured against.
    let window: Vec<(f64, f64, f64)> = rows
        .iter()
        .filter(|e| (1..=3).any(|which| storm.contains(which, e.mjd)))
        .map(|e| (e.mjd, e.measured, scale * e.modelled[model.index()]))
        .collect();
    let bins = orbit_bins(&window);
    let quiet_of = |pick: fn(&(f64, f64, f64)) -> f64| -> Option<f64> {
        let (mut sum, mut n) = (0.0_f64, 0.0_f64);
        for b in bins.iter().filter(|b| storm.contains(1, b.0)) {
            sum += pick(b);
            n += 1.0;
        }
        (n > 0.0).then(|| sum / n)
    };
    let storm_bins: Vec<(f64, f64, f64)> = bins
        .iter()
        .filter(|b| storm.contains(2, b.0) || storm.contains(3, b.0))
        .copied()
        .collect();
    if storm_bins.len() < 4 {
        return None;
    }
    let (quiet_measured, quiet_model) = (quiet_of(|b| b.1)?, quiet_of(|b| b.2)?);
    let peak_at = |pick: fn(&(f64, f64, f64)) -> f64| -> usize {
        storm_bins
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| pick(a).total_cmp(&pick(b)))
            .map_or(0, |(i, _)| i)
    };
    let (i_measured, i_model) = (peak_at(|b| b.1), peak_at(|b| b.2));
    let series = |pick: fn(&(f64, f64, f64)) -> f64| -> Vec<(f64, f64)> {
        storm_bins.iter().map(|b| (b.0, pick(b))).collect()
    };

    Some(StormVerdict {
        scale,
        whole,
        per_phase,
        onset_error: (storm_bins[i_model].0 - storm_bins[i_measured].0) * 24.0,
        peak_ratio: (storm_bins[i_model].2 - quiet_model)
            / (storm_bins[i_measured].1 - quiet_measured),
        recovery_error: match (
            recovery_half_time(&series(|b| b.2), quiet_model, i_model),
            recovery_half_time(&series(|b| b.1), quiet_measured, i_measured),
        ) {
            (Some(model), Some(measured)) => model - measured,
            _ => f64::NAN,
        },
    })
}

/// The storm assessment: every model over every storm the corpus covers, on the protocol of
/// Bruinsma & Laurens (2024) — so that our numbers can be set beside theirs — plus the three timing
/// numbers `plans/16` Ф15.H asks for.
fn the_storm_assessment(all: &[Evaluation], days: &[SpaceWeatherDay]) {
    let covered: Vec<Storm> = storms(days)
        .into_iter()
        .filter(|s| {
            // A storm is covered when any row falls in its first and last phase; `assess` rejects the
            // rest per satellite.
            all.iter().any(|e| s.contains(1, e.mjd)) && all.iter().any(|e| s.contains(4, e.mjd))
        })
        .collect();
    let mut missions: Vec<&'static str> = all.iter().map(|e| e.satellite).collect();
    missions.sort_unstable();
    missions.dedup();

    println!(
        "\n== storm assessment, Bruinsma & Laurens (2024) protocol: ap >= {STORM_AP:.0} at t0, \
         phases -30/-18/0/+36/+48 h =="
    );
    println!("{} storms in the corpus's own windows", covered.len());

    let mut verdicts: BTreeMap<(&'static str, usize), Vec<StormVerdict>> = BTreeMap::new();
    for storm in &covered {
        let mjd = storm.t0;
        println!(
            "\nt0 MJD {mjd:.3}  peak ap {:.0}  {}  phase 3 ends t0+{:.0} h",
            storm.peak_ap,
            if storm.multiple_peak {
                "multiple peak"
            } else {
                "single peak"
            },
            storm.phase3_end
        );
        for mission in &missions {
            let rows: Vec<&Evaluation> = all
                .iter()
                .filter(|e| e.satellite == *mission && (storm.t0 - e.mjd).abs() < 4.0)
                .collect();
            if rows.is_empty() {
                continue;
            }
            println!(
                "  {mission}  n {} in the storm's own windows\n    \
                 model             scale   <O/C>  StD %   R      n  \
                 onset h  peak M/O  d(half) h",
                rows.len()
            );
            for &model in Model::ALL {
                let Some(v) = assess(&rows, storm, model) else {
                    println!("    {:<18} rejected: a phase has no data", model.name());
                    continue;
                };
                println!(
                    "    {:<18}{:6.3}{:8.3}{:7.1}{:7.3}{:7}{:8.1}{:10.3}{:9.1}",
                    model.name(),
                    v.scale,
                    v.whole.mean_ratio,
                    v.whole.scatter,
                    v.whole.correlation,
                    v.whole.n,
                    v.onset_error,
                    v.peak_ratio,
                    v.recovery_error
                );
                verdicts
                    .entry((mission, model.index()))
                    .or_default()
                    .push(v);
            }
        }
    }

    the_storm_average_per_satellite(&verdicts);
    the_timing_verdict(&verdicts);

    // The protocol itself has to have produced something for every model, or the table above is a
    // statement about the reader rather than about the models.
    assert!(!covered.is_empty(), "no storm in the corpus reached ap 80");
    assert_eq!(
        verdicts.len() % Model::ALL.len(),
        0,
        "a model was assessed on a different set of storms from the others"
    );
    for ((mission, index), set) in &verdicts {
        for v in set {
            assert!(
                v.whole.mean_ratio.is_finite() && v.whole.mean_ratio > 0.0,
                "{mission} {} produced {:?}",
                Model::ALL[*index].name(),
                v.whole
            );
            assert!((0.0..=200.0).contains(&v.whole.scatter));
            assert!((-1.0..=1.0).contains(&v.whole.correlation));
        }
    }
}

/// The paper's own four columns, averaged over storms per satellite and model -- the shape its
/// Table 4 has, so that the two can be read side by side.
fn the_storm_average_per_satellite(verdicts: &BTreeMap<(&'static str, usize), Vec<StormVerdict>>) {
    println!("\n== averaged over storms, per satellite: the paper's own four columns ==");
    println!(
        "satellite / model                  storms  <O/C>  StD<O/C> %  mean StD %  mean R  \
         P2 <O/C>  P3 <O/C>  P4 <O/C>"
    );
    let geometric = |xs: &[f64]| -> f64 {
        libm::exp(xs.iter().map(|x| libm::log(*x)).sum::<f64>() / xs.len() as f64)
    };
    let mean = |xs: &[f64]| -> f64 { xs.iter().sum::<f64>() / xs.len() as f64 };
    for ((mission, index), set) in verdicts {
        let ratios: Vec<f64> = set.iter().map(|v| v.whole.mean_ratio).collect();
        let centre = geometric(&ratios);
        let spread = 100.0
            * (ratios
                .iter()
                .map(|r| (libm::log(r / centre)).powi(2))
                .sum::<f64>()
                / ratios.len() as f64)
                .sqrt();
        let phase = |k: usize| {
            geometric(
                &set.iter()
                    .map(|v| v.per_phase[k].mean_ratio)
                    .collect::<Vec<_>>(),
            )
        };
        let scatter = mean(&set.iter().map(|v| v.whole.scatter).collect::<Vec<_>>());
        let correlation = mean(&set.iter().map(|v| v.whole.correlation).collect::<Vec<_>>());
        println!(
            "{:<24} {:<12}{:4}{:8.3}{:11.1}{:12.1}{:8.3}{:10.3}{:10.3}{:10.3}",
            mission.split(' ').next().unwrap_or(mission),
            Model::ALL[*index].name(),
            set.len(),
            centre,
            spread,
            scatter,
            correlation,
            phase(1),
            phase(2),
            phase(3)
        );
        // Debiasing is what makes the five comparable, and the measured spread of these aggregates is
        // narrow: 0.967 to 1.050 in the centre and 0.830 to 0.898 in the correlation (§10.14). A model
        // outside these bands is not a worse model, it is a driver assembled wrong -- which is the
        // failure this campaign exists to catch, and the reason the bounds are this loose.
        let name = Model::ALL[*index].name();
        assert!(
            (0.85..=1.20).contains(&centre),
            "{mission} {name} storm <O/C> {centre:.3}"
        );
        assert!(
            correlation > 0.70,
            "{mission} {name} storm R {correlation:.3}"
        );
        assert!(
            scatter < 40.0,
            "{mission} {name} storm scatter {scatter:.1}%"
        );
    }
}

/// The three timing numbers `plans/16` Ф15.H states its verdict in.
fn the_timing_verdict(verdicts: &BTreeMap<(&'static str, usize), Vec<StormVerdict>>) {
    let mean = |xs: &[f64]| -> f64 { xs.iter().sum::<f64>() / xs.len() as f64 };
    println!(
        "\n== the timing verdict Ф15.H asks for: onset +/-3 h, recovery +/-6 h, peak within 25 % =="
    );
    println!(
        "model               cases  |onset| h  worst  |peak M/O - 1| %  worst  |d(half)| h  worst"
    );
    for &model in Model::ALL {
        let set: Vec<&StormVerdict> = verdicts
            .iter()
            .filter(|((_, index), _)| *index == model.index())
            .flat_map(|(_, v)| v.iter())
            .collect();
        if set.is_empty() {
            continue;
        }
        let onset: Vec<f64> = set.iter().map(|v| v.onset_error.abs()).collect();
        let peak: Vec<f64> = set
            .iter()
            .map(|v| 100.0 * (v.peak_ratio - 1.0).abs())
            .collect();
        let tau: Vec<f64> = set
            .iter()
            .map(|v| v.recovery_error.abs())
            .filter(|x| x.is_finite())
            .collect();
        let worst = |xs: &[f64]| xs.iter().copied().fold(0.0_f64, f64::max);
        println!(
            "{:<20}{:4}{:10.1}{:8.1}{:17.1}{:8.1}{:12.1}{:8.1}",
            model.name(),
            set.len(),
            mean(&onset),
            worst(&onset),
            mean(&peak),
            worst(&peak),
            if tau.is_empty() { f64::NAN } else { mean(&tau) },
            worst(&tau)
        );
    }
}

/// The storm detection and its phase boundaries, on an `ap` series written by hand.
///
/// Everything the assessment reports hangs off `t0`, and `t0` is the one quantity in it that comes
/// from a search rather than from a formula. A single interval's error moves every phase boundary by
/// three hours, which is the size of the verdict's own tolerance.
#[test]
fn the_storm_phases_are_anchored_on_the_first_peak_of_ap() {
    // Four quiet days, then a rise to 200 on the third interval of day five, then quiet again.
    let mut days: Vec<SpaceWeatherDay> = (0..10)
        .map(|k| {
            let base = space_weather::parse_space_weather(CELESTRAK)[0];
            let (year, month, day) = (2024, 5, 1 + k);
            SpaceWeatherDay {
                year,
                month,
                day,
                ap: 5.0,
                ap3h: [5.0; 8],
                ..base
            }
        })
        .collect();
    days[4].ap3h = [5.0, 40.0, 200.0, 120.0, 90.0, 20.0, 5.0, 5.0];

    let found = storms(&days);
    assert_eq!(found.len(), 1, "{found:?}");
    let storm = found[0];
    assert!((storm.peak_ap - 200.0).abs() < 1e-12);
    assert!(!storm.multiple_peak, "one run, one peak");
    assert!((storm.phase3_end - 36.0).abs() < 1e-12);
    // t0 is the centre of 06-09 UT on 2024-05-05, the first interval at or above 80 that is also a
    // local maximum -- not the 03-06 interval, which is below the threshold, and not the 09-12 one.
    let day = MJD_AT_UNIX_EPOCH + days_since_1970(2024, 5, 5) as f64;
    assert!(
        (storm.t0 - (day + 7.5 / 24.0)).abs() < 1e-9,
        "t0 is {} h into the day",
        (storm.t0 - day) * 24.0
    );
    // The four windows, at their own boundaries: closed below, open above, and contiguous from
    // t0 - 30 h to t0 + 48 h with a twelve-hour gap between phases 1 and 2 by construction.
    for (which, (from, to)) in [
        (1, (-30.0, -18.0)),
        (2, (-18.0, 0.0)),
        (3, (0.0, 36.0)),
        (4, (36.0, 48.0)),
    ] {
        assert!(
            storm.contains(which, storm.t0 + from / 24.0),
            "phase {which} excludes its own start"
        );
        assert!(
            !storm.contains(which, storm.t0 + to / 24.0),
            "phase {which} includes its own end"
        );
        assert_eq!(storm.phase(which), (from, to));
    }
    assert!(!storm.contains(1, storm.t0 - 31.0 / 24.0));
    assert!(!storm.contains(4, storm.t0 + 49.0 / 24.0));

    // A second peak 12 h after the first makes it a multiple-peak storm and extends phase 3.
    days[4].ap3h = [5.0, 40.0, 200.0, 120.0, 20.0, 100.0, 5.0, 5.0];
    let multiple = storms(&days);
    assert_eq!(
        multiple.len(),
        1,
        "a second peak inside the window is the same storm"
    );
    assert!(multiple[0].multiple_peak);
    // The last interval at or above 80 is 15-18 UT, centred 9 h after t0, so phase 3 ends at +45 h.
    assert!(
        (multiple[0].phase3_end - 45.0).abs() < 1e-9,
        "phase 3 ends at t0+{} h",
        multiple[0].phase3_end
    );
    assert_eq!(multiple[0].phase(4), (45.0, 57.0));

    // And a quiet feed has no storms at all rather than one of zero length.
    let quiet: Vec<SpaceWeatherDay> = days
        .iter()
        .map(|d| SpaceWeatherDay {
            ap3h: [5.0; 8],
            ..*d
        })
        .collect();
    assert_eq!(storms(&quiet), [] as [Storm; 0]);
}

/// The full stratified table over everything `tools/fetch-density-data.sh` downloads.
///
/// `#[ignore]`d: it reads ~230 MB that is deliberately not vendored. Run with
/// `cargo test -p spacelab-propagation --test measured_density -- --ignored --nocapture`.
#[test]
#[ignore = "needs the tools/fetch-density-data.sh download in data/validation/density"]
fn every_downloaded_month_stratified_by_flux_and_storm_level() {
    // One row in six: the products are 10 s and 30 s cadence and the thermosphere carries no
    // independent information at that rate. Stated rather than silent — it is a 6× reduction in
    // every sample count below.
    const DECIMATION: usize = 6;

    let dir = Path::new("../../data/validation/density");
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();
    let series = DriverSeries {
        days: space_weather::parse_space_weather(&read("SW-All.csv")),
    };
    let d = series.view();

    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains("_DNS_"))
                && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt"))
        })
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no density products in {}",
        dir.display()
    );

    let mut all: Vec<Evaluation> = Vec::new();
    // Kept per file as well as pooled: a product file *is* one satellite-month, which is the unit the
    // held-out split below has to be taken in.
    let mut arcs: Vec<(String, Vec<Evaluation>)> = Vec::new();
    println!("== per product file, for comparison with tools/orekit-oracle/DensityOracle.java ==");
    print!("{:<28} {:>7}", "file", "n");
    for &m in Model::ALL {
        print!("  {:>17}", m.name());
    }
    println!();
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let satellite = satellite_of(&name);
        let observations = parse_density(&text);
        assert!(
            !observations.is_empty(),
            "{name} parsed to nothing — the Swarm product is nine columns wide, not twelve"
        );
        let one: Vec<Evaluation> = observations
            .iter()
            .step_by(DECIMATION)
            .filter_map(|o| evaluate(o, satellite, d))
            .collect();
        print!("{name:<28} {:>7}", one.len());
        for &m in Model::ALL {
            let f = fit_of(m, &one);
            print!("  {:>7.3} ±{:<8.3}", f.bias, f.rms);
        }
        println!();
        all.extend(one.iter().copied());
        arcs.push((name, one));
    }
    println!("\ntotal evaluations: {}", all.len());

    let table = |title: &str, key: &dyn Fn(&Evaluation) -> String| {
        let mut bins: BTreeMap<String, Vec<Evaluation>> = BTreeMap::new();
        for e in &all {
            bins.entry(key(e)).or_default().push(*e);
        }
        println!("\n== {title} ==");
        print!("{:<30} {:>8}", "bin", "n");
        for &m in Model::ALL {
            print!("  {:>17}", m.name());
        }
        println!();
        for (name, set) in bins {
            print!("{:<30} {:>8}", name, set.len());
            for &m in Model::ALL {
                let f = fit_of(m, &set);
                print!("  {:>7.3} ±{:<8.3}", f.bias, f.rms);
            }
            println!();
        }
    };

    table("by solar flux", &|e| flux_bin(e.f107).to_string());
    table("by geomagnetic level", &|e| storm_bin(e.ap).to_string());
    table("by altitude", &|e| {
        let lower = (e.altitude_km / 20.0).floor() * 20.0;
        format!("{lower:.0}-{:.0} km", lower + 20.0)
    });
    table("by satellite", &|e| e.satellite.to_string());
    table("by flux x geomagnetic", &|e| {
        format!("{} / {}", flux_bin(e.f107), storm_bin(e.ap))
    });

    mape_against_the_published_band(&all);
    held_out_by_satellite_month(&arcs);
    the_interval_covers_every_regime(&all);
    the_storm_assessment(&all, d.days);
}

/// The shipped interval's coverage, per regime, over the whole campaign.
///
/// The fixture-day test asserts that the interval does not under-cover on two days; this asserts it for
/// every cell of the table over 348 693 evaluations, which is the only way a single wrong `σ` would
/// show. It over-covers throughout and is expected to: the residual here is *in sample* — the fit saw
/// all twelve months — while `σ` was measured on a month held out (`docs/verification.md` §10.12).
fn the_interval_covers_every_regime(all: &[Evaluation]) {
    println!("\n== the shipped 90 % interval, by regime ==");
    let mut cells: BTreeMap<usize, Vec<(f64, f64, SdtmDrivers)>> = BTreeMap::new();
    for e in all {
        cells
            .entry(DensityUncertainty::regime(&e.sdtm))
            .or_default()
            .push((e.modelled[Model::Sdtm1.index()], e.measured, e.sdtm));
    }
    for (cell, rows) in &cells {
        let covered = coefficients::UNCERTAINTY
            .coverage(rows, sdtm::NORMAL_90)
            .expect("a non-empty cell");
        println!(
            "regime {cell}  n {:>7}  sigma {:.3}  covers {:5.1}%",
            rows.len(),
            coefficients::UNCERTAINTY.table()[*cell],
            100.0 * covered
        );
        assert!(
            covered >= 0.90,
            "regime {cell} under-covers at {:.1}%",
            100.0 * covered
        );
    }
    let pooled: Vec<(f64, f64, SdtmDrivers)> = cells.values().flatten().copied().collect();
    let covered = coefficients::UNCERTAINTY
        .coverage(&pooled, sdtm::NORMAL_90)
        .expect("a non-empty campaign");
    println!(
        "pooled    n {:>7}                covers {:5.1}%",
        pooled.len(),
        100.0 * covered
    );
    // Every regime the table has must be exercised by twelve months of two satellites, or the table
    // carries a number nothing has ever checked.
    assert_eq!(cells.len(), sdtm::REGIMES, "some regime has no evaluations");
}

/// The same rows on the metric the literature reports, so the campaign can be set beside a published
/// bar instead of only beside itself.
///
/// Acciarini et al., *Space Weather* **22**, e2023SW003652 (2024) report 40–60 % MAPE for empirical
/// models and ~20 % for a machine-learned one, on POD densities from the same satellite family.
/// NRLMSISE-00 landing inside their empirical band is the check that this campaign measures what
/// theirs measured; DTM2020 reaching their ML figure with no learned component is the finding, and it
/// is not independent evidence, because DTM2020 was fitted on this data family.
fn mape_against_the_published_band(all: &[Evaluation]) {
    println!("\n== mean absolute percentage error ==");
    print!("{:<30} {:>8}", "bin", "n");
    for &m in Model::ALL {
        print!("  {:>17}", m.name());
    }
    println!();
    let mape_row = |name: &str, set: &[Evaluation]| {
        print!("{:<30} {:>8}", name, set.len());
        for &m in Model::ALL {
            print!("  {:>16.1}%", fit_of(m, set).mape);
        }
        println!();
    };
    mape_row("all", all);
    for (label, key) in [
        ("flux", &flux_bin as &dyn Fn(f64) -> &'static str),
        ("geomag", &storm_bin),
    ] {
        let mut bins: BTreeMap<&'static str, Vec<Evaluation>> = BTreeMap::new();
        for e in all {
            let v = if label == "flux" { e.f107 } else { e.ap };
            bins.entry(key(v)).or_default().push(*e);
        }
        for (name, set) in bins {
            mape_row(name, &set);
        }
    }

    let pooled = |m| fit_of(m, all).mape;
    assert!(
        (40.0..60.0).contains(&pooled(Model::Nrlmsise00StormArray)),
        "NRLMSISE-00 pooled MAPE {:.1}% is outside the published empirical band",
        pooled(Model::Nrlmsise00StormArray)
    );
}

/// Out of sample, by satellite-month: the correction is fitted on every arc but one and scored on the
/// one held back, so the number is extrapolation to an arc the fit never saw rather than
/// interpolation within it.
///
/// SDTM-1 is in this loop for a reason that is not the others': it carries sun-fixed harmonics of its
/// own, so how much a four-coefficient sun-fixed correction can still add is a measure of whether those
/// harmonics did their job. What is held out here is the *correction's* fit, not the model's — SDTM-1's
/// own coefficients saw every arc (`docs/verification.md` §10.10). Rows from one pass are not independent of each other, which is why the
/// split is by arc and not by row — and the arc is the product file, because a block cut at a fixed
/// row count would straddle two files and leave each half of a month training on the other.
fn held_out_by_satellite_month(arcs: &[(String, Vec<Evaluation>)]) {
    println!(
        "\n== held-out calibration, MAPE (leave-one-out over {} satellite-months) ==",
        arcs.len()
    );
    for m in [Model::Nrlmsise00StormArray, Model::Sdtm1] {
        let (mut base_abs, mut cal_abs, mut n) = (0.0_f64, 0.0_f64, 0usize);
        for (held, rows) in arcs {
            let train: Vec<DensitySample> = arcs
                .iter()
                .filter(|(k, _)| k != held)
                .flat_map(|(_, v)| v.iter())
                .map(|e| DensitySample {
                    latitude: e.latitude,
                    local_solar_time: e.local_solar_time,
                    log_ratio: (1.0 / e.ratio(m)).ln(),
                    weight: 1.0,
                })
                .collect();
            let Some(correction) = DensityCorrection::<4>::fit_ridge(&train, 1e-3) else {
                continue;
            };
            for e in rows {
                let r = e.ratio(m);
                base_abs += (r - 1.0).abs();
                cal_abs += (r * correction.factor(e.latitude, e.local_solar_time) - 1.0).abs();
                n += 1;
            }
        }
        assert!(n > 0, "{} had no held-out rows", m.name());
        let (base, calibrated) = (100.0 * base_abs / n as f64, 100.0 * cal_abs / n as f64);
        println!(
            "{:<17} n {n:>7}  base {:5.1}%  ->  calibrated {:5.1}%",
            m.name(),
            base,
            calibrated
        );
        // Four sun-fixed coefficients carry a model whose error has sun-fixed structure and cannot
        // carry one whose error does not. The gate is therefore asymmetric: a large gain is required
        // where the structure exists, and everywhere else only that the correction does not make the
        // model worse by more than it could plausibly be worth.
        if m == Model::Nrlmsise00StormArray {
            assert!(
                calibrated < 0.65 * base,
                "the correction should take a third off NRLMSISE-00: {base:.1}% -> {calibrated:.1}%"
            );
        } else {
            assert!(
                calibrated < base + 2.0,
                "{} got worse out of sample: {base:.1}% -> {calibrated:.1}%",
                m.name()
            );
        }
    }
}

/// The satellite-months both the operational WAM-IPE archive (from 2023-07-01) and the density
/// download cover.
const WAM_IPE_MONTHS: [&str; 2] = ["2024_05", "2025_06"];
/// GFS's Earth radius (`con_rerth`), the sphere WAM-IPE's heights are measured from if they are not
/// measured from the ellipsoid.
const WAM_SPHERE_KM: f64 = 6371.2;
const WGS84_A_KM: f64 = 6378.137;
/// The base models the nowcast is fitted over: the two public empirical models, and ours.
const WAM_BASE_LIST: &[Model] = &[Model::Nrlmsise00StormArray, Model::Sdtm1];
const WAM_BASE_COUNT: usize = WAM_BASE_LIST.len();
/// [`WAM_BASE_LIST`] as an array, so that a per-base result is a fixed-size value.
const WAM_BASES: [Model; WAM_BASE_COUNT] = {
    let mut bases = [Model::Sdtm1; WAM_BASE_COUNT];
    let mut i = 0;
    while i < WAM_BASE_COUNT {
        bases[i] = WAM_BASE_LIST[i];
        i += 1;
    }
    bases
};
/// Every other ring and longitude of the 2° × 4° grid: 45 × 45 nodes for at most 25 coefficients.
const WAM_STRIDE: usize = 2;
/// The trailing window an hour's correction is taken relative to in the anomaly columns, and the
/// fewest fitted hours it must hold. Chosen before the first run, not after it: three days is
/// longer than a storm's main phase and short enough that the base model's own drift inside it is
/// small.
const WAM_ANOMALY_HOURS: i64 = 72;
const WAM_ANOMALY_MIN_HOURS: usize = 48;

/// The two readings of WAM-IPE's undocumented height coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WamHeight {
    /// Geodetic height over WGS84, with geodetic latitude.
    Geodetic,
    /// Radius minus [`WAM_SPHERE_KM`], with geocentric latitude.
    Spherical,
}

impl WamHeight {
    const BOTH: [Self; 2] = [Self::Geodetic, Self::Spherical];

    const fn name(self) -> &'static str {
        match self {
            Self::Geodetic => "geodetic",
            Self::Spherical => "spherical",
        }
    }

    /// A geodetic point's `(height km, latitude °)` in the product's coordinates.
    fn of(self, altitude_km: f64, latitude: f64) -> (f64, f64) {
        match self {
            Self::Geodetic => (altitude_km, latitude.to_degrees()),
            Self::Spherical => {
                let (r, psi) = spherical_from_geodetic(altitude_km, latitude);
                (r - WAM_SPHERE_KM, psi.to_degrees())
            }
        }
    }
}

/// `(radius km, geocentric latitude rad)` of a WGS84 geodetic point.
fn spherical_from_geodetic(altitude_km: f64, latitude: f64) -> (f64, f64) {
    let e2 = FLATTENING * (2.0 - FLATTENING);
    let n = WGS84_A_KM / (1.0 - e2 * latitude.sin().powi(2)).sqrt();
    let p = (n + altitude_km) * latitude.cos();
    let z = (n * (1.0 - e2) + altitude_km) * latitude.sin();
    (p.hypot(z), z.atan2(p))
}

/// Civil date of a day number since 1970-01-01 (Howard Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (
        i32::try_from(year).unwrap(),
        u32::try_from(month).unwrap(),
        u32::try_from(day).unwrap(),
    )
}

fn unix_seconds(e: &Evaluation) -> f64 {
    (e.mjd - MJD_AT_UNIX_EPOCH) * 86_400.0
}

/// Mean local solar time, `UT + λ/15`, hours — the convention the calibration layer applies a
/// correction in, which the product's own column need not share.
fn mean_solar_time(unix: f64, longitude_deg: f64) -> f64 {
    (unix.rem_euclid(86_400.0) / 3_600.0 + longitude_deg / 15.0).rem_euclid(24.0)
}

/// A grid node dressed as an observation, so the base models are evaluated there through the same
/// [`evaluate`] every measured row goes through.
fn node_observation(
    unix: f64,
    altitude_km: f64,
    latitude: f64,
    longitude_deg: f64,
) -> DensityObservation {
    let day = unix.div_euclid(86_400.0);
    let (year, month, d) = civil_from_days(day as i64);
    DensityObservation {
        year,
        month,
        day: d,
        seconds_of_day: unix - day * 86_400.0,
        time_scale: TimeScale::Utc,
        altitude: altitude_km * 1000.0,
        longitude: longitude_deg.to_radians(),
        latitude,
        local_solar_time: mean_solar_time(unix, longitude_deg),
        density: 1.0,
        orbit_mean_density: None,
        nominal: true,
    }
}

/// The pinned nowcast files, by their valid hour in Unix seconds.
fn wam_ipe_hours() -> BTreeMap<i64, PathBuf> {
    let dir = std::env::var_os("WAM_IPE_DIR").map_or_else(
        || PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/spacelab/wam-ipe"),
        PathBuf::from,
    );
    let list = std::fs::read_to_string("../../tools/wam-ipe.sha256").unwrap();
    list.lines()
        .map(|line| {
            let key = line.split_whitespace().nth(1).unwrap();
            let stamp = key.rsplit('.').nth(1).unwrap();
            let num = |a: usize, b: usize| stamp[a..b].parse::<u32>().unwrap();
            let days = days_since_1970(i32::try_from(num(0, 4)).unwrap(), num(4, 6), num(6, 8));
            let seconds = days * 86_400 + i64::from(num(9, 11) * 3600 + num(11, 13) * 60);
            (seconds, dir.join(key))
        })
        .collect()
}

/// One fitted correction set: a scale, degree 2 and degree 4.
#[derive(Clone, Copy, Debug)]
struct Corrections {
    scale: DensityCorrection<1>,
    degree2: DensityCorrection<9>,
    degree4: DensityCorrection<25>,
}

impl Corrections {
    const DEGREES: [&str; 3] = ["scale", "degree 2", "degree 4"];

    fn fit(samples: &[DensitySample]) -> Self {
        Self {
            scale: DensityCorrection::fit(samples).unwrap(),
            degree2: DensityCorrection::fit(samples).unwrap(),
            degree4: DensityCorrection::fit(samples).unwrap(),
        }
    }

    fn log(&self, degree: usize, latitude_deg: f64, lst: f64) -> f64 {
        let lat = latitude_deg.to_radians();
        match degree {
            0 => self.scale.log_correction(lat, lst),
            1 => self.degree2.log_correction(lat, lst),
            _ => self.degree4.log_correction(lat, lst),
        }
    }

    /// This correction less the mean of `trailing`'s, coefficient by coefficient: what WAM-IPE
    /// says has *changed* against the base model, with whatever it says always cancelled.
    fn anomaly(&self, trailing: &[&Self]) -> Self {
        fn less_mean<const K: usize>(
            c: &DensityCorrection<K>,
            all: impl Iterator<Item = [f64; K]>,
            n: f64,
        ) -> DensityCorrection<K> {
            let mut out = *c.coefficients();
            for coefficients in all {
                for (o, x) in out.iter_mut().zip(coefficients) {
                    *o -= x / n;
                }
            }
            DensityCorrection::from_coefficients(out)
        }
        let n = trailing.len() as f64;
        Self {
            scale: less_mean(
                &self.scale,
                trailing.iter().map(|t| *t.scale.coefficients()),
                n,
            ),
            degree2: less_mean(
                &self.degree2,
                trailing.iter().map(|t| *t.degree2.coefficients()),
                n,
            ),
            degree4: less_mean(
                &self.degree4,
                trailing.iter().map(|t| *t.degree4.coefficients()),
                n,
            ),
        }
    }
}

/// What one hourly field contributes: `ln ρ` at every observation within an hour of it, under both
/// height readings, and per satellite the corrections fitted on the shell it flew through — under
/// the geodetic reading only, which is the one the latitude bands select.
struct HourResult {
    hour: i64,
    /// `(satellite, row, [ln ρ geodetic, ln ρ spherical])`.
    values: Vec<(usize, usize, [f64; 2])>,
    /// `(satellite, one per base)`.
    fits: Vec<(usize, [Corrections; WAM_BASE_COUNT])>,
}

fn wam_ipe_hour(
    hour: i64,
    path: &Path,
    satellites: &[Vec<Evaluation>],
    d: Drivers<'_>,
) -> HourResult {
    let field = WamIpeField::parse(&std::fs::read(path).unwrap()).unwrap();
    if field.holes() > 0 {
        println!("{}: {} holes", path.display(), field.holes());
    }
    assert_eq!(
        field.unix_seconds(),
        hour as f64,
        "{} is not valid at its hour",
        path.display()
    );
    let t = hour as f64;
    let stride = NonZeroUsize::new(WAM_STRIDE).unwrap();
    let mut values = Vec::new();
    let mut fits = Vec::new();
    for (s, rows) in satellites.iter().enumerate() {
        let from = rows.partition_point(|e| unix_seconds(e) <= t - 3600.0);
        let to = rows.partition_point(|e| unix_seconds(e) < t + 3600.0);
        for (n, e) in rows[from..to].iter().enumerate() {
            let ln = WamHeight::BOTH.map(|c| {
                let (h, lat) = c.of(e.altitude_km, e.latitude);
                field
                    .density(h, lat, e.longitude.to_degrees())
                    .unwrap()
                    .ln()
            });
            values.push((s, from + n, ln));
        }
        let near: Vec<&Evaluation> = rows[from..to]
            .iter()
            .filter(|e| (unix_seconds(e) - t).abs() <= 1800.0)
            .collect();
        if near.is_empty() {
            continue;
        }
        let shell = near.iter().map(|e| e.altitude_km).sum::<f64>() / near.len() as f64;
        // Each node is evaluated once for all three bases.
        let mut cache: HashMap<(i64, i64), Option<Evaluation>> = HashMap::new();
        let per_base = WAM_BASES.map(|m| {
            let samples = field.calibration_samples(shell, stride, |lat, lon| {
                let key = ((lat * 8.0).round() as i64, (lon * 8.0).round() as i64);
                cache
                    .entry(key)
                    .or_insert_with(|| {
                        let node = node_observation(t, shell, lat.to_radians(), lon);
                        evaluate(&node, "", d)
                    })
                    .map_or(f64::NAN, |x| x.modelled[m.index()])
            });
            Corrections::fit(&samples)
        });
        fits.push((s, per_base));
    }
    HourResult { hour, values, fits }
}

/// The driver rows within `before` days ahead of and five days after a month, so that the linear
/// searches every evaluation makes run over a season rather than seventy years.
fn drivers_near(year: i32, month: u32, before: i64, all: Drivers<'_>) -> DriverSeries {
    let start = days_since_1970(year, month, 1) - before;
    let end = days_since_1970(year, month, 1) + 36;
    let inside = |d: i64| (start..end).contains(&d);
    DriverSeries {
        days: all
            .days
            .iter()
            .filter(|d| inside(days_since_1970(d.year, d.month, d.day)))
            .copied()
            .collect(),
    }
}

/// The columns every WAM-IPE table reports, in the order [`wam_ipe_row`] fills them: the bases,
/// WAM-IPE under each height reading, then each base corrected to WAM-IPE's level and to its
/// anomaly, at each degree.
fn wam_ipe_columns() -> Vec<String> {
    let mut columns: Vec<String> = WAM_BASES.iter().map(|m| m.name().to_string()).collect();
    for c in WamHeight::BOTH {
        columns.push(format!("WAM-IPE, {} height", c.name()));
    }
    for kind in ["level", "anomaly"] {
        for m in WAM_BASES {
            for d in Corrections::DEGREES {
                columns.push(format!("{} + WAM-IPE {kind}, {d}", m.name()));
            }
        }
    }
    columns
}

/// One hour's contribution to a row: `ln ρ` under both readings, and per base the correction to
/// WAM-IPE's level and to its anomaly.
type HourValues<'a> = (
    [f64; 2],
    &'a [Corrections; WAM_BASE_COUNT],
    &'a [Corrections; WAM_BASE_COUNT],
);

/// One observation's `ρ/ρ_measured` in every column, a fraction `w` of the way from the hour
/// `before` to the hour `after`.
fn wam_ipe_row(e: &Evaluation, w: f64, before: HourValues<'_>, after: HourValues<'_>) -> Vec<f64> {
    let t = unix_seconds(e);
    let mut row: Vec<f64> = WAM_BASES.iter().map(|&m| e.ratio(m)).collect();
    for c in 0..2 {
        row.push(((1.0 - w) * before.0[c] + w * after.0[c]).exp() / e.measured);
    }
    let lst = mean_solar_time(t, e.longitude.to_degrees());
    let latitude = e.latitude.to_degrees();
    for (b0, b1) in [(before.1, after.1), (before.2, after.2)] {
        for (b, &m) in WAM_BASES.iter().enumerate() {
            for d in 0..Corrections::DEGREES.len() {
                let log = (1.0 - w) * b0[b].log(d, latitude, lst) + w * b1[b].log(d, latitude, lst);
                row.push(e.ratio(m) * log.exp());
            }
        }
    }
    row
}

/// Every hour of a month, split across the machine's threads.
fn wam_ipe_month_hours(
    month_hours: &[(i64, &PathBuf)],
    satellites: &[Vec<Evaluation>],
    d: Drivers<'_>,
) -> Vec<HourResult> {
    let threads = std::thread::available_parallelism().map_or(4, NonZeroUsize::get);
    let chunk = month_hours.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = month_hours
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|(h, p)| wam_ipe_hour(*h, p, satellites, d))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
}

/// Hours from the run's cycle time to the hour a pinned file is valid at, from its key.
fn wam_ipe_lead(hour: i64, path: &Path) -> i64 {
    let key = path.to_string_lossy();
    let at = key.find("wfs.").unwrap() + 4;
    let num = |a: usize, b: usize| key[at + a..at + b].parse::<u32>().unwrap();
    let cycle = days_since_1970(i32::try_from(num(0, 4)).unwrap(), num(4, 6), num(6, 8)) * 86_400
        + i64::from(num(9, 11)) * 3600;
    (hour - cycle) / 3600
}

/// Score one satellite-month pair, adding each row's ratios under its satellite-month, under
/// `all`, under its geomagnetic level, its latitude band and the lead of the run it came from.
fn score_wam_ipe_month(
    month: &str,
    dir: &Path,
    all: Drivers<'_>,
    hours: &BTreeMap<i64, PathBuf>,
    ratios: &mut BTreeMap<(String, usize), Vec<f64>>,
) {
    // One row in six, as in the stratified campaign.
    const DECIMATION: usize = 6;
    let year: i32 = month[..4].parse().unwrap();
    let mon: u32 = month[5..].parse().unwrap();
    let near = drivers_near(year, mon, 120, all);
    let d = near.view();
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("txt")))
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .filter(|n| n.contains("_DNS_") && n.contains(month))
        .collect();
    names.sort();
    assert_eq!(
        names.len(),
        2,
        "GRACE-FO C and Swarm A for {month}: {names:?}"
    );
    let satellites: Vec<Vec<Evaluation>> = names
        .iter()
        .map(|n| {
            parse_density(&std::fs::read_to_string(dir.join(n)).unwrap())
                .iter()
                .step_by(DECIMATION)
                .filter_map(|o| evaluate(o, satellite_of(n), d))
                .collect()
        })
        .collect();

    let month_start = days_since_1970(year, mon, 1) * 86_400;
    let month_end = days_since_1970(year + i32::from(mon == 12), mon % 12 + 1, 1) * 86_400;
    let month_hours: Vec<(i64, &PathBuf)> = hours
        .range(month_start..=month_end)
        .map(|(h, p)| (*h, p))
        .collect();
    let mut at_hour: HashMap<(usize, usize, i64), [f64; 2]> = HashMap::new();
    let mut fitted: HashMap<(usize, i64), [Corrections; WAM_BASE_COUNT]> = HashMap::new();
    for r in wam_ipe_month_hours(&month_hours, &satellites, d) {
        for (s, n, v) in r.values {
            at_hour.insert((s, n, r.hour), v);
        }
        for (s, f) in r.fits {
            fitted.insert((s, r.hour), f);
        }
    }

    // Each hour's correction less the mean of those fitted in the trailing window before it, where
    // the window holds enough of them.
    let mut anomaly: HashMap<(usize, i64), [Corrections; WAM_BASE_COUNT]> = HashMap::new();
    for (&(s, hour), now) in &fitted {
        let trailing: Vec<&[Corrections; WAM_BASE_COUNT]> = (1..=WAM_ANOMALY_HOURS)
            .filter_map(|k| fitted.get(&(s, hour - k * 3600)))
            .collect();
        if trailing.len() >= WAM_ANOMALY_MIN_HOURS {
            let per_base = std::array::from_fn(|b| {
                now[b].anomaly(&trailing.iter().map(|t| &t[b]).collect::<Vec<_>>())
            });
            anomaly.insert((s, hour), per_base);
        }
    }

    for (s, rows) in satellites.iter().enumerate() {
        let group = format!("{} {month}", &names[s][..2]);
        let (mut scored, mut unscored) = (0usize, 0usize);
        for (n, e) in rows.iter().enumerate() {
            let t = unix_seconds(e);
            let h0 = (t / 3600.0).floor() as i64 * 3600;
            let hour = |h: i64| {
                Some((
                    *at_hour.get(&(s, n, h))?,
                    fitted.get(&(s, h))?,
                    anomaly.get(&(s, h))?,
                ))
            };
            let (Some(before), Some(after)) = (hour(h0), hour(h0 + 3600)) else {
                unscored += 1;
                continue;
            };
            scored += 1;
            let row = wam_ipe_row(e, (t - h0 as f64) / 3600.0, before, after);
            let storm = if e.ap >= 50.0 { "Ap >= 50" } else { "Ap < 50" };
            let band = match e.latitude.to_degrees().abs() {
                x if x < 30.0 => "|latitude| 0-30",
                x if x < 60.0 => "|latitude| 30-60",
                _ => "|latitude| 60-90",
            };
            let lead = format!("lead {:+} h", wam_ipe_lead(h0, &hours[&h0]));
            for g in [group.as_str(), "all", storm, band, &lead] {
                for (k, r) in row.iter().enumerate() {
                    ratios.entry((g.to_string(), k)).or_default().push(*r);
                }
            }
        }
        println!(
            "{group}: {scored} rows scored, {unscored} without both hourly fields and their \
             trailing {WAM_ANOMALY_HOURS} h"
        );
    }
}

/// WAM-IPE, NOAA's public physics nowcast, against the same accelerometers — on its own, and as
/// the input the calibration layer fits a correction to over each base model (`plans/16` Ф18.B).
///
/// No density measurement enters any fit: each hour's correction is fitted to WAM-IPE's own field
/// on the shell the satellite flew through that hour, so every row scored here is held out from
/// everything fitted on it. SDTM-1's numbers carry their usual caveat — its coefficients saw these
/// two satellite-months (`docs/verification.md` §10.10).
#[test]
#[ignore = "needs tools/fetch-density-data.sh and tools/fetch-wam-ipe.sh (4.1 GB)"]
fn wam_ipe_against_the_accelerometers_alone_and_as_a_calibration_input() {
    let dir = Path::new("../../data/validation/density");
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();
    let series = DriverSeries {
        days: space_weather::parse_space_weather(&read("SW-All.csv")),
    };
    let hours = wam_ipe_hours();
    let mut ratios: BTreeMap<(String, usize), Vec<f64>> = BTreeMap::new();
    for month in WAM_IPE_MONTHS {
        score_wam_ipe_month(month, dir, series.view(), &hours, &mut ratios);
    }

    let mut groups: Vec<String> = ratios.keys().map(|(g, _)| g.clone()).collect();
    groups.dedup();
    for g in &groups {
        println!("\n== {g} ==");
        // The bands and leads are diagnostics of WAM-IPE itself, so they stop at its two columns.
        let shown = if g.starts_with('|') || g.starts_with("lead") {
            WAM_BASES.len() + 2
        } else {
            usize::MAX
        };
        for (k, name) in wam_ipe_columns().iter().enumerate().take(shown) {
            let f = fit(ratios[&(g.clone(), k)].iter().copied());
            println!(
                "{name:<44} n {:>6}  bias {:.3}  rms {:.3}  mape {:5.1}%",
                f.n, f.bias, f.rms, f.mape
            );
        }
    }
}
