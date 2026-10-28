//! Can SDTM-1 be run *now*, from files a caller is allowed to fetch, with no SET product in the loop?
//!
//! Everything else about this model is measured against a past month with the whole feed in hand.
//! Ф15.G asks the operational question instead, and it has three parts that fail separately:
//!
//! * **latency** — how far behind the present the last *measured* day is, which is the newest epoch a
//!   density can be computed for without forecasting anything;
//! * **the forecast baseline** — what happens past that day, and which of persistence and 27-day
//!   recurrence is worth using at which lead time;
//! * **the driver conventions** — whether the library assembles the three SDTM-1 drivers the way the
//!   fit did, which `space_weather::sdtm_drivers` now owns rather than each call site.
//!
//! The inputs are CelesTrak's `SW-All.csv` alone: F10.7, its trailing 81-day mean and the 3-hourly
//! `ap`. No SOLFSMY, no DTCFILE, no HASDM — the whole point of the exercise is that the operational
//! path stays inside the licences `docs/verification.md` §10.9 admitted.
//!
//! `docs/verification.md` §10.13 carries the numbers these tests print.

// The provider names (CelesTrak, NOAA, SET) are the domain vocabulary, as in the module under test.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    // At a lead of one rotation the two baselines sum identical values in identical order, so the
    // equality below is exact and asserting it to a tolerance would weaken the check.
    clippy::float_cmp
)]

use std::path::Path;

use spacelab_propagation::sdtm::{SdtmPoint, coefficients};
use spacelab_propagation::space_weather::{
    self, FLUX_WINDOW, ForecastMethod, SOLAR_ROTATION, SpaceWeatherDay,
};

const DIR: &str = "../../data/validation/density";

fn feed() -> Vec<SpaceWeatherDay> {
    let days = space_weather::parse_space_weather(
        &std::fs::read_to_string(Path::new(DIR).join("SW-All.csv")).unwrap(),
    );
    assert!(days.len() > 20_000, "the bulk file goes back to 1957");
    days
}

/// Day number since 1970-01-01 — the test's own copy, because the crate's is private and the
/// comparison it is used for here is exactly the one that must not be taken on trust.
fn ordinal(d: &SpaceWeatherDay) -> i64 {
    let y = i64::from(d.year) - i64::from(d.month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from(if d.month > 2 {
        d.month - 3
    } else {
        d.month + 9
    });
    let doy = (153 * mp + 2) / 5 + i64::from(d.day) - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

fn last_measured(days: &[SpaceWeatherDay]) -> usize {
    days.iter().rposition(|d| d.quality.is_measured()).unwrap()
}

/// The verdict Ф15.G asks for: a density at the newest epoch the feed supports, from one file.
#[test]
#[ignore = "needs tools/fetch-density-data.sh's SW-All.csv in data/validation/density"]
fn a_density_at_the_last_measured_day_from_one_fetched_file() {
    let days = feed();
    let last = last_measured(&days);
    let now = days[last];
    let newest = days.last().unwrap();
    let field = &coefficients::FIELD;

    println!("\n== latency, from the file alone ==");
    println!(
        "last measured day      {:04}-{:02}-{:02}  F10.7 {:.1}  trailing 81-day mean {:.1}  Ap {:.0}",
        now.year, now.month, now.day, now.f107_observed, now.f107_last81, now.ap
    );
    println!(
        "last row of any kind   {:04}-{:02}-{:02}  ({:?}), {} days further on",
        newest.year,
        newest.month,
        newest.day,
        newest.quality,
        ordinal(newest) - ordinal(&now)
    );

    // Latency measured against the file's own modification time rather than against the wall clock:
    // what is being measured is the *feed*, and a stale download would otherwise read as a slow
    // provider. Seconds since the epoch divided by 86 400 is the UTC day, which is what the feed's
    // dates are in.
    let fetched = std::fs::metadata(Path::new(DIR).join("SW-All.csv"))
        .and_then(|m| m.modified())
        .and_then(|t| {
            t.duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map_err(std::io::Error::other)
        })
        .map(|d| i64::try_from(d.as_secs() / 86_400).unwrap())
        .unwrap();
    let latency = fetched - ordinal(&now);
    println!("the feed's last measurement is {latency} day(s) behind the download itself");
    assert!(
        (0..=3).contains(&latency),
        "the last measured day is {latency} days behind the download, which is not an operational feed"
    );

    // 12:00 UT at the last measured day: the `ap` history ends at 09-12 UT, all of it published.
    let drivers = space_weather::sdtm_drivers::<{ coefficients::HISTORY }>(
        &days,
        field.storm(),
        now.year,
        now.month,
        now.day,
        12.0 * 3600.0,
    )
    .expect("the last measured day has a previous day and a full ap history");
    let doy = f64::from(
        u16::try_from(
            ordinal(&now)
                - ordinal(&SpaceWeatherDay {
                    month: 1,
                    day: 1,
                    ..now
                })
                + 1,
        )
        .unwrap(),
    );
    println!(
        "drivers                F10.7 {:.1} (yesterday)  F10.7a {:.1} (trailing)  storm {:.1} ap",
        drivers.f107, drivers.f107a, drivers.storm
    );

    println!("\n== SDTM-1 at noon local solar time, equator ==");
    for altitude in [300e3, 400e3, 500e3, 600e3] {
        let point = SdtmPoint {
            altitude,
            latitude: 0.0,
            local_solar_time: 12.0,
            day_of_year: doy,
        };
        let rho = field.density(&point, &drivers);
        let (low, high) = coefficients::UNCERTAINTY.interval(
            rho,
            &drivers,
            spacelab_propagation::sdtm::NORMAL_90,
        );
        println!(
            "{:>4.0} km  T_inf {:6.0} K  rho {rho:9.3e}  90 % interval {low:9.3e} .. {high:9.3e} kg/m^3",
            altitude / 1000.0,
            field.exospheric_temperature(&point, &drivers)
        );
        assert!(rho.is_finite() && rho > 0.0, "{altitude} m gave {rho}");
        assert!(low < rho && rho < high);
    }
    // A 400 km density outside this band would mean the drivers were assembled wrong, not that the
    // thermosphere did something: the whole measured corpus of §10.2 spans 7e-13 to 9e-12 there.
    let four_hundred = field.density(
        &SdtmPoint {
            altitude: 400e3,
            latitude: 0.0,
            local_solar_time: 12.0,
            day_of_year: doy,
        },
        &drivers,
    );
    assert!(
        (1e-13..1e-11).contains(&four_hundred),
        "400 km noon density {four_hundred:e} is not a thermosphere"
    );
}

/// The trailing mean this crate recomputes for a forecast day is the one the provider publishes for a
/// measured day.
///
/// `extend_forecast` recomputes both 81-day means over the extended dailies rather than carrying the
/// last published value forward. That is only correct if our window is the provider's window, and this
/// is the one place the two can be compared: on measured days, where CelesTrak publishes both.
#[test]
#[ignore = "needs tools/fetch-density-data.sh's SW-All.csv in data/validation/density"]
fn the_eighty_one_day_means_are_the_providers_own_window() {
    let days = feed();
    let last = last_measured(&days);
    let width = usize::try_from(FLUX_WINDOW).unwrap();

    let (mut worst_trailing, mut worst_centred, mut at, mut n) = (0.0_f64, 0.0_f64, 0, 0_u32);
    for i in width - 1..last {
        // Only where the window is contiguous and every day in it is measured -- an `INT` day is a
        // value the series did not have, and the provider's mean may or may not contain it.
        let window = &days[i + 1 - width..=i];
        if ordinal(&days[i]) - ordinal(&window[0]) != FLUX_WINDOW - 1
            || !window.iter().all(|d| d.quality.is_measured())
        {
            continue;
        }
        let (mut sum, mut count) = (0.0_f64, 0.0_f64);
        for d in window {
            sum += d.f107_observed;
            count += 1.0;
        }
        let trailing = (sum / count - days[i].f107_last81).abs();
        if trailing > worst_trailing {
            (worst_trailing, at) = (trailing, i);
        }
        // The centred mean of the same day, forty days each side.
        let centred_window = &days[i - width / 2..=i + width / 2];
        if centred_window.len() == width
            && ordinal(&days[i + width / 2]) - ordinal(&days[i - width / 2]) == FLUX_WINDOW - 1
            && centred_window.iter().all(|d| d.quality.is_measured())
        {
            let (mut s, mut c) = (0.0_f64, 0.0_f64);
            for d in centred_window {
                s += d.f107_observed;
                c += 1.0;
            }
            worst_centred = worst_centred.max((s / c - days[i].f107_center81).abs());
        }
        n += 1;
    }
    println!(
        "\n== our 81-day window against the published one, {n} fully measured windows ==\n\
         worst trailing difference {worst_trailing:.4} flux units (at {:04}-{:02}-{:02}), \
         worst centred {worst_centred:.4}",
        days[at].year, days[at].month, days[at].day
    );
    assert!(n > 20_000, "only {n} windows were fully measured");
    // The file publishes the means to one decimal, so a tenth is the rounding and anything past it is
    // a different window.
    assert!(worst_trailing < 0.05, "trailing off by {worst_trailing}");
    assert!(worst_centred < 0.05, "centred off by {worst_centred}");
}

/// Persistence against 27-day recurrence, by lead time, on the observed series.
///
/// Both are baselines, so the question is not whether either is good but which to use at which lead —
/// and that has an answer only once it is measured. NOAA's own `PRD` forecast cannot be scored here:
/// the file carries it only for days that have *not* happened yet, so scoring it would need an archive
/// of past snapshots that this repository does not keep.
#[test]
#[ignore = "needs tools/fetch-density-data.sh's SW-All.csv in data/validation/density"]
fn the_forecast_baselines_are_scored_by_lead_time() {
    // A prime stride so the sampling does not beat against the 27-day rotation, over the span the
    // file's own dailies are dense in.
    const STRIDE: usize = 13;
    const HISTORY: usize = 220;
    /// One lead per day of the rotation the recurrence forecast copies from.
    const LEADS: usize = 27;

    assert_eq!(i64::try_from(LEADS).unwrap(), SOLAR_ROTATION);
    let days = feed();
    let last = last_measured(&days);
    let leads = LEADS;
    let first = last.saturating_sub(15 * 365);

    let mut flux = [[(0.0_f64, 0.0_f64); 2]; LEADS];
    let mut storm = [[(0.0_f64, 0.0_f64); 2]; LEADS];
    let mut origins = 0_u32;
    for i in (first..last - leads).step_by(STRIDE) {
        if !days[i].quality.is_measured() || i < HISTORY {
            continue;
        }
        let horizon = &days[i + leads];
        if ordinal(horizon) - ordinal(&days[i]) != SOLAR_ROTATION {
            continue;
        }
        let prefix = &days[i + 1 - HISTORY..=i];
        let forecasts = [ForecastMethod::Persistence, ForecastMethod::Recurrence].map(|m| {
            space_weather::extend_forecast(prefix, m, horizon.year, horizon.month, horizon.day)
        });
        let (Some(persistence), Some(recurrence)) = (&forecasts[0], &forecasts[1]) else {
            continue;
        };
        origins += 1;
        for lead in 1..=leads {
            let truth = &days[i + lead];
            if !truth.quality.is_measured() {
                continue;
            }
            for (m, series) in [persistence, recurrence].into_iter().enumerate() {
                let row = &series[HISTORY - 1 + lead];
                flux[lead - 1][m].0 += (row.f107_observed - truth.f107_observed).abs();
                flux[lead - 1][m].1 += 1.0;
                storm[lead - 1][m].0 += (row.ap - truth.ap).abs();
                storm[lead - 1][m].1 += 1.0;
            }
        }
    }

    println!(
        "\n== forecast baselines, mean absolute error by lead, {origins} origins over the last 15 years =="
    );
    println!("lead  F10.7 persistence  F10.7 recurrence   Ap persistence   Ap recurrence");
    let mae = |c: (f64, f64)| c.0 / c.1;
    for lead in 1..=leads {
        println!(
            "{lead:>4}{:17.2}{:18.2}{:17.2}{:16.2}",
            mae(flux[lead - 1][0]),
            mae(flux[lead - 1][1]),
            mae(storm[lead - 1][0]),
            mae(storm[lead - 1][1])
        );
    }
    let crossover = (1..=leads).find(|&l| mae(flux[l - 1][1]) < mae(flux[l - 1][0]));
    println!(
        "F10.7: recurrence first beats persistence at lead {}",
        crossover.map_or("never".to_string(), |l| l.to_string())
    );
    let storm_crossover = (1..=leads).find(|&l| mae(storm[l - 1][1]) < mae(storm[l - 1][0]));
    println!(
        "Ap:    recurrence first beats persistence at lead {}",
        storm_crossover.map_or("never".to_string(), |l| l.to_string())
    );

    assert!(origins > 300, "only {origins} origins were scorable");
    // At a lead of exactly one rotation the two baselines are the same forecast: recurrence copies the
    // day 27 back, which is the origin, which is what persistence holds. Structural, so it is an
    // equality and not a tolerance -- and it is the check that the lead indexing is not off by a day.
    assert_eq!(mae(flux[leads - 1][0]), mae(flux[leads - 1][1]));
    assert_eq!(mae(storm[leads - 1][0]), mae(storm[leads - 1][1]));
    // And in the middle of the rotation, where the two genuinely differ, recurrence is ahead on both
    // indices -- measured at lead 20 as 16.5 against 14.7 in F10.7 and 7.7 against 7.0 in Ap.
    assert!(mae(flux[19][1]) < mae(flux[19][0]));
    assert!(mae(storm[19][1]) < mae(storm[19][0]));
    // Persistence has to win at lead one: it *is* the last measurement, and recurrence is a month-old
    // one. Anything else means the indexing is off by a day somewhere.
    assert!(mae(flux[0][0]) < mae(flux[0][1]));
    assert!(mae(storm[0][0]) < mae(storm[0][1]));
    // And every error has to be finite and bounded by the range of the index itself.
    for lead in 1..=leads {
        for m in 0..2 {
            assert!(mae(flux[lead - 1][m]).is_finite() && mae(flux[lead - 1][m]) < 100.0);
            assert!(mae(storm[lead - 1][m]).is_finite() && mae(storm[lead - 1][m]) < 100.0);
        }
    }
}
