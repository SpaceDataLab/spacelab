//! What an hourly geomagnetic index says that a three-hourly one cannot.
//!
//! Kp and its linear twin `ap` are published every three hours and saturate at 9 and 400. Both
//! limits were reached by the May 2024 "Gannon" storm, the largest since 2003, and both are visible
//! in the two products this file reads side by side over the same week:
//!
//! * `celestrak_sw_validation.csv` — the 3-hourly Kp/ap the density models are specified against.
//! * `gfz_hp60_gannon.txt`, `gfz_hp30_gannon.txt` — GFZ's Hpo family, the same construction from the
//!   same 13 observatories at 60- and 30-minute cadence, and **uncapped** above Kp 9.
//!
//! The two are not a coarse and a fine sampling of one quantity: the Hpo standardisation differs, so
//! `Hp60` is not the hour-by-hour breakdown of the `Kp` that covers it, and nothing here treats it as
//! one. What is compared is what each product can *express* — the ceiling, and the variation inside
//! one three-hour interval — and then what that is worth in density.
//!
//! Both fixtures are committed, so this runs under plain `cargo test`. The GFZ files are CC BY 4.0
//! and their 30-line header, which carries the licence and the citation, is kept verbatim in the
//! fixture.

// `start_hour` is a decimal the file states exactly -- 0.0, 0.5, 1.0 -- so an exact comparison is
// the right one here: a tolerance would hide a cadence the parser had misread.
#![allow(clippy::expect_used, clippy::float_cmp, clippy::unwrap_used)]

use spacelab_propagation::nrlmsise00::{Nrlmsise00Flags, Nrlmsise00Input, gtd7d};
use spacelab_propagation::space_weather::{self, HpoInterval};

const CELESTRAK: &str = include_str!("fixtures/celestrak_sw_validation.csv");
const HP60: &str = include_str!("fixtures/gfz_hp60_gannon.txt");
const HP30: &str = include_str!("fixtures/gfz_hp30_gannon.txt");

/// The top of the published `ap` table, which is Kp 9.0 — the largest disturbance the three-hourly
/// product has a symbol for.
const AP_CEILING: f64 = 400.0;

/// The storm's peak day, and the day both products are compared on.
const YEAR: i32 = 2024;
/// May.
const MONTH: u32 = 5;
/// The 10th — the onset; the 11th is the recovery, and both are in the fixture.
const DAY: u32 = 10;

fn on(intervals: &[HpoInterval], year: i32, month: u32, day: u32) -> Vec<HpoInterval> {
    intervals
        .iter()
        .filter(|i| i.year == year && i.month == month && i.day == day)
        .copied()
        .collect()
}

#[test]
fn the_week_parses_at_both_cadences_and_the_two_agree_about_the_days_they_cover() {
    let hp60 = space_weather::parse_hpo(HP60);
    let hp30 = space_weather::parse_hpo(HP30);

    // Seven whole days, 2024-05-08 to 05-14, at 24 and 48 intervals a day.
    assert_eq!(hp60.len(), 7 * 24, "Hp60 rows");
    assert_eq!(hp30.len(), 7 * 48, "Hp30 rows");

    // A row read off the file by hand, so a column shift cannot pass: the storm's peak hour.
    let peak = hp60
        .iter()
        .find(|i| i.year == 2024 && i.month == 5 && i.day == 10 && i.start_hour == 22.0)
        .expect("2024-05-10 22 UT is in the fixture");
    assert!(
        (peak.hp - 10.667).abs() < 1e-9,
        "Hp60 at the peak: {}",
        peak.hp
    );
    assert!(
        (peak.ap - 456.0).abs() < 1e-9,
        "ap60 at the peak: {}",
        peak.ap
    );

    // The lookup is by instant, and it must land in the interval that *contains* it rather than the
    // one that starts nearest: 22:59 UT is still the 22:00 interval at hourly cadence and the 22:30
    // one at half-hourly.
    let at60 = space_weather::hpo_at(&hp60, 2024, 5, 10, 22.0 * 3600.0 + 3540.0).unwrap();
    let at30 = space_weather::hpo_at(&hp30, 2024, 5, 10, 22.0 * 3600.0 + 3540.0).unwrap();
    assert_eq!(at60.start_hour, 22.0);
    assert_eq!(at30.start_hour, 22.5);
}

#[test]
fn the_three_hourly_product_saturates_where_the_hourly_one_keeps_reading() {
    let days = space_weather::parse_space_weather(CELESTRAK);
    let hp60 = space_weather::parse_hpo(HP60);

    let three_hourly_peak = days.iter().flat_map(|d| d.ap3h).fold(f64::MIN, f64::max);
    let hourly_peak = hp60.iter().map(|i| i.ap).fold(f64::MIN, f64::max);
    let hp_peak = hp60.iter().map(|i| i.hp).fold(f64::MIN, f64::max);

    println!(
        "week of 2024-05-08: peak 3-hourly ap {three_hourly_peak}, peak ap60 {hourly_peak}, \
         peak Hp60 {hp_peak}"
    );

    // The three-hourly series is *at* its ceiling — not near it, at it — so it is not measuring the
    // peak any more, it is reporting the end of its own scale.
    assert!(
        (three_hourly_peak - AP_CEILING).abs() < 1e-9,
        "the 3-hourly peak is {three_hourly_peak}, not the ceiling; is this still the Gannon week?"
    );
    // And the hourly one is past it, by 14 %, which is the size of the statement the older product
    // cannot make.
    assert!(
        hourly_peak > AP_CEILING,
        "ap60 peaked at {hourly_peak}, which the 3-hourly scale could have expressed"
    );
    assert!(
        hp_peak > 9.0,
        "Hp60 peaked at {hp_peak}, which Kp could have expressed"
    );
}

#[test]
fn one_three_hour_interval_of_the_onset_holds_a_factor_of_three() {
    let hp60 = on(&space_weather::parse_hpo(HP60), YEAR, MONTH, DAY);
    assert_eq!(hp60.len(), 24);

    let mut worst = (0.0_f64, 0_u32);
    for bin in 0..8_u32 {
        let start = f64::from(bin) * 3.0;
        let inside: Vec<f64> = hp60
            .iter()
            .filter(|i| i.start_hour >= start && i.start_hour < start + 3.0)
            .map(|i| i.ap)
            .collect();
        assert_eq!(inside.len(), 3, "three hours in each 3-hour interval");
        let lo = inside.iter().copied().fold(f64::MAX, f64::min);
        let hi = inside.iter().copied().fold(f64::MIN, f64::max);
        let ratio = hi / lo.max(1.0);
        println!(
            "{YEAR}-{MONTH:02}-{DAY:02} {:02}-{:02} UT: ap60 {inside:?}, spread {ratio:.2}x",
            bin * 3,
            bin * 3 + 3
        );
        if ratio > worst.0 {
            worst = (ratio, bin);
        }
    }

    // A single number stands for all three of those hours in every model specified on Kp. At the
    // onset the three differ by more than a factor of three, so whichever value is chosen is wrong
    // by that much at one end of the interval or the other.
    assert!(
        worst.0 > 3.0,
        "the worst within-interval spread was {:.2}x in the {:02}-{:02} UT interval",
        worst.0,
        worst.1 * 3,
        worst.1 * 3 + 3
    );
}

#[test]
fn inside_one_three_hour_interval_the_modelled_density_moves_and_the_published_driver_cannot_see_it()
 {
    let days = space_weather::parse_space_weather(CELESTRAK);
    let hp60 = space_weather::parse_hpo(HP60);

    // The 15-18 UT interval of the storm day, swept hour by hour. The published `ap` is one number
    // for all three; `ap60` is 80, 94 and 294. Everything else about the evaluation is held fixed,
    // so the spread below is the driver's and nothing else's.
    let mut published = Vec::new();
    let mut resolved = Vec::new();
    for hour in [15.5_f64, 16.5, 17.5] {
        let sec = hour * 3600.0;
        let array = space_weather::nrlmsise_ap_array(&days, YEAR, MONTH, DAY, sec)
            .expect("the fixture covers the look-back");
        let hourly = space_weather::hpo_at(&hp60, YEAR, MONTH, DAY, sec).expect("in the file");

        // Only slot 1 changes -- "the 3-hourly ap of the current interval". The history slots stay
        // as published, so what is measured is the effect of resolving *this* interval alone.
        let mut sharpened = array;
        sharpened[1] = hourly.ap;

        let base = Nrlmsise00Input {
            year: YEAR,
            doy: 131, // 2024-05-10
            sec,
            alt: 400.0,
            g_lat: 0.0,
            g_long: 0.0,
            lst: hour,
            f107a: 176.2,
            f107: 233.2, // 2024-05-09 observed, the previous day the model asks for
            ap: 105.0,
            ap_array: Some(array),
        };
        let mut storm = Nrlmsise00Flags::default();
        storm.switches[9] = -1;

        let rho_pub = gtd7d(&base, &storm).d[5] * 1000.0;
        let rho_res = gtd7d(
            &Nrlmsise00Input {
                ap_array: Some(sharpened),
                ..base
            },
            &storm,
        )
        .d[5]
            * 1000.0;
        println!(
            "{YEAR}-{MONTH:02}-{DAY:02} {hour:04.1} UT, 400 km: ap {} -> {}, \
             rho {rho_pub:.4e} -> {rho_res:.4e} kg/m^3, {:+.1} %",
            array[1],
            hourly.ap,
            (rho_res / rho_pub - 1.0) * 100.0
        );
        published.push(rho_pub);
        resolved.push(rho_res);
    }

    // The measured numbers, and they are smaller than a first guess would put them. Substituting
    // the onset hour's own `ap` for the interval's moves NRLMSISE-00's density at 400 km by about
    // -4.5 %, -3.9 % and +5.9 % across the three hours, because the model's storm response is
    // dominated by the 12-33 h and 36-57 h history slots rather than by the current interval: the
    // thermosphere at 400 km integrates hours of heating, and one hour of it is a correction rather
    // than the signal. **That is the honest verdict of this ticket** -- an hourly index is worth
    // single digits of percent through NRLMSISE-00's published storm parameterization, and the case
    // for it is the two statements above plus the one below, not a large density number.
    let deviations: Vec<f64> = published
        .iter()
        .zip(&resolved)
        .map(|(p, r)| r / p - 1.0)
        .collect();
    let worst = deviations
        .iter()
        .copied()
        .fold(0.0_f64, |a, d| a.max(d.abs()));
    assert!(
        (0.03..0.10).contains(&worst),
        "the worst single-hour deviation was {:+.1} %, outside the measured 3-10 % band",
        worst * 100.0
    );
    // It changes sign inside the interval, which is the smearing stated as arithmetic: one number
    // over-drives the first hour and under-drives the last.
    assert!(
        deviations[0] < 0.0 && deviations[2] > 0.0,
        "the published `ap` should over-drive the first hour and under-drive the last: {deviations:?}"
    );

    // And the consequence that is worth more than the percentages. Under the published driver the
    // density falls monotonically through the onset -- the diurnal term is the only thing moving,
    // so the storm arriving at 17 UT leaves no trace at all in the modelled series. Resolved, the
    // fall is arrested in the third hour. The three-hourly index does not merely mis-scale the
    // onset; it removes its shape.
    //
    // Measured rather than assumed, and the first attempt at this assertion said the resolved
    // series should be the *wider* of the two, which the measurement refuted: it is narrower
    // (1.06x against 1.17x), because the correction pushes back against the diurnal fall.
    let falls = |v: &[f64]| v.windows(2).all(|w| w[1] < w[0]);
    assert!(
        falls(&published),
        "the published driver should give a monotone diurnal fall here: {published:?}"
    );
    assert!(
        !falls(&resolved),
        "resolving the onset hour should break the monotone fall: {resolved:?}"
    );
}
