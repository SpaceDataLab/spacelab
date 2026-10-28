//! The USGS hourly Dst over the May 2024 "Gannon" storm, beside GFZ's Hp60 for the same week.
//!
//! `usgs_dst_gannon.txt` is `tools/fetch-usgs-dst.sh excerpt 2024-05-08 2024-05-14`: U.S. public
//! domain, its header carrying the credit and Rigler (2017), doi:10.3133/ofr20171037. The two
//! indices measure different things — Dst the ring current's field at low latitude, Hp60 the
//! hour's range at mid latitude — so what is checked is the storm both record: its depth, about
//! −400 nT, and the hour it peaks.

#![allow(clippy::expect_used, clippy::float_cmp)]

use spacelab_propagation::space_weather::{self, UsgsDstHour};

const DST: &str = include_str!("fixtures/usgs_dst_gannon.txt");
const HP60: &str = include_str!("fixtures/gfz_hp60_gannon.txt");

fn deepest(hours: &[UsgsDstHour], index: fn(&UsgsDstHour) -> Option<f64>) -> (UsgsDstHour, f64) {
    hours
        .iter()
        .filter_map(|h| index(h).map(|v| (*h, v)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("the week has values")
}

#[test]
fn the_week_parses_as_its_168_consecutive_hours() {
    let hours = space_weather::parse_usgs_dst(DST);
    assert_eq!(hours.len(), 7 * 24);
    for (k, h) in hours.iter().enumerate() {
        let (day, hour) = (8 + k / 24, k % 24);
        assert_eq!(
            (h.year, h.month, h.day, h.hour),
            (
                2024,
                5,
                u32::try_from(day).expect("day"),
                u32::try_from(hour).expect("hour")
            ),
            "row {k}"
        );
        assert!(h.dst4.is_some() && h.dst3.is_some(), "{h:?}");
    }
    // A row read off the file by hand, so a column swap cannot pass: the Dst4 minimum.
    assert_eq!(
        space_weather::usgs_dst_at(&hours, 2024, 5, 11, 2.5 * 3600.0),
        Some(UsgsDstHour {
            year: 2024,
            month: 5,
            day: 11,
            hour: 2,
            dst4: Some(-399.304_633_333_333_36),
            dst3: Some(-374.654_216_666_666_74),
        })
    );
}

/// Measured: Dst4 reaches −399.30 nT at 02 UT on 2024-05-11, Dst3 −421.92 nT at 04 UT. Hp60
/// reaches the week's maximum, 10.667, at 22 UT on the 10th and again at 02 UT on the 11th.
#[test]
fn the_storm_is_about_400_nt_deep_and_dst4_bottoms_out_in_an_hour_of_peak_hp60() {
    let hours = space_weather::parse_usgs_dst(DST);
    let (at4, dst4) = deepest(&hours, |h| h.dst4);
    let (at3, dst3) = deepest(&hours, |h| h.dst3);
    assert!((-410.0..-390.0).contains(&dst4), "Dst4 minimum {dst4} nT");
    assert!((-430.0..-410.0).contains(&dst3), "Dst3 minimum {dst3} nT");
    assert_eq!((at4.day, at4.hour), (11, 2));
    assert_eq!((at3.day, at3.hour), (11, 4));

    let hp60 = space_weather::parse_hpo(HP60);
    let peak = hp60.iter().map(|i| i.hp).fold(f64::NEG_INFINITY, f64::max);
    let at_dst4 = space_weather::hpo_at(&hp60, 2024, 5, 11, 2.5 * 3600.0).expect("Hp60 at 02 UT");
    assert_eq!(peak, 10.667);
    assert_eq!(at_dst4.hp, peak);
}
