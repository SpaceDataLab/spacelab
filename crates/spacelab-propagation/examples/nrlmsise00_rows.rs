//! NRLMSISE-00 total mass density on many rows, for the corpus campaign of `tools/fit-sdtm-corpus.py`.
//!
//! `cargo run -p spacelab-propagation --release --example nrlmsise00_rows -- SW-All.csv < rows > rho`
//!
//! Each input line is `year month day seconds_of_day altitude_km latitude_deg longitude_deg lst_h`;
//! each output line is the density in kg/m³, or `nan` where the feed lacks a day the model needs.
//! The drivers are the ones `tests/measured_density.rs` scores NRLMSISE-00 with (§10.2): the 81-day
//! *centred* F10.7, the previous day's observed F10.7, and the storm-time `ap[]` history
//! (`switches[9] = -1`) through `gtd7d`, whose total carries anomalous oxygen.

#![allow(clippy::expect_used)]

use std::io::{BufRead, BufWriter, Write};

use spacelab_propagation::space_weather::{self, SpaceWeatherDay};
use spacelab_propagation::{Nrlmsise00Flags, Nrlmsise00Input, gtd7d};

/// The feed days a row needs: its own, the previous one (yesterday's flux) and the three before that
/// (`ap[]` reaches 57 h back). Searching this slice keeps `nrlmsise_ap_array`'s linear scan short.
fn window(days: &[SpaceWeatherDay], year: i32, month: u32, day: u32) -> Option<&[SpaceWeatherDay]> {
    let i = days
        .binary_search_by(|d| (d.year, d.month, d.day).cmp(&(year, month, day)))
        .ok()?;
    days.get(i.checked_sub(4)?..=i)
}

fn day_of_year(year: i32, month: u32, day: u32) -> i32 {
    const CUMULATIVE: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_index = usize::try_from(month - 1).expect("month 1..=12");
    CUMULATIVE[month_index] + i32::try_from(day).expect("day 1..=31") + i32::from(leap && month > 2)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: nrlmsise00_rows SW-All.csv < rows");
    let days = space_weather::parse_space_weather(
        &std::fs::read_to_string(path).expect("readable SW-All.csv"),
    );
    let mut storm = Nrlmsise00Flags::default();
    storm.switches[9] = -1;

    let mut out = BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("readable stdin");
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [year, month, day, rest @ ..] = &fields[..] else {
            panic!("expected 8 fields, got {line:?}");
        };
        let year: i32 = year.parse().expect("integer year");
        let month: u32 = month.parse().expect("integer month");
        let day: u32 = day.parse().expect("integer day");
        let f: Vec<f64> = rest
            .iter()
            .map(|x| x.parse().expect("numeric field"))
            .collect();
        let [sec, alt, lat, lon, lst] = f[..] else {
            panic!("expected 8 fields, got {line:?}");
        };
        let rho = window(&days, year, month, day).and_then(|w| {
            let today = w[4];
            let ap_array = space_weather::nrlmsise_ap_array(w, year, month, day, sec)?;
            let input = Nrlmsise00Input {
                year,
                doy: day_of_year(year, month, day),
                sec,
                alt,
                g_lat: lat,
                g_long: lon,
                lst,
                f107a: today.f107_center81,
                f107: w[3].f107_observed,
                ap: today.ap,
                ap_array: Some(ap_array),
            };
            // `d[5]` is g/cm³ under the default switches.
            Some(gtd7d(&input, &storm).d[5] * 1000.0)
        });
        writeln!(out, "{}", rho.unwrap_or(f64::NAN)).expect("writable stdout");
    }
}
