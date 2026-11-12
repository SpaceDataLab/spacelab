//! SDTM-1's two fields and the clean-room JB2008 on many rows, for the prospective blind test of
//! `docs/sdtm1-blind-protocol.md` (`plans/17` Ф30.J). NRLMSISE-00 comes from `nrlmsise00_rows`.
//!
//! `cargo run -p spacelab-propagation --release --example blind_rows -- SW-All.csv HP60 OMNI SOLFSMY.TXT DTCFILE.TXT [DEFAULT.field SOLAR_WIND.field] < rows > out`
//!
//! `HP60` is GFZ's `Hp60_ap60_complete_series.txt`, `OMNI` the `omni2_sw_hourly.txt` that
//! `tools/fetch-omni.sh` writes. Each input line is `satellite year month day seconds_of_day
//! altitude_m latitude_deg longitude_deg lst_h`, the satellite as the corpus's two-letter code. Each
//! output line is SDTM-1's density, the solar-wind field's and the clean-room JB2008's, kg/m³, `nan`
//! where a feed lacks an hour or a day the model needs.
//!
//! - **The fields.** With no field file, the shipped ones: [`coefficients::FIELD`] and
//!   [`solar_wind_coefficients::FIELD`] with its [`solar_wind_coefficients::DRIVE`]. With two, the
//!   fields `tools/evaluate-sdtm.py export` wrote for a fold, through the checked constructors.
//! - **The drivers** are the library's: [`space_weather::sdtm_drivers`] for the 3-hourly field and
//!   [`SolarWindDrive::storm`] over the hours before the row's hour for the solar-wind field.
//! - **Three conventions are the fit's, so that a row compares with the harness's to round-off.** The
//!   storm state is run from 120 days before the row, as the fit runs it. The day of year is the day
//!   and the whole hour, as `tools/gen_sdtm_coefficients.py` reads it. Each
//!   satellite's log scale from the field's table multiplies the density, as the harness scores a
//!   fold and as `docs/verification.md` §10.38 printed it; GRACE-FO C is the reference, at zero.
//! - **JB2008** is driven as `tests/measured_density.rs` drives it (§10.2): SET's indices at their
//!   lags, and the hour angle from the row's local solar time.

#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};

use spacelab_propagation::ephemeris::{AnalyticSun, Ephemeris};
use spacelab_propagation::jacchia_bowman;
use spacelab_propagation::sdtm::{
    CouplingMap, DRIVERS, FluxExcess, NitricOxideCooling, SdtmDrivers, SdtmPoint, SolarWindDrive,
    StormGain, StormHour, StormResponse, Thermosphere, coefficients, solar_wind_coefficients,
};
use spacelab_propagation::space_weather::{
    self, DstTemperatureDay, SolarIndexDay, SpaceWeatherDay,
};

const HARMONICS: usize = coefficients::HARMONICS;
const FLATTENING: f64 = 1.0 / 298.257_223_563;
/// The storm state's warm-up, 3-hourly intervals and hours: the 120 days the fit runs it from
/// (`tools/fit-sdtm-corpus.py`, `CorpusModel.first`). The tables' own `HISTORY` is the shortest a caller
/// needs for the shipped constants; a fold's slower nitric oxide (40 h at the 2020 cut against the shipped
/// 30 h) leaves 5e-4 in `ln ρ` after its 15 days, and a scorer reproduces the fit.
const WARM_UP_INTERVALS: usize = 120 * 8;
const WARM_UP_HOURS: i64 = 120 * 24;
const MJD_AT_UNIX_EPOCH: f64 = 40587.0;

/// A field and each satellite's log scale against GRACE-FO C.
struct Scored {
    field: Thermosphere<HARMONICS>,
    scale: HashMap<String, f64>,
}

impl Scored {
    fn density(&self, satellite: &str, point: &SdtmPoint, drivers: &SdtmDrivers) -> f64 {
        let scale = if satellite == "GC" {
            0.0
        } else {
            *self
                .scale
                .get(satellite)
                .unwrap_or_else(|| panic!("no log scale for {satellite}"))
        };
        self.field.density(point, drivers) * scale.exp()
    }
}

/// The shipped table's satellite names, by the corpus's codes.
fn shipped_scale(table: &[(&str, f64)]) -> HashMap<String, f64> {
    let code = |name: &str| match name {
        "Swarm A" => "SA",
        "Swarm B" => "SB",
        "Swarm C" => "SC",
        "GRACE A" => "GA",
        "GRACE B" => "GB",
        "CHAMP" => "CH",
        "GOCE" => "GO",
        _ => panic!("an unknown satellite {name} in the shipped table"),
    };
    table
        .iter()
        .map(|&(name, s)| (code(name).to_owned(), s))
        .collect()
}

/// `key value…` lines: one number for most keys, a row for `coefficients` and the knots.
fn read_keys(path: &str) -> (HashMap<String, Vec<f64>>, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut value = HashMap::new();
    let mut rows = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut cells = line.split_whitespace();
        let key = cells.next().expect("a key");
        let numbers: Vec<f64> = cells
            .map(|c| c.parse().unwrap_or_else(|_| panic!("{path}: {line:?}")))
            .collect();
        if key == "coefficients" {
            rows.push(numbers);
        } else {
            value.insert(key.to_owned(), numbers);
        }
    }
    (value, rows)
}

fn read_field(path: &str) -> (Scored, HashMap<String, Vec<f64>>) {
    let (value, rows) = read_keys(path);
    let get = |k: &str| value.get(k).unwrap_or_else(|| panic!("{path}: no {k}"))[0];
    assert_eq!(
        rows.len(),
        DRIVERS,
        "{path}: one coefficient row per driver"
    );
    let mut coefficients = [[0.0; HARMONICS]; DRIVERS];
    for (out, row) in coefficients.iter_mut().zip(&rows) {
        out.copy_from_slice(row);
    }
    let field = Thermosphere::from_coefficients(
        coefficients,
        get("sigma"),
        get("boundary_scale"),
        StormResponse::Cooling(NitricOxideCooling {
            heating: get("heating_s"),
            nitric_oxide: get("nitric_oxide_s"),
            saturation: get("saturation"),
        }),
    )
    .and_then(|f| {
        f.with_flux_excess(FluxExcess::Saturated {
            scale: get("excess_scale"),
        })
    })
    .and_then(|f| {
        f.with_storm_gain(StormGain::FluxPower {
            exponent: get("gain_exponent"),
        })
    })
    .unwrap_or_else(|| panic!("{path}: not a usable field"));
    let scale = value
        .iter()
        .filter_map(|(k, v)| Some((k.strip_prefix("scale_")?.to_owned(), v[0])))
        .collect();
    (Scored { field, scale }, value)
}

fn read_drive<const M: usize>(value: &HashMap<String, Vec<f64>>) -> SolarWindDrive<M> {
    let knots = |k: &str| -> [f64; M] {
        value
            .get(k)
            .unwrap_or_else(|| panic!("no {k}"))
            .as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("{k}: not {M} knots"))
    };
    CouplingMap::new(knots("knot_coupling"), knots("knot_ap60"))
        .and_then(|map| SolarWindDrive::new(value["share"][0], map))
        .expect("a usable coupling map and share")
}

/// An hourly series on a contiguous grid from its first hour, NaN where an hour is missing.
struct Hourly<T> {
    first: i64,
    values: Vec<T>,
}

impl<T: Copy> Hourly<T> {
    fn at(&self, hour: i64) -> Option<T> {
        usize::try_from(hour - self.first)
            .ok()
            .and_then(|i| self.values.get(i).copied())
    }
}

/// Hours since 1970-01-01T00.
fn hour_number(year: i32, month: u32, day: u32, hour: i64) -> i64 {
    days_since_1970(year, month, day) * 24 + hour
}

/// The hour of the day `hours` falls in, `0..=23`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a floored number in [0, 23] is exact in any integer"
)]
fn hour_of_day(hours: f64) -> i64 {
    hours.floor().clamp(0.0, 23.0) as i64
}

fn read_ap60(text: &str) -> Hourly<f64> {
    let intervals = space_weather::parse_hpo(text);
    let hour = |i: &space_weather::HpoInterval| {
        // An Hp60 interval starts on the whole hour.
        hour_number(i.year, i.month, i.day, hour_of_day(i.start_hour))
    };
    let first = hour(intervals.first().expect("an ap60 interval"));
    let last = hour(intervals.last().expect("an ap60 interval"));
    let mut values = vec![f64::NAN; usize::try_from(last - first + 1).expect("ascending")];
    for i in &intervals {
        values[usize::try_from(hour(i) - first).expect("ascending")] = i.ap;
    }
    Hourly { first, values }
}

/// One number of an OMNI stamp `YYYY-MM-DDThh`.
fn stamp_field<T: std::str::FromStr>(stamp: &str, range: std::ops::Range<usize>) -> T {
    stamp
        .get(range)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("not an OMNI hour stamp: {stamp}"))
}

/// OMNI's `hour imf sw B Bx By Bz T n V P`, `null` where missing: `(V, By, Bz)` by hour.
fn read_omni(text: &str) -> Hourly<[f64; 3]> {
    let mut first = None;
    let mut values = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let stamp = f[0];
        let hour = hour_number(
            stamp_field(stamp, 0..4),
            stamp_field(stamp, 5..7),
            stamp_field(stamp, 8..10),
            stamp_field(stamp, 11..13),
        );
        let first = *first.get_or_insert(hour);
        let index = i64::try_from(values.len()).expect("fewer hours than i64 holds");
        assert_eq!(hour - first, index, "OMNI is not contiguous");
        let value = |s: &str| {
            if s == "null" {
                f64::NAN
            } else {
                s.parse().expect("a number or null")
            }
        };
        values.push([value(f[9]), value(f[5]), value(f[6])]);
    }
    Hourly {
        first: first.expect("an OMNI hour"),
        values,
    }
}

/// The solar-wind field's storm driver at the row's hour: the drive over the `WARM_UP_HOURS` hours that
/// end with the hour before it.
fn solar_wind_storm<const M: usize>(
    drive: &SolarWindDrive<M>,
    response: &StormResponse,
    ap60: &Hourly<f64>,
    omni: &Hourly<[f64; 3]>,
    hour: i64,
) -> Option<f64> {
    let hours: Option<Vec<StormHour>> = (hour - WARM_UP_HOURS..hour)
        .map(|h| {
            let [speed, by, bz] = omni.at(h).unwrap_or([f64::NAN; 3]);
            StormHour::from_values(ap60.at(h)?, speed, by, bz)
        })
        .collect();
    Some(drive.storm(response, &hours?))
}

/// Proleptic-Gregorian day number since 1970-01-01 (Howard Hinnant's `days_from_civil`).
fn days_since_1970(year: i32, month: u32, day: u32) -> i64 {
    let y = i64::from(year) - i64::from(month <= 2);
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = i64::from(if month > 2 { month - 3 } else { month + 9 });
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn day_of_year(year: i32, month: u32, day: u32) -> u32 {
    const CUMULATIVE: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_index = usize::try_from(month - 1).expect("month 1..=12");
    CUMULATIVE[month_index] + day + u32::from(leap && month > 2)
}

/// The clean-room JB2008 on SET's argument list: `AMJD, SUN(1..2), SAT(1..3)` and the nine drivers.
/// Only the hour angle `sat_ra − sun_ra` enters, so it is taken from the row's local solar time and
/// the Sun's right ascension is zero.
fn jb2008_open(row: &Row, solar: &[SolarIndexDay], dst: &[DstTemperatureDay]) -> Option<f64> {
    let d = space_weather::jb2008_drivers(solar, dst, row.year, row.day_of_year, row.sec)?;
    let day = i32::try_from(days_since_1970(row.year, row.month, row.day)).expect("a day in i32");
    let mjd = MJD_AT_UNIX_EPOCH + f64::from(day) + row.sec / 86_400.0;
    let sun = AnalyticSun {
        epoch_j2000_seconds: (mjd - 51_544.5) * 86_400.0,
    }
    .position(0.0);
    let sun_dec = (sun[2] / (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt()).asin();
    let one_minus_f = 1.0 - FLATTENING;
    let input = jacchia_bowman::Input {
        mjd,
        sun_ra: 0.0,
        sun_dec,
        sat_ra: (row.lst - 12.0) * (std::f64::consts::PI / 12.0),
        sat_lat: (one_minus_f * one_minus_f * row.latitude.tan()).atan(),
        height_km: row.altitude / 1000.0,
        indices: jacchia_bowman::SolarIndices {
            f10: d.f10,
            f10_bar: d.f10b,
            s10: d.s10,
            s10_bar: d.s10b,
            m10: d.xm10,
            m10_bar: d.xm10b,
            y10: d.y10,
            y10_bar: d.y10b,
        },
        dtc_storm: d.dstdtc,
    };
    // A refusal at a corpus height is a defect to see, not a row to drop.
    Some(
        jacchia_bowman::evaluate(&input)
            .unwrap_or_else(|e| panic!("JB2008 open refused {input:?}: {e:?}"))
            .density,
    )
}

struct Row {
    year: i32,
    month: u32,
    day: u32,
    day_of_year: u32,
    sec: f64,
    altitude: f64,
    latitude: f64,
    lst: f64,
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let (feeds, fields) = paths.split_at(paths.len().min(5));
    let [sw, hp60, omni_path, solfsmy, dtcfile] = feeds else {
        panic!(
            "usage: blind_rows SW-All.csv HP60 OMNI SOLFSMY.TXT DTCFILE.TXT [DEFAULT.field SOLAR_WIND.field] < rows"
        );
    };
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let feeds = Feeds {
        days: space_weather::parse_space_weather(&read(sw)),
        ap60: read_ap60(&read(hp60)),
        omni: read_omni(&read(omni_path)),
        solar: space_weather::parse_solfsmy(&read(solfsmy)),
        dst: space_weather::parse_dtcfile(&read(dtcfile)),
    };
    match fields {
        [] => score_rows(
            &feeds,
            &Scored {
                field: coefficients::FIELD,
                scale: shipped_scale(&coefficients::SATELLITE_SCALE),
            },
            &Scored {
                field: solar_wind_coefficients::FIELD,
                scale: shipped_scale(&solar_wind_coefficients::SATELLITE_SCALE),
            },
            &solar_wind_coefficients::DRIVE,
        ),
        [default_path, wind_path] => {
            let (default, _) = read_field(default_path);
            let (wind, value) = read_field(wind_path);
            // A map's knot count is in its type, and a fold's map has a knot per `ap60` class its
            // training hours held: 33 before 2020, 34 in the shipped one.
            match value.get("knot_ap60").map(Vec::len) {
                Some(33) => score_rows(&feeds, &default, &wind, &read_drive::<33>(&value)),
                Some(34) => score_rows(&feeds, &default, &wind, &read_drive::<34>(&value)),
                n => panic!("{wind_path}: {n:?} knots, which no arm here reads"),
            }
        }
        _ => panic!("either no field file or two: DEFAULT.field SOLAR_WIND.field"),
    }
}

/// The driver files `main` reads.
struct Feeds {
    days: Vec<SpaceWeatherDay>,
    ap60: Hourly<f64>,
    omni: Hourly<[f64; 3]>,
    solar: Vec<SolarIndexDay>,
    dst: Vec<DstTemperatureDay>,
}

/// Every row of stdin under the two fields and the clean-room JB2008, one output line each.
fn score_rows<const M: usize>(
    feeds: &Feeds,
    default: &Scored,
    wind: &Scored,
    drive: &SolarWindDrive<M>,
) {
    let Feeds {
        days,
        ap60,
        omni,
        solar,
        dst,
    } = feeds;
    let mut out = BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("readable stdin");
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [satellite, year, month, day, rest @ ..] = &fields[..] else {
            panic!("expected 9 fields, got {line:?}");
        };
        let year: i32 = year.parse().expect("integer year");
        let month: u32 = month.parse().expect("integer month");
        let day: u32 = day.parse().expect("integer day");
        let f: Vec<f64> = rest
            .iter()
            .map(|x| x.parse().expect("numeric field"))
            .collect();
        let [sec, altitude, lat, _lon, lst] = f[..] else {
            panic!("expected 9 fields, got {line:?}");
        };
        let row = Row {
            year,
            month,
            day,
            day_of_year: day_of_year(year, month, day),
            sec,
            altitude,
            latitude: lat.to_radians(),
            lst,
        };
        let whole_hour = (sec / 3_600.0).floor();
        let point = SdtmPoint {
            altitude,
            latitude: row.latitude,
            local_solar_time: lst,
            day_of_year: f64::from(row.day_of_year) + whole_hour / 24.0,
        };
        // The flux drivers are the same for both fields; the solar-wind field replaces the storm.
        let drivers = space_weather::sdtm_drivers::<WARM_UP_INTERVALS>(
            days,
            default.field.storm(),
            year,
            month,
            day,
            sec,
        );
        let rho_default = drivers.map(|d| default.density(satellite, &point, &d));
        let hour = hour_number(year, month, day, hour_of_day(sec / 3_600.0));
        let rho_wind = drivers
            .zip(solar_wind_storm(
                drive,
                wind.field.storm(),
                ap60,
                omni,
                hour,
            ))
            .map(|(d, storm)| wind.density(satellite, &point, &SdtmDrivers { storm, ..d }));
        let rho_jb = jb2008_open(&row, solar, dst);
        let nan = f64::NAN;
        writeln!(
            out,
            "{:e} {:e} {:e}",
            rho_default.unwrap_or(nan),
            rho_wind.unwrap_or(nan),
            rho_jb.unwrap_or(nan)
        )
        .expect("writable stdout");
    }
}
