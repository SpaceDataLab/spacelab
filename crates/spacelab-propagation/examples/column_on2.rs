//! The column ratio O/N2 above the N2 depth of 1e17 cm⁻² (Strickland et al. 1995), from SDTM-1 and
//! NRLMSISE-00, at the points GOLD and GUVI retrieve it (`plans/17` Ф23.P).
//!
//! `cargo run -p spacelab-propagation --release --example column_on2 -- SW-All.csv < points > columns`
//! `cargo run -p spacelab-propagation --release --example column_on2 -- --local 520 SW-All.csv < points`
//! `cargo run -p spacelab-propagation --release --example column_on2 -- --check`
//!
//! Each input line is `year month day seconds_of_day latitude_deg longitude_deg lst_h`; each output line is
//! `t_inf_k sdtm_ratio sdtm_depth_km msis_ratio msis_depth_km`, `nan` where the feed lacks a day a model
//! needs. The column is the vertical one in geodetic altitude, `∫_z^top n dz` to 5000 km, by 32-point
//! Gauss–Legendre on six pieces; the depth is where N2's column is 1e21 m⁻², by Newton on the column's
//! logarithm, whose derivative is `−n/N`. `--local ALT_KM` prints instead each model's local ratio `n_O/n_N2` at
//! that geodetic altitude, in the ratio columns, and the altitude in the depth columns: what an in-situ mass
//! spectrometer measures (TianMu-1's NGMS).
//!
//! SDTM-1's species are its closed form (`derivations/sdtm_column.mac`): the boundary composition and scale
//! and the Bates profile at the field's `T∞` and `σ`, so its ratio is the geometric one the gate computes.
//! NRLMSISE-00's are `gtd7`'s O and N2, driven as `nrlmsise00_rows` drives `gtd7d`. `--check` holds
//! SDTM-1's depth and ratio at `T∞` = 700, 1000 and 1500 K to `derivations/sdtm_column.gp`'s 60-digit
//! values and exits 1 on a miss.

#![allow(clippy::expect_used)]

use std::io::{BufRead, BufWriter, Write};

use spacelab_propagation::sdtm::{
    BOUNDARY, BOUNDARY_ALTITUDE, BOUNDARY_GRAVITY, BOUNDARY_TEMPERATURE, SdtmPoint, coefficients,
    geopotential_height,
};
use spacelab_propagation::space_weather::{self, SpaceWeatherDay};
use spacelab_propagation::{Nrlmsise00Flags, Nrlmsise00Input, gtd7};

/// Boltzmann's constant, J/K (SI, exact) — `sdtm/mod.rs`'s.
const BOLTZMANN: f64 = 1.380_649e-23;
/// N2's column at the depth, m⁻².
const DEPTH_COLUMN: f64 = 1e21;
/// The column's top, geodetic altitude, m — `derivations/sdtm_column.mac`'s.
const TOP: f64 = 5.0e6;
/// The pieces' upper ends above the level, m; the last piece runs to [`TOP`].
const PIECES: [f64; 5] = [2.0e4, 6.0e4, 1.5e5, 4.0e5, 1.0e6];
const NODES: usize = 32;
/// Nodes on one Newton step, a few kilometres: the column is updated by the step's integral, not recomputed.
const STEP_NODES: usize = 16;
/// [`BOUNDARY`]'s order: N2, O2, O, Ar, He.
const N2: usize = 0;
const O: usize = 2;
/// The gate's values, `T∞` K, depth m, ratio.
#[expect(
    clippy::excessive_precision,
    reason = "derivations/sdtm_column.gp's 60-digit values, to 22 digits"
)]
const GATE: [(f64, f64, f64); 3] = [
    (
        700.0,
        131_917.056_445_192_885_311,
        0.485_971_357_515_928_361_550_7,
    ),
    (
        1000.0,
        133_576.241_735_151_973_527,
        0.487_542_453_257_362_112_562_8,
    ),
    (
        1500.0,
        136_970.768_911_998_138_774,
        0.490_553_275_956_073_607_843_9,
    ),
];

/// Gauss–Legendre nodes and weights on `[−1, 1]`, by Newton on `P_N`.
fn gauss_legendre<const N: usize>() -> ([f64; N], [f64; N]) {
    let order = f64::from(u32::try_from(N).expect("small"));
    let (mut nodes, mut weights) = ([0.0; N], [0.0; N]);
    let legendre = |root: f64| {
        let (mut p0, mut p1) = (1.0, root);
        for degree in 2..=N {
            let degree = f64::from(u32::try_from(degree).expect("small"));
            (p0, p1) = (
                p1,
                ((2.0 * degree - 1.0) * root * p1 - (degree - 1.0) * p0) / degree,
            );
        }
        (p1, order * (root * p1 - p0) / (root * root - 1.0))
    };
    for (index, (node, weight)) in nodes.iter_mut().zip(&mut weights).enumerate() {
        let index = f64::from(u32::try_from(index).expect("small"));
        let mut root = (std::f64::consts::PI * (index + 0.75) / (order + 0.5)).cos();
        for _ in 0..100 {
            let (value, slope) = legendre(root);
            let step = value / slope;
            root -= step;
            if step.abs() < 1e-16 {
                break;
            }
        }
        let (_, slope) = legendre(root);
        *node = root;
        *weight = 2.0 / ((1.0 - root * root) * slope * slope);
    }
    (nodes, weights)
}

/// The column's rule on each piece and the Newton step's.
struct Rules {
    column: ([f64; NODES], [f64; NODES]),
    step: ([f64; STEP_NODES], [f64; STEP_NODES]),
}

/// `[O, N2]` integrals from `low` to `high`, signed, of the densities `density(z) = [n_O, n_N2]` in m⁻³.
fn integral<const N: usize>(
    density: &impl Fn(f64) -> [f64; 2],
    rule: &([f64; N], [f64; N]),
    low: f64,
    high: f64,
) -> [f64; 2] {
    let (middle, half) = (f64::midpoint(low, high), 0.5 * (high - low));
    let mut sum = [0.0; 2];
    for (x, w) in rule.0.iter().zip(&rule.1) {
        let n = density(middle + half * x);
        sum[0] += w * half * n[0];
        sum[1] += w * half * n[1];
    }
    sum
}

/// `[O, N2]` columns above `z`, m⁻².
fn columns(density: &impl Fn(f64) -> [f64; 2], rules: &Rules, z: f64) -> [f64; 2] {
    let mut sum = [0.0; 2];
    let mut low = z;
    for high in PIECES.iter().map(|p| z + p).chain(std::iter::once(TOP)) {
        let piece = integral(density, &rules.column, low, high);
        sum = [sum[0] + piece[0], sum[1] + piece[1]];
        low = high;
    }
    sum
}

/// The depth, m, and the ratio there; `None` if Newton leaves `[floor, 400 km]` or does not converge. The
/// columns are computed once at 140 km and carried: each step subtracts its own integral.
fn depth(density: &impl Fn(f64) -> [f64; 2], rules: &Rules, floor: f64) -> Option<(f64, f64)> {
    let mut z = 140e3;
    let mut column = columns(density, rules, z);
    for _ in 0..60 {
        let step = (column[1] / DEPTH_COLUMN).ln() * column[1] / density(z)[1];
        if !(floor..=4.0e5).contains(&(z + step)) {
            return None;
        }
        let slice = integral(density, &rules.step, z, z + step);
        column = [column[0] - slice[0], column[1] - slice[1]];
        z += step;
        if step.abs() < 1e-7 {
            return Some((z, column[0] / column[1]));
        }
    }
    None
}

/// SDTM-1's `[n_O, n_N2]`, m⁻³, at geodetic altitude `z` for exospheric temperature `t_inf`.
fn sdtm(t_inf: f64, z: f64) -> [f64; 2] {
    let (sigma, scale) = (
        coefficients::FIELD.sigma(),
        coefficients::FIELD.boundary_scale(),
    );
    let zeta = geopotential_height(z);
    let t = t_inf - (t_inf - BOUNDARY_TEMPERATURE) * (-sigma * zeta).exp();
    let integral = (sigma * zeta + (t / BOUNDARY_TEMPERATURE).ln()) / (sigma * t_inf);
    [O, N2].map(|i| {
        let s = BOUNDARY[i];
        scale
            * s.number_density
            * (BOUNDARY_TEMPERATURE / t).powf(1.0 + s.thermal_diffusion)
            * (-s.mass * BOUNDARY_GRAVITY / BOLTZMANN * integral).exp()
    })
}

fn check(rules: &Rules) -> bool {
    let mut ok = true;
    for (t_inf, z_gate, ratio_gate) in GATE {
        let (z, ratio) = depth(&|z| sdtm(t_inf, z), rules, BOUNDARY_ALTITUDE).expect("a depth");
        let (dz, dr) = (z / z_gate - 1.0, ratio / ratio_gate - 1.0);
        let pass = dz.abs() < 1e-12 && dr.abs() < 1e-12;
        ok &= pass;
        println!(
            "T∞ {t_inf} K: depth {z:.6} m ({dz:+.1e}), ratio {ratio:.15} ({dr:+.1e}) against the gate: {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }
    ok
}

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

/// One input line: the civil date, UT seconds, geodetic latitude and longitude in degrees, local solar time.
struct Point {
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
    lat: f64,
    lon: f64,
    lst: f64,
}

fn parse(line: &str) -> Point {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let [year, month, day, rest @ ..] = &fields[..] else {
        panic!("expected 7 fields, got {line:?}");
    };
    let f: Vec<f64> = rest
        .iter()
        .map(|x| x.parse().expect("numeric field"))
        .collect();
    let [sec, lat, lon, lst] = f[..] else {
        panic!("expected 7 fields, got {line:?}");
    };
    Point {
        year: year.parse().expect("integer year"),
        month: month.parse().expect("integer month"),
        day: day.parse().expect("integer day"),
        sec,
        lat,
        lon,
        lst,
    }
}

/// SDTM-1's `(T∞, ratio, height)`: the column ratio and its depth, or the local ratio at `local`.
fn sdtm_at(
    days: &[SpaceWeatherDay],
    p: &Point,
    local: Option<f64>,
    rules: &Rules,
) -> Option<(f64, f64, f64)> {
    let field = coefficients::FIELD;
    let drivers = space_weather::sdtm_drivers::<{ coefficients::HISTORY }>(
        days,
        field.storm(),
        p.year,
        p.month,
        p.day,
        p.sec,
    )?;
    let point = SdtmPoint {
        altitude: 4.0e5,
        latitude: p.lat.to_radians(),
        local_solar_time: p.lst.rem_euclid(24.0),
        day_of_year: f64::from(day_of_year(p.year, p.month, p.day)) + p.sec / 86_400.0,
    };
    let t_inf = field.exospheric_temperature(&point, &drivers);
    match local {
        Some(z) => {
            let [o, n2] = sdtm(t_inf, z);
            Some((t_inf, o / n2, z))
        }
        None => depth(&|z| sdtm(t_inf, z), rules, BOUNDARY_ALTITUDE).map(|(z, r)| (t_inf, r, z)),
    }
}

/// NRLMSISE-00's `(height, ratio)`, as [`sdtm_at`].
fn msis_at(
    days: &[SpaceWeatherDay],
    p: &Point,
    local: Option<f64>,
    rules: &Rules,
) -> Option<(f64, f64)> {
    let mut flags = Nrlmsise00Flags::default();
    flags.switches[9] = -1;
    let w = window(days, p.year, p.month, p.day)?;
    let today = w[4];
    let ap_array = space_weather::nrlmsise_ap_array(w, p.year, p.month, p.day, p.sec)?;
    let msis = |z: f64| {
        let input = Nrlmsise00Input {
            year: p.year,
            doy: day_of_year(p.year, p.month, p.day),
            sec: p.sec,
            alt: z / 1000.0,
            g_lat: p.lat,
            g_long: p.lon,
            lst: p.lst,
            f107a: today.f107_center81,
            f107: w[3].f107_observed,
            ap: today.ap,
            ap_array: Some(ap_array),
        };
        let d = gtd7(&input, &flags).d;
        [d[1] * 1e6, d[2] * 1e6]
    };
    match local {
        Some(z) => {
            let [o, n2] = msis(z);
            Some((z, o / n2))
        }
        None => depth(&msis, rules, 9.0e4),
    }
}

fn main() {
    let rules = Rules {
        column: gauss_legendre(),
        step: gauss_legendre(),
    };
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: column_on2 [--local ALT_KM] SW-All.csv < points | column_on2 --check";
    let (local, path) = match args.get(1).map(String::as_str) {
        Some("--check") => std::process::exit(i32::from(!check(&rules))),
        Some("--local") => (
            Some(
                1000.0
                    * args
                        .get(2)
                        .expect(usage)
                        .parse::<f64>()
                        .expect("altitude, km"),
            ),
            args.get(3).expect(usage),
        ),
        Some(_) => (None, &args[1]),
        None => panic!("{usage}"),
    };
    let days = space_weather::parse_space_weather(
        &std::fs::read_to_string(path).expect("readable SW-All.csv"),
    );
    let mut out = BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let point = parse(&line.expect("readable stdin"));
        let (t_inf, sdtm_ratio, sdtm_z) =
            sdtm_at(&days, &point, local, &rules).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        let (msis_z, msis_ratio) =
            msis_at(&days, &point, local, &rules).unwrap_or((f64::NAN, f64::NAN));
        writeln!(
            out,
            "{t_inf:.3} {sdtm_ratio:.6} {:.4} {msis_ratio:.6} {:.4}",
            sdtm_z / 1000.0,
            msis_z / 1000.0
        )
        .expect("writable stdout");
    }
}
