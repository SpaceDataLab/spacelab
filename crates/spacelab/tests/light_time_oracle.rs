//! `spacelab_estimation::observed`'s down leg, and the range and range rate it feeds, on a
//! two-body orbit.
//!
//! `derivations/light_time_down_leg.mac` decides (plans/17 Ф19.G): the orbit by Lagrange's f and g
//! at 40 digits, the flight time as the fixed point of `cτ = |r(t_r − τ) − s(t_r)|`, and the range
//! and range rate at the emission state, at five reception epochs. Orekit's own light-time solution
//! and its `Range` and `RangeRate` models decided these tests until then; they are a comparator
//! now, parked below and published in `docs/comparisons.md`.
//!
//! # The comparison isolates one variable, deliberately
//!
//! The orbit is **two-body Keplerian**, propagated in closed form on both sides from the same
//! Cartesian epoch state with the same `mu`. That is not a convenience: with an SGP4 or a numerical
//! orbit the two trajectories would differ by metres, and metres is the size of the effect under
//! test, so a residual could not be attributed. With two-body the trajectories agree to round-off
//! and what is left is the light time.
//!
//! Everything is in **EME2000**, and the station's inertial position and velocity come from the
//! fixture rather than from our own frame chain, for the same reason: a frame difference is worth
//! about a metre here, and it is not what is being measured. (The frame chain is compared with
//! Orekit on its own, in `crates/spacelab-frames/tests/orekit_oracle.rs`.)
//!
//! # What the fixture carries, and what each column is for
//!
//! `tau_resolved` is Orekit's fixed point with the propagator re-evaluated at every iterate;
//! `tau_shifted` is what Orekit's own `Range` uses internally, the emitter Keplerian-shifted from
//! the reception state. The difference between them is Orekit's own shortcut, and the flight-time
//! row reports it rather than assuming it negligible: it comes out at **8.0e-8 m**, so the two are
//! one number at this level and the comparison is against either.
//!
//! The derivation reads only the fixture's inputs, the epoch state and the station's states, as the
//! doubles Rust parses them to; none of Orekit's outputs.
//!
//! Regenerate with `tools/orekit-oracle/OrekitLightTimeOracle.java` — see that file's header and
//! `tools/orekit-oracle/README.md` for the jars and their published checksums.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)] // Orekit, EME2000, Keplerian are names in prose
#![allow(clippy::excessive_precision)] // the goldens carry Maxima's 20 digits, as derived

use spacelab_astro::propagate as two_body;
use spacelab_core::Vec3;
use spacelab_estimation::light_time::SPEED_OF_LIGHT;
use spacelab_estimation::measurement::Station;
use spacelab_estimation::modifier::{Range, RangeRate, partials};
use spacelab_estimation::observed::{Signal, partials_at_emission, reduce};

/// The gravitational parameter the oracle was generated with, to the digit.
const MU: f64 = 3.986_004_418e14;

/// One row of the fixture: everything Orekit reported for one reception epoch.
struct Row {
    t: f64,
    station: [f64; 6],
    reception: [f64; 6],
    emission: [f64; 6],
    tau_shifted: f64,
    tau_resolved: f64,
    range_one_way: f64,
    range_two_way: f64,
    range_rate: f64,
    range_instantaneous: f64,
}

/// The epoch state, and every row, as the oracle wrote them.
fn fixture() -> ([f64; 6], Vec<Row>) {
    let text = include_str!("fixtures/orekit_light_time.csv");
    let mut epoch_state = [0.0; 6];
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# epoch_state=") {
            for (slot, field) in epoch_state.iter_mut().zip(rest.split(',')) {
                *slot = field.trim().parse().expect("epoch state field");
            }
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse().expect("fixture field"))
            .collect();
        assert_eq!(f.len(), 25, "fixture row width");
        rows.push(Row {
            t: f[0],
            station: [f[1], f[2], f[3], f[4], f[5], f[6]],
            reception: [f[7], f[8], f[9], f[10], f[11], f[12]],
            emission: [f[13], f[14], f[15], f[16], f[17], f[18]],
            tau_shifted: f[19],
            tau_resolved: f[20],
            range_one_way: f[21],
            range_two_way: f[22],
            range_rate: f[23],
            range_instantaneous: f[24],
        });
    }
    assert_eq!(rows.len(), 470, "fixture rows");
    (epoch_state, rows)
}

/// Our own two-body propagation from the epoch state, as the ephemeris `reduce` walks.
fn ephemeris(epoch_state: [f64; 6]) -> impl Fn(f64) -> Option<[f64; 6]> {
    move |t: f64| {
        let (r, v) = two_body(
            Vec3::new(epoch_state[0], epoch_state[1], epoch_state[2]),
            Vec3::new(epoch_state[3], epoch_state[4], epoch_state[5]),
            MU,
            t,
        );
        Some([r.x, r.y, r.z, v.x, v.y, v.z])
    }
}

fn separation(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn station_of(row: &Row) -> Station {
    Station {
        position: [row.station[0], row.station[1], row.station[2]],
        velocity: [row.station[3], row.station[4], row.station[5]],
    }
}

/// The fixture rows the goldens are taken at: reception epochs 0, 11 740, 23 960, 35 710 and
/// 86 170 s past the epoch state, five passes over the day.
const GOLDEN_ROWS: [usize; 5] = [0, 117, 234, 351, 469];

/// `derivations/light_time_down_leg.mac` at 40 digits, the fixture's inputs taken as the doubles
/// they parse to: the flight time `τ`, s.
const TAU_MAXIMA: [f64; 5] = [
    7.394_662_066_087_709_782_4e-3,
    5.504_793_718_585_873_746e-3,
    6.226_590_286_126_634_527_3e-3,
    5.887_070_072_865_112_475_6e-3,
    9.021_780_083_327_492_286_5e-3,
];
/// The range at the emission state, `cτ`, m.
const RANGE_MAXIMA: [f64; 5] = [
    2.216_863_916_871_792_959_2e6,
    1.650_295_639_677_819_374_4e6,
    1.866_684_806_836_827_064_2e6,
    1.764_899_207_562_471_171_5e6,
    2.704_661_626_716_193_731_6e6,
];
/// The range rate at the emission state, `(r_e − s)·(v_e − ṡ)/|r_e − s|`, m/s.
const RANGE_RATE_MAXIMA: [f64; 5] = [
    5.068_324_136_374_337_697_6e3,
    8.691_218_517_585_328_196_6e1,
    1.412_089_375_701_055_839_3e3,
    -5.269_198_033_209_631_035_3e3,
    5.663_608_016_436_142_451_8e3,
];
/// The instantaneous range `|r(t_r) − s|` the light time replaces, m.
const INSTANTANEOUS_RANGE_MAXIMA: [f64; 5] = [
    2.216_904_106_176_157_436_4e6,
    1.650_295_308_446_732_734_7e6,
    1.866_694_867_956_533_410_9e6,
    1.764_866_071_155_877_521e6,
    2.704_715_869_022_666_765_1e6,
];

/// (G3) The down leg against `derivations/light_time_down_leg.mac`: the flight time, the range and
/// the range rate at the emission state, and the instantaneous range, each against its 40-digit
/// value on the exact two-body orbit.
#[test]
fn the_down_leg_is_the_light_time_fixed_point_on_the_exact_orbit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let mut worst = [0.0_f64; 4];
    let mut convention = 0.0_f64;
    for (k, &index) in GOLDEN_ROWS.iter().enumerate() {
        let row = &rows[index];
        let station = station_of(row);
        let observed = reduce(Signal::down_leg(), row.t, &station, &orbit)
            .expect("a Keplerian orbit reaches every epoch");
        let range = partials_at_emission(&Range { station }, &observed, [0.0; 3], &[]).value;
        let rate = partials_at_emission(&RangeRate { station }, &observed, [0.0; 3], &[]).value;
        let now = partials(&Range { station }, &orbit(row.t).unwrap(), &[]).value;
        for (slot, d) in worst.iter_mut().zip([
            observed.transit().delay - TAU_MAXIMA[k],
            range - RANGE_MAXIMA[k],
            rate - RANGE_RATE_MAXIMA[k],
            now - INSTANTANEOUS_RANGE_MAXIMA[k],
        ]) {
            *slot = slot.max(d.abs());
        }
        convention = convention.max((INSTANTANEOUS_RANGE_MAXIMA[k] - RANGE_MAXIMA[k]).abs());
    }
    let [tau, range, rate, instantaneous] = worst;
    println!(
        "against the 40-digit values: flight time {tau:.3e} s, range {range:.3e} m, range rate \
         {rate:.3e} m/s, instantaneous range {instantaneous:.3e} m; the instantaneous convention \
         is {convention:.1} m from the light-time range"
    );
    // Measured 1.1e-16 s, 3.3e-8 m, 1.7e-10 m/s and 1.3e-7 m: the f64 two-body propagation over a
    // day, fifteen revolutions, not the light time, whose iteration stops at 1e-15 s.
    assert!(tau < 5.0e-16, "flight time off by {tau} s");
    assert!(range < 1.5e-7, "range off by {range} m");
    assert!(rate < 8.0e-10, "range rate off by {rate} m/s");
    assert!(
        instantaneous < 5.0e-7,
        "instantaneous range off by {instantaneous} m"
    );
}

fn comparison_row(quantity: &str, unit: &str, distances: &[f64]) {
    let max = distances.iter().copied().fold(0.0, f64::max);
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | Orekit | 13.1.7 | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// The distance from our two-body propagation to Orekit's, at reception and at emission: every
/// row below is only meaningful while this one is small.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn two_body_distance_to_orekit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let reception: Vec<f64> = rows
        .iter()
        .map(|row| separation(&orbit(row.t).unwrap(), &row.reception))
        .collect();
    let emission: Vec<f64> = rows
        .iter()
        .map(|row| separation(&orbit(row.t - row.tau_resolved).unwrap(), &row.emission))
        .collect();
    comparison_row("two-body position at reception", "m", &reception);
    comparison_row("two-body position at emission", "m", &emission);
}

/// The distance to Orekit's flight time, its fixed point with the propagator re-evaluated at every
/// iterate; Orekit's own Keplerian-shift shortcut is reported beside it, so the row says which
/// Orekit.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn flight_time_distance_to_orekit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let mut distances = Vec::new();
    let mut shortcut = 0.0_f64;
    for row in &rows {
        let observed = reduce(Signal::down_leg(), row.t, &station_of(row), &orbit)
            .expect("a Keplerian orbit reaches every epoch");
        distances.push((observed.transit().delay - row.tau_resolved).abs());
        shortcut = shortcut.max((row.tau_shifted - row.tau_resolved).abs());
    }
    comparison_row("down-leg flight time", "s", &distances);
    println!(
        "Orekit's shift shortcut against its own fixed point: {shortcut:.3e} s ({:.3e} m)",
        shortcut * SPEED_OF_LIGHT
    );
}

/// The distance to Orekit's one-way `Range`, ours at the emission epoch and, for scale, the
/// instantaneous convention; Orekit's two-way range is a different observable and is reported
/// beside them.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn range_distance_to_orekit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let (mut light_time, mut instantaneous, mut same_convention) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut two_way = 0.0_f64;
    for row in &rows {
        let station = station_of(row);
        let observed = reduce(Signal::down_leg(), row.t, &station, &orbit).unwrap();
        let modelled = partials_at_emission(&Range { station }, &observed, [0.0; 3], &[]).value;
        light_time.push((modelled - row.range_one_way).abs());
        let now = partials(&Range { station }, &orbit(row.t).unwrap(), &[]).value;
        instantaneous.push((now - row.range_one_way).abs());
        same_convention.push((now - row.range_instantaneous).abs());
        two_way = two_way.max((row.range_two_way - row.range_one_way).abs());
    }
    comparison_row("one-way range, light-time", "m", &light_time);
    comparison_row(
        "one-way range, instantaneous convention",
        "m",
        &instantaneous,
    );
    comparison_row(
        "instantaneous range, against Orekit's instantaneous",
        "m",
        &same_convention,
    );
    println!("Orekit's two-way range minus its one-way: {two_way:.4} m");
}

/// The distance to Orekit's one-way `RangeRate`, ours at the emission epoch and, for scale, the
/// instantaneous convention.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn range_rate_distance_to_orekit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let (mut light_time, mut instantaneous) = (Vec::new(), Vec::new());
    for row in &rows {
        let station = station_of(row);
        let observed = reduce(Signal::down_leg(), row.t, &station, &orbit).unwrap();
        let modelled = partials_at_emission(&RangeRate { station }, &observed, [0.0; 3], &[]).value;
        light_time.push((modelled - row.range_rate).abs());
        let now = partials(&RangeRate { station }, &orbit(row.t).unwrap(), &[]).value;
        instantaneous.push((now - row.range_rate).abs());
    }
    comparison_row("one-way range rate, light-time", "m/s", &light_time);
    comparison_row(
        "one-way range rate, instantaneous convention",
        "m/s",
        &instantaneous,
    );
}
