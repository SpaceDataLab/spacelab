//! `spacelab_estimation::observed` against Orekit's own light-time solution and `Range` model.
//!
//! Orekit implements the IERS Conventions directly and descends from neither SOFA nor Vallado, so
//! it is the third lineage the frame chain and the DSST are already validated against. Here it
//! answers a narrower question: given one trajectory and one station, does our down-leg solution
//! agree with theirs, and does the range that comes out of it agree with the range their
//! measurement model reports?
//!
//! # The experiment isolates one variable, deliberately
//!
//! The orbit is **two-body Keplerian**, propagated in closed form on both sides from the same
//! Cartesian epoch state with the same `mu`. That is not a convenience: with an SGP4 or a numerical
//! orbit the two trajectories would differ by metres, and metres is the size of the effect under
//! test, so a residual could not be attributed. With two-body the trajectories agree to round-off
//! and what is left is the light time.
//!
//! Everything is in **EME2000**, and the station's inertial position and velocity come from the
//! fixture rather than from our own frame chain, for the same reason: a frame difference is worth
//! about a metre here, and it is not what is being measured. (The frame chain has its own oracle —
//! `crates/spacelab-frames/tests/orekit_oracle.rs`.)
//!
//! # What the fixture carries, and what each column is for
//!
//! `tau_resolved` is Orekit's fixed point with the propagator re-evaluated at every iterate;
//! `tau_shifted` is what Orekit's own `Range` uses internally, the emitter Keplerian-shifted from
//! the reception state. The difference between them is Orekit's own shortcut, and the test reports
//! it rather than assuming it negligible: it comes out at **8.0e-8 m**, so the two are one number
//! at this level and our comparison is against either.
//!
//! Regenerate with `tools/orekit-oracle/OrekitLightTimeOracle.java` — see that file's header and
//! `tools/orekit-oracle/README.md` for the jars and their published checksums.

#![allow(clippy::expect_used, clippy::unwrap_used)]
#![allow(clippy::doc_markdown)] // Orekit, EME2000, Keplerian are names in prose

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

/// Before anything is compared, the two trajectories have to *be* one trajectory. Reported first
/// because every other residual below is only meaningful if this one is small: a metre of orbit
/// disagreement would swallow the whole effect.
#[test]
fn the_two_keplerian_propagations_are_the_same_orbit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let mut worst_reception = 0.0_f64;
    let mut worst_emission = 0.0_f64;
    for row in &rows {
        worst_reception = worst_reception.max(separation(&orbit(row.t).unwrap(), &row.reception));
        let emit = row.t - row.tau_resolved;
        worst_emission = worst_emission.max(separation(&orbit(emit).unwrap(), &row.emission));
    }
    println!(
        "two-body agreement: {worst_reception:.3e} m at reception, {worst_emission:.3e} m at emission"
    );
    assert!(
        worst_reception < 1.0e-6 && worst_emission < 1.0e-6,
        "the two propagations are not one orbit: {worst_reception} / {worst_emission} m"
    );
}

/// The flight time itself, against Orekit's fixed point. Also reports the cost of Orekit's own
/// Keplerian-shift shortcut, so that "we agree with Orekit" says which Orekit.
#[test]
fn the_down_leg_agrees_with_orekits_own_fixed_point() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let mut worst = 0.0_f64;
    let mut worst_shortcut = 0.0_f64;
    for row in &rows {
        let station = Station {
            position: [row.station[0], row.station[1], row.station[2]],
            velocity: [row.station[3], row.station[4], row.station[5]],
        };
        let observed = reduce(Signal::down_leg(), row.t, &station, &orbit)
            .expect("a Keplerian orbit reaches every epoch");
        worst = worst.max((observed.transit().delay - row.tau_resolved).abs());
        worst_shortcut = worst_shortcut.max((row.tau_shifted - row.tau_resolved).abs());
    }
    println!(
        "flight time: {:.3e} s ({:.3e} m) from Orekit; its own shift shortcut costs {:.3e} s ({:.3e} m)",
        worst,
        worst * SPEED_OF_LIGHT,
        worst_shortcut,
        worst_shortcut * SPEED_OF_LIGHT
    );
    // A picosecond is 0.3 mm. Two implementations of one fixed point, on one orbit, in one frame,
    // have nothing left to disagree about above that.
    assert!(
        worst * SPEED_OF_LIGHT < 1.0e-3,
        "flight time differs by {worst} s"
    );
}

/// The whole point: the range an orbit determination models. Orekit's one-way `Range` against ours
/// evaluated at the epoch `reduce` chose, and — the number this ticket exists for — against the
/// instantaneous convention that was the only one available before.
#[test]
fn the_modelled_range_agrees_with_orekit_and_the_instantaneous_one_does_not() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let mut worst_light_time = 0.0_f64;
    let mut worst_instantaneous = 0.0_f64;
    let mut worst_two_way = 0.0_f64;
    for row in &rows {
        let station = Station {
            position: [row.station[0], row.station[1], row.station[2]],
            velocity: [row.station[3], row.station[4], row.station[5]],
        };
        let observable = Range { station };

        let observed = reduce(Signal::down_leg(), row.t, &station, &orbit).unwrap();
        let modelled = partials_at_emission(&observable, &observed, [0.0; 3], &[]).value;
        worst_light_time = worst_light_time.max((modelled - row.range_one_way).abs());

        let instantaneous = partials(&observable, &orbit(row.t).unwrap(), &[]).value;
        worst_instantaneous = worst_instantaneous.max((instantaneous - row.range_one_way).abs());
        // The fixture's own instantaneous column, so the comparison is not our arithmetic twice.
        assert!(
            (instantaneous - row.range_instantaneous).abs() < 1.0e-6,
            "our instantaneous range disagrees with Orekit's by {} m",
            (instantaneous - row.range_instantaneous).abs()
        );

        worst_two_way = worst_two_way.max((row.range_two_way - row.range_one_way).abs());
    }
    println!(
        "range: light-time {worst_light_time:.3e} m from Orekit, instantaneous \
         {worst_instantaneous:.4} m; two-way minus one-way {worst_two_way:.4} m"
    );
    assert!(
        worst_light_time < 1.0e-3,
        "the light-time range differs from Orekit by {worst_light_time} m"
    );
    // The defect, measured against a third lineage rather than against ourselves.
    assert!(
        worst_instantaneous > 50.0,
        "the instantaneous convention was only wrong by {worst_instantaneous} m here"
    );
    // Two-way is a *different observable*, not a refinement of this one: the station moves between
    // transmission and reception. `spacelab_slr::two_way_light_time` models it, and the gap is the
    // reason `observed` does not pretend to.
    assert!(
        worst_two_way > 1.0,
        "the two-way range differs from the one-way by only {worst_two_way} m"
    );
}

/// A range-rate is the observable the acceleration argument reaches, so it is the one that shows
/// the light-time correction is not a range-only story. Orekit's one-way `RangeRate` is the
/// reference; ours is evaluated at the emission epoch.
#[test]
fn the_range_rate_at_the_emission_epoch_agrees_with_orekit() {
    let (epoch_state, rows) = fixture();
    let orbit = ephemeris(epoch_state);
    let mut worst = 0.0_f64;
    let mut worst_instantaneous = 0.0_f64;
    for row in &rows {
        let station = Station {
            position: [row.station[0], row.station[1], row.station[2]],
            velocity: [row.station[3], row.station[4], row.station[5]],
        };
        let observable = RangeRate { station };
        let observed = reduce(Signal::down_leg(), row.t, &station, &orbit).unwrap();
        let modelled = partials_at_emission(&observable, &observed, [0.0; 3], &[]).value;
        worst = worst.max((modelled - row.range_rate).abs());
        let instantaneous = partials(&observable, &orbit(row.t).unwrap(), &[]).value;
        worst_instantaneous = worst_instantaneous.max((instantaneous - row.range_rate).abs());
    }
    println!(
        "range rate: light-time {worst:.4e} m/s from Orekit, instantaneous \
         {worst_instantaneous:.4e} m/s"
    );
    assert!(worst < 1.0e-3, "range rate differs by {worst} m/s");
}
