//! The DSST mean state of an ILRS prediction file (`plans/17` Ф23.U): the ANDE spheres have no element
//! sets in the historical archive the lifetime campaign reads (planet4589.org), so they enter `tools/lifetime-campaign.py spheres` through their laser
//! predictions (CPF, positions in ITRF every 60 s).
//!
//!     cargo run -p spacelab --release --example cpf_mean_state -- FILE ...
//!     cargo run -p spacelab --release --example cpf_mean_state -- --check
//!
//! For each file it prints `file unix a e i raan argp M`: the state at the file's fifth node, mean,
//! in TEME, metres and radians, as the lifetime campaign reads an element set.
//!
//! ```text
//! CPF ─nine-node central difference─▶ ITRF velocity ─EOP, Vallado (C-1)─▶ TEME
//!     ─▶ osculating elements ─zonal short period, inverted─▶ mean
//! ```
//!
//! The velocity's weights and their error, below 1e-8 m/s at 60 s, are
//! `derivations/central_difference.mac`. The mean state is the fixed point of
//! `mean ← mean + (osculating − η(mean))`, `η` the zonal short-period map of the J2–J6 that
//! `spacelab-propagation`'s `lifetime_campaign` example integrates, with its constants: the mean state
//! is that worker's own. `--check` runs the chain backwards on a known mean state.

#![allow(clippy::expect_used)] // an example that cannot run should say so loudly

use std::f64::consts::{PI, TAU};

use spacelab::{EopSource, eop::EopSeries, teme_to_itrf_at};
use spacelab_core::{Angle, UtcDateTime, Vec3};
use spacelab_elements::KeplerianElements;
use spacelab_frames::{Itrf, StateVector};
use spacelab_propagation::dsst::zonal::DsstZonal;

/// As `lifetime_campaign`: EGM2008, tide-free, degrees 2–6.
const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
const C_BAR: [f64; 5] = [
    -0.484_165_143_790_815e-3,
    0.957_161_207_093_473e-6,
    0.539_965_866_638_991e-6,
    0.686_702_913_736_681e-7,
    -0.149_953_927_978_527e-6,
];
/// `derivations/central_difference.mac`, (1).
const WEIGHTS: [f64; 9] = [
    1.0 / 280.0,
    -4.0 / 105.0,
    1.0 / 5.0,
    -4.0 / 5.0,
    0.0,
    4.0 / 5.0,
    -1.0 / 5.0,
    4.0 / 105.0,
    -1.0 / 280.0,
];
const PASSES: usize = 50;
const MEAN_TOLERANCE: f64 = 1e-13;
const MJD_UNIX: i32 = 40_587;
const MJD_J2000: i32 = 51_544;

struct Node {
    mjd: i32,
    sod: f64,
    position: Vec3,
}

/// The step, s, and the first nine nodes. Panics on anything but ITRF positions at a fixed step with
/// no leap second among the nine.
fn read_cpf(text: &str) -> (f64, Vec<Node>) {
    let mut step = None;
    let mut nodes = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split_whitespace().collect();
        match cells.first() {
            Some(&"H2") => {
                assert_eq!(cells[19], "0", "CPF reference frame 0, ITRF");
                step = Some(cells[16].parse::<f64>().expect("a step"));
            }
            Some(&"10") if nodes.len() < WEIGHTS.len() => {
                assert_eq!(cells[4], "0", "no leap second among the nodes");
                let number = |k: usize| cells[k].parse::<f64>().expect("a number");
                nodes.push(Node {
                    mjd: cells[2].parse().expect("an MJD"),
                    sod: number(3),
                    position: Vec3::new(number(5), number(6), number(7)),
                });
            }
            _ => {}
        }
    }
    let step = step.expect("an H2 record");
    assert_eq!(nodes.len(), WEIGHTS.len(), "nine nodes");
    for pair in nodes.windows(2) {
        let gap = f64::from(pair[1].mjd - pair[0].mjd) * 86_400.0 + pair[1].sod - pair[0].sod;
        assert!((gap - step).abs() < 1e-6, "nodes at the file's step");
    }
    (step, nodes)
}

fn central_difference(positions: &[Vec3], step: f64) -> Vec3 {
    let mut v = Vec3::new(0.0, 0.0, 0.0);
    for (w, p) in WEIGHTS.iter().zip(positions) {
        v = v + p.scale(*w);
    }
    v.scale(1.0 / step)
}

fn wrap(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

fn equinoctial(k: &KeplerianElements) -> [f64; 6] {
    let (raan, argp) = (k.raan.radians(), k.arg_periapsis.radians());
    let half = (k.inclination.radians() / 2.0).tan();
    [
        k.semi_major_axis,
        k.eccentricity * (argp + raan).cos(),
        k.eccentricity * (argp + raan).sin(),
        half * raan.cos(),
        half * raan.sin(),
        k.mean_anomaly().radians() + argp + raan,
    ]
}

/// `[a, e, i, raan, argp, M]` of an equinoctial `[a, k, h, q, p, λ]`, angles in `[0, 2π)`.
fn classical(s: [f64; 6]) -> [f64; 6] {
    let raan = s[4].atan2(s[3]);
    let longitude = s[2].atan2(s[1]);
    [
        s[0],
        s[1].hypot(s[2]),
        2.0 * s[3].hypot(s[4]).atan(),
        raan.rem_euclid(TAU),
        (longitude - raan).rem_euclid(TAU),
        (s[5] - longitude).rem_euclid(TAU),
    ]
}

fn mean_of(zonal: &DsstZonal, osculating: [f64; 6]) -> [f64; 6] {
    let mut mean = osculating;
    for _ in 0..PASSES {
        let image = zonal.osculating(mean);
        let mut step = 0.0_f64;
        for i in 0..6 {
            let delta = if i == 5 {
                wrap(osculating[i] - image[i])
            } else {
                osculating[i] - image[i]
            };
            mean[i] += delta;
            step = step.max(delta.abs() / if i == 0 { osculating[0] } else { 1.0 });
        }
        if step <= MEAN_TOLERANCE {
            return mean;
        }
    }
    panic!("the zonal short-period map did not converge");
}

fn zonal() -> DsstZonal {
    let j: Vec<f64> = (2..)
        .zip(C_BAR)
        .map(|(n, c): (u32, f64)| -f64::from(2 * n + 1).sqrt() * c)
        .collect();
    DsstZonal::from_j(MU, RE, &j)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a second of the day in nanoseconds is below 2^47"
)]
fn utc(mjd: i32, sod: f64) -> UtcDateTime {
    let ns = i128::from(mjd - MJD_J2000) * 86_400_000_000_000 - 43_200_000_000_000
        + (sod * 1e9).round() as i128;
    UtcDateTime::from_naive_ns_since_j2000(ns, false)
}

fn eop() -> EopSeries {
    let snapshot = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/eop");
    let path = std::fs::read_dir(&snapshot)
        .unwrap_or_else(|e| panic!("run tools/fetch-eop.sh first ({}): {e}", snapshot.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "txt"))
        .expect("an Earth-orientation snapshot");
    EopSeries::parse_finals2000a(&std::fs::read_to_string(&path).expect("readable"))
        .expect("the published series parses")
}

/// A known mean state, through `η` to osculating, to a Kepler arc sampled at 60 s, back through the
/// central difference and the inversion.
fn check(zonal: &DsstZonal) {
    let mean = [6_778_000.0, 0.0012, -0.0009, 0.21, 0.37, 2.4];
    let osculating = zonal.osculating(mean);
    let elements = classical(osculating);
    let (position, true_velocity) = KeplerianElements {
        semi_major_axis: elements[0],
        eccentricity: elements[1],
        inclination: Angle::from_radians(elements[2]),
        raan: Angle::from_radians(elements[3]),
        arg_periapsis: Angle::from_radians(elements[4]),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    }
    .at_mean_anomaly(Angle::from_radians(elements[5]))
    .to_state();
    let arc: Vec<Vec3> = (-4..=4)
        .map(|j| {
            spacelab_astro::kepler::propagate(position, true_velocity, MU, 60.0 * f64::from(j)).0
        })
        .collect();
    let velocity = central_difference(&arc, 60.0);
    let miss_v = (velocity - true_velocity).norm();
    let back = mean_of(
        zonal,
        equinoctial(&KeplerianElements::from_state(position, velocity, MU)),
    );
    let miss_a = (back[0] - mean[0]).abs();
    let miss_angle = (1..6)
        .map(|i| wrap(back[i] - mean[i]).abs())
        .fold(0.0, f64::max);
    println!(
        "velocity miss {miss_v:.3e} m/s; mean a miss {miss_a:.3e} m; largest other miss {miss_angle:.3e}"
    );
    assert!(
        miss_v < 1e-6 && miss_a < 1e-3 && miss_angle < 1e-9,
        "--check failed"
    );
    println!("check: PASS");
}

fn main() {
    let zonal = zonal();
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.first().is_some_and(|arg| arg == "--check") {
        check(&zonal);
        return;
    }
    let series = eop();
    for path in &paths {
        let (step, nodes) = read_cpf(&std::fs::read_to_string(path).expect("a CPF file"));
        let centre = &nodes[4];
        let positions: Vec<Vec3> = nodes.iter().map(|n| n.position).collect();
        let itrf = StateVector::<Itrf>::new(centre.position, central_difference(&positions, step));
        let (transform, _) =
            teme_to_itrf_at(utc(centre.mjd, centre.sod), EopSource::Series(&series))
                .expect("Earth orientation at the node");
        let teme = transform.inverse().state(itrf);
        let osculating = equinoctial(&KeplerianElements::from_state(
            teme.position.vector,
            teme.velocity.vector,
            MU,
        ));
        let [a, e, i, raan, argp, m] = classical(mean_of(&zonal, osculating));
        let unix = f64::from(centre.mjd - MJD_UNIX) * 86_400.0 + centre.sod;
        println!("{path}\t{unix:.3}\t{a}\t{e}\t{i}\t{raan}\t{argp}\t{m}");
    }
}
