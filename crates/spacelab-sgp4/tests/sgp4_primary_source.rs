//! The published primary source: Vallado's own `tcppver.out`.
//!
//! `tests/sgp4_verification.rs` compares us against numbers produced *today* by the
//! reference implementation. This file compares us against the numbers Vallado
//! *printed* in 2006 alongside AIAA 2006-6753. The two checks answer different
//! questions, and only together do they close the loop:
//!
//! - against the running reference: did we transcribe it correctly?
//! - against the printed file: has the reference itself drifted since publication?
//!
//! Neither establishes that SGP4 describes the sky. Nothing can; every implementation
//! in circulation descends from the same listing.
//!
//! # The floor of this comparison is the printing, not the arithmetic
//!
//! `tcppver.out` prints position to eight decimals of a kilometre and velocity to
//! nine decimals of a kilometre per second. A printed value is therefore only known
//! to within half of its last digit — \\(5\\) nm per component in km, i.e.
//! \\(5\\,\mu\mathrm{m}\\) in metres, and \\(0.5\\,\mu\mathrm{m/s}\\) for velocity.
//! No agreement tighter than that can be claimed from this file however good the
//! implementation is, and the test says so rather than pretending to a tighter bound.
//!
//! Source: `tcppver.out`, redistributed under MIT with `python-sgp4`, which carries
//! Vallado's published output verbatim.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::Satellite;
use spacelab_testkit::nan_max;

const CASES: &str = include_str!("fixtures/sgp4_ver_tle.txt");
const PUBLISHED: &str = include_str!("fixtures/tcppver.out");

/// Half of the last printed digit of a position component, in metres.
///
/// The file prints kilometres to eight decimals: one unit in the last place is
/// \\(10^{-8}\\) km = \\(10\\,\mu\mathrm{m}\\), so a printed component is known to
/// \\(\pm 5\\,\mu\mathrm{m}\\).
const POSITION_PRINT_ULP_M: f64 = 0.5e-8 * 1000.0;

/// Half of the last printed digit of a velocity component, in metres per second.
const VELOCITY_PRINT_ULP_MS: f64 = 0.5e-9 * 1000.0;

/// The print floor for a *vector*: three components each uncertain by half an ulp.
const POSITION_PRINT_FLOOR_M: f64 = 1.732_050_807_568_877_2 * POSITION_PRINT_ULP_M;

/// Agreement bound for the whole file, metres.
///
/// Deliberately the same millimetre used against the running reference. The printed
/// file cannot resolve better than `POSITION_PRINT_FLOOR_M`, but the long-arc
/// deep-space cases diverge by more than that from accumulated rounding in the
/// resonance integrator — so the bound that has to hold is the physical one, and the
/// print floor is reported separately as the level below which the file says nothing.
const TOL_POSITION_M: f64 = 1e-3;

/// Velocity agreement bound, metres per second.
const TOL_VELOCITY_MS: f64 = 1e-6;

/// One printed state: minutes since epoch, position and velocity in metres.
struct Published {
    minutes: f64,
    position: [f64; 3],
    velocity: [f64; 3],
}

/// One catalog section of the file, in file order.
struct Section {
    norad: u32,
    states: Vec<Published>,
}

/// Split `tcppver.out` into its catalog sections.
///
/// The format is a header line `NNNNN xx` followed by one line per state. Every state
/// line begins with seven numbers — the time in minutes and the position and velocity
/// in km and km/s. Most also carry the osculating elements and a calendar date, which
/// we ignore: they are a derived view of the same state.
fn published_sections() -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    for line in PUBLISHED.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() == 2 && fields[1] == "xx" {
            sections.push(Section {
                norad: fields[0].parse().expect("a catalog number"),
                states: Vec::new(),
            });
            continue;
        }
        if fields.len() < 7 {
            continue;
        }
        let v: Vec<f64> = fields[..7]
            .iter()
            .map(|f| f.parse().expect("a printed number"))
            .collect();
        sections
            .last_mut()
            .expect("a state line before any section header")
            .states
            .push(Published {
                minutes: v[0],
                position: [v[1] * 1000.0, v[2] * 1000.0, v[3] * 1000.0],
                velocity: [v[4] * 1000.0, v[5] * 1000.0, v[6] * 1000.0],
            });
    }
    sections
}

/// The element sets, in the same order the published file lists them.
fn cases() -> Vec<(u32, Option<Satellite>)> {
    CASES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split('|').collect();
            let norad = f[1][2..7].trim().parse().expect("a catalog number");
            (norad, Satellite::from_tle_unchecked(f[0], f[1]).ok())
        })
        .collect()
}

/// The published file and the element-set fixture describe the same run, in the same
/// order — including the catalog number that appears twice, with two different spans.
///
/// This is a structural check, and it is the reason the comparison below can zip the
/// two by index rather than keying on the catalog number, which is not unique.
#[test]
fn the_published_file_and_the_element_sets_are_the_same_run() {
    let sections = published_sections();
    let cases = cases();

    assert_eq!(
        sections.len(),
        cases.len(),
        "the published file has {} sections but the fixture has {} element sets",
        sections.len(),
        cases.len(),
    );
    for (i, (section, (norad, _))) in sections.iter().zip(cases.iter()).enumerate() {
        assert_eq!(
            section.norad, *norad,
            "section {i} is catalog {} in the published file and {norad} in the fixture",
            section.norad,
        );
    }

    let duplicated = sections
        .iter()
        .enumerate()
        .filter(|(i, s)| sections[..*i].iter().any(|p| p.norad == s.norad))
        .count();
    assert_eq!(
        duplicated, 1,
        "catalog 20413 is published twice, with a short and a long arc; \
         found {duplicated} repeated sections",
    );
}

#[test]
fn we_reproduce_the_published_output_to_the_precision_it_was_printed_at() {
    let sections = published_sections();
    let cases = cases();

    let mut compared = 0_usize;
    let mut within_print_floor = 0_usize;
    let mut refused_objects = 0_usize;
    let mut refused_states = 0_usize;
    let mut worst_pos = 0.0_f64;
    let mut worst_vel = 0.0_f64;
    let mut worst_at = (0_u32, 0.0_f64);
    let mut sum_sq = 0.0_f64;

    for (section, (_, satellite)) in sections.iter().zip(cases.iter()) {
        let Some(satellite) = satellite else {
            refused_objects += 1;
            continue;
        };
        for state in &section.states {
            // Where we refuse to answer — the mean elements left the model's domain,
            // or the object re-entered — the published file still prints a number,
            // because Vallado's driver prints the buffer regardless of the error flag.
            // Comparing against it would be comparing against uninitialised arithmetic.
            let Ok(ours) = satellite.propagate_minutes(state.minutes) else {
                refused_states += 1;
                continue;
            };

            let dp = ((ours.position.vector.x - state.position[0]).powi(2)
                + (ours.position.vector.y - state.position[1]).powi(2)
                + (ours.position.vector.z - state.position[2]).powi(2))
            .sqrt();
            let dv = ((ours.velocity.vector.x - state.velocity[0]).powi(2)
                + (ours.velocity.vector.y - state.velocity[1]).powi(2)
                + (ours.velocity.vector.z - state.velocity[2]).powi(2))
            .sqrt();

            compared += 1;
            sum_sq += dp * dp;
            if dp <= POSITION_PRINT_FLOOR_M {
                within_print_floor += 1;
            }
            if dp > worst_pos {
                worst_pos = dp;
                worst_at = (section.norad, state.minutes);
            }
            worst_vel = nan_max(worst_vel, dv);
        }
    }

    // Counts of a few hundred; the cast is exact well past any plausible fixture size.
    #[allow(clippy::cast_precision_loss)]
    let (rms, inside) = (
        (sum_sq / compared as f64).sqrt(),
        100.0 * within_print_floor as f64 / compared as f64,
    );
    println!(
        "tcppver.out: {compared} states compared, {refused_objects} element sets refused, \
         {refused_states} states we decline to answer.\n\
         worst position {:.3} µm (catalog {} at {} min), RMS {:.3} µm, worst velocity {:.4} µm/s.\n\
         print floor {:.3} µm position / {:.3} µm/s velocity per component; \
         {within_print_floor} states ({inside:.1}%) agree to within it.",
        worst_pos * 1e6,
        worst_at.0,
        worst_at.1,
        rms * 1e6,
        worst_vel * 1e6,
        POSITION_PRINT_FLOOR_M * 1e6,
        VELOCITY_PRINT_ULP_MS * 1e6,
    );

    assert!(
        compared > 600,
        "only {compared} states compared; the published file should yield far more",
    );
    assert!(
        worst_pos < TOL_POSITION_M,
        "worst position residual against the published output {worst_pos:.3e} m \
         exceeds {TOL_POSITION_M:.0e} m",
    );
    assert!(
        worst_vel < TOL_VELOCITY_MS,
        "worst velocity residual against the published output {worst_vel:.3e} m/s \
         exceeds {TOL_VELOCITY_MS:.0e} m/s",
    );
    // Most of the file is short-arc, and there the only thing separating us from the
    // 2006 printout is the printing itself. If that fraction collapses, something has
    // drifted even though the millimetre bound above would still pass.
    assert!(
        inside > 80.0,
        "only {inside:.1}% of states agree to the printed precision; \
         the published output and this implementation have diverged",
    );
}
