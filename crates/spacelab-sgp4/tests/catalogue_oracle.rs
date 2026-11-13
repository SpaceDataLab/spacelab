//! Comparator: the whole catalogue against satellite.js, re-measured offline from the committed
//! fixture. Benchmark material, not a gate. SGP4 is decided by its document:
//! `derivations/sgp4_document.mac` runs the document's routines at 40 digits and reproduces the
//! published `tcppver.out` (G5), `tests/sgp4_verification.rs` and `tests/sgp4_primary_source.rs`
//! hold the crate to those printed numbers, and `tests/document_refusals.rs` to every refusal the
//! document's error tests make.
//!
//! `tests/sgp4_verification.rs` runs the published set: 31 element sets, chosen by Vallado to
//! exercise every branch, compared against `tcppver.out`. That is the specification bar and it is
//! narrow by design. What it does not answer is whether the propagator is right on the *catalogue* —
//! 16,000 element sets whose eccentricities, drag terms and epoch ages are whatever the fitting
//! process produced that day, including the objects nobody would pick as a test case.
//!
//! `tools/sgp4-catalogue-oracle/run.sh` answers that offline. It propagates Celestrak's `GROUP=active`
//! catalogue with satellite.js and with this crate at 5-minute steps across ±24 hours of each
//! object's epoch, and writes what it found into `fixtures/sgp4_catalogue_oracle.txt`: the sweep's
//! own counts, the decade histogram of the position residual, and the worst objects with the
//! oracle's answers for them. This file re-runs our half of the comparison against those committed
//! answers, so the bar is checked on every `cargo test` with no network, no Node and no oracle.
//!
//! ## The measurement, as of the committed fixture
//!
//! Catalogue retrieved 2026-08-15; 16,085 objects, 9,281,045 samples compared. **Worst position
//! difference 4.220168e-5 m, worst velocity difference 7.718614e-9 m/s.** 92% of samples land in the
//! two decades between 1 nm and 100 nm, and 4% agree to the last bit. Neither implementation
//! rejected an element set the other accepted, and neither refused a propagation the other
//! completed — over nine million opportunities to disagree about the domain, and none taken.
//!
//! ## What that does and does not establish
//!
//! satellite.js is a JavaScript transcription of the same Vallado listing the `sgp4` crate ports, so
//! this is a transcription check across the catalogue and not an independent confirmation of the
//! model. `src/lib.rs` states the limit for SGP4 generally: there is no independent implementation
//! to be had, because the specification *is* a particular listing. Forty micrometres over a day is
//! evidence that both ports read the same source correctly at every branch the catalogue reaches —
//! which is exactly the failure mode a 31-case set can miss and this cannot.
//!
//! The residual's *size* is worth naming rather than admiring: it is accumulated floating-point
//! difference, not modelling difference. The two implementations evaluate the same expressions in
//! different orders, in one case through JavaScript's `Math`, and 4e-5 m is what that costs at the
//! extreme of a 16,000-object sweep.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::{Convention, Regime, Satellite};

const FIXTURE: &str = include_str!("fixtures/sgp4_catalogue_oracle.txt");

/// One object from the fixture: its element set and the oracle's answers for it.
struct Case {
    norad_id: u32,
    regime: String,
    worst_position_m: f64,
    line1: String,
    line2: String,
    samples: Vec<(f64, [f64; 3], [f64; 3])>,
}

fn parse() -> (Vec<(String, String)>, Vec<Case>) {
    let mut header = Vec::new();
    let mut cases: Vec<Case> = Vec::new();
    for line in FIXTURE.lines() {
        if let Some(rest) = line.strip_prefix("#: ") {
            let (key, value) = rest.split_once(' ').expect("header line is `key value`");
            header.push((key.to_owned(), value.to_owned()));
        } else if let Some(rest) = line.strip_prefix("T ") {
            let mut f = rest.split_ascii_whitespace();
            cases.push(Case {
                norad_id: f.next().expect("catalog number").parse().expect("numeric"),
                regime: f.next().expect("regime").to_owned(),
                worst_position_m: f.next().expect("worst position").parse().expect("numeric"),
                line1: String::new(),
                line2: String::new(),
                samples: Vec::new(),
            });
        } else if let Some(rest) = line.strip_prefix("S ") {
            let mut f = rest
                .split_ascii_whitespace()
                .map(|t| t.parse::<f64>().expect("sample field is numeric"));
            let minutes = f.next().expect("offset");
            let r = [
                f.next().expect("x"),
                f.next().expect("y"),
                f.next().expect("z"),
            ];
            let v = [
                f.next().expect("vx"),
                f.next().expect("vy"),
                f.next().expect("vz"),
            ];
            cases
                .last_mut()
                .expect("a case is open")
                .samples
                .push((minutes, r, v));
        } else if line.starts_with("1 ") {
            line.clone_into(&mut cases.last_mut().expect("a case is open").line1);
        } else if line.starts_with("2 ") {
            line.clone_into(&mut cases.last_mut().expect("a case is open").line2);
        }
    }
    (header, cases)
}

fn header_value<'a>(header: &'a [(String, String)], key: &str) -> &'a str {
    header.iter().find(|(name, _)| name == key).map_or_else(
        || panic!("fixture header has no `{key}` line"),
        |(_, value)| value.as_str(),
    )
}

/// The distances to satellite.js on the catalogue's worst objects.
///
/// The cases are the tail of the distribution, not a sample of it: whatever disagrees most across
/// nine million samples is what is committed here.
#[test]
#[ignore = "comparator: the distance to satellite.js, not a verdict"]
fn worst_objects_distance_to_satellite_js() {
    let (_, cases) = parse();
    assert!(cases.len() >= 40, "fixture carries {} cases", cases.len());

    let (mut positions, mut velocities) = (Vec::new(), Vec::new());
    for case in &cases {
        let satellite = Satellite::from_tle(&case.line1, &case.line2)
            .unwrap_or_else(|e| panic!("catalog {}: {e}", case.norad_id));
        assert_eq!(
            u64::from(satellite.norad_id().get()),
            u64::from(case.norad_id),
            "the fixture's catalog number and its element set disagree"
        );
        for &(minutes, r, v) in &case.samples {
            let state = satellite
                .propagate_minutes(minutes)
                .unwrap_or_else(|e| panic!("catalog {} at {minutes} min: {e}", case.norad_id));
            positions.push(distance(state.position.vector, r));
            velocities.push(distance(state.velocity.vector, v));
        }
    }
    comparison_row("position on the catalogue's worst objects", "m", &positions);
    comparison_row(
        "velocity on the catalogue's worst objects",
        "m/s",
        &velocities,
    );
}

/// `|ours − theirs|` in SI units, the fixture's vector being in kilometres (per second).
fn distance(ours: spacelab_core::Vec3, theirs_km: [f64; 3]) -> f64 {
    ((ours.x - theirs_km[0] * 1.0e3).powi(2)
        + (ours.y - theirs_km[1] * 1.0e3).powi(2)
        + (ours.z - theirs_km[2] * 1.0e3).powi(2))
    .sqrt()
}

/// One `comparison | ...` row of `tools/compare.py`.
fn comparison_row(quantity: &str, unit: &str, distances: &[f64]) {
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of a few hundred distances"
    )]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | satellite.js | 4.1.4 | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// The fixture's per-case worst is an upper bound on what these seven offsets can show. (S2)
///
/// The recorded number is the object's worst over the full 577-point sweep; the committed samples
/// are seven of those points. So re-measuring on the subset must land at or below the record. A
/// violation means the fixture and the element set beside it came from different runs.
#[test]
fn the_recorded_worst_bounds_what_the_committed_offsets_reproduce() {
    let (_, cases) = parse();
    for case in &cases {
        let satellite = Satellite::from_tle(&case.line1, &case.line2).expect("element set parses");
        for &(minutes, r, _) in &case.samples {
            let state = satellite.propagate_minutes(minutes).expect("propagates");
            let dp = ((state.position.vector.x - r[0] * 1.0e3).powi(2)
                + (state.position.vector.y - r[1] * 1.0e3).powi(2)
                + (state.position.vector.z - r[2] * 1.0e3).powi(2))
            .sqrt();
            assert!(
                dp <= case.worst_position_m * (1.0 + 1.0e-9),
                "catalog {} at {minutes} min: {dp:.6e} m exceeds the recorded worst {:.6e} m",
                case.norad_id,
                case.worst_position_m
            );
        }
    }
}

/// Every SGP4 branch is represented, and the fixture says so in its own words. (S2)
///
/// A "whole catalogue" sweep that happened to commit 49 near-Earth objects would check one arm of a
/// two-algorithm model. The generator forces in the worst object of each [`Regime`] for exactly this
/// reason, and this test is what would notice if that stopped happening — the half-day resonance in
/// particular, which is one object in the fixture and the rarest branch in the catalogue.
#[test]
fn all_four_regimes_reach_the_committed_fixture() {
    let (_, cases) = parse();
    for wanted in ["near-earth", "deep-space", "synchronous", "half-day"] {
        assert!(
            cases.iter().any(|c| c.regime == wanted),
            "no {wanted} object in the fixture"
        );
    }
    // And the label is our own classification, not a string the generator could have mislaid.
    for case in &cases {
        let satellite = Satellite::from_tle(&case.line1, &case.line2).expect("element set parses");
        let name = match satellite.regime() {
            Regime::NearEarth => "near-earth",
            Regime::DeepSpaceNonResonant => "deep-space",
            Regime::DeepSpaceSynchronous => "synchronous",
            Regime::DeepSpaceHalfDay => "half-day",
        };
        assert_eq!(name, case.regime, "catalog {}", case.norad_id);
    }
}

/// The sweep's count of propagations one side refused and the other completed.
///
/// Nine million chances to disagree on whether a propagation is possible at all. The count is the
/// sweep's record, which this reads back as a row; the document's own refusals are decided by
/// `tests/document_refusals.rs`.
#[test]
#[ignore = "comparator: the distance to satellite.js, not a verdict"]
fn refusal_disagreements_with_satellite_js() {
    let (header, _) = parse();
    let field = |line: &str, key: &str| -> f64 {
        line.split_ascii_whitespace()
            .find_map(|f| f.strip_prefix(key))
            .unwrap_or_else(|| panic!("no {key} in {line}"))
            .parse()
            .expect("numeric")
    };
    let samples = header_value(&header, "samples");
    let init = header_value(&header, "init");
    let disagreements = field(samples, "only_ours_refused=")
        + field(samples, "only_oracle_refused=")
        + field(init, "oracle_rejected=")
        + field(init, "we_rejected=");
    println!(
        "comparison | satellite.js | 4.1.4 | propagations or element sets one side refused | count | {} | {disagreements:.3e} | {disagreements:.3e}",
        field(samples, "compared=")
    );
}

/// The convention the sweep ran under is the one the oracle implements. (S2)
///
/// satellite.js uses WGS-72 and the AFSPC sidereal expression; [`Convention::Reference`] selects
/// exactly that, and it is the default. If the default ever changed, the committed residual would
/// stop describing the code under test — so the coupling is asserted rather than assumed.
#[test]
fn the_default_convention_is_the_one_the_oracle_implements() {
    assert_eq!(Convention::default(), Convention::Reference);

    let (_, cases) = parse();
    let case = cases.first().expect("at least one case");
    let satellite = Satellite::from_tle(&case.line1, &case.line2).expect("element set parses");
    assert_eq!(satellite.convention(), Convention::Reference);
}

/// The distance from the IAU convention to satellite.js' WGS-72 answers.
///
/// The same element sets propagated under [`Convention::Iau`] — WGS-84 gravity and the IAU sidereal
/// expression instead of the oracle's WGS-72 and AFSPC — land metres to kilometres away: the scale a
/// model change has, against the micrometres of the transcription distance above.
#[test]
#[ignore = "comparator: the distance to satellite.js, not a verdict"]
fn iau_convention_distance_to_satellite_js() {
    let (_, cases) = parse();
    let mut positions = Vec::new();
    for case in cases.iter().take(8) {
        let elements = sgp4::Elements::from_tle(None, case.line1.as_bytes(), case.line2.as_bytes())
            .expect("element set parses");
        let iau = Satellite::from_elements(&elements, Convention::Iau).expect("usable elements");
        for &(minutes, r, _) in &case.samples {
            if let Ok(state) = iau.propagate_minutes(minutes) {
                positions.push(distance(state.position.vector, r));
            }
        }
    }
    comparison_row(
        "position under the IAU convention, eight objects",
        "m",
        &positions,
    );
}
