//! The CARA conjunctions, read from the CDMs instead of transcribed from them.
//!
//! `conjunction.rs`'s `alfano_cara_cases_reproduce_published_reference` is the workspace's primary
//! `Pc` oracle, and every number in it was copied by hand out of the NASA CARA sample messages. A
//! hand transcription of 210 covariance entries is a place a digit goes missing quietly: the wrong
//! digit lands in a low-order position, `Pc` moves by less than the 3·10⁻⁴ the test allows, and
//! nothing ever says so.
//!
//! This test removes the hand from the loop. It reads the same eleven files with
//! [`spacelab_ndm::parse_cdm`], rebuilds the geometry from what the file says, and holds the
//! resulting `Pc` to the same published references at the same tolerance. Two things pass or fail
//! together as a result — the reader, and the transcription it replaces.
//!
//! # Why the data is not in the repository
//!
//! `nasa/CARA_Analysis_Tools` ships four NASA Open Source Agreement documents and no LICENSE. NOSA
//! is OSI-approved but not permissive, and this workspace is heading for a permissive release, so
//! the messages are fetched rather than vendored — `tools/fetch-cara-cdms.sh`, into an ignored
//! directory, with a committed manifest recording URL, digest and size. **When the directory is
//! absent this test skips**, which is why the transcribed table in `conjunction.rs` stays where it
//! is: it is the leg of the oracle that survives a checkout with no network.
//!
//! The reference probabilities are quoted directly, because a number published in a paper (Alfano
//! 2009, *Satellite Conjunction Monte Carlo Analysis*) is a fact and not an expressive work.

// Exact `f64` comparison is the point wherever it appears here: the two duplicate messages must
// agree bit for bit, and a value parsed out of a decimal must be the double that decimal denotes.
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use std::path::{Path, PathBuf};

use spacelab_estimation::conjunction::ConjunctionGeometry;
use spacelab_ndm::{Cdm, parse_cdm, symmetric_from_lower};

/// Case number, combined hard-body radius \[m], published reference `Pc`.
///
/// The radius is listed here and *checked against* `COMMENT HBR` in the file rather than read from
/// it, because it is the one quantity CDM 1.0 has no keyword for: reading it without asserting it
/// would make the test agree with whatever convention the producer happened to use.
const REFERENCES: [(u32, f64, f64); 10] = [
    (1, 15.0, 0.146_749_549),
    (2, 4.0, 0.006_222_267),
    (3, 15.0, 0.100_351_176),
    (4, 15.0, 0.049_323_406),
    (5, 10.0, 0.044_487_386),
    (6, 10.0, 0.004_335_455),
    (7, 10.0, 0.000_158_147),
    (8, 4.0, 0.036_948_008),
    (9, 6.0, 0.290_146_291),
    (11, 4.0, 0.002_672_026),
];

/// Where the fetched messages live, `CARA_DIR` overriding the default for an out-of-tree copy.
fn cara_dir() -> PathBuf {
    std::env::var_os("CARA_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/validation/cara"),
        PathBuf::from,
    )
}

/// Read and parse one case, or `None` when the suite has not been fetched.
fn case(number: u32) -> Option<Cdm> {
    let path = cara_dir().join(format!("AlfanoTestCase{number:02}.cdm"));
    let text = std::fs::read_to_string(&path).ok()?;
    Some(parse_cdm(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display())))
}

/// The RTN position covariance: the leading 3×3 block of the 6×6 the message states.
fn position_covariance(covariance: &[f64; 21]) -> [[f64; 3]; 3] {
    let full = symmetric_from_lower(covariance);
    let mut block = [[0.0; 3]; 3];
    for (row, source) in block.iter_mut().zip(&full) {
        row.copy_from_slice(&source[..3]);
    }
    block
}

fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// Every published `Pc` is reproduced from the message rather than from a transcription of it.
#[test]
fn cara_cdms_reproduce_the_published_probabilities() {
    let Some(first) = case(1) else {
        println!(
            "skipped: {} holds no messages; run tools/fetch-cara-cdms.sh",
            cara_dir().display()
        );
        return;
    };
    drop(first);

    let mut worst = 0.0_f64;
    for (number, hbr, reference) in REFERENCES {
        let cdm = case(number).unwrap();

        assert_eq!(
            cdm.commented_hard_body_radius(),
            Some(hbr),
            "case {number}: COMMENT HBR disagrees with the radius the reference Pc was computed at"
        );

        let relative_position = cdm.relative_position.unwrap();
        let relative_velocity = cdm.relative_velocity.unwrap();

        // The message states the miss distance and relative speed as separate keywords, so the
        // vectors the reader assembled can be checked against the file's own scalars. Both are
        // written to six decimals of a metre, which bounds the disagreement at ~10⁻⁶ absolute.
        assert!(
            (norm(relative_position) - cdm.miss_distance).abs() < 1e-5,
            "case {number}: |Δr| = {} against MISS_DISTANCE = {}",
            norm(relative_position),
            cdm.miss_distance
        );
        assert!(
            (norm(relative_velocity) - cdm.relative_speed.unwrap()).abs() < 1e-5,
            "case {number}: |Δv| = {} against RELATIVE_SPEED = {}",
            norm(relative_velocity),
            cdm.relative_speed.unwrap()
        );

        // Same construction as the transcribed test: the combined radius split evenly, because the
        // model reads only the sum.
        let geometry = ConjunctionGeometry::new(
            relative_position,
            relative_velocity,
            position_covariance(&cdm.object1.covariance),
            position_covariance(&cdm.object2.covariance),
            0.5 * hbr,
            0.5 * hbr,
        );
        let foster = geometry.project().unwrap().probability_foster();
        let error = (foster - reference).abs() / reference;
        worst = worst.max(error);
        println!("CARA {number:02}: ref={reference:.9e} foster={foster:.9e} rel={error:.2e}");
        assert!(
            error < 3e-4,
            "case {number}: {foster:.6e} against the published {reference:.6e}, {error:.2e} relative"
        );
    }
    println!("worst relative error, Pc from the file against the published reference: {worst:.2e}");
}

/// Case 10 is case 9 with different catalogue numbers — the reason the oracle covers ten of eleven.
///
/// `conjunction.rs` states that as prose and then silently omits the case. With the files present it
/// is checkable: every number in the two messages agrees, so case 10 adds no geometry, and the
/// omission is a fact about the suite rather than a gap in the coverage.
#[test]
fn case_ten_is_case_nine_under_another_catalogue_number() {
    let (Some(nine), Some(ten)) = (case(9), case(10)) else {
        println!("skipped: run tools/fetch-cara-cdms.sh");
        return;
    };

    assert_eq!(nine.tca, ten.tca);
    assert_eq!(nine.miss_distance, ten.miss_distance);
    assert_eq!(nine.relative_position, ten.relative_position);
    assert_eq!(nine.relative_velocity, ten.relative_velocity);
    assert_eq!(
        nine.commented_hard_body_radius(),
        ten.commented_hard_body_radius()
    );
    for (nine, ten) in [(&nine.object1, &ten.object1), (&nine.object2, &ten.object2)] {
        assert_eq!(nine.position, ten.position);
        assert_eq!(nine.velocity, ten.velocity);
        assert_eq!(nine.covariance, ten.covariance);
        assert_ne!(
            nine.object_designator, ten.object_designator,
            "the two cases must at least be distinguishable"
        );
    }
}

/// The reader survives what an operational message actually contains.
///
/// These are not the annex examples: they carry `NaN` in eleven optional keywords, a comment with an
/// `=` in it that is not the hard-body radius, and units on `RELATIVE_VELOCITY_*` that say `[m]`
/// where the standard says `[m/s]`. Each of those is a way a stricter reader would fail on a file
/// that is nonetheless the one the industry sends.
#[test]
fn operational_messages_parse_despite_nan_fields_and_a_mislabelled_unit() {
    let Some(cdm) = case(1) else {
        println!("skipped: run tools/fetch-cara-cdms.sh");
        return;
    };

    assert!(
        cdm.object1.area_pc.unwrap().is_nan(),
        "AREA_PC is NaN in this suite and must arrive as NaN, not as a parse failure"
    );
    // `COMMENT Apogee Altitude = NaN [km]` splits on `=` exactly like the HBR comment does; only the
    // key tells them apart.
    assert!(
        cdm.comments
            .iter()
            .any(|comment| comment.starts_with("Apogee Altitude")),
        "a comment carrying an `=` must survive as a comment"
    );
    assert_eq!(cdm.commented_hard_body_radius(), Some(15.0));
    // Units are stripped and never read, which is why the file's `[m]` on a velocity costs nothing:
    // the components arrive as the exact doubles the decimals denote, with no scaling attempted.
    assert_eq!(
        cdm.relative_velocity.unwrap(),
        [-0.009_963, 0.010_037, -0.000_001]
    );
}
