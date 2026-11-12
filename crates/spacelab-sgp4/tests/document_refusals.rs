//! Where the document's own model refuses to answer, and that we refuse exactly there.
//!
//! SGP4's refusals belong to the document (D-18 a): `sgp4` in Vallado et al. (2006) sets
//! `satrec.error` when the mean eccentricity leaves [−0.001, 1) (code 1), the mean motion is not
//! positive (2), the perturbed eccentricity leaves [0, 1] (3), the semi-latus rectum is negative
//! (4), or the radius falls below one Earth radius (6). Codes 1–4 are
//! [`PropagationError::Diverged`] here, code 6 [`PropagationError::Decayed`].
//!
//! `derivations/sgp4_document.mac` evaluates those tests inside the document's routines at 40
//! digits, on the grid of the document's own driver, and reads `DOCUMENT_REFUSALS` back. The
//! printed `tcppver.out` fixes where each section's first refusal falls: the driver prints t = 0
//! regardless and then stops at the first error; it says nothing past that point, which is why the
//! table, not the printout, decides the rest.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::{PropagationError, Satellite};

const CASES: &str = include_str!("fixtures/sgp4_ver_tle.txt");
const PUBLISHED: &str = include_str!("fixtures/tcppver.out");

/// Every refusal the document's model makes on its driver's grid, as runs of consecutive grid
/// times: set (row of `fixtures/sgp4_ver_tle.txt`, from 0), first minute, last minute, class (1:
/// codes 1–4, 6: code 6). Derived by `derivations/sgp4_document.mac`, which reads it back. Catalog
/// 33333 (set 29) answers between its three runs.
#[rustfmt::skip]
const DOCUMENT_REFUSALS: [f64; 36] = [
    11.0, 494.202_867_2, 1440.0, 1.0,
    22.0, 1560.0, 2880.0, 1.0,
    25.0, 55.0, 60.0, 6.0,
    26.0, 440.0, 440.0, 6.0,
    29.0, 25.0, 45.0, 1.0,
    29.0, 85.0, 110.0, 1.0,
    29.0, 145.0, 150.0, 1.0,
    30.0, 0.0, 1440.0, 1.0,
    32.0, 1_844_345.0, 1_844_355.0, 6.0,
];

struct Set {
    line1: &'static str,
    line2: &'static str,
    start: f64,
    stop: f64,
    step: f64,
}

fn sets() -> Vec<Set> {
    CASES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('|').collect();
            Set {
                line1: f[0],
                line2: f[1],
                start: f[2].parse().unwrap(),
                stop: f[3].parse().unwrap(),
                step: f[4].parse().unwrap(),
            }
        })
        .collect()
}

/// The driver's grid (`testcpp`): t = 0, then start, start + step, … with the last step clamped to
/// stop, accumulated in binary64 as the driver accumulates it.
fn grid(set: &Set) -> Vec<f64> {
    let mut times = vec![0.0];
    let mut t = set.start;
    if t.abs() > 1e-8 {
        t -= set.step;
    }
    while t < set.stop {
        t += set.step;
        if t > set.stop {
            t = set.stop;
        }
        times.push(t);
    }
    times
}

/// What the document's model answers for set `k` at `minutes`.
fn document_verdict(k: usize, minutes: f64) -> Result<(), PropagationError> {
    // Grid steps are at least a minute; the bound only absorbs the binary64 accumulation.
    let set = f64::from(u32::try_from(k).expect("33 sets"));
    let run = DOCUMENT_REFUSALS
        .chunks(4)
        .find(|r| (r[0] - set).abs() < 0.5 && minutes >= r[1] - 1e-6 && minutes <= r[2] + 1e-6);
    match run {
        None => Ok(()),
        Some(r) if (r[3] - 6.0).abs() < 0.5 => Err(PropagationError::Decayed),
        Some(_) => Err(PropagationError::Diverged),
    }
}

#[test]
fn we_refuse_exactly_where_the_document_refuses() {
    let (mut answered, mut diverged, mut decayed) = (0, 0, 0);
    for (k, set) in sets().iter().enumerate() {
        let satellite = Satellite::from_tle_unchecked(set.line1, set.line2)
            .expect("every published set builds");
        for t in grid(set) {
            let ours = satellite.propagate_minutes(t).map(|_| ());
            assert_eq!(
                ours,
                document_verdict(k, t),
                "set {k} (catalog {}) at {t} min",
                satellite.norad_id().get()
            );
            match ours {
                Ok(()) => answered += 1,
                Err(PropagationError::Decayed) => decayed += 1,
                Err(_) => diverged += 1,
            }
        }
    }
    println!("document grid: {answered} answered, {diverged} diverged, {decayed} decayed");
    // Both refusal classes and the answers between 33333's runs are what this test is for.
    assert_eq!((diverged, decayed), (1515, 6), "refusal counts moved");
}

/// (G5) The published sections end where we first refuse: the driver prints t = 0 regardless,
/// then every grid time until the first error.
#[test]
fn every_published_section_ends_where_we_first_refuse() {
    let mut rows: Vec<usize> = Vec::new();
    for line in PUBLISHED.lines() {
        if line.trim_end().ends_with(" xx") {
            rows.push(0);
        } else if !line.trim().is_empty() {
            *rows.last_mut().expect("a row before any section header") += 1;
        }
    }
    let sets = sets();
    assert_eq!(
        rows.len(),
        sets.len(),
        "one published section per element set"
    );
    for (k, (set, printed)) in sets.iter().zip(rows).enumerate() {
        let satellite = Satellite::from_tle_unchecked(set.line1, set.line2).unwrap();
        let times = grid(set);
        let ours = if satellite.propagate_minutes(0.0).is_err() {
            1
        } else {
            1 + times[1..]
                .iter()
                .take_while(|&&t| satellite.propagate_minutes(t).is_ok())
                .count()
        };
        assert_eq!(
            ours,
            printed,
            "set {k} (catalog {}): we stop after {ours} rows, the printout after {printed}",
            satellite.norad_id().get()
        );
    }
}
