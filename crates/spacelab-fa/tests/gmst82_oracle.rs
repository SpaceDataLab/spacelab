//! Comparator: `gmst82` against `pyerfa` over ten millennia. Benchmark material, not a
//! gate — the gate is `gmst_is_the_exact_rational_polynomial_reduced_to_a_turn` in
//! `src/earth_rotation.rs`, against the polynomial in exact rationals
//! (`derivations/gmst82.mac`).
//!
//! The fixtures are generated offline by `derivations/gmst82.py` using `pyerfa`,
//! and committed. `pyerfa` wraps ERFA, a fork of SOFA: the distance measures how
//! far our evaluation order lies from SOFA's, which grows with `|B·T|` four thousand
//! years from J2000.

// Panicking on a malformed fixture is the correct behaviour: a corrupt comparator file
// is not a condition to recover from. The workspace ban on `expect` applies to library
// code, where a panic is a denial of service.
#![allow(clippy::expect_used)]

use spacelab_core::time::{Epoch, Ut1};
use spacelab_fa::gmst82;

const FIXTURES: &str = include_str!("fixtures/gmst82_pyerfa.csv");

#[test]
#[ignore = "comparator: the distance to pyerfa, not a verdict"]
fn gmst82_distance_to_pyerfa_across_ten_millennia() {
    let distances: Vec<f64> = FIXTURES
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(i, line)| {
            let (days, expected) = line
                .split_once(',')
                .unwrap_or_else(|| panic!("row {}: not two columns: {line:?}", i + 1));
            let days: f64 = days.trim().parse().expect("column 1 is a float");
            let expected: f64 = expected.trim().parse().expect("column 2 is a float");
            // The angle wraps, so compare on the circle rather than on the line.
            circular_distance(
                gmst82(Epoch::<Ut1>::from_days_since_j2000(days).expect("a finite day count"))
                    .radians(),
                expected,
            )
        })
        .collect();
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(clippy::cast_precision_loss, reason = "a count of a few dozen epochs")]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | pyerfa | not recorded | GMST82 at epochs over ten millennia | rad | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// Shortest angular separation between two angles, in `[0, π]`.
fn circular_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}
