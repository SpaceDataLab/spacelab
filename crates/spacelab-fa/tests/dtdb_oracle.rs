//! Comparator: `tdb_minus_tt` against `pyerfa` over two centuries. Benchmark material,
//! not a gate — the gate is `the_offset_is_its_series_summed_exactly` in `src/tdb.rs`,
//! against the series summed in 40-digit arithmetic (`derivations/tdb_series.mac`).
//!
//! The fixtures are generated offline by `derivations/tdb_tables.py` using `pyerfa`,
//! and committed; each row stores the exact nanosecond count the oracle was called
//! at. `pyerfa` wraps ERFA, a fork of SOFA, and the coefficient table was extracted
//! from that same source, so the distance measures how far our summation order lies
//! from ERFA's, not how far either lies from the series.

// Panicking on a malformed fixture is the correct behaviour: a corrupt comparator file
// is not a condition to recover from.
#![allow(clippy::expect_used)]

use spacelab_core::time::{Epoch, Tt};
use spacelab_fa::tdb_minus_tt;

const FIXTURES: &str = include_str!("fixtures/dtdb_pyerfa.csv");

#[test]
#[ignore = "comparator: the distance to pyerfa, not a verdict"]
fn tdb_minus_tt_distance_to_pyerfa_over_two_centuries() {
    let distances: Vec<f64> = FIXTURES
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let (ns, want) = line.split_once(',').expect("two columns");
            let ns: i128 = ns.trim().parse().expect("an integer nanosecond count");
            let want: f64 = want.trim().parse().expect("a float");
            (tdb_minus_tt(Epoch::<Tt>::from_ns_since_j2000(ns)) - want).abs()
        })
        .collect();
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of a few hundred epochs"
    )]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | pyerfa | not recorded | TDB−TT at epochs over two centuries | s | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}
