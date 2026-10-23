//! Differential test of `tdb_minus_tt` against an external oracle.
//!
//! The fixtures are generated offline by `derivations/tdb_tables.py` using `pyerfa`,
//! and committed. CI never runs Python: it reads numbers, not code.
//!
//! `pyerfa` wraps ERFA, which is a fork of SOFA, and the coefficient table was
//! extracted from that same source — so this proves transcription, not independence.
//! What it *does* prove is the part that can silently rot: that 787 amplitudes, five
//! group boundaries and a Horner order survived the trip into Rust intact. The
//! derivation additionally reproduces the published SOFA `t_dtdb` self-test vector,
//! which is topocentric and therefore exercises blocks this geocentric entry point
//! does not reach. See `plans/04-validation-and-risks.md` §1.2.

// Panicking on a malformed fixture is the correct behaviour: a corrupt oracle is
// not a condition to recover from.
#![allow(clippy::expect_used)]

use spacelab_core::time::{Epoch, Tt};
use spacelab_fa::tdb_minus_tt;

const FIXTURES: &str = include_str!("fixtures/dtdb_pyerfa.csv");

/// Agreement required against the oracle, in seconds.
///
/// **Derived, not tuned.** `TDB − TT` peaks at `1.7e-3` s, whose unit in the last
/// place is `2^-62 ≈ 2.17e-19` s. The sum runs over 787 terms in the published order;
/// the amplitudes span eleven decades, so the accumulated rounding is bounded by the
/// number of additions times the ulp of the largest partial sum. Taking `128` ulp —
/// `2.8e-17` s — and rounding up gives `5e-17` s, which is `1.5e-8` metres of light
/// travel and nine orders below the microsecond at which any consumer of this scale
/// operates.
///
/// The epoch is not part of the budget: the fixture stores the exact nanosecond count
/// and the oracle was called on the same `(day, fraction)` split, so both sides
/// evaluate at the same `t`. Over the 441 committed epochs the worst residual is
/// **2.168e-19 s** — one ulp of the peak, which is the summation reproducing ERFA's
/// term by term. If a change pushes it to a large fraction of the budget, the
/// summation has become worse and the tolerance must not be widened.
const TOLERANCE_S: f64 = 5e-17;

#[test]
fn tdb_minus_tt_matches_pyerfa_over_two_centuries() {
    let mut checked = 0;
    let mut worst = 0.0_f64;
    let mut worst_at = 0_i128;

    for line in FIXTURES.lines().filter(|l| !l.trim().is_empty()) {
        let (ns, want) = line.split_once(',').expect("two columns");
        let ns: i128 = ns.trim().parse().expect("an integer nanosecond count");
        let want: f64 = want.trim().parse().expect("a float");

        let got = tdb_minus_tt(Epoch::<Tt>::from_ns_since_j2000(ns));
        let delta = (got - want).abs();
        if delta > worst {
            worst = delta;
            worst_at = ns;
        }
        assert!(
            delta < TOLERANCE_S,
            "{ns} ns: got {got:.19e}, oracle {want:.19e}, differ by {delta:.3e} s"
        );
        checked += 1;
    }

    assert!(checked >= 200, "only {checked} epochs in the fixture");
    println!("dtdb: {checked} epochs, worst {worst:.3e} s at {worst_at} ns");
}
