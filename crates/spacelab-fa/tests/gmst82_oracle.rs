//! Differential test of `gmst82` against an external oracle.
//!
//! The fixtures are generated offline by `derivations/gmst82.py` using `pyerfa`,
//! and committed. CI never runs Python: it reads numbers, not code.
//!
//! `pyerfa` wraps ERFA, which is a fork of SOFA. It is therefore an oracle of the
//! *SOFA lineage*, and it can only prove that our implementation agrees with SOFA
//! — not that SOFA is right. A second oracle of a different lineage is required
//! before that stronger claim can be made. See `plans/04-validation-and-risks.md`
//! §1.2.

// Panicking on a malformed fixture is the correct behaviour: a corrupt oracle is
// not a condition to recover from. The workspace ban on `expect` applies to library
// code, where a panic is a denial of service.
#![allow(clippy::expect_used)]

use spacelab_core::time::{Epoch, Ut1};
use spacelab_fa::gmst82;

const FIXTURES: &str = include_str!("fixtures/gmst82_pyerfa.csv");

/// Agreement required against the oracle, in radians, as a function of epoch.
///
/// **This tolerance is derived, not tuned.** The dominant intermediate in the GMST
/// polynomial is `B·T` with `B = 8640184.812866` seconds per Julian century. Its
/// unit in the last place is `ε·|B·T|` seconds; two mathematically equivalent
/// evaluation orders may differ by a small number of them. Converting seconds of
/// angle to radians multiplies by `2π/86400`.
///
/// A flat `10⁻¹²` rad would be *below the resolution of the computation itself*
/// four thousand years from J2000, and a test that demands more precision than
/// `f64` can carry is testing the summation order, not the formula.
///
/// The bound stays sharp where it matters: within a century of J2000 it is about
/// `10⁻¹²` rad, which is 0.2 nanometres of rotation at the equator. A wrong
/// coefficient anywhere in the polynomial would miss by many orders of magnitude
/// more, so nothing is being let through.
fn tolerance_rad(days: f64) -> f64 {
    /// Constant term, seconds.
    const C0: f64 = 67_310.548_41;
    /// Linear coefficient, seconds per Julian century.
    const B: f64 = 8_640_184.812_866;
    /// Bound on the whole-day contribution, seconds.
    const DAY: f64 = 86_400.0;
    /// Radians per second of time.
    const S2R: f64 = std::f64::consts::TAU / DAY;
    /// Slack for the differing summation orders of two implementations.
    const ULPS: f64 = 8.0;

    let t = (days / 36_525.0).abs();

    // Each intermediate contributes a rounding error proportional to its own
    // magnitude. Summing magnitudes bounds the sum of the errors.
    let magnitude_seconds = C0 + DAY + B * t;

    // Plus one more rounding at the final reduction into [0, 2π).
    ULPS * f64::EPSILON * (magnitude_seconds * S2R + std::f64::consts::TAU)
}

#[test]
fn agrees_with_pyerfa_across_ten_millennia() {
    let mut checked = 0_usize;
    let mut worst = 0.0_f64;
    let mut worst_at = 0.0_f64;

    for (line_no, line) in FIXTURES.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (days_s, expected_s) = line
            .split_once(',')
            .unwrap_or_else(|| panic!("line {}: not two columns: {line:?}", line_no + 1));

        let days: f64 = days_s.trim().parse().expect("column 1 is a float");
        let expected: f64 = expected_s.trim().parse().expect("column 2 is a float");

        let got = gmst82(Epoch::<Ut1>::from_days_since_j2000(days)).radians();

        // The angle wraps, so compare on the circle rather than on the line.
        let delta = circular_distance(got, expected);
        let tol = tolerance_rad(days);
        let headroom = delta / tol;
        if headroom > worst {
            worst = headroom;
            worst_at = days;
        }
        assert!(
            delta < tol,
            "days {days}: got {got:.17}, oracle {expected:.17}, delta {delta:.3e}, tol {tol:.3e}"
        );
        checked += 1;
    }

    assert!(
        checked >= 40,
        "expected a meaningful fixture set, got {checked}"
    );
    let percent = worst * 100.0;
    println!(
        "gmst82: {checked} epochs; worst delta used {percent:.1}% of its tolerance, at day {worst_at}"
    );
}

/// Shortest angular separation between two angles, in `[0, π]`.
fn circular_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}
