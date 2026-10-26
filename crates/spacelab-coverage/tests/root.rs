//! Brent against roots computed to 34 digits in PARI/GP, and the failure modes Orekit names.
//!
//! The reference roots are derived in `derivations/root_reference_roots.mac` (Maxima) and
//! re-solved in `derivations/root_reference_roots.gp` (PARI/GP). A root finder that only agrees
//! with itself proves nothing; these compare against an independent high-precision oracle, and
//! against the conformance cases (`RootPlusToleranceHasWrongSign`, double roots, endpoint roots)
//! that `notes/13-orekit.md` §9 lists as the ones a detector must not fail.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::approx_constant,
    clippy::type_complexity
)]

use spacelab_coverage::{RootError, brent};

/// PARI/GP `solve`, 34 significant digits.
const ROOT_CUBIC: f64 = 2.094_551_481_542_326_591_482_386_540_579_303; // x^3 - 2x - 5
const ROOT_DOTTIE: f64 = 0.739_085_133_215_160_641_655_312_087_673_873_4; // cos(x) - x
const ROOT_OMEGA: f64 = 0.567_143_290_409_783_872_999_968_662_210_355_5; // x - exp(-x)
const ROOT_SQRT2: f64 = 1.414_213_562_373_095_048_801_688_724_209_698; // x^2 - 2

fn assert_valid_bracket(b: &spacelab_coverage::Bracket, tol: f64) {
    assert!(b.lo <= b.hi, "bracket not ordered: {b:?}");
    assert!(
        b.f_lo == 0.0 || b.f_hi == 0.0 || b.f_lo.signum() != b.f_hi.signum(),
        "bracket endpoints do not straddle the root: f_lo={}, f_hi={}",
        b.f_lo,
        b.f_hi
    );
    // zeroin converges at |xm| <= 2·eps·|b| + tol/2, so the bracket width is <= tol + 4·eps·|b|.
    let bound = tol + 8.0 * f64::EPSILON * b.estimate().abs().max(1.0);
    assert!(
        b.width() <= bound,
        "bracket width {} exceeds the guaranteed tol + 4·eps·|x| = {bound}",
        b.width()
    );
}

/// (G3) Brent's bracket contains, and its estimate lies within `tol + 8·ε·max(|x|, 1)` of, the exact
/// roots of `x³ − 2x − 5` (Cardano), `cos x − x`, `x − e⁻ˣ` (`W(1)`) and `x² − 2`, derived in
/// `derivations/root_reference_roots.mac` and re-solved in `derivations/root_reference_roots.gp`.
#[test]
fn matches_pari_reference_roots_to_near_machine_precision() {
    let cases: &[(fn(f64) -> f64, f64, f64, f64)] = &[
        (|x| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, ROOT_CUBIC),
        (|x| x.cos() - x, 0.0, 1.5, ROOT_DOTTIE),
        (|x| x - (-x).exp(), 0.0, 1.0, ROOT_OMEGA),
        (|x| x * x - 2.0, 0.0, 2.0, ROOT_SQRT2),
    ];
    for &(f, a, b, root) in cases {
        let tol = 1e-12;
        let br = brent(f, a, b, tol).expect("bracketed");
        assert_valid_bracket(&br, tol);
        // The guarantee is that the true root lies in the bracket, so the error is at most the
        // bracket width. Checked against an independent 34-digit oracle, which also rules out
        // converging to a *different* root.
        let err = (br.estimate() - root).abs();
        let bound = tol + 8.0 * f64::EPSILON * root.abs().max(1.0);
        assert!(
            err <= bound,
            "root {} vs 34-digit reference {root}: error {err:.3e} exceeds the bracket bound {bound:.3e}",
            br.estimate()
        );
        assert!(
            br.lo <= root && root <= br.hi,
            "the 34-digit root is not inside the bracket {br:?}"
        );
    }
}

#[test]
fn the_returned_bracket_always_straddles_the_root() {
    // Orekit's testRootPlusToleranceHasWrongSign: the bracket must genuinely contain the root, so
    // that resuming on either side has a known, correct sign. Check across many tolerances.
    for p in 4..=15 {
        let tol = 10f64.powi(-p);
        let br = brent(|x| x.cos() - x, 0.0, 1.5, tol).expect("bracketed");
        // assert_valid_bracket already checks the endpoints straddle the root (opposite signs).
        assert_valid_bracket(&br, tol);
        // and the true 34-digit root lies within [lo, hi]
        assert!(
            br.lo <= ROOT_DOTTIE && ROOT_DOTTIE <= br.hi,
            "root not inside {br:?}"
        );
    }
}

#[test]
fn refuses_an_unbracketed_interval_instead_of_guessing() {
    // Same sign at both ends: GMAT would extrapolate; we refuse.
    assert_eq!(
        brent(|x| x * x + 1.0, -1.0, 1.0, 1e-9),
        Err(RootError::NotBracketed)
    );
    assert_eq!(
        brent(|x| x - 5.0, 0.0, 1.0, 1e-9),
        Err(RootError::NotBracketed)
    );
}

#[test]
fn endpoint_roots_are_reported_not_iterated_toward() {
    let br = brent(|x| x, 0.0, 1.0, 1e-9).expect("root at a");
    assert_eq!(br.estimate(), 0.0);
    assert_eq!(
        br.evaluations, 2,
        "an endpoint root costs exactly the two endpoint evals"
    );

    let br = brent(|x| x - 1.0, 0.0, 1.0, 1e-9).expect("root at b");
    assert_eq!(br.estimate(), 1.0);
}

#[test]
fn rejects_bad_arguments() {
    assert_eq!(brent(|x| x, 0.0, 1.0, 0.0), Err(RootError::BadTolerance));
    assert_eq!(brent(|x| x, 0.0, 1.0, -1e-9), Err(RootError::BadTolerance));
    assert_eq!(
        brent(|x| x, 0.0, 1.0, f64::NAN),
        Err(RootError::BadTolerance)
    );
    assert_eq!(brent(|x| x, 1.0, 1.0, 1e-9), Err(RootError::BadInterval));
    assert_eq!(
        brent(|x| x, f64::INFINITY, 1.0, 1e-9),
        Err(RootError::BadInterval)
    );
}

#[test]
fn converges_superlinearly_on_smooth_functions_not_by_bisection() {
    // Bisecting any of these to 1e-12 takes ~40 evaluations. Brent's interpolation must do far
    // better, and the bound is set just above what it actually costs (7–11 here) rather than
    // generously: interpolation is self-correcting, so a broken inverse-quadratic step still
    // reaches the right root by bisection, and only the count reports the damage.
    let cases: &[(fn(f64) -> f64, f64, f64)] = &[
        (|x| x.cos() - x, 0.0, 1.5),
        (|x| x * x * x - 2.0 * x - 5.0, 2.0, 3.0),
        (|x| x - (-x).exp(), 0.0, 1.0),
        (|x| x * x - 2.0, 0.0, 2.0),
        (|x| x.exp() - 5.0, 0.0, 3.0),
        (|x| x.atan() - 0.5, -1.0, 4.0),
    ];
    for (i, &(f, a, b)) in cases.iter().enumerate() {
        let br = brent(f, a, b, 1e-12).expect("bracketed");
        assert!(
            br.evaluations <= 12,
            "case {i} took {} evals — that is bisection, not interpolation",
            br.evaluations
        );
    }
}

/// The measured cost of `brent` on a battery whose arithmetic is bit-reproducible.
///
/// Only polynomials and rationals: `+ - * /` and `powi` are exact IEEE operations, so these counts
/// are the same on every platform, where `cos` and `exp` come from the system libm and may differ
/// in the last ulp — enough to move an evaluation count, not enough to matter numerically. The
/// transcendental cases stay under the loose bound above.
///
/// Each entry is `(f, a, b, tol, evaluations)`. The brackets are deliberately varied in shape:
/// tight and wide, symmetric about zero and skewed, roots at the centre and hard against an
/// endpoint, `|f|` from `1e-300` to `1e300`, flat roots of order 3, 5 and 25, and one `f` that
/// diverges at a bracket end.
#[rustfmt::skip]
const COST: &[(fn(f64) -> f64, f64, f64, f64, u32)] = &[
    (|x| x * x * x - 2.0 * x - 5.0,         2.0,   3.0,   1e-12,  8),
    (|x| x * x - 2.0,                       0.0,   2.0,   1e-12,  9),
    (|x| x * x - 2.0,                       1.0,   2.0,   1e-12,  8),
    (|x| x.powi(5) - x - 1.0,               1.0,   2.0,   1e-12, 10),
    (|x| x.powi(7) - 1.0,                   0.5,   1.5,   1e-12, 11),
    (|x| (x - 1.0) * (x - 2.0) * (x - 3.0), 0.5,   1.5,   1e-12, 10),
    (|x| (x * x - 4.0) / (x * x + 1.0),     0.0,   3.0,   1e-12, 10),
    (|x| 1.0 / x - 3.0,                     0.1,   1.0,   1e-12, 11),
    (|x| 1e9 * (x - 0.5),                   0.0,   1.0,   1e-12,  3),
    (|x| (x - 0.3).powi(3),                 0.0,   1.0,   1e-10, 85),
    (|x| (x - 0.3).powi(5),                 0.0,   1.0,   1e-10, 90),
    (|x| (x - 0.3).powi(25),                0.0,   1.0,   1e-10, 95),
    (|x| x * x - 1.0,                       0.0,   1e3,   1e-12, 24),
    (|x| x * x - 2.0,                      -1e3,   1.0,   1e-12, 27),
    (|x| x - 3e-7,                          0.0,   1e-6,  1e-18,  4),
    (|x| x * x * x - 8.0,                  -1e6,   1e6,   1e-9,  48),
    // Symmetric about zero, so `b + a` is zero where `b - a` is the whole bracket: the one shape
    // that resolves the initial `e`, which otherwise only ever scales an acceptance threshold.
    (|x| x - 0.1,                          -1.0,   1.0,   1e-12,  4),
    (|x| x - 0.9999,                       -1.0,   1.0,   1e-12,  3),
    (|x| x + 0.9999,                       -1.0,   1.0,   1e-12,  3),
    (|x| x - 999.9,                        -1e3,   1e3,   1e-12,  3),
    (|x| x * x * x - 0.008,                -2.0,   2.0,   1e-12, 16),
    (|x| x * x * x + 2.0 * x - 5.0,        -5.0,   5.0,   1e-12, 13),
    (|x| x.powi(5) - 19.2,                 -4.0,   4.0,   1e-12, 14),
    (|x| (x - 5.5) / (x * x + 1.0),        -6.0,   6.0,   1e-12,  3),
    // |f| far from unity, where a ratio of function values is the only scale-free quantity.
    (|x| 1e300 * (x - 1.3),                 0.0,   2.0,   1e-12,  4),
    (|x| 1e-300 * (x - 1.3),                0.0,   2.0,   1e-12,  4),
    // `f` diverges at the lower end, so `s = f(b)/f(a)` is exactly ±0 and the secant numerator
    // `p = 2·xm·s` is exactly zero — the only state in which `p > 0` and `p >= 0` choose differently,
    // and so the only shape that can see the sign test guarding `q`. A switching function that runs
    // to infinity at a bracket end is not contrived: an inverse-range or link-margin `g` does it.
    (|x| 1.0 / x - 3.0,                     0.0,   1.0,   1e-12, 11),
    // A flat cubic root approached from a bracket wide enough that `b` and `a` are far apart and of
    // opposite sign for most of the descent, where `b − a` and `b + a` are genuinely different
    // numbers — the reset of `d` after `c` is re-established.
    (|x| (x - 0.1).powi(3),                -2.0,   5.0,   1e-12, 111),
];

#[test]
fn the_cost_of_every_bracket_shape_is_the_measured_one() {
    // Brent is self-correcting: an inverse-quadratic step computed wrongly is rejected by the
    // acceptance rule and bisection finds the root anyway. So *every* test that checks only the
    // answer passes on a broken interpolation, and the evaluation count is the only thing that
    // reports the damage. It is pinned exactly rather than bounded, because the damage is often a
    // single evaluation on a single bracket shape — measured here, breaking one operator of the
    // secant, the inverse-quadratic step or the acceptance rule moves this total by −111 to +37,
    // but on many individual brackets by nothing at all.
    //
    // A change that legitimately alters these numbers is a change to the algorithm, not to its
    // implementation, and should be argued for rather than re-baselined.
    let mut total = 0;
    for (i, &(f, a, b, tol, expected)) in COST.iter().enumerate() {
        let br = brent(f, a, b, tol).expect("bracketed");
        assert_valid_bracket(&br, tol);
        assert_eq!(
            br.evaluations, expected,
            "case {i} on [{a}, {b}] cost {} evaluations, not the measured {expected}",
            br.evaluations
        );
        total += br.evaluations;
    }
    assert_eq!(total, 642, "the battery's total cost moved");
}

#[test]
fn the_bracket_width_is_the_interval_it_reports() {
    // `assert_valid_bracket` only ever compares the width against an upper bound, which a width
    // that always answered zero would satisfy everywhere. The width has to be the real interval:
    // equal to hi − lo, and strictly positive whenever the search stopped on a tolerance rather
    // than on an exact zero.
    let br = brent(|x| x.cos() - x, 0.0, 1.5, 1e-9).expect("bracketed");
    assert_eq!(br.width(), br.hi - br.lo);
    assert!(br.width() > 0.0, "converged bracket has no width: {br:?}");
    assert!(br.width() <= 1e-9 + 8.0 * f64::EPSILON);

    // An exact root is the one case allowed to be degenerate, and it must be exactly zero.
    let exact = brent(|x| x - 0.25, 0.0, 1.0, 1e-12).expect("bracketed");
    assert_eq!(exact.width(), 0.0);
}

#[test]
fn still_converges_on_a_pathological_function_via_bisection_fallback() {
    // A function whose derivative blows up at the root defeats interpolation; bisection must save
    // it. sign(x)·√|x| has an infinite slope at 0. Brent must still bracket the root.
    let f = |x: f64| x.signum() * x.abs().sqrt();
    let br = brent(f, -1.0, 2.0, 1e-10).expect("bracketed");
    assert_valid_bracket(&br, 1e-10);
    assert!(
        br.estimate().abs() < 1e-9,
        "root near 0, got {}",
        br.estimate()
    );
    // and it must not have run away: the loop is bounded even in the worst case.
    assert!(br.evaluations < 100, "too many evals: {}", br.evaluations);
}

#[test]
fn a_root_at_a_reversed_interval_is_found() {
    // b < a: the method must not care about the order of the endpoints.
    let br = brent(|x| x * x - 2.0, 2.0, 0.0, 1e-12).expect("bracketed");
    assert!((br.estimate() - ROOT_SQRT2).abs() < 1e-13);
}

#[test]
fn tolerance_is_the_abscissa_not_the_ordinate() {
    // A function with a huge slope: |f| at the bracket ends is large, but the bracket in x is tiny.
    // If the tolerance were on f, this would "converge" at a wide x-interval. It must not.
    let steep = |x: f64| 1e9 * (x - 0.5);
    let br = brent(steep, 0.0, 1.0, 1e-9).expect("bracketed");
    assert!(
        br.width() <= 1e-9 + 8.0 * f64::EPSILON,
        "width {} is not an abscissa tolerance",
        br.width()
    );
    assert!((br.estimate() - 0.5).abs() < 1e-9);
}

#[test]
fn the_progress_safeguard_bounds_iterations_on_a_flat_root() {
    // A high-order root is nearly flat, so interpolation creeps and the "must beat bisection"
    // acceptance rule is what keeps the iteration count bounded. Without it, secant/IQI take an
    // order of magnitude more steps. `(x-0.3)^25` costs ~99 evals correctly and ~725 if the
    // acceptance rule is weakened — this bound sits between and pins the safeguard.
    let f = |x: f64| (x - 0.3).powi(25);
    let br = brent(f, 0.0, 1.0, 1e-10).expect("bracketed");
    assert!(
        (br.estimate() - 0.3).abs() < 1e-9,
        "root at 0.3, got {}",
        br.estimate()
    );
    assert!(
        br.evaluations <= 110,
        "took {} evals on a flat root — the bisection-progress safeguard is not doing its job",
        br.evaluations
    );
}

#[test]
fn the_acceptance_margin_decides_where_the_last_interpolated_step_lands() {
    // The `− |tol1·q|` in `min1` is a *margin*: it keeps an accepted interpolated step at least
    // half a tolerance clear of the far end of the bracket. Widening the margin (`+`) or scaling it
    // away (`/`) still converges, still costs the same 12 evaluations, and still returns a bracket
    // inside the requested tolerance — so neither the cost battery nor the width guarantee can see
    // it. What moves is where the final step lands, and therefore the bracket that comes back.
    //
    // A loose tolerance on a wide bracket is what exposes it: the margin only bites once `|xm|` is
    // within a few multiples of `tol1`, which here is the last step rather than a limit never
    // reached. Measured, not chosen — the numbers below are what the algorithm produces.
    let br = brent(|x: f64| x.powi(9) - 0.5, -0.125, 8.0, 1e-1).expect("bracketed");
    assert_valid_bracket(&br, 1e-1);
    assert_eq!(br.evaluations, 12);
    assert_eq!(br.lo, 0.873_522_294_155_801_1, "lo {}", br.lo);
    assert_eq!(br.hi, 0.935_198_647_077_900_6, "hi {}", br.hi);
}
