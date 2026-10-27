//! Filter consistency diagnostics — NEES, NIS, and the χ²-per-degree-of-freedom covariance-realism
//! test (Bar-Shalom, *Estimation with Applications to Tracking and Navigation*, §5.4).
//!
//! A filter can report a covariance that is *precise* but not *realistic*: the estimate looks tight
//! while the true error is systematically larger (overconfident) or smaller (underconfident) than `P`
//! claims. Nothing in the filter recursion detects this — it needs an external check against either
//! ground truth (NEES, in a Monte-Carlo) or the filter's own innovations (NIS, online):
//!
//! * **NEES** (normalized estimation error squared) `= (x−x̂)ᵀ P⁻¹ (x−x̂)` — needs the true state, so it
//!   is a simulation metric. Expectation `= n` (state dimension) for a consistent filter.
//! * **NIS** (normalized innovation squared) `= νᵀ S⁻¹ ν` — needs only the innovation `ν` and its
//!   covariance `S`, so it runs online. Expectation `= m` (measurement dimension).
//!
//! Both are computed here from the **square root** of the covariance (`P = L Lᵀ`, `L` lower-triangular)
//! by forward substitution, never by forming `P⁻¹` — the same square-root discipline as the rest of
//! this crate, so the condition number is not squared. Averaged over a run, `χ²ᵥ = mean / dof` is the
//! factor that rescales an over/under-confident covariance back to realism, and
//! [`chi_square_mean_interval`] gives the acceptance band that decides whether the deviation is real.
//!
//! Gate: `derivations/chi_square_quantiles.mac` shows the forward substitution is `rᵀ(L Lᵀ)⁻¹r`, and
//! `derivations/chi_square_quantiles.gp` computes the exact χ² quantiles at 60 digits (re-derived by
//! series in the .mac) and the Wilson–Hilferty error envelope documented below. The module tests
//! `the_normalized_squared_error_is_the_quadratic_form` (G1) and
//! `the_mean_interval_is_the_cube_of_the_exact_quantile_within_its_envelope` (G3) tie the code to it.

// The math symbols (P, L, S, ν, z, k) are one letter by convention; `as f64` on small dimensions.
#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]

/// The normalized squared error `rᵀ P⁻¹ r`, from the covariance square root `sqrt` (`P = L Lᵀ`, `L`
/// lower-triangular) by forward substitution `L y = r`, returning `‖y‖²`. `P⁻¹` is never formed.
///
/// With `r = x − x̂` (true minus estimate) and `sqrt` the filter's covariance root, this is the NEES;
/// with `r` an innovation and `sqrt` the innovation-covariance root it is the NIS. `sqrt` must be a
/// nonsingular lower-triangular factor (as [`crate::SquareRootKalman::covariance_sqrt`] returns).
#[must_use]
pub fn normalized_squared<const M: usize>(sqrt: &[[f64; M]; M], residual: &[f64; M]) -> f64 {
    let mut acc = 0.0;
    let mut y = [0.0; M];
    for i in 0..M {
        let mut s = residual[i];
        for k in 0..i {
            s -= sqrt[i][k] * y[k];
        }
        y[i] = s / sqrt[i][i];
        acc += y[i] * y[i];
    }
    acc
}

/// The scalar NIS `ν² / S` for a one-dimensional measurement, from the innovation `ν` and its variance
/// `S` — exactly the `(innovation, innovation_variance)` a scalar
/// [`crate::SquareRootKalman::measurement_update`] consumes and returns. Expectation `1` when consistent.
#[must_use]
pub fn scalar_nis(innovation: f64, innovation_variance: f64) -> f64 {
    innovation * innovation / innovation_variance
}

/// The χ²-per-degree-of-freedom covariance rescale factor `mean_normalized / dof`. For a consistent
/// filter the average NEES/NIS equals `dof`, so a factor `> 1` means the filter is overconfident
/// (inflate `P`/`R` by it) and `< 1` underconfident. The classic single-number realism gauge.
#[must_use]
pub fn chi_square_scale_factor(mean_normalized: f64, dof: usize) -> f64 {
    mean_normalized / dof as f64
}

/// Two-sided acceptance interval `(lo, hi)` for the **mean** NEES/NIS over `runs` independent samples
/// of `dof` degrees of freedom each, at standard-normal quantile `z` (1.96 → 95 %, 2.576 → 99 %). The
/// filter is consistent when the average normalized error lies in `(lo, hi)`; both bounds straddle
/// `dof`.
///
/// The sum of `runs` independent χ²(dof) is χ²(runs·dof); its quantiles use the Wilson–Hilferty cube
/// approximation `χ²ₚ(k) ≈ k(1 − 2/9k + zₚ√(2/9k))³`, divided by `runs` to give the interval for
/// the mean.
///
/// Against the exact quantiles (`derivations/chi_square_quantiles.gp`, 60 digits, k = 1…400) the
/// cube is off by at most 0.27 in the quantile of the sum for `|z| ≤ 3`, and 0.10 at `z = 1.96`,
/// so the bounds of the mean are within `0.27/runs`: below 0.01 from 27 runs, not at every count.
/// The error falls with `k` — 1.2e-4 on either bound at `dof = 2, runs = 50, z = 1.96` — but at
/// `k = 1` the lower bound is negative for `z > 7/(3√2) ≈ 1.65`, where the exact quantile is small
/// and positive (9.8e-4 at `z = 1.96`).
#[must_use]
pub fn chi_square_mean_interval(dof: usize, runs: usize, z: f64) -> (f64, f64) {
    let k = (dof * runs) as f64;
    let a = 2.0 / (9.0 * k);
    let wilson_hilferty = |zp: f64| {
        let t = 1.0 - a + zp * libm::sqrt(a);
        k * t * t * t
    };
    let n = runs as f64;
    (wilson_hilferty(-z) / n, wilson_hilferty(z) / n)
}

/// Whether the average normalized error `mean_normalized` is consistent with `dof` degrees of freedom
/// over `runs` samples at quantile `z` — i.e. it lies in [`chi_square_mean_interval`].
#[must_use]
pub fn is_consistent(mean_normalized: f64, dof: usize, runs: usize, z: f64) -> bool {
    let (lo, hi) = chi_square_mean_interval(dof, runs, z);
    (lo..=hi).contains(&mean_normalized)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::SquareRootKalman;

    #[test]
    fn normalized_squared_reduces_to_the_squared_norm_for_identity_covariance() {
        let id = [[1.0, 0.0], [0.0, 1.0]];
        let r = [3.0, 4.0];
        assert!((normalized_squared(&id, &r) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_squared_matches_the_explicit_quadratic_form() {
        // L = [[2,0],[1,3]] ⇒ P = LLᵀ = [[4,2],[2,10]], det 36, P⁻¹ = [[10,-2],[-2,4]]/36.
        let l = [[2.0, 0.0], [1.0, 3.0]];
        let r = [1.0, 1.0];
        // rᵀ P⁻¹ r = (10 - 2 - 2 + 4)/36 = 10/36.
        let expected = 10.0 / 36.0;
        assert!((normalized_squared(&l, &r) - expected).abs() < 1e-12);
    }

    /// (G1) The forward substitution is the quadratic form `rᵀ(L Lᵀ)⁻¹r` that
    /// `derivations/chi_square_quantiles.mac` shows it equals on a general 3 × 3 factor; on
    /// `L = [[2,0,0],[1,3,0],[−1,4,5]]`, `r = (1, −2, ½)` that is 763/450, rounded once.
    #[test]
    fn the_normalized_squared_error_is_the_quadratic_form() {
        let l = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 4.0, 5.0]];
        let got = normalized_squared(&l, &[1.0, -2.0, 0.5]);
        let exact = 763.0 / 450.0;
        assert!(
            (got / exact - 1.0).abs() <= 4.0 * f64::EPSILON,
            "{got} against {exact}"
        );
    }

    /// (G3) The acceptance band is the Wilson–Hilferty cube of the two quantiles, to rounding, and
    /// that cube sits where the documentation says against the exact quantiles of χ²(k):
    /// `derivations/chi_square_quantiles.gp` at 60 digits (cross-derived in
    /// `derivations/chi_square_quantiles.mac`), divided by `runs`. At `dof = 2, runs = 50` the bounds
    /// are within 1.2e-4 of the exact ones; at one sample of one degree of freedom the upper bound is
    /// 0.096 low and the lower one is negative where the exact is 9.8e-4 — inside the 0.27/runs
    /// envelope, and the reason that envelope is stated per run.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn the_mean_interval_is_the_cube_of_the_exact_quantile_within_its_envelope() {
        for (dof, runs, cube, exact, envelope) in [
            (
                2,
                50,
                (
                    1.484_315_666_121_386_441_104_170,
                    2.591_291_898_871_754_848_333_416,
                ),
                (
                    1.484_430_215_023_646_332_532_425,
                    2.591_236_034_543_314_253_648_493,
                ),
                1.2e-4,
            ),
            (
                1,
                1,
                (
                    -0.003_123_345_637_774_382_519_105_020,
                    4.928_019_916_282_493_175_386_046,
                ),
                (
                    9.819_037_012_087_134_823_160_270e-4,
                    5.024_031_992_629_765_320_376_545,
                ),
                0.097,
            ),
        ] {
            let (lo, hi) = chi_square_mean_interval(dof, runs, 1.96);
            for (got, want) in [(lo, cube.0), (hi, cube.1)] {
                assert!(
                    (got - want).abs() <= 8.0 * f64::EPSILON * want.abs().max(1.0),
                    "dof {dof}, runs {runs}: {got} against the cube {want}"
                );
            }
            assert!((lo - exact.0).abs() < envelope && (hi - exact.1).abs() < envelope);
            assert!((hi - exact.1).abs() <= 0.27 / runs as f64);
        }
        assert!(chi_square_mean_interval(1, 1, 1.96).0 < 0.0);
    }

    #[test]
    fn nees_from_a_filters_covariance_root_is_finite_and_correct() {
        // Seed a filter with P = diag(4, 9); a 1σ error in each axis gives NEES = 1 + 1 = 2.
        let filter =
            SquareRootKalman::from_covariance([0.0, 0.0], &[[4.0, 0.0], [0.0, 9.0]]).unwrap();
        let error = [2.0, 3.0]; // (x_true − x̂) at exactly 1σ per axis
        let nees = normalized_squared(&filter.covariance_sqrt(), &error);
        assert!((nees - 2.0).abs() < 1e-12, "nees {nees}");
    }

    #[test]
    fn scalar_nis_is_the_squared_sigma_ratio() {
        // A 2σ innovation (ν = 2, S = 1) gives NIS = 4.
        assert!((scalar_nis(2.0, 1.0) - 4.0).abs() < 1e-12);
    }

    /// The NIS is a square over a variance, at operands where nothing else lands on the same value.
    ///
    /// The test above uses `ν = 2, S = 1`, and 1 is the fixed point of division while 2 is the fixed
    /// point of `ν + ν = ν²` — so a sum and a product were indistinguishable there. With `ν = 3` and
    /// `S = 4` the answer is 9/4, against 6/4 for a sum and 36 for a multiplication.
    #[test]
    fn scalar_nis_separates_the_square_from_a_sum_and_the_division_from_a_product() {
        assert!((scalar_nis(3.0, 4.0) - 2.25).abs() < 1e-12);
    }

    #[test]
    fn scale_factor_flags_over_and_under_confidence() {
        assert!((chi_square_scale_factor(4.0, 2) - 2.0).abs() < 1e-12); // overconfident ×2
        assert!((chi_square_scale_factor(1.0, 2) - 0.5).abs() < 1e-12); // underconfident ×0.5
    }

    #[test]
    fn mean_interval_matches_known_chi_square_quantiles() {
        // dof 2, 50 runs → χ²(100). Known 2.5 %/97.5 % quantiles 74.222 / 129.561, ÷50 = 1.4844 / 2.5912.
        let (lo, hi) = chi_square_mean_interval(2, 50, 1.96);
        assert!((lo - 1.4844).abs() < 1e-2, "lo {lo}");
        assert!((hi - 2.5912).abs() < 1e-2, "hi {hi}");
        // The band straddles the expected value dof = 2.
        assert!(lo < 2.0 && 2.0 < hi);
    }

    #[test]
    fn is_consistent_accepts_the_expected_value_and_rejects_a_gross_deviation() {
        assert!(is_consistent(2.0, 2, 50, 1.96)); // exactly dof → consistent
        assert!(!is_consistent(5.0, 2, 50, 1.96)); // 2.5× overconfident → rejected
    }
}
