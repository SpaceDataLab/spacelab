//! Consider covariance (Schmidt 1966; Tapley, Schutz & Born, *Statistical Orbit Determination*,
//! §4.14) — the honest covariance of an estimate that deliberately does **not** solve for some
//! uncertain parameters.
//!
//! Orbit determination always has parameters that are uncertain but unwise to estimate from the data
//! at hand: drag and SRP coefficients, station coordinates, gravity-field errors, measurement biases.
//! Estimating them can corrupt the fit (they are weakly observable), but pretending they are perfectly
//! known makes the reported covariance optimistic — the "estimated-only" covariance `P` understates the
//! true error. The **consider** covariance restores honesty: it inflates `P` by the effect of the
//! *considered* parameters' uncertainty on the solution, without ever estimating (or updating) them.
//!
//! With estimated states `x`, considered parameters `c` of covariance `Pcc`, information matrix
//! `Λ = HᵀR⁻¹H` partitioned into `Λxx`, `Λxc`, and the estimated-only covariance `P = Λxx⁻¹`:
//!
//! * the **sensitivity** of the solution to the considered parameters is `S = ∂x̂/∂c = −P·Λxc`
//!   ([`consider_sensitivity`]);
//! * the **consider covariance** is `Pc = P + S·Pcc·Sᵀ` ([`consider_covariance`]).
//!
//! `Pc` is exactly the covariance of the estimation error `e = P·Hxᵀ·R⁻¹·(Hc·c + n)` that the ignored
//! parameters and the measurement noise together produce, so `Pc ⪰ P` always — the consider covariance
//! never claims to be tighter than the estimated one. `Pcc` is an input and is never modified, matching
//! the Schmidt filter's rule that the considered block is not updated.
//!
//! Gate: `derivations/consider_covariance.mac` derives `S` as `∂x̂/∂c̄` and `Pc` as `E[e eᵀ]` from
//! the estimator itself, with `Pc = P` at `Pcc = 0`; `derivations/linear_filters.gp` recomputes the
//! fixture. The module test `the_consider_covariance_is_the_error_covariance_of_the_fixture` (G2)
//! holds both functions to it.

// Matrix algebra over const-generic dimensions: `X` estimated states, `C` considered parameters, with
// single-letter index and dimension names as is universal for linear algebra.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

/// The consider sensitivity `S = ∂x̂/∂c = −P·Λxc` (`X × C`): how the estimate of the `X` solved-for
/// states shifts per unit error in each of the `C` considered parameters. `estimated_covariance` is
/// `P = Λxx⁻¹`; `cross_information` is `Λxc = Hxᵀ·R⁻¹·Hc`.
#[must_use]
pub fn consider_sensitivity<const X: usize, const C: usize>(
    estimated_covariance: &[[f64; X]; X],
    cross_information: &[[f64; C]; X],
) -> [[f64; C]; X] {
    let mut s = [[0.0; C]; X];
    for i in 0..X {
        for j in 0..C {
            let mut acc = 0.0;
            for k in 0..X {
                acc += estimated_covariance[i][k] * cross_information[k][j];
            }
            s[i][j] = -acc;
        }
    }
    s
}

/// The consider covariance `Pc = P + S·Pcc·Sᵀ` (`X × X`): the estimated-only covariance `P` inflated
/// by the considered parameters' uncertainty `Pcc` (`C × C`) through the sensitivity `S`
/// ([`consider_sensitivity`], `X × C`). `Pc ⪰ P`; `Pcc` itself is not modified.
#[must_use]
pub fn consider_covariance<const X: usize, const C: usize>(
    estimated_covariance: &[[f64; X]; X],
    sensitivity: &[[f64; C]; X],
    consider_covariance: &[[f64; C]; C],
) -> [[f64; X]; X] {
    // M = S·Pcc (X × C).
    let mut m = [[0.0; C]; X];
    for i in 0..X {
        for j in 0..C {
            let mut acc = 0.0;
            for k in 0..C {
                acc += sensitivity[i][k] * consider_covariance[k][j];
            }
            m[i][j] = acc;
        }
    }
    // Pc = P + M·Sᵀ (X × X).
    let mut out = *estimated_covariance;
    for i in 0..X {
        for j in 0..X {
            let mut acc = 0.0;
            for k in 0..C {
                acc += m[i][k] * sensitivity[j][k];
            }
            out[i][j] += acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]

    use super::*;

    /// (G2) On a three-measurement, two-state, two-parameter fixture the sensitivity and the
    /// consider covariance are the derivative of the estimate and the covariance of its error —
    /// `derivations/consider_covariance.mac` derives both from the estimator and evaluates them in
    /// exact rationals (25 digits below). `Hx = [[1,0],[1,1],[1,2]]`, `Hc = [[1,0],[½,1],[¼,−1]]`,
    /// `R = diag(¼, ½, 1)`, `Pcc = [[2,½],[½,1]]`, so `Λxc = [[21/4, 1], [3/2, 0]]` is exact in f64
    /// and `P = [[3/13, −2/13], [−2/13, 7/26]]` is rounded once on input. The bound is 8 ulps of the
    /// largest entry, 2.4.
    #[test]
    fn the_consider_covariance_is_the_error_covariance_of_the_fixture() {
        let p = [[3.0 / 13.0, -2.0 / 13.0], [-2.0 / 13.0, 7.0 / 26.0]];
        let cross = [[5.25, 1.0], [1.5, 0.0]];
        let pcc = [[2.0, 0.5], [0.5, 1.0]];
        let s_exact = [
            [
                -9.807_692_307_692_307_692_307_692e-1,
                -2.307_692_307_692_307_692_307_692e-1,
            ],
            [
                4.038_461_538_461_538_461_538_462e-1,
                1.538_461_538_461_538_461_538_462e-1,
            ],
        ];
        let pc_exact = [
            [
                2.434_171_597_633_136_094_674_556,
                -1.103_550_295_857_988_165_680_473,
            ],
            [
                -1.103_550_295_857_988_165_680_473,
                6.812_130_177_514_792_899_408_284e-1,
            ],
        ];
        let s = consider_sensitivity(&p, &cross);
        let pc = consider_covariance(&p, &s, &pcc);
        let tol = 8.0 * f64::EPSILON * 2.4;
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (s[i][j] - s_exact[i][j]).abs() <= tol,
                    "S[{i}][{j}] {}",
                    s[i][j]
                );
                assert!(
                    (pc[i][j] - pc_exact[i][j]).abs() <= tol,
                    "Pc[{i}][{j}] {}",
                    pc[i][j]
                );
            }
        }
        // (G3) With nothing considered the consider covariance is the estimated-only one, exactly.
        assert_eq!(consider_covariance(&p, &s, &[[0.0; 2]; 2]), p);
    }

    #[test]
    fn scalar_consider_matches_the_closed_form_error_variance() {
        // One state, one consider parameter, one measurement: y = hx·x + hc·c + n, var(n) = r.
        // The estimate x̂ = y/hx has error (hc·c + n)/hx, variance (hc²·Pcc + r)/hx².
        let (hx, hc, r, pcc) = (2.0, 3.0, 0.5, 4.0);
        let p = [[r / (hx * hx)]]; // Λxx⁻¹ = r/hx²
        let cross = [[hx * hc / r]]; // Λxc = hx·hc/r
        let s = consider_sensitivity(&p, &cross);
        assert!((s[0][0] - (-hc / hx)).abs() < 1e-12, "S {}", s[0][0]); // = −hc/hx
        let pc = consider_covariance(&p, &s, &[[pcc]]);
        let closed = (hc * hc * pcc + r) / (hx * hx);
        assert!(
            (pc[0][0] - closed).abs() < 1e-12,
            "Pc {} vs {closed}",
            pc[0][0]
        );
        assert!(pc[0][0] > p[0][0]); // consider covariance never tighter than estimated-only
    }

    #[test]
    fn two_state_one_consider_matches_the_hand_computed_matrix() {
        // Diagonal P so the matrix path is checkable by hand.
        let p = [[0.1, 0.0], [0.0, 0.2]];
        let cross = [[5.0], [-3.0]]; // Λxc (2×1)
        let s = consider_sensitivity(&p, &cross);
        // S = −P·Λxc = [−0.1·5, −0.2·(−3)] = [−0.5, 0.6].
        assert!(
            (s[0][0] + 0.5).abs() < 1e-12 && (s[1][0] - 0.6).abs() < 1e-12,
            "{s:?}"
        );
        let pc = consider_covariance(&p, &s, &[[2.0]]);
        // S·Pcc·Sᵀ = 2·[[0.25, −0.3], [−0.3, 0.36]]; Pc = P + that.
        let expected = [[0.6, -0.6], [-0.6, 0.92]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (pc[i][j] - expected[i][j]).abs() < 1e-12,
                    "Pc[{i}][{j}] {}",
                    pc[i][j]
                );
            }
        }
        // Symmetric and diagonal-inflated.
        assert!((pc[0][1] - pc[1][0]).abs() < 1e-15);
        assert!(pc[0][0] > p[0][0] && pc[1][1] > p[1][1]);
    }

    #[test]
    fn consider_sensitivity_matches_the_direct_information_route() {
        // For a scalar measurement Hx = [1, 0.5], Hc = 0.8, R = 0.25: Λxc = Hxᵀ·R⁻¹·Hc, and the
        // sensitivity S = −P·Λxc must equal −(P·Hxᵀ)·(Hc/R) — the same quantity computed two ways —
        // and Pc = P + S·Pcc·Sᵀ then follows. An independent check of both routines.
        let (hx0, hx1, hc, r, pcc) = (1.0, 0.5, 0.8, 0.25, 3.0);
        let p = [[0.4, 0.1], [0.1, 0.5]];
        let cross = [[hx0 * hc / r], [hx1 * hc / r]]; // Λxc = Hxᵀ·R⁻¹·Hc (2×1)
        let s = consider_sensitivity(&p, &cross);

        let phx = [p[0][0] * hx0 + p[0][1] * hx1, p[1][0] * hx0 + p[1][1] * hx1]; // P·Hxᵀ (2×1)
        for i in 0..2 {
            assert!(
                (s[i][0] - (-phx[i] * hc / r)).abs() < 1e-12,
                "S[{i}] {}",
                s[i][0]
            );
        }

        let pc = consider_covariance(&p, &s, &[[pcc]]);
        for i in 0..2 {
            for j in 0..2 {
                let expected = p[i][j] + (phx[i] * hc / r) * pcc * (phx[j] * hc / r);
                assert!(
                    (pc[i][j] - expected).abs() < 1e-12,
                    "Pc[{i}][{j}] {}",
                    pc[i][j]
                );
            }
        }
    }
}
