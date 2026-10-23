//! Richardson extrapolation as sequence acceleration and estimator debiasing.
//!
//! An estimate that depends on a scalar control `t` — the number of iterations of an optimiser, or a
//! regularisation parameter — typically carries a bias with a known *rate* but an unknown *constant*:
//!
//! ```text
//! x_t = x_* + c₁ t^α + c₂ t^{2α} + …
//! ```
//!
//! with `α = −1` when `t` is an iteration count driven to `∞`, and `α = +1` when `t` is a
//! regularisation parameter driven to `0`. Richardson extrapolation combines estimates at several `t`
//! to cancel the leading bias terms *without knowing the constants* `cⱼ`. In the variable `u = t^α`
//! the expansion is an ordinary power series, so polynomial extrapolation to `u = 0` over `m` nodes
//! removes the first `m − 1` terms and leaves `O(t^{mα})` — proven in
//! `derivations/richardson_acceleration.mac`.
//!
//! This is the debiasing primitive of Bach, *On the Effectiveness of Richardson Extrapolation in Data
//! Science* (SIAM J. Math. Data Sci. 3(4), 2021): it turns an `O(1/k)` averaged-gradient / Frank–Wolfe
//! rate into `O(1/k²)`, and removes the leading `O(λ)` bias of a ridge / Tikhonov estimator. It is the
//! statistical cousin of the same extrapolation the [`crate`]'s sibling
//! `spacelab_propagation::gbs` applies to ODE steps. Estimator-agnostic: it only sees the outputs.
//!
//! `no_std`; the caller supplies the estimates.

#![allow(clippy::cast_precision_loss)] // the node count M is tiny, exactly representable in f64

/// The two-point Richardson estimate of `x_*` from estimates at controls `t1` and `t2` (`t1 ≠ t2`),
/// under the leading model `x_t = x_* + c·t^α`. Cancels the `c·t^α` bias term.
///
/// With `t2 = 2·t1` this is Bach's `(2^α x₁ − x₂)/(2^α − 1)`. For iteration counts use `α = −1`; for a
/// regularisation parameter use `α = +1`.
#[must_use]
pub fn richardson2<const N: usize>(
    x1: &[f64; N],
    t1: f64,
    x2: &[f64; N],
    t2: f64,
    alpha: f64,
) -> [f64; N] {
    let (u1, u2) = (libm::pow(t1, alpha), libm::pow(t2, alpha));
    // x_* = (u2 x1 − u1 x2)/(u2 − u1): the line through (u1,x1),(u2,x2) evaluated at u = 0.
    let inv = 1.0 / (u2 - u1);
    core::array::from_fn(|i| (u2 * x1[i] - u1 * x2[i]) * inv)
}

/// The full `M`-point Richardson extrapolation of a vector estimate to the `t^α → 0` limit.
///
/// `nodes[i]` are the controls (all positive and distinct, ordered coarse→fine, i.e. `|nodes|`
/// decreasing for `α > 0` and increasing for `α < 0`); `values[i]` is the estimate at `nodes[i]`.
/// Returns the extrapolated estimate (the tableau diagonal `T_{M,M}`, order `M` in the node scale) and
/// an error estimate `‖T_{M,M} − T_{M,M−1}‖₂` — the change from the previous order, the usual
/// convergence gauge.
#[must_use]
pub fn richardson_extrapolate<const N: usize, const M: usize>(
    nodes: &[f64; M],
    values: &[[f64; N]; M],
    alpha: f64,
) -> ([f64; N], f64) {
    let u: [f64; M] = core::array::from_fn(|i| libm::pow(nodes[i], alpha));
    let mut prev = [[0.0; N]; M]; // T_{i-1,·}
    let mut row = [[0.0; N]; M];
    let mut err = 0.0;
    for i in 0..M {
        row = [[0.0; N]; M];
        row[0] = values[i];
        for c in 1..=i {
            // T_{i,c+1} from T_{i,c} and T_{i-1,c}; denom = u_{i-c}/u_i − 1 (Neville to u = 0).
            let denom = u[i - c] / u[i] - 1.0;
            for k in 0..N {
                row[c][k] = row[c - 1][k] + (row[c - 1][k] - prev[c - 1][k]) / denom;
            }
        }
        if i >= 1 {
            err = libm::sqrt(
                (0..N)
                    .map(|k| {
                        let d = row[i][k] - row[i - 1][k];
                        d * d
                    })
                    .sum(),
            );
        }
        prev = row;
    }
    (row[M - 1], err)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]
    use super::*;

    /// The reported error is the change from the previous order, on a tableau known in closed form.
    ///
    /// With α = 1 and nodes 4, 2, 1 the Neville denominators are exactly 1, 1 and 3, so a quadratic
    /// `f(h) = L + a h + b h²` gives an integer tableau: the second column is `L − 8b` and `L − 2b`,
    /// the diagonal is `L`, and the reported error is `|T₂,₂ − T₂,₁| = 2|b|`. At `L = 5, a = 3,
    /// b = 7` that is 14, against 4 for a sum in place of the difference, 0 for an index that reads
    /// the same row twice, 2·14 under a square root for a sum of the terms and 1 for their ratio.
    /// The existing tests all assert that the extrapolated value is close to the limit, which the
    /// error estimate does not enter.
    #[test]
    fn the_reported_error_is_the_gap_between_the_last_two_orders() {
        let f = |h: f64| 5.0 + 3.0 * h + 7.0 * h * h;
        let nodes = [4.0, 2.0, 1.0];
        let values: [[f64; 1]; 3] = core::array::from_fn(|i| [f(nodes[i])]);

        let (value, err) = richardson_extrapolate::<1, 3>(&nodes, &values, 1.0);
        assert!((value[0] - 5.0).abs() < 1e-12, "T₂,₂ = {}", value[0]);
        assert!((err - 14.0).abs() < 1e-12, "err = {err}");
    }

    /// The iteration-count case (α = −1): a `1/k` bias expansion, as in averaged SGD / Frank–Wolfe.
    /// Three-point Richardson removes the `1/k` and `1/k²` terms, recovering the limit far better than
    /// the raw estimate.
    #[test]
    fn accelerates_a_one_over_k_sequence() {
        let x_star = 5.0;
        let seq = |k: f64| [x_star + 3.0 / k - 2.0 / (k * k) + 0.5 / (k * k * k)];
        let nodes = [10.0, 20.0, 40.0];
        let values = [seq(10.0), seq(20.0), seq(40.0)];
        let (ext, err) = richardson_extrapolate(&nodes, &values, -1.0);
        let raw_err = (values[2][0] - x_star).abs(); // best raw sample (k = 40)
        let ext_err = (ext[0] - x_star).abs();
        println!(
            "1/k accel: raw(k=40) err {raw_err:.3e}, Richardson err {ext_err:.3e}, gauge {err:.3e}"
        );
        assert!(
            ext_err < raw_err / 100.0,
            "extrapolation did not accelerate: {ext_err:.3e} vs {raw_err:.3e}"
        );
    }

    /// The regularisation case (α = +1): a scalar ridge estimate `β̂(λ) = s·β/(s+λ)` has bias `O(λ)`;
    /// two-point Richardson over `(λ, 2λ)` cuts it to `O(λ²)` — Bach's kernel-ridge debiasing.
    #[test]
    fn debiases_ridge_regression() {
        let (s, beta) = (4.0_f64, 2.0_f64); // XᵀX and the true coefficient (noiseless)
        let ridge = |lam: f64| [s * beta / (s + lam)];
        let lam = 0.1;
        let deb = richardson2(&ridge(lam), lam, &ridge(2.0 * lam), 2.0 * lam, 1.0);
        let ridge_bias = (ridge(lam)[0] - beta).abs();
        let deb_bias = (deb[0] - beta).abs();
        println!("ridge debias: ridge(λ) bias {ridge_bias:.3e}, debiased bias {deb_bias:.3e}");
        assert!(
            deb_bias < ridge_bias / 20.0,
            "debiasing failed: {deb_bias:.3e} vs {ridge_bias:.3e}"
        );
        // and the debiased bias should scale like λ²: halving λ quarters it (roughly).
        let deb_half = richardson2(&ridge(lam / 2.0), lam / 2.0, &ridge(lam), lam, 1.0);
        let ratio = deb_bias / (deb_half[0] - beta).abs();
        assert!(ratio > 3.0, "debiased bias is not O(λ²) (ratio {ratio:.2})");
    }

    /// The general tableau agrees with the two-point closed form when given two nodes.
    #[test]
    fn two_point_matches_the_tableau() {
        let x1 = [1.25, -0.5];
        let x2 = [1.10, -0.2];
        let (t1, t2, a) = (0.2, 0.4, 1.0);
        let closed = richardson2(&x1, t1, &x2, t2, a);
        let (tab, _) = richardson_extrapolate(&[t1, t2], &[x1, x2], a);
        for i in 0..2 {
            assert!(
                (closed[i] - tab[i]).abs() < 1e-14,
                "axis {i}: {} vs {}",
                closed[i],
                tab[i]
            );
        }
    }
}
