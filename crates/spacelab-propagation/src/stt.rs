//! The second-order state-transition tensor (STT), by integrating the second-order variational
//! equations with an AD Hessian.
//!
//! The first-order STM `Φ` in [`crate::variational`] is the linear map of a deviation: it answers
//! "how does a small change in the initial state move the final state, to first order". For
//! *nonlinear* uncertainty propagation that is not enough — a Gaussian pushed through a curved flow
//! is no longer Gaussian, and the linear map cannot see the curvature that skews the mean and
//! inflates the covariance. The **STT** `Ψ` is that curvature. Expanding the flow `x(t) = φ(t; x₀)`
//! about a nominal `x₀`,
//!
//! ```text
//! δxᵢ(t) = Φᵢⱼ δx₀ⱼ + ½ Ψᵢⱼₖ δx₀ⱼ δx₀ₖ + O(δ³),
//! Φᵢⱼ = ∂xᵢ/∂x₀ⱼ,   Ψᵢⱼₖ = ∂²xᵢ/∂x₀ⱼ∂x₀ₖ  (symmetric in j, k).
//! ```
//!
//! `Φ` and `Ψ` are the solution of the variational equations (Park & Scheeres 2006):
//!
//! ```text
//! Φ̇ᵢₖ = Aᵢₗ Φₗₖ,                              A = ∂f/∂x,
//! Ψ̇ᵢⱼₖ = Aᵢₗ Ψₗⱼₖ + (∂²fᵢ/∂xₗ∂xₘ) Φₗⱼ Φₘₖ,
//! Φ(t₀) = I,   Ψ(t₀) = 0,
//! ```
//!
//! integrated alongside the state with the same method. Both `A = ∂f/∂x` and the force Hessian
//! `∂²f/∂x²` come from one second-order automatic-differentiation pass over the *same*
//! [`ScalarSystem`] force that computes the value ([`spacelab_autodiff::hessian`]) — exact to
//! rounding, never hand-derived, never able to drift from the model. This is where the crate exceeds
//! Orekit, which ships only the first-order STM.
//!
//! # Storage
//!
//! `Ψ` is symmetric in its last two indices, so the augmented state stores only the upper triangle:
//! `N·N(N+1)/2` entries, not `N³`. The augmented vector is `[x (N); Φ (N², row-major); Ψ (N·T,
//! component-major over a packed (j,k) triangle)]`, dimension `M = N + N² + N·N(N+1)/2`. The caller
//! supplies both `N` and `M` (there is no stable const arithmetic to derive `M`), and a runtime
//! assertion guards the pairing. For an orbit, `N = 6`, `T = 21`, `M = 168`.
//!
//! # Moment mapping
//!
//! [`map_deviation`] pushes a single deviation `δx₀ → δx(t)` to second order. [`map_covariance`]
//! pushes a Gaussian: with `δx₀ ~ N(0, P₀)`, the nonlinearity biases the mean by
//! `μᵢ = ½ Ψᵢₐᵦ P₀ₐᵦ` and maps the covariance to `Pᵢⱼ = Φᵢₐ P₀ₐ𝒸 Φⱼ𝒸 + ½ Ψᵢₐᵦ Ψⱼ𝒸𝒹 P₀ₐ𝒸 P₀ᵦ𝒹`
//! (Park & Scheeres 2006, the Gaussian second-order moment update via Isserlis' theorem).

// The variational tensors index several packed arrays by hand; sequential-index iterators would
// obscure the ODE. The moment sums run over the Greek indices (a, b, c, d) of the Park–Scheeres
// equations — the domain's names, not terse locals.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use spacelab_autodiff::{Scalar, hessian};

use crate::DynamicalSystem;
use crate::variational::ScalarSystem;

/// The augmented dimension `M = N + N² + N·N(N+1)/2` for a state of size `N`.
#[must_use]
pub const fn augmented_len(n: usize) -> usize {
    n + n * n + n * n * (n + 1) / 2
}

/// The length `T = N(N+1)/2` of one component's packed symmetric `(j, k)` triangle.
const fn triangle_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// The packed index of the symmetric pair `(j, k)` with `j ≤ k`, into a length-`T` upper triangle.
///
/// Row `j` starts at `j·n − (j²−j)/2` (the running count of the wider rows above it), then the offset
/// within the row is `k − j`. `j ≤ k` is a precondition — the caller iterates `k` from `j`.
const fn packed_index(j: usize, k: usize, n: usize) -> usize {
    j * n - (j * j - j) / 2 + (k - j)
}

/// Wraps a [`ScalarSystem`] as the augmented `[state; Φ; Ψ]` second-order variational system of
/// dimension `M = N + N² + N·N(N+1)/2`.
#[derive(Debug, Clone, Copy)]
pub struct VariationalStt<'a, const N: usize, F> {
    force: &'a F,
}

impl<'a, const N: usize, F> VariationalStt<'a, N, F> {
    /// Wrap a force model. Pair with `M = augmented_len(N)` when integrating.
    #[must_use]
    pub const fn new(force: &'a F) -> Self {
        Self { force }
    }
}

impl<const N: usize, const M: usize, F: ScalarSystem<N>> DynamicalSystem<M>
    for VariationalStt<'_, N, F>
{
    fn derivative(&self, t: f64, y: &[f64; M], dydt: &mut [f64; M]) {
        assert!(
            M == augmented_len(N),
            "augmented dimension M must equal N + N² + N·N(N+1)/2"
        );
        let tri = triangle_len(N);
        let psi_base = N + N * N;

        // State, Jacobian A[i][l] = ∂fᵢ/∂xₗ, and Hessian H[i][l][m] = ∂²fᵢ/∂xₗ∂xₘ — one AD pass, the
        // value doubling as f(t, s).
        let mut s = [0.0; N];
        s.copy_from_slice(&y[..N]);
        let (f, a, h) = hessian::<N, N, _>(|v| self.force.eval(Scalar::from_f64(t), v), &s);
        dydt[..N].copy_from_slice(&f);

        // Φ̇ᵢₖ = Σₗ Aᵢₗ Φₗₖ, with Φ (and Φ̇) row-major: Φₗₖ = y[N + l·N + k].
        for i in 0..N {
            for k in 0..N {
                let mut acc = 0.0;
                for l in 0..N {
                    acc += a[i][l] * y[N + l * N + k];
                }
                dydt[N + i * N + k] = acc;
            }
        }

        // Ψ̇ᵢⱼₖ = Σₗ Aᵢₗ Ψₗⱼₖ + Σₗₘ Hᵢₗₘ Φₗⱼ Φₘₖ, computed for j ≤ k (Ψ is symmetric in j, k, and so
        // is Ψ̇ — the coupling term is symmetric because H is symmetric in l, m).
        for i in 0..N {
            for j in 0..N {
                for k in j..N {
                    let mut acc = 0.0;
                    for l in 0..N {
                        acc += a[i][l] * y[psi_base + l * tri + packed_index(j, k, N)];
                    }
                    for l in 0..N {
                        for m in 0..N {
                            acc += h[i][l][m] * y[N + l * N + j] * y[N + m * N + k];
                        }
                    }
                    dydt[psi_base + i * tri + packed_index(j, k, N)] = acc;
                }
            }
        }
    }
}

/// Build the augmented initial state `[y0; I; 0]` of length `M = augmented_len(N)`: the state, the
/// identity STM `Φ(t₀) = I`, and the zero STT `Ψ(t₀) = 0`.
///
/// # Panics
///
/// If `M != augmented_len(N)`.
#[must_use]
pub fn augment_stt<const N: usize, const M: usize>(y0: &[f64; N]) -> [f64; M] {
    assert!(
        M == augmented_len(N),
        "augmented dimension M must equal N + N² + N·N(N+1)/2"
    );
    let mut a = [0.0; M];
    a[..N].copy_from_slice(y0);
    for i in 0..N {
        a[N + i * N + i] = 1.0; // Φ(t₀) = I
    }
    a // Ψ(t₀) = 0, already zeroed
}

/// Extract the state `y` (first `N`), the STM `Φ` (`N×N`, row-major), and the STT `Ψ` (`N×N×N`, the
/// packed triangle expanded to the full tensor, symmetric in the last two indices) from an augmented
/// vector.
///
/// # Panics
///
/// If `M != augmented_len(N)`.
#[must_use]
#[allow(clippy::type_complexity)] // (state, Φ, Ψ) is the natural triple the caller consumes
pub fn split_stt<const N: usize, const M: usize>(
    aug: &[f64; M],
) -> ([f64; N], [[f64; N]; N], [[[f64; N]; N]; N]) {
    assert!(
        M == augmented_len(N),
        "augmented dimension M must equal N + N² + N·N(N+1)/2"
    );
    let tri = triangle_len(N);
    let psi_base = N + N * N;

    let mut y = [0.0; N];
    y.copy_from_slice(&aug[..N]);

    let mut phi = [[0.0; N]; N];
    for i in 0..N {
        for k in 0..N {
            phi[i][k] = aug[N + i * N + k];
        }
    }

    let mut psi = [[[0.0; N]; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in j..N {
                let v = aug[psi_base + i * tri + packed_index(j, k, N)];
                psi[i][j][k] = v;
                psi[i][k][j] = v; // fill the symmetric partner
            }
        }
    }
    (y, phi, psi)
}

/// Map a state deviation `δx₀ → δx(t)` to second order: `δxᵢ = Φᵢⱼ δx₀ⱼ + ½ Ψᵢⱼₖ δx₀ⱼ δx₀ₖ`.
///
/// The linear term alone is the STM prediction; the quadratic term is the STT correction that
/// captures the leading nonlinearity, cutting the residual against the true propagated deviation from
/// `O(‖δx₀‖²)` to `O(‖δx₀‖³)`.
#[must_use]
pub fn map_deviation<const N: usize>(
    phi: &[[f64; N]; N],
    psi: &[[[f64; N]; N]; N],
    dx0: &[f64; N],
) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        let mut lin = 0.0;
        for j in 0..N {
            lin += phi[i][j] * dx0[j];
        }
        let mut quad = 0.0;
        for j in 0..N {
            for k in 0..N {
                quad += psi[i][j][k] * dx0[j] * dx0[k];
            }
        }
        out[i] = lin + 0.5 * quad;
    }
    out
}

/// Map a Gaussian `δx₀ ~ N(0, P₀)` through the flow to second order (Park & Scheeres 2006).
///
/// Returns `(mean_shift, covariance)`:
/// * `mean_shift[i] = ½ Σₐᵦ Ψᵢₐᵦ P₀ₐᵦ` — the bias the curvature adds. The propagated mean is the
///   STM-propagated nominal *plus* this shift; a linear filter, which omits it, is biased.
/// * `covariance[i][j] = Σₐ𝒸 Φᵢₐ Φⱼ𝒸 P₀ₐ𝒸 + ½ Σₐᵦ𝒸𝒹 Ψᵢₐᵦ Ψⱼ𝒸𝒹 P₀ₐ𝒸 P₀ᵦ𝒹` — the linear covariance
///   `Φ P₀ Φᵀ` plus the second-order inflation, from the fourth Gaussian moment (Isserlis) with the
///   mean-shift outer product folded in.
#[must_use]
#[allow(clippy::type_complexity)] // (mean_shift, covariance) is the natural moment pair
pub fn map_covariance<const N: usize>(
    phi: &[[f64; N]; N],
    psi: &[[[f64; N]; N]; N],
    p0: &[[f64; N]; N],
) -> ([f64; N], [[f64; N]; N]) {
    let mut mean = [0.0; N];
    for i in 0..N {
        let mut s = 0.0;
        for a in 0..N {
            for b in 0..N {
                s += psi[i][a][b] * p0[a][b];
            }
        }
        mean[i] = 0.5 * s;
    }

    // The quartic ΨΨP₀P₀ term is O(N⁴) per (i,j) — for the orbit N = 6 this is a one-shot moment
    // update, not a per-step cost, so the direct sum is kept for legibility against the closed form.
    let mut cov = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut lin = 0.0;
            for a in 0..N {
                for c in 0..N {
                    lin += phi[i][a] * phi[j][c] * p0[a][c];
                }
            }
            let mut sec = 0.0;
            for a in 0..N {
                for b in 0..N {
                    for c in 0..N {
                        for d in 0..N {
                            sec += psi[i][a][b] * psi[j][c][d] * p0[a][c] * p0[b][d];
                        }
                    }
                }
            }
            cov[i][j] = lin + 0.5 * sec;
        }
    }
    (mean, cov)
}

#[cfg(test)]
mod tests {
    use super::augmented_len;

    /// The augmented length against its own documented layout, at literal values.
    ///
    /// `tests/stt.rs` writes `const M: usize = augmented_len(6)` and then asserts `M ==
    /// augmented_len(N)`, so a wrong formula sizes the arrays and the assertion consistently: the
    /// state, STM and tensor blocks are packed from the front by `N`, `N²` and `N·N(N+1)/2`
    /// separately, and whatever length is left over is simply zeros nobody reads. Three mutants of
    /// the expression survived on that. The oracle here is the layout stated in the doc comment,
    /// evaluated by hand -- `N` state, `N²` first-order STM, `N` triangles of `N(N+1)/2`.
    #[test]
    fn the_augmented_length_is_the_state_the_stm_and_the_packed_tensor() {
        for (n, want) in [(1, 3), (2, 12), (3, 30), (4, 60), (6, 168)] {
            assert_eq!(
                augmented_len(n),
                want,
                "augmented_len({n}) should be {n} + {n}² + {n}·{n}({n}+1)/2 = {want}"
            );
        }
    }
}
