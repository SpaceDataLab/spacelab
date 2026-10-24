//! Forward-mode automatic differentiation.
//!
//! A [`Dual<N>`] carries a value and its `N` partial derivatives, propagating them through every
//! operation by the exact chain rule. Seed input `i` as a unit direction and the gradient of every
//! output falls out — a full Jacobian in one evaluation of the *same* code that computes the value.
//!
//! # Why this exists, and why it is an accuracy tool
//!
//! Orbit determination, the state transition matrix, and every sensitivity need partial derivatives
//! of the force model and the measurement geometry. The alternatives are worse: hand-coded partials
//! fall out of sync with the model the moment either changes, and finite differences have a floor —
//! the truncation error falls as `h` while the rounding error rises as `ε/h`, so the best achievable
//! relative accuracy is `√ε ≈ 1e-8`, eight digits thrown away in the one matrix that decides how much
//! to trust each observation. Dual numbers have no such floor: the derivative is exact to the same
//! rounding as the value.
//!
//! # The `Scalar` abstraction
//!
//! Write the geometry once, generic over [`Scalar`], and evaluate it in `f64` for the value or in
//! [`Dual<N>`] for the value *and* the Jacobian — the type system guarantees they are the same
//! function. This is Orekit's `Field`/`Gradient` idea, in a `Copy` type with no allocation.
//!
//! `tests/derivatives.rs` holds the derivatives to closed forms — the elementary functions, the chain,
//! product and quotient rules, and the Kepler-acceleration Jacobian and Hessian — and sets them beside
//! a finite difference.

#![cfg_attr(not(test), no_std)]
#![allow(clippy::needless_range_loop)] // the loops index several arrays by one i; iterators would obscure them

mod double_double;
mod dual;
mod hyperdual;
mod scalar;

pub use double_double::DoubleDouble;
pub use dual::Dual;
pub use hyperdual::HyperDual;
pub use scalar::Scalar;

/// Compute the `M×N` Jacobian of `f: ℝᴺ → ℝᴹ` at `x` by one forward-mode pass.
///
/// `jac[m][n] = ∂fₘ/∂xₙ`, exact to rounding. `f` is written once, generic over [`Scalar`], so the
/// same source gives the value in `f64` and the Jacobian here.
pub fn jacobian<const N: usize, const M: usize, F>(f: F, x: &[f64; N]) -> ([f64; M], [[f64; M]; N])
where
    F: Fn(&[Dual<N>; N]) -> [Dual<N>; M],
{
    // seed each input as x_i + ε·e_i
    let mut seeds = [Dual::constant(0.0); N];
    for i in 0..N {
        seeds[i] = Dual::variable(x[i], i);
    }
    let out = f(&seeds);
    let mut value = [0.0; M];
    let mut jac = [[0.0; M]; N];
    for m in 0..M {
        value[m] = out[m].value();
        for n in 0..N {
            jac[n][m] = out[m].grad()[n];
        }
    }
    (value, jac)
}

/// Compute the value, Jacobian, and full Hessian of `f: ℝᴺ → ℝᴹ` at `x` by one forward second-order
/// pass — the second-order companion to [`jacobian`], used to build the state-transition tensor.
///
/// Returns `(value, grad, hess)` in **output-major** layout: `grad[i][j] = ∂fᵢ/∂xⱼ` (so `grad[i]` is
/// `∇fᵢ`, the `i`-th row of the Jacobian `A`) and `hess[i][j][k] = ∂²fᵢ/∂xⱼ∂xₖ` (so `hess[i]` is the
/// symmetric Hessian of output `i`). This differs on purpose from [`jacobian`]'s input-major
/// `jac[n][m] = ∂fₘ/∂xₙ`: the variational STT equations read `A` and the force Hessian directly in
/// this form. `f` is written once, generic over [`Scalar`], so the same source gives the value in
/// `f64`, the Jacobian at [`Dual`], and the Hessian here — none can drift from the others.
#[allow(clippy::type_complexity)] // (value, grad, hess) is the natural triple; a struct would not clarify it
pub fn hessian<const N: usize, const M: usize, F>(
    f: F,
    x: &[f64; N],
) -> ([f64; M], [[f64; N]; M], [[[f64; N]; N]; M])
where
    F: Fn(&[HyperDual<N>; N]) -> [HyperDual<N>; M],
{
    // seed each input as a unit first-direction with zero second-order part
    let mut seeds = [HyperDual::constant(0.0); N];
    for i in 0..N {
        seeds[i] = HyperDual::variable(x[i], i);
    }
    let out = f(&seeds);
    let mut value = [0.0; M];
    let mut grad = [[0.0; N]; M];
    let mut hess = [[[0.0; N]; N]; M];
    for m in 0..M {
        value[m] = out[m].value();
        grad[m] = out[m].grad();
        hess[m] = out[m].hess();
    }
    (value, grad, hess)
}
