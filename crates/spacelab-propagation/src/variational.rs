//! The state transition matrix, by integrating the variational equations with an AD Jacobian.
//!
//! The STM `Φ(t, t₀) = ∂y(t)/∂y(t₀)` is what a batch least-squares or a Kalman filter needs: it
//! propagates the covariance and maps an observation's partials back to the epoch state
//! ([notes/14-estimation.md](../notes/14-estimation.md)). It is the solution of the variational
//! equation
//!
//! ```text
//! dΦ/dt = A(t) Φ,   Φ(t₀, t₀) = I,   A(t) = ∂f/∂y
//! ```
//!
//! integrated alongside the state with the same method. The Jacobian `A` comes from
//! [`spacelab_autodiff`] — exact to rounding, from the *same* force-model source that computes the
//! value, so the two cannot drift apart. Finite-difference partials would cap the STM at ~1e-8
//! relative (see [`spacelab_autodiff`]); AD does not.
//!
//! The force model is a [`ScalarSystem`], written once over any [`spacelab_autodiff::Scalar`]. The
//! augmented `[state (N); Φ (N×N, row-major)]` state has dimension `M = N + N²`; the caller supplies
//! both (there is no stable const arithmetic to derive `M` from `N`), and a runtime assertion guards
//! the pairing. For an orbit, `N = 6`, `M = 42`.

use spacelab_autodiff::{Scalar, jacobian};

use crate::DynamicalSystem;

/// A force model written generically over a scalar, so the same source gives the value (`f64`) and
/// the Jacobian (`Dual<N>`).
pub trait ScalarSystem<const N: usize> {
    /// The derivative `dy/dt` at `(t, y)`.
    ///
    /// `t` is the same scalar as the state, not an `f64`, because [`crate::taylor`] steps the clock
    /// along with the solution: its jet recurrence needs `∂f/∂t` at every order, and a force that
    /// cannot supply it truncates the series after the first one. Forces whose time dependence goes
    /// through an `f64`-only subsystem — an ephemeris, a density table, the frame rotation — take
    /// `t.value()` and are legitimately flat in `t`; [`crate::rosenbrock`] documents what that costs
    /// a method that needs `∂f/∂t` explicitly.
    fn eval<S: Scalar>(&self, t: S, y: &[S; N]) -> [S; N];

    /// f64-only fast path, defaulting to `eval::<f64>`. Overridden (e.g. by
    /// [`crate::forces::Newtonian`]) to route through the acceleration f64 fast path;
    /// [`crate::forces::Plain`] calls this for f64 propagation, leaving the AD/STM variational path
    /// on the generic `eval`.
    #[inline]
    fn eval_f64(&self, t: f64, y: &[f64; N]) -> [f64; N] {
        self.eval::<f64>(t, y)
    }
}

/// Wraps a [`ScalarSystem`] as the augmented `[state; STM]` system of dimension `M = N + N²`.
#[derive(Debug, Clone, Copy)]
pub struct Variational<'a, const N: usize, F> {
    force: &'a F,
}

impl<'a, const N: usize, F> Variational<'a, N, F> {
    /// Wrap a force model. Pair with `M = N + N²` when integrating.
    #[must_use]
    pub const fn new(force: &'a F) -> Self {
        Self { force }
    }
}

impl<const N: usize, const M: usize, F: ScalarSystem<N>> DynamicalSystem<M>
    for Variational<'_, N, F>
{
    fn derivative(&self, t: f64, y: &[f64; M], dydt: &mut [f64; M]) {
        assert!(M == N + N * N, "augmented dimension M must equal N + N²");

        // state, and its Jacobian A in one AD pass — value doubles as f(t, s).
        let mut s = [0.0; N];
        s.copy_from_slice(&y[..N]);
        // jac[n][m] = ∂fₘ/∂yₙ, so A[i][j] = ∂fᵢ/∂yⱼ = jac[j][i].
        let (f, jac) = jacobian::<N, N, _>(|v| self.force.eval(Scalar::from_f64(t), v), &s);
        dydt[..N].copy_from_slice(&f);

        // dΦ = A Φ, with Φ and dΦ row-major in y[N..] and dydt[N..].
        for i in 0..N {
            for k in 0..N {
                let mut acc = 0.0;
                for j in 0..N {
                    acc += jac[j][i] * y[N + j * N + k];
                }
                dydt[N + i * N + k] = acc;
            }
        }
    }
}

/// Build the augmented initial state `[y0; I]` of length `M = N + N²`.
///
/// # Panics
///
/// If `M != N + N²`.
#[must_use]
pub fn augment<const N: usize, const M: usize>(y0: &[f64; N]) -> [f64; M] {
    assert!(M == N + N * N, "augmented dimension M must equal N + N²");
    let mut a = [0.0; M];
    a[..N].copy_from_slice(y0);
    for i in 0..N {
        a[N + i * N + i] = 1.0; // Φ(t₀) = I
    }
    a
}

/// Extract the state `y` (first `N`) and the STM `Φ` (`N×N`, row-major) from an augmented vector.
///
/// # Panics
///
/// If `M != N + N²`.
#[must_use]
pub fn split<const N: usize, const M: usize>(aug: &[f64; M]) -> ([f64; N], [[f64; N]; N]) {
    assert!(M == N + N * N, "augmented dimension M must equal N + N²");
    let mut y = [0.0; N];
    y.copy_from_slice(&aug[..N]);
    let mut phi = [[0.0; N]; N];
    for i in 0..N {
        for k in 0..N {
            phi[i][k] = aug[N + i * N + k];
        }
    }
    (y, phi)
}
