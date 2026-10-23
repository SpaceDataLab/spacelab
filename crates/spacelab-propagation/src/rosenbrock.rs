//! Rosenbrock GRK4T, order 4: a *linearly-implicit* Runge–Kutta for stiff systems.
//!
//! # Linearly implicit — the middle ground
//!
//! [`Radau`](crate::radau) is fully implicit: each step solves a nonlinear stage system by Newton
//! iteration, several `3N×3N` solves per step. An explicit method needs none of that but is
//! unstable on stiff problems. A Rosenbrock method sits between: it keeps the Jacobian `J = ∂f/∂y`
//! in the method (so it inherits an implicit method's stability) but arranges the stages so each is
//! a **single linear solve**, no iteration. One `N×N` LU factorisation of `(I/(γh) − J)` per step is
//! reused across all four stages — four back-substitutions. That is the whole appeal: A-stable
//! behaviour at a fixed, predictable cost per step.
//!
//! # Coefficients
//!
//! GRK4T is the order-4 method of Kaps & Rentrop (1979), `A(89.3°)`-stable with small truncation
//! error — the method Numerical Recipes ships. The tableau is transcribed verbatim from the
//! authoritative Hairer–Wanner reference code `ros4.f` (`SUBROUTINE GRK4T`), the primary source, not
//! from memory. `derivations/rosenbrock_grk4t.mac` re-derives the linear order conditions by
//! applying the exact stage recurrence below to the Dahlquist test `y' = λy` and confirming the
//! stability function matches `eᶻ` through `z⁴` (the four linear order conditions); the nonlinear
//! order conditions are exercised by the order-4 convergence test on the Kepler problem.
//!
//! # Applicability
//!
//! Unlike the symplectic and Gauss–Jackson methods, a Rosenbrock method places no restriction on the
//! force — the Jacobian is taken of the whole `[r, v]` system, so velocity-dependent forces (drag)
//! are fine. The f-evaluations use the stage times `t + cᵢh`, and each stage also carries the
//! `h·dᵢ·∂f/∂t` term of the full `ros4.f`, without which the method is **order one** on any force
//! that reads `t` — which every force in `spacelab::forces` does. Measured on `y' = y·cos t` over
//! `[1, 3]`: 2.2815e-3 → 1.3957e-4 at n = 20 … 320 without the terms, every ratio 2.00–2.02.
//!
//! The `dᵢ = Σⱼ γᵢⱼ` weights are not transcribed: `derivations/rosenbrock_grk4t.mac` derives them
//! from the `cᵢⱼ` above by `Γ = (diag(1/γ) − C)⁻¹`, and confirms the derivation by recovering the
//! published stage times `c₂` and `c₃` from the same matrix. `∂f/∂t` comes from the one-sided
//! difference `ros4.f` itself uses when the caller supplies no analytic form (`IDFX = 0`). The
//! autodiff that gives `∂f/∂y` exactly *can* reach `t` — [`ScalarSystem::eval`] takes it as the
//! generic scalar — but it would report `∂f/∂t = 0` for every force whose clock runs through an
//! `f64`-only subsystem, which is every force this term exists for. The difference quotient sees
//! that dependence; a differentiated `t` would silently not.

// The tableau entries are full-precision literals from ros4.f; the step count is a small integer.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use spacelab_autodiff::{Scalar, jacobian};

use crate::driver::{
    HermiteDense, IntegrationError, Trajectory, integrate_fixed_hermite, step_count,
};
use crate::linalg::lu_factor;
use crate::variational::ScalarSystem;

const GAMMA: f64 = 0.231;
const A21: f64 = 0.2000000000000000e1;
const A31: f64 = 0.4524708207373116e1;
const A32: f64 = 0.4163528788597648e1;
const C21: f64 = -0.5071675338776316e1;
const C31: f64 = 0.6020152728650786e1;
const C32: f64 = 0.1597506846727117e0;
const C41: f64 = -0.1856343618686113e1;
const C42: f64 = -0.8505380858179826e1;
const C43: f64 = -0.2084075136023187e1;
const B1: f64 = 0.3957503746640777e1;
const B2: f64 = 0.4624892388363313e1;
const B3: f64 = 0.6174772638750108e0;
const B4: f64 = 0.1282612945269037e1;
/// Stage times `cᵢ` (only affect a non-autonomous force).
const C2: f64 = 0.4620000000000000e0;
const C3: f64 = 0.8802083333333334e0;
/// `dᵢ = Σⱼ γᵢⱼ`, the weights of the `h·dᵢ·∂f/∂t` stage terms, from
/// `derivations/rosenbrock_grk4t.mac` — derived from the `cᵢⱼ` above at `fpprec: 60` rather than
/// transcribed, so they are consistent with the `cᵢⱼ` this file actually uses to the last bit.
/// They land within 1.1 ulp of the values `ros4.f` prints (`-0.3962966775244303e-1`,
/// `0.5507789395789127e0`, `-0.5535098457052764e-1`), which is the check that the derivation is
/// the right one. `d₁ = γ₁₁ = γ` is an identity of the tableau, so it is written as one.
const D1: f64 = GAMMA;
const D2: f64 = -0.39629667752443025e-1;
const D3: f64 = 0.5507789395789128e0;
const D4: f64 = -0.5535098457052765e-1;

/// One GRK4T step of size `h` from `(t, y)`: one LU factorisation, four linear stage solves.
fn rosenbrock_step<const N: usize>(
    force: &impl ScalarSystem<N>,
    t: f64,
    y: &[f64; N],
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    // f and its exact Jacobian at (t, y). jac[n][m] = ∂fₘ/∂yₙ ⇒ J[r][c] = jac[c][r].
    let (f0, jac) = jacobian::<N, N, _>(|v| force.eval(Scalar::from_f64(t), v), y);

    // ∂f/∂t, one-sided at the increment ros4.f picks for `IDFX = 0`: δ = √(u·max(1e-5, |t|)),
    // which scales with the epoch, so it stays above `ulp(t)` at a J2000 `t` and above zero at
    // `t = 0`. One extra f-evaluation per step, and not replaceable by AD: `eval` takes `t` as the
    // generic scalar, but a force reading the clock through an ephemeris or a density table takes
    // `t.value()` and differentiates to zero, while the difference still sees it.
    let delta = (f64::EPSILON * t.abs().max(1.0e-5)).sqrt();
    let f_dt = force.eval::<f64>(t + delta, y);
    let mut dfdt = [0.0; N];
    for i in 0..N {
        dfdt[i] = (f_dt[i] - f0[i]) / delta;
    }

    // E = (1/(γh)) I − J, factored once and reused for every stage.
    let fac = 1.0 / (GAMMA * h);
    let mut e = [[0.0; N]; N];
    for r in 0..N {
        for c in 0..N {
            e[r][c] = -jac[c][r];
        }
        e[r][r] += fac;
    }
    let lu = lu_factor(e);
    let solve = |rhs: [f64; N]| {
        lu.solve(rhs)
            .ok_or(IntegrationError::StageSolveFailed { t })
    };
    let inv_h = 1.0 / h;

    // Stage 1: E k1 = f(y) + h d1 ∂f/∂t.
    let mut rhs = [0.0; N];
    for i in 0..N {
        rhs[i] = f0[i] + h * D1 * dfdt[i];
    }
    let k1 = solve(rhs)?;

    // Stage 2: E k2 = f(y + a21 k1) + (c21/h) k1 + h d2 ∂f/∂t.
    let mut y2 = *y;
    for i in 0..N {
        y2[i] += A21 * k1[i];
    }
    let f2 = force.eval::<f64>(t + C2 * h, &y2);
    for i in 0..N {
        rhs[i] = f2[i] + inv_h * C21 * k1[i] + h * D2 * dfdt[i];
    }
    let k2 = solve(rhs)?;

    // Stage 3: E k3 = f(y + a31 k1 + a32 k2) + (c31 k1 + c32 k2)/h + h d3 ∂f/∂t.
    let mut y3 = *y;
    for i in 0..N {
        y3[i] += A31 * k1[i] + A32 * k2[i];
    }
    let f3 = force.eval::<f64>(t + C3 * h, &y3);
    for i in 0..N {
        rhs[i] = f3[i] + inv_h * (C31 * k1[i] + C32 * k2[i]) + h * D3 * dfdt[i];
    }
    let k3 = solve(rhs)?;

    // Stage 4: E k4 = f3 (reused) + (c41 k1 + c42 k2 + c43 k3)/h + h d4 ∂f/∂t.
    for i in 0..N {
        rhs[i] = f3[i] + inv_h * (C41 * k1[i] + C42 * k2[i] + C43 * k3[i]) + h * D4 * dfdt[i];
    }
    let k4 = solve(rhs)?;

    let mut yn = *y;
    for i in 0..N {
        yn[i] += B1 * k1[i] + B2 * k2[i] + B3 * k3[i] + B4 * k4[i];
    }
    Ok(yn)
}

/// Integrate `force` from `t0` to `t1` with the fixed-step Rosenbrock GRK4T method. The step is
/// adjusted so an integer number of steps spans `[t0, t1]`; `h` is the target size.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span/step is not positive;
/// [`IntegrationError::StageSolveFailed`] if the `(I/(γh) − J)` factorisation is singular.
pub fn integrate_rosenbrock4<const N: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    if !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, 1)?;
    let step = (t1 - t0) / n_steps as f64;

    let mut y = *y0;
    let mut t = t0;
    for _ in 0..n_steps {
        y = rosenbrock_step(force, t, &y, step)?;
        t += step;
    }
    Ok(y)
}

/// Integrate as [`integrate_rosenbrock4`], returning a dense cubic-Hermite [`Trajectory`] for the
/// event detector. Each segment is built from the state and its derivative at the step endpoints.
///
/// # Errors
///
/// As [`integrate_rosenbrock4`].
pub fn integrate_rosenbrock4_dense<const N: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<Trajectory<N, HermiteDense<N>>, IntegrationError> {
    if !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, 1)?;
    integrate_fixed_hermite(
        t0,
        t1,
        y0,
        n_steps,
        |t, y| force.eval::<f64>(t, y),
        |t, y, step| rosenbrock_step(force, t, y, step),
    )
}
