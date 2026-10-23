//! Radau IIA, 3-stage, order 5: the L-stable implicit collocation Runge–Kutta.
//!
//! # Why an implicit method
//!
//! Every explicit integrator in this crate — DOP853, RKN12(10), the symplectic family — has a
//! bounded stability region: on a *stiff* problem (a fast-decaying mode alongside the slow dynamics,
//! e.g. a tight atmospheric-drag boundary layer, or a control loop with a short time constant) they
//! must take absurdly small steps to stay stable, regardless of the accuracy wanted. Radau IIA is
//! **L-stable**: its stability region contains the entire left half-plane and it damps the
//! infinitely-stiff mode completely, so the step is set by accuracy alone. That is the whole reason
//! to pay for the implicit stage solve.
//!
//! # Construction
//!
//! The method is the collocation RK on the three right-Radau nodes (`c₃ = 1`, which makes it
//! *stiffly accurate*: `bⱼ = a₃ⱼ`, hence L-stable). The full tableau — nodes as the roots of
//! `d²/dx²[x²(x−1)³]`, the `aᵢⱼ`/`bⱼ` as Lagrange-collocation integrals — is derived and verified to
//! order 5 in `derivations/radau5_coeffs.mac` (order conditions `B(1..5)` and the row sums all
//! vanish). The literals below are transcribed from that CAS output, never a table.
//!
//! # Stage solve
//!
//! The three coupled stage equations `Kᵢ = f(t + cᵢh, y + h Σⱼ aᵢⱼ Kⱼ)` are solved by *simplified*
//! Newton: the Jacobian `J = ∂f/∂y` is evaluated once per step by [automatic differentiation]
//! (exact, from the same [`ScalarSystem`] source as the value), frozen across iterations, and the
//! `3N×3N` Newton system `(I − h A⊗J) ΔK = −R` is solved by dense LU with partial pivoting. Because
//! `N` cannot be multiplied by 3 in a const generic on stable Rust, the caller passes `SN = 3N`
//! explicitly (as [`crate::variational`] passes `M = N + N²`); a runtime assertion guards it.
//!
//! [automatic differentiation]: spacelab_autodiff

// The tableau entries are full-precision literals; the step count is a small integer.
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
use crate::linalg::lu_solve;
use crate::variational::ScalarSystem;

/// Number of stages.
const S: usize = 3;
/// Newton iteration cap before the step is declared unconverged.
const MAX_NEWTON: usize = 50;
/// Relative increment threshold for Newton convergence.
const NEWTON_TOL: f64 = 1e-12;

/// Collocation nodes `c` — the right-Radau points `(4∓√6)/10, 1` (see the module derivation).
const C: [f64; S] = [
    0.155051025721682190180271592529410861,
    0.644948974278317809819728407470589139,
    1.0,
];

/// Butcher matrix `aᵢⱼ = ∫₀^{cᵢ} Lⱼ`, row-major.
const A: [[f64; S]; S] = [
    [
        0.19681547722366042586838614299182989,
        -0.0655354258501983881085227825696086918,
        0.023770974348220152420408232107189663,
    ],
    [
        0.394424314739087276997411671458497581,
        0.292073411665228463020502745897058999,
        -0.0415487521259979301981860098849674408,
    ],
    [
        0.376403062700467275050075442369280795,
        0.512485826188421613838813446519608094,
        0.111111111111111111111111111111111111,
    ],
];

/// Quadrature weights `bⱼ = ∫₀¹ Lⱼ`. Equal to `A[2]` (stiff accuracy), stated separately for clarity.
const B: [f64; S] = [
    0.376403062700467275050075442369280795,
    0.512485826188421613838813446519608094,
    0.111111111111111111111111111111111111,
];

/// One Radau IIA-5 step of size `h` from `(t, y)`, solving the stage system by simplified Newton.
fn radau_step<const N: usize, const SN: usize>(
    force: &impl ScalarSystem<N>,
    t: f64,
    y: &[f64; N],
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    // Frozen Jacobian at (t, y): jac[n][m] = ∂fₘ/∂yₙ, so J[r][c] = ∂f_r/∂y_c = jac[c][r].
    let (f0, jac) = jacobian::<N, N, _>(|v| force.eval(Scalar::from_f64(t), v), y);

    // Newton matrix (I − h A⊗J), constant across iterations because J is frozen.
    let mut newton = [[0.0; SN]; SN];
    for bi in 0..S {
        for bj in 0..S {
            for r in 0..N {
                for c in 0..N {
                    let mut v = -h * A[bi][bj] * jac[c][r];
                    if bi == bj && r == c {
                        v += 1.0;
                    }
                    newton[bi * N + r][bj * N + c] = v;
                }
            }
        }
    }

    // Initial stage-derivative guess: f at the current point for every stage.
    let mut kk = [0.0; SN];
    for bi in 0..S {
        kk[bi * N..bi * N + N].copy_from_slice(&f0);
    }

    let mut converged = false;
    for _ in 0..MAX_NEWTON {
        // Residual Rᵢ = Kᵢ − f(t + cᵢh, y + h Σⱼ aᵢⱼ Kⱼ).
        let mut neg_resid = [0.0; SN];
        for bi in 0..S {
            let mut z = *y;
            for bj in 0..S {
                for r in 0..N {
                    z[r] += h * A[bi][bj] * kk[bj * N + r];
                }
            }
            let fi = force.eval::<f64>(t + C[bi] * h, &z);
            for r in 0..N {
                neg_resid[bi * N + r] = fi[r] - kk[bi * N + r];
            }
        }

        let delta =
            lu_solve::<SN>(newton, neg_resid).ok_or(IntegrationError::StageSolveFailed { t })?;

        let mut d2 = 0.0;
        let mut k2 = 0.0;
        for i in 0..SN {
            kk[i] += delta[i];
            d2 += delta[i] * delta[i];
            k2 += kk[i] * kk[i];
        }
        if d2.sqrt() <= NEWTON_TOL * (1.0 + k2.sqrt()) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(IntegrationError::StageSolveFailed { t });
    }

    // y_{n+1} = y + h Σⱼ bⱼ Kⱼ (= the third stage value, by stiff accuracy).
    let mut yn = *y;
    for bj in 0..S {
        for r in 0..N {
            yn[r] += h * B[bj] * kk[bj * N + r];
        }
    }
    Ok(yn)
}

/// Integrate `force` from `t0` to `t1` with the fixed-step Radau IIA-5 method. The step is adjusted
/// so an integer number of steps spans `[t0, t1]`; `h` is the target size. `SN` must equal `3·N`.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if `SN ≠ 3N` or the span/step is not positive;
/// [`IntegrationError::StageSolveFailed`] if a stage solve fails to converge.
pub fn integrate_radau5<const N: usize, const SN: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    if SN != S * N || !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, 1)?;
    let step = (t1 - t0) / n_steps as f64;

    let mut y = *y0;
    let mut t = t0;
    for _ in 0..n_steps {
        y = radau_step::<N, SN>(force, t, &y, step)?;
        t += step;
    }
    Ok(y)
}

/// Integrate as [`integrate_radau5`], returning a dense cubic-Hermite [`Trajectory`] for the event
/// detector. Each segment is built from the state and its derivative at the step endpoints.
///
/// # Errors
///
/// As [`integrate_radau5`].
pub fn integrate_radau5_dense<const N: usize, const SN: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<Trajectory<N, HermiteDense<N>>, IntegrationError> {
    if SN != S * N || !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, 1)?;
    integrate_fixed_hermite(
        t0,
        t1,
        y0,
        n_steps,
        |t, y| force.eval::<f64>(t, y),
        |t, y, step| radau_step::<N, SN>(force, t, y, step),
    )
}
