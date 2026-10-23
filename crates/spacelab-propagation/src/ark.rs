//! ARK4(3)6L`2`SA — an additive (implicit–explicit) Runge–Kutta of order 4 (Kennedy & Carpenter 2003).
//!
//! # Splitting the physics
//!
//! A real simulator has terms with wildly different time constants: gravity and drag are non-stiff,
//! but an attitude control loop or a chemistry term can be orders of magnitude faster. Treating the
//! whole right-hand side explicitly forces the fast term to dictate a tiny step; treating it all
//! implicitly pays for an expensive nonlinear solve on terms that never needed one. An **IMEX**
//! method splits `y' = f_E(t,y) + f_I(t,y)` and integrates the non-stiff `f_E` with an explicit
//! Runge–Kutta and the stiff `f_I` with an implicit one, coupled so the pair is 4th order. Only the
//! stiff part is solved implicitly — one `N×N` Newton solve per stage, not the whole system.
//!
//! The implicit tableau is an ESDIRK (explicit first stage, single diagonal `γ = 1/4`, L-stable and
//! stiffly accurate); the explicit tableau is an ERK; the two share the nodes `c` and weights `b`.
//! The coefficients are the exact rationals of the reference implementation (SUNDIALS/ARKODE), and
//! `derivations/ark_order.mac` checks all eighteen order-4 conditions — the four shared, the four
//! single-method conditions for each tableau, and the two coupling conditions — vanish to ~1e-27.

// The tableau entries are exact rationals written as divisions; the stage count is small.
#![allow(
    clippy::needless_range_loop,
    clippy::unreadable_literal,
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use spacelab_autodiff::{Scalar, jacobian};

use crate::driver::{IntegrationError, step_count};
use crate::linalg::lu_factor;
use crate::variational::ScalarSystem;

const S: usize = 6;
const GAMMA: f64 = 0.25;
const MAX_NEWTON: usize = 50;
const NEWTON_TOL: f64 = 1e-12;

const C: [f64; S] = [0.0, 0.5, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0];
const B: [f64; S] = [
    82889.0 / 524892.0,
    0.0,
    15625.0 / 83664.0,
    69875.0 / 102672.0,
    -2260.0 / 8211.0,
    0.25,
];

/// Explicit ERK Butcher matrix (strictly lower triangular).
const AEX: [[f64; S]; S] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [13861.0 / 62500.0, 6889.0 / 62500.0, 0.0, 0.0, 0.0, 0.0],
    [
        -116923316275.0 / 2393684061468.0,
        -2731218467317.0 / 15368042101831.0,
        9408046702089.0 / 11113171139209.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -451086348788.0 / 2902428689909.0,
        -2682348792572.0 / 7519795681897.0,
        12662868775082.0 / 11960479115383.0,
        3355817975965.0 / 11060851509271.0,
        0.0,
        0.0,
    ],
    [
        647845179188.0 / 3216320057751.0,
        73281519250.0 / 8382639484533.0,
        552539513391.0 / 3454668386233.0,
        3354512671639.0 / 8306763924573.0,
        4040.0 / 17871.0,
        0.0,
    ],
];

/// Implicit ESDIRK Butcher matrix (lower triangular, diagonal `γ`, first stage explicit).
const AIM: [[f64; S]; S] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
    [8611.0 / 62500.0, -1743.0 / 31250.0, 0.25, 0.0, 0.0, 0.0],
    [
        5012029.0 / 34652500.0,
        -654441.0 / 2922500.0,
        174375.0 / 388108.0,
        0.25,
        0.0,
        0.0,
    ],
    [
        15267082809.0 / 155376265600.0,
        -71443401.0 / 120774400.0,
        730878875.0 / 902184768.0,
        2285395.0 / 8070912.0,
        0.25,
        0.0,
    ],
    [
        82889.0 / 524892.0,
        0.0,
        15625.0 / 83664.0,
        69875.0 / 102672.0,
        -2260.0 / 8211.0,
        0.25,
    ],
];

/// Solve the stiff stage equation `Y = x + hγ·f_I(t, Y)` by simplified Newton (frozen AD Jacobian).
fn stage_solve<const N: usize>(
    implicit: &impl ScalarSystem<N>,
    t: f64,
    x: &[f64; N],
    hg: f64,
) -> Result<[f64; N], IntegrationError> {
    let mut y = *x;
    // Freeze the Newton matrix M = I − hγ J at the initial guess (J[r][c] = ∂f_r/∂y_c = jac[c][r]).
    let (_, jac) = jacobian::<N, N, _>(|v| implicit.eval(Scalar::from_f64(t), v), &y);
    let mut m = [[0.0; N]; N];
    for r in 0..N {
        for c in 0..N {
            m[r][c] = -hg * jac[c][r] + if r == c { 1.0 } else { 0.0 };
        }
    }
    let lu = lu_factor(m);

    for _ in 0..MAX_NEWTON {
        let fi = implicit.eval::<f64>(t, &y);
        let mut neg = [0.0; N];
        for k in 0..N {
            neg[k] = -(y[k] - x[k] - hg * fi[k]);
        }
        let dy = lu
            .solve(neg)
            .ok_or(IntegrationError::StageSolveFailed { t })?;
        let mut d2 = 0.0;
        let mut y2 = 0.0;
        for k in 0..N {
            y[k] += dy[k];
            d2 += dy[k] * dy[k];
            y2 += y[k] * y[k];
        }
        if d2.sqrt() <= NEWTON_TOL * (1.0 + y2.sqrt()) {
            return Ok(y);
        }
    }
    Err(IntegrationError::StageSolveFailed { t })
}

/// Integrate `y' = explicit(t,y) + implicit(t,y)` from `t0` to `t1` with ARK4(3)6L`2`SA. The step is
/// adjusted so an integer number spans `[t0, t1]`; `h` is the target size. Only `implicit` is solved
/// implicitly (its Jacobian is taken by AD); `explicit` is evaluated directly.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] for a non-positive span/step; [`IntegrationError::StageSolveFailed`]
/// if a stage's Newton iteration fails.
pub fn integrate_ark<const N: usize>(
    explicit: &impl ScalarSystem<N>,
    implicit: &impl ScalarSystem<N>,
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
        let mut ke = [[0.0; N]; S];
        let mut ki = [[0.0; N]; S];
        // Stage 0 is explicit (both A rows are zero): Y_0 = y.
        ke[0] = explicit.eval::<f64>(t, &y);
        ki[0] = implicit.eval::<f64>(t, &y);

        for i in 1..S {
            let mut x = y;
            for k in 0..N {
                for j in 0..i {
                    x[k] += step * (AEX[i][j] * ke[j][k] + AIM[i][j] * ki[j][k]);
                }
            }
            let ti = t + C[i] * step;
            let yi = stage_solve(implicit, ti, &x, step * GAMMA)?;
            ke[i] = explicit.eval::<f64>(ti, &yi);
            ki[i] = implicit.eval::<f64>(ti, &yi);
        }

        for k in 0..N {
            for i in 0..S {
                y[k] += step * B[i] * (ke[i][k] + ki[i][k]);
            }
        }
        t += step;
    }
    Ok(y)
}
