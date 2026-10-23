//! Dormand–Prince 8(5,3), the `DOP853` method, with Hairer's 7th-order dense output.
//!
//! # Provenance and verification
//!
//! The tableau is machine-generated from scipy's `dop853_coefficients.py` (BSD-3-Clause), which are
//! the published Hairer–Nørsett–Wanner coefficients — see `src/dop853_consts.rs` and
//! `tools/gen_dop853.py`. Two hundred constants typed by hand is the surest way to ship an order-8
//! method that is secretly lower; generating them removes that risk, and the order conditions are
//! verified to machine precision (`derivations/dop853_order.mac`) while the measured convergence
//! order (`tests/dop853.rs`) reads the *actual* order off the numbers. None of it is from
//! *Numerical Recipes*.
//!
//! The step uses 12 stages; the FSAL 13th (`f(t+h, y_next)`) feeds the error estimate; three further
//! stages build the interpolant. The error norm is `DOP853`'s own combination of a 5th- and a
//! 3rd-order estimate (Hairer II.5.2), not a plain RMS — replicated from scipy.

#![allow(clippy::needless_range_loop)] // the stage loops index the 16×16 tableau; iterators would obscure them
#![allow(clippy::cast_precision_loss)] // N is a small state dimension, exactly representable
#![allow(clippy::many_single_char_names)] // i, j, s, f are the algorithm's own names

use crate::dop853_consts::{A, B, C, D, E3, E5};
use crate::driver::{Method, MethodStep};
use crate::{DenseOutput, DynamicalSystem, Tolerance};

const N_STAGES: usize = 12;
const N_EXTENDED: usize = 16;
const INTERP: usize = 7;

/// The 7th-order interpolant for one `DOP853` step, valid on `[t0, t0 + h]`.
#[derive(Debug, Clone, Copy)]
pub struct Dop853Dense<const N: usize> {
    t0: f64,
    h: f64,
    y_old: [f64; N],
    f: [[f64; N]; INTERP],
}

impl<const N: usize> DenseOutput<N> for Dop853Dense<N> {
    fn eval(&self, t: f64, out: &mut [f64; N]) {
        let x = (t - self.t0) / self.h;
        // Hairer/scipy Horner: fold F[6..0], multiplying by x on even indices and (1−x) on odd.
        let mut acc = [0.0; N];
        for (i, f) in self.f.iter().rev().enumerate() {
            let m = if i % 2 == 0 { x } else { 1.0 - x };
            for j in 0..N {
                acc[j] = (acc[j] + f[j]) * m;
            }
        }
        for j in 0..N {
            out[j] = acc[j] + self.y_old[j];
        }
    }
}

/// The Dormand–Prince 8(5,3) stepper. Stateless; one instance serves any dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dop853;

impl Dop853 {
    /// Stages that form the order-8 solution itself.
    pub const STAGES: usize = N_STAGES;
    /// The FSAL stage `f(t+h, y_next)`, which the error estimate needs and the step does not.
    pub const ERROR_STAGES: usize = 1;
    /// Stages evaluated only to build [`Dop853Dense`]. Paid on every attempted step, accepted or
    /// rejected, whether or not the caller asks for the interpolant.
    pub const INTERPOLANT_STAGES: usize = N_EXTENDED - N_STAGES - Self::ERROR_STAGES;
    /// Every derivative evaluation one attempted step costs.
    pub const EXTENDED_STAGES: usize = N_EXTENDED;
}

impl<const N: usize> Method<N> for Dop853 {
    type Dense = Dop853Dense<N>;

    fn step_exponent(&self) -> f64 {
        0.125 // 1/(error-estimator order 7 + 1)
    }

    fn try_step(
        &self,
        sys: &impl DynamicalSystem<N>,
        t: f64,
        y: &[f64; N],
        h: f64,
        tol: Tolerance,
    ) -> MethodStep<N, Dop853Dense<N>> {
        // All 16 stages: 12 for the step, the FSAL 13th for the error, three for the interpolant.
        let mut s = [[0.0; N]; N_EXTENDED];
        let mut arg = [0.0; N];

        sys.derivative(t, y, &mut s[0]);

        // All stages 1..=15 via the tableau, in order, so each has its predecessors. Stage 12's `a`
        // row equals B, so `s[12]` evaluates at `y_next = y + h·Σ B·s` automatically (the FSAL
        // stage); stages 13..15 build the interpolant.
        for stage in 1..N_EXTENDED {
            for i in 0..N {
                let mut acc = 0.0;
                for j in 0..stage {
                    acc += A[stage][j] * s[j][i];
                }
                arg[i] = y[i] + h * acc;
            }
            sys.derivative(t + C[stage] * h, &arg, &mut s[stage]);
        }

        // Form y_next only now that every main stage exists. Accumulate the increment among the
        // like-magnitude stage terms first, then add it to the large state once — as DOPRI5 does —
        // so the low bits a term-by-term sum onto `y ≈ 7e6` would drop are kept, and the dense-output
        // `delta = y_next − y` below is `h·Σ B·s` exactly.
        let mut y_next = *y;
        for i in 0..N {
            let mut dy = 0.0;
            for j in 0..N_STAGES {
                dy += B[j] * s[j][i];
            }
            y_next[i] = y[i] + h * dy;
        }

        // DOP853 error norm: 5th- and 3rd-order estimates combined (Hairer II.5.2).
        let mut err5_2 = 0.0;
        let mut err3_2 = 0.0;
        for i in 0..N {
            let mut e5 = 0.0;
            let mut e3 = 0.0;
            for j in 0..=N_STAGES {
                e5 += E5[j] * s[j][i];
                e3 += E3[j] * s[j][i];
            }
            let sc = tol.scale(y[i], y_next[i]);
            let r5 = e5 / sc;
            let r3 = e3 / sc;
            err5_2 += r5 * r5;
            err3_2 += r3 * r3;
        }
        let error_norm = if err5_2 == 0.0 && err3_2 == 0.0 {
            0.0
        } else {
            let denom = err5_2 + 0.01 * err3_2;
            libm::fabs(h) * err5_2 / libm::sqrt(denom * N as f64)
        };

        // Dense output F-coefficients (scipy Dop853DenseOutput).
        let mut f = [[0.0; N]; INTERP];
        for i in 0..N {
            let delta = y_next[i] - y[i];
            f[0][i] = delta;
            f[1][i] = h * s[0][i] - delta;
            f[2][i] = 2.0 * delta - h * (s[N_STAGES][i] + s[0][i]);
        }
        for row in 0..(INTERP - 3) {
            for i in 0..N {
                let mut acc = 0.0;
                for j in 0..N_EXTENDED {
                    acc += D[row][j] * s[j][i];
                }
                f[3 + row][i] = h * acc;
            }
        }

        MethodStep {
            y_next,
            dense: Dop853Dense {
                t0: t,
                h,
                y_old: *y,
                f,
            },
            error_norm,
        }
    }
}

impl Dop853 {
    /// One fixed-step advance, ignoring the error estimate — used to read the convergence order.
    #[must_use]
    pub fn step_fixed<const N: usize>(
        self,
        sys: &impl DynamicalSystem<N>,
        t: f64,
        y: &[f64; N],
        h: f64,
    ) -> [f64; N] {
        let mut s = [[0.0; N]; N_STAGES];
        let mut arg = [0.0; N];
        sys.derivative(t, y, &mut s[0]);
        for stage in 1..N_STAGES {
            for i in 0..N {
                let mut acc = 0.0;
                for j in 0..stage {
                    acc += A[stage][j] * s[j][i];
                }
                arg[i] = y[i] + h * acc;
            }
            sys.derivative(t + C[stage] * h, &arg, &mut s[stage]);
        }
        let mut y_next = *y;
        for i in 0..N {
            let mut dy = 0.0;
            for j in 0..N_STAGES {
                dy += B[j] * s[j][i];
            }
            y_next[i] = y[i] + h * dy;
        }
        y_next
    }
}
