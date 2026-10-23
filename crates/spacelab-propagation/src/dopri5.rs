//! Dormand–Prince 5(4), the `DOPRI5` embedded pair, with Hairer's 4th-order dense output.
//!
//! # Provenance and verification
//!
//! The tableau is from Dormand & Prince (1980). Every order condition through order 5 is checked
//! symbolically in `derivations/dopri5_order.mac` (all zero; the embedded pair is order 4). The
//! error coefficients `e = b − b̂` are recomputed there. The dense-output `d`-coefficients are
//! Hairer's `contd5` (Hairer–Nørsett–Wanner, *Solving ODEs I*); the interpolant is verified to
//! reproduce the endpoints exactly and the analytic solution mid-step to the method's order in
//! `tests/dopri5.rs`. None of it comes from *Numerical Recipes*.
//!
//! `DOPRI5` is FSAL — the last stage `k7 = f(t+h, y_next)` is the first stage of the next step — so
//! a driver can save one derivative evaluation per accepted step. This stepper computes all seven
//! for a clean, stateless interface; the FSAL reuse is a driver-level optimisation.

#![allow(clippy::needless_range_loop)] // the stage loops index up to eight arrays; iterators would obscure them

use crate::{DenseOutput, DynamicalSystem, StepTrial};

// --- nodes c and the strictly-lower-triangular a, from Dormand & Prince (1980) ---
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// --- 5th-order solution weights b (also the a7 row, giving FSAL) ---
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

// --- error weights e = b − b̂, recomputed exactly in derivations/dopri5_order.mac ---
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17_253.0 / 339_200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

// --- dense-output coefficients d, Hairer contd5 ---
const D1: f64 = -12_715_105_075.0 / 11_282_082_432.0;
const D3: f64 = 87_487_479_700.0 / 32_700_410_799.0;
const D4: f64 = -10_690_763_975.0 / 1_880_347_072.0;
const D5: f64 = 701_980_252_875.0 / 199_316_789_632.0;
const D6: f64 = -1_453_857_185.0 / 822_651_844.0;
const D7: f64 = 69_997_945.0 / 29_380_423.0;

/// The 4th-order interpolant for one `DOPRI5` step, valid on `[t0, t0 + h]`.
#[derive(Debug, Clone, Copy)]
pub struct Dopri5Dense<const N: usize> {
    t0: f64,
    h: f64,
    r1: [f64; N],
    r2: [f64; N],
    r3: [f64; N],
    r4: [f64; N],
    r5: [f64; N],
}

impl<const N: usize> DenseOutput<N> for Dopri5Dense<N> {
    fn eval(&self, t: f64, out: &mut [f64; N]) {
        let theta = (t - self.t0) / self.h;
        let theta1 = 1.0 - theta;
        for i in 0..N {
            // Horner in θ / (1−θ): r1 + θ(r2 + (1−θ)(r3 + θ(r4 + (1−θ)r5)))
            out[i] = self.r1[i]
                + theta
                    * (self.r2[i]
                        + theta1 * (self.r3[i] + theta * (self.r4[i] + theta1 * self.r5[i])));
        }
    }
}

/// The Dormand–Prince 5(4) stepper. Stateless; one instance serves any dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dopri5;

impl Dopri5 {
    /// Take one trial step of size `h` from `(t, y)`.
    ///
    /// Returns the 5th-order state, the componentwise error estimate (order 4), and the dense
    /// interpolant. The caller's step controller decides acceptance from the error norm.
    #[must_use]
    pub fn try_step<const N: usize>(
        self,
        sys: &impl DynamicalSystem<N>,
        t: f64,
        y: &[f64; N],
        h: f64,
    ) -> StepTrial<N, Dopri5Dense<N>> {
        let mut k1 = [0.0; N];
        let mut k2 = [0.0; N];
        let mut k3 = [0.0; N];
        let mut k4 = [0.0; N];
        let mut k5 = [0.0; N];
        let mut k6 = [0.0; N];
        let mut k7 = [0.0; N];
        let mut tmp = [0.0; N];

        sys.derivative(t, y, &mut k1);

        for i in 0..N {
            tmp[i] = y[i] + h * A21 * k1[i];
        }
        sys.derivative(t + C2 * h, &tmp, &mut k2);

        for i in 0..N {
            tmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        sys.derivative(t + C3 * h, &tmp, &mut k3);

        for i in 0..N {
            tmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        sys.derivative(t + C4 * h, &tmp, &mut k4);

        for i in 0..N {
            tmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        sys.derivative(t + C5 * h, &tmp, &mut k5);

        for i in 0..N {
            tmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        sys.derivative(t + h, &tmp, &mut k6);

        // 5th-order solution (the a7 row is b, so y_next is the stage argument for k7 — FSAL).
        let mut y_next = *y;
        for i in 0..N {
            y_next[i] += h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
        }
        sys.derivative(t + h, &y_next, &mut k7);

        // Error estimate e·k (order 4), and the dense interpolant.
        let mut y_err = [0.0; N];
        let (mut r1, mut r2, mut r3, mut r4, mut r5) =
            ([0.0; N], [0.0; N], [0.0; N], [0.0; N], [0.0; N]);
        for i in 0..N {
            y_err[i] =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);

            let ydiff = y_next[i] - y[i];
            let bspl = h * k1[i] - ydiff;
            r1[i] = y[i];
            r2[i] = ydiff;
            r3[i] = bspl;
            r4[i] = ydiff - h * k7[i] - bspl;
            r5[i] =
                h * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i] + D7 * k7[i]);
        }

        StepTrial {
            y_next,
            y_err,
            dense: Dopri5Dense {
                t0: t,
                h,
                r1,
                r2,
                r3,
                r4,
                r5,
            },
        }
    }
}

impl<const N: usize> crate::driver::Method<N> for Dopri5 {
    type Dense = Dopri5Dense<N>;

    fn step_exponent(&self) -> f64 {
        0.2 // 1/(error-estimator order 4 + 1)
    }

    fn beta(&self) -> f64 {
        0.04 // Hairer's dopri5.f PI-controller memory
    }

    fn try_step(
        &self,
        sys: &impl DynamicalSystem<N>,
        t: f64,
        y: &[f64; N],
        h: f64,
        tol: crate::Tolerance,
    ) -> crate::driver::MethodStep<N, Dopri5Dense<N>> {
        let trial = Dopri5::try_step(*self, sys, t, y, h);
        let error_norm = tol.error_norm(y, &trial.y_next, &trial.y_err);
        crate::driver::MethodStep {
            y_next: trial.y_next,
            dense: trial.dense,
            error_norm,
        }
    }
}
