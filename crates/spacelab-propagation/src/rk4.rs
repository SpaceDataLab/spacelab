//! Classical fourth-order Runge–Kutta, fixed step — the baseline every other method is checked
//! against.
//!
//! RK4 is the one integrator whose coefficients are beyond dispute (`b = [1/6, 1/3, 1/3, 1/6]`,
//! `c = [0, 1/2, 1/2, 1]`), so it needs no tableau verification. It has no error estimate and no
//! dense output; its job is to be obviously correct and to serve as the reference in the
//! convergence-order and differential tests of the adaptive methods.

use crate::DynamicalSystem;

/// One fixed step of classical RK4, returning the state at `t + h`.
#[must_use]
pub fn rk4_step<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t: f64,
    y: &[f64; N],
    h: f64,
) -> [f64; N] {
    let mut k1 = [0.0; N];
    let mut k2 = [0.0; N];
    let mut k3 = [0.0; N];
    let mut k4 = [0.0; N];
    let mut tmp = [0.0; N];

    sys.derivative(t, y, &mut k1);

    for i in 0..N {
        tmp[i] = y[i] + 0.5 * h * k1[i];
    }
    sys.derivative(t + 0.5 * h, &tmp, &mut k2);

    for i in 0..N {
        tmp[i] = y[i] + 0.5 * h * k2[i];
    }
    sys.derivative(t + 0.5 * h, &tmp, &mut k3);

    for i in 0..N {
        tmp[i] = y[i] + h * k3[i];
    }
    sys.derivative(t + h, &tmp, &mut k4);

    let mut out = *y;
    for i in 0..N {
        out[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

/// Integrate from `t0` to `t1` in `steps` equal RK4 steps, returning the final state.
///
/// Fixed-step, so the caller owns the accuracy/step trade-off. Used as the reference in tests and
/// as the simplest possible propagator.
#[must_use]
pub fn rk4_integrate<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    steps: u32,
) -> [f64; N] {
    let h = (t1 - t0) / f64::from(steps);
    let mut t = t0;
    let mut y = *y0;
    for _ in 0..steps {
        y = rk4_step(sys, t, &y, h);
        t += h;
    }
    y
}
