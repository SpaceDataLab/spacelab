//! Gauss–Jackson, order 8: the second-sum predictor–corrector for `x'' = a(x)`, the workhorse of
//! high-precision orbit propagation (used by USSPACECOM for the public catalog).
//!
//! # Why a second-order-native method
//!
//! A gravitational problem is a *second-order* ODE — the force depends on position, not velocity.
//! Rewriting it as a first-order system and handing it to Adams or a Runge–Kutta throws that
//! structure away. Gauss–Jackson integrates the second-order equation directly by summing
//! accelerations twice, which buys two things: the same order-8 accuracy at **one** force
//! evaluation per step (PECE costs two), and a dramatically lower round-off growth, because the
//! large, slowly varying part of the trajectory is carried in running sums rather than recomputed
//! as the near-cancelling combination `2 xₙ − xₙ₋₁` every step.
//!
//! The position advances with the Störmer (explicit) / Cowell (implicit) coefficients; the velocity
//! with the Adams (first-sum) companion — the natural pairing, since the first sum of accelerations
//! *is* the velocity and the second *is* the position. The implementation keeps the running first
//! difference `Δ = xₙ − xₙ₋₁` and updates `Δ ← Δ + h²·(corrector·a)`, `x ← x + Δ`, both
//! Kahan-compensated: this is the summed form that gives the method its low round-off floor.
//!
//! Gate: `derivations/gauss_jackson8_generating.mac` derives the Störmer, Cowell and Adams–Moulton
//! weights from their generating functions in exact rationals, shows them exact on `t^m` through
//! degree nine (eight for the velocity) and not beyond, and checks every shipped literal is the
//! exact weight correctly rounded at its last digit; `derivations/gauss_jackson8_generating.gp`
//! solves the moment systems instead and agrees. The module test
//! `the_shipped_weights_are_the_generating_function_rationals` (G3) holds each `f64` weight to
//! the correctly rounded rational, bit for bit.
//!
//! # Constraint
//!
//! Like the [symplectic](crate::symplectic) methods, the acceleration must be a function of
//! **position only**. Gravity (`PointMass`, `Zonal`, `ThirdBody`) qualifies; drag does not — its
//! velocity dependence breaks the second-order form the summation assumes. The system is taken as a
//! first-order `[r, v]` state whose derivative is `[v, a]`; the acceleration is read from the
//! second half, so `N` must be even.

// The Störmer/Cowell weights are full-precision literals; the step counts are small integers.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::vec::Vec;

use crate::abm::AM8;
use crate::driver::{Controls, HermiteDense, IntegrationError, Trajectory, integrate, step_count};
use crate::{Dop853, DynamicalSystem, Tolerance};

const ORDER: usize = 8;

/// Störmer predictor weights (explicit), on `aₙ, aₙ₋₁, …, aₙ₋₇`. Derived as the moment solution
/// in `derivations/gauss_jackson_coeffs.mac` (sum = 1; exact rationals `22081/15120, −4511/2240,
/// 40933/10080, −300227/60480, 9857/2520, −39017/20160, 3319/6048, −275/4032`).
const STORMER: [f64; ORDER] = [
    1.46038359788359788359788359788359788,
    -2.01383928571428571428571428571428571,
    4.06081349206349206349206349206349206,
    -4.9640707671957671957671957671957672,
    3.91150793650793650793650793650793651,
    -1.93536706349206349206349206349206349,
    0.548776455026455026455026455026455026,
    -0.0682043650793650793650793650793650794,
];

/// Cowell corrector weights (implicit), on `aₙ₊₁, aₙ, …, aₙ₋₆`. Same derivation (sum = 1; exact
/// rationals `275/4032, 13831/15120, −2099/20160, 811/3360, −11477/60480, 29/315, −517/20160,
/// 19/6048`).
const COWELL: [f64; ORDER] = [
    0.0682043650793650793650793650793650794,
    0.914748677248677248677248677248677249,
    -0.104117063492063492063492063492063492,
    0.241369047619047619047619047619047619,
    -0.18976521164021164021164021164021164,
    0.0920634920634920634920634920634920635,
    -0.0256448412698412698412698412698412698,
    0.00314153439153439153439153439153439153,
];

/// Compensated (Kahan) addition `*y += inc`, carrying the lost low bits in `*c` — the same running
/// summation that keeps the second-sum accumulators at the round-off floor (see
/// [`crate::symplectic`] for the full argument).
#[inline]
fn kahan_add(y: &mut f64, c: &mut f64, inc: f64) {
    let inc_c = inc + *c;
    let y_new = *y + inc_c;
    *c = (*y - y_new) + inc_c;
    *y = y_new;
}

fn eval<const N: usize>(sys: &impl DynamicalSystem<N>, t: f64, y: &[f64; N]) -> [f64; N] {
    let mut d = [0.0; N];
    sys.derivative(t, y, &mut d);
    d
}

/// Integrate `sys` from `t0` to `t1` with the fixed-step Gauss–Jackson-8 PECE scheme. The step is
/// adjusted so an integer number of steps spans `[t0, t1]` exactly; `h` is the target size.
///
/// The state is `[r, v]` with `dr/dt = v` and a **position-only** acceleration `a = (dv/dt)`; `N`
/// must be even.
///
/// # Errors
///
/// [`IntegrationError`] if `N` is odd, the span/step is not positive, or the `DOP853` startup fails.
pub fn integrate_gj8<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    if !(N.is_multiple_of(2) && t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let d = N / 2;
    let n_steps = step_count(t0, t1, h, ORDER - 1)?;
    let step = (t1 - t0) / n_steps as f64;
    let h2 = step * step;

    // Startup: the eight past node states, each from an independent high-accuracy DOP853 run
    // t0 → node. Taking each run's final state (rather than one run's dense-output interpolant)
    // keeps the startup error at DOP853's global-error floor, ~1e-9 m at this tolerance, instead of
    // the ~1e-7 m dense-output interpolation error — the multistep scheme can be no more accurate
    // than the seed it is handed.
    let mut node_state = [[0.0; N]; ORDER];
    node_state[0] = *y0;
    for node in 1..ORDER {
        node_state[node] = integrate(
            &Dop853,
            sys,
            t0,
            y0,
            t0 + node as f64 * step,
            step,
            Controls::new(Tolerance::new(1e-14, 1e-14), 1e-10),
        )?
        .final_state();
    }

    // Acceleration history: ah[0] = a at the newest node (tₙ), ah[7] = a at tₙ₋₇. Only the first
    // `d` entries of each row are used (the acceleration is the second half of the derivative).
    let mut ah = [[0.0; N]; ORDER];
    for k in 0..ORDER {
        let node = ORDER - 1 - k;
        let tj = t0 + node as f64 * step;
        let dj = eval(sys, tj, &node_state[node]);
        ah[k][..d].copy_from_slice(&dj[d..]);
    }

    let mut t = t0 + f64::from(ORDER as u32 - 1) * step;
    let mut y = node_state[ORDER - 1]; // [rₙ, vₙ] at the newest node
    let mut delta = [0.0; N]; // running first difference Δ = rₙ − rₙ₋₁ (position half only)
    for i in 0..d {
        delta[i] = y[i] - node_state[ORDER - 2][i];
    }
    let mut c_r = [0.0; N]; // Kahan compensation for the position accumulator
    let mut c_delta = [0.0; N]; // … for Δ
    let mut c_v = [0.0; N]; // … for the velocity accumulator

    for _ in (ORDER - 1)..n_steps {
        // Predict position (Störmer): r_pred = rₙ + Δ + h²·Σ STORMER[j]·aₙ₋ⱼ.
        let mut y_pred = y;
        for i in 0..d {
            let mut acc = 0.0;
            for j in 0..ORDER {
                acc += STORMER[j] * ah[j][i];
            }
            y_pred[i] = y[i] + delta[i] + h2 * acc;
        }
        // Evaluate the acceleration at the predicted position (velocity ignored by the force).
        let d_pred = eval(sys, t + step, &y_pred);

        // Correct: Δ ← Δ + h²·(COWELL[0]·a_pred + Σ_{j≥1} COWELL[j]·aₙ₊₁₋ⱼ), then r ← r + Δ.
        // Velocity by the Adams–Moulton first-sum companion: v ← v + h·(AM8·accelerations).
        let mut corr = [0.0; N];
        let mut vinc = [0.0; N];
        for i in 0..d {
            let mut cacc = COWELL[0] * d_pred[d + i];
            let mut vacc = AM8[0] * d_pred[d + i];
            for j in 1..ORDER {
                cacc += COWELL[j] * ah[j - 1][i];
                vacc += AM8[j] * ah[j - 1][i];
            }
            corr[i] = h2 * cacc;
            vinc[i] = step * vacc;
        }
        for i in 0..d {
            kahan_add(&mut delta[i], &mut c_delta[i], corr[i]);
            kahan_add(&mut y[i], &mut c_r[i], delta[i]);
            kahan_add(&mut y[d + i], &mut c_v[i], vinc[i]);
        }

        // Evaluate at the corrected state and shift the history (PECE).
        t += step;
        let d_new = eval(sys, t, &y);
        for j in (1..ORDER).rev() {
            ah[j] = ah[j - 1];
        }
        ah[0][..d].copy_from_slice(&d_new[d..]);
    }

    Ok(y)
}

/// Integrate as [`integrate_gj8`], returning a dense cubic-Hermite [`Trajectory`] for the event
/// detector. A segment is built between each pair of adjacent nodes (startup and corrector alike)
/// from the `[r, v]` state and its derivative there.
///
/// # Errors
///
/// As [`integrate_gj8`].
pub fn integrate_gj8_dense<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<Trajectory<N, HermiteDense<N>>, IntegrationError> {
    if !(N.is_multiple_of(2) && t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let d = N / 2;
    let n_steps = step_count(t0, t1, h, ORDER - 1)?;
    let step = (t1 - t0) / n_steps as f64;
    let h2 = step * step;

    let mut node_state = [[0.0; N]; ORDER];
    node_state[0] = *y0;
    for node in 1..ORDER {
        node_state[node] = integrate(
            &Dop853,
            sys,
            t0,
            y0,
            t0 + node as f64 * step,
            step,
            Controls::new(Tolerance::new(1e-14, 1e-14), 1e-10),
        )?
        .final_state();
    }

    // Collect (state, derivative) at every node; the startup nodes come from node_state.
    let mut ynodes = Vec::with_capacity(n_steps + 1);
    let mut fnodes = Vec::with_capacity(n_steps + 1);
    let mut ah = [[0.0; N]; ORDER];
    for node in 0..ORDER {
        let dj = eval(sys, t0 + node as f64 * step, &node_state[node]);
        ah[ORDER - 1 - node][..d].copy_from_slice(&dj[d..]);
        ynodes.push(node_state[node]);
        fnodes.push(dj);
    }

    let mut t = t0 + f64::from(ORDER as u32 - 1) * step;
    let mut y = node_state[ORDER - 1];
    let mut delta = [0.0; N];
    for i in 0..d {
        delta[i] = y[i] - node_state[ORDER - 2][i];
    }
    let mut c_r = [0.0; N];
    let mut c_delta = [0.0; N];
    let mut c_v = [0.0; N];

    for _ in (ORDER - 1)..n_steps {
        let mut y_pred = y;
        for i in 0..d {
            let mut acc = 0.0;
            for j in 0..ORDER {
                acc += STORMER[j] * ah[j][i];
            }
            y_pred[i] = y[i] + delta[i] + h2 * acc;
        }
        let d_pred = eval(sys, t + step, &y_pred);

        let mut corr = [0.0; N];
        let mut vinc = [0.0; N];
        for i in 0..d {
            let mut cacc = COWELL[0] * d_pred[d + i];
            let mut vacc = AM8[0] * d_pred[d + i];
            for j in 1..ORDER {
                cacc += COWELL[j] * ah[j - 1][i];
                vacc += AM8[j] * ah[j - 1][i];
            }
            corr[i] = h2 * cacc;
            vinc[i] = step * vacc;
        }
        for i in 0..d {
            kahan_add(&mut delta[i], &mut c_delta[i], corr[i]);
            kahan_add(&mut y[i], &mut c_r[i], delta[i]);
            kahan_add(&mut y[d + i], &mut c_v[i], vinc[i]);
        }

        t += step;
        let d_new = eval(sys, t, &y);
        ynodes.push(y);
        fnodes.push(d_new);
        for j in (1..ORDER).rev() {
            ah[j] = ah[j - 1];
        }
        ah[0][..d].copy_from_slice(&d_new[d..]);
    }

    let mut segments = Vec::with_capacity(n_steps);
    let mut ends = Vec::with_capacity(n_steps);
    for k in 0..n_steps {
        let ta = t0 + k as f64 * step;
        let tb = t0 + (k + 1) as f64 * step;
        segments.push(HermiteDense::new(
            ta,
            tb,
            ynodes[k],
            fnodes[k],
            ynodes[k + 1],
            fnodes[k + 1],
        ));
        ends.push(tb);
    }
    Ok(Trajectory::from_segments(
        t0,
        t1,
        y,
        segments,
        ends,
        n_steps as u32,
        0,
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use core::cell::Cell;

    use super::*;

    /// The test span. It starts away from the origin, where `t1 − t0` and `t1 + t0` differ.
    const T0: f64 = -0.7;
    const T1: f64 = 0.8;

    /// Coefficients of the manufactured position, ascending powers, one row per component.
    const POLY: [[f64; ORDER + 1]; 2] = [
        [-1.0, 4.0, 0.0, -1.0, 0.0, 2.0, 0.0, -3.0, 1.0],
        [7.0, 2.0, -5.0, 0.0, 3.0, 0.0, 1.0, 0.0, -2.0],
    ];

    /// Strength of the feedback term. Small enough that `λ·step²` is deep inside the stability
    /// region, large enough that a corrupted predictor grows rather than being damped away.
    const FEEDBACK: f64 = 4.0;

    /// `r'' = P''(t) + λ·(r − P(t))`, whose exact solution through `(t, P(t), P'(t))` is the degree-8
    /// polynomial `P`. Störmer, Cowell and the Adams–Moulton velocity companion all have zero
    /// residual on `t^m` up to `m = 9` (`derivations/gauss_jackson_coeffs.mac` for the first two, the
    /// same moment check for the third), so the comparison is against round-off and not against a
    /// truncation tolerance. The `λ` term keeps the acceleration dependent on the position — a
    /// predictor that drifts is fed back through `d_pred` instead of being discarded — and the
    /// explicit `t` makes the evaluation *times* enter the answer at all.
    struct Manufactured;

    impl Manufactured {
        fn position(t: f64) -> [f64; 2] {
            let mut p = [0.0; 2];
            for i in 0..2 {
                for m in (0..=ORDER).rev() {
                    p[i] = p[i] * t + POLY[i][m];
                }
            }
            p
        }

        fn velocity(t: f64) -> [f64; 2] {
            let mut v = [0.0; 2];
            for i in 0..2 {
                for m in (1..=ORDER).rev() {
                    v[i] = v[i] * t + m as f64 * POLY[i][m];
                }
            }
            v
        }

        fn acceleration(t: f64) -> [f64; 2] {
            let mut a = [0.0; 2];
            for i in 0..2 {
                for m in (2..=ORDER).rev() {
                    a[i] = a[i] * t + (m * (m - 1)) as f64 * POLY[i][m];
                }
            }
            a
        }

        fn state(t: f64) -> [f64; 4] {
            let p = Self::position(t);
            let v = Self::velocity(t);
            [p[0], p[1], v[0], v[1]]
        }
    }

    impl DynamicalSystem<4> for Manufactured {
        fn derivative(&self, t: f64, y: &[f64; 4], dydt: &mut [f64; 4]) {
            let p = Self::position(t);
            let a = Self::acceleration(t);
            for i in 0..2 {
                dydt[i] = y[2 + i];
                dydt[2 + i] = a[i] + FEEDBACK * (y[i] - p[i]);
            }
        }
    }

    /// `r'' = −r`, exact solution `cos(t − t0)` from `(1, 0)`. Not a polynomial, so the grid the
    /// method chose is visible in the answer — which a polynomial trajectory hides, being reproduced
    /// exactly at any step count.
    struct Oscillator;

    impl DynamicalSystem<2> for Oscillator {
        fn derivative(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
            dydt[0] = y[1];
            dydt[1] = -y[0];
        }
    }

    /// An odd `N`, which cannot be split into a position and a velocity half.
    struct Odd;

    impl DynamicalSystem<3> for Odd {
        fn derivative(&self, _t: f64, _y: &[f64; 3], dydt: &mut [f64; 3]) {
            *dydt = [0.0; 3];
        }
    }

    /// Counts the derivative evaluations made strictly after `after`. The returned state cannot show
    /// *where* the method sampled the dynamics, and for a force model with a validity window that is
    /// the property that matters.
    struct CountAfter<S> {
        inner: S,
        after: f64,
        hits: Cell<usize>,
    }

    impl<S> CountAfter<S> {
        fn new(inner: S, after: f64) -> Self {
            Self {
                inner,
                after,
                hits: Cell::new(0),
            }
        }
    }

    impl<const N: usize, S: DynamicalSystem<N>> DynamicalSystem<N> for CountAfter<S> {
        fn derivative(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]) {
            if t > self.after {
                self.hits.set(self.hits.get() + 1);
            }
            self.inner.derivative(t, y, dydt);
        }
    }

    fn max_abs_diff<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
        (0..N).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
    }

    /// The ordinate weights as exact rationals `(numerator, denominator)`, printed by
    /// `derivations/gauss_jackson8_generating.mac` from the generating functions
    /// `t²/((1 − t) log²(1 − t))`, `t²/log²(1 − t)` and `−t/log(1 − t)`.
    const STORMER_EXACT: [(i32, i32); ORDER] = [
        (22081, 15120),
        (-4511, 2240),
        (40933, 10080),
        (-300_227, 60480),
        (9857, 2520),
        (-39017, 20160),
        (3319, 6048),
        (-275, 4032),
    ];
    const COWELL_EXACT: [(i32, i32); ORDER] = [
        (275, 4032),
        (13831, 15120),
        (-2099, 20160),
        (811, 3360),
        (-11477, 60480),
        (29, 315),
        (-517, 20160),
        (19, 6048),
    ];
    const AM8_EXACT: [(i32, i32); ORDER] = [
        (5257, 17280),
        (139_849, 120_960),
        (-4511, 4480),
        (123_133, 120_960),
        (-88547, 120_960),
        (1537, 4480),
        (-11351, 120_960),
        (275, 24192),
    ];

    /// (G3) Every Störmer, Cowell and Adams–Moulton weight the scheme multiplies by is the exact
    /// rational of its generating function, rounded once to `f64`.
    ///
    /// `derivations/gauss_jackson8_generating.mac` derives the weights from the generating
    /// functions, shows they are exact on `t^m` through degree nine (eight for the velocity) and on
    /// nothing further, and that each 34–37-digit literal in this file and in `abm.rs` is the exact
    /// value correctly rounded at its last digit; `derivations/gauss_jackson8_generating.gp`
    /// solves the moment systems instead and lands on the same rationals. What is left for `f64` is
    /// the compiler's rounding of the literal, which must agree with the rounding of the rational:
    /// numerator and denominator are exact doubles and IEEE division rounds correctly, so
    /// `num / den` is the nearest double to the weight, bit for bit.
    #[test]
    fn the_shipped_weights_are_the_generating_function_rationals() {
        for (name, shipped, exact) in [
            ("Störmer", STORMER, STORMER_EXACT),
            ("Cowell", COWELL, COWELL_EXACT),
            ("Adams–Moulton", AM8, AM8_EXACT),
        ] {
            for (j, (&w, (num, den))) in shipped.iter().zip(exact).enumerate() {
                let want = f64::from(num) / f64::from(den);
                assert_eq!(
                    w.to_bits(),
                    want.to_bits(),
                    "{name} w{j}: {w:e} vs {num}/{den}"
                );
            }
        }
    }

    #[test]
    fn compensated_addition_keeps_the_lost_low_bits() {
        // Four additions of 2⁻⁵³ to 1.0. Each is exactly half an ulp, so round-to-nearest-even
        // discards every one of them and the naive sum never leaves 1.0. The compensated sum has to
        // reach the exact total 1 + 4·2⁻⁵³ = 1 + 2⁻⁵¹, which is representable — hand-derived from
        // IEEE-754, not read off a run. A compensation carried with the wrong sign cancels itself on
        // alternate calls and lands back on the naive answer.
        let inc = f64::EPSILON / 2.0;
        let mut y = 1.0;
        let mut c = 0.0;
        let mut naive = 1.0;
        for _ in 0..4 {
            kahan_add(&mut y, &mut c, inc);
            naive += inc;
        }
        assert_eq!(
            naive.to_bits(),
            1.0_f64.to_bits(),
            "the increment is visible"
        );
        assert_eq!(
            y.to_bits(),
            (1.0 + f64::EPSILON * 2.0).to_bits(),
            "compensation lost: {y:.17e}"
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        // The odd `N` and the negative-zero step come first on purpose: both are refused by the guard
        // alone, and a guard weakened enough to let them through runs on to a state — a saturating
        // step count, an unsplittable state vector — that the later cases must not be asked to reach.
        assert_eq!(
            integrate_gj8(&Odd, T0, &[0.0; 3], T1, 0.05),
            Err(IntegrationError::BadArguments),
            "accepted an odd N"
        );
        assert_eq!(
            integrate_gj8_dense(&Odd, T0, &[0.0; 3], T1, 0.05).err(),
            Some(IntegrationError::BadArguments),
            "dense accepted an odd N"
        );

        let y0 = Manufactured::state(T0);
        for (t0, t1, h) in [
            (T0, T1, -0.0),  // negative zero: `h >= 0.0` admits it and derives a positive step
            (T0, T0, 0.05),  // empty span
            (T1, T0, 0.05),  // backwards: the fixed-step form integrates forwards only
            (T0, T1, 0.0),   // zero step
            (T0, T1, -0.05), // backwards step over a forwards span
        ] {
            assert_eq!(
                integrate_gj8(&Manufactured, t0, &y0, t1, h),
                Err(IntegrationError::BadArguments),
                "accepted t0={t0} t1={t1} h={h}"
            );
            assert_eq!(
                integrate_gj8_dense(&Manufactured, t0, &y0, t1, h).err(),
                Some(IntegrationError::BadArguments),
                "dense accepted t0={t0} t1={t1} h={h}"
            );
        }
    }

    #[test]
    fn reproduces_a_degree_eight_trajectory_exactly() {
        let y0 = Manufactured::state(T0);
        let want = Manufactured::state(T1);
        let h = (T1 - T0) / 40.0;

        let y = integrate_gj8(&Manufactured, T0, &y0, T1, h).unwrap();
        let err = max_abs_diff(&y, &want);
        println!("GJ8 on a degree-8 trajectory: |Δ[r, v]| = {err:.3e}");
        assert!(err < 1e-10, "not exact on a degree-8 trajectory: {err:.3e}");

        let traj = integrate_gj8_dense(&Manufactured, T0, &y0, T1, h).unwrap();
        let derr = max_abs_diff(&traj.final_state(), &want);
        println!("GJ8 dense on a degree-8 trajectory: |Δ[r, v]| = {derr:.3e}");
        assert!(derr < 1e-10, "dense form not exact: {derr:.3e}");
    }

    #[test]
    fn walks_the_node_schedule_the_caller_asked_for() {
        let h = 0.07; // 1.5 / 0.07 = 21.43: the span is not a whole number of steps
        let n_steps = ((T1 - T0) / h).round() as usize;
        let step = (T1 - T0) / n_steps as f64;
        // Between the last startup node and the first corrector node: the startup owns everything at
        // or below it, and the PECE loop owns everything above.
        let after = T0 + (ORDER as f64 - 0.5) * step;
        let y0 = Manufactured::state(T0);
        let want = 2 * (n_steps - (ORDER - 1));

        let probe = CountAfter::new(Manufactured, after);
        integrate_gj8(&probe, T0, &y0, T1, h).unwrap();
        assert_eq!(probe.hits.get(), want, "PECE evaluations past the startup");

        let probe = CountAfter::new(Manufactured, after);
        let traj = integrate_gj8_dense(&probe, T0, &y0, T1, h).unwrap();
        assert_eq!(
            probe.hits.get(),
            want,
            "dense PECE evaluations past the startup"
        );
        assert_eq!(traj.step_counts().0 as usize, n_steps);
        assert_eq!(traj.step_ends().len(), n_steps);
        let last = *traj.step_ends().last().unwrap();
        assert!(
            (last - T1).abs() < 1e-12,
            "last node is not t1: {last:.17e}"
        );
    }

    #[test]
    fn a_span_below_the_startup_length_is_startup_only() {
        // Two steps asked for; seven startup intervals are the floor, so the startup covers the whole
        // span and no Gauss–Jackson step is taken. Three radians of a unit oscillator per step is far
        // outside the method's stability region, so any other step count is loud.
        let t0 = 2.0_f64;
        let t1 = 23.0_f64;
        let h = (t1 - t0) / 2.0;
        let want = [(t1 - t0).cos(), -(t1 - t0).sin()];

        let y = integrate_gj8(&Oscillator, t0, &[1.0, 0.0], t1, h).unwrap();
        let err = max_abs_diff(&y, &want);
        println!("GJ8 over a sub-startup span: |Δ[r, v]| = {err:.3e}");
        assert!(err < 1e-9, "not integrated on the startup grid: {err:.3e}");

        let traj = integrate_gj8_dense(&Oscillator, t0, &[1.0, 0.0], t1, h).unwrap();
        let derr = max_abs_diff(&traj.final_state(), &want);
        println!("GJ8 dense over a sub-startup span: |Δ[r, v]| = {derr:.3e}");
        assert!(
            derr < 1e-9,
            "dense not integrated on the startup grid: {derr:.3e}"
        );
        assert_eq!(traj.step_counts().0 as usize, ORDER - 1);
    }

    #[test]
    fn dense_interpolates_between_nodes() {
        let n_steps: usize = 120;
        let step = (T1 - T0) / n_steps as f64;
        let y0 = Manufactured::state(T0);
        let traj = integrate_gj8_dense(&Manufactured, T0, &y0, T1, step).unwrap();

        // Mid-segment. On a node every cubic Hermite returns the node state whatever abscissa or end
        // derivative it was built from, so a node query tests nothing about the segment's geometry.
        let t = T0 + 72.5 * step;
        let err = max_abs_diff(&traj.eval(t), &Manufactured::state(t));
        println!("GJ8 dense mid-segment error: {err:.3e}");
        assert!(
            err < 1e-6,
            "Hermite segment is not on the right abscissa: {err:.3e}"
        );
    }
}
