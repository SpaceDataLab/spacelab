//! Adams–Bashforth–Moulton, order 8, fixed step, in predict–evaluate–correct–evaluate (PECE) form.
//!
//! A multistep method reuses the derivatives it already computed: an order-8 Adams step costs **two**
//! force evaluations, where DOP853 costs twelve, so on a long smooth arc with an expensive force
//! model it is several times cheaper for the same order. The price is a fixed step and a startup
//! phase — it needs eight past points before it can begin, which [`DOP853`](crate::Dop853) provides.
//!
//! The Adams–Bashforth (predictor) and Adams–Moulton (corrector) coefficients are the integrals of
//! the Lagrange interpolant of the stored derivatives, derived from first principles and verified in
//! `derivations/adams_coeffs.mac` (both sum to 1; leading terms 16083/4480 and 5257/17280) — not a
//! table from memory. Variable order/step (the Nordsieck form) is the natural next refinement.

// The Adams weights are transcribed full-precision literals; the step counts are small integers.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::vec::Vec;

use crate::driver::{Controls, HermiteDense, IntegrationError, Trajectory, integrate, step_count};
use crate::{Dop853, DynamicalSystem, Tolerance};

const ORDER: usize = 8;

/// Adams–Bashforth-8 predictor weights, on `f_n, f_{n-1}, …, f_{n-7}` (full-precision literals).
const AB8: [f64; ORDER] = [
    3.58995535714285714285714285714286,
    -9.52520667989417989417989417989418,
    18.0545386904761904761904761904762,
    -22.0277529761904761904761904761905,
    17.3796544312169312169312169312169,
    -8.61212797619047619047619047619048,
    2.44516369047619047619047619047619,
    -0.304224537037037037037037037037037,
];

/// Adams–Moulton-8 corrector weights, on `f_{n+1}, f_n, …, f_{n-6}`. Shared with [`crate::gj`],
/// where the same first-sum (Adams) companion advances the velocity of the Gauss–Jackson scheme.
pub(crate) const AM8: [f64; ORDER] = [
    0.304224537037037037037037037037037,
    1.15615906084656084656084656084656,
    -1.00691964285714285714285714285714,
    1.01796461640211640211640211640212,
    -0.732035383597883597883597883597884,
    0.343080357142857142857142857142857,
    -0.0938409391534391534391534391534392,
    0.0113673941798941798941798941798942,
];

fn eval<const N: usize>(sys: &impl DynamicalSystem<N>, t: f64, y: &[f64; N]) -> [f64; N] {
    let mut d = [0.0; N];
    sys.derivative(t, y, &mut d);
    d
}

/// Integrate `sys` from `t0` to `t1` with the fixed-step ABM8 PECE scheme. The step is adjusted so an
/// integer number of steps spans `[t0, t1]` exactly; `h` is the target size.
///
/// # Errors
///
/// [`IntegrationError`] if the `DOP853` startup fails or the span/step is not positive.
pub fn integrate_abm8<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    if !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, ORDER - 1)?;
    let step = (t1 - t0) / n_steps as f64;

    // Startup: high-accuracy DOP853 over the first seven intervals, sampled at each node.
    let warm = integrate(
        &Dop853,
        sys,
        t0,
        y0,
        t0 + f64::from(ORDER as u32 - 1) * step,
        step,
        Controls::new(Tolerance::new(1e-14, 1e-14), 1e-10),
    )?;

    // Derivative history: fhist[0] = f at the newest node (t_n), fhist[7] = f at t_{n-7}.
    let mut fhist = [[0.0; N]; ORDER];
    for j in 0..ORDER {
        let tj = t0 + j as f64 * step;
        let yj = if j == 0 { *y0 } else { warm.eval(tj) };
        fhist[ORDER - 1 - j] = eval(sys, tj, &yj); // oldest first, so index 0 ends up newest
    }
    let mut y = warm.eval(t0 + f64::from(ORDER as u32 - 1) * step);
    let mut t = t0 + f64::from(ORDER as u32 - 1) * step;

    for _ in (ORDER - 1)..n_steps {
        // Predict (Adams–Bashforth): y_p = y + h·Σ AB8[j]·f_{n-j}.
        let mut y_pred = y;
        for i in 0..N {
            let mut acc = 0.0;
            for j in 0..ORDER {
                acc += AB8[j] * fhist[j][i];
            }
            y_pred[i] += step * acc;
        }
        let f_pred = eval(sys, t + step, &y_pred);

        // Correct (Adams–Moulton): y_{n+1} = y + h·(AM8[0]·f_p + Σ_{j≥1} AM8[j]·f_{n+1-j}).
        let mut y_next = y;
        for i in 0..N {
            let mut acc = AM8[0] * f_pred[i];
            for j in 1..ORDER {
                acc += AM8[j] * fhist[j - 1][i];
            }
            y_next[i] += step * acc;
        }

        // Evaluate at the corrected state and shift the history.
        t += step;
        y = y_next;
        let f_new = eval(sys, t, &y);
        for j in (1..ORDER).rev() {
            fhist[j] = fhist[j - 1];
        }
        fhist[0] = f_new;
    }

    Ok(y)
}

/// Integrate as [`integrate_abm8`], returning a dense cubic-Hermite [`Trajectory`] for the event
/// detector. A segment is built between each pair of adjacent nodes (startup and corrector alike)
/// from the state and its derivative there.
///
/// # Errors
///
/// As [`integrate_abm8`].
pub fn integrate_abm8_dense<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<Trajectory<N, HermiteDense<N>>, IntegrationError> {
    if !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, ORDER - 1)?;
    let step = (t1 - t0) / n_steps as f64;

    let warm = integrate(
        &Dop853,
        sys,
        t0,
        y0,
        t0 + f64::from(ORDER as u32 - 1) * step,
        step,
        Controls::new(Tolerance::new(1e-14, 1e-14), 1e-10),
    )?;

    // Node states and derivatives, collected at every node so the Hermite segments span [t0, t1].
    let mut fhist = [[0.0; N]; ORDER];
    let mut ynodes = Vec::with_capacity(n_steps + 1);
    let mut fnodes = Vec::with_capacity(n_steps + 1);
    for j in 0..ORDER {
        let tj = t0 + j as f64 * step;
        let yj = if j == 0 { *y0 } else { warm.eval(tj) };
        let fj = eval(sys, tj, &yj);
        fhist[ORDER - 1 - j] = fj;
        ynodes.push(yj);
        fnodes.push(fj);
    }
    let mut y = ynodes[ORDER - 1];
    let mut t = t0 + f64::from(ORDER as u32 - 1) * step;

    for _ in (ORDER - 1)..n_steps {
        let mut y_pred = y;
        for i in 0..N {
            let mut acc = 0.0;
            for j in 0..ORDER {
                acc += AB8[j] * fhist[j][i];
            }
            y_pred[i] += step * acc;
        }
        let f_pred = eval(sys, t + step, &y_pred);

        let mut y_next = y;
        for i in 0..N {
            let mut acc = AM8[0] * f_pred[i];
            for j in 1..ORDER {
                acc += AM8[j] * fhist[j - 1][i];
            }
            y_next[i] += step * acc;
        }

        t += step;
        y = y_next;
        let f_new = eval(sys, t, &y);
        ynodes.push(y);
        fnodes.push(f_new);
        for j in (1..ORDER).rev() {
            fhist[j] = fhist[j - 1];
        }
        fhist[0] = f_new;
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

    /// Coefficients of the manufactured solution, ascending powers, one row per component.
    const POLY: [[f64; ORDER + 1]; 2] = [
        [-1.0, 4.0, 0.0, -1.0, 0.0, 2.0, 0.0, -3.0, 1.0],
        [7.0, 2.0, -5.0, 0.0, 3.0, 0.0, 1.0, 0.0, -2.0],
    ];

    /// Strength of the feedback term. Small enough that `λh` is deep inside the stability region.
    const FEEDBACK: f64 = 0.5;

    /// `y' = P'(t) + λ·(y − P(t))`, whose exact solution through `(t, P(t))` is the degree-8
    /// polynomial `P`. An order-8 linear multistep method reproduces a degree-8 trajectory exactly, so
    /// the comparison is against round-off and not against a tolerance. The `λ` term keeps the right
    /// side dependent on `y` as well as on `t`: a predictor that drifts is fed back through `f_pred`
    /// instead of being discarded, and the evaluation *times* enter the answer at all.
    struct Manufactured;

    impl Manufactured {
        fn exact(t: f64) -> [f64; 2] {
            let mut y = [0.0; 2];
            for i in 0..2 {
                for m in (0..=ORDER).rev() {
                    y[i] = y[i] * t + POLY[i][m];
                }
            }
            y
        }

        fn exact_dot(t: f64) -> [f64; 2] {
            let mut d = [0.0; 2];
            for i in 0..2 {
                for m in (1..=ORDER).rev() {
                    d[i] = d[i] * t + m as f64 * POLY[i][m];
                }
            }
            d
        }
    }

    impl DynamicalSystem<2> for Manufactured {
        fn derivative(&self, t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
            let p = Self::exact(t);
            let dp = Self::exact_dot(t);
            for i in 0..2 {
                dydt[i] = dp[i] + FEEDBACK * (y[i] - p[i]);
            }
        }
    }

    /// `y' = y`, exact solution `e^t`. Non-polynomial, so the grid the method chooses is visible in
    /// the answer — which a polynomial trajectory hides, being integrated exactly at any step.
    struct Exponential;

    impl DynamicalSystem<1> for Exponential {
        fn derivative(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
            dydt[0] = y[0];
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

    #[test]
    fn rejects_bad_arguments() {
        let y0 = Manufactured::exact(T0);
        // The zero step is tried twice. Away from the origin a step count derived from `h = 0`
        // saturates, the startup nodes it implies round back onto `t0`, and the startup would refuse
        // the collapsed span on its own; from the origin they stay apart, so the guard is the only
        // thing standing between a zero step and an integration.
        for (t0, t1, h) in [
            (T0, T0, 0.05),  // empty span
            (T1, T0, 0.05),  // backwards: the fixed-step form integrates forwards only
            (T0, T1, 0.0),   // zero step
            (0.0, T1, 0.0),  // zero step from the origin
            (T0, T1, -0.05), // backwards step over a forwards span
        ] {
            assert_eq!(
                integrate_abm8(&Manufactured, t0, &y0, t1, h),
                Err(IntegrationError::BadArguments),
                "accepted t0={t0} t1={t1} h={h}"
            );
            assert_eq!(
                integrate_abm8_dense(&Manufactured, t0, &y0, t1, h).err(),
                Some(IntegrationError::BadArguments),
                "dense accepted t0={t0} t1={t1} h={h}"
            );
        }
    }

    #[test]
    fn reproduces_a_degree_eight_polynomial_exactly() {
        let y0 = Manufactured::exact(T0);
        let want = Manufactured::exact(T1);
        let h = (T1 - T0) / 40.0;

        let y = integrate_abm8(&Manufactured, T0, &y0, T1, h).unwrap();
        let err = max_abs_diff(&y, &want);
        println!("ABM8 on a degree-8 trajectory: |Δy| = {err:.3e}");
        assert!(err < 1e-10, "not exact on a degree-8 trajectory: {err:.3e}");

        let traj = integrate_abm8_dense(&Manufactured, T0, &y0, T1, h).unwrap();
        let derr = max_abs_diff(&traj.final_state(), &want);
        println!("ABM8 dense on a degree-8 trajectory: |Δy| = {derr:.3e}");
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
        let y0 = Manufactured::exact(T0);
        let want = 2 * (n_steps - (ORDER - 1));

        let probe = CountAfter::new(Manufactured, after);
        integrate_abm8(&probe, T0, &y0, T1, h).unwrap();
        assert_eq!(probe.hits.get(), want, "PECE evaluations past the startup");

        let probe = CountAfter::new(Manufactured, after);
        let traj = integrate_abm8_dense(&probe, T0, &y0, T1, h).unwrap();
        assert_eq!(
            probe.hits.get(),
            want,
            "dense PECE evaluations past the startup"
        );
        assert_eq!(traj.step_counts().0 as usize, n_steps);
        assert_eq!(traj.step_ends().len(), n_steps);
    }

    #[test]
    fn a_span_below_the_startup_length_is_startup_only() {
        let t1 = 7.0_f64;
        let h = t1 / 2.0; // two steps asked for; seven startup intervals are the floor
        let want = t1.exp();

        let probe = CountAfter::new(Exponential, t1);
        let y = integrate_abm8(&probe, 0.0, &[1.0], t1, h).unwrap();
        let rel = (y[0] / want - 1.0).abs();
        println!("ABM8 over a sub-startup span: relative error {rel:.3e}");
        assert!(rel < 1e-9, "not integrated on the startup grid: {rel:.3e}");
        assert_eq!(probe.hits.get(), 0, "the startup sampled beyond t1");

        let probe = CountAfter::new(Exponential, t1);
        let traj = integrate_abm8_dense(&probe, 0.0, &[1.0], t1, h).unwrap();
        let drel = (traj.final_state()[0] / want - 1.0).abs();
        println!("ABM8 dense over a sub-startup span: relative error {drel:.3e}");
        assert!(
            drel < 1e-9,
            "dense not integrated on the startup grid: {drel:.3e}"
        );
        assert_eq!(probe.hits.get(), 0, "the dense startup sampled beyond t1");
        assert_eq!(traj.step_counts().0 as usize, ORDER - 1);
    }

    #[test]
    fn dense_interpolates_between_nodes() {
        let n_steps: usize = 120;
        let step = (T1 - T0) / n_steps as f64;
        let y0 = Manufactured::exact(T0);
        let traj = integrate_abm8_dense(&Manufactured, T0, &y0, T1, step).unwrap();

        // Mid-segment. On a node every cubic Hermite returns the node state whatever abscissa or end
        // derivative it was built from, so a node query tests nothing about the segment's geometry.
        let t = T0 + 72.5 * step;
        let err = max_abs_diff(&traj.eval(t), &Manufactured::exact(t));
        println!("ABM8 dense mid-segment error: {err:.3e}");
        assert!(
            err < 1e-6,
            "Hermite segment is not on the right abscissa: {err:.3e}"
        );
    }
}
