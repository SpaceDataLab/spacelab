//! The RKN12(10)17 Runge–Kutta–Nyström pair of Dormand, El-Mikkawy & Prince (1987): a
//! twelfth-order integrator for the special second-order system `r̈ = f(t, r)`, with an embedded
//! tenth-order solution for adaptive step control.
//!
//! # Why Nyström, and when
//!
//! An orbit is `r̈ = a(r)` — the acceleration depends on position, not velocity. A Runge–Kutta method
//! ignores that: it doubles the system to first order and spends stages transporting `v` it already
//! knows. A Nyström method integrates the second-order form directly, so at a given order it needs
//! fewer stages and less work. RKN12(10) reaches twelfth order in 17 stages, where a comparable RK
//! would need many more. The price is the **special form**: the acceleration must not depend on
//! velocity, so this is for conservative gravity (`PointMass`, `Zonal`, `ThirdBody`, `SolarRadiationPressure`),
//! **not** drag. For drag, use [`crate::Dop853`].
//!
//! # Coefficients
//!
//! The tableau is `crate::rkn1210_consts`, machine-generated from a ~30-digit primary listing
//! (`tools/gen_rkn1210.py`) so no constant is truncated by hand. The gate below is the proof they
//! transcribed correctly: a wrong coefficient fails an order condition.
//!
//! Gate: `derivations/rkn1210_order.mac` parses those literals into exact rationals and shows that
//! they meet every special Runge–Kutta–Nyström order condition — 563 position and 1104 velocity
//! conditions through order 12 for `b̂`, `b̂'`, 151 and 288 through order 10 for the embedded `b`,
//! `b'` — to `1.0e-28` of each condition's scale, the 30-digit transcription, while conditions of
//! order 13 and 11 fail; `derivations/rkn1210_order.gp` re-checks all of them. The module test
//! `the_shipped_tableau_meets_the_nystrom_order_conditions` (G1) reads the same conditions off the
//! shipped `step` through each tree's polynomial system, in `f64`.

#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use crate::driver::IntegrationError;
use crate::rkn1210_consts::{A, B, BHAT, BP, BPHAT, C};

const STAGES: usize = 17;

/// One RKN12(10) step from `(r, v)` at time `t` over a step `h`, for `r̈ = accel(t, r)`.
///
/// Returns the propagated twelfth-order state `(r, v)` and the position and velocity error norms
/// separately — the twelfth-minus-tenth-order differences `‖Δr‖`, `‖Δv‖` — so the controller can hold
/// each against its own scale (position and velocity differ by ~`‖r‖/‖v‖ ≈ 900` in LEO; a single
/// scale under-controls one of them by that factor).
pub fn step<const D: usize, F: Fn(f64, &[f64; D]) -> [f64; D]>(
    accel: &F,
    t: f64,
    r: &[f64; D],
    v: &[f64; D],
    h: f64,
) -> ([f64; D], [f64; D], f64, f64) {
    let h2 = h * h;
    let mut f = [[0.0; D]; STAGES];
    for i in 0..STAGES {
        let mut g = [0.0; D];
        for d in 0..D {
            let mut acc = r[d] + C[i] * h * v[d];
            for j in 0..i {
                acc += h2 * A[i][j] * f[j][d];
            }
            g[d] = acc;
        }
        f[i] = accel(t + C[i] * h, &g);
    }

    let mut r_new = *r;
    let mut v_new = *v;
    let mut err_pos2 = 0.0;
    let mut err_vel2 = 0.0;
    for d in 0..D {
        let (mut s_bhat, mut s_bphat, mut s_db, mut s_dbp) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..STAGES {
            s_bhat += BHAT[i] * f[i][d];
            s_bphat += BPHAT[i] * f[i][d];
            s_db += (BHAT[i] - B[i]) * f[i][d];
            s_dbp += (BPHAT[i] - BP[i]) * f[i][d];
        }
        r_new[d] = r[d] + h * v[d] + h2 * s_bhat;
        v_new[d] = v[d] + h * s_bphat;
        err_pos2 += (h2 * s_db) * (h2 * s_db);
        err_vel2 += (h * s_dbp) * (h * s_dbp);
    }
    (r_new, v_new, libm::sqrt(err_pos2), libm::sqrt(err_vel2))
}

fn norm<const D: usize>(x: &[f64; D]) -> f64 {
    libm::sqrt(x.iter().map(|v| v * v).sum())
}

/// Integrate `r̈ = accel(t, r)` from `t0` to `t1` with adaptive steps controlled by the embedded
/// error estimate against `atol + rtol · max(‖r‖, ‖v‖)`.
///
/// `h0` is the initial step (a fraction of an orbital period is a good guess); the step then adapts.
/// The exponent is `1/(10 + 1)`: the error estimator carries the embedded tenth order, so the step
/// scales on that. `max_steps` bounds the passes — accepted plus rejected — the controller may take;
/// pass [`DEFAULT_MAX_STEPS`](crate::driver::DEFAULT_MAX_STEPS) unless the run is known to need
/// more. Returns the final `(r, v)`.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] for a non-increasing span, a non-positive initial step, a
/// `max_steps` of zero, or non-finite tolerances;
/// [`IntegrationError::StepLimit`] if the run needs more than `max_steps` passes. The step floor
/// alone does not bound the work: at `h_min = 16 ε (t1 − t0)` a force model that never satisfies
/// the tolerance still advances, one floor-sized step at a time, for `1/(16 ε) ≈ 2.8e14` passes.
// The Nystrom form splits the state into position and velocity, so the same seven controls the
// first-order integrators take arrive here as eight.
#[allow(clippy::too_many_arguments)]
pub fn integrate<const D: usize, F: Fn(f64, &[f64; D]) -> [f64; D]>(
    accel: &F,
    t0: f64,
    r0: &[f64; D],
    v0: &[f64; D],
    t1: f64,
    h0: f64,
    tol: crate::Tolerance,
    max_steps: u32,
) -> Result<([f64; D], [f64; D]), IntegrationError> {
    if !(t1 > t0 && h0 > 0.0 && max_steps > 0 && tol.atol.is_finite() && tol.rtol.is_finite()) {
        return Err(IntegrationError::BadArguments);
    }
    let mut t = t0;
    let (mut r, mut v) = (*r0, *v0);
    let mut h = h0.min(t1 - t0);
    let h_min = 16.0 * f64::EPSILON * (t1 - t0);
    // The budget is a range iterator, not a counter incremented by hand: `passes += 1` mutated to
    // `passes *= 1` pins the count at zero and the guard never fires, which is the one runaway this
    // guard exists to stop. A finite iterator cannot be turned back into an unbounded loop by
    // changing an operator.
    let mut budget = 0..max_steps;

    while t < t1 {
        if budget.next().is_none() {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(max_steps),
            });
        }
        if t + h > t1 {
            h = t1 - t;
        }
        let (r_new, v_new, err_r, err_v) = step(accel, t, &r, &v, h);
        // Hold position and velocity each against its own scale, then take the worst case — a single
        // shared scale would leave velocity under-controlled by ~‖r‖/‖v‖.
        let scale_r = tol.atol + tol.rtol * norm(&r_new);
        let scale_v = tol.atol + tol.rtol * norm(&v_new);
        let err = (err_r / scale_r).max(err_v / scale_v);

        if err <= 1.0 || h <= h_min {
            t += h;
            r = r_new;
            v = v_new;
        }
        // PI-free step control on the tenth-order estimator; err is normalised to 1. Clamp the growth.
        let ratio = if err > 0.0 { 1.0 / err } else { 5.0 };
        let factor = 0.9 * libm::pow(ratio, 1.0 / 11.0);
        h *= factor.clamp(0.2, 5.0);
        h = h.max(h_min);
    }
    Ok((r, v))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::expect_used,
        // the force closures must match `integrate`'s `Fn(f64, &[f64; D]) -> [f64; D]` bound,
        // which is by reference for every D including the 8-byte D = 1
        clippy::trivially_copy_pass_by_ref
    )]

    use super::*;
    use crate::Tolerance;
    use crate::driver::DEFAULT_MAX_STEPS;
    use core::cell::RefCell;

    const MU: f64 = 3.986_004_418e14;

    /// Point-mass gravity. Autonomous, so on its own it says nothing about the stage times.
    fn kepler(_t: f64, r: &[f64; 3]) -> [f64; 3] {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let k = -MU / (r2 * r2.sqrt());
        [k * r[0], k * r[1], k * r[2]]
    }

    /// No force at all. Every stage derivative is zero, so the twelfth- and tenth-order solutions
    /// agree bit for bit and the scaled error is exactly `0.0` — the one input that reaches the
    /// controller's zero-error branch, where the growth factor is the fixed `0.9·5^(1/11)`.
    fn free(_t: f64, _r: &[f64; 1]) -> [f64; 1] {
        [0.0]
    }

    /// Periapsis state and period of the orbit `(a, e)`.
    fn orbit(a: f64, e: f64) -> ([f64; 3], [f64; 3], f64) {
        let r_p = a * (1.0 - e);
        let v_p = (MU * (1.0 + e) / r_p).sqrt();
        let period = core::f64::consts::TAU * (a * a * a / MU).sqrt();
        ([r_p, 0.0, 0.0], [0.0, v_p, 0.0], period)
    }

    fn dist<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut s = 0.0;
        for d in 0..D {
            s += (a[d] - b[d]) * (a[d] - b[d]);
        }
        s.sqrt()
    }

    /// The trajectory `r(t) = Σ_k POLY[k]·t^k`, degree twelve.
    ///
    /// Under a forcing that depends on `t` alone the step is pure quadrature on the nodes `c_i`, and
    /// the pair reproduces the trajectory exactly for as long as the quadrature conditions hold:
    /// `Σ b̂_i c_i^{k-1} = 1/(k(k+1))` for the position, which the tableau satisfies to `k = 11`
    /// (the `h²` prefactor buys one order, so a degree-10 forcing is the limit), and
    /// `Σ b̂'_i c_i^{k-1} = 1/k` for the velocity, satisfied one order further. Degree twelve is
    /// therefore the highest trajectory exact in both, and the assertion is round-off.
    const POLY: [f64; 13] = [
        0.7, -1.3, 0.45, 2.1, -0.85, 1.6, -0.35, 0.95, -1.15, 0.6, -0.25, 1.05, -0.5,
    ];

    fn derivative_of(c: &[f64; 13]) -> [f64; 13] {
        let mut d = [0.0; 13];
        for k in 1..13 {
            d[k - 1] = c[k] * k as f64;
        }
        d
    }

    fn horner(c: &[f64; 13], t: f64) -> f64 {
        c.iter().rev().fold(0.0, |acc, &ck| acc * t + ck)
    }

    /// A derivative-evaluation trace.
    ///
    /// A step controller is self-correcting: any variant that happens to take small enough steps
    /// still lands on the right answer, so its arithmetic is invisible to an accuracy assertion.
    /// The cost is not, and this observes it without a step counter in the public API.
    #[derive(Default)]
    struct Trace(RefCell<Vec<f64>>);

    impl Trace {
        /// `(evaluations, accepted steps, rejected steps)`. `C[0] == 0`, so every `STAGES`-th call
        /// carries the base time of one attempt; a rejected attempt repeats it unchanged, and the
        /// last attempt is always accepted because only acceptance ends the loop.
        fn cost(&self) -> (usize, usize, usize) {
            let calls = self.0.borrow();
            let base: Vec<f64> = calls.iter().copied().step_by(STAGES).collect();
            let accepted =
                base.windows(2).filter(|w| w[0] != w[1]).count() + usize::from(!base.is_empty());
            (calls.len(), accepted, base.len() - accepted)
        }
    }

    /// A twelfth-order pair integrates a degree-twelve trajectory exactly when the forcing depends
    /// on `t` alone — which makes every stage *time* observable. Under the autonomous gravity the
    /// rest of the suite uses, `accel`'s first argument is dead and nothing constrains `t + c_i·h`.
    #[test]
    fn a_time_dependent_polynomial_trajectory_is_integrated_exactly() {
        // A time origin off zero and a step off one, so `t + c_i·h` is distinguishable from
        // `t − c_i·h`, `t·c_i·h`, `t + c_i + h` and `t + c_i/h` alike.
        const STEPS: usize = 4;
        let (t0, h) = (-0.4, 0.35);
        let d1 = derivative_of(&POLY);
        let d2 = derivative_of(&d1);
        let forcing = |t: f64, _r: &[f64; 1]| [horner(&d2, t)];

        let (mut r, mut v) = ([horner(&POLY, t0)], [horner(&d1, t0)]);
        for k in 0..STEPS {
            let (rn, vn, _, _) = step(&forcing, t0 + k as f64 * h, &r, &v, h);
            r = rn;
            v = vn;
        }

        let t1 = t0 + STEPS as f64 * h;
        let dr = (r[0] - horner(&POLY, t1)).abs();
        let dv = (v[0] - horner(&d1, t1)).abs();
        println!("degree-12 trajectory: |Δr| = {dr:.3e}, |Δv| = {dv:.3e}");
        assert!(dr < 1e-13, "position not exact on a polynomial: {dr:.3e}");
        assert!(dv < 1e-13, "velocity not exact on a polynomial: {dv:.3e}");
    }

    /// The velocity half of the embedded estimate, held to its own order. It is returned separately
    /// from the position half precisely because the two carry different scales, and every other test
    /// discards it — an estimate nothing reads can be any function of the stage derivatives at all.
    #[test]
    fn the_velocity_error_estimate_tracks_its_tenth_order() {
        let (r0, v0, period) = orbit(8_000e3, 0.1);
        let (_, _, _, coarse) = step(&kepler, 0.0, &r0, &v0, period / 8.0);
        let (_, _, _, fine) = step(&kepler, 0.0, &r0, &v0, period / 16.0);
        println!("velocity error estimate: {coarse:.3e} → {fine:.3e}");
        assert!(
            coarse.is_finite() && fine.is_finite() && coarse > 0.0 && fine > 0.0,
            "velocity error estimate is not positive and finite: {coarse:e} → {fine:e}"
        );
        let order = (coarse / fine).log2();
        assert!(
            order > 9.0,
            "velocity estimator order {order:.2} below its tenth-order design"
        );
    }

    /// The controller's cost, pinned. Accepted and rejected step counts are the observable an
    /// accuracy assertion cannot absorb: an eccentric orbit started with a half-period step forces
    /// the reject branch and the shrink limiter, and both scaled errors take turns driving the
    /// decision, so neither the position nor the velocity scale is free to change.
    #[test]
    fn the_step_controller_cost_is_pinned() {
        let (r0, v0, period) = orbit(8_000e3, 0.6);
        let trace = Trace::default();
        let watched = |t: f64, y: &[f64; 3]| {
            trace.0.borrow_mut().push(t);
            kepler(t, y)
        };
        let (r, v) = integrate(
            &watched,
            0.0,
            &r0,
            &v0,
            period,
            period / 2.0,
            Tolerance::new(1e-8, 1e-10),
            DEFAULT_MAX_STEPS,
        )
        .expect("one orbit fits the default step budget");

        let cost = trace.cost();
        let (dr, dv) = (dist(&r, &r0), dist(&v, &v0));
        println!(
            "one orbit at (1e-8, 1e-10): {cost:?}, closure |Δr| = {dr:.3e} m, |Δv| = {dv:.3e} m/s"
        );
        assert_eq!(cost, (510, 20, 10), "step controller cost moved");
        // Independent of the reference: after exactly one period the orbit must close on itself.
        assert!(dr < 1e-3, "orbit did not close: {dr:.3e} m");
        assert!(dv < 1e-5, "orbit did not close in velocity: {dv:.3e} m/s");
    }

    /// Tightening the tolerance must cost steps at the rate the estimator's order predicts: the step
    /// scales as `tol^(1/11)`, so `10⁴` of tolerance is `10^(4/11) ≈ 2.3` times the work.
    #[test]
    fn tightening_the_tolerance_costs_steps_at_the_estimator_order() {
        let (r0, v0, period) = orbit(8_000e3, 0.6);
        let accepted = |scale: f64| {
            let trace = Trace::default();
            let watched = |t: f64, y: &[f64; 3]| {
                trace.0.borrow_mut().push(t);
                kepler(t, y)
            };
            let _ = integrate(
                &watched,
                0.0,
                &r0,
                &v0,
                period,
                period / 2.0,
                Tolerance::new(1e-8 * scale, 1e-10 * scale),
                DEFAULT_MAX_STEPS,
            );
            trace.cost().1 as f64
        };
        let (loose, tight) = (accepted(1.0), accepted(1e-4));
        let growth = tight / loose;
        let predicted = libm::pow(10.0, 4.0 / 11.0);
        println!(
            "accepted steps {loose} → {tight}, growth {growth:.2} vs predicted {predicted:.2}"
        );
        assert!(
            (growth / predicted) > 0.8 && (growth / predicted) < 1.25,
            "step growth {growth:.2} is not the {predicted:.2} the estimator order predicts"
        );
    }

    /// An initial step far below the floor is raised to `h_min` and ramps back geometrically. The
    /// ramp is the controller's design read off in closed form — factor `0.9·5^(1/11)` per step from
    /// `16 ε (t1 − t0)` — so the step count is predicted with no reference to the code under test.
    #[test]
    fn a_step_below_the_floor_ramps_back_geometrically() {
        let (t0, t1) = (-0.5, 1.0);
        let (r0, v0) = ([2.0], [0.75]);
        let trace = Trace::default();
        let watched = |t: f64, y: &[f64; 1]| {
            trace.0.borrow_mut().push(t);
            free(t, y)
        };
        let (r, v) = integrate(
            &watched,
            t0,
            &r0,
            &v0,
            t1,
            1e-30,
            Tolerance::new(1e-10, 1e-10),
            DEFAULT_MAX_STEPS,
        )
        .expect("the ramp fits the default step budget");

        let q = 0.9 * libm::pow(5.0, 1.0 / 11.0);
        let h_min = 16.0 * f64::EPSILON * (t1 - t0);
        let ramp = ((1.0 + (t1 - t0) * (q - 1.0) / h_min).ln() / q.ln()).ceil() as usize;
        let attempts = trace.cost().0 / STAGES;
        println!(
            "ramp from {h_min:.3e} at ×{q:.6}: {attempts} attempts, predicted {}",
            ramp + 1
        );
        // The wasted step at `h0` itself, then the ramp.
        assert!(
            attempts.abs_diff(ramp + 1) <= 1,
            "ramp took {attempts} steps, the geometric model predicts {}",
            ramp + 1
        );
        assert_eq!(
            r[0],
            r0[0] + (t1 - t0) * v0[0],
            "drift is not a straight line"
        );
        assert_eq!(v[0], v0[0], "a force-free velocity changed");
    }

    /// An initial step longer than the whole span is cut to the span, in one step. From a time
    /// origin off zero `t1 − t0`, `t1 + t0` and `t1/t0` are three different numbers, and only the
    /// first covers the interval exactly once.
    #[test]
    fn the_initial_step_never_exceeds_the_span() {
        let (t0, t1) = (-0.5, 1.0);
        let trace = Trace::default();
        let watched = |t: f64, y: &[f64; 1]| {
            trace.0.borrow_mut().push(t);
            free(t, y)
        };
        let (r, _) = integrate(
            &watched,
            t0,
            &[2.0],
            &[0.75],
            t1,
            10.0,
            Tolerance::new(1e-10, 1e-10),
            DEFAULT_MAX_STEPS,
        )
        .expect("one step fits the default step budget");
        let cost = trace.cost();
        println!("h0 = 10 over a span of {}: {cost:?}", t1 - t0);
        assert_eq!(cost, (STAGES, 1, 0), "the span was not covered in one step");
        assert_eq!(r[0], 2.0 + (t1 - t0) * 0.75, "drift is not a straight line");
    }

    /// A force model the controller can never satisfy is refused, not integrated forever.
    ///
    /// The step floor is not a bound on the work: below it the step is accepted regardless, so `t`
    /// still advances — by `h_min = 16 ε (t1 − t0)` a time, i.e. `1/(16 ε) ≈ 2.8e14` passes to cross
    /// the span. At the measured cost of one 17-stage pass that is months of wall clock with no
    /// error and no way for the caller to interrupt. `max_steps` is what makes it terminate, and the
    /// count it stops at is exactly the count it was given.
    #[test]
    fn a_force_model_that_never_meets_the_tolerance_is_refused_not_run_forever() {
        // Alternating sign at every evaluation: no step size makes the twelfth- and tenth-order
        // solutions agree, so the scaled error never falls to 1 however far the step shrinks.
        let flips = RefCell::new(0u64);
        let adversarial = |_t: f64, _r: &[f64; 1]| {
            let mut n = flips.borrow_mut();
            *n += 1;
            [if n.is_multiple_of(2) { 1e12 } else { -1e12 }]
        };
        for cap in [1u32, 7, 250] {
            let got = integrate(
                &adversarial,
                0.0,
                &[0.0],
                &[0.0],
                1.0,
                0.5,
                Tolerance::new(1e-14, 1e-14),
                cap,
            );
            assert_eq!(
                got,
                Err(IntegrationError::StepLimit {
                    steps: f64::from(cap)
                }),
                "cap {cap} did not stop the run"
            );
        }

        // And the arguments a caller can get wrong are refused before the loop, not inside it.
        for (t1, h0, cap) in [(0.0, 1.0, 1u32), (1.0, 0.0, 1), (1.0, 1.0, 0)] {
            assert_eq!(
                integrate(
                    &free,
                    0.0,
                    &[0.0],
                    &[0.0],
                    t1,
                    h0,
                    Tolerance::new(1e-10, 1e-10),
                    cap
                ),
                Err(IntegrationError::BadArguments),
                "t1 = {t1}, h0 = {h0}, max_steps = {cap} was accepted"
            );
        }
    }

    /// A special-Nyström tree: the children of an `f` vertex, each the velocity leaf or a subtree.
    #[derive(Clone)]
    enum Child {
        Vel,
        Sub(FTree),
    }

    #[derive(Clone)]
    struct FTree(Vec<Child>);

    impl Child {
        fn weight(&self) -> usize {
            match self {
                Child::Vel => 1,
                Child::Sub(s) => s.size() + 2,
            }
        }
    }

    impl FTree {
        /// `|T|`: the power of `h` the tree carries in `f(Rᵢ)`.
        fn size(&self) -> usize {
            self.0.iter().map(Child::weight).sum()
        }

        /// `q(T) = ∏ q(S)/((|S|+1)(|S|+2))` over the subtrees: the exact weight is `θ^{|T|} q(T)`.
        fn q(&self) -> f64 {
            self.0
                .iter()
                .map(|c| match c {
                    Child::Vel => 1.0,
                    Child::Sub(s) => s.q() / ((s.size() + 1) * (s.size() + 2)) as f64,
                })
                .product()
        }

        /// The elementary weight `Φᵢ(T)` from the shipped table, and the same with `|A|`, `|c|`.
        fn phi(&self) -> ([f64; STAGES], [f64; STAGES]) {
            let (mut p, mut m) = ([1.0; STAGES], [1.0; STAGES]);
            for child in &self.0 {
                let (w, wm) = match child {
                    Child::Vel => (C, C.map(f64::abs)),
                    Child::Sub(s) => {
                        let (sp, sm) = s.phi();
                        let mut w = [0.0; STAGES];
                        let mut wm = [0.0; STAGES];
                        for i in 0..STAGES {
                            for j in 0..i {
                                w[i] += A[i][j] * sp[j];
                                wm[i] += A[i][j].abs() * sm[j];
                            }
                        }
                        (w, wm)
                    }
                };
                for i in 0..STAGES {
                    p[i] *= w[i];
                    m[i] *= wm[i];
                }
            }
            (p, m)
        }
    }

    /// Every special-Nyström tree of size `0..=n`, by size: an `f` vertex over each multiset of
    /// velocity leaves and smaller trees whose weights sum to the size, taken in pool order — the
    /// enumeration `derivations/rkn1210_order.gp` uses.
    fn ftrees(n: usize) -> Vec<Vec<FTree>> {
        fn multisets(
            pool: &[Child],
            from: usize,
            left: usize,
            cur: &mut Vec<Child>,
            out: &mut Vec<FTree>,
        ) {
            if left == 0 {
                out.push(FTree(cur.clone()));
                return;
            }
            for i in from..pool.len() {
                let w = pool[i].weight();
                if w <= left {
                    cur.push(pool[i].clone());
                    multisets(pool, i, left - w, cur, out);
                    cur.pop();
                }
            }
        }
        let mut by_size = vec![vec![FTree(Vec::new())]];
        for m in 1..=n {
            let mut pool = vec![Child::Vel];
            for smaller in by_size.iter().take(m.saturating_sub(1)) {
                pool.extend(smaller.iter().cloned().map(Child::Sub));
            }
            let mut out = Vec::new();
            multisets(&pool, 0, m, &mut Vec::new(), &mut out);
            by_size.push(out);
        }
        by_size
    }

    /// Components of the tree system: up to 12 vertices for the trees of size `<= 11` read here.
    const TREE_DIM: usize = 12;

    /// The Nyström tree system of `t`: one component per vertex, from `r = 0`. A velocity leaf has
    /// `r'' = 0, v(0) = 1`, so `r(τ) = τ`; an `f` vertex has `r'' = ∏` of its children and
    /// `v(0) = 0`. The system is polynomial and nilpotent, so the step truncates nothing: from
    /// `t = 0` with `h = 1` the root holds `Σ b̂ᵢ Φᵢ(T)` in position and `Σ b̂'ᵢ Φᵢ(T)` in velocity,
    /// up to rounding. With `clock` set every velocity leaf is read from the stage time
    /// `t + cᵢh` instead — the leaf's exact solution is the clock itself.
    struct NystromTreeSystem {
        children: Vec<Vec<usize>>,
        vel: Vec<bool>,
        clock: bool,
    }

    impl NystromTreeSystem {
        fn new(t: &FTree, clock: bool) -> Self {
            fn walk(t: &FTree, sys: &mut NystromTreeSystem) -> usize {
                let me = sys.children.len();
                sys.children.push(Vec::new());
                sys.vel.push(false);
                for child in &t.0 {
                    let id = match child {
                        Child::Vel => {
                            sys.children.push(Vec::new());
                            sys.vel.push(true);
                            sys.children.len() - 1
                        }
                        Child::Sub(s) => walk(s, sys),
                    };
                    sys.children[me].push(id);
                }
                me
            }
            let mut sys = Self {
                children: Vec::new(),
                vel: Vec::new(),
                clock,
            };
            walk(t, &mut sys);
            assert!(sys.children.len() <= TREE_DIM, "tree system too large");
            sys
        }

        fn accel(&self, t: f64, r: &[f64; TREE_DIM]) -> [f64; TREE_DIM] {
            let mut a = [0.0; TREE_DIM];
            for (k, kids) in self.children.iter().enumerate() {
                if !self.vel[k] {
                    a[k] = kids
                        .iter()
                        .map(|&u| if self.clock && self.vel[u] { t } else { r[u] })
                        .product();
                }
            }
            a
        }

        fn v0(&self) -> [f64; TREE_DIM] {
            let mut v = [0.0; TREE_DIM];
            for (k, &is_vel) in self.vel.iter().enumerate() {
                v[k] = if is_vel { 1.0 } else { 0.0 };
            }
            v
        }
    }

    /// `|residual| / scale` of one condition, the scale being `Σ |bᵢ| |Φᵢ| + |target|` with `|Φ|`
    /// formed from `|A|` and `|c|` — the magnitude every rounding in the sum is relative to.
    fn rel(got: f64, target: f64, w: &[f64; STAGES], phi_abs: &[f64; STAGES]) -> f64 {
        let scale: f64 = (0..STAGES).map(|i| w[i].abs() * phi_abs[i]).sum::<f64>() + target.abs();
        (got - target).abs() / scale
    }

    /// (G1) The shipped step and table meet every special Runge–Kutta–Nyström order condition the
    /// pair claims, and fail at the next order — the conditions `derivations/rkn1210_order.mac`
    /// evaluates in exact rationals on the same literals (worst relative residual `1.0e-28`, the
    /// 30-digit transcription), re-checked by `derivations/rkn1210_order.gp`.
    ///
    /// The twelfth-order solution is read off the shipped `step` through each tree's system, with
    /// the velocity leaves from their components (the `cᵢ h v` term) and again from the stage time
    /// (the `t + cᵢ h` passed to `accel`): position conditions through order 12 (563 trees),
    /// velocity through order 12 (1104). The embedded tenth-order weights `b`, `b'`, which `step`
    /// only returns inside an error norm, are checked from the table: 151 and 288 conditions. Each
    /// residual is held to `16 ε` of its scale, which bounds the rounding of sums over at most 17
    /// stages nested five deep with room to spare (measured worst: `1.25 ε`). At order `p + 1` some
    /// condition of every weight vector fails by more than `1e-6` of that scale: 9, 16, 17 and 25
    /// of them for `b̂`, `b̂'`, `b`, `b'` (the derivation's scale, `Σ |bᵢ Φᵢ|`, is smaller and
    /// counts more).
    #[test]
    fn the_shipped_tableau_meets_the_nystrom_order_conditions() {
        let forest = ftrees(12);
        let counts: Vec<usize> = forest.iter().map(Vec::len).collect();
        assert_eq!(counts, [1, 1, 2, 3, 6, 10, 20, 36, 72, 137, 275, 541, 1098]);
        let bar = 16.0 * f64::EPSILON;
        let mut worst = 0.0_f64;
        let mut fails = [0_usize; 4];
        for (n, ts) in forest.iter().enumerate() {
            for t in ts {
                let (phi, phi_abs) = t.phi();
                let q = t.q();
                let pos_target = q / ((n + 1) * (n + 2)) as f64;
                let vel_target = q / (n + 1) as f64;
                if n <= 11 {
                    for clock in [false, true] {
                        let sys = NystromTreeSystem::new(t, clock);
                        let accel = |tt: f64, r: &[f64; TREE_DIM]| sys.accel(tt, r);
                        let (r1, v1, _, _) = step(&accel, 0.0, &[0.0; TREE_DIM], &sys.v0(), 1.0);
                        let rv = rel(v1[0], vel_target, &BPHAT, &phi_abs);
                        let rp = rel(r1[0], pos_target, &BHAT, &phi_abs);
                        if n <= 10 {
                            assert!(rp <= bar, "b̂ position, size {n}, clock {clock}: {rp:e}");
                        } else if !clock && rp > 1e-6 {
                            fails[0] += 1;
                        }
                        assert!(rv <= bar, "b̂' velocity, size {n}, clock {clock}: {rv:e}");
                        worst = worst.max(rv);
                        if n <= 10 {
                            worst = worst.max(rp);
                        }
                    }
                }
                let dot = |w: &[f64; STAGES]| (0..STAGES).map(|i| w[i] * phi[i]).sum::<f64>();
                if n == 12 && rel(dot(&BPHAT), vel_target, &BPHAT, &phi_abs) > 1e-6 {
                    fails[1] += 1;
                }
                let rp = rel(dot(&B), pos_target, &B, &phi_abs);
                let rv = rel(dot(&BP), vel_target, &BP, &phi_abs);
                match n {
                    0..=8 => {
                        assert!(rp <= bar && rv <= bar, "b, b' size {n}: {rp:e}, {rv:e}");
                        worst = worst.max(rp).max(rv);
                    }
                    9 => {
                        assert!(rv <= bar, "b' size 9: {rv:e}");
                        fails[2] += usize::from(rp > 1e-6);
                    }
                    10 => fails[3] += usize::from(rv > 1e-6),
                    _ => {}
                }
            }
        }
        println!(
            "RKN12(10) order conditions in f64: worst {:.3} ε of scale; p+1 failures {fails:?}",
            worst / f64::EPSILON
        );
        assert!(
            fails.iter().all(|&f| f > 0),
            "a weight vector is of higher order: {fails:?}"
        );
    }
}
