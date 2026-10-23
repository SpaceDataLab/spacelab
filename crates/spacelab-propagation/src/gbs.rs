//! Gragg–Bulirsch–Stoer: modified-midpoint integration with Richardson extrapolation to high order.
//!
//! # Provenance and verification
//!
//! Each macro-step `H` is integrated with Gragg's modified-midpoint method at increasing substep
//! counts `n_j = 2, 4, 6, 8, …` (the harmonic sequence), and the results are polynomial-extrapolated
//! to `h → 0` in the node `h²`. Gragg (1965) showed the modified-midpoint error expands in *even*
//! powers of `h`, so each extrapolation column gains two orders: the `j`-th diagonal entry is order
//! `2j`. That order structure is proven symbolically in `derivations/gbs_extrapolation.mac` — Maxima
//! builds `S_n(z)` for the linear test `y' = λy` from the recurrence and confirms the diagonal reaches
//! orders 2, 4, 6, 8, 10 exactly. The extrapolation denominators `(n_j / n_{j-k+1})² − 1` are computed
//! from the sequence, not tabulated.
//!
//! The adaptive order-and-step control follows the Deuflhard/Hairer ODEX scheme (as in Apache Commons
//! Math's `GraggBulirschStoerIntegrator`): the tableau supplies an error estimate
//! `‖T_{j,j} − T_{j,j-1}‖`, the step is grown or shrunk from it, and the working order `k_opt` tracks
//! where convergence lands. The `bulirsch` crate implements the same method but pulls in `ndarray` and
//! is not `no_std`, so it is not a dependency; this is the self-contained suite member.
//!
//! GBS shines on smooth, non-stiff problems where a very tight tolerance is wanted cheaply — its order
//! rises to meet the accuracy demand, unlike a fixed-order Runge–Kutta pair. First-order ODEs via
//! [`DynamicalSystem`]; SI units are the caller's.

#![allow(clippy::cast_precision_loss)] // substep/row counts are <= 16, exactly representable in f64

#[cfg(test)]
use crate::driver::Controls;
use crate::{DynamicalSystem, Tolerance, driver::IntegrationError};

/// Maximum number of extrapolation rows (columns). Row `j` yields order `2j`, so 8 reaches order 16.
const KMAX: usize = 8;
/// Step-controller safety factor and growth/shrink clamps.
const SAFE: f64 = 0.9;
const FACMIN: f64 = 0.1;
const FACMAX: f64 = 4.0;

/// Gragg's modified-midpoint method: advance `(t0, y0)` across `H` in `n` substeps (n even), with the
/// symmetric endpoint smoothing that gives the even-power error expansion. Returns the state at
/// `t0 + H`. Cost: `n + 1` derivative evaluations.
fn modified_midpoint<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    big_h: f64,
    n: usize,
) -> [f64; N] {
    let h = big_h / n as f64;
    let mut f = [0.0; N];
    sys.derivative(t0, y0, &mut f);
    let mut ym = *y0; // Δ₀
    let mut yc = [0.0; N]; // Δ₁ = Δ₀ + h f(Δ₀)
    for i in 0..N {
        yc[i] = y0[i] + h * f[i];
    }
    for m in 1..n {
        sys.derivative(t0 + m as f64 * h, &yc, &mut f);
        let mut yn = [0.0; N]; // Δ_{m+1} = Δ_{m-1} + 2h f(Δ_m)
        for i in 0..N {
            yn[i] = ym[i] + 2.0 * h * f[i];
        }
        ym = yc;
        yc = yn;
    }
    // smoothing: S = ½(Δ_n + Δ_{n-1} + h f(t0+H, Δ_n)),  ym = Δ_{n-1}, yc = Δ_n
    sys.derivative(t0 + big_h, &yc, &mut f);
    let mut s = [0.0; N];
    for i in 0..N {
        s[i] = 0.5 * (yc[i] + ym[i] + h * f[i]);
    }
    s
}

/// Integrate `sys` from `t0` to `t1` with the adaptive Gragg–Bulirsch–Stoer method.
///
/// `h0` is the initial macro-step guess; the controller adapts both step and order to `tol`.
/// `max_steps` bounds the passes it may take doing so — pass
/// [`DEFAULT_MAX_STEPS`](crate::driver::DEFAULT_MAX_STEPS) unless the run is known to need more.
/// The step *floor* is not a parameter and deliberately so: it is `span * 1e-14`, and a caller who
/// set it from the span alone would get a floor below the resolution of its own clock whenever
/// `|t| >> span` — the case measured in `the_span_is_measured_from_the_epoch_not_added_to_it`.
/// Returns the state at `t1`.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] for a non-increasing span, a `max_steps` of zero, or
/// non-finite controls;
/// [`IntegrationError::StepUnderflow`] if a pass leaves the macro-step non-finite, below the floor,
/// or unable to make progress — the problem is too stiff or too rough for this non-stiff
/// extrapolation method; [`IntegrationError::StepLimit`] if the run needs more than `max_steps`
/// passes.
#[allow(clippy::needless_range_loop)] // the tableau loops index columns 0..j across N-vectors
pub fn integrate_gbs<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h0: f64,
    tol: Tolerance,
    max_steps: u32,
) -> Result<[f64; N], IntegrationError> {
    if !(t1 > t0 && h0 > 0.0 && max_steps > 0 && tol.atol.is_finite() && tol.rtol.is_finite()) {
        return Err(IntegrationError::BadArguments);
    }
    let span = t1 - t0;
    let min_step = span * 1e-14;
    let mut y = *y0;
    let mut t = t0;
    let mut h = h0.min(span);
    let mut k_opt = 4usize; // working order index (order 2·k_opt)
    let mut passes = 0u32;
    // Named because the loop and the invariant below both need it, and two copies of the same
    // endpoint expression is a place for a mutant to hide: change one and the other still agrees.
    let done = |t: f64| t >= t1 - 1e-12 * span;

    while !done(t) {
        if passes >= max_steps {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(passes),
            });
        }
        passes += 1;
        let t_before = t;
        if t + h > t1 {
            h = t1 - t;
        }
        // Captured after the endpoint clamp: the step this pass actually attempts is what the next
        // one has to beat, and the clamp is not the controller shrinking anything.
        let h_before = h;
        let kmax_try = (k_opt + 1).min(KMAX);
        let mut prev = [[0.0; N]; KMAX]; // T_{j-1,·}
        let mut accepted: Option<(usize, f64, [f64; N])> = None;
        let mut last_err = f64::INFINITY;

        for j in 1..=kmax_try {
            let s = modified_midpoint(sys, t, &y, h, 2 * j);
            let mut row = [[0.0; N]; KMAX]; // T_{j,·}; row[c] = T_{j,c+1}
            row[0] = s;
            for c in 1..j {
                let ratio = j as f64 / (j - c) as f64; // n_j / n_{j-c}
                let denom = ratio * ratio - 1.0;
                for i in 0..N {
                    row[c][i] = row[c - 1][i] + (row[c - 1][i] - prev[c - 1][i]) / denom;
                }
            }
            if j >= 2 {
                let mut errv = [0.0; N];
                for i in 0..N {
                    errv[i] = row[j - 1][i] - row[j - 2][i]; // T_{j,j} − T_{j,j-1}
                }
                last_err = tol.error_norm(&y, &row[j - 1], &errv);
                if last_err <= 1.0 {
                    accepted = Some((j, last_err, row[j - 1]));
                    break;
                }
            }
            prev = row;
        }

        if let Some((j, err, y_new)) = accepted {
            y = y_new;
            t += h;
            let expo = 1.0 / (2.0 * j as f64 - 1.0);
            let fac = if err > 0.0 {
                (SAFE * (1.0 / err).powf(expo)).clamp(FACMIN, FACMAX)
            } else {
                FACMAX
            };
            h *= fac;
            // Order control: converging below the working order means the order is too high; needing
            // the full window means raise it.
            if j < k_opt && k_opt > 2 {
                k_opt -= 1;
            } else if j >= k_opt && k_opt < KMAX - 1 {
                k_opt += 1;
            }
        } else {
            let expo = 1.0 / (2.0 * kmax_try as f64 - 1.0);
            let fac = (SAFE * (1.0 / last_err).powf(expo)).clamp(FACMIN, 0.7);
            h *= fac;
        }
        // What a pass has to leave behind, in place of the two one-sided `h < min_step` checks that
        // stood here. Both were blind in the same direction, and each of the three ways out of this
        // loop went through that blindness:
        //
        // * `<` is false for NaN, and `f64::clamp` returns NaN when its input is NaN — so one NaN
        //   out of the force model becomes a NaN `fac`, a NaN `h`, and a step that walks past a
        //   `h < min_step` guard untouched, leaving a loop with nothing left in it that can end.
        // * A rejected pass moves nothing but `h`, so a `h` that failed to shrink is the same pass
        //   again, forever. The floor cannot see it while `h` stays above the floor.
        // * The floor bounds how *small* the step gets, never how *many* are taken; at `1e-14` of
        //   the span it permits 1e14 of them, which is the unbounded-count hole the sibling drivers
        //   were given a ceiling for.
        //
        // Consecutive rejections need no separate bound: the clamp above caps `fac` at 0.7, so the
        // floor is reached in at most `ln(1e-14)/ln(0.7)` ≈ 90 of them.
        //
        // Asked only of a pass that has another one after it: a run that has just reached `t1` is
        // allowed to leave behind whatever step it likes, and the old guard exempted it the same
        // way.
        let progressed = t > t_before || h < h_before;
        let usable = h.is_finite() && h >= min_step && progressed;
        if !(done(t) || usable) {
            return Err(IntegrationError::StepUnderflow { t });
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
    use crate::driver::DEFAULT_MAX_STEPS;

    use super::*;

    // Measured, not derived: the evaluation counts the current controller spends on the problems
    // below. They are checked for equality, not to a band. A +-10% band was the first attempt and
    // it left 28 controller mutants alive -- the exponent `1/(2j-1)`, the `k_opt` raise and lower
    // rules, the rejection branch's own exponent and its 0.7 clamp all move the trajectory by less
    // than a tenth of the total work. The controller is deterministic f64 arithmetic over a fixed
    // problem, so equality is the honest precision of the measurement; if a toolchain bump moves
    // `powf` by an ulp and shifts one of these by one or two, re-measure and say so in the commit,
    // do not widen the band back.
    const EXP_LOOSE: u32 = 135;
    const EXP_TIGHT: u32 = 155;
    const ORBIT_LOOSE: u32 = 711;
    const ORBIT_TIGHT: u32 = 2160;
    const POLE_REJECT: u32 = 6482; // the pole run, which is all rejection branch and step floor
    // The pole again, from a nonzero epoch over a span of 4. This one was **106 809** before the
    // progress invariant landed, and the drop is the invariant working rather than a measurement
    // that moved: at `t0 = 1000` the span-relative floor is `4e-14` while `ulp(1002)` is `1.137e-13`,
    // nearly three times larger, so a step can be too small to move the clock and still sit
    // comfortably above the floor. Measured at the stop: pass 2602 was *accepted* with
    // `h = 5.683e-14`, exactly half an ulp, which `t + h` rounded straight back to `t`; the
    // controller then grew it by `FACMAX` to `2.273e-13` and the run spent another 71 020
    // evaluations oscillating around that wall before `h` finally fell under `4e-14`. None of that
    // work moved `t`.
    const SHIFTED_POLE: u32 = 35_789;
    const CEILING: u32 = 3661; // the quadrature run, whose accepted orders spread across 5..8

    /// y' = y on [0,1]; exact solution e. High order must hit e to near machine precision.
    struct Exponential;
    impl DynamicalSystem<1> for Exponential {
        fn derivative(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
            dydt[0] = y[0];
        }
    }

    #[test]
    fn exponential_reaches_e_to_machine_precision() {
        let y = integrate_gbs(
            &Exponential,
            0.0,
            &[1.0],
            1.0,
            0.1,
            Tolerance::new(1e-13, 1e-13),
            DEFAULT_MAX_STEPS,
        )
        .unwrap();
        let err = (y[0] - core::f64::consts::E).abs();
        println!("GBS y(1) error vs e = {err:.3e}");
        assert!(err < 1e-11, "exp error {err:.3e}");
    }

    /// Planar two-body: a circular orbit must return to its start after one period.
    struct TwoBody {
        mu: f64,
    }
    impl DynamicalSystem<4> for TwoBody {
        fn derivative(&self, _t: f64, y: &[f64; 4], dydt: &mut [f64; 4]) {
            let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
            let r3 = r * r * r;
            dydt[0] = y[2];
            dydt[1] = y[3];
            dydt[2] = -self.mu * y[0] / r3;
            dydt[3] = -self.mu * y[1] / r3;
        }
    }

    #[test]
    fn two_body_circular_orbit_closes() {
        let mu = 398_600.441_8; // km³/s², Earth
        let r = 7000.0_f64; // km
        let v = (mu / r).sqrt();
        let period = core::f64::consts::TAU * (r * r * r / mu).sqrt();
        let sys = TwoBody { mu };
        let y0 = [r, 0.0, 0.0, v];
        let y = integrate_gbs(
            &sys,
            0.0,
            &y0,
            period,
            period / 100.0,
            Tolerance::new(1e-12, 1e-12),
            DEFAULT_MAX_STEPS,
        )
        .unwrap();
        let dr = ((y[0] - r).powi(2) + y[1].powi(2)).sqrt();
        let dv = ((y[2]).powi(2) + (y[3] - v).powi(2)).sqrt();
        println!("GBS one-period closure: |Δr| = {dr:.3e} km, |Δv| = {dv:.3e} km/s");
        assert!(dr < 1e-6, "position drift {dr:.3e} km");
        assert!(dv < 1e-9, "velocity drift {dv:.3e} km/s");
    }

    /// Cross-check against the independent DOP853 driver on the same orbit — two unrelated methods must
    /// agree to well below their tolerances.
    #[test]
    fn agrees_with_dop853() {
        let mu = 398_600.441_8;
        let sys = TwoBody { mu };
        let y0 = [7000.0, 0.0, 0.0, (mu / 7000.0_f64).sqrt() * 1.1]; // eccentric
        let t1 = 6000.0;
        let tol = Tolerance::new(1e-11, 1e-11);
        let y_gbs = integrate_gbs(&sys, 0.0, &y0, t1, 60.0, tol, DEFAULT_MAX_STEPS).unwrap();
        let traj =
            crate::driver::integrate_dop853(&sys, 0.0, &y0, t1, 60.0, Controls::new(tol, 1e-6))
                .unwrap();
        let y_ref = traj.final_state();
        let diff = (0..4)
            .map(|i| (y_gbs[i] - y_ref[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("GBS vs DOP853 final-state diff = {diff:.3e}");
        assert!(diff < 1e-5, "GBS and DOP853 disagree by {diff:.3e}");
    }

    /// Wraps a system and counts the derivative evaluations the controller asks for.
    struct Counted<'a, const N: usize, S: DynamicalSystem<N>> {
        inner: &'a S,
        calls: core::cell::Cell<u32>,
    }

    impl<const N: usize, S: DynamicalSystem<N>> DynamicalSystem<N> for Counted<'_, N, S> {
        fn derivative(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]) {
            self.calls.set(self.calls.get() + 1);
            self.inner.derivative(t, y, dydt);
        }
    }

    /// Nothing above can see the step and order controller. Extrapolation converges to the tolerance
    /// whatever `h` and `k_opt` do, so `SAFE`, the `FACMIN`/`FACMAX` clamps, the exponent
    /// `1/(2j-1)` and the whole order-raising rule are free: 63 of this module's mutants live in
    /// `integrate_gbs` and every one of them survived. What they change is the price, and the price
    /// is one `Cell` away from being an assertion.
    #[test]
    fn the_controller_costs_what_its_order_and_step_rules_earn() {
        // (system, span, h0, tolerance, evaluations)
        let mu = 398_600.441_8;
        let r = 7000.0_f64;
        let v = (mu / r).sqrt();
        let period = core::f64::consts::TAU * (r * r * r / mu).sqrt();

        let exp_cost = |tol: f64| {
            let sys = Counted {
                inner: &Exponential,
                calls: core::cell::Cell::new(0),
            };
            integrate_gbs(
                &sys,
                0.0,
                &[1.0],
                1.0,
                0.1,
                Tolerance::new(tol, tol),
                DEFAULT_MAX_STEPS,
            )
            .unwrap();
            sys.calls.get()
        };
        let orbit_cost = |tol: f64| {
            let inner = TwoBody { mu };
            let sys = Counted {
                inner: &inner,
                calls: core::cell::Cell::new(0),
            };
            integrate_gbs(
                &sys,
                0.0,
                &[r, 0.0, 0.0, v],
                period,
                period / 100.0,
                Tolerance::new(tol, tol),
                DEFAULT_MAX_STEPS,
            )
            .unwrap();
            sys.calls.get()
        };

        let pole_cost = || {
            struct Pole;
            impl DynamicalSystem<1> for Pole {
                fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                    dydt[0] = 1.0 / (t - 0.5);
                }
            }
            let sys = Counted {
                inner: &Pole,
                calls: core::cell::Cell::new(0),
            };
            let _ = integrate_gbs(
                &sys,
                0.0,
                &[0.0],
                1.0,
                0.1,
                Tolerance::new(1e-10, 1e-10),
                DEFAULT_MAX_STEPS,
            );
            sys.calls.get()
        };

        let (loose_exp, tight_exp) = (exp_cost(1e-8), exp_cost(1e-13));
        let (loose_orbit, tight_orbit) = (orbit_cost(1e-8), orbit_cost(1e-12));
        let pole = pole_cost();
        println!(
            "GBS evaluations: exp {loose_exp}/{tight_exp}, orbit {loose_orbit}/{tight_orbit}, pole {pole}"
        );
        for (label, got, want) in [
            ("exp 1e-8", loose_exp, EXP_LOOSE),
            ("exp 1e-13", tight_exp, EXP_TIGHT),
            ("orbit 1e-8", loose_orbit, ORBIT_LOOSE),
            ("orbit 1e-12", tight_orbit, ORBIT_TIGHT),
            ("pole 1e-10", pole, POLE_REJECT),
        ] {
            assert_eq!(got, want, "{label}: {got} evaluations, expected {want}");
        }
        // Physics rather than a fitted number: five more digits cost more work, on both problems.
        assert!(
            tight_exp > loose_exp,
            "tightening the tolerance cost nothing on y' = y"
        );
        assert!(
            tight_orbit > loose_orbit,
            "tightening the tolerance cost nothing on the orbit"
        );
    }

    /// A right-hand side with a pole inside the span: no macro-step converges near it, the controller
    /// shrinks until the floor, and the run must come back as [`IntegrationError::StepUnderflow`] at
    /// roughly the pole rather than grind on. That arm -- the whole rejection branch and its own
    /// exponent and clamp -- is otherwise never taken.
    #[test]
    fn a_pole_inside_the_span_underflows_at_the_pole() {
        struct Pole;
        impl DynamicalSystem<1> for Pole {
            fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0 / (t - 0.5);
            }
        }
        let out = integrate_gbs(
            &Pole,
            0.0,
            &[0.0],
            1.0,
            0.1,
            Tolerance::new(1e-10, 1e-10),
            DEFAULT_MAX_STEPS,
        );
        match out {
            Err(IntegrationError::StepUnderflow { t }) => {
                assert!(
                    (t - 0.5).abs() < 1e-3,
                    "underflowed at {t}, not at the pole"
                );
            }
            other => panic!("expected StepUnderflow at the pole, got {other:?}"),
        }
    }

    /// A right-hand side that stops being a number partway through the span. This is what a domain
    /// error in a force model looks like from the controller's side — an atmosphere table asked for
    /// a height it does not tabulate, a `sqrt` of a quantity that went negative, a division by a
    /// radius that reached zero. It arrives as a NaN, not as a large number, and the two are
    /// nothing alike here.
    ///
    /// A NaN error norm makes `fac` NaN, and `f64::clamp` returns NaN for a NaN input rather than
    /// clamping it into the range, so the step becomes NaN as well. `h < min_step` is false for a
    /// NaN `h` — every comparison against NaN is — so the guard that stood here passed it through,
    /// and nothing else in the loop could end the run: a rejected pass does not move `t`, so the
    /// endpoint test kept failing on the same instant for as long as the process was left alive.
    #[test]
    fn a_right_hand_side_that_goes_nan_stops_the_run() {
        struct Domain;
        impl DynamicalSystem<1> for Domain {
            fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = if t > 0.5 { f64::NAN } else { 1.0 };
            }
        }
        match integrate_gbs(
            &Domain,
            0.0,
            &[0.0],
            1.0,
            0.1,
            Tolerance::new(1e-10, 1e-10),
            DEFAULT_MAX_STEPS,
        ) {
            Err(IntegrationError::StepUnderflow { t }) => {
                assert!(
                    (0.4..=0.6).contains(&t),
                    "stopped at {t}, not where the right-hand side stopped being a number"
                );
            }
            other => panic!("expected StepUnderflow at the domain edge, got {other:?}"),
        }
    }

    /// The half of the guard no step floor can cover: every pass of this run is healthy — the step
    /// is nowhere near the floor and each one advances the clock — and there are merely more of them
    /// than the caller allowed. A floor is blind to that by construction, because nothing ever
    /// approaches the floor.
    ///
    /// One period of the circular orbit at 1e-12 costs 169 passes, so a ceiling of five is reached
    /// early and cheaply, and the same run with the default ceiling still has to succeed — without
    /// that half the test would pass just as well against a driver that refused everything.
    #[test]
    fn gbs_stops_at_the_step_ceiling_it_is_given() {
        let mu = 398_600.441_8;
        let r = 7000.0_f64;
        let v = (mu / r).sqrt();
        let period = core::f64::consts::TAU * (r * r * r / mu).sqrt();
        let sys = TwoBody { mu };
        let y0 = [r, 0.0, 0.0, v];
        let tol = Tolerance::new(1e-12, 1e-12);

        match integrate_gbs(&sys, 0.0, &y0, period, period / 100.0, tol, 5) {
            // Exactly the ceiling, not merely past it. The count is an integer well inside f64's
            // exact range, so equality is the honest comparison and a tolerance would accept a
            // guard that fired a pass late.
            Err(IntegrationError::StepLimit { steps }) => {
                assert!(
                    steps.to_bits() == 5.0_f64.to_bits(),
                    "stopped after {steps} passes, not at the ceiling"
                );
            }
            other => panic!("expected StepLimit at five passes, got {other:?}"),
        }
        assert!(
            integrate_gbs(
                &sys,
                0.0,
                &y0,
                period,
                period / 100.0,
                tol,
                DEFAULT_MAX_STEPS
            )
            .is_ok(),
            "the same orbit must still integrate under the default ceiling"
        );
        assert_eq!(
            integrate_gbs(&sys, 0.0, &y0, period, period / 100.0, tol, 0),
            Err(IntegrationError::BadArguments),
            "a ceiling of zero admits no run at all, so it is an argument error"
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        assert_eq!(
            integrate_gbs(
                &Exponential,
                1.0,
                &[1.0],
                0.0,
                0.1,
                Tolerance::default(),
                DEFAULT_MAX_STEPS
            ),
            Err(IntegrationError::BadArguments)
        );
        // The guard is `!(t1 > t0 && h0 > 0.0 && …)`, and each `>` has a `>=` mutant that only an
        // argument sitting exactly on the boundary can see -- `1.0 → 0.0` above passes either way.
        // Mutated, an empty span skips the loop and reports the initial state as a success, and a
        // zero step is worse: `t + 0.0 > t1` is false forever, so the loop never advances.
        assert_eq!(
            integrate_gbs(
                &Exponential,
                1.0,
                &[1.0],
                1.0,
                0.1,
                Tolerance::default(),
                DEFAULT_MAX_STEPS
            ),
            Err(IntegrationError::BadArguments)
        );
        assert_eq!(
            integrate_gbs(
                &Exponential,
                0.0,
                &[1.0],
                1.0,
                0.0,
                Tolerance::default(),
                DEFAULT_MAX_STEPS
            ),
            Err(IntegrationError::BadArguments)
        );
        for bad in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                integrate_gbs(
                    &Exponential,
                    0.0,
                    &[1.0],
                    1.0,
                    0.1,
                    Tolerance::new(bad, 1e-9),
                    DEFAULT_MAX_STEPS
                ),
                Err(IntegrationError::BadArguments)
            );
            assert_eq!(
                integrate_gbs(
                    &Exponential,
                    0.0,
                    &[1.0],
                    1.0,
                    0.1,
                    Tolerance::new(1e-9, bad),
                    DEFAULT_MAX_STEPS
                ),
                Err(IntegrationError::BadArguments)
            );
        }
    }

    /// `y' = cos(t)`, so `y(t) = sin(t) + c` — the only system in this file whose right-hand side
    /// reads `t` at all.
    ///
    /// `Exponential`, `TwoBody` and the cost fixtures are all autonomous: they take `_t` and ignore
    /// it. That makes `modified_midpoint`'s two stage times, `t0 + m·h` and `t0 + H`, free
    /// parameters — replace either `+` with `-` or `*` and every one of them still integrates the
    /// same trajectory. `Pole` does read `t`, but only to be asked whether it underflows near 0.5.
    struct Quadrature;
    impl DynamicalSystem<1> for Quadrature {
        fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
            dydt[0] = t.cos();
        }
    }

    #[test]
    fn a_non_autonomous_right_hand_side_is_sampled_at_the_stage_times() {
        // `t0 = 0.7`, not zero: at `t0 = 0` the three mutants of `t0 + m·h` collapse onto each
        // other, because `0 + x`, `0 - x` and `0 * x` differ only in a sign or a zero the fixture
        // supplied. The span crosses π/2 so the integrand changes sign inside it — a quadrature
        // that samples the wrong abscissae cannot land on the right answer by symmetry.
        let (t0, t1) = (0.7_f64, 3.9_f64);
        let y = integrate_gbs(
            &Quadrature,
            t0,
            &[t0.sin()],
            t1,
            0.4,
            Tolerance::new(1e-13, 1e-13),
            DEFAULT_MAX_STEPS,
        )
        .unwrap();
        let err = (y[0] - t1.sin()).abs();
        println!("GBS ∫cos over [{t0}, {t1}]: error vs sin(t₁) = {err:.3e}");
        assert!(err < 1e-12, "quadrature error {err:.3e}");
    }

    /// The driver's own bookkeeping — `span`, the step floor derived from it, and the underflow exit
    /// that guards an *accepted* step — is invisible to every test above, because a problem that
    /// converges never touches any of it. Reaching it needs an initial step guess that is already
    /// below the floor, which is a legitimate call: `h0` is documented as a guess, and `span·1e-14`
    /// is not a number a caller can be expected to know.
    ///
    /// `t0 = 100` and `span = 4` are both deliberate. At `t0 = 0` the span is `t1` either way, so
    /// `t1 - t0` and `t1 + t0` agree; at `span = 1` the floor and the loop's slack are the same
    /// number whether `1e-12` multiplies or divides by it.
    #[test]
    // Exact equality is the assertion: `t` must be the *unmoved* start time, and a tolerance would
    // accept a step that did advance it.
    #[allow(clippy::float_cmp)]
    fn a_step_guess_below_the_floor_underflows_before_it_can_grow() {
        // min_step = 4·1e-14; h0 = 1e-15 is under it, and one FACMAX growth (×4) does not escape.
        let out = integrate_gbs(
            &Exponential,
            100.0,
            &[1.0],
            104.0,
            1e-15,
            Tolerance::new(1e-10, 1e-10),
            DEFAULT_MAX_STEPS,
        );
        match out {
            Err(IntegrationError::StepUnderflow { t }) => {
                // 1e-15 is below ulp(100) = 1.42e-14, so the step cannot move `t` at all: the
                // accumulator, not the tolerance, is what has run out.
                assert_eq!(t, 100.0, "underflowed at {t}, not at the first step");
            }
            other => panic!("expected StepUnderflow at the first step, got {other:?}"),
        }
    }

    /// The same pole as [`a_pole_inside_the_span_underflows_at_the_pole`], moved off zero. That test
    /// cannot see `span = t1 - t0` become `t1 + t0`, because its `t0` is 0; here the two differ by a
    /// factor of 501, the step floor with them, and the number of rejections needed to reach the
    /// floor by that factor's logarithm. The count is the observable — the returned `t` is the same
    /// either way, since a rejected step does not advance it.
    #[test]
    fn the_span_is_measured_from_the_epoch_not_added_to_it() {
        struct ShiftedPole;
        impl DynamicalSystem<1> for ShiftedPole {
            fn derivative(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0 / (t - 1002.0);
            }
        }
        let sys = Counted {
            inner: &ShiftedPole,
            calls: core::cell::Cell::new(0),
        };
        let out = integrate_gbs(
            &sys,
            1000.0,
            &[0.0],
            1004.0,
            0.4,
            Tolerance::new(1e-10, 1e-10),
            DEFAULT_MAX_STEPS,
        );
        let calls = sys.calls.get();
        println!("GBS shifted pole: {calls} evaluations, outcome {out:?}");
        match out {
            Err(IntegrationError::StepUnderflow { t }) => {
                assert!(
                    (t - 1002.0).abs() < 1e-3,
                    "underflowed at {t}, not at the pole"
                );
            }
            other => panic!("expected StepUnderflow at the pole, got {other:?}"),
        }
        assert_eq!(
            calls, SHIFTED_POLE,
            "{calls} evaluations, expected {SHIFTED_POLE}"
        );
    }

    /// Where the working order stops climbing — `KMAX - 1`, and the `&&` that pairs the ceiling with
    /// the raise test — is invisible to every fixture above, and not for the obvious reason. The
    /// ceiling is reached: `exp` at 1e-13 already drives `k_opt` to 7, and twenty orbits at 1e-15
    /// accept at `j = 7` on 349 of 403 steps. But `k_opt` enters the arithmetic only through
    /// `kmax_try = (k_opt + 1).min(KMAX)`, which saturates at `KMAX` for every `k_opt >= KMAX - 1`,
    /// so a controller pinned at 7 and one allowed to sit at 8 or 9 do exactly the same work. The
    /// ceiling becomes observable one step *after* the order comes back down: the raised controller
    /// lowers to 7 where the real one lowers to 6, and only then does `kmax_try` differ. That needs
    /// a run that ratchets *both* ways often, which means a mixed accepted-order distribution rather
    /// than a high one.
    ///
    /// Chosen by measurement, not by argument — with the ceiling opened from `KMAX - 1` to `KMAX`,
    /// the two runs that merely sit at the top (`orbit 1e-12`, twenty orbits at 1e-15) cost exactly
    /// the same 24 036 evaluations, while the three with a spread of accepted orders separate:
    /// 21 927 → 22 593, 60 083 → 63 528, and this one, 3661 → 2973. It is the cheapest of the three
    /// and has the widest margin.
    #[test]
    fn the_working_order_stops_at_its_ceiling() {
        let sys = Counted {
            inner: &Quadrature,
            calls: core::cell::Cell::new(0),
        };
        let (t0, t1) = (0.7_f64, 40.7_f64);
        let y = integrate_gbs(
            &sys,
            t0,
            &[t0.sin()],
            t1,
            4.0,
            Tolerance::new(1e-16, 1e-16),
            DEFAULT_MAX_STEPS,
        )
        .unwrap();
        let calls = sys.calls.get();
        let err = (y[0] - t1.sin()).abs();
        println!("GBS ceiling run: {calls} evaluations, error vs sin(t₁) = {err:.3e}");
        assert!(err < 1e-12, "quadrature error {err:.3e}");
        assert_eq!(calls, CEILING, "{calls} evaluations, expected {CEILING}");
    }

    /// The endpoint tolerance is `1e-12` *of the span*, and every fixture above is on a span near 1,
    /// where a relative tolerance and an absolute one are the same number. On a microsecond span the
    /// two part company by twelve orders of magnitude: `1e-12 · span` is `1e-18`, while `1e-12 / span`
    /// is `1e-6` — the whole interval. A driver holding the second form has `done(t0)` true on entry,
    /// takes no step at all, and returns the initial condition as the answer.
    ///
    /// A microsecond of an exponential is a legitimate call, not a degenerate one: `y(1e-6)` differs
    /// from `y(0)` in the sixth digit, which is 1e6 times the tolerance the run is asked for.
    #[test]
    fn a_span_far_below_one_is_integrated_rather_than_declared_finished() {
        let t1 = 1e-6;
        let y = integrate_gbs(
            &Exponential,
            0.0,
            &[1.0],
            t1,
            t1 / 4.0,
            Tolerance::new(1e-13, 1e-13),
            DEFAULT_MAX_STEPS,
        )
        .unwrap();
        let err = (y[0] - libm::exp(t1)).abs();
        println!("GBS microsecond span: y = {:.17e}, error {err:.3e}", y[0]);
        assert!(err < 1e-15, "short-span error {err:.3e}");
    }
}
