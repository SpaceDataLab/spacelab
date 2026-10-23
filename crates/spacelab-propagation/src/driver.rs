//! Adaptive-step driver with Hairer's PI controller, generic over the integration method.
//!
//! The controller is the PI step-size law of Hairer–Nørsett–Wanner (`β = 0.04`, safety `0.9`, step
//! ratio bounds `[0.1, 5]`), which damps the oscillation a pure I-controller shows. The method — its
//! Butcher tableau, its error estimate, its dense output — is a [`Method`], so `DOPRI5` and `DOP853`
//! share this one driver. That is the Bridge the architecture promised: method and step controller
//! vary independently.
//!
//! The result is a [`Trajectory`]: every accepted step's dense interpolant, so the state at any `t`
//! in `[t0, t1]` is available without re-integrating — what [`spacelab_coverage::find_events`] needs.

// The integrator signatures carry the full IVP (sys, span, y0, h0) plus the controller's limits,
// which travel together as [`Controls`]. Dense-output loops index by component.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::vec::Vec;

use crate::{DenseOutput, DynamicalSystem, Tolerance};

/// One trial step of a method: the new state, its interpolant, and the scaled error norm (accept
/// when `≤ 1`). The method owns its error-norm computation, because `DOP853`'s combined 5th/3rd-order
/// estimate is not a plain RMS of a single error vector.
pub struct MethodStep<const N: usize, D> {
    /// The proposed state at `t + h`.
    pub y_next: [f64; N],
    /// The interpolant valid on `[t, t + h]`.
    pub dense: D,
    /// The scaled error norm; the step is accepted when this is `≤ 1`.
    pub error_norm: f64,
}

/// An embedded, dense-output integration method plugged into [`integrate`].
pub trait Method<const N: usize> {
    /// The method's dense-output type.
    type Dense: DenseOutput<N>;

    /// The base step-size exponent `1 / (error_estimator_order + 1)` — `0.2` for `DOPRI5`
    /// (estimator order 4), `0.125` for `DOP853` (estimator order 7).
    fn step_exponent(&self) -> f64;

    /// The PI-controller memory coefficient `β`. `DOPRI5` uses Hairer's `0.04`; `DOP853` ships `β = 0`
    /// (a pure I-controller — its `dop853.f` and scipy both do), so the default is `0`. A method
    /// tuned for a different `β` overrides this rather than borrowing another method's tuning.
    fn beta(&self) -> f64 {
        0.0
    }

    /// Take one trial step, returning the new state, its interpolant, and the scaled error norm.
    fn try_step(
        &self,
        sys: &impl DynamicalSystem<N>,
        t: f64,
        y: &[f64; N],
        h: f64,
        tol: Tolerance,
    ) -> MethodStep<N, Self::Dense>;
}

/// Why an integration could not complete.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegrationError {
    /// The step size fell below `min_step` while still failing the tolerance — the problem is too
    /// stiff or too discontinuous for this non-stiff method. Reported, never silently accepted (the
    /// mistake `notes/15-integrators.md` §1 records in GMAT's shadow-boundary kludge).
    StepUnderflow {
        /// The time at which the step collapsed.
        t: f64,
    },
    /// `t1` is not strictly after `t0`, or a tolerance/step argument was not finite and positive.
    BadArguments,
    /// An implicit method's stage solve failed: Newton did not converge within its cap, or the
    /// linear system was singular. The step is too large for the stage system to resolve. Reported
    /// rather than accepting an unconverged stage, which would silently drop the method's order.
    StageSolveFailed {
        /// The time at which the stage solve failed.
        t: f64,
    },
    /// The run wants more steps than it is allowed. A fixed-step method asks up front, and is
    /// refused before the loop is entered against [`MAX_STEPS`], because the loop that would run is
    /// not one a caller can interrupt. An adaptive run cannot be asked in advance — its step count
    /// is the controller's answer, not the caller's — so it is counted instead and stopped at
    /// [`Controls::max_steps`].
    StepLimit {
        /// How many steps were asked for, or taken. An `f64` because the ask can exceed
        /// `usize::MAX`, which is the case this variant exists for.
        steps: f64,
    },
}

/// The most equal steps any fixed-step integrator here will take over one span.
///
/// A fixed-step run costs one step per `h`, so a cap on the count bounds both the wall clock and —
/// for the dense forms, which push one interpolant per step — the memory. The largest count this
/// workspace's own tests reach is 15 000, measured over the 890 calls a `cargo test --workspace`
/// makes; the largest a plausible caller reaches is a year of LEO at a one-minute step, 5.3e5. Ten
/// million is nineteen times that, and 1.8e12 times *below* the `usize::MAX` the saturating cast
/// used to hand the loop. It is a ceiling on a runaway, not a budget to integrate against.
pub const MAX_STEPS: usize = 10_000_000;

/// How many equal steps of nominal size `h` span `[t0, t1]`, at least `floor` of them.
///
/// Not `((t1 - t0) / h).round().max(floor) as usize`, which every integrator in this crate used to
/// write inline. A float-to-integer `as` cast in Rust *saturates*: `f64::INFINITY as usize` is
/// `usize::MAX`, silently. So `h = 0`, or a `t0` a mutation turned into a divisor, or simply a span
/// and a step whose ratio exceeds `1.8e19`, produced a loop of `usize::MAX` iterations — each one
/// cheap, the whole thing unbounded, and nothing to distinguish it from a hang. Two mutants of
/// `integrate_whfast` were detected only by *not terminating*, which is what pointed at this.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the ratio is not finite, and
/// [`IntegrationError::StepLimit`] if it exceeds [`MAX_STEPS`].
pub(crate) fn step_count(
    t0: f64,
    t1: f64,
    h: f64,
    floor: usize,
) -> Result<usize, IntegrationError> {
    let asked = (t1 - t0) / h;
    // A negative ratio means `t1 > t0` or `h > 0` was violated, which every caller checks before
    // reaching here. Refusing it rather than clamping keeps the cast below provably in range.
    if !(asked.is_finite() && asked >= 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    // Compared as an `f64` on purpose: `MAX_STEPS` is exact in one, and rounding first would put
    // the very values this refuses back through the saturating cast it exists to avoid.
    let asked = asked.round();
    if asked > MAX_STEPS as f64 {
        return Err(IntegrationError::StepLimit { steps: asked });
    }
    // Both guards above are what make this cast exact rather than saturating: `asked` is finite,
    // not negative, and no greater than `MAX_STEPS`.
    #[allow(clippy::cast_sign_loss)]
    let steps = asked as usize;
    Ok(steps.max(floor))
}

/// A dense solution over `[t0, t1]`: the accepted steps' interpolants, evaluable at any interior `t`.
#[derive(Debug, Clone)]
pub struct Trajectory<const N: usize, D> {
    segments: Vec<D>,
    ends: Vec<f64>, // ends[k] = end time of segment k, ascending — the binary-search key
    t0: f64,
    t1: f64,
    y_final: [f64; N],
    accepted: u32,
    rejected: u32,
}

impl<const N: usize, D: DenseOutput<N>> Trajectory<N, D> {
    /// Assemble a trajectory from a method's own step segments — the door for the fixed-step and
    /// self-adaptive integrators (multistep, Gauss–Jackson, Radau, IAS15, Taylor, …) that do not run
    /// through the PI-controller [`integrate`] but still expose dense output for the event detector.
    ///
    /// `ends[k]` is the end time of `segments[k]`, ascending; `segments` must be non-empty and the
    /// same length as `ends`. `accepted`/`rejected` are recorded for [`Self::step_counts`].
    ///
    /// # Panics
    ///
    /// If `segments` is empty or `segments.len() != ends.len()`.
    #[must_use]
    pub fn from_segments(
        t0: f64,
        t1: f64,
        y_final: [f64; N],
        segments: Vec<D>,
        ends: Vec<f64>,
        accepted: u32,
        rejected: u32,
    ) -> Self {
        assert!(
            !segments.is_empty() && segments.len() == ends.len(),
            "segments/ends mismatch"
        );
        Self {
            segments,
            ends,
            t0,
            t1,
            y_final,
            accepted,
            rejected,
        }
    }

    /// The state at time `t`, interpolated. Outside `[t0, t1]` this extrapolates the nearest segment.
    #[must_use]
    pub fn eval(&self, t: f64) -> [f64; N] {
        let k = match self
            .ends
            .binary_search_by(|e| e.partial_cmp(&t).unwrap_or(core::cmp::Ordering::Less))
        {
            Ok(i) | Err(i) => i.min(self.segments.len() - 1),
        };
        let mut out = [0.0; N];
        self.segments[k].eval(t, &mut out);
        out
    }

    /// The final state at `t1`.
    #[must_use]
    pub fn final_state(&self) -> [f64; N] {
        self.y_final
    }

    /// The integration span.
    #[must_use]
    pub fn span(&self) -> (f64, f64) {
        (self.t0, self.t1)
    }

    /// Accepted and rejected step counts — a cheap window on how hard the problem was.
    #[must_use]
    pub fn step_counts(&self) -> (u32, u32) {
        (self.accepted, self.rejected)
    }

    /// The end time of every accepted step, ascending — the step-size history the controller
    /// actually chose, since `ends[k] - ends[k-1]` is the step it took.
    ///
    /// [`Self::step_counts`] says how many steps there were; this says where they were. That is
    /// what distinguishes a controller that settled from one that oscillated between a large step
    /// and a rejection, and the two are indistinguishable from the counts alone.
    #[must_use]
    pub fn step_ends(&self) -> &[f64] {
        &self.ends
    }
}

/// Cubic-Hermite dense output over one step `[t0, t1]`, from the state and its derivative at both
/// ends. It is `C¹` and third-order accurate — the uniform interpolant for the fixed-step methods
/// that have no cheaper high-order form of their own. For a method whose own step is short (the fixed
/// step is set by accuracy, so it is), the interpolation error over one step is far below the state
/// error, which is what the event detector needs to bracket a sign change and seed Brent.
#[derive(Debug, Clone, Copy)]
pub struct HermiteDense<const N: usize> {
    t0: f64,
    h: f64,
    y0: [f64; N],
    y1: [f64; N],
    f0: [f64; N],
    f1: [f64; N],
}

impl<const N: usize> HermiteDense<N> {
    /// Build the interpolant from the endpoints `(t0, y0, f0)` and `(t1, y1, f1)`, where `f = dy/dt`.
    #[must_use]
    pub fn new(t0: f64, t1: f64, y0: [f64; N], f0: [f64; N], y1: [f64; N], f1: [f64; N]) -> Self {
        Self {
            t0,
            h: t1 - t0,
            y0,
            y1,
            f0,
            f1,
        }
    }
}

impl<const N: usize> DenseOutput<N> for HermiteDense<N> {
    fn eval(&self, t: f64, out: &mut [f64; N]) {
        let s = if self.h == 0.0 {
            0.0
        } else {
            (t - self.t0) / self.h
        };
        let s2 = s * s;
        let s3 = s2 * s;
        // Hermite basis on [0,1].
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        for i in 0..N {
            out[i] = h00 * self.y0[i]
                + h10 * self.h * self.f0[i]
                + h01 * self.y1[i]
                + h11 * self.h * self.f1[i];
        }
    }
}

/// Drive a stateless fixed-step method over `n_steps` and collect a cubic-[`HermiteDense`] trajectory.
///
/// `advance(t, y, h)` takes one step from `(t, y)`; `deriv(t, y)` is `dy/dt` (reused across the
/// segment boundary, so each node's derivative is computed once). For a one-step method whose step is
/// a pure function of `(t, y)` — Radau, Rosenbrock — this is the whole dense driver. Multistep and
/// compensated-state methods (Adams, Gauss–Jackson, symplectic) carry history and build their
/// segments in their own loops instead.
///
/// # Errors
///
/// Whatever `advance` returns on a failed step.
pub fn integrate_fixed_hermite<const N: usize>(
    t0: f64,
    t1: f64,
    y0: &[f64; N],
    n_steps: usize,
    mut deriv: impl FnMut(f64, &[f64; N]) -> [f64; N],
    mut advance: impl FnMut(f64, &[f64; N], f64) -> Result<[f64; N], IntegrationError>,
) -> Result<Trajectory<N, HermiteDense<N>>, IntegrationError> {
    let step = (t1 - t0) / n_steps as f64;
    let mut y = *y0;
    let mut t = t0;
    let mut f0 = deriv(t, &y);
    let mut segments = Vec::with_capacity(n_steps);
    let mut ends = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let y_new = advance(t, &y, step)?;
        let tn = t + step;
        let f1 = deriv(tn, &y_new);
        segments.push(HermiteDense::new(t, tn, y, f0, y_new, f1));
        y = y_new;
        f0 = f1;
        t = tn;
        ends.push(tn);
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

// Hairer PI-controller constants, shared by every method (the method supplies the exponent and β).
// The step is `h_new = h / fac`, `fac` clamped to `[FACC2, FACC1] = [0.1, 5]`, so `h` grows at most
// ×10 (`fac = 0.1`) and shrinks at most ÷5 (`fac = 5`).
const SAFE: f64 = 0.9;
const FACC1: f64 = 1.0 / 0.2; // = 5: the shrink limit, h_new >= h/5
const FACC2: f64 = 1.0 / 10.0; // = 0.1: the growth limit, h_new <= 10 h

/// The step ceiling [`Controls::new`] installs, and the fixed one
/// [`ias15`](crate::ias15::integrate_ias15) holds itself to: Hairer's `nmax` for `DOP853`, from his
/// `dop853.f`.
///
/// The largest count this workspace's own adaptive runs reach is **7803**, measured over a
/// `cargo test -p spacelab-propagation`; 52 runs pass 200 steps and the next-largest are 6210 and
/// 5320. IAS15's own runs are an order of magnitude below that again — 606 passes at the widest,
/// with 2006 of them at 61. A hundred thousand is 12.8 times the measured maximum, which is margin
/// enough that a legitimate integration never meets it, and — at the 112 051 steps/s this crate's
/// test profile sustains — is reached in about a second, which is what makes the ceiling a guard
/// rather than a formality.
pub const DEFAULT_MAX_STEPS: u32 = 100_000;

/// What the adaptive step controller is allowed to do: the tolerance it holds, how small a step may
/// get, and how many steps it may take.
///
/// The three travel together — every caller of [`integrate`] supplies all three and none is
/// meaningful without the others — so they are one value rather than three positional arguments.
///
/// `min_step` and `max_steps` bound *different* failure modes and neither implies the other. The
/// floor catches a controller driven to a step it cannot represent progress with. The ceiling
/// catches a controller that settles at a step which is legal but far too small to finish: a
/// one-day DSST mean propagation whose healthy cost is 1160 steps was measured, under a corrupted
/// force model, holding `h ≈ 3e-4` against a `min_step` of `1e-6` and needing **6e8** steps — 200
/// hours, with one dense interpolant retained per accepted step. The floor cannot see that, because
/// the step never approaches it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Controls {
    /// The error tolerance the controller holds.
    pub tol: Tolerance,
    /// The step size below which the run is declared failed rather than left to crawl.
    pub min_step: f64,
    /// The most steps, accepted plus rejected, before the run is declared failed.
    pub max_steps: u32,
}

impl Controls {
    /// Controls holding `tol`, with a floor of `min_step` and the [`DEFAULT_MAX_STEPS`] ceiling.
    #[must_use]
    pub const fn new(tol: Tolerance, min_step: f64) -> Self {
        Self {
            tol,
            min_step,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    /// The same controls with a different step ceiling.
    #[must_use]
    pub const fn with_max_steps(self, max_steps: u32) -> Self {
        Self { max_steps, ..self }
    }
}

/// Integrate `sys` from `t0` to `t1` with `method`, adapting the step to `controls`.
///
/// `h0` is the initial step guess; the controller takes over immediately.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] for a non-increasing span or non-finite controls;
/// [`IntegrationError::StepUnderflow`] if the step collapses below [`Controls::min_step`];
/// [`IntegrationError::StepLimit`] if the run passes [`Controls::max_steps`].
pub fn integrate<const N: usize, M: Method<N>>(
    method: &M,
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h0: f64,
    controls: Controls,
) -> Result<Trajectory<N, M::Dense>, IntegrationError> {
    let Controls {
        tol,
        min_step,
        max_steps,
    } = controls;
    let arguments_ok =
        t1 > t0 && h0 > 0.0 && min_step > 0.0 && max_steps > 0 && tol.atol.is_finite();
    if !arguments_ok {
        return Err(IntegrationError::BadArguments);
    }

    let beta = method.beta();
    let expo1 = method.step_exponent() - 0.75 * beta;
    let mut t = t0;
    let mut y = *y0;
    let mut h = h0.min(t1 - t0);
    let mut facold: f64 = 1e-4;

    let mut segments = Vec::new();
    let mut ends = Vec::new();
    let mut accepted = 0u32;
    let mut rejected = 0u32;

    while t < t1 {
        if t + h > t1 {
            h = t1 - t;
        }
        let step = method.try_step(sys, t, &y, h, tol);
        // A non-finite error norm is not a small one. `err > 0.0` is false for NaN, which would put
        // it on the same branch as an exactly-zero error — the "the step was perfect, grow it"
        // branch — and `FACC1.min(fac11 / SAFE)` would then be 0.111, so a rejected step would come
        // back nine times *longer*. A force model returning NaN could therefore never drive `h` down
        // to the floor the underflow guard watches, and the only thing that stopped the run was the
        // step ceiling, 100 000 evaluations later. Mapped to the worst finite case instead: rejected,
        // and shrunk by the hard limit, which reaches the floor from `h = 60 s` in twelve steps.
        let err = if step.error_norm.is_nan() {
            f64::INFINITY
        } else {
            step.error_norm
        };

        let fac11 = if err > 0.0 { err.powf(expo1) } else { FACC2 };
        let fac = (fac11 / facold.powf(beta) / SAFE).clamp(FACC2, FACC1);
        let h_new = h / fac;

        if err <= 1.0 {
            segments.push(step.dense);
            t += h;
            ends.push(t);
            y = step.y_next;
            facold = err.max(1e-4);
            accepted += 1;
            h = h_new;
        } else {
            rejected += 1;
            h /= FACC1.min(fac11 / SAFE);
        }
        if h < min_step && t < t1 {
            return Err(IntegrationError::StepUnderflow { t });
        }
        // A step that stays above the floor never trips it, however many of them the run needs, and
        // the segment vector grows by one interpolant for every accepted one. Counted rather than
        // derived from `min_step`: an accepted step advances `t` by at least the floor, so
        // `span / min_step` is an upper bound no run can reach and would not bound anything.
        if accepted + rejected >= max_steps {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(accepted) + f64::from(rejected),
            });
        }
    }

    Ok(Trajectory {
        segments,
        ends,
        t0,
        t1,
        y_final: y,
        accepted,
        rejected,
    })
}

/// Convenience wrapper: integrate with [`crate::Dopri5`].
///
/// # Errors
///
/// As [`integrate`].
pub fn integrate_dopri5<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h0: f64,
    controls: Controls,
) -> Result<Trajectory<N, crate::dopri5::Dopri5Dense<N>>, IntegrationError> {
    integrate(&crate::Dopri5, sys, t0, y0, t1, h0, controls)
}

/// Convenience wrapper: integrate with [`crate::Dop853`].
///
/// # Errors
///
/// As [`integrate`].
pub fn integrate_dop853<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h0: f64,
    controls: Controls,
) -> Result<Trajectory<N, crate::dop853::Dop853Dense<N>>, IntegrationError> {
    integrate(&crate::Dop853, sys, t0, y0, t1, h0, controls)
}

#[cfg(test)]
// `expect` on a fixture the test itself just proved integrable is the right failure. The exact
// float comparison is the assertion: the reported count is an integer carried in an `f64`, and a
// tolerance would accept a ceiling that fired a step out.
#[allow(clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::{
        Controls, DEFAULT_MAX_STEPS, DynamicalSystem, IntegrationError, MAX_STEPS, Tolerance,
        integrate, step_count,
    };

    /// The three answers [`step_count`] can give, at the boundaries between them.
    ///
    /// The middle case is the one this function exists for. `((t1 - t0) / h).round() as usize` --
    /// what all ten fixed-step integrators here computed inline -- saturates rather than
    /// overflowing: `1e300 as usize` is `usize::MAX`, silently. What follows is 1.8e19 iterations
    /// of a cheap body, which no caller can interrupt and no test can tell from a hang. Two mutants
    /// of `integrate_whfast` were caught by exactly that non-termination, and a mutant caught by
    /// running forever is a defect the suite reported in the only way left to it.
    #[test]
    fn the_step_count_refuses_what_it_cannot_walk() {
        assert_eq!(step_count(0.0, 10.0, 1.0, 1), Ok(10));
        // The floor applies after the count, so a span shorter than one step still gets its
        // startup points -- the reason `abm` and `gj` pass `ORDER - 1` rather than 1.
        assert_eq!(step_count(0.0, 1e-9, 1.0, 7), Ok(7));

        let Err(IntegrationError::StepLimit { steps }) = step_count(0.0, 1.0, 1e-300, 1) else {
            panic!("an ask of 1e300 steps was not refused");
        };
        // Not `== 1e300`: neither `1e-300` nor its reciprocal is exact in binary, so the payload
        // reports the count actually asked for, not the decimal that produced it.
        assert!((steps / 1e300 - 1.0).abs() < 1e-15, "reported {steps:e}");
        // Exactly at the cap is allowed and one step past it is not, so the boundary is where the
        // constant says it is rather than one out.
        let h = 1.0 / MAX_STEPS as f64;
        assert_eq!(step_count(0.0, 1.0, h, 1), Ok(MAX_STEPS));
        assert!(matches!(
            step_count(0.0, 1.0 + h, h, 1),
            Err(IntegrationError::StepLimit { .. })
        ));

        // Below `1 / f64::MAX` the ratio is not a large step count but no step count at all, and
        // `f64::INFINITY as usize` saturates the same way a merely huge one does.
        assert_eq!(
            step_count(0.0, 1.0, f64::from_bits(1), 1),
            Err(IntegrationError::BadArguments)
        );
        assert_eq!(
            step_count(0.0, f64::NAN, 1.0, 1),
            Err(IntegrationError::BadArguments)
        );
        // A backward span reaches here only if a caller's own `t1 > t0` guard was weakened; it is
        // refused rather than clamped to the floor, which is what silently reversed the step.
        assert_eq!(
            step_count(10.0, 0.0, 1.0, 1),
            Err(IntegrationError::BadArguments)
        );
    }

    /// A stiff-looking oscillator: cheap, and its step is set by `ω` rather than by the span, so the
    /// run needs a step count the caller never chose.
    struct Oscillator;

    impl DynamicalSystem<2> for Oscillator {
        fn derivative(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
            const W: f64 = 1.0e3;
            dydt[0] = W * y[1];
            dydt[1] = -W * y[0];
        }
    }

    /// The ceiling bounds a run the floor cannot see.
    ///
    /// `min_step` is `1e-30` here, thirty orders below any step the controller actually takes, so
    /// [`IntegrationError::StepUnderflow`] is unreachable by construction — which is the real case,
    /// not a contrived one: a one-day DSST mean propagation under a corrupted force model was
    /// measured holding `h ≈ 3e-4` against a `min_step` of `1e-6` and needing 6e8 steps.
    ///
    /// The cap is taken from the run's own measured cost rather than written down, so the test
    /// cannot pass by the fixture being unintegrable: the first integration proves it completes and
    /// says how dearly, and the second proves the ceiling bites at exactly the count asked for.
    #[test]
    fn the_step_ceiling_bounds_a_run_whose_step_never_approaches_the_floor() {
        let generous = Controls::new(Tolerance::new(1e-12, 1e-12), 1e-30);
        let (accepted, rejected) = integrate(
            &crate::Dop853,
            &Oscillator,
            0.0,
            &[1.0, 0.0],
            10.0,
            1e-3,
            generous,
        )
        .expect("the unbounded run completes")
        .step_counts();

        let needed = accepted + rejected;
        assert!(needed > 8, "fixture too cheap to bound: {needed} steps");
        assert!(
            needed < DEFAULT_MAX_STEPS,
            "fixture already exceeds the default ceiling: {needed} steps"
        );

        let cap = needed / 2;
        let got = integrate(
            &crate::Dop853,
            &Oscillator,
            0.0,
            &[1.0, 0.0],
            10.0,
            1e-3,
            generous.with_max_steps(cap),
        )
        .map(|traj| traj.step_counts());
        match got {
            Err(IntegrationError::StepLimit { steps }) => assert_eq!(steps, f64::from(cap)),
            other => panic!("expected StepLimit at {cap} steps, got {other:?}"),
        }
    }

    /// A ceiling of zero admits no step at all, so it is an argument error rather than a run that
    /// fails on its first step — the same treatment `min_step <= 0` already gets.
    #[test]
    fn a_zero_step_ceiling_is_refused_as_an_argument() {
        let controls = Controls::new(Tolerance::new(1e-9, 1e-9), 1e-12).with_max_steps(0);
        let got = integrate(
            &crate::Dop853,
            &Oscillator,
            0.0,
            &[1.0, 0.0],
            1.0,
            1e-3,
            controls,
        )
        .map(|traj| traj.step_counts());
        assert_eq!(got, Err(IntegrationError::BadArguments));
        assert_eq!(
            Controls::new(Tolerance::new(1e-9, 1e-9), 1e-12).max_steps,
            DEFAULT_MAX_STEPS
        );
    }

    /// A force model that returns a non-finite derivative reaches the floor, and reaches it in the
    /// dozen steps the shrink limit allows rather than at the ceiling.
    ///
    /// The error norm of such a step is NaN, and `err > 0.0` is false for NaN — the same branch an
    /// *exactly zero* error takes, which asks for the largest growth the controller permits. The
    /// step was rejected and then made nine times longer, so it could never fall to `min_step` and
    /// the run ended only at the step ceiling: measured at 100 000 steps and 15 s per attempt before
    /// the mapping to the worst finite case. Asserting the step *count* is what makes this a
    /// regression test rather than a restatement — the failure mode was never a wrong answer, it was
    /// four orders of magnitude of wasted work on the way to the right one.
    #[test]
    fn a_non_finite_derivative_reaches_the_step_floor_rather_than_the_ceiling() {
        struct Unevaluable;
        impl DynamicalSystem<2> for Unevaluable {
            fn derivative(&self, _t: f64, _y: &[f64; 2], dydt: &mut [f64; 2]) {
                dydt[0] = f64::INFINITY;
                dydt[1] = f64::INFINITY;
            }
        }

        let controls = Controls::new(Tolerance::new(1e-12, 1e-12), 1e-6);
        let got = integrate(
            &crate::Dop853,
            &Unevaluable,
            0.0,
            &[1.0, 0.0],
            10.0,
            60.0,
            controls,
        );
        let Err(IntegrationError::StepUnderflow { t }) = got else {
            panic!("expected StepUnderflow, got {got:?}")
        };
        assert_eq!(
            t, 0.0,
            "no step was ever accepted, so the run never left t = 0"
        );

        // `h` starts at the span it was given and is divided by the shrink limit each rejection, so
        // the count is `ceil(log_5(10 / 1e-6))` = 11 — the ceiling is 100 000.
        let counted = integrate(
            &crate::Dop853,
            &Unevaluable,
            0.0,
            &[1.0, 0.0],
            10.0,
            60.0,
            controls.with_max_steps(11),
        );
        assert!(
            matches!(counted, Err(IntegrationError::StepUnderflow { .. })),
            "the floor was not reached inside 11 steps: {counted:?}"
        );
    }
}
