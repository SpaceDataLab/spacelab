//! An adaptive Taylor-series integrator, driven by automatic differentiation of the flow.
//!
//! # The idea
//!
//! The exact solution of `y' = f(y)` is its Taylor series `y(t+h) = Σ aₖ hᵏ`. A Taylor integrator
//! sums that series directly: `a₀ = y`, `a₁ = f(y)`, and the recurrence `a_{k+1} = [f(y(·))]ₖ /(k+1)`
//! gives every higher coefficient, where `[f(y(·))]ₖ` is the `k`-th Taylor coefficient of the
//! right-hand side *along the solution*. Because that coefficient depends only on `a₀ … aₖ`, the
//! coefficients bootstrap one order at a time. The method reaches arbitrary order from one object —
//! the [`Jet`], a truncated power series that carries a value and its normalised derivatives through
//! every arithmetic operation and elementary function by the classical Taylor recurrences (Cauchy
//! product for `×`, and the `w = √u`, `1/u`, `uᵖ`, `eᵘ`, `sin`, `cos`, `atan2` recurrences below).
//!
//! A `Jet` **is** a [`Scalar`], so the same force model that yields the
//! value at `f64` and the Jacobian at [`Dual`](spacelab_autodiff) yields, at `Jet`, the full jet of
//! the acceleration — no hand-written derivatives. This is automatic differentiation used not for a
//! Jacobian but for the flow itself.
//!
//! # Step control
//!
//! With the series in hand, the step is chosen from its tail (Jorba & Zou 2005): the last two
//! coefficients `a_{p-1}, a_p` bound the truncation error, so `h = min((ε/‖a_{p-1}‖)^{1/(p-1)},
//! (ε/‖a_p‖)^{1/p})` (with a safety factor) keeps the discarded terms under the tolerance `ε`. Order
//! is fixed at compile time by `M = p+1`; step size adapts. `N` is the state dimension and `M` the
//! jet length; the caller supplies both (as [`crate::variational`] supplies `N` and `M`).
//!
//! # Applicability
//!
//! The clock is a jet like the state — along the step it is `t(h) = t + h` — so `∂f/∂t` enters the
//! recurrence at every order and a non-autonomous force keeps full order. That is only as good as
//! the force's own time dependence: one that reaches an `f64`-only subsystem (an ephemeris, a
//! density table, the frame rotation — see [`ScalarSystem::eval`]) is flat in `t` under
//! differentiation, and its series still truncates after the first order.
//!
//! Gate: `derivations/taylor_jet_recurrences.mac` shows, on symbolic jets, that every recurrence
//! here — product, quotient, `recip`, `sqrt`, `powf`, `exp`, `ln`, `sin`, `cos`, `atan2` — yields
//! the Taylor coefficients of its function, and that the clocked bootstrap reproduces the exact
//! series of the two-body problem (against its Lie series) and of `y′ = cos(t)·y`;
//! `derivations/taylor_jet_recurrences.gp` re-checks the recurrences against PARI's power series and
//! the two solutions by Picard iteration. The module test `the_solution_jets_are_the_exact_series`
//! (G2) holds [`solution_jets`] to those exact rationals.

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use core::ops::{Add, Div, Mul, Neg, Sub};
use std::vec::Vec;

use spacelab_autodiff::Scalar;

use crate::DenseOutput;
use crate::driver::{DEFAULT_MAX_STEPS, IntegrationError, Trajectory};
use crate::variational::ScalarSystem;

/// A truncated Taylor series `Σ_{k<M} c[k] τᵏ` — a jet of `M` coefficients. `c[0]` is the value.
#[derive(Debug, Clone, Copy)]
pub struct Jet<const M: usize> {
    c: [f64; M],
}

impl<const M: usize> Jet<M> {
    /// A jet with a given value and all higher coefficients zero.
    fn value_only(x: f64) -> Self {
        let mut c = [0.0; M];
        c[0] = x;
        Self { c }
    }

    /// Sum the series at `h` by Horner's rule.
    fn eval(&self, h: f64) -> f64 {
        let mut acc = 0.0;
        for k in (0..M).rev() {
            acc = acc * h + self.c[k];
        }
        acc
    }

    /// `d/dτ` as a jet: `(u')[k] = (k+1) u[k+1]`, top coefficient zero.
    fn deriv(&self) -> Self {
        let mut c = [0.0; M];
        for k in 0..M - 1 {
            c[k] = (k + 1) as f64 * self.c[k + 1];
        }
        Self { c }
    }
}

impl<const M: usize> Add for Jet<M> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut c = [0.0; M];
        for k in 0..M {
            c[k] = self.c[k] + o.c[k];
        }
        Self { c }
    }
}

impl<const M: usize> Sub for Jet<M> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        let mut c = [0.0; M];
        for k in 0..M {
            c[k] = self.c[k] - o.c[k];
        }
        Self { c }
    }
}

impl<const M: usize> Neg for Jet<M> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut c = [0.0; M];
        for k in 0..M {
            c[k] = -self.c[k];
        }
        Self { c }
    }
}

impl<const M: usize> Mul for Jet<M> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        // Cauchy product: (u·v)[k] = Σ_{j≤k} u[j] v[k−j].
        let mut c = [0.0; M];
        for k in 0..M {
            let mut acc = 0.0;
            for j in 0..=k {
                acc += self.c[j] * o.c[k - j];
            }
            c[k] = acc;
        }
        Self { c }
    }
}

impl<const M: usize> Div for Jet<M> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        // w[k] = (u[k] − Σ_{1≤j≤k} v[j] w[k−j]) / v[0].
        let mut w = [0.0; M];
        for k in 0..M {
            let mut acc = self.c[k];
            for j in 1..=k {
                acc -= o.c[j] * w[k - j];
            }
            w[k] = acc / o.c[0];
        }
        Self { c: w }
    }
}

impl<const M: usize> Scalar for Jet<M> {
    fn from_f64(x: f64) -> Self {
        Self::value_only(x)
    }

    fn value(self) -> f64 {
        self.c[0]
    }

    fn recip(self) -> Self {
        let mut w = [0.0; M];
        w[0] = 1.0 / self.c[0];
        for k in 1..M {
            let mut acc = 0.0;
            for j in 1..=k {
                acc += self.c[j] * w[k - j];
            }
            w[k] = -acc / self.c[0];
        }
        Self { c: w }
    }

    fn sqrt(self) -> Self {
        let mut w = [0.0; M];
        w[0] = libm::sqrt(self.c[0]);
        for k in 1..M {
            let mut acc = self.c[k];
            for j in 1..k {
                acc -= w[j] * w[k - j];
            }
            w[k] = acc / (2.0 * w[0]);
        }
        Self { c: w }
    }

    fn powf(self, p: f64) -> Self {
        // w[k] = (1/(k u[0])) Σ_{j<k} (p(k−j) − j) u[k−j] w[j].
        let mut w = [0.0; M];
        w[0] = libm::pow(self.c[0], p);
        for k in 1..M {
            let mut acc = 0.0;
            for j in 0..k {
                acc += (p * (k - j) as f64 - j as f64) * self.c[k - j] * w[j];
            }
            w[k] = acc / (k as f64 * self.c[0]);
        }
        Self { c: w }
    }

    fn exp(self) -> Self {
        let mut w = [0.0; M];
        w[0] = libm::exp(self.c[0]);
        for k in 1..M {
            let mut acc = 0.0;
            for j in 1..=k {
                acc += j as f64 * self.c[j] * w[k - j];
            }
            w[k] = acc / k as f64;
        }
        Self { c: w }
    }

    /// `u·w' = u'` read off order by order: `k u[0] w[k] = k u[k] − Σ_{0<j<k} j w[j] u[k−j]`.
    ///
    /// A non-positive `u[0]` is off the domain and is left to IEEE, as `sqrt`, `recip` and division
    /// above leave theirs: `libm::log` returns `−∞` or `NaN` and every higher coefficient inherits
    /// it through the division by `k u[0]`.
    fn ln(self) -> Self {
        let mut w = [0.0; M];
        w[0] = libm::log(self.c[0]);
        for k in 1..M {
            let mut acc = k as f64 * self.c[k];
            for j in 1..k {
                acc -= j as f64 * w[j] * self.c[k - j];
            }
            w[k] = acc / (k as f64 * self.c[0]);
        }
        Self { c: w }
    }

    fn sin(self) -> Self {
        // Coupled recurrence for (sin, cos); return sin.
        let (mut s, mut co) = ([0.0; M], [0.0; M]);
        s[0] = libm::sin(self.c[0]);
        co[0] = libm::cos(self.c[0]);
        for k in 1..M {
            let (mut sa, mut ca) = (0.0, 0.0);
            for j in 1..=k {
                let ju = j as f64 * self.c[j];
                sa += ju * co[k - j];
                ca += ju * s[k - j];
            }
            s[k] = sa / k as f64;
            co[k] = -ca / k as f64;
        }
        Self { c: s }
    }

    fn cos(self) -> Self {
        let (mut s, mut co) = ([0.0; M], [0.0; M]);
        s[0] = libm::sin(self.c[0]);
        co[0] = libm::cos(self.c[0]);
        for k in 1..M {
            let (mut sa, mut ca) = (0.0, 0.0);
            for j in 1..=k {
                let ju = j as f64 * self.c[j];
                sa += ju * co[k - j];
                ca += ju * s[k - j];
            }
            s[k] = sa / k as f64;
            co[k] = -ca / k as f64;
        }
        Self { c: co }
    }

    fn atan2(self, x: Self) -> Self {
        // w' = (x y' − y x') / (x² + y²); integrate: w[0] = atan2(y[0],x[0]), w[k] = d[k−1]/k.
        let y = self;
        let d = (x * y.deriv() - y * x.deriv()) / (x * x + y * y);
        let mut w = [0.0; M];
        w[0] = libm::atan2(y.c[0], x.c[0]);
        for k in 1..M {
            w[k] = d.c[k - 1] / k as f64;
        }
        Self { c: w }
    }
}

/// Whether stepping from `t` by `h` moves the clock at all.
///
/// The exact condition the `while t < t1 { … t += h }` loops depend on, written as itself rather than
/// as a floor on `h`. A floor cannot express it: `h` is a duration and `t` an absolute epoch, so the
/// smallest step that advances the clock is `ulp(t)/2` — `1.2e-7 s` at a J2000 epoch of `1e9 s`,
/// `2.2e-308 s` at `t = 0`, fifteen orders apart. The series-tail estimate legitimately returns a
/// step below that whenever the tolerance is tighter than the epoch's own resolution: `y' = y` at
/// `t = 1e9` with `tol = 1e-300` asks for `h = 7.5e-15`, and `t + h` is exactly `t`.
///
/// `!advances` also covers `h ≤ 0` and `h` NaN, so the caller's negation is the whole failure set.
fn advances(t: f64, h: f64) -> bool {
    t + h > t
}

/// One adaptive Taylor step from `(t, y)`: bootstrap the solution jets and choose the step from the
/// series tail (Jorba–Zou), clamped to `max_h`. Returns the jets (which *are* the dense interpolant)
/// and the chosen step `h`.
fn taylor_step<const N: usize, const M: usize>(
    force: &impl ScalarSystem<N>,
    t: f64,
    y: &[f64; N],
    tol: f64,
    max_h: f64,
) -> ([Jet<M>; N], f64) {
    let order = M - 1;
    // Bootstrap: Y[i] starts as the value; each order is filled from the matching Taylor coefficient
    // of the right-hand side evaluated on the jets so far.
    let mut yj = [Jet::<M>::value_only(0.0); N];
    for i in 0..N {
        yj[i] = Jet::value_only(y[i]);
    }
    // The independent variable is a jet as well: along the step it is t(h) = t + h, so c[1] = 1 and
    // every higher coefficient vanishes. Seeding it `value_only` would hold the clock fixed inside
    // the recurrence and hand the force ∂/∂t = 0, which is right for an autonomous system and wrong
    // for every other one — it truncates the series after the first order and drops the method to
    // O(h²) whatever `M` says.
    let mut tj = Jet::<M>::value_only(t);
    tj.c[1] = 1.0;
    for k in 0..order {
        let f = force.eval(tj, &yj);
        for i in 0..N {
            yj[i].c[k + 1] = f[i].c[k] / (k + 1) as f64;
        }
    }

    let mut norm_p = 0.0_f64;
    let mut norm_pm1 = 0.0_f64;
    for i in 0..N {
        norm_p = norm_p.max(yj[i].c[order].abs());
        norm_pm1 = norm_pm1.max(yj[i].c[order - 1].abs());
    }
    let hp = if norm_p > 0.0 {
        (tol / norm_p).powf(1.0 / order as f64)
    } else {
        max_h
    };
    let hpm1 = if norm_pm1 > 0.0 {
        (tol / norm_pm1).powf(1.0 / (order - 1) as f64)
    } else {
        max_h
    };
    // No floor: a step the tail estimate says is too small to take is [`integrate_taylor`]'s to
    // report, not this function's to round up. A floor here can only be a constant, and no constant
    // is right — whether a step advances the clock depends on the *epoch*, not on the step alone.
    let h = (0.9 * hp.min(hpm1)).min(max_h);
    (yj, h)
}

/// The Taylor coefficients of the solution at `(t, y)` up to order `M − 1`: `coeffs[i][k]` is the
/// `k`-th normalised derivative of component `i` (so `yᵢ(t+τ) = Σₖ coeffs[i][k] τᵏ`). Shared with
/// [`crate::adams_nordsieck`], which builds its startup Nordsieck vector `zᵢ[k] = hᵏ·coeffs[i][k]`
/// from these exact scaled derivatives instead of a bootstrap of its own.
#[must_use]
pub fn solution_jets<const N: usize, const M: usize>(
    force: &impl ScalarSystem<N>,
    t: f64,
    y: &[f64; N],
) -> [[f64; M]; N] {
    let (yj, _) = taylor_step::<N, M>(force, t, y, 1.0, 0.0);
    let mut out = [[0.0; M]; N];
    for i in 0..N {
        out[i] = yj[i].c;
    }
    out
}

/// Integrate `force` from `t0` to `t1` with an adaptive Taylor method of order `M − 1`.
///
/// `tol` bounds the per-step truncation error, which sets the step size. `M` is the jet length
/// (order + 1); `N` is the state dimension.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span or tolerance is not positive, or `M < 3`.
///
/// [`IntegrationError::StepUnderflow`] if the series-tail step estimate stops advancing the
/// clock — a tolerance tighter than the epoch's own `f64` resolution can ask for one.
///
/// [`IntegrationError::StepLimit`] if the run passes [`DEFAULT_MAX_STEPS`] steps. `advances` bounds
/// how *small* a step can be, not how many of them a span holds: the floor is `ulp(t)/2`, which at
/// a J2000 epoch of `1e9 s` is `1.2e-7 s`, so a one-day span admits 7.3e11 of them — a run no
/// caller can interrupt and none would want. Counted rather than derived from the floor, for the
/// reason `driver::integrate` counts: the floor is an epoch-dependent quantity and the bound it
/// gives is no bound at all.
pub fn integrate_taylor<const N: usize, const M: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    tol: f64,
) -> Result<[f64; N], IntegrationError> {
    if !(t1 > t0 && tol > 0.0) || M < 3 {
        return Err(IntegrationError::BadArguments);
    }
    let mut y = *y0;
    let mut t = t0;
    let mut steps = 0u32;
    while t < t1 {
        let (yj, h) = taylor_step::<N, M>(force, t, &y, tol, t1 - t);
        if !advances(t, h) {
            return Err(IntegrationError::StepUnderflow { t });
        }
        for i in 0..N {
            y[i] = yj[i].eval(h);
        }
        t += h;
        steps += 1;
        if steps >= DEFAULT_MAX_STEPS {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(steps),
            });
        }
    }
    Ok(y)
}

/// The Taylor method's native dense output over one step: the solution's own truncated Taylor
/// polynomial, summed by Horner. Order `M − 1` — the same order as the step, so the interpolant is as
/// accurate as the step itself, not a low-order graft. This is the "excellent dense output" a Taylor
/// integrator gives for free (Jorba–Zou; `heyoka`).
#[derive(Debug, Clone, Copy)]
pub struct TaylorDense<const N: usize, const M: usize> {
    t0: f64,
    coeffs: [[f64; M]; N],
}

impl<const N: usize, const M: usize> DenseOutput<N> for TaylorDense<N, M> {
    fn eval(&self, t: f64, out: &mut [f64; N]) {
        let tau = t - self.t0;
        for i in 0..N {
            let mut acc = 0.0;
            for k in (0..M).rev() {
                acc = acc * tau + self.coeffs[i][k];
            }
            out[i] = acc;
        }
    }
}

/// Integrate as [`integrate_taylor`], returning a dense [`Trajectory`] whose every segment is the
/// step's order-`(M−1)` Taylor polynomial — evaluable at any `t`, for the event detector.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span or tolerance is not positive, or `M < 3`.
///
/// [`IntegrationError::StepUnderflow`] if the series-tail step estimate stops advancing the
/// clock — a tolerance tighter than the epoch's own `f64` resolution can ask for one.
///
/// [`IntegrationError::StepLimit`] if the run passes [`DEFAULT_MAX_STEPS`] steps. `advances` bounds
/// how *small* a step can be, not how many of them a span holds: the floor is `ulp(t)/2`, which at
/// a J2000 epoch of `1e9 s` is `1.2e-7 s`, so a one-day span admits 7.3e11 of them — a run no
/// caller can interrupt and none would want. Counted rather than derived from the floor, for the
/// reason `driver::integrate` counts: the floor is an epoch-dependent quantity and the bound it
/// gives is no bound at all.
pub fn integrate_taylor_dense<const N: usize, const M: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    tol: f64,
) -> Result<Trajectory<N, TaylorDense<N, M>>, IntegrationError> {
    if !(t1 > t0 && tol > 0.0) || M < 3 {
        return Err(IntegrationError::BadArguments);
    }
    let mut y = *y0;
    let mut t = t0;
    let mut segments = Vec::new();
    let mut ends = Vec::new();
    let mut taken = 0u32;
    while t < t1 {
        let (yj, h) = taylor_step::<N, M>(force, t, &y, tol, t1 - t);
        if !advances(t, h) {
            return Err(IntegrationError::StepUnderflow { t });
        }
        let mut coeffs = [[0.0; M]; N];
        for i in 0..N {
            coeffs[i] = yj[i].c;
        }
        segments.push(TaylorDense { t0: t, coeffs });
        for i in 0..N {
            y[i] = yj[i].eval(h);
        }
        t += h;
        ends.push(t);
        taken += 1;
        if taken >= DEFAULT_MAX_STEPS {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(taken),
            });
        }
    }
    let steps = segments.len() as u32;
    Ok(Trajectory::from_segments(
        t0, t1, y, segments, ends, steps, 0,
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unwrap_used)] // exact recurrence identities; test unwraps
    use super::*;
    use core::f64::consts::FRAC_PI_2;

    const M: usize = 8;

    /// The variable jet `x0 + τ`: value `x0`, first derivative 1, higher coefficients zero.
    fn var(x0: f64) -> Jet<M> {
        let mut c = [0.0; M];
        c[0] = x0;
        c[1] = 1.0;
        Jet { c }
    }

    fn factorial(k: usize) -> f64 {
        (1..=k).map(|j| j as f64).product()
    }

    /// A step too small to move the clock is reported, not rounded up to one that is.
    ///
    /// `y' = y` is the mildest system there is, and `t = 1e9 s` an ordinary J2000 epoch — but a
    /// tolerance of `1e-300` asks the order-20 tail for `h = 7.5e-15 s`, and `ulp(1e9)` is
    /// `1.2e-7`, so `t + h` is exactly `t`. Left as a floor at `f64::MIN_POSITIVE` this is an
    /// infinite loop: the estimate never grows, because the state it is computed from never
    /// changes. The same call at `t = 0` runs to completion, which is the whole point — the failure
    /// is a property of the epoch, and no constant floor on `h` can see it.
    #[test]
    fn a_step_that_cannot_move_the_clock_is_reported() {
        struct Exp;
        impl crate::variational::ScalarSystem<1> for Exp {
            fn eval<S: spacelab_autodiff::Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
                [y[0]]
            }
        }
        assert_eq!(
            integrate_taylor::<1, 21>(&Exp, 1.0e9, &[1.0], 1.0e9 + 1.0, 1e-300),
            Err(IntegrationError::StepUnderflow { t: 1.0e9 })
        );
        assert_eq!(
            integrate_taylor_dense::<1, 21>(&Exp, 1.0e9, &[1.0], 1.0e9 + 1.0, 1e-300).err(),
            Some(IntegrationError::StepUnderflow { t: 1.0e9 })
        );
        // The identical call near `t = 0`, where the ulp is far below the step, is not an error and
        // still integrates: the report is about the clock, not about the tolerance. The span is cut
        // to the step's own scale, because `1.16e-15` per step is as far as this tolerance reaches.
        let span = 1e-14;
        let y = integrate_taylor::<1, 21>(&Exp, 0.0, &[1.0], span, 1e-300).unwrap();
        // Nine accumulation steps of an `O(1)` value, so the bar is a few ulps of the *value*, not of
        // the `1e-14` increment: `e^span − 1` is forty-five ulps wide and cannot be resolved better.
        assert!((y[0] - span.exp()).abs() < 4.0 * f64::EPSILON, "{}", y[0]);
    }

    #[test]
    fn exp_cos_and_powf_have_the_analytic_taylor_coefficients() {
        let x0 = 0.7;
        let u = var(x0);

        // exp(x0 + τ) = exp(x0)·Σ τᵏ/k!  ⇒  c[k] = exp(x0)/k!.
        let e = u.exp();
        for k in 0..M {
            let want = x0.exp() / factorial(k);
            assert!((e.c[k] - want).abs() <= 1e-13 * want.abs(), "exp c[{k}]");
        }
        // cosᵏ(x0) = cos(x0 + k·π/2)  ⇒  c[k] = cos(x0 + k·π/2)/k!.
        let co = u.cos();
        for k in 0..M {
            let want = (x0 + k as f64 * FRAC_PI_2).cos() / factorial(k);
            assert!((co.c[k] - want).abs() <= 1e-13, "cos c[{k}]");
        }
        // powf: c[k] = [Π_{j<k}(p−j)]/k! · x0^(p−k).
        let p = 1.7;
        let pw = u.powf(p);
        for k in 0..M {
            let binom: f64 = (0..k).map(|j| p - j as f64).product::<f64>() / factorial(k);
            let want = binom * x0.powf(p - k as f64);
            assert!((pw.c[k] - want).abs() <= 1e-12 * want.abs(), "powf c[{k}]");
        }
    }

    /// The logarithm's jet, against the analytic coefficients and against `exp` run backwards.
    ///
    /// `ln(x₀ + τ) = ln x₀ + Σ_{k≥1} (−1)^{k+1} τᵏ/(k x₀ᵏ)`, which fixes every coefficient
    /// independently of the recurrence that produced it. The round trip is the second opinion: it
    /// couples `ln` to the `exp` above, which has its own analytic test, and it exercises a jet
    /// whose higher coefficients are not zero — the linear jet alone leaves most of the convolution
    /// untouched, the same blind spot `powf` has below. Measured at `x₀ = 0.7`, `M = 8`: worst
    /// **2.2e-16** relative on the analytic coefficients, and the round trip returns `u` bit-exactly
    /// (worst `|exp(ln(u))ₖ − uₖ|` is `0`).
    #[test]
    fn ln_has_the_analytic_taylor_coefficients_and_inverts_exp() {
        let x0 = 0.7;
        let l = var(x0).ln();
        assert!((l.c[0] - x0.ln()).abs() <= 1e-15, "ln c[0]");
        for k in 1..M {
            let want = if k % 2 == 0 { -1.0 } else { 1.0 } / (k as f64 * x0.powf(k as f64));
            assert!((l.c[k] - want).abs() <= 1e-13 * want.abs(), "ln c[{k}]");
        }

        // exp(ln(u)) = u on a jet with a full set of nonzero coefficients.
        let u = var(x0).exp();
        let round = u.ln().exp();
        for k in 0..M {
            assert!(
                (round.c[k] - u.c[k]).abs() <= 1e-13 * u.c[k].abs().max(1.0),
                "exp(ln(u)) c[{k}] = {} vs {}",
                round.c[k],
                u.c[k]
            );
        }
    }

    /// `powf` weights every term of its sum by `k − j`, and a jet whose only nonzero derivative is
    /// the first never gives that weight a value but one: for `u = x₀ + τ` the sum
    /// `Σ_{j<k} (p(k−j) − j) u[k−j] w[j]` keeps only `j = k−1`, where `p·1` and `p/1` agree. That
    /// linear jet is the only argument `powf` ever sees in this crate, because the force models
    /// reach for `sqrt().recip()` and not `powf(−1.5)`. Squaring it brings `k − j = 2` into the
    /// sum: `(1 + τ)²` raised to `±3/2` is `(1 + τ)^{±3}`, whose coefficients are `C(±3, k)`.
    #[test]
    fn powf_of_a_quadratic_jet_is_the_binomial_series() {
        let mut c = [0.0; M];
        c[0] = 1.0;
        c[1] = 2.0;
        c[2] = 1.0;
        let u = Jet::<M> { c };
        let cases: [(f64, [f64; M]); 2] = [
            (1.5, [1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            (-1.5, [1.0, -3.0, 6.0, -10.0, 15.0, -21.0, 28.0, -36.0]),
        ];
        for (p, want) in cases {
            let w = u.powf(p);
            for k in 0..M {
                let bar = 1e-12 * want[k].abs().max(1.0);
                assert!(
                    (w.c[k] - want[k]).abs() <= bar,
                    "powf({p}) c[{k}] = {}",
                    w.c[k]
                );
            }
        }
    }

    #[test]
    fn atan2_reproduces_the_arctangent_along_the_solution() {
        // atan2(y0 + τ, x0) summed at h reproduces atan2(y0 + h, x0) to truncation order; this also
        // drives deriv(), which atan2 uses internally.
        let (y0, x0) = (0.4, 1.3);
        let y = var(y0);
        let x = Jet::<M>::from_f64(x0);
        let w = y.atan2(x);
        assert!((w.c[0] - y0.atan2(x0)).abs() < 1e-15);
        // d/dτ atan2(y0+τ, x0) = x0/(x0² + y0²) at τ = 0.
        assert!((w.c[1] - x0 / (x0 * x0 + y0 * y0)).abs() < 1e-13);
        for &h in &[1e-2, -1e-2] {
            assert!((w.eval(h) - (y0 + h).atan2(x0)).abs() < 1e-12);
        }
    }

    /// `atan2` differentiates *both* of its arguments, and a constant `x` hides half the formula:
    /// `x′ = 0` erases the second term of the `x y′ − y x′` numerator outright, and `y = y₀ + τ`
    /// makes the first term's factor `y′` the constant 1, which divides exactly as it multiplies.
    /// Put the two arguments on the unit circle instead — `sin` and `cos` coefficients written out,
    /// not produced by the jet recurrences under test — and the answer is the angle itself:
    /// `atan2(sin τ, cos τ) = τ`, so every coefficient vanishes but the first.
    #[test]
    fn atan2_of_the_unit_circle_is_the_angle_itself() {
        let (mut s, mut co) = ([0.0; M], [0.0; M]);
        s[1] = 1.0;
        s[3] = -1.0 / 6.0;
        s[5] = 1.0 / 120.0;
        s[7] = -1.0 / 5040.0;
        co[0] = 1.0;
        co[2] = -1.0 / 2.0;
        co[4] = 1.0 / 24.0;
        co[6] = -1.0 / 720.0;
        let w = Jet::<M> { c: s }.atan2(Jet::<M> { c: co });
        for k in 0..M {
            let want = if k == 1 { 1.0 } else { 0.0 };
            assert!((w.c[k] - want).abs() < 1e-14, "atan2 c[{k}] = {}", w.c[k]);
        }
    }

    #[test]
    fn the_jet_operators_obey_their_algebra() {
        let a = var(1.1);
        let b = Jet::<M> {
            c: [0.3, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        };

        // Sub and Neg are componentwise; value is c[0].
        let d = a - b;
        let n = -a;
        for k in 0..M {
            assert_eq!(d.c[k], a.c[k] - b.c[k]);
            assert_eq!(n.c[k], -a.c[k]);
        }
        assert_eq!(a.value(), a.c[0]);

        // Division is the inverse of the Cauchy product: (a / b) · b == a.
        let back = (a / b) * b;
        for k in 0..M {
            assert!((back.c[k] - a.c[k]).abs() < 1e-12, "div/mul c[{k}]");
        }

        // deriv: (u′)[k] = (k+1)·u[k+1], top coefficient zero.
        let da = a.deriv();
        for k in 0..M - 1 {
            assert_eq!(da.c[k], (k as f64 + 1.0) * a.c[k + 1]);
        }
        assert_eq!(da.c[M - 1], 0.0);
    }

    /// The planar two-body problem, `μ = 1`, with `|r|⁻³` formed as `(r²)^{−3/2}` through `powf`
    /// (`POWF`) or as `1/(√r² · r²)` through `sqrt` and `recip`, the force models' spelling.
    struct TwoBody<const POWF: bool>;
    impl<const POWF: bool> ScalarSystem<4> for TwoBody<POWF> {
        fn eval<S: Scalar>(&self, _t: S, y: &[S; 4]) -> [S; 4] {
            let r2 = y[0] * y[0] + y[1] * y[1];
            let k = if POWF {
                r2.powf(-1.5)
            } else {
                (r2.sqrt() * r2).recip()
            };
            [y[2], y[3], -(y[0] * k), -(y[1] * k)]
        }
    }

    /// `y′ = cos(t)·y`: the right-hand side reads the clock, so its jet needs `t + τ`, not `t`.
    struct CosClock;
    impl ScalarSystem<1> for CosClock {
        fn eval<S: Scalar>(&self, t: S, y: &[S; 1]) -> [S; 1] {
            [t.cos() * y[0]]
        }
    }

    /// The bootstrap's coefficients through `τ⁷` as exact rationals, printed by
    /// `derivations/taylor_jet_recurrences.mac` for the two-body state `x = (3/5, 4/5)`,
    /// `v = (−1/2, 1)` and for `exp(sin t)`.
    const TWO_BODY_JET: [[(i32, i32); M]; 4] = [
        [
            (3, 5),
            (-1, 2),
            (-3, 10),
            (7, 30),
            (-9, 80),
            (1, 30),
            (7, 800),
            (-149, 6300),
        ],
        [
            (4, 5),
            (1, 1),
            (-2, 5),
            (1, 30),
            (7, 120),
            (-1, 15),
            (167, 3600),
            (-1081, 50400),
        ],
        [
            (-1, 2),
            (-3, 5),
            (7, 10),
            (-9, 20),
            (1, 6),
            (21, 400),
            (-149, 900),
            (5989, 33600),
        ],
        [
            (1, 1),
            (-4, 5),
            (1, 10),
            (7, 30),
            (-1, 3),
            (167, 600),
            (-1081, 7200),
            (953, 50400),
        ],
    ];
    const EXP_SIN_JET: [(i32, i32); M] = [
        (1, 1),
        (1, 1),
        (1, 2),
        (0, 1),
        (-1, 8),
        (-1, 15),
        (-1, 240),
        (1, 90),
    ];

    /// (G2) `solution_jets` returns the solution's own Taylor series: the coefficients the
    /// bootstrap `a_{k+1} = [f(y(·))]ₖ/(k+1)` must produce by its defining equation, which
    /// `derivations/taylor_jet_recurrences.mac` derives in exact rationals — every jet recurrence
    /// against Maxima's series, the two-body bootstrap against the Lie series `Lᵏx/k!`, the
    /// clocked one against `exp(sin t)` — and `derivations/taylor_jet_recurrences.gp` re-derives by
    /// Picard iteration on PARI's power series.
    ///
    /// The two-body state is the nearest double to `(3/5, 4/5, −1/2, 1)`, so `|r| = 1` only to an
    /// ulp; the coefficients are `O(1)` and come out of at most eight nested recurrences, and the bar
    /// is `64 ε`. `exp(sin t)` from `t = 0` is the case a clock frozen at `t₀` gets wrong from `τ²`
    /// on (it gives `1/k!`).
    #[test]
    fn the_solution_jets_are_the_exact_series() {
        let bar = 64.0 * f64::EPSILON;
        let state = [0.6, 0.8, -0.5, 1.0];
        for (name, jets) in [
            ("powf", solution_jets::<4, M>(&TwoBody::<true>, 0.0, &state)),
            (
                "sqrt·recip",
                solution_jets::<4, M>(&TwoBody::<false>, 0.0, &state),
            ),
        ] {
            for (i, row) in TWO_BODY_JET.iter().enumerate() {
                for (k, &(num, den)) in row.iter().enumerate() {
                    let want = f64::from(num) / f64::from(den);
                    let d = (jets[i][k] - want).abs();
                    assert!(
                        d <= bar,
                        "two-body ({name}) y{i} c[{k}] = {:.17e}, want {num}/{den}",
                        jets[i][k]
                    );
                }
            }
        }
        let jets = solution_jets::<1, M>(&CosClock, 0.0, &[1.0]);
        for (k, &(num, den)) in EXP_SIN_JET.iter().enumerate() {
            let want = f64::from(num) / f64::from(den);
            assert!(
                (jets[0][k] - want).abs() <= bar,
                "exp(sin t) c[{k}] = {:.17e}, want {num}/{den}",
                jets[0][k]
            );
        }
    }

    /// `y′ = -y` — exponential decay, for a real integration and the dense/bad-argument paths.
    struct Decay;
    impl ScalarSystem<1> for Decay {
        fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
            [S::from_f64(-1.0) * y[0]]
        }
    }

    /// `y′ = 0` — a constant solution whose series tail vanishes, so the tail-based step falls back to
    /// `max_h`.
    struct Zero;
    impl ScalarSystem<1> for Zero {
        fn eval<S: Scalar>(&self, _t: S, _y: &[S; 1]) -> [S; 1] {
            [S::from_f64(0.0)]
        }
    }

    /// `y′ = a + b·y²` — one right-hand side in the three shapes the step rule has to tell apart.
    /// `b = 0` truncates the solution at degree one; `a = 0` has the closed-form jet
    /// `aₖ = y₀^{k+1}`; `a = b = 1` from `y₀ = 0` is `tan`, whose even coefficients vanish.
    struct Riccati {
        a: f64,
        b: f64,
    }
    impl ScalarSystem<1> for Riccati {
        fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
            [S::from_f64(self.a) + S::from_f64(self.b) * y[0] * y[0]]
        }
    }

    fn assert_step(got: f64, want: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-12 * want,
            "{what}: h = {got:.17e}, want {want:.17e}"
        );
    }

    /// The step **is** the Jorba–Zou tail estimate, not merely a step small enough to be accurate.
    ///
    /// Every other test here reads the answer, and the answer is what a step-size rule is best at
    /// hiding: halve the step and the same trajectory comes back, only more slowly. `y′ = y²` has
    /// the closed form `y₀/(1 − y₀t)` and therefore the jet `aₖ = y₀^{k+1}`, so at `y₀ = 2` and
    /// `M = 8` the two tail norms are exactly `2⁷` and `2⁸` and the estimate is a number rather
    /// than a bound. Four rows: each estimate binding in turn, and each vanishing norm falling
    /// back to `max_h` in turn.
    #[test]
    fn the_step_is_the_jorba_zou_tail_estimate() {
        let square = Riccati { a: 0.0, b: 1.0 };
        let (jets, h) = taylor_step::<1, 8>(&square, 0.0, &[2.0], 1e-7, 100.0);
        for k in 0..M {
            assert_eq!(jets[0].c[k], f64::from(1_i32 << (k + 1)), "y² jet c[{k}]");
        }
        // ‖a₆‖ = 2⁷ and ‖a₇‖ = 2⁸, and at this tolerance the order-6 estimate is the smaller of the
        // two: h = 0.9·(1e-7/128)^{1/6} = 2.731330039513127545081863569047e-2 (Maxima, 40 digits).
        assert_step(h, 2.731_330_039_513_127_5e-2, "order-6 tail binds");
        // The same series at a tolerance above 2 puts the order-7 estimate below the other:
        // h = 0.9·(2.56e9/256)^{1/7} = 0.9·10.
        let (_, h) = taylor_step::<1, 8>(&square, 0.0, &[2.0], 2.56e9, 100.0);
        assert_step(h, 9.0, "order-7 tail binds");
        // b = 0 ⇒ y = y₀ + t, so with M = 3 the top coefficient a₂ is zero and only the order-1
        // estimate is real. It is tol/‖a₁‖ = 10, above `max_h`, so the clamp leaves 0.9·max_h — a
        // fallback of `∞` rather than `max_h` would leave `max_h` itself.
        let (_, h) = taylor_step::<1, 3>(&Riccati { a: 1.0, b: 0.0 }, 0.0, &[0.0], 10.0, 1.0);
        assert_step(h, 0.9, "vanishing top coefficient");
        // `tan` has a₂ = 0 and a₃ = 1/3, so with M = 4 the *next-to-top* coefficient is the zero
        // one and the order-3 estimate is the only real one: (9/(1/3))^{1/3} = 3, clamped again.
        let (_, h) = taylor_step::<1, 4>(&Riccati { a: 1.0, b: 1.0 }, 0.0, &[0.0], 9.0, 1.0);
        assert_step(h, 0.9, "vanishing next-to-top coefficient");
    }

    /// `advances` is the whole underflow refusal, every caller reads it negated, and the loop it
    /// guards has no other exit — so a wrong answer here is not a wrong number, it is a hang. Its
    /// three answers are asserted directly: a hang is a detection by the wall clock, and an
    /// assertion is the cheaper detector.
    #[test]
    fn advancing_the_clock_is_an_epoch_relative_question() {
        assert!(advances(0.0, f64::MIN_POSITIVE)); // at t = 0 the smallest subnormal is a step
        assert!(advances(1.0e9, 1.0e-6)); // ulp(1e9) is 1.2e-7, so a microsecond still moves it
        assert!(!advances(1.0e9, 7.5e-15)); // … and 7.5 femtoseconds does not
        assert!(!advances(0.0, 0.0)); // `h = 0` advances no clock at any epoch
        assert!(!advances(0.0, f64::NAN)); // nor does a NaN step, which the caller must refuse too
    }

    #[test]
    fn a_constant_system_takes_the_max_step_and_holds_its_value() {
        let y = integrate_taylor::<1, 4>(&Zero, 0.0, &[5.0], 1.0, 1e-9).unwrap();
        assert!((y[0] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn a_decaying_system_matches_the_exponential_and_rejects_bad_arguments() {
        let y = integrate_taylor::<1, 8>(&Decay, 0.0, &[2.0], 1.0, 1e-12).unwrap();
        assert!(
            (y[0] - 2.0 * (-1.0_f64).exp()).abs() < 1e-9,
            "decay {}",
            y[0]
        );
        assert!(matches!(
            integrate_taylor_dense::<1, 8>(&Decay, 0.0, &[2.0], -1.0, 1e-12),
            Err(IntegrationError::BadArguments),
        ));
    }

    /// Every refusal of the entry guard, named rather than counted. `is_err()` is not the assertion
    /// to make: a tolerance of zero drives the tail estimate to a zero step, so the call fails
    /// either way — with `StepUnderflow`, which is a different statement about a different failure,
    /// made after the work rather than before it. `M` is the jet length, so `M = 3` is the shortest
    /// jet the tail estimate has two coefficients for and has to be accepted, while `M = 2` leaves
    /// the order-`(M−2)` exponent at `1/0` and must not be. Both entry points carry their own copy
    /// of the guard, so both owe the same four answers.
    #[test]
    fn the_entry_guard_names_every_argument_it_refuses() {
        let bad = Some(IntegrationError::BadArguments);
        let y = [5.0];
        assert_eq!(
            integrate_taylor::<1, 8>(&Zero, 3.0, &y, 3.0, 1e-9).err(),
            bad,
            "t1 == t0"
        );
        assert_eq!(
            integrate_taylor::<1, 8>(&Zero, 0.0, &y, 1.0, 0.0).err(),
            bad,
            "tol == 0"
        );
        assert_eq!(
            integrate_taylor::<1, 2>(&Zero, 0.0, &y, 1.0, 1e-9).err(),
            bad,
            "M == 2"
        );
        assert!(
            integrate_taylor::<1, 3>(&Zero, 0.0, &y, 1.0, 1e-9).is_ok(),
            "M == 3"
        );
        let dense_span = integrate_taylor_dense::<1, 8>(&Zero, 3.0, &y, 3.0, 1e-9);
        assert_eq!(dense_span.err(), bad, "dense t1 == t0");
        let dense_tol = integrate_taylor_dense::<1, 8>(&Zero, 0.0, &y, 1.0, 0.0);
        assert_eq!(dense_tol.err(), bad, "dense tol == 0");
        let dense_short = integrate_taylor_dense::<1, 2>(&Zero, 0.0, &y, 1.0, 1e-9);
        assert_eq!(dense_short.err(), bad, "dense M == 2");
        assert!(
            integrate_taylor_dense::<1, 3>(&Zero, 0.0, &y, 1.0, 1e-9).is_ok(),
            "dense M == 3"
        );
    }
}
