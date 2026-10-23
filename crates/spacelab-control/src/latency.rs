//! What the loop knew, and when it knew it.
//!
//! # Three delays, not one
//!
//! A closed loop reads a sensor, decides, and commands an actuator, and each of those three costs
//! time. [`Latency`] keeps them apart because they are bought down separately — a star tracker's
//! integration time with aperture, the compute latency with a faster processor, the actuator lag
//! with a different wheel — and a stability margin quoted against their sum tells an engineer
//! nothing about which one to spend money on. The sum is available from [`Latency::total`] for the
//! places that genuinely need only the loop delay, such as the characteristic equation of the
//! linearised system.
//!
//! # Commensurate delays are exact, and the others are refused rather than rounded
//!
//! An attitude loop runs at a fixed rate, so its delays are whole numbers of steps and the state at
//! `t − τ` is a *stored sample*. No interpolation, no interpolation error. [`Latency::steps`]
//! returns that whole number, and returns `None` when the delay is not one — because the alternative
//! is rounding `2.5` steps to `2` and reporting a margin for a loop nobody is running.
//!
//! The tolerance on "is a whole number of steps" is four ulps of the larger of `τ` and `dt`, and the
//! four is arithmetic rather than taste. `total()` is two additions, each correctly rounded, so it
//! carries at most about one epsilon of relative error; `k · dt` is one multiplication and carries
//! half of one. Two quantities that are equal in exact arithmetic can therefore differ by roughly
//! one and a half epsilons after rounding, and four leaves 2.7× of headroom over that bound without
//! being loose enough to admit a delay that is a genuine fraction of a step.
//!
//! # The initial history is part of the problem, not a detail of the buffer
//!
//! A delayed system needs a state at `t − τ` from the first step, and before the loop has run that
//! long there is nothing to read. What fills the gap is the **initial history** `φ(t)`, `t < t₀`,
//! and it is a boundary condition of the problem rather than an implementation choice —
//! [`History::seeded`] takes it explicitly for that reason, and there is no constructor that starts
//! the buffer empty and quietly hands back zeros.
//!
//! It matters beyond bookkeeping. Where `φ` does not agree with the solution the system would have
//! produced, the derivative jumps at `t₀`, the second derivative jumps at `t₀ + τ`, the third at
//! `t₀ + 2τ`, and so on: each delay interval smooths the solution by exactly one order. A fixed-step
//! method walks over those breakpoints without noticing. An adaptive one must place a mesh point at
//! each of them or it loses its order while its error estimate keeps saying the step was fine.

/// The three latencies of a control loop, in seconds.
///
/// Construct through [`Latency::new`], which rejects anything not finite and non-negative, or
/// through [`Latency::none`] for the instantaneous loop that most textbook analysis assumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Latency {
    sensor: f64,
    compute: f64,
    actuator: f64,
}

impl Latency {
    /// The instantaneous loop: every latency zero.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            sensor: 0.0,
            compute: 0.0,
            actuator: 0.0,
        }
    }

    /// `None` unless all three are finite and non-negative. A negative latency is a loop that acts
    /// on a measurement it has not taken yet, and there is no sensible clamp for it.
    #[must_use]
    pub fn new(sensor: f64, compute: f64, actuator: f64) -> Option<Self> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        (ok(sensor) && ok(compute) && ok(actuator)).then_some(Self {
            sensor,
            compute,
            actuator,
        })
    }

    /// The time between the state a measurement describes and the moment the resulting torque acts.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.sensor + self.compute + self.actuator
    }

    /// The delay from the sensor alone.
    #[must_use]
    pub const fn sensor(&self) -> f64 {
        self.sensor
    }

    /// The delay from deciding.
    #[must_use]
    pub const fn compute(&self) -> f64 {
        self.compute
    }

    /// The delay from the actuator responding.
    #[must_use]
    pub const fn actuator(&self) -> f64 {
        self.actuator
    }

    /// How many whole steps of `dt` the total latency is, or `None` if it is not a whole number of
    /// them to within four ulps.
    ///
    /// Refusing is the point. A loop whose delay is 2.5 steps is a different problem from one whose
    /// delay is 2 or 3, and it needs the dense output of the step the delay falls inside; rounding
    /// here would answer a question nobody asked and label it with the caller's `τ`.
    #[must_use]
    pub fn steps(&self, dt: f64) -> Option<usize> {
        let total = self.total();
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        if total == 0.0 {
            return Some(0);
        }
        let k = libm::round(total / dt);
        #[allow(clippy::cast_precision_loss)]
        let ceiling = usize::MAX as f64;
        if k < 0.0 || k > ceiling {
            return None;
        }
        let tol = 4.0 * f64::EPSILON * if total > dt { total } else { dt };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        (libm::fabs(total - k * dt) <= tol).then_some(k as usize)
    }
}

/// A fixed-capacity ring of past states, seeded with the problem's initial history.
///
/// `N` is the state's dimension and `C` its capacity in samples. A delay of `k` steps needs
/// `C > k`; [`History::delayed`] returns `None` rather than aliasing when it is asked for more.
#[derive(Debug, Clone)]
pub struct History<const N: usize, const C: usize> {
    buf: [[f64; N]; C],
    /// Index one past the newest sample, so `buf[head - 1 - k]` is the sample `k` steps back.
    head: usize,
}

impl<const N: usize, const C: usize> History<N, C> {
    /// Seed every slot with a constant initial history `φ(t) = state` for `t ≤ t₀`.
    ///
    /// Constant history is the ordinary initial condition for a delayed system and the one a loop
    /// started from rest is genuinely in. It is also the one that makes the derivative jump at `t₀`
    /// unless the state happens to be an equilibrium — which is a property of the problem and not
    /// something this type should hide.
    #[must_use]
    pub const fn seeded(state: [f64; N]) -> Self {
        Self {
            buf: [state; C],
            head: 0,
        }
    }

    /// Seed slot `k` steps back with `history(k)`, for a non-constant `φ`.
    #[must_use]
    pub fn seeded_with(mut history: impl FnMut(usize) -> [f64; N]) -> Self {
        let mut buf = [[0.0; N]; C];
        for (k, slot) in buf.iter_mut().enumerate() {
            *slot = history(k);
        }
        // `seeded_with` fills by age, so slot `k` is `k` steps back with `head` at zero: index
        // `(0 + C - 1 - k) % C` must be `k`, which holds for the reversed fill below.
        buf.reverse();
        Self { buf, head: 0 }
    }

    /// Record the newest state, evicting the oldest.
    pub fn push(&mut self, state: [f64; N]) {
        self.head = (self.head + 1) % C;
        self.buf[(self.head + C - 1) % C] = state;
    }

    /// The state `k` steps back, where `0` is the newest sample. `None` when `k >= C`.
    ///
    /// `k` is not reduced modulo `C` on the way in: the guard it sits behind has already
    /// established `k < C`, so a reduction would be the identity, and writing one would suggest a
    /// wrap this deliberately refuses to perform.
    #[must_use]
    pub fn delayed(&self, k: usize) -> Option<[f64; N]> {
        (k < C).then(|| self.buf[(self.head + C - 1 - k) % C])
    }

    /// The capacity, in samples.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        C
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::float_cmp)]

    use super::*;

    #[test]
    fn a_negative_or_non_finite_latency_is_refused_rather_than_clamped() {
        assert!(Latency::new(-1e-9, 0.0, 0.0).is_none());
        assert!(Latency::new(0.0, f64::NAN, 0.0).is_none());
        assert!(Latency::new(0.0, 0.0, f64::INFINITY).is_none());
        assert!(Latency::new(0.0, 0.0, 0.0).is_some());
    }

    #[test]
    fn the_three_latencies_are_kept_apart_and_only_summed_on_request() {
        let l = Latency::new(0.10, 0.02, 0.05).expect("finite and non-negative");
        assert!((l.sensor() - 0.10).abs() < 1e-15);
        assert!((l.compute() - 0.02).abs() < 1e-15);
        assert!((l.actuator() - 0.05).abs() < 1e-15);
        assert!((l.total() - 0.17).abs() < 1e-15);
    }

    /// The tolerance is arithmetic, so it has to admit a delay that is exactly `k · dt` in the reals
    /// and only nearly so in `f64` — and reject one that is genuinely between two steps.
    ///
    /// Finding a case that *exercises* the tolerance took searching, and the first candidate did not.
    /// `0.1 + 0.1 + 0.1` is famously `0.30000000000000004` rather than `0.3`, but `3.0 * 0.1` rounds
    /// to the very same value, so the comparison here misses by exactly zero and a tolerance of zero
    /// would pass it too. `0.05 + 2.9 + 0.7` against `dt = 0.05` is a case where the two genuinely
    /// differ: measured, the miss is 8.882e-16 against a tolerance of 3.242e-15, so the rule is doing
    /// work at 27 % of its budget.
    #[test]
    fn commensurability_admits_a_rounding_error_and_refuses_half_a_step() {
        let dt = 0.05;
        let l = Latency::new(0.05, 2.9, 0.7).expect("valid");
        let miss = libm::fabs(l.total() - 73.0 * dt);
        let tol = 4.0 * f64::EPSILON * l.total();
        println!("commensurability: miss {miss:.3e} against tolerance {tol:.3e}");
        assert!(
            miss > 0.0,
            "this case must not be exact, or it tests nothing"
        );
        assert!(miss <= tol);
        assert_eq!(l.steps(dt), Some(73));

        let half = Latency::new(0.025, 0.0, 0.0).expect("valid");
        assert_eq!(
            half.steps(0.01),
            None,
            "2.5 steps is not a whole number of them"
        );

        assert_eq!(Latency::none().steps(dt), Some(0));
        assert_eq!(l.steps(0.0), None, "a non-positive step has no step count");
        assert_eq!(l.steps(f64::NAN), None);
    }

    #[test]
    fn the_ring_returns_the_sample_that_many_steps_back() {
        let mut h = History::<1, 4>::seeded([0.0]);
        for i in 1..=3 {
            h.push([f64::from(i)]);
        }
        assert_eq!(h.delayed(0), Some([3.0]));
        assert_eq!(h.delayed(1), Some([2.0]));
        assert_eq!(h.delayed(2), Some([1.0]));
        assert_eq!(h.delayed(3), Some([0.0]), "still the seeded history");
        assert_eq!(h.delayed(4), None, "beyond capacity, not wrapped");
        assert_eq!(h.capacity(), 4);
    }

    /// Seeding is not decoration: before the loop has run for `τ`, every delayed read is the initial
    /// history, and a buffer that started at zero would feed the controller a state the vehicle was
    /// never in. Here the seed is a non-zero rate and the first three delayed reads must return it.
    #[test]
    fn the_seed_is_what_the_loop_reads_until_it_has_run_for_the_delay() {
        let mut h = History::<1, 8>::seeded([7.0]);
        h.push([1.0]);
        assert_eq!(h.delayed(0), Some([1.0]));
        for k in 1..8 {
            assert_eq!(h.delayed(k), Some([7.0]), "step {k} back is the seed");
        }
    }

    #[test]
    fn a_non_constant_seed_is_indexed_by_age() {
        let h = History::<1, 4>::seeded_with(|k| [f64::from(u32::try_from(k).expect("small"))]);
        for k in 0..4 {
            assert_eq!(
                h.delayed(k),
                Some([f64::from(u32::try_from(k).expect("small"))])
            );
        }
    }
    /// The three ways `steps` can refuse or admit a ratio that is not an ordinary small integer.
    ///
    /// A latency far below one ulp of the step is `0` steps and not a refusal — the loop is
    /// instantaneous to the resolution the caller chose. A ratio past `usize::MAX` is a refusal
    /// even though it is a perfectly exact whole number of steps, because there is no index to
    /// return; and the ceiling itself is the largest one that is not.
    #[test]
    fn the_step_count_admits_a_latency_below_one_ulp_and_refuses_one_past_the_index_range() {
        let dt = 1.0;
        let sub_ulp = Latency::new(1.0e-16, 0.0, 0.0).expect("finite and non-negative");
        assert!(
            sub_ulp.total() > 0.0,
            "the fixture must not take the zero-total path"
        );
        assert_eq!(sub_ulp.steps(dt), Some(0));

        #[allow(clippy::cast_precision_loss)]
        let ceiling = usize::MAX as f64;
        let at = Latency::new(ceiling, 0.0, 0.0).expect("finite and non-negative");
        println!("ceiling {ceiling:e}: steps {:?}", at.steps(dt));
        assert!(at.steps(dt).is_some(), "the ceiling itself was refused");

        let over = Latency::new(1.0, 0.0, 0.0).expect("finite and non-negative");
        assert_eq!(over.steps(1.0e-20), None);
    }

    /// The newest sample is readable straight after seeding, when `head` is still zero. Every other
    /// fixture pushes first, which moves `head` away from the one value that makes the ring index
    /// arithmetic run backwards through zero.
    #[test]
    fn the_newest_sample_is_readable_before_the_first_push() {
        let history = History::<2, 4>::seeded_with(|k| {
            #[allow(clippy::cast_precision_loss)]
            let age = k as f64;
            [age, -age]
        });
        assert_eq!(history.delayed(0), Some([0.0, 0.0]));
        assert_eq!(history.delayed(3), Some([3.0, -3.0]));
        assert_eq!(history.delayed(4), None);
    }
}
