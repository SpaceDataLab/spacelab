//! The largest latency at which the loop still comes back — the one number [`crate::Latency`]
//! exists to be compared against.
//!
//! # What is being computed
//!
//! Linearise one principal axis about the reference. The plant is `J δθ̈ = u`, and a law acting on
//! the state as it was `τ` ago gives
//!
//! ```text
//! J δθ̈(t) + k_ω δθ̇(t − τ) + k_θ δθ(t − τ) = 0
//! ```
//!
//! whose characteristic equation `J λ² + (k_ω λ + k_θ) e^{−λτ} = 0` is transcendental and has
//! infinitely many roots. At `τ = 0` they all sit in the left half plane; as `τ` grows a conjugate
//! pair walks right, and the **delay margin** is the `τ` at which it reaches the imaginary axis.
//! Setting `λ = jω` and eliminating `τ` between the real and imaginary parts leaves a polynomial
//! that `derivations/delay_margin.mac` derives and factors:
//!
//! ```text
//! J²ω⁴ − k_ω²ω² − k_θ² = 0      ⟹   ω_c² = (k_ω² + √(k_ω⁴ + 4J²k_θ²)) / (2J²)
//! ```
//!
//! and the phase then gives `τ = atan2(k_ω ω_c, k_θ) / ω_c`. Both branches of the real part are
//! positive at the crossing, so `ω_c τ` is in the first quadrant and this is the *smallest* positive
//! delay that destabilises the loop, not some later one.
//!
//! # Two results worth stating outright
//!
//! **The margin is not monotone in the damping.** `τ_max · ω_n` depends only on the damping ratio,
//! and it peaks at **ζ = 0.6394** with `τ_max ω_n = 0.7415`, against **0.6474** at critical damping —
//! 14.5 % more margin at less damping. Past that the rate term's own phase lag under delay costs
//! more than the damping buys. "More damping is safer" is the natural assumption and it is false
//! here, which is why [`SecondOrderLoop::normalised_delay_margin`] exists as its own function.
//!
//! **An undamped loop has exactly zero margin.** With `k_ω = 0` the formula gives `ω_c = ω_n` and
//! `τ = atan2(0, k_θ)/ω_c = 0`: a marginally-stable second-order system is destabilised by *any*
//! delay. Not a special case in the code — it falls out of the same expression.
//!
//! # What this does not cover
//!
//! One axis, linearised, constant delay, and a delay on the whole feedback path rather than on the
//! measurement and the actuator separately. For a three-axis body the binding axis is the one with
//! the **smallest** principal moment, because `τ_max ∝ 1/ω_n ∝ √(J/k_θ)` at fixed gains —
//! [`worst_principal_axis`] does that minimisation and nothing more. A non-diagonal inertia tensor
//! has to be diagonalised by the caller; this module will not pretend its diagonal is its spectrum.

use spacelab_propagation::attitude::Vec3;

/// The gains of a control law's linearisation about the reference attitude.
///
/// Separate from [`crate::ControlLaw`] on purpose. A linearisation is an approximation with a
/// domain, and a law is free not to have one — a bang-bang or a saturating law does not. Making it
/// a required method of the law trait would advertise a small-angle model as if it were free.
pub trait SmallAngleGains {
    /// `(k_θ, k_ω)` in N·m/rad and N·m·s/rad: the torque per unit of small angle error, and per unit
    /// of rate error.
    fn small_angle_gains(&self) -> (f64, f64);
}

/// One principal axis of a rate-and-angle feedback loop, linearised.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecondOrderLoop {
    moment: f64,
    angle_gain: f64,
    rate_gain: f64,
}

impl SecondOrderLoop {
    /// `None` unless the moment and the angle gain are finite and strictly positive and the rate
    /// gain is finite and non-negative.
    ///
    /// A zero rate gain is admitted rather than refused, because the answer it produces — zero
    /// margin — is the correct one and is worth being able to compute.
    #[must_use]
    pub fn new(moment: f64, angle_gain: f64, rate_gain: f64) -> Option<Self> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        (positive(moment) && positive(angle_gain) && rate_gain.is_finite() && rate_gain >= 0.0)
            .then_some(Self {
                moment,
                angle_gain,
                rate_gain,
            })
    }

    /// The loop a control law closes around a principal axis of moment `moment`.
    #[must_use]
    pub fn of(law: &impl SmallAngleGains, moment: f64) -> Option<Self> {
        let (angle_gain, rate_gain) = law.small_angle_gains();
        Self::new(moment, angle_gain, rate_gain)
    }

    /// `ω_n = √(k_θ / J)`, rad/s.
    #[must_use]
    pub fn natural_frequency(&self) -> f64 {
        libm::sqrt(self.angle_gain / self.moment)
    }

    /// `ζ = k_ω / (2 √(J k_θ))`.
    #[must_use]
    pub fn damping_ratio(&self) -> f64 {
        self.rate_gain / (2.0 * libm::sqrt(self.moment * self.angle_gain))
    }

    /// The delay margin and the frequency at which the loop crosses.
    #[must_use]
    pub fn delay_margin(&self) -> DelayMargin {
        // `hypot(k_ω², 2 J k_θ)` is `√(k_ω⁴ + 4J²k_θ²)` without ever forming `k_ω⁴`, which overflows
        // at a rate gain of 1e77 and loses precision long before that. Both terms of the sum are
        // non-negative, so there is no cancellation anywhere in this expression.
        let rate2 = self.rate_gain * self.rate_gain;
        let root = libm::hypot(rate2, 2.0 * self.moment * self.angle_gain);
        // `√(x/2)/J` rather than `√(x/(2J²))`: same value, and it does not square the moment.
        let crossing_frequency = libm::sqrt(f64::midpoint(rate2, root)) / self.moment;
        let seconds =
            libm::atan2(self.rate_gain * crossing_frequency, self.angle_gain) / crossing_frequency;
        DelayMargin {
            seconds,
            crossing_frequency,
        }
    }

    /// `τ_max · ω_n`, which depends only on the damping ratio.
    ///
    /// The scale-free form of the answer, and the one to quote: it says the margin is a fixed
    /// fraction of the loop's own period, so buying bandwidth spends margin proportionally. It is
    /// also where the non-monotonicity in `ζ` is visible — see the module documentation.
    #[must_use]
    pub fn normalised_delay_margin(&self) -> f64 {
        self.delay_margin().seconds() * self.natural_frequency()
    }
}

/// The delay at which a loop's dominant pole reaches the imaginary axis, and where it crosses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayMargin {
    seconds: f64,
    crossing_frequency: f64,
}

impl DelayMargin {
    /// The margin, in seconds. Compare it against [`crate::Latency::total`].
    #[must_use]
    pub const fn seconds(&self) -> f64 {
        self.seconds
    }

    /// The frequency at which the pole pair crosses, rad/s — the frequency the loop rings at when
    /// the delay is exactly the margin.
    #[must_use]
    pub const fn crossing_frequency(&self) -> f64 {
        self.crossing_frequency
    }
}

/// The binding axis of a three-axis body and its margin: the index of the principal moment with the
/// smallest margin, and that margin.
///
/// At fixed gains `τ_max ∝ 1/ω_n ∝ √(J/k_θ)`, so the answer is always the smallest moment. It is
/// computed rather than assumed, because the relation holds for these laws and not for a law whose
/// gains are scheduled on the inertia.
///
/// `None` if any moment is not finite and positive, or if the gains are not admissible.
#[must_use]
pub fn worst_principal_axis(
    law: &impl SmallAngleGains,
    moments: Vec3,
) -> Option<(usize, DelayMargin)> {
    let mut worst: Option<(usize, DelayMargin)> = None;
    for (axis, moment) in moments.into_iter().enumerate() {
        let margin = SecondOrderLoop::of(law, moment)?.delay_margin();
        if worst.is_none_or(|(_, best)| margin.seconds() < best.seconds()) {
            worst = Some((axis, margin));
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    // The reference values are Maxima's 40-digit bigfloats truncated to what `f64` can hold. They
    // are transcribed at the CAS's precision rather than at the type's, so that widening the
    // assertion later does not silently re-round the oracle.
    #![allow(clippy::excessive_precision, clippy::expect_used, clippy::float_cmp)]

    use super::*;
    use crate::attitude::{MrpLyapunov, QuaternionPd};

    /// The Maxima run in `derivations/delay_margin.mac`, 40 significant digits.
    ///
    /// It does not merely restate the same formula: it eliminates `τ` between the real and
    /// imaginary parts by resultant, factors the result, and then substitutes the recovered
    /// `(ω_c, τ_max)` back into the *original transcendental* equation. That residual is 2.296e-41
    /// for the first case and 8.968e-44 for the cubesat — so the pair really is a crossing, not a
    /// solution of a rearrangement that lost a branch.
    const CRITICAL_UNIT: (f64, f64) = (1.455_346_690_225_354_808, 9.155_748_756_873_190_881e-1);

    /// `τ_max · ω_n` against damping ratio, same run. Non-monotone: it rises to ζ ≈ 0.64 and falls.
    const NORMALISED: [(f64, f64); 8] = [
        (0.125, 2.448_251_508_099_830_653e-1),
        (0.25, 4.594_888_917_496_084_290e-1),
        (0.5, 7.111_186_487_157_930_191e-1),
        (0.75, 7.280_753_460_980_845_694e-1),
        (1.0, 6.474_092_032_825_335_893e-1),
        (1.5, 4.839_914_895_787_282_487e-1),
        (2.0, 3.763_929_901_981_862_884e-1),
        (4.0, 1.943_730_930_834_528_366e-1),
    ];

    /// The unit case: `J = 1`, quaternion PD with `K = 1` so `k_θ = ½`, critically damped so
    /// `k_ω = P = √(2 J K) = √2`.
    fn unit_loop() -> SecondOrderLoop {
        let law = QuaternionPd::critically_damped(1.0, 1.0).expect("valid gains");
        SecondOrderLoop::of(&law, 1.0).expect("valid loop")
    }

    #[test]
    fn the_crossing_matches_the_cas_of_record() {
        let margin = unit_loop().delay_margin();
        let (omega, tau) = CRITICAL_UNIT;
        let d_omega = (margin.crossing_frequency() - omega).abs() / omega;
        let d_tau = (margin.seconds() - tau).abs() / tau;
        println!("vs Maxima: omega_c rel {d_omega:.3e}, tau_max rel {d_tau:.3e}");
        assert!(d_omega < 1e-15, "crossing frequency rel {d_omega:.3e}");
        assert!(d_tau < 1e-15, "delay margin rel {d_tau:.3e}");
    }

    /// The residual of the transcendental characteristic equation at the computed crossing, in
    /// `f64` this time rather than in 40-digit bigfloats. It has to vanish to rounding, and it is
    /// the only check here that does not go through the closed form the code implements.
    #[test]
    fn the_characteristic_equation_vanishes_at_the_computed_crossing() {
        let mut worst: f64 = 0.0;
        for &(zeta, _) in &NORMALISED {
            let (moment, angle_gain) = (1.0, 1.0);
            let rate_gain = 2.0 * zeta * libm::sqrt(moment * angle_gain);
            let loop_ = SecondOrderLoop::new(moment, angle_gain, rate_gain).expect("valid");
            let m = loop_.delay_margin();
            let (w, t) = (m.crossing_frequency(), m.seconds());
            // J λ² + (k_ω λ + k_θ) e^{−λτ} at λ = jω, split by hand.
            let (sin, cos) = (libm::sin(w * t), libm::cos(w * t));
            let real = -moment * w * w + angle_gain * cos + rate_gain * w * sin;
            let imag = rate_gain * w * cos - angle_gain * sin;
            let scale = moment * w * w;
            worst = worst.max((libm::hypot(real, imag) / scale).abs());
        }
        println!("worst |char. eq.| / J omega^2 over the 8 damping ratios: {worst:.3e}");
        assert!(worst < 1e-15, "residual {worst:.3e}");
    }

    /// The margin is a fixed fraction of the loop's own period, and the fraction is *not* largest at
    /// critical damping. Both halves are published claims, so both are asserted.
    #[test]
    fn the_normalised_margin_depends_only_on_the_damping_and_peaks_below_critical() {
        let mut worst: f64 = 0.0;
        for &(zeta, expected) in &NORMALISED {
            // Two loops with the same ζ and different scales: same normalised margin.
            for (moment, angle_gain) in [(1.0, 1.0), (0.2, 2.0e-3)] {
                let rate_gain = 2.0 * zeta * libm::sqrt(moment * angle_gain);
                let loop_ = SecondOrderLoop::new(moment, angle_gain, rate_gain).expect("valid");
                assert!((loop_.damping_ratio() - zeta).abs() < 1e-15);
                worst = worst.max((loop_.normalised_delay_margin() - expected).abs() / expected);
            }
        }
        println!("worst normalised-margin deviation from Maxima: {worst:.3e}");
        assert!(worst < 1e-14, "deviation {worst:.3e}");

        let at = |zeta: f64| {
            SecondOrderLoop::new(1.0, 1.0, 2.0 * zeta)
                .expect("valid")
                .normalised_delay_margin()
        };
        assert!(
            at(0.64) > at(1.0),
            "the margin is not monotone in the damping: {:.6} at zeta = 0.64 against {:.6} at \
             critical damping",
            at(0.64),
            at(1.0)
        );
        assert!(
            at(0.64) > at(0.25) && at(0.64) > at(2.0),
            "0.64 is the peak"
        );
    }

    #[test]
    fn an_undamped_loop_has_exactly_no_margin() {
        let loop_ = SecondOrderLoop::new(1.0, 1.0, 0.0).expect("a zero rate gain is admissible");
        let m = loop_.delay_margin();
        assert_eq!(m.seconds(), 0.0, "any delay destabilises a marginal loop");
        assert!((m.crossing_frequency() - loop_.natural_frequency()).abs() < 1e-15);
    }

    #[test]
    fn a_gain_or_moment_that_is_not_admissible_is_refused() {
        assert!(SecondOrderLoop::new(0.0, 1.0, 1.0).is_none());
        assert!(SecondOrderLoop::new(-1.0, 1.0, 1.0).is_none());
        assert!(SecondOrderLoop::new(1.0, 0.0, 1.0).is_none());
        assert!(SecondOrderLoop::new(1.0, 1.0, -1e-30).is_none());
        assert!(SecondOrderLoop::new(f64::NAN, 1.0, 1.0).is_none());
        assert!(SecondOrderLoop::new(1.0, f64::INFINITY, 1.0).is_none());
        assert!(SecondOrderLoop::new(1.0, 1.0, f64::NAN).is_none());
    }

    /// The MRP law's `σ → δθ/4` against the quaternion law's `q_v → δθ/2`: at the same nominal `K`
    /// the MRP loop is half as stiff, so it is slower and its margin is larger by `√2`.
    #[test]
    fn the_two_laws_linearise_to_different_angle_gains() {
        let pd = QuaternionPd::new(1.0, 1.0).expect("valid");
        let mrp = MrpLyapunov::new(1.0, 1.0).expect("valid");
        assert_eq!(pd.small_angle_gains(), (0.5, 1.0));
        assert_eq!(mrp.small_angle_gains(), (0.25, 1.0));

        let ratio = SecondOrderLoop::of(&mrp, 1.0)
            .expect("valid")
            .natural_frequency()
            / SecondOrderLoop::of(&pd, 1.0)
                .expect("valid")
                .natural_frequency();
        assert!((ratio - 0.5_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn the_binding_axis_is_the_smallest_principal_moment() {
        let law = QuaternionPd::new(1.0, 1.0).expect("valid");
        let (axis, margin) = worst_principal_axis(&law, [10.0, 12.0, 8.0]).expect("valid");
        assert_eq!(axis, 2, "the 8 kg m^2 axis");
        let alone = SecondOrderLoop::of(&law, 8.0)
            .expect("valid")
            .delay_margin();
        assert_eq!(margin, alone);
        assert!(worst_principal_axis(&law, [10.0, -1.0, 8.0]).is_none());

        // Two axes tied at the minimum. The margins are bit-identical — the same moment through
        // the same arithmetic — so the answer is decided entirely by the tie-break, and the first
        // of the tied axes is the one reported.
        let (tied, _) = worst_principal_axis(&law, [8.0, 8.0, 10.0]).expect("valid");
        assert_eq!(tied, 0, "a tie must resolve to the first of the tied axes");
        let (last, _) = worst_principal_axis(&law, [10.0, 8.0, 8.0]).expect("valid");
        assert_eq!(last, 1, "a tie must resolve to the first of the tied axes");
    }

    /// A 12U cubesat sized against a real latency budget — the number this ticket exists to produce.
    ///
    /// `J = 0.20 kg m²` (smallest principal moment), control bandwidth `ω_n = 0.10 rad/s`,
    /// critically damped: `K = 2 J ω_n² = 4.0e-3 N·m/rad`, `P = 2 J ω_n = 4.0e-2 N·m·s/rad`. A
    /// plausible worst-case sensor-to-actuator budget is a 0.25 s star-tracker integration, 0.10 s
    /// of processing and a 0.20 s wheel command lag: 0.55 s.
    #[test]
    fn a_realistic_reaction_wheel_loop_has_margin_to_spare_and_the_ratio_says_how_much() {
        let (moment, natural) = (0.20, 0.10);
        let law = QuaternionPd::new(2.0 * moment * natural * natural, 2.0 * moment * natural)
            .expect("valid gains");
        let loop_ = SecondOrderLoop::of(&law, moment).expect("valid loop");
        let margin = loop_.delay_margin();

        let budget = crate::Latency::new(0.25, 0.10, 0.20).expect("valid latency");
        let ratio = margin.seconds() / budget.total();
        // The bandwidth at which the budget *is* the margin: τ_max ω_n is fixed, so this is a
        // division rather than a search.
        let breaking_bandwidth = loop_.normalised_delay_margin() / budget.total();
        println!(
            "12U: omega_n {:.4} rad/s, zeta {:.3}, tau_max {:.4} s, budget {:.2} s, ratio {ratio:.2}x, \
             loop breaks at omega_n = {breaking_bandwidth:.3} rad/s",
            loop_.natural_frequency(),
            loop_.damping_ratio(),
            margin.seconds(),
            budget.total()
        );
        assert!((loop_.natural_frequency() - natural).abs() < 1e-15);
        assert!((margin.seconds() - 6.474_092_032_825_335_893).abs() < 1e-13);
        assert!(ratio > 11.0, "the margin is {ratio:.2}x the budget");
        assert!(
            (breaking_bandwidth - 1.177_107_642_331_879_253).abs() < 1e-12,
            "a loop faster than this rad/s is unstable on this budget"
        );
    }

    /// The independent leg: integrate the delay differential equation the margin is *about*, and
    /// find the crossing by watching the solution grow or decay.
    ///
    /// This shares no algebra with [`SecondOrderLoop::delay_margin`] — it is the method of steps
    /// from [`crate::delayed`] driving the closed loop, and the answer comes out of the trajectory
    /// rather than out of a formula. A sign error in the phase recovery, or the wrong branch of the
    /// quartic, would put the analytic `τ_max` somewhere the solution is plainly decaying.
    #[cfg(feature = "std")]
    #[test]
    fn integrating_the_delayed_loop_finds_the_same_crossing() {
        use spacelab_propagation::{Dop853, Tolerance, driver::Controls};

        use crate::delayed::{DelayedSystem, InitialValue, Past, integrate_delayed};

        /// `J θ̈ = −k_θ θ(t − τ) − k_ω θ̇(t − τ)`, as the state `[θ, θ̇]`.
        struct Loop {
            tau: [f64; 1],
            moment: f64,
            angle_gain: f64,
            rate_gain: f64,
        }
        impl DelayedSystem<2> for Loop {
            fn delays(&self) -> &[f64] {
                &self.tau
            }
            fn derivative(&self, t: f64, y: &[f64; 2], past: &impl Past<2>, dydt: &mut [f64; 2]) {
                let old = past.at(t - self.tau[0]);
                dydt[0] = y[1];
                dydt[1] = -(self.angle_gain * old[0] + self.rate_gain * old[1]) / self.moment;
            }
        }

        let (moment, angle_gain, zeta) = (1.0, 1.0, 1.0);
        let rate_gain = 2.0 * zeta * libm::sqrt(moment * angle_gain);
        let analytic = SecondOrderLoop::new(moment, angle_gain, rate_gain)
            .expect("valid")
            .delay_margin();

        // Fifteen ring cycles: long enough for the growth rate to separate from the transient, short
        // enough that a run below the margin has not decayed into rounding.
        let span = 15.0 * core::f64::consts::TAU / analytic.crossing_frequency();
        let controls = Controls::new(Tolerance::new(1e-11, 1e-11), 1e-10);

        // The log-amplitude slope over the second half of the run: positive means the pole pair has
        // crossed. The amplitude is the phase-plane radius scaled by the crossing frequency, so it
        // does not dip to zero between peaks and confuse the fit.
        let growth = |tau: f64| -> f64 {
            let sys = Loop {
                tau: [tau],
                moment,
                angle_gain,
                rate_gain,
            };
            let sol = integrate_delayed(
                &Dop853,
                &sys,
                InitialValue::at_rest(0.0, [1.0e-2, 0.0]),
                span,
                0.05,
                controls,
            )
            .expect("the delayed loop integrates");
            let amplitude = |t: f64| {
                let y = sol.eval(t);
                libm::log(libm::hypot(y[0], y[1] / analytic.crossing_frequency()))
            };
            let (a, b) = (0.5 * span, span);
            (amplitude(b) - amplitude(a)) / (b - a)
        };

        assert!(
            growth(0.90 * analytic.seconds()) < 0.0,
            "below the margin it must decay"
        );
        assert!(
            growth(1.10 * analytic.seconds()) > 0.0,
            "above it must grow"
        );

        // Bisect on the sign of the growth rate. Twenty halvings of a 40 % bracket is 4e-7 of it.
        let (mut lo, mut hi) = (0.80 * analytic.seconds(), 1.20 * analytic.seconds());
        for _ in 0..20 {
            let mid = f64::midpoint(lo, hi);
            if growth(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let measured = f64::midpoint(lo, hi);
        let deviation = (measured - analytic.seconds()).abs() / analytic.seconds();
        println!(
            "delay margin: analytic {:.9} s, from the integrated DDE {measured:.9} s, \
             relative {deviation:.3e}",
            analytic.seconds()
        );
        assert!(
            deviation < 1e-4,
            "the integrated loop crosses at {measured:.9} s, the formula says {:.9} s",
            analytic.seconds()
        );
    }
}
