//! The ring: plant, sensor delay, law, actuator delay, back to the plant.
//!
//! # Why the loop is a type and the controller is only a trait
//!
//! A control law is stateless algebra — error in, torque out. Everything that makes a real loop
//! behave unlike the textbook lives *between* the law and the plant: the measurement describes a
//! state that is already old, the decision takes time, and the actuator answers late.
//! [`AttitudeLoop`] owns that arrangement so a law never has to know it exists, which is also what
//! lets the same law be run at several latencies and compared.
//!
//! # The seed is a boundary condition
//!
//! Before the loop has been running for `τ` there is no measured past to read, and what the
//! controller sees instead is the initial history — supplied to [`AttitudeLoop::run`] rather than
//! defaulted, because it is part of the problem statement. A loop whose vehicle starts 30° off but
//! whose history says "at the reference" commands exactly zero torque for the first `τ`, and that is
//! the correct answer for that problem, not a bug to be smoothed over.
//!
//! # Fixed step, exact delay
//!
//! The plant is advanced by [`spacelab_propagation::attitude::integrate_attitude_torqued`] at a
//! fixed step, so a delay that is a whole number of steps is a stored sample and carries no
//! interpolation error at all. [`crate::Latency::steps`] refuses a delay that is not, and the loop
//! refuses to be built on one — the machinery for a non-commensurate delay is dense output plus
//! breakpoint tracking, and it is deliberately elsewhere.

use crate::attitude::{AttitudeError, ControlLaw};
use crate::latency::{History, Latency};
use spacelab_propagation::attitude::{Inertia, Mat3, Vec3, integrate_attitude_torqued};

/// A rigid body's orientation and rate — one point of the plant's state.
///
/// It exists as a type rather than a `(Mat3, Vec3)` pair because [`AttitudeLoop::run`] takes two of
/// them, the true initial condition and the initial history, and a bare pair at each of those
/// positions is an argument order waiting to be swapped silently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudeState {
    /// Inertial→body change of basis.
    pub rotation: Mat3,
    /// Body rate in body axes, rad/s.
    pub rate: Vec3,
}

impl AttitudeState {
    /// A state at `rotation` with rate `rate`.
    #[must_use]
    pub const fn new(rotation: Mat3, rate: Vec3) -> Self {
        Self { rotation, rate }
    }

    /// The flat 12-element encoding a [`History`] stores: nine rotation entries row-major, then the
    /// three rate components.
    fn pack(&self) -> [f64; 12] {
        let mut out = [0.0; 12];
        for i in 0..3 {
            for j in 0..3 {
                out[i * 3 + j] = self.rotation[i][j];
            }
        }
        out[9..12].copy_from_slice(&self.rate);
        out
    }

    /// The inverse of [`AttitudeState::pack`].
    fn unpack(state: &[f64; 12]) -> Self {
        let mut rotation = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rotation[i][j] = state[i * 3 + j];
            }
        }
        Self {
            rotation,
            rate: [state[9], state[10], state[11]],
        }
    }
}

/// One sample of a closed-loop run, handed to the observer at every step.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Time since the run started, seconds.
    pub time: f64,
    /// The true state at `time`.
    pub state: AttitudeState,
    /// The error the controller acted on — of the *delayed* state, when there is latency.
    pub error: AttitudeError,
    /// The moment that was commanded, N·m, body axes.
    pub torque: Vec3,
}

/// A rigid-body attitude loop at a fixed rate and a fixed latency.
#[derive(Debug, Clone, Copy)]
pub struct AttitudeLoop<C> {
    law: C,
    latency: Latency,
    step: f64,
    lag: usize,
}

impl<C: ControlLaw> AttitudeLoop<C> {
    /// `None` unless `step` is finite and positive and `latency` is a whole number of steps of it.
    #[must_use]
    pub fn new(law: C, latency: Latency, step: f64) -> Option<Self> {
        let lag = latency.steps(step)?;
        Some(Self {
            law,
            latency,
            step,
            lag,
        })
    }

    /// The loop delay in whole steps.
    #[must_use]
    pub const fn lag(&self) -> usize {
        self.lag
    }

    /// The latency this loop was built with.
    #[must_use]
    pub const fn latency(&self) -> Latency {
        self.latency
    }

    /// Run `steps` steps from `initial` towards a fixed `reference`, with `seed` as the initial
    /// history the controller reads until the delay has elapsed.
    ///
    /// `H` is the history capacity in samples and must exceed the lag; `None` when it does not,
    /// rather than a wrapped read of a state from a different era.
    ///
    /// `observe` is called once per step, after the torque for that step has been chosen and before
    /// the plant is advanced by it.
    pub fn run<const H: usize>(
        &self,
        initial: AttitudeState,
        seed: AttitudeState,
        reference: &Mat3,
        inertia: &Inertia,
        steps: usize,
        mut observe: impl FnMut(&Sample),
    ) -> Option<AttitudeState> {
        if self.lag >= H {
            return None;
        }
        let mut history = History::<12, H>::seeded(seed.pack());
        let mut state = initial;

        for n in 0..steps {
            history.push(state.pack());
            let measured = AttitudeState::unpack(&history.delayed(self.lag)?);
            let error = AttitudeError::new(&measured.rotation, reference, measured.rate, [0.0; 3]);
            let torque = self.law.torque(&error, inertia);

            #[allow(clippy::cast_precision_loss)]
            let time = self.step * n as f64;
            observe(&Sample {
                time,
                state,
                error,
                torque,
            });

            let (rotation, rate) = integrate_attitude_torqued(
                &state.rotation,
                state.rate,
                time,
                inertia,
                self.step,
                1,
                |_, _, _| torque,
            );
            state = AttitudeState { rotation, rate };
        }
        Some(state)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::float_cmp)]

    use super::*;
    use crate::attitude::{MrpLyapunov, QuaternionPd};
    use spacelab_propagation::attitude::exp_so3;

    const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn inertia() -> Inertia {
        Inertia::principal([10.0, 12.0, 8.0]).expect("positive definite")
    }

    /// The verdict the module exists for, and it is measured rather than derived. Schaub–Junkins
    /// promises `V̇ = −P‖δω‖² ≤ 0` for *any* positive gains and *any* initial error; the test starts
    /// 120° away with a tumble on all three axes — far outside anything a linearisation covers — and
    /// asserts that the measured `V` never rises.
    ///
    /// Measured on this case: `V` falls from 0.3134 to 4.902e-7 over 4000 steps of 0.05 s, and the
    /// largest single-step *increase* anywhere in the run is 0.000e0. The sequence is monotone, not
    /// merely trending — which is the difference between checking the theorem and checking that the
    /// controller happens to work.
    #[test]
    fn the_lyapunov_function_never_rises_from_a_large_tumbling_error() {
        let law = MrpLyapunov::new(0.5, 4.0).expect("valid");
        let i = inertia();
        let start = exp_so3([0.0, 0.0, 120.0_f64.to_radians()]);
        let loop_ = AttitudeLoop::new(law, Latency::none(), 0.05).expect("commensurate");

        let mut values = alloc_vec();
        loop_
            .run::<4>(
                AttitudeState::new(start, [0.05, -0.04, 0.03]),
                AttitudeState::new(start, [0.05, -0.04, 0.03]),
                &IDENTITY,
                &i,
                4000,
                |s| values.push(law.lyapunov(&s.error, &i)),
            )
            .expect("capacity exceeds the zero lag");

        let mut worst_rise = 0.0_f64;
        for pair in values.windows(2) {
            worst_rise = worst_rise.max(pair[1] - pair[0]);
        }
        println!(
            "lyapunov: V0 {:.4} -> V_end {:.3e}, worst single-step rise {:.3e}",
            values[0],
            values[values.len() - 1],
            worst_rise
        );
        assert!(
            worst_rise <= 0.0,
            "V rose by {worst_rise} somewhere in the run"
        );
        assert!(
            values[values.len() - 1] < 1e-4 * values[0],
            "V ended at {} from {}",
            values[values.len() - 1],
            values[0]
        );
    }

    /// The sharp check on the PD law: at small angles the closed loop *is* `I δθ̈ + P δθ̇ + (K/2)δθ = 0`,
    /// so a critically damped design has the closed-form response `δθ(t) = δθ₀(1 + ωₙt)e^{−ωₙt}` and
    /// the simulation has to reproduce it pointwise. This is an identity, not a tolerance: the only
    /// discrepancies are the `sin(δθ/2) ≈ δθ/2` truncation and the integrator's own error.
    ///
    /// Measured with `δθ₀ = 0.5°`: worst relative deviation 2.672e-4 over the whole transient.
    #[test]
    fn a_critically_damped_small_slew_matches_the_analytic_second_order_response() {
        let moment = 10.0;
        let i = Inertia::principal([moment, moment, moment]).expect("positive definite");
        let law = QuaternionPd::critically_damped(0.4, moment).expect("valid");
        let wn = law.natural_frequency(moment);
        assert!((law.damping_ratio(moment) - 1.0).abs() < 1e-14);

        let theta0 = 0.5_f64.to_radians();
        let start = exp_so3([0.0, 0.0, theta0]);
        let loop_ = AttitudeLoop::new(law, Latency::none(), 0.01).expect("commensurate");

        let mut worst = 0.0_f64;
        loop_
            .run::<4>(
                AttitudeState::new(start, [0.0; 3]),
                AttitudeState::new(start, [0.0; 3]),
                &IDENTITY,
                &i,
                3000,
                |s| {
                    let analytic = theta0 * (1.0 + wn * s.time) * libm::exp(-wn * s.time);
                    worst = worst.max((s.error.angle() - analytic).abs() / theta0);
                },
            )
            .expect("capacity exceeds the zero lag");

        println!("second-order: worst relative deviation {worst:.3e}");
        assert!(
            worst < 5e-3,
            "worst relative deviation from the analytic response was {worst}"
        );
    }

    /// A value test, not a difference test. The vehicle starts 30° off; the history says it is at
    /// the reference. For the first `lag` steps the controller therefore sees zero error and
    /// commands **exactly** zero torque — a number, not a smaller number. A loop that ignored its
    /// latency would command the full 30°-error torque on the very first step.
    #[test]
    fn until_the_delay_has_elapsed_the_controller_acts_on_the_seed_and_commands_nothing() {
        let law = QuaternionPd::new(0.4, 4.0).expect("valid");
        let i = inertia();
        let start = exp_so3([0.0, 0.0, 30.0_f64.to_radians()]);
        let latency = Latency::new(0.10, 0.05, 0.05).expect("valid");
        let loop_ = AttitudeLoop::new(law, latency, 0.01).expect("20 steps");
        assert_eq!(loop_.lag(), 20);

        let mut torques = alloc_vec();
        loop_
            .run::<32>(
                AttitudeState::new(start, [0.0; 3]),
                AttitudeState::new(IDENTITY, [0.0; 3]),
                &IDENTITY,
                &i,
                40,
                |s| torques.push(s.torque),
            )
            .expect("capacity exceeds the lag");

        for (n, t) in torques.iter().enumerate().take(20) {
            let magnitude = libm::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
            assert!(magnitude == 0.0, "step {n} commanded {magnitude}, not zero");
        }
        let after = torques[20];
        let magnitude = libm::sqrt(after[0] * after[0] + after[1] * after[1] + after[2] * after[2]);
        assert!(
            magnitude > 0.05,
            "the first post-delay step commanded {magnitude}"
        );
    }

    /// The ring stores a flat twelve, so the encoding has to be its own inverse or the controller
    /// reads a transposed rotation and a permuted rate with nothing to say so.
    #[test]
    fn the_flat_encoding_round_trips_and_is_row_major() {
        let state = AttitudeState::new(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [0.1, -0.2, 0.3],
        );
        let packed = state.pack();
        assert_eq!(
            packed,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.1, -0.2, 0.3],
            "row-major rotation then rate"
        );
        assert_eq!(AttitudeState::unpack(&packed), state);
    }

    #[test]
    fn a_history_shorter_than_the_lag_is_refused_rather_than_wrapped() {
        let law = QuaternionPd::new(0.4, 4.0).expect("valid");
        let latency = Latency::new(0.1, 0.0, 0.0).expect("valid");
        let loop_ = AttitudeLoop::new(law, latency, 0.01).expect("10 steps");
        assert_eq!(loop_.lag(), 10);
        assert!(
            loop_
                .run::<4>(
                    AttitudeState::new(IDENTITY, [0.0; 3]),
                    AttitudeState::new(IDENTITY, [0.0; 3]),
                    &IDENTITY,
                    &inertia(),
                    5,
                    |_| {},
                )
                .is_none()
        );
    }

    #[test]
    fn a_non_commensurate_latency_never_becomes_a_loop() {
        let law = QuaternionPd::new(0.4, 4.0).expect("valid");
        let latency = Latency::new(0.025, 0.0, 0.0).expect("valid");
        assert!(AttitudeLoop::new(law, latency, 0.01).is_none());
        assert!(AttitudeLoop::new(law, Latency::none(), 0.0).is_none());
    }

    fn alloc_vec<T>() -> std::vec::Vec<T> {
        std::vec::Vec::new()
    }
}
