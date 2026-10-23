//! Closed-loop control — the feedback layer the workspace did not have.
//!
//! # The gap
//!
//! [`spacelab_astro`] generates trajectories: Lambert arcs, low-thrust primer-vector solutions, FNPEG
//! entry guidance, powered descent, the Vary/Achieve corrector. [`spacelab_propagation`] integrates a
//! plant and prescribes a *reference* orientation through
//! [`spacelab_propagation::attitude_law::AttitudeProvider`]. [`spacelab_estimation`] estimates state
//! from measurements. What no crate did was close the ring: take the error between where the vehicle
//! is and where it was told to be, and produce the actuation that removes it.
//!
//! Without that ring there is no such thing as a delay margin, a settling time, or a station-keeping
//! budget, because each of those is a property of a *loop* rather than of any one of its parts.
//!
//! [`spacelab_astro`]: https://docs.rs/spacelab-astro
//! [`spacelab_estimation`]: https://docs.rs/spacelab-estimation
//!
//! # Latency is the load-bearing idea, not the controller
//!
//! A control law is a few lines of algebra; the interesting behaviour lives in *when* each of those
//! lines sees its input. A real loop reads a sensor that integrated light some time ago, spends
//! processor time deciding, and commands an actuator that responds with its own lag. [`Latency`]
//! keeps those three as three numbers rather than one, because they have different physical origins
//! and different fixes: a star tracker's integration time is bought down with aperture, compute
//! latency with a faster processor, actuator lag with a different wheel. Collapsing them into a
//! single `τ` gives a margin nobody can act on.
//!
//! # Two paths to a delayed state, and what is still missing
//!
//! [`closed_loop`] advances the plant at a **fixed step**, which makes the delayed lookup exact
//! whenever the delay is a whole number of steps: the state at `t − τ` is a stored sample, not an
//! interpolation. That is the commensurate case, it is the case an ADCS loop running at a fixed rate
//! is actually in, and it costs no accuracy. [`Latency::steps`] refuses a non-commensurate delay
//! rather than rounding one silently.
//!
//! The non-commensurate case has its own path. [`delayed`] is the method of steps over an adaptive
//! integrator's dense output, with the breakpoints at `t = kτ` tracked — a mismatched initial history
//! puts a jump in `x'` at the origin, in `x''` at `τ` and in `x'''` at `2τ`, and a mesh that steps
//! over one loses the method's order. [`margin`] and [`formation`] are both measured through it.
//!
//! What is still absent is the join: [`closed_loop`] does not run on [`delayed`], so a loop whose
//! latency is not a whole number of steps cannot be simulated end to end with the estimator in it.
//! Neither of the two delay results published so far needs that — both linearise to a plant the DDE
//! integrator drives directly — so it is a gap and not yet a limitation.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod attitude;
pub mod closed_loop;
#[cfg(feature = "std")]
pub mod delayed;
#[cfg(feature = "std")]
pub mod formation;
pub mod latency;
pub mod margin;
pub mod orbit;

pub use attitude::{AttitudeError, ControlLaw, MrpLyapunov, QuaternionPd};
pub use closed_loop::{AttitudeLoop, AttitudeState, Sample};
#[cfg(feature = "std")]
pub use delayed::{
    ConstantHistory, DelayError, DelayedSolution, DelayedSystem, InitialValue, Past,
};
#[cfg(feature = "std")]
pub use formation::{ConsensusNetwork, DelayBound, Graph, GraphError, Protocol};
pub use latency::{History, Latency};
pub use margin::{DelayMargin, SecondOrderLoop, SmallAngleGains, worst_principal_axis};
pub use orbit::{ElementFeedback, Lqr, LqrError, clohessy_wiltshire_plant, lqr};
