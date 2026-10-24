//! Two-body and conic mission design — the transfer layer for cislunar and interplanetary flight.
//!
//! This is the first slice of the deep-space pillar: the *conic* primitives that need no ephemeris
//! and are verifiable against textbook cases. The high-fidelity propagation they seed lives in
//! `spacelab_propagation`; the planetary ephemerides
//! that turn a transfer into a dated launch window are Stage A (`anise`/DE440).
//!
//! * [`kepler`] — universal-variable two-body propagation (Stumpff formulation), the workhorse that
//!   advances a state by `Δt` for any conic (ellipse, parabola, hyperbola) through one branchless path.
//! * [`lambert`](mod@lambert) — the transfer arc between two positions in a given time, the core
//!   of injection and porkchop design.
//! * [`cr3bp`] — the circular restricted three-body problem: pseudo-potential, Jacobi constant, the
//!   five libration points, and the synodic equations of motion (Earth–Moon and Sun–Earth).
//!
//! Units are the caller's, consistent: state in (length, length/time), `mu` in length³/time². The
//! two-body functions are unit-agnostic; the CR3BP is nondimensional (separation and mean motion 1).

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod cr3bp;
pub mod cw;
pub mod edl;
pub mod iod;
pub mod kepler;
pub mod lambert;
pub mod lowthrust;
pub mod optimal;
pub mod patched;
pub mod periodic;
pub mod powered;
pub mod richardson;
pub mod synodic;
pub mod targeting;
pub mod transfer;

pub use cr3bp::Cr3bp;
pub use cw::{
    curvilinear_from_lvlh, cw_propagate, cw_propagate_curvilinear, lvlh_basis,
    lvlh_from_curvilinear, relative_lvlh_state,
};
pub use edl::{
    Atmosphere, EntryOutcome, EntryVehicle, GuidedEntry, fnpeg_bank_command, guided_entry,
    predicted_range, propagate_entry,
};
pub use iod::{
    IodError, Sighting, gauss, gibbs, gooding, herrick_gibbs, laplace, orbit_from_angles,
};
pub use kepler::propagate;
pub use lambert::{lambert, lambert_multirev};
pub use lowthrust::{Steering, propagate_lowthrust};
pub use optimal::{MinEnergyTransfer, min_energy_transfer, optimal_control_at};
pub use patched::{BPlane, HyperbolicArc, b_plane, hyperbolic_arc, sphere_of_influence};
pub use periodic::{
    ManifoldSeed, PeriodicOrbit, halo_family, halo_orbit, lyapunov_orbit, monodromy,
    richardson_halo, stable_manifold_seed, unstable_manifold_seed,
};
pub use powered::{Landing, powered_descent, zem_zev};
pub use richardson::richardson_halo_guess;
pub use synodic::{inertial_to_synodic, synodic_to_inertial};
pub use targeting::{Control, DifferentialCorrector, Goal, TargeterError, TargeterReport, target};
pub use transfer::{Transfer, transfer};
