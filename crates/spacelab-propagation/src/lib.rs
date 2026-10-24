//! Numerical ODE integration.
//!
//! SGP4 is analytic and needs no integrator; this crate is for what SGP4 cannot do — high-fidelity
//! force models, deep space, manoeuvres, orbit determination, attitude.
//!
//! Two rules govern everything here, and they come before any method:
//!
//! * **Coefficients are verified, never transcribed from memory.** Every Butcher tableau is checked
//!   against its order conditions in a computer algebra system (`derivations/dopri5_order.mac`) and
//!   against analytic-solution and convergence-order tests. A method claiming order 5 that is
//!   secretly order 4 is caught by reading the measured order off the numbers, not by hoping.
//! * **Dense output is the interface.** A propagator without a continuous interpolant forces the
//!   caller into the ring-buffer spline hack GMAT is stuck with. Every adaptive method carries its
//!   [`DenseOutput`], so [`spacelab_coverage::find_events`] can evaluate `g(t)` at any `t` inside a
//!   step without re-integrating.
//!
//! The state is a fixed-size `[f64; N]`: `[rx, ry, rz, vx, vy, vz]` for an orbit, a quaternion plus
//! body rate for attitude, `6·M` for `M` bodies. Const-generic `N` keeps it on the stack, `no_std`,
//! and friendly to the autovectoriser.

#![cfg_attr(not(test), no_std)]

// The degree-120 geopotential coefficient tables are heap-allocated (a ~0.8 MB `GeoPotential` on the
// stack overflows a thread stack once a few nest); `alloc` provides the `Box` for that. A no_std
// consumer must therefore supply a global allocator.
extern crate alloc;

pub mod attitude;
pub mod attitude_law;
pub mod density_calibration;
pub mod density_data;
pub mod density_rom;
pub mod dop853;
pub(crate) mod dop853_consts;
pub mod dopri5;
pub mod drag_coefficient;
pub mod earth_radiation;
pub mod ephemeris;
pub mod forces;
pub mod geopotential;
pub mod geopotential_coeffs;
pub mod harris_priester;
pub mod jb2008;
pub mod lifetime;
pub mod lightcurve;
pub mod linalg;
pub mod lunar;
pub mod lunar_coeffs;
pub mod maneuver;
pub mod nrlmsise00;
pub mod ocean_tide;
pub mod ocean_tide_admittance;
pub mod ocean_tide_fes2004;
pub mod ocean_tide_fes2014;
pub mod ocean_tide_got;
pub mod relativity;
pub mod rk4;
pub mod rkn1210;
pub(crate) mod rkn1210_consts;
pub mod sdtm;
pub mod shadow;
pub mod solid_tide;
pub mod space_weather;
pub mod spacecraft;
pub mod stt;
pub mod symplectic;
mod tide_step2_tables;
pub mod time_variable_gravity;
pub mod tolerance;
pub mod torque;
pub mod variational;
pub mod wind_data;
pub mod wind_model;

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
pub mod abm;
#[cfg(feature = "std")]
pub mod adams_nordsieck;
#[cfg(feature = "std")]
pub mod ark;
#[cfg(feature = "std")]
pub mod driver;
#[cfg(feature = "std")]
pub mod dsst;
#[cfg(feature = "std")]
pub mod gbs;
#[cfg(feature = "std")]
pub mod gj;
#[cfg(feature = "std")]
pub mod ias15;
#[cfg(feature = "std")]
pub mod ks;
#[cfg(feature = "std")]
pub mod mercurius;
#[cfg(feature = "std")]
pub mod mri;
#[cfg(feature = "std")]
pub mod radau;
#[cfg(feature = "std")]
pub mod rosenbrock;
#[cfg(feature = "std")]
pub mod saba;
#[cfg(feature = "std")]
pub mod taylor;
#[cfg(feature = "std")]
pub mod whfast;

pub use attitude_law::{
    AttitudeProvider, LocalOrbitalFrame, LofType, SpinStabilized, SunPointing, YawSteering,
};
// The reduced-order density model's own POD needs a symmetric eigensolver, and so does anything
// that has to read a spectrum — a graph Laplacian's, for one. Re-exported at the root rather than
// left behind `density_rom`, whose name says nothing about linear algebra, and rather than copied.
pub use density_rom::jacobi_symmetric;
pub use dop853::Dop853;
pub use dopri5::Dopri5;
pub use drag_coefficient::{
    FreeMolecularConditions, SentmanCoefficients, sentman_panel, speed_ratio,
};
#[cfg(feature = "std")]
pub use dsst::{
    DsstAuxiliary, DsstMeanPropagator, DsstSolarRadiationPressure, DsstTesseral, DsstThirdBody,
    DsstZonal,
};
pub use earth_radiation::{AngularResolution, KnockeEarthRadiation};
pub use geopotential::GeoPotential;
pub use harris_priester::{HarrisPriester, HarrisPriesterAtmosphere, harris_priester};
pub use jb2008::{Jb2008Atmosphere, Jb2008Input, Jb2008Output, jb2008};
pub use maneuver::{
    AccelerationSignal, ConstantThrust, EmpiricalAcceleration, Frame, HarmonicSignal,
    ImpulsiveManeuver, PolynomialSignal,
};
pub use nrlmsise00::{
    Nrlmsise00Atmosphere, Nrlmsise00Flags, Nrlmsise00Input, Nrlmsise00Output, gtd7, gtd7d,
};
pub use ocean_tide::OceanTide;
pub use relativity::{DeSitter, LenseThirring, Schwarzschild};
pub use solid_tide::{
    MeanPole, SolidEarthTide, ocean_pole_tide_corrections, pole_tide_corrections, step2_corrections,
};
pub use spacecraft::{
    BoxWing, Cannonball, DragModel, Facet, PanelDrag, PanelNormal, PanelSrp, RadiationModel,
};
pub use time_variable_gravity::{conventional_zonal_coefficients, secular_zonal_correction};
pub use tolerance::Tolerance;

/// The right-hand side of an ODE: writes `dy/dt` at `(t, y)` into `dydt`.
///
/// `t` is seconds from a reference epoch the caller tracks; the integrator never handles an
/// `Epoch<S>` in its hot loop, which keeps `i128` time arithmetic out of the derivative evaluation.
pub trait DynamicalSystem<const N: usize> {
    /// Evaluate the derivative. Called once per stage — this is the hot path.
    fn derivative(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]);
}

/// A continuous representation of the solution over one accepted step, so events and output need no
/// re-integration. Valid on `[t0, t0 + h]`; behaviour outside that is extrapolation and unspecified.
pub trait DenseOutput<const N: usize> {
    /// Interpolate the state at time `t`, writing it into `out`.
    fn eval(&self, t: f64, out: &mut [f64; N]);
}

/// The outcome of one trial step of an embedded method: the new state, the componentwise error
/// estimate, and the dense interpolant for the step just taken.
#[derive(Debug, Clone, Copy)]
pub struct StepTrial<const N: usize, D> {
    /// The proposed state at `t + h`.
    pub y_next: [f64; N],
    /// The embedded error estimate, componentwise. Its scaled norm decides acceptance.
    pub y_err: [f64; N],
    /// The interpolant for `[t, t + h]`.
    pub dense: D,
}
