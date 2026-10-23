//! Orbit-determination kernels.
//!
//! The one decision that governs the whole layer: **least squares by orthogonal factorisation, never
//! by the normal equations.** Forming `HᵀWH` squares the condition number, and orbit-determination
//! Jacobians are ill-conditioned enough that the difference is between eight correct digits and one
//! ([notes/14-estimation.md](../notes/14-estimation.md) §1). Everything here is built on the
//! square-root information filter [`Srif`], and the nonlinear solver [`GaussNewton`] wraps it.
//!
//! This is the linear-algebra core. Wiring it to a measurement model (range, range-rate, angles) and
//! the state transition matrix — which maps an observation's partials back to the epoch state — is
//! the next layer; the STM it needs is built (`spacelab_propagation::variational`).

#![cfg_attr(not(test), no_std)]

pub mod conjunction;
pub mod conjunction_domain;
pub mod consider;
pub mod consistency;
pub mod converter;
pub mod ensemble;
pub mod ephemeris;
pub mod fixed_lag;
pub mod gauss_newton;
pub mod hinf;
pub mod iekf;
pub mod imm;
pub mod kalman;
pub mod lie;
pub mod light_time;
pub mod lincov;
pub mod measurement;
pub mod mekf;
pub mod mixture;
pub mod modifier;
pub mod multistatic;
pub mod observed;
pub mod particle;
pub mod process_noise;
pub mod radiometric;
pub mod richardson;
pub mod robust;
pub mod rts;
pub mod sampling;
pub mod screening;
pub mod se23;
pub mod se3;
pub mod srif;
pub mod srukf;
pub mod unscented;

pub use conjunction::{ConjunctionError, ConjunctionGeometry, EncounterPlane};
pub use conjunction_domain::{
    CurvilinearEncounter, DomainError, RectilinearEncounter, RelativeSample, UsageViolations,
    usage_violations,
};
pub use consider::{consider_covariance, consider_sensitivity};
pub use consistency::{
    chi_square_mean_interval, chi_square_scale_factor, is_consistent, normalized_squared,
    scalar_nis,
};
pub use converter::{
    ConverterError, ConverterFit, DifferentiablePropagator, PropagatorBuilder, PropagatorConverter,
    SampledPropagator, TrajectoryResiduals, TrajectorySample, residual_statistics,
};
pub use ensemble::{EnsembleError, EnsembleKalman};
pub use ephemeris::{Component, StateComponent};
pub use fixed_lag::FixedLagSmoother;
pub use gauss_newton::{GaussNewton, GaussNewtonReport, ResidualModel};
pub use hinf::{HInfinityError, HInfinityFilter};
pub use iekf::InvariantEkf;
pub use imm::{ImmError, InteractingMultipleModel};
pub use kalman::SquareRootKalman;
pub use lie::UnitQuaternion;
pub use light_time::{SPEED_OF_LIGHT, Transit, down_leg};
pub use lincov::LinearCovariance;
pub use measurement::{
    Station, Topocentric, azimuth, declination, elevation, range, range_rate, right_ascension,
};
pub use mekf::MultiplicativeEkf;
pub use mixture::{GaussianComponent, GaussianMixture};
pub use modifier::{
    AtmosphericDelay, Azimuth, Bias, Declination, Elevation, EstimatedBias, EstimatedDrift,
    EstimatedShapiroDelay, Modified, Modifier, ModifierError, Observable, Partials, Range,
    RangeRate, RightAscension, ShapiroDelay, SlantDelay, SolarCoronaDelay, partials, shapiro_delay,
    shapiro_delay_ppn,
};
pub use multistatic::{
    BistaticGeometry, BistaticRange, BistaticRangeRate, Fdoa, InterSatelliteRange,
    InterSatelliteRangeRate, MultistaticError, PairObservable, PairPartials, Tdoa, TdoaGeometry,
    Terminal, bistatic_range, bistatic_range_rate, fdoa, inter_satellite_range,
    inter_satellite_range_rate, pair_partials, tdoa,
};
pub use observed::{Observed, Signal, partials_at_emission, reduce};
pub use particle::{ParticleError, ParticleFilter, Resampling};
pub use process_noise::{
    DiscreteStep, ProcessNoiseError, dynamic_model_compensation, matrix_exp, process_noise_sqrt,
    rotate_position_velocity, state_noise_compensation, van_loan,
};
pub use radiometric::{
    AKSIM_PAVLOV_2022, Band, CASSINI_KA_TURNAROUND_RATIO, CoronaSeries, CoronaTerm, DopplerLink,
    DopplerReference, DopplerSignature, PLASMA_DELAY_COEFFICIENT, PlasmaSignature, Quasar,
    RadiometricError, RampSegment, RampTable, SOLAR_RADIUS, averaged_doppler, delta_dor,
    measured_frequency, range_modulus, sequential_range, solar_corona_delay,
};
pub use richardson::{richardson_extrapolate, richardson2};
pub use robust::{RobustOutcome, RobustPolicy, RobustTally, chi_square_gate, robust_update};
pub use rts::{RtsError, RtsSmoother, SmoothingStep};
pub use sampling::Rng;
pub use screening::{
    CloseApproach, MAX_SAMPLES, ScreeningError, closest_approaches, radially_separated,
};
pub use se3::Se3;
pub use se23::{ImuNoise, NavigationEkf, Se23};
pub use srif::{Srif, SrifError};
pub use srukf::SquareRootUnscentedKalman;
pub use unscented::{UnscentedParams, unscented_transform};
