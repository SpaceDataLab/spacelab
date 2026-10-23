//! **spacelab-gnss** — multi-constellation GNSS broadcast orbits, observations, geometry metrics and
//! RINEX parsing.
//!
//! The Keplerian constellations (GPS/Galileo/BeiDou/QZSS/NavIC) broadcast a
//! Keplerian-plus-corrections model turned into an Earth-fixed state by the IS-GPS-200 algorithm
//! ([`gps`]); GLONASS instead broadcasts a PZ-90 Cartesian state integrated numerically ([`glonass`]).
//! [`rinex`] reads either out of a RINEX 3 navigation file, [`almanac`] reads the coarse,
//! long-validity GPS almanac in both distributed formats, and [`dop()`] turns a set of satellite
//! positions into the dilution-of-precision geometry metrics.
//!
//! The measurement side starts at [`rinex_obs`], which reads a RINEX 3 observation file:
//! [`combinations`] forms the dual-frequency linear combinations from those observables,
//! [`cycle_slip`] cuts a carrier-phase arc where its integer ambiguity jumped, and [`lambda`] solves
//! the integer least-squares problem that fixes the ambiguity itself.
//!
//! This exceeds Orekit's `GNSSPropagator` reach: all six constellations plus the DOP metrics GNSS
//! engineers actually use. The orbit math is `no_std` + `libm`; the text parser needs `std`.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

pub mod combinations;
pub mod dop;
pub mod glonass;
pub mod gps;
pub mod higher_order_iono;
pub mod models;
pub mod rtcm;

#[cfg(any(feature = "std", test))]
pub mod gpt3;

#[cfg(any(feature = "std", test))]
pub mod almanac;

#[cfg(any(feature = "std", test))]
pub mod cycle_slip;

#[cfg(any(feature = "std", test))]
pub mod lambda;

#[cfg(any(feature = "std", test))]
pub mod ntrip;

#[cfg(any(feature = "std", test))]
pub mod rinex;
pub mod rinex4;

#[cfg(any(feature = "std", test))]
pub mod sinex_bias;

#[cfg(any(feature = "std", test))]
pub mod rinex_obs;

#[cfg(any(feature = "std", test))]
pub mod sp3;

#[cfg(any(feature = "std", test))]
pub mod vmf_grid;

pub use combinations::{
    CombinationError, FrequencyPair, carrier_frequency, geometry_free, graphic,
};
pub use dop::{Dop, dop};
pub use glonass::{GlonassEphemeris, glonass_l1_frequency, glonass_l2_frequency};
pub use gps::{BroadcastEphemeris, EARTH_GM, EARTH_ROTATION_RATE, GnssSystem, kepler_solve};
pub use models::{
    gmf_mapping, ionospheric_mapping_factor, klobuchar_iono_delay, niell_mapping,
    obliquity_mapping, saastamoinen_zenith_hydrostatic, saastamoinen_zenith_wet, vmf1_mapping,
    vmf3_mapping,
};
pub use rtcm::{
    ClockCorrection, DecodeError, Frame, Message, OrbitCorrection, RtcmDecoder, SsrClockBlock,
    SsrOrbitBlock, StationReferencePoint, apply_orbit, crc24q, decode,
};

#[cfg(any(feature = "std", test))]
pub use gpt3::{Gpt3Result, gpt3};

#[cfg(any(feature = "std", test))]
pub use almanac::{AlmanacEntry, parse_sem, parse_yuma, resolve_week};

#[cfg(any(feature = "std", test))]
pub use cycle_slip::{DetectorError, GeometryFreeDetector, PolynomialFitDetector, SlipVerdict};

#[cfg(any(feature = "std", test))]
pub use lambda::{Candidate, LambdaError, Reduction};

#[cfg(any(feature = "std", test))]
pub use ntrip::{
    ConnectError, Credentials, NtripClient, NtripError, Response, Revision, SourceTable,
    SourceTableEntry,
};

#[cfg(any(feature = "std", test))]
pub use rinex::{parse_glonass_nav, parse_gps_nav, parse_nav};

#[cfg(any(feature = "std", test))]
pub use sinex_bias::{Bias, BiasKind, BiasSinex, BiasUnit, SinexEpoch};

#[cfg(any(feature = "std", test))]
pub use rinex_obs::{
    ObsCode, ObsEpoch, ObsHeader, ObsKind, Observation, RinexObs, RinexObsError, SatObservation,
};

#[cfg(any(feature = "std", test))]
pub use sp3::{Itrs, ReferenceFrame, SatId, Sp3, TimeSystem};

#[cfg(any(feature = "std", test))]
pub use vmf_grid::{VmfCoefficients, VmfGrid, VmfGridError, VmfModel};
