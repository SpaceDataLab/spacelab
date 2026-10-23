//! **earthvision** — Earth-observation foundations, the first Phase-8 line.
//!
//! Two building blocks that sit under every optical remote-sensing pipeline:
//!
//! - [`observation_geometry`] — the sun/sensor angles (solar and view zenith, relative azimuth,
//!   scattering angle, air mass) that drive atmospheric correction and surface-BRDF inversion. The
//!   satellite tells you *where* it looked; these tell you *how the light arrived*.
//! - [`monte_carlo_slab`] — a photon Monte Carlo of radiative transfer through a plane-parallel
//!   scattering-and-absorbing atmosphere (Marchuk's method), returning the layer's reflectance,
//!   transmittance and absorptance. This is the path-radiance / atmospheric-correction core: what
//!   fraction of sunlight is scattered back to space, reaches the surface, or is absorbed. What a
//!   scattering event does is a [`PhaseFunction`] — analytic, or tabulated from [`mie`] theory or an
//!   [`ice`]-crystal ray trace, which is what lets cirrus scatter like cirrus rather than like a
//!   one-parameter fit with no halo in it. Those three are flux fractions, over the whole hemisphere;
//!   [`path_reflectance`] is the same layer seen down one line of sight, by next-event estimation —
//!   which is what a sensor measures, and what the 6S path-reflectance term means.
//! - [`swath_geometry`] — the sensor's ground reach: from an off-nadir look angle, the Earth-central
//!   angle, elevation, ground and slant ranges of the footprint edge, with [`ground_point_at`] to
//!   trace footprint circles and ground tracks. The *where it looked* the two above take as given.
//!
//! `no_std` + `libm`, deterministic (a seeded `xoshiro256**` stream), no external dependencies.

#![cfg_attr(not(any(test, feature = "onnx")), no_std)]

extern crate alloc;

pub mod atmospheric_correction;
pub mod camera;
pub mod features;
pub mod gas;
pub mod geometry;
pub mod ice;
pub mod image_quality;
pub mod mie;
pub mod mvs;
pub mod ortho;
pub mod pansharpen;
pub mod phase;
pub mod radiative_transfer;
pub mod rpc;
pub mod rpc_adjust;
pub mod sfm;
pub mod stereo;
pub mod swath;

#[cfg(feature = "onnx")]
pub mod cloud;

pub use atmospheric_correction::{
    AtmosphericCoupling, CorrelatedKCoupling, brdf_reflectance, correlated_k_coupling,
    coupling_from_slab, coupling_from_stack, li_sparse_reciprocal_kernel, ross_thick_kernel,
};
pub use camera::{
    Observation, OutlierScale, PinholeCamera, bundle_adjust, bundle_adjust_robust,
    ransac_resection, resection, triangulate,
};
pub use features::{
    Corner, FeatureMatch, HarrisConfig, Image, NccConfig, detect_corners, harris_response,
    match_ncc,
};
pub use gas::{GasError, KDistribution, gas_optical_depths, gas_optical_depths_binned};
pub use geometry::{ObservationGeometry, air_mass, observation_geometry};
pub use ice::{
    GeometricOptics, HexagonalCrystal, MAX_LOBE_PANELS, RayTracingOptics, add_diffraction,
    diffraction_on_grid, trace_on_grid,
};
pub use mie::{BulkOptics, Complex, SizeDistribution};
pub use mvs::plane_sweep_mvs;
pub use ortho::{
    DemGrid, MapGrid, SourceTile, orthorectify, orthorectify_point, ray_dem_intersection,
    sample_bilinear,
};
pub use pansharpen::{brovey_pan_sharpen, gram_schmidt_pan_sharpen};
pub use phase::{PhaseError, PhaseFunction, TabulatedPhase, tabulate_legendre, tabulate_mie};
pub use radiative_transfer::{
    Atmosphere, Layer, MAX_LAYERS, SlabResult, monte_carlo_slab, monte_carlo_stack,
    path_reflectance, stack_path_reflectance,
};
pub use rpc::{GroundPoint, InverseError, RationalPolynomialCamera};
pub use rpc_adjust::{
    AdjustmentModel, GroundControlPoint, RpcAdjustment, adjustment_residual_rms,
    estimate_adjustment, forward_adjusted,
};
pub use sfm::{Observation2d, Track, build_tracks, triangulate_tracks};
pub use stereo::{
    DIRECTIONS_4, DIRECTIONS_8, SgmConfig, aggregate_sgm, census_cost_volume, depth_from_disparity,
    disparity_wta, semi_global_match,
};
pub use swath::{SwathGeometry, ground_point_at, horizon_look_angle, swath_geometry};

#[cfg(feature = "onnx")]
pub use cloud::{CloudMasker, Footprint, OnnxError, SwathLine, usable_footprints};
